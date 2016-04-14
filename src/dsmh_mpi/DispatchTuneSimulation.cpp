#include <cmath>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cstdio>
#include "dw_rand.h"
#include "dw_math.h"
#include "CSampleIDWeight.hpp"
#include "CEquiEnergyModel.hpp"
#include "CMetropolis.hpp"
#include "storage_constant.hpp"
#include "mpi_constant.hpp"
#include "dw_matrix.h"
#include "mdd_constant.hpp"
#include "mdd_function.h"
#include "master_deploying.hpp"
#include "EstimateLogMDD.hpp"
#include "option.hpp"

#include <time.h>

using namespace std; 

vector<string> glob(const string &pattern);
void GetWeightedVarianceMatrix(CEquiEnergyModel &model, int stage, const std::vector<CSampleIDWeight> &); 
bool ReadScaleFromFile(const string &file_name, double &c); 
bool WriteScaleToFile(const string &file_name, double c); 
bool FileExist(const string &file_name); 
double ScaleFit(double*, double*, const int, CEquiEnergyModel &model, int stage, int nNode, int nInitial);
vector<int> StriationDistribution(const vector<CSampleIDWeight> &samples, const CEquiEnergyModel &model, int stage); 

void DispatchTuneSimulation(double *sPackage, double *rPackage, const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model,const CSampleIDWeight &mode, int simulation_length, const Diagnosis &option, bool save_space_flag)
{
	// log_file
	string log_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + string(".log");
	ofstream log_file(log_filename.c_str(), iostream::out); 
	if (!log_file)
	{
		cerr << "DispatchTuneSimulation: Error opening " << log_filename << endl; 
		exit(1); 
	}
	// diagnostic statistics
	// consistency
	vector<double> consistency(model.parameter->number_energy_stage, 0.0), average_consistency(model.parameter->number_energy_stage, 0.0), std_consistency(model.parameter->number_energy_stage, 0.0), LB_ESS(model.parameter->number_energy_stage, 0.0); 
	string consistency_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + string(".LogMDD.txt") ; 
	ofstream consistency_file(consistency_filename.c_str(), iostream::out); 
	if (!consistency_file)
        {
        	cerr << "DispatchTuneSimulation: Error opening " << consistency_filename << endl;
                exit(1); 
        }
	
	vector<vector<int > > striation_distribution; 
	string distribution_filename; 
	ofstream distribution_file; 
	if (option & OPT_DSTR_STR)
	{
		striation_distribution.resize(model.parameter->number_energy_stage); 
		distribution_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + string(".Distribution.txt"); 
		distribution_file.open(distribution_filename.c_str(), iostream::out); 
		if (!distribution_file.is_open())
		{
                	cerr << "DispatchTuneSimulation: Error opening " << distribution_filename << endl;
                	exit(1);
        	}	
	}

	string jump_filename; 
	ofstream jump_file; 
	if (option & OPT_JMP_RT || option & OPT_TRAN_STR)
	{
		jump_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + string(".Jump.txt");
		jump_file.open(jump_filename.c_str(), iostream::out); 
		if (!jump_file.is_open())
		{
			cerr << "DispatchTuneSimulation: Error opening " << jump_filename << endl;
                        exit(1);
		}
	}
	
	vector<double> logMDD; 
	if (option & OPT_MLLR)
		logMDD.resize(model.parameter->number_energy_stage+1); 

	vector<CSampleIDWeight> samples;
	string start_point_filename; 
	ofstream output_file; 

	time_t rawtime;
	double alpha_0 = LOWER_ALPHA, alpha_1 = UPPER_ALPHA; 

	/////////////////////////////////////////////////////////////////////////////////
	// Highest + 1 stage 
	model.storage->InitializeBin(model.parameter->highest_stage+1);
	if (model.parameter->highest_stage == model.parameter->number_energy_stage-1)
	{
		// Dispatch Hill Climb jobs to obtain start points
		// draw highest stage + 1 sample
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - drawing from stage=" << model.parameter->highest_stage+1  << " " << ctime(&rawtime) << endl;

		// samples = samples of highest+1 stage
		const bool NPSOL_HIGHEST_PLUS_1 = false; 
		if (NPSOL_HIGHEST_PLUS_1)
			samples = HighestPlus1Stage(sPackage, rPackage, N_MESSAGE, nNode, nInitial, model, mode, alpha_0, alpha_1, log_file, jump_file);   // At highest+1 level, first find a local maximum of the heated posterior, and do adaptive Metropolis Hasting
		else 
			samples = HighestPlus1Stage_Prior(sPackage, rPackage, N_MESSAGE, nNode, nInitial, model, jump_file);   // Alternatively, at the highest+1 level, draw from prior distribution
				
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - done drawing from stage=" << model.parameter->highest_stage+1 << " " << ctime(&rawtime) << endl;
	}
	else 
		samples = model.storage->DrawAllSample(model.parameter->highest_stage+1);
	if (option & OPT_MLLR)
        {
                consistency[model.parameter->highest_stage+1] = logMDD[model.parameter->highest_stage+1] = LogMDD(samples, model, model.parameter->lambda[model.parameter->highest_stage+1], USE_TRUNCATED_POWER, LIKELIHOOD_HEATED);
                consistency_file << "Stage " << model.parameter->highest_stage+1 << ":\t\t\t\t" << logMDD[model.parameter->highest_stage+1] << endl;
                time(&rawtime);
                log_file << "DispatchTuneSimulation() - done computing MDD: stage=" << model.parameter->highest_stage+1 << " " << ctime(&rawtime) << endl;
        }
        else
                consistency[model.parameter->highest_stage+1] = LogMDD(samples, model, model.parameter->lambda[model.parameter->highest_stage+1], USE_TRUNCATED_POWER, LIKELIHOOD_HEATED);

 
	// Starting from highest down to lowest
	for (int stage=model.parameter->highest_stage; stage>=model.parameter->lowest_stage; stage--)
	{
		// samples = samples of the previous stage
		consistency[stage] = CheckConvergency(samples, model, stage, stage+1, consistency[stage+1], average_consistency[stage], std_consistency[stage], LB_ESS[stage], LIKELIHOOD_HEATED, nInitial); 
                consistency_file << "Stage " << stage << ": " << setprecision(20) << consistency[stage] << "\t" << average_consistency[stage] << "\t" << std_consistency[stage]<< "\t" << LB_ESS[stage]; 
	
		////////////////////////////////////////////////////////////////////////////////
		// Starting points
	  	time(&rawtime);
	  	log_file << "DispatchTuneSimulation() - top of loop: stage= " << stage << " " << ctime(&rawtime) << endl;
		model.storage->InitializeBin(stage); 
		vector<CSampleIDWeight> start_points; 
		if (!samples.empty() )
			start_points = model.Initialize_WeightedSampling(samples, nInitial, stage+1);
		if (samples.empty() || start_points.empty())
		{
			start_points.resize(nInitial); 
			for (int i=0; i<nInitial; i++)
				start_points[i] = mode; 
		}

		start_point_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + START_POINT; 
        	output_file.open(start_point_filename.c_str(), ios::binary|ios::out);
        	if (!output_file.is_open())
		{
               		cerr << "Error in writing to " << start_point_filename << endl; 
			exit(1); 
		}
        	else
		{
			for (int i=0; i<(int)(start_points.size()); i++)
               			write(output_file, &(start_points[i]));
		}
        	output_file.close();
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - done getting initial points: stage=" << stage << " " << ctime(&rawtime) << endl;


		/////////////////////// Tuning
		string block_file_name = model.parameter->storage_dir  + model.parameter->run_id + string("/") + model.parameter->run_id + BLOCK;
		if (!FileExist(block_file_name))
                        GetWeightedVarianceMatrix(model, stage, samples);
	
		double alpha = ScaleFit(sPackage, rPackage, N_MESSAGE, model, stage, nNode, nInitial); 
		log_file << "Metropolis acceptance rate at stage " <<  stage << " using the scale matrix of the previous stage " << alpha << endl; 

		int counter = 0; 
		while (alpha <= alpha_0 || alpha >= alpha_1) 
		{
			if (counter == 0)
				GetWeightedVarianceMatrix(model, stage, samples); 
			else 
			{
				if (log(alpha) <= 5.0*log(0.5*(alpha_0+alpha_1)))
					model.metropolis->SetScale(0.2); 
				else if (5.0*log(0.5*(alpha_0+alpha_1)) < log(alpha) && log(alpha) < 0.2*log(0.5*(alpha_0+alpha_1)))
					model.metropolis->SetScale(log(0.5*(alpha_0+alpha_1))/log(alpha)); 
				else if (log(alpha) >= 0.2*log(0.5*(alpha_0+alpha_1) ) )
					model.metropolis->SetScale(5.0); 

        			if (!model.metropolis->WriteBlocks(block_file_name))
        			{
                			cerr << "DispatchTuneSimulation() : Error in writing BMatrix file.\n";
                			exit(1); 
        			}
			}
			
			counter ++; 

			alpha = ScaleFit(sPackage, rPackage, N_MESSAGE, model, stage, nNode, nInitial); 
			log_file << "Metropolis acceptance rate at stage " <<  stage << " using the scale matrix of the previous stage " << alpha << endl; 
		} 
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - done tuning: stage=" << stage << " " << ctime(&rawtime) << endl;	

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// simualtion
		// samples = samples of the current stage
		samples = DispatchSimulation(sPackage, rPackage, N_MESSAGE, nNode, nInitial, model, model.parameter->simulation_length, stage, SIMULATION_TAG, jump_file);
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - done simulating (" << model.parameter->simulation_length << "): stage=" << stage << " " << ctime(&rawtime) << endl;
	
		// Number of draws in each striation where the striation is defined at the pervious stage
		if (option & OPT_DSTR_STR)
		{	
			striation_distribution[stage] = StriationDistribution(samples, model, stage+1);
			distribution_file << "Stage " << stage << ": \n"; 
			for (int i=0; i<(int)(striation_distribution[stage].size()); i++)
			{
				if (striation_distribution[stage][i])
					distribution_file << "Striation " << i << ": " << striation_distribution[stage][i] << endl; 
			}
			distribution_file.flush(); 
		}	 	
	
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// logMDD using Mueller's method
		if (option & OPT_MLLR)
		{
	     		logMDD[stage] = LogMDD(samples, model, model.parameter->lambda[stage], USE_TRUNCATED_POWER, LIKELIHOOD_HEATED);
			consistency_file << setprecision(20) << "\t" << logMDD[stage]; 	
			time(&rawtime);
			log_file << "DispatchTuneSimulation() - done computing MDD: stage=" << stage << " " << ctime(&rawtime) << endl;
		}
		consistency_file << endl; 
		consistency_file.flush(); 

		// save samples in ascii if stage == 0
		if (stage == 0)
		{
			string ascii_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + string(".draw.text");
			ofstream ascii_file(ascii_filename.c_str(), iostream::out); 
			if (!ascii_file)
			{
				cerr << "Error in writing to " << ascii_filename << endl; 
				exit(1); 
			}
			for (int i=0; i<(int)(samples.size()); i++)
			{
				if (i) 
					ascii_file << endl; 
				ascii_file << samples[i].weight << "\t" << samples[i].reserved; 
				for (int j=0; j<samples[i].data.Dimension(); j++)
					ascii_file << "\t" << samples[i].data[j]; 
			}
			ascii_file.close(); 	
		}
			
		
		// to save space, remove stage+1 samples
                if (save_space_flag ) 
		{
			model.storage->ClearSample(stage+1);  
			stringstream convert; 
			convert.str(string());
                        convert << model.parameter->run_id << "/" << model.parameter->run_id << "*." << stage+1 << "*";;
                        string remove_file_pattern = model.parameter->storage_dir + convert.str();
			vector<string> remove_file = glob(remove_file_pattern); 
                	for (int i=0; i<(int)remove_file.size(); i++)
                        	remove(remove_file[i].c_str());
			
			convert.str(string());
                        convert << model.parameter->run_id << "/" << model.parameter->run_id << "*.sample." << stage+1;
			remove_file_pattern = model.parameter->storage_dir + convert.str();
			remove_file = glob(remove_file_pattern); 
                        for (int i=0; i<(int)remove_file.size(); i++)
                                remove(remove_file[i].c_str());
		}
		model.storage->ClearBin(stage+1); 
		time(&rawtime);
		log_file << "DispatchTuneSimulation() - bottom of loop: stage=" << stage << " " << ctime(&rawtime) << endl;
	}

	log_file.close(); 
	consistency_file.close(); 
	if (option & OPT_DSTR_STR)
		distribution_file.close();
	if (option & OPT_JMP_RT || option & OPT_TRAN_STR)
		jump_file.close(); 
}

void GetWeightedVarianceMatrix(CEquiEnergyModel &model, int stage, const std::vector<CSampleIDWeight> &samples)
{
	// Calculate B for AdaptiveAfterSimulation_WeightedSampling_OnePass
	// weight
	vector<double> log_weight = model.Reweight(samples, stage, stage+1);
       	double log_weight_sum = log_weight[0];
       	for (int i=1; i<(int)log_weight.size(); i++)
		log_weight_sum = AddLogs(log_weight_sum, log_weight[i]);
       	vector<double> weight(log_weight.size(), 0.0);
       	for (int i=0; i<(int)log_weight.size(); i++)
                weight[i] = exp(log_weight[i] - log_weight_sum);
			
	// block_scheme
        string block_scheme_file_name = model.parameter->storage_dir  + model.parameter->run_id + string("/") + model.parameter->run_id + BLOCK_SCHEME;
	if (!model.metropolis->ReadBlockScheme(block_scheme_file_name)) 
		model.metropolis->SetBlockScheme(vector<TIndex>(1,TIndex(0,samples[0].data.dim-1))); 

	model.metropolis->GetBlockMatrix_WeightedSampling(samples, weight); 

	// Bmatrix file
	string bmatrix_file_name = model.parameter->storage_dir  + model.parameter->run_id  + string("/") + model.parameter->run_id + BLOCK; 
	if (!model.metropolis->WriteBlocks(bmatrix_file_name))
	{
		cerr << "DispatchTuneSimulation() : Error in writing BMatrix file.\n"; 
		abort(); 
	}
}

bool FileExist(const string &file_name)
{
	struct stat buffer;   
  	return (stat (file_name.c_str(), &buffer) == 0);			
}

bool ReadScaleFromFile(const string &file_name, double &c)
{
	ifstream iFile; 
	iFile.open(file_name.c_str(), iostream::in | iostream::binary); 
	if (!iFile)
		return false; 
	iFile.read((char *)&c, sizeof(double)); 
	iFile.close(); 
	return true; 
}

bool WriteScaleToFile(const string &file_name, double c)
{
	ofstream oFile; 
	oFile.open(file_name.c_str(), iostream::out | iostream::binary); 
	if (!oFile)
		return false; 
	oFile.write((char *)&c, sizeof(double)); 
	oFile.close(); 
	return true; 
}

double ScaleFit(double *sPackage, double *rPackage, const int N_MESSAGE, CEquiEnergyModel &model, int stage, int nNode, int nInitial)
{
        sPackage[THIN_INDEX] = 1;
        sPackage[LEVEL_INDEX] = stage;
        sPackage[PEE_INDEX] = model.parameter->pee;
       	sPackage[LENGTH_INDEX] = 500;
        sPackage[BURN_INDEX] = model.parameter->burn_in_length;

        MPI_Status status;

	for (int i=1; i<nNode; i++)
	{
		sPackage[GROUP_INDEX] = dw_uniform_int(nInitial); 
		sPackage[GROUP_NUMBER_INDEX] = 1; 
		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, i, SCALE_MATRIX_FIT_TAG, MPI_COMM_WORLD); 
	}

	std::vector<int> nJump(2,0);
	for (int i=1; i<nNode; i++)
	{
		MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, SCALE_MATRIX_FIT_TAG, MPI_COMM_WORLD, &status);
		nJump[0] += rPackage[RETURN_INDEX_1]; // EE jump
                nJump[1] += rPackage[RETURN_INDEX_2]; // MH jump
	}
	double avg_accpt_rate = (double)nJump[1]/((nNode-1)*sPackage[LENGTH_INDEX]); 

	return avg_accpt_rate; 
}

vector<int> StriationDistribution(const vector<CSampleIDWeight> &samples, const CEquiEnergyModel &model, int stage)
{
	int bin_index; 
	double log_posterior_stage; 
	vector<int> distribution(model.parameter->number_striation+1,0); 
	for (int i=0; i<(int)samples.size(); i++)
	{
		log_posterior_stage = samples[i].reserved*model.parameter->lambda[stage] + (samples[i].weight-samples[i].reserved); 
		bin_index = model.storage->BinIndex(stage,-log_posterior_stage); 
		distribution[bin_index]	++; 
	}
	return distribution; 
}

