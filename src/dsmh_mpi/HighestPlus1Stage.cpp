#include <vector>
#include <string>
#include <mpi.h>
#include <algorithm>
#include <ctime>
#include "dw_rand.h"
#include "dw_math.h"
#include "CEquiEnergyModel.hpp"
#include "CMetropolis.hpp"
#include "mpi_constant.hpp"
#include "storage_constant.hpp"
#include "master_deploying.hpp"

using namespace std; 

vector<string> glob(const string &pattern); 
double ScaleFit(double*, double*, const int, CEquiEnergyModel &model, int stage, int nNode, int nInitial);

std::vector<CSampleIDWeight> HighestPlus1Stage(double *sPackage, double *rPackage, const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model, const CSampleIDWeight &mode, double alpha_0, double alpha_1,  ofstream &log_file, ofstream &jump_file)
{
	// Dispatch hill climb task
	int nFeasibleSolutionPerNode = ceil( (double)(nInitial*10.0)/(double)(nNode-1.0) ); 
        sPackage[LENGTH_INDEX] = nFeasibleSolutionPerNode;
        sPackage[LEVEL_INDEX] = model.parameter->number_energy_stage ;

	for (int i=1; i<nNode; i++)
        {
                sPackage[GROUP_INDEX] = dw_uniform_int(nInitial);
                sPackage[GROUP_NUMBER_INDEX] = 1;
                MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, i, HILL_CLIMB_TAG, MPI_COMM_WORLD);
        }

        MPI_Status status;
        for (int i=1; i<nNode; i++)
                MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, HILL_CLIMB_TAG, MPI_COMM_WORLD, &status);

	// Consolidate gm_mean to get starters
	stringstream convert;
        convert.str(string());
        convert <<  model.parameter->run_id << "/" << model.parameter->run_id << GM_MEAN_COVARIANCE << ".*";
        string filename_pattern =  model.parameter->storage_dir + convert.str();
	
	vector<string> merge_file_name = glob(filename_pattern);
	vector<CSampleIDWeight> samples; 
	for (int i=0; i<(int)(merge_file_name.size()); i++)
	{
		vector<CSampleIDWeight> read_from_file = LoadSampleFromFile(merge_file_name[i]); 
		if (!read_from_file.empty())
			samples.insert(samples.end(), read_from_file.begin(), read_from_file.end()); 
	}
	
	std::vector<CSampleIDWeight> starters(nInitial); 
	if (!samples.empty())
	{
		CSampleIDWeight_Sorter comparator(model.parameter->lambda[model.parameter->number_energy_stage]); 
		sort(samples.begin(), samples.end(), comparator); 

		// importance weight to select starters 
		vector<double> log_weight_sum(samples.size()), weight_sum(samples.size());  
		log_weight_sum[0] = samples[0].weight;
		for (int i=1; i<(int)(samples.size()); i++)
			log_weight_sum[i] = AddLogs(log_weight_sum[i-1], samples[i].weight); 
		for (int i=0; i<(int)(samples.size()); i++)
			weight_sum[i] = exp(log_weight_sum[i] - log_weight_sum.back()); 
		
		for (int i=0; i<nInitial; i++)
		{
			double random_number = dw_uniform_rnd();
                	int position = std::lower_bound(weight_sum.begin(), weight_sum.end(), random_number)-weight_sum.begin();
                	starters[i] = samples[position]; 
		}
	}
	else 
	{
		for (int i=0; i<nInitial; i++)
			starters[i] = mode; 
	}

	// write starter file
	string start_point_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + START_POINT;
        ofstream output_file(start_point_filename.c_str(), ios::binary|ios::out);
        if (!output_file.is_open())
        {
        	cerr << "Error in writing to " << start_point_filename << endl;
                exit(1);
        }
        else
        {
        	for (int i=0; i<(int)(starters.size()); i++)
                    write(output_file, &(starters[i]));
        }
        output_file.close();

	time_t rawtime; 
	time(&rawtime);
       	log_file << "DispatchTuneSimulation() - done getting initial points: stage=" << model.parameter->number_energy_stage << " " << ctime(&rawtime) << endl;

	// Tuning; 
	// block scheme
	string block_scheme_file_name = model.parameter->storage_dir  + model.parameter->run_id + string("/") + model.parameter->run_id + BLOCK_SCHEME;
        if (!model.metropolis->ReadBlockScheme(block_scheme_file_name))
                model.metropolis->SetBlockScheme(vector<TIndex>(1,TIndex(0,starters[0].data.dim-1)));

	// One block with identity matrix as the jump kernel initially
	vector<TDenseMatrix> block(1); 
	block[0].Identity(starters[0].data.dim); 
	model.metropolis->SetBlocks(block); 
	string block_file_name = model.parameter->storage_dir  + model.parameter->run_id + string("/") + model.parameter->run_id + BLOCK;
	if (!model.metropolis->WriteBlocks(block_file_name))
        {
        	cerr << "DispatchTuneSimulation() : Error in writing BMatrix file.\n";
                exit(1);
        }

	double alpha = ScaleFit(sPackage, rPackage, N_MESSAGE, model, model.parameter->number_energy_stage, nNode, nInitial);
        log_file << "Metropolis acceptance rate at stage " <<  model.parameter->number_energy_stage << " using the diagonal scale matrix " << alpha << endl;
	
	int counter = 0;
        while (alpha <= alpha_0 || alpha >= alpha_1)
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

                counter ++;

                alpha = ScaleFit(sPackage, rPackage, N_MESSAGE, model, model.parameter->number_energy_stage, nNode, nInitial);
                log_file << "Metropolis acceptance rate at stage " <<  model.parameter->number_energy_stage << " using the scale matrix of the previous stage " << alpha << endl;
        }
        time(&rawtime);
        log_file << "DispatchTuneSimulation() - done tuning: stage=" << model.parameter->number_energy_stage << " " << ctime(&rawtime) << endl;
	
	// simulation
	samples.clear(); 
	samples = DispatchSimulation(sPackage, rPackage, N_MESSAGE, nNode, nInitial, model, model.parameter->simulation_length, model.parameter->number_energy_stage, SIMULATION_TAG, jump_file);		
	time(&rawtime);
        log_file << "DispatchTuneSimulation() - done simulating (" << model.parameter->simulation_length << "): stage=" << model.parameter->number_energy_stage << " " << ctime(&rawtime) << endl;
	return samples; 	
}
