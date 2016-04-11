#include <cmath>
#include <algorithm>
#include <vector> 
#include <functional>
#include <ctime>
#include <fstream>
#include "CSampleIDWeight.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "dw_dense_matrix.hpp"
#include "dw_math.h"
#include "CEquiEnergyModel.hpp"
#include "CMetropolis.hpp"
#include "dw_rand.h"

using namespace std;

bool CEquiEnergyModel::JumpAcrossStriation(const CSampleIDWeight &y_end, const CSampleIDWeight &y_initial, TDenseMatrix &jump_table) const 
{
	if (energy_stage == parameter->number_energy_stage)
		return false; 
	double heated_initial = y_initial.reserved*parameter->lambda[energy_stage+1] + (y_initial.weight - y_initial.reserved);
	double heated_end = y_end.reserved*parameter->lambda[energy_stage+1] + (y_end.weight - y_end.reserved); 
	int bin_initial = storage->BinIndex(energy_stage+1, -heated_initial); 
	int bin_end = storage->BinIndex(energy_stage+1, -heated_end); 
	if (jump_table.rows <= bin_initial || jump_table.cols <= bin_end) 
		return false; 
	jump_table(bin_initial,bin_end) ++; 
	return true;  
}

bool CEquiEnergyModel::MakeEquiEnergyJump(CSampleIDWeight &y_end, const CSampleIDWeight &y_initial)
{
	double heated_initial = y_initial.reserved*parameter->lambda[energy_stage+1] + (y_initial.weight - y_initial.reserved); 
	if(storage->DrawSample(energy_stage+1, storage->BinIndex(energy_stage+1,-heated_initial), y_end) ) // if a sample is successfully draw from bin
	{
		// calculate log_ratio in the current and the higher stages
		double log_ratio = parameter->LogRatio_Stage(y_initial, y_end, energy_stage+1); 
		log_ratio += parameter->LogRatio_Stage(y_end, y_initial, energy_stage); 
		if (log(dw_uniform_rnd()) <= log_ratio)
			return true; 
	}
	y_end = y_initial; 
	return false; 
}

void CEquiEnergyModel::SaveSampleToStorage(const CSampleIDWeight &sample)
{
	double heated = sample.reserved * parameter->lambda[energy_stage] + (sample.weight - sample.reserved); 
        storage->DepositSample(energy_stage, storage->BinIndex(energy_stage, -heated), sample);
}

void CEquiEnergyModel::Take_New_Sample_As_Current_Sample(const CSampleIDWeight &x_new)
{
	current_sample = x_new; 
	current_sample.id = timer_when_started;
}

int CEquiEnergyModel::EE_Draw()
{
	CSampleIDWeight x_new; 
	int new_sample_code = NO_JUMP; 

	if (dw_uniform_rnd() <= parameter->pee ) // EE jump
	{
		if (MakeEquiEnergyJump(x_new, current_sample))
		{
			Take_New_Sample_As_Current_Sample(x_new); 
			new_sample_code = EQUI_ENERGY_JUMP; 
		}
	}
	else 
	{
		double bounded_log_posterior_new; 
		if (metropolis->BlockRandomWalkMetropolis(bounded_log_posterior_new, x_new, current_sample, 1))
		{
			Take_New_Sample_As_Current_Sample(x_new); 
			new_sample_code = METROPOLIS_JUMP; 
		}
	}
	
	return new_sample_code; 
}


std::vector<int> CEquiEnergyModel::BurnIn(int burn_in_length)
{
	CSampleIDWeight x_new; 
	int nMHJump =0; 
	double bounded_log_posterior_new; 

	for (int i=0; i<burn_in_length; i++)
	{
		if (metropolis->BlockRandomWalkMetropolis(bounded_log_posterior_new, x_new, current_sample, 1) )
		{
			Take_New_Sample_As_Current_Sample(x_new); 
			nMHJump ++; 
		}
	}
	std::vector<int> nJump(2);
        nJump[0] = 0;   // EE jump
        nJump[1] = nMHJump; // MH jump
        return nJump;
}

std::vector<int> CEquiEnergyModel::Simulation_Prior(bool if_storage, const string &sample_file_name)
{
	CSampleIDWeight x_new(TDenseVector(current_sample.data.dim));   
        bool if_write_file = false;
        ofstream output_file;
        if (!sample_file_name.empty() )
        {
                output_file.open(sample_file_name.c_str(), ios::binary | ios::out);
                if (output_file)
                        if_write_file = true;
        }

	for (int i=0; i<parameter->simulation_length; i++)
	{
		do
		{
			DrawParametersFromPrior(x_new.data.vector); 
			x_new.DataChanged(); 
			log_posterior_function(x_new);
		} while (x_new.weight <= MINUS_INFINITY); 
		Take_New_Sample_As_Current_Sample(x_new); 
		if (if_storage)
               	      SaveSampleToStorage(current_sample);
                if (if_write_file)
                      write(output_file, &current_sample);
	}
	if (if_write_file)
                output_file.close();
	return std::vector<int>(2,0); 
}

std::vector<int> CEquiEnergyModel::Simulation_Within(TDenseMatrix &jump_table, bool if_storage, const string &sample_file_name) 
{
	CSampleIDWeight x_new; 
	int nMHJump =0; 
	double bounded_log_posterior_new; 
	bool if_write_file = false; 
	ofstream output_file; 
	if (!sample_file_name.empty() )
	{
		output_file.open(sample_file_name.c_str(), ios::binary | ios::out); 
		if (output_file)
			if_write_file = true; 
	}

	for (int i=0; i<parameter->simulation_length; i++)
	{
		for (int j=0; j<parameter->THIN; j++)
		{
			if (metropolis->BlockRandomWalkMetropolis(bounded_log_posterior_new, x_new, current_sample, 1))
			{// when x_new is accepted
				if (jump_table.rows && jump_table.cols)
					JumpAcrossStriation(x_new, current_sample, jump_table); 
				Take_New_Sample_As_Current_Sample(x_new);
                        	nMHJump ++;
			}
		}
		// if (jump_table.rows && jump_table.cols)
		// 	JumpAcrossStriation(x_new, current_sample, jump_table); 
		// Take_New_Sample_As_Current_Sample(x_new);
		
		if (if_storage)
			SaveSampleToStorage(current_sample);
		if (if_write_file)
			write(output_file, &current_sample); 
	}
	if (if_write_file)
		output_file.close(); 
	std::vector<int> nJump(2);
        nJump[0] = 0; // EE jump
        nJump[1] = nMHJump; // MH jump
	return nJump; 
}

std::vector<int> CEquiEnergyModel::Simulation_Cross(TDenseMatrix &jump_table, bool if_storage, const string &sample_file_name)
{
	CSampleIDWeight x_old; 

	std::vector<int> nJump(2,0); // nJump[0]: EE, nJump[1]: MH
	bool if_write_file = false;
        ofstream output_file;
        if (!sample_file_name.empty() )
        {
                output_file.open(sample_file_name.c_str(), ios::binary | ios::out);
                if (output_file)
                        if_write_file = true;
        }
	for (int i=0; i<parameter->simulation_length; i++)
	{
		for (int j=0; j<parameter->THIN; j++)
		{
			x_old = current_sample; 
			int jump_code = EE_Draw(); 
			if (jump_code == EQUI_ENERGY_JUMP)
				nJump[0] ++; // nEEJump++; 
			else if (jump_code == METROPOLIS_JUMP)
			{ // current_sample is already the new sample
				nJump[1] ++; // nMHJump++; 
				if (jump_table.rows && jump_table.cols )
					JumpAcrossStriation(current_sample, x_old, jump_table);
			}
		}	
		//if (jump_table.rows && jump_table.cols )
		//	JumpAcrossStriation(current_sample, x_old, jump_table);
	
		if (if_storage)
			SaveSampleToStorage(current_sample); 
		if (if_write_file)
			write(output_file, &current_sample); 
	}

	if (if_write_file)
		output_file.close(); 	

	return nJump; 
	// cout << "EE Jump " << nEEJump << " out of " << parameter->simulation_length *parameter->thin<< " in simulation.\n"; 
	// cout << "MH Jump " << nMHJump << " out of " << parameter->simulation_length *parameter->thin<< " in simulation.\n"; 
}

CEquiEnergyModel::CEquiEnergyModel() : 
gmm_mean(vector<TDenseVector>(0)), gmm_covariance_sqrt(vector<TDenseMatrix>(0)), 
gmm_covariance_sqrt_log_determinant(vector<double>(0)), gmm_covariance_sqrt_inverse(vector<TDenseMatrix>(0)), 
if_bounded(true), energy_stage(0), lambda(1.0), current_sample(CSampleIDWeight()), timer_when_started(-1), metropolis(NULL), parameter(NULL), storage(NULL)
{}

CEquiEnergyModel::CEquiEnergyModel(bool _if_bounded, int eL, double _lambda, const CSampleIDWeight &_x, time_t _time, CMetropolis *_metropolis, CEESParameter *_parameter, CStorageHead *_storage) :
gmm_mean(vector<TDenseVector>(0)), gmm_covariance_sqrt(vector<TDenseMatrix>(0)), 
gmm_covariance_sqrt_log_determinant(vector<double>(0)), gmm_covariance_sqrt_inverse(vector<TDenseMatrix>(0)),
if_bounded(_if_bounded), energy_stage(eL), lambda(_lambda), current_sample(_x), timer_when_started(_time), metropolis(_metropolis), parameter(_parameter), storage(_storage) 
{
}


