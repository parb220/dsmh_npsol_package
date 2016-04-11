#include <vector>
#include <sstream>
#include <mpi.h> 
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include "dw_math.h"
#include "CSampleIDWeight.hpp"
#include "CEquiEnergyModel.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "storage_constant.hpp"
#include "mpi_constant.hpp"

using namespace std;  

vector<string> glob(const string &pattern); 

std::vector<CSampleIDWeight> DispatchSimulation(double *sPackage, double *rPackage, const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model, int simulation_length, int stage, int message_tag, ofstream &jump_file)
{
	sPackage[THIN_INDEX] = model.parameter->THIN;
       	sPackage[LEVEL_INDEX] = stage;
	sPackage[PEE_INDEX] = model.parameter->pee; 
	std::vector<CSampleIDWeight> samples(0); 

	MPI_Status status;
	std::vector<int> nTotalJump(2,0.0); // nTotalJump[0]: EE, nTotalJump[1] : MH
	TDenseMatrix jump_table(model.parameter->number_striation, model.parameter->number_striation,0.0); 
	
	vector<int> available_node(nNode-1); 
	for (int i=0; i<(int)(available_node.size()); i++)
		available_node[i] = i+1; 

	if (message_tag == SIMULATION_PRIOR_TAG) 
	{
		int length_per_node = (int)ceil((double)simulation_length/(double)available_node.size()); 
		while(!available_node.empty())
                {
                        sPackage[LENGTH_INDEX] = length_per_node;
                        MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, available_node.back(), message_tag, MPI_COMM_WORLD);
                        available_node.pop_back();
                }
	}	
	else 
	{
		std::vector<int> length_task, group_index_task;
		int length_per_initial = (int)ceil((double)simulation_length/(double)nInitial);
	
		if (length_per_initial > TASK_LENGTH_SHORT)
		{
			if (length_per_initial > TASK_LENGTH_LONG)
			{
				for (int iInitial=0; iInitial<nInitial; iInitial++)
				{
					for (int k=0; k<length_per_initial/TASK_LENGTH_LONG; k++)
					{
						length_task.push_back(TASK_LENGTH_LONG);
						group_index_task.push_back(iInitial);
					}
					if (length_per_initial%TASK_LENGTH_LONG)
					{
						length_task.push_back(length_per_initial % TASK_LENGTH_LONG);
						group_index_task.push_back(iInitial);
					}
				}

			}
			else 
			{
				for (int iInitial=0; iInitial<nInitial; iInitial++)
				{
					length_task.push_back(length_per_initial);
					group_index_task.push_back(iInitial);
				}	
			}

			for (int iTask = 0; iTask<(int)(length_task.size()); iTask ++)
			{
				sPackage[LENGTH_INDEX] = length_task[iTask];  
				sPackage[BURN_INDEX] = model.parameter->burn_in_length;
				sPackage[GROUP_INDEX] = group_index_task[iTask]; 
				sPackage[GROUP_NUMBER_INDEX] = 1; 
				if (available_node.empty())
				{
					MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, message_tag, MPI_COMM_WORLD, &status); 
					nTotalJump[0] += (int) rPackage[RETURN_INDEX_1];
                                        nTotalJump[1] += (int) rPackage[RETURN_INDEX_2];
					if ((int)rPackage[RESERVE_INDEX_START]) 
					{
						TDenseMatrix tmp_jump_table(model.parameter->number_striation, model.parameter->number_striation,0.0); 
						for (int j=0; j<tmp_jump_table.cols; j++)
							for (int i=0; i<tmp_jump_table.rows; i++)
								tmp_jump_table(i,j) = rPackage[RESERVE_INDEX_START+1 + j*tmp_jump_table.rows + i]; 
						jump_table = jump_table + tmp_jump_table; 
					}
					available_node.push_back(status.MPI_SOURCE); 
				}
				MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, available_node.back(), message_tag, MPI_COMM_WORLD);
				available_node.pop_back(); 
			}
		}
		else
		{
			int group_index_start = 0; 
			int group_index_stepsize = (int)ceil(nInitial/(float)available_node.size()); 
			while(!available_node.empty())
			{
		        	sPackage[LENGTH_INDEX] = length_per_initial; 
                        	sPackage[BURN_INDEX] = model.parameter->burn_in_length;
                        	sPackage[GROUP_INDEX] = group_index_start; 
                        	sPackage[GROUP_NUMBER_INDEX] = group_index_stepsize; 
                        	MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, available_node.back(), message_tag, MPI_COMM_WORLD);
                        	available_node.pop_back();
				group_index_start+=group_index_stepsize; 
                	}
		}
	} 

	for (int j=0; j<(nNode-1)-(int)available_node.size(); j++)
	{
		MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, message_tag, MPI_COMM_WORLD, &status);
		nTotalJump[0] += (int) rPackage[RETURN_INDEX_1];
                nTotalJump[1] += (int) rPackage[RETURN_INDEX_2];
		if ((int)rPackage[RESERVE_INDEX_START])
  		{
                	TDenseMatrix tmp_jump_table(model.parameter->number_striation, model.parameter->number_striation,0.0);
                	for (int j=0; j<tmp_jump_table.cols; j++)
                		for (int i=0; i<tmp_jump_table.rows; i++)
                			tmp_jump_table(i,j) = rPackage[RESERVE_INDEX_START+1 + j*tmp_jump_table.rows + i];
                	jump_table = jump_table + tmp_jump_table;
                }
	}

	if (jump_file.is_open())
	{
		jump_file << "Stage " << stage  << endl; 
		jump_file << "EE jump rate " << (double) nTotalJump[0]/(double)(model.parameter->simulation_length*model.parameter->THIN) << endl;
		jump_file << "MH jump rate " << (double)nTotalJump[1]/(double)(model.parameter->simulation_length*model.parameter->THIN) << endl; 
		jump_file << "Out of " << nTotalJump[1] << " MH jumps:" << endl; 
		for (int j=0; j<jump_table.cols; j++)
		{
			for (int i=0; i<jump_table.rows; i++)
			{
				if (i!=j && jump_table(i,j))
					jump_file << "Number of jumps from striation " << i << " to striation " << j << " " << jump_table(i,j) << endl; 
			}
		}
	}

	// binning
	if (message_tag != TUNE_TAG_SIMULATION_FIRST && message_tag != SCALE_MATRIX_FIT_TAG) 
	{
		samples = model.storage->binning_equal_size(stage, model.parameter->number_striation, model.parameter->lambda[stage]);

		sPackage[LEVEL_INDEX] = (double)stage;
        	sPackage[RESERVE_INDEX_START] = (double)(model.storage->GetNumber_Bin(stage));
        	for (int i=0; i<model.storage->GetNumber_Bin(stage); i++)
        		sPackage[RESERVE_INDEX_START + i + 1] = model.storage->GetEnergyLowerBound(stage, i);

        	for (int i=1; i<nNode; i++)
        		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, i, BINNING_INFO, MPI_COMM_WORLD);

        	for (int i=1; i<nNode; i++)
        		MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, BINNING_INFO, MPI_COMM_WORLD, &status);
	}
	return samples;
}
