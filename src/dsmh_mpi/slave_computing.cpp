#include <mpi.h>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include "dw_dense_matrix.hpp"
#include "CEquiEnergyModel.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "CSampleIDWeight.hpp"
#include "mpi_constant.hpp"
#include "storage_constant.hpp"
#include "prcsn.h"
#include "slave_computing.hpp"

using namespace std;

void slave_computing(const int N_MESSAGE, CEquiEnergyModel &model, const CSampleIDWeight &mode)
{
	int my_rank, nNode; 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &nNode);  
	MPI_Status status; 
	
	double *rPackage = new double [N_MESSAGE], *sPackage = new double [N_MESSAGE];    
	int group_index; 

	while (1)
	{
		MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
		if (status.MPI_TAG == END_TAG)
		{
			delete []rPackage; 
			delete []sPackage; 
			exit(0); 
		}
		else if (status.MPI_TAG == BINNING_INFO)
		{
			model.energy_stage = (int)(rPackage[LEVEL_INDEX]); 
			int number_ring = (int)(rPackage[RESERVE_INDEX_START]); 
			model.storage-> ResizeBin(model.energy_stage, number_ring); 
			for (int i=0; i<number_ring; i++)
				model.storage->SetEnergyLowerBound(model.energy_stage, i, rPackage[RESERVE_INDEX_START+i+1]);
			if (model.energy_stage+1 <= model.parameter->number_energy_stage)
				model.storage->ClearBin(model.energy_stage+1); 
			sPackage[RETURN_INDEX_1] = 0;
                        sPackage[RETURN_INDEX_2] = 0;
		}		
		else if (status.MPI_TAG == HILL_CLIMB_TAG)
		{
			vector<CSampleIDWeight> solutions = model.HillClimb_NPSOL(rPackage[LENGTH_INDEX]);
			if (!solutions.empty())
			{
				stringstream convert;
                        	convert.str(string());
                        	convert <<  model.parameter->run_id << "/" << model.parameter->run_id << GM_MEAN_COVARIANCE << "." << my_rank;
                        	string gm_file = model.parameter->storage_dir + convert.str();
				if (!SaveSampleToFile(gm_file, solutions))
				{
					cerr << "SaveSampleToFile() : Error occurred.\n"; 
					exit(1); 
				}
			}
			sPackage[RETURN_INDEX_1] = 0;
                        sPackage[RETURN_INDEX_2] = 0;
		}
		else if (status.MPI_TAG == TUNE_TAG_SIMULATION_FIRST || status.MPI_TAG == SIMULATION_TAG || status.MPI_TAG == SIMULATION_PRIOR_TAG || status.MPI_TAG == SCALE_MATRIX_FIT_TAG) 
		{	
			model.energy_stage = (int)(rPackage[LEVEL_INDEX]);
			group_index = (int)(rPackage[GROUP_INDEX]); 
			int nGroup = (int)(rPackage[GROUP_NUMBER_INDEX]); 
			// model.timer_when_started = group_index; 
		
			if (!GetCommunicationParameter(rPackage, N_MESSAGE, model.parameter))
			{
				cerr << "GetCommunicationParameter() : Error occurred.\n"; 
				exit(1); 
			}
			model.lambda = model.parameter->lambda[model.energy_stage];
			TDenseMatrix jump_table; 
			std::vector<int> nJump = ExecutingSimulationTask(jump_table, model, my_rank, group_index, nGroup,  mode, status.MPI_TAG); 
			sPackage[RETURN_INDEX_1] = nJump[0];
                        sPackage[RETURN_INDEX_2] = nJump[1];

			if (jump_table.rows && jump_table.cols)
			{
				sPackage[RESERVE_INDEX_START] = jump_table.rows*jump_table.cols; 
				for (int j=0; j<jump_table.cols; j++)
					for (int i=0; i<jump_table.rows; i++)
						sPackage[RESERVE_INDEX_START+1 + j*jump_table.rows +i ] = jump_table(i,j); 
			}
			else 
				sPackage[RESERVE_INDEX_START] = 0; 
		}
		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD); 
	}
}

