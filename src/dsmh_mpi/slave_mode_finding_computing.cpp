#include <mpi.h>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include "dw_ascii.hpp"
#include "dw_dense_matrix.hpp"
#include "CEquiEnergyModel.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "CSampleIDWeight.hpp"
#include "mpi_constant.hpp"
#include "storage_constant.hpp"

#include <time.h>

using namespace std;

void slave_mode_finding_computing(const int N_MESSAGE, CEquiEnergyModel &model, int optimizationN, int perturbationN, double perturbationS) 
{
	int my_rank, nNode; 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &nNode);  
	MPI_Status status; 
	
	double *rPackage = new double [N_MESSAGE], *sPackage = new double [N_MESSAGE];    

	while (1)
	{
		MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status); 
		if (status.MPI_TAG == END_TAG)
		{
			delete []rPackage; 
			delete []sPackage; 
			exit(0); 
		}
		else if (status.MPI_TAG == HILL_CLIMB_TAG)
		{
			model.energy_stage = 0; 

			int start_index = (int)(rPackage[GROUP_INDEX]);
			int end_index = start_index+(int)rPackage[LENGTH_INDEX]-1; 
			// start point
			string start_point_file = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + START_POINT; 
			vector<CSampleIDWeight> start_points = LoadSampleFromFile(start_point_file); 
			if (start_points.empty() || end_index >= (int)start_points.size())
			{
				cerr << "slave_mode_finding_computing() : Error occurred reading start point files " << start_point_file << endl; 
				exit(1); 
			}
			vector<CSampleIDWeight>solutions((int)rPackage[LENGTH_INDEX]); 
			for (int j=start_index; j<=end_index; j++)
			{
				model.current_sample = start_points[j]; 
				vector<CSampleIDWeight> one_solution=model.HillClimb_NPSOL(1, optimizationN, perturbationN, perturbationS, 1.0, model.current_sample.data); // nSolution=1 
				solutions[j-start_index] = one_solution[0]; 
			}

			// Save gm_mean and gm_covariance_sqrt into files 
			string output_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + OPTIMIZATION + string(".") + cluster_to_string(start_index); 
			ofstream output_file(output_filename.c_str(), iostream::out); 
			if (!output_file)
			{
				cerr << "slave_mode_finding_computing() : Error occurred writing optimization solutions to " << output_filename << endl;
				exit(1);
			}
			for (int j=0; j<(int)solutions.size(); j++)
			{
				output_file << solutions[j].weight << "\t" << solutions[j].reserved;
				for (int i=0; i<solutions[j].data.Dimension(); i++)
					output_file << "\t" << solutions[j].data[i];
				output_file << endl; 
			}
		
			output_file.close(); 
			sPackage[RETURN_INDEX_1] = (double)(false); 
		}
		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD); 
	}
}

