#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <mpi.h>
#include "CEquiEnergyModel.hpp"
#include "mpi_constant.hpp"
#include "storage_constant.hpp"

using namespace std;

void master_mode_finding_deploying(const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model)
{	
	double *sPackage= new double [N_MESSAGE];
	double *rPackage= new double [N_MESSAGE];   
	MPI_Status status;

	// binning 
	model.storage->InitializeBin(0); // stage = 0; 
        vector<CSampleIDWeight> samples = model.storage->binning_equal_size(0, model.parameter->number_striation, 1.0); // stage=0, lambda=1.0 
	if (samples.empty())
	{
		cerr << "master_mode_finding_deploying(): Error in loading samples from the simulation.\n"; 
		exit(1); 
	}
	vector<CSampleIDWeight> start_points = model.Initialize_WeightedSampling(samples, nInitial);
	if (start_points.empty())
	{
		cerr << "master_mode_finding_deploying(): Error in initializing.\n"; 
		exit(1); 
	}

	string start_point_filename = model.parameter->storage_dir + model.parameter->run_id + string("/") + model.parameter->run_id + START_POINT;
        ofstream output_file; 
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


	// send out hill_climb jobs
	
	int start_index =0; 
	int end_index = ceil((double)nInitial/(double)(nNode-1)); 
	end_index = end_index < nInitial ? end_index : nInitial-1; 
	for (int i=1; i<nNode; i++)
	{
		sPackage[GROUP_INDEX] = start_index; 
		sPackage[LENGTH_INDEX] = end_index-start_index; 	
		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, i, HILL_CLIMB_TAG, MPI_COMM_WORLD);
		start_index = end_index; 
		end_index = ceil((double)nInitial*(i+1)/(double)(nNode-1));
		end_index = end_index < nInitial ? end_index : nInitial-1;
	}

        for (int i=1; i<nNode; i++)
                MPI_Recv(rPackage, N_MESSAGE, MPI_DOUBLE, MPI_ANY_SOURCE, HILL_CLIMB_TAG, MPI_COMM_WORLD, &status);

	// tell all the slaves to exit by sending an empty messag with 0 simulation length 
	for (int i=1; i<nNode; i++)
		MPI_Send(sPackage, N_MESSAGE, MPI_DOUBLE, i, END_TAG, MPI_COMM_WORLD);
        delete [] sPackage;
        delete [] rPackage;
}

