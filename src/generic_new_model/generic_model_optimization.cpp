#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#include <mpi.h>
#include <getopt.h>
#include <vector>
#include <iomanip>

#include "dw_rand.h"
#include "dw_ascii.hpp"

// dsmh_basic related header files
#include "CSampleIDWeight.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "CMetropolis.hpp"

// dsmh_mpi related header files
#include "TaskScheduling.hpp"
#include "mpi_constant.hpp"
#include "storage_constant.hpp"
#include "option.hpp"

// models derived from generic_model
#include "generic_model.hpp"
#include "generic_model_example.hpp"
#include "CEquiEnergy_generic_model.hpp"

using namespace std; 

int main(int argc, char **argv)
{
	static struct option long_options[] =
        {
		// Simulation options
		{"output directory", required_argument, 0, 'F'},
		{"ID", required_argument, 0, 'R'},
		{"number of optimization", required_argument, 0, 'N'},
                {"number of iterations for optimization", required_argument, 0,  'O'},
                {"number of iterations for perturbaion", required_argument, 0, 'P'},
                {"scaler for perturbation", required_argument, 0, 's'},
		{0, 0, 0, 0}
	}; 

	CEESParameter sim_option;
	sim_option.storage_dir = string("./"); // default directory for saving results
        sim_option.storage_marker = 10000; // related to how frequncy to replenish memory from disk
	sim_option.number_energy_stage = sim_option.number_striation = 1;  // default values for number of energy stages and number of striations
        sim_option.run_id = string(); // default value for ID
	int n_optimization = -1;
        int i_optimization = 10, i_perturbation=10;
        double s_perturbation =1.0;

	// Command line option processing
	int option_index = 0;	
	while (1)
        {
                int c = getopt_long(argc, argv, "F:R:N:O:P:s:", long_options, &option_index);
                if (c == -1)
                        break;
		switch(c)
                {
			case 'F':
				sim_option.storage_dir = string(optarg); break; 
			case 'R':
                                sim_option.run_id = string(optarg); break;
			case 'N':
                                n_optimization = atoi(optarg); break;
                        case 'O':
                                i_optimization = atoi(optarg); break;
                        case 'P':
                                i_perturbation = atoi(optarg); break;
                        case 's':
                                s_perturbation = atof(optarg); break;
			default: 
				break; 
		}
	}
	
	if ( sim_option.run_id.empty() || n_optimization < 0)
        {
                cerr << "Usage: " << argv[0] << " -R run_ID -N number_optimization\n";
                exit(1);
        }	

      	//////////////////////////////////////////////////////
      	//
      	// Fill details for the model derived from generic_model
      	Generic_Model_Example target_model; 
      	//
      	/////////////////////////////////////////////////////

	/////////////////////////////////////////////////////////////////////
	// DSMH Model
	MPI_Init(&argc, &argv);
        int my_rank, nNode;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nNode);
	
	dw_initialize_generator(time(NULL));

	CEquiEnergy_GenericModel simulation_model;
        simulation_model.target_model = &target_model; // target_model should have been specified as above, as an object derived from Generic_Model
        simulation_model.timer_when_started = -1;
        simulation_model.parameter = &sim_option;

	// set current parameter
	TDenseVector parameter_vector(target_model.GetNumberParameters(),0.0); 
	if (!target_model.DrawParametersFromPrior(parameter_vector.vector, parameter_vector.Dimension()))
	{
		cerr << "Error in drawing parameters from prior.\n" ; 
		exit(1); 
	}
        simulation_model.current_sample = CSampleIDWeight(parameter_vector, 0, target_model.log_posterior_function(parameter_vector.vector, parameter_vector.Dimension()), true);
        CSampleIDWeight mode = simulation_model.current_sample;
	simulation_model.storage = new CStorageHead (my_rank, sim_option.run_id, sim_option.storage_marker, sim_option.storage_dir, sim_option.number_energy_stage);

	// for communicating lower energy bound, and number of jumps from i-th striation to the j-th striation
	const int N_MESSAGE = (RESERVE_INDEX_START +1) + (sim_option.number_striation+1) + sim_option.number_striation*sim_option.number_striation; 

	if (my_rank == 0)
        {
                if (!simulation_model.storage->makedir())
                {
                        cerr << "Error in making directory for " << sim_option.run_id << endl;
                        double *sMessage= new double [N_MESSAGE];
                        for (int i=1; i<nNode; i++)
                                MPI_Send(sMessage, N_MESSAGE, MPI_DOUBLE, i, END_TAG, MPI_COMM_WORLD);
                        delete [] sMessage;
                        exit(1);
                }
		master_mode_finding_deploying(N_MESSAGE, nNode, n_optimization, simulation_model);
        }
        else
		slave_mode_finding_computing(N_MESSAGE, simulation_model, i_optimization, i_perturbation, s_perturbation);
	
  	return 0;
}
