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
                {"Pure Metropolis-Hasting", no_argument, 0, 'o'},
                {"Number of stages", required_argument, 0, 'H'},
		{"Number of striations", required_argument, 0, 'M'},
                {"Lambda_1", required_argument, 0, 'T'},
		{"Pee", required_argument, 0, 'P'}, 
                {"Thinning factor", required_argument, 0, 'I'},
                {"Number of draws per group", required_argument, 0, 'N'},
		{"Burn-in length", required_argument, 0, 'B'}, 
                {"Number of groups", required_argument, 0, 'G'},
		// Diaoganistic options
		{"Display jump rate", no_argument, 0, 'j'}, 
		{"Display transitions across striations", no_argument, 0, 't'},
		{"Display distrition among striations", no_argument, 0, 'd'},
		{"Use Mueller's method", no_argument, 0, 'm'}, 
		{0, 0, 0, 0}
	}; 

        int nGroup = 1;	// default value for number of groups
	bool if_pure_MH = false; // default: not pure metropolis hasting
	
	CEESParameter sim_option;
	sim_option.storage_dir = string("./"); // default directory for saving results
        sim_option.storage_marker = 10000; // related to how frequncy to replenish memory from disk
        sim_option.run_id = cluster_to_string(time(NULL)); // default value for ID
	sim_option.number_energy_stage = sim_option.number_striation = 1;  // default values for number of energy stages and number of striations
        sim_option.lambda_1 = 0.1; // default value for lambda_1
        sim_option.THIN = 50; // default value for thinning factor
        sim_option.pee = 1.0/(10.0*sim_option.THIN); // defulat value for frequency of equi-energy jump
	sim_option.burn_in_length = 0;  // default value for burn-in length
	sim_option.simulation_length = 200000; // defalut value for length of simulation

	Diagnosis diagnosis_option = OPT_ESS; // option to print out diagnostic information 

	// Command line option processing
	int option_index = 0;	
	while (1)
        {
                int c = getopt_long(argc, argv, "F:R:oH:M:T:P:I:N:B:G:jtdm", long_options, &option_index);
                if (c == -1)
                        break;
		switch(c)
                {
			case 'F':
				sim_option.storage_dir = string(optarg); break; 
			case 'R':
                                sim_option.run_id = string(optarg); break;
                        case 'o':
                                if_pure_MH = true; break; 
                        case 'H':
				sim_option.number_energy_stage = atoi(optarg); break; 
			case 'M':
				sim_option.number_striation = atoi(optarg); break; 
			case 'P':
				sim_option.pee = atof(optarg); break; 
			case 'T':
                                sim_option.lambda_1 = atof(optarg); break;
                        case 'I':
                                sim_option.THIN = atoi(optarg); break;
                        case 'N':
                                sim_option.simulation_length = atoi(optarg); break;
			case 'B': 
				sim_option.burn_in_length = atoi(optarg);  break; 
                        case 'G':
                                nGroup= atoi(optarg); break;
			case 'j':
				diagnosis_option = static_cast<Diagnosis>(static_cast<int>(diagnosis_option) | static_cast<int>(OPT_JMP_RT)); break; 
			case 't':
				diagnosis_option = static_cast<Diagnosis>(static_cast<int>(diagnosis_option) | static_cast<int>(OPT_TRAN_STR)); break; 
			case 'd':
				diagnosis_option = static_cast<Diagnosis>(static_cast<int>(diagnosis_option) | static_cast<int>(OPT_DSTR_STR)); break; 
			case 'm':
				diagnosis_option = static_cast<Diagnosis>(static_cast<int>(diagnosis_option) | static_cast<int>(OPT_MLLR)); break; 
			default: 
				break; 
		}
	}		

	if (if_pure_MH)
        {
                sim_option.number_energy_stage = 1;
                sim_option.highest_stage = sim_option.lowest_stage = 0;
		sim_option.pee = 0.0; 
        }
	else 
	{
		sim_option.highest_stage = sim_option.number_energy_stage-1; 
		sim_option.lowest_stage = 0; 
	}
	sim_option.simulation_length *= nGroup;
	sim_option.SetTemperature_quadratic();  

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
        if (if_pure_MH)
                simulation_model.if_bounded = false;
        simulation_model.metropolis = new CMetropolis(&simulation_model); 
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
		cout << "Lambda : " << endl; 
		for (int i=0; i<(int)(sim_option.lambda.size()); i++)
			cout << setprecision(20) << sim_option.lambda[i] << "\t"; 
		cout << endl; 
                if (!simulation_model.storage->makedir())
                {
                        cerr << "Error in making directory for " << sim_option.run_id << endl;
                        double *sMessage= new double [N_MESSAGE];
                        for (int i=1; i<nNode; i++)
                                MPI_Send(sMessage, N_MESSAGE, MPI_DOUBLE, i, END_TAG, MPI_COMM_WORLD);
                        delete [] sMessage;
                        exit(1);
                }
		master_deploying(N_MESSAGE, nNode, nGroup, simulation_model, mode, diagnosis_option); 
        }
        else
		slave_computing(N_MESSAGE, simulation_model, mode); 

  	return 0;
}
