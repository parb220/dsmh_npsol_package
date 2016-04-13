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
#include "dw_ascii.h"
#include "sbvar.hpp"
#include "CEquiEnergy_TimeSeries.hpp"
#include "CSampleIDWeight.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "mpi_constant.hpp"
#include "CMetropolis.hpp"
#include "storage_constant.hpp"
#include "TaskScheduling.hpp"
#include "maximization_option.hpp"
#include "option.hpp"

using namespace std; 

int main(int argc, char **argv)
{
	static struct option long_options[] =
        {
                {"data_file", required_argument, 0, 'D'},
                {"restriction_file", required_argument, 0, 'S'},
		{"output directory", required_argument, 0, 'F'}, 
		// Simulation options
		{"ID", required_argument, 0, 'R'},
		{"number of optimization", required_argument, 0, 'N'},
		{"number of iterations for optimization", required_argument, 0,  'O'},
		{"number of iterations for perturbaion", required_argument, 0, 'P'}, 
		{"scaler for perturbation", required_argument, 0, 's'},
		{0, 0, 0, 0}
	}; 

	int option_index = 0;
	
	CEESParameter sim_option;
	sim_option.storage_dir = string("./"); // getenv("HOME")+string("/DW_TZ_GIT/projects_dw/work/sbvar/results/");
	sim_option.storage_marker = 10000;
	sim_option.number_energy_stage = sim_option.number_striation = 1;
        sim_option.run_id = string(); 
	int n_optimization = -1; 
	int i_optimization = 10, i_perturbation=10; 
	double s_perturbation =1.0; 

	string data_file_name, restriction_file; 

	while (1)
        {
                int c = getopt_long(argc, argv, "D:S:F:R:N:O:P:s:", long_options, &option_index);
                if (c == -1)
                        break;
		switch(c)
                {
                        case 'D':
                                data_file_name = string(optarg); break;
			case 'S':
                                restriction_file = string(optarg); break;
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
	if (data_file_name.empty() || restriction_file.empty() || sim_option.run_id.empty() || n_optimization < 0 ) 
	{
		cerr << "Usage: " << argv[0] << " -D data file -S restriction file -R run_ID -N number_optimization\n"; 
		exit(1); 
	}	

      	//////////////////////////////////////////////////////
      	// restrictions file
      	ifstream input; 
      	input.open(restriction_file.c_str(), ifstream::in);
      	if (!input.is_open()) 
	{
		cerr << "Unable to open restrictions file" << restriction_file << endl; 
		exit(1); 
	}
      	vector<TDenseMatrix> U, V;
      	SetupRestrictionMatrices(U,V,input);
      	input.close();
	
      	//////////////////////////////////////////////////////
      	// constant term, n_vars, n_predetermined, n_lags, n_exogenous, n_parameters
      	bool IsConstant=true;
	int n_vars=U[0].rows; 
	int n_predetermined = V[0].rows; 
      	int n_lags=NumberLags(TDenseMatrix(n_vars,n_vars,0.0),TDenseMatrix(n_vars,n_predetermined,0.0),IsConstant); 
      	int n_parameters=0;
      	for (int i=0; i < n_vars; i++) 
		n_parameters+=U[i].cols+V[i].cols;

      	///////////////////////////////////////////////////////
      	//Generate/read data
      	int n_lines = dw_NumberLines((FILE *)NULL, (char*)data_file_name.c_str()); 
      	int n_obs= n_lines-n_lags; // since 1988.01 
	if (n_obs <= 0)
	{
		cout << "There are not sufficient data in " << data_file_name <<endl;
                exit(1); 
	}
      	TDenseMatrix rawdata;

	input.open(data_file_name.c_str(), ifstream::in);
      	if (!input.is_open()) 
	{
		cout << "Unable to open data file" << data_file_name <<endl;
		exit(1); 
	}
      	rawdata.Resize(n_obs+n_lags,1+n_vars);
      	input >> rawdata;
      	input.close();
	
      	TData_predetermined Data(n_lags, IsConstant, rawdata, TIndex(1,rawdata.cols-1), TIndex(), n_lags,rawdata.rows-1);
	
      	// Sims-Zha prior 
      	TDenseVector mu(6);
       	// mu(0)=1.0; mu(1)=1.0; mu(2)=1.0; mu(3)=1.2; mu(4)=3.0; mu(5)=3.0; 
       	mu(0) = 0.7; mu(1) = 0.5; mu(2) = 0.1; mu(3) = 1.2; mu(4) = 1.0; mu(5) = 1.0;
      	double periods_per_year=12.0;
      	SBVAR_symmetric_linear sbvar(&Data,mu,periods_per_year, 1.0, U, V);
      	sbvar.DefaultParameters();

	// Estimate parameters
	sbvar.MaximizePosterior(1.0e-5,false);
     	TDenseVector EstimatedParameters(sbvar.NumberParameters());
      	sbvar.GetParameters(EstimatedParameters.vector);

	/////////////////////////////////////////////////////////////////////
	// EquiEnergyModel
	MPI_Init(&argc, &argv);
        int my_rank, nNode;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nNode);
	
	dw_initialize_generator(time(NULL));

	CEquiEnergy_TimeSeries simulation_model;
        simulation_model.target_model = &sbvar;
        simulation_model.timer_when_started = -1;
        simulation_model.parameter = &sim_option;
        simulation_model.current_sample = CSampleIDWeight(EstimatedParameters, 0, sbvar.LogPosterior(EstimatedParameters.vector), true);
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
