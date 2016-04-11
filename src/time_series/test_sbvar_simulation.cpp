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
#include "option.hpp"

using namespace std; 

int main(int argc, char **argv)
{
	static struct option long_options[] =
        {
                {"data_file", required_argument, 0, 'D'},
                {"restriction_file", required_argument, 0, 'S'},
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

	int option_index = 0;
        int nGroup = 1;
	bool if_pure_MH = false;
	
	CEESParameter sim_option;
	sim_option.storage_dir = string("./"); // getenv("HOME")+string("/DW_TZ_GIT/projects_dw/work/sbvar/results/");
        sim_option.storage_marker = 10000;
        sim_option.run_id = cluster_to_string(time(NULL));
	sim_option.number_energy_stage = sim_option.number_striation = 1; 
        sim_option.lambda_1 = 0.1;
        sim_option.THIN = 50;
        sim_option.pee = 1.0/(10.0*sim_option.THIN);
	sim_option.burn_in_length = 0;        
	sim_option.simulation_length = 200000;

	Diagnosis diagnosis_option = OPT_ESS; 
	string data_file_name, restriction_file; 

	while (1)
        {
                int c = getopt_long(argc, argv, "D:S:F:R:oH:M:T:P:I:N:B:G:jtdm", long_options, &option_index);
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
	if (data_file_name.empty() || restriction_file.empty()) 
	{
		cerr << "Usage: " << argv[0] << " -D data file -S restriction file.\n"; 
		exit(1); 
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
        if (if_pure_MH)
                simulation_model.if_bounded = false;
        simulation_model.metropolis = new CMetropolis(&simulation_model); 
        simulation_model.parameter = &sim_option;
        simulation_model.current_sample = CSampleIDWeight(EstimatedParameters, 0, sbvar.LogPosterior(EstimatedParameters.vector), true);
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
                
		// direct calculation of logMDD
		/*if (sim_option.highest_stage == sim_option.number_energy_stage-1)
		{
			for (int stage = sim_option.highest_stage; stage >= sim_option.lowest_stage; stage--)
			{
				sbvar.SetTemperature(sim_option.lambda[stage],1.0); 
				cout << "true logMDD at stage " << stage << setprecision(20) << ": " << sbvar.LogPosteriorIntegral() << endl; 
			}
			sbvar.SetTemperature(1.0,1.0); 
		}*/
		master_deploying(N_MESSAGE, nNode, nGroup, simulation_model, mode, diagnosis_option); 
        }
        else
		slave_computing(N_MESSAGE, simulation_model, mode); 

  	return 0;
}
