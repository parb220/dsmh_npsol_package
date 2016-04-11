#include <cstdlib>
#include <iomanip>
#include <getopt.h>
#include <fstream>
#include <ctime>
#include <mpi.h>
#include "dw_rand.h"
#include "dw_ascii.h"
#include "dw_dense_matrix.hpp"
#include "dw_data.hpp"
#include "dw_ascii.hpp"
#include "regime_processes.hpp"
#include "tvsbvar.hpp"
#include "CEESParameter.hpp"
#include "CSampleIDWeight.hpp"
#include "CStorageHead.hpp"
#include "maximization_option.hpp"
#include "mpi_constant.hpp"	
#include "CEquiEnergy_TimeSeries.hpp"
#include "CMetropolis.hpp"
#include "storage_constant.hpp"
#include "TaskScheduling.hpp"
#include "option.hpp"

using namespace std; 

int ParseRestrictionMatrix(std::vector< std::vector<int> > &R, TDenseMatrix &C, const StringMatrix &M, int pos, int n_rows, int n_cols=-1);
int FindIdentifier(const StringMatrix &M, const std::string &id);
int ParseInteger(int &k, const StringMatrix &M, int pos); 

int main(int argc, char **argv)
{
	// Initialize MPI
	MPI_Init(&argc, &argv);
        int my_rank, nNode;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nNode);

	// random number generator
	dw_initialize_generator(time(NULL)+my_rank*100000); 

	// command line options
	static struct option long_options[] =
        {
                {"data_file", required_argument, 0, 'D'},
		{"markov_regime_process_file", required_argument, 0, 'V'},
                {"deterministic_regime_process_file", required_argument, 0, 'C'},
		{"restriction_file", required_argument, 0, 'S'},
		{"hyper_parameter_file", required_argument, 0, 'Y'},
		{"output directory", required_argument, 0, 'F'},
		// Simulation options
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
	sim_option.storage_dir = string("./"); //getenv("HOME")+string("/DW_TZ_GIT/projects_dw/work/tvsbvar/results/"); 
	sim_option.storage_marker = 10000; 
	sim_option.run_id = cluster_to_string(time(NULL)); 
	sim_option.number_energy_stage = sim_option.number_striation = 1;
        sim_option.lambda_1 = 0.1;
	sim_option.THIN = 50; 
	sim_option.pee = 1.0/(10.0*sim_option.THIN);
	sim_option.burn_in_length = 0;  // BURN_IN_LENGTH; 
	sim_option.simulation_length = 200000; // SIMULATION_LENGTH; 
	
	Diagnosis diagnosis_option = OPT_ESS;

	string data_file_name, markov_regime_process_file, deterministic_regime_process_file, restriction_file, hyper_parameter_file; 
	
	while (1)
        {
                int c = getopt_long(argc, argv, "D:V:C:S:Y:F:R:oH:M:T:P:I:N:B:G:jtdm", long_options, &option_index);
                if (c == -1)
                        break;
                switch(c)
                {
                        case 'D':
                                data_file_name = string(optarg); break;
			case 'V': 
				markov_regime_process_file = string(optarg); break; 
			case 'C':
				deterministic_regime_process_file = string(optarg); break;
			case 'S': 
				restriction_file = string(optarg); break; 
			case 'Y':
				hyper_parameter_file = string(optarg); break; 
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
				sim_option.burn_in_length = atoi(optarg); break; 
			case 'G': 
				nGroup = atoi(optarg); break; 
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
        if (data_file_name.empty() || (markov_regime_process_file.empty() && deterministic_regime_process_file.empty() )|| restriction_file.empty() || hyper_parameter_file.empty())
	{
		cerr << "Usage: " << argv[0] << " -D data file -V markov regime process file -C deterministic regime process file -S restriction file -Y hyper parameter file\n"; 
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

	/////////////////////////////////// Data //////////////////////////////////////////
	// obtain n_lags, n_vars from model restriction file
        ifstream input_file;
	input_file.open(restriction_file.c_str(), fstream::in);
        if (!input_file)
        {
                cerr << "Error in opening" << restriction_file << endl;
                abort();
        }
	StringMatrix M(input_file, string("//**"));
        int pos, n_vars, n_lags;
        if ( (pos=FindIdentifier(M, "NumberVariables")) < 0 || !ParseInteger(n_vars, M, pos+1) || n_vars < 0 )
	{
                cerr << "Error in parsing //== NumberVariables in " << restriction_file << endl;
		exit(1);  
	}
        if ( (pos=FindIdentifier(M, "NumberLags")) < 0 || !ParseInteger(n_lags, M, pos+1) || n_lags < 0)
	{
                cerr << "Error in parsing //== NumberLags in " << restriction_file << endl; 
		exit(1); 
	}

	input_file.close(); 
	
	int n_lines = dw_NumberLines((FILE *)NULL, (char*)data_file_name.c_str());
        int n_obs= n_lines-n_lags; // since 1988.01 
        if (n_obs <= 0)
        {
                cout << "There are not sufficient data in " << data_file_name <<endl;
                exit(1);
        }
	
	TDenseMatrix data_from_file(n_obs+n_lags,n_vars+1);  //570,4); // data_from_file(90, 37); // 
        input_file.open(data_file_name.c_str(), fstream::in);
        if (!input_file)
        {
                cerr << "Error in opening " << data_file_name << endl;
                abort();
        }
        input_file >> data_from_file;
        input_file.close();
	// append a constant vector Ones(data_from_file.rows) to the end of raw_data
	TDenseMatrix raw_data(data_from_file.rows, data_from_file.cols+1);
        raw_data.Insert(0,0,data_from_file);
        raw_data.InsertColumnMatrix(0,data_from_file.cols,TDenseVector(data_from_file.rows,1.0));
	// Generate TData_predetermined from raw_data
        TData_predetermined input_data(n_lags,false,raw_data, TIndex(1,data_from_file.cols-1), TIndex(data_from_file.cols), n_lags, data_from_file.rows-1);// TIndex(8)(2)(35)(12)(20)(1), TIndex(data_from_file.cols), n_lags+16, data_from_file.rows-1); //

	//////////////////////////// Regime process: independent markov ///////////////////////
	// markov
	TRegimeProcessArray regime_process_array;
        if (!markov_regime_process_file.empty() )
        {
                try { regime_process_array = SetupMarkovArrayRegimeProcess(markov_regime_process_file.c_str()); }
                catch (dw_exception &e)
                {
                        cerr << "Error in reading or parsing " << markov_regime_process_file << endl << e.what() << endl;
                        abort();
                }
        }
	// deterministic
	if (!deterministic_regime_process_file.empty())
        {
                try
                {
                        int rows=74;
                        input_file.open(deterministic_regime_process_file.c_str());
                        if (!input_file.is_open())
                                throw dw_exception("unable to open regimes file");
                        TDenseMatrix regime_file(rows,3);
                        input_file >> regime_file;
                        input_file.close();
                        vector<int> regimes(rows);
                        for (int i=0; i < 2; i++)
                        {
                                int n_regimes = 1;
                                for (int t=0; t<rows; t++)
                                {
                                        regimes[t]=regime_file(t,i)-1;
                                        if (regimes[t] < 0)
                                                throw dw_exception("regimes cannot be negative");
                                        if (regimes[t] >= n_regimes)
                                                n_regimes=regimes[t]+1;
                                }
                                regime_process_array.push_back(new TRegimeProcess_deterministic(n_regimes,regimes));
                        }
                }
                catch (dw_exception &e)
                {
                        cerr << "Error in reading or parsing " << deterministic_regime_process_file << endl << e.what() << endl;
                        abort();
                }
        }
	TRegimeProcess_independent independent_regime_process(regime_process_array);
        TDenseVector parameters(independent_regime_process.NumberParameters(),0.0);
        independent_regime_process.SetParameters(parameters.vector);

	/////////////////////////// Hyper parameter ////////////////////////////////////////
	input_file.open(hyper_parameter_file.c_str(), fstream::in);
        if (!input_file)
        {
                cerr << "Error in opening " << hyper_parameter_file << endl;
                abort();
        }
        THyperParameter_Vector hyper_parameter(input_file);
        input_file.close();
	
	//////////////////////////// tvsbvar: /////////////////////////////////////////////////
	input_file.open(restriction_file.c_str(), fstream::in);
        if (!input_file)
        {
                cerr << "Error in opening" << restriction_file << endl;
                abort();
        }

        Sims_Zha::log2pi = 1.837877066409345;
        TVSBVAR tvsbvar_model(input_file, (TData_predetermined *)(&input_data), &independent_regime_process, &hyper_parameter, 4, 1.0);
        input_file.close();

        TDenseVector parameters_initial(tvsbvar_model.NumberParameters(),0.0), parameters_optimal;
        tvsbvar_model.DefaultParameters();
        tvsbvar_model.GetParameters(parameters_initial.vector);


	////////////////////////////  EquiEnergyModel ///////////////////////////////////////
	CEquiEnergy_TimeSeries simulation_model;
        simulation_model.target_model = &tvsbvar_model;
        simulation_model.timer_when_started = time(NULL);
        if (if_pure_MH)
                simulation_model.if_bounded = false;
        simulation_model.metropolis = new CMetropolis(&simulation_model); 
        simulation_model.parameter = &sim_option;
        simulation_model.current_sample = CSampleIDWeight(parameters_initial, 0, tvsbvar_model.LogPosterior(parameters_initial), true);
        CSampleIDWeight mode = simulation_model.current_sample;
	simulation_model.storage = new CStorageHead(my_rank, sim_option.run_id, sim_option.storage_marker, sim_option.storage_dir, sim_option.number_energy_stage); 

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
		cout << "Lambda : " << endl; 
		for (int i=0; i<sim_option.lambda.size(); i++)
			cout << setprecision(20) << sim_option.lambda[i] << "\t"; 
		cout << endl; 
		master_deploying(N_MESSAGE, nNode, nGroup, simulation_model, mode, diagnosis_option);
	}
	else
	{
		slave_computing(N_MESSAGE, simulation_model, mode);
	}
}
