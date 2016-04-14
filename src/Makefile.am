bin_PROGRAMS = test_sbvar_simulation test_tvsbvar_simulation generic_model_simulation test_sbvar_optimization test_tvsbvar_optimization generic_model_optimization

utilities_SOURCES = utilities_dw/dw_array.c utilities_dw/dw_ascii.c utilities_dw/dw_dense_matrix.cpp utilities_dw/dw_elliptical.c utilities_dw/dw_math.c utilities_dw/dw_matrix.c utilities_dw/dw_matrix_rand.c utilities_dw/dw_parse_cmd.c utilities_dw/dw_rand_gsl.c utilities_dw/bmatrix.c utilities_dw/dw_error.c utilities_dw/tzmatlab.c utilities_dw/dw_matrix_sort.c utilities_dw/dw_ascii_cpp.cpp
dsmh_basci_SOURCES = dsmh_basic/CEESParameter.cpp dsmh_basic/CEquiEnergyModel.cpp dsmh_basic/CEquiEnergyModel_Initialize.cpp dsmh_basic/CEquiEnergyModel_HillClimb.cpp dsmh_basic/CMetropolis.cpp dsmh_basic/CPutGetBin.cpp dsmh_basic/CSampleIDWeight.cpp dsmh_basic/CStorageHead.cpp dsmh_basic/Glob.cpp
dsmh_mpi_SOURCES = dsmh_mpi/DispatchSimulation.cpp dsmh_mpi/DispatchTuneSimulation.cpp dsmh_mpi/EstimateLogMDD.cpp dsmh_mpi/ExecutingSimulationTask.cpp dsmh_mpi/GetCommunicationParameter.cpp dsmh_mpi/HighestPlus1Stage.cpp dsmh_mpi/HighestPlus1Stage_Prior.cpp dsmh_mpi/master_deploying.cpp dsmh_mpi/mdd_function.c dsmh_mpi/slave_computing.cpp dsmh_mpi/master_mode_finding_deploying.cpp dsmh_mpi/slave_mode_finding_computing.cpp
time_series_SOURCES = time_series/dw_data.cpp time_series/regime_processes.cpp time_series/CEquiEnergy_TimeSeries.cpp
generic_new_model_SOURCES = generic_new_model/CEquiEnergy_generic_model.cpp

test_sbvar_simulation_SOURCES = time_series/sbvar.cpp time_series/sbvar_blocks.cpp time_series/test_sbvar_simulation.cpp $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES) $(time_series_SOURCES)

test_tvsbvar_simulation_SOURCES = time_series/sbvar.cpp time_series/sbvar_blocks.cpp time_series/hyper_parameter.cpp time_series/tvsbvar.cpp time_series/Sims_Zha.cpp time_series/Restriction.cpp time_series/tvsbvar_blocks.cpp time_series/test_tvsbvar_simulation.cpp $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES) $(time_series_SOURCES)

generic_model_simulation_SOURCES = generic_new_model/generic_model_simulation.cpp $(generic_new_model_SOURCES) $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES)

test_sbvar_optimization_SOURCES = time_series/test_sbvar_optimization.cpp time_series/sbvar.cpp time_series/sbvar_blocks.cpp $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES) $(time_series_SOURCES) 

test_tvsbvar_optimization_SOURCES = time_series/test_tvsbvar_optimization.cpp time_series/sbvar.cpp time_series/sbvar_blocks.cpp time_series/hyper_parameter.cpp time_series/tvsbvar.cpp time_series/Sims_Zha.cpp time_series/Restriction.cpp time_series/tvsbvar_blocks.cpp $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES) $(time_series_SOURCES)

generic_model_optimization_SOURCES = generic_new_model/generic_model_optimization.cpp $(generic_new_model_SOURCES) $(utilities_SOURCES) $(dsmh_basci_SOURCES) $(dsmh_mpi_SOURCES)

AM_CPPFLAGS = -Iinclude/utilities_dw -Iinclude/dsmh_basic -Iinclude/dsmh_mpi -Iinclude/time_series -Iinclude/generic_new_model -DDATADIR=\"$(pkgdatadir)\"
#AM_LDFLAGS = -L/home/hongweiwu/npsol/Linux-x86_64/optimized/lib -lnpsol_f77 -llssol_f77
