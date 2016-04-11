#ifndef MPI_CONSTANT_HEADER_
#define MPI_CONSTANT_HEADER_

const int LENGTH_INDEX = 0; 
const int LEVEL_INDEX = 1; 
const int BURN_INDEX = 2; 
const int SCALE_INDEX = 3; 
const int THIN_INDEX = 4; 
const int GROUP_INDEX = 5; 
const int PEE_INDEX = 6; 
const int GROUP_NUMBER_INDEX = 7; 

const int RETURN_INDEX_0 = 10; 
const int RETURN_INDEX_1 = 11; 
const int RETURN_INDEX_2 = 12; 
const int RETURN_INDEX_3 = 13; 
const int RETURN_INDEX_4 = 14; 
const int RETURN_INDEX_5 = 15; 

const int RESERVE_INDEX_START = 20; 

const int BURN_IN_LENGTH = 5000; 
const int SIMULATION_LENGTH = 50000; 

const int TUNE_TAG_BEFORE_SIMULATION = 28; 

const int TUNE_TAG_SIMULATION_FIRST = 32; 

const int SIMULATION_TAG = 60; 
const int SIMULATION_PRIOR_TAG = 63; 

const int BINNING_INFO  = 1000; 

const int SCALE_MATRIX_FIT_TAG = 91; 
const int HILL_CLIMB_TAG = 70;

const int END_TAG = 0;  
const int TASK_LENGTH_LONG = 1000; 
const int TASK_LENGTH_SHORT = 100; 

const double LOWER_ALPHA = 0.234*0.8; 
const double UPPER_ALPHA = 0.234*1.25; 
#endif
