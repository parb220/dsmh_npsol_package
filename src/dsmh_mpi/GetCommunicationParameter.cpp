#include "CEESParameter.hpp"
#include "mpi_constant.hpp"

bool GetCommunicationParameter(const double *rPackage, int package_size, CEESParameter *parameter)
{
	parameter->simulation_length = (int)(rPackage[LENGTH_INDEX]); 
	parameter->burn_in_length = (int)(rPackage[BURN_INDEX]); 
	parameter->THIN = (int)(rPackage[THIN_INDEX]); 
	parameter->pee = rPackage[PEE_INDEX]; 

        // parameter->SetTemperature();
	return true; 
}

