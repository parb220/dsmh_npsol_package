#include <vector>
#include <string>
#include "CEquiEnergyModel.hpp"
#include "mpi_constant.hpp"
#include "master_deploying.hpp"

using namespace std; 

vector<string> glob(const string &pattern); 

std::vector<CSampleIDWeight> HighestPlus1Stage_Prior(double *sPackage, double *rPackage, const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model, ofstream &jump_file)
{
	return DispatchSimulation(sPackage, rPackage, N_MESSAGE, nNode, nInitial, model, model.parameter->simulation_length,  model.parameter->number_energy_stage, SIMULATION_PRIOR_TAG, jump_file);
}
