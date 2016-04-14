#ifndef MASTER_DEPLOY_HEADER_
#define MASTER_DEPLOY_HEADER_

#include "option.hpp"

std::vector<CSampleIDWeight> HighestPlus1Stage(double *, double *, const int, int nNode, int nInitial, CEquiEnergyModel &model, const CSampleIDWeight &, double, double, std::ofstream &, std::ofstream &);

std::vector<CSampleIDWeight> HighestPlus1Stage_Prior(double *sPackage, double *rPackage, const int N_MESSAGE, int nNode, int nInitial, CEquiEnergyModel &model, ofstream &jump_file); 

void DispatchTuneSimulation(double *, double *, const int, int nNode, int nInitial, CEquiEnergyModel &model, const CSampleIDWeight &mode, int simulation_length, const Diagnosis &option=OPT_ESS, bool save_space_flag = true);

std::vector<CSampleIDWeight> DispatchSimulation(double *, double *, const int, int nNode, int nInitial, CEquiEnergyModel &model, int simulation_length, int stage, int message_tag, std::ofstream &) ; 


#endif
