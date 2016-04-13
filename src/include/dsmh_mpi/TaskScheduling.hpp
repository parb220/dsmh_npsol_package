#ifndef TASK_SCHEDULING_HEADER
#define TASK_SCHEDULING_HEADER

#include "option.hpp"

void master_deploying(const int, int, int, CEquiEnergyModel &, const CSampleIDWeight &mode, const Diagnosis &option=OPT_ESS );
void slave_computing(const int, CEquiEnergyModel &, const CSampleIDWeight &mode);

void master_mode_finding_deploying(const int N_MESSAGE, int nNode, int n_optimization,  CEquiEnergyModel &model);
void slave_mode_finding_computing(const int N_MESSAGE, CEquiEnergyModel &model, int , int, double);

#endif
