#ifndef TASK_SCHEDULING_HEADER
#define TASK_SCHEDULING_HEADER

#include "option.hpp"

void master_deploying(const int, int, int, CEquiEnergyModel &, const CSampleIDWeight &mode, const Diagnosis &option=OPT_ESS );
void slave_computing(const int, CEquiEnergyModel &, const CSampleIDWeight &mode);
#endif
