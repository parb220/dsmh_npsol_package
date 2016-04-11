#ifndef ESTIMATE_LOG_MDD_HEADER_
#define ESTIMATE_LOG_MDD_HEADER_

const int POSTERIOR_HEATED = 0;
const int LIKELIHOOD_HEATED = 1;
const int PRIOR_ONLY = 2;

double LogMDD(const vector<CSampleIDWeight> &posterior, CEquiEnergyModel &model, double t, int proposal_type, int posterior_type = POSTERIOR_HEATED);

double LogMDD_Importance(const vector<CSampleIDWeight> &proposal, CEquiEnergyModel &model,  int previous_stage, int current_stage, double logMDD_previous, int posterior_type=POSTERIOR_HEATED);

double CheckConvergency(std::vector<CSampleIDWeight> &samples, CEquiEnergyModel &model, int stage, int previous_stage,  double convergency_previous, double &average_consistency, double &std_consistency, double &LB_ESS, int posterior_type = POSTERIOR_HEATED, int nGroup_NSE = 1);

#endif 
