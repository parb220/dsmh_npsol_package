#include <vector>
#include <ctime>
#include "dw_dense_matrix.hpp"
#include "dw_rand.h"
#include "dw_time_series.hpp"
#include "CEquiEnergy_TimeSeries.hpp"

using namespace std; 

double CEquiEnergy_TimeSeries :: log_posterior_function(CSampleIDWeight &x)
{
	if (!x.calculated)
	{
		x.reserved = target_model->LogLikelihood(x.data); 
		x.weight = x.reserved + target_model->LogPrior(x.data); 
		x.calculated = true; 
	}
	double bounded_log_posterior; 
	if (if_bounded)
		bounded_log_posterior = x.reserved*lambda + (x.weight-x.reserved); 
	else 
		bounded_log_posterior = x.weight; 
	return bounded_log_posterior; 
}

double CEquiEnergy_TimeSeries :: log_likelihood_function(const CSampleIDWeight &x)
{
	return target_model->LogLikelihood(x.data.vector); 
}

double CEquiEnergy_TimeSeries :: log_prior_function(const CSampleIDWeight &x)
{
	return target_model->LogPrior(x.data.vector); 
}

double CEquiEnergy_TimeSeries :: log_posterior_function(const double *x, int n)
{
	double raw_lpf = target_model->LogPosterior((double *)x); 
	double bounded_log_posterior; 
	if (if_bounded)
	{
		double log_prior = target_model->LogPrior((double *)x);
		bounded_log_posterior = (raw_lpf-log_prior)*lambda + log_prior; 
	}
	else 
		bounded_log_posterior = raw_lpf;
	return bounded_log_posterior; 
}

double CEquiEnergy_TimeSeries :: log_likelihood_function(const double *x, int n)
{
	return target_model->LogLikelihood((double *)x); 
}

double CEquiEnergy_TimeSeries :: log_prior_function(const double *x, int n)
{
	return target_model->LogPrior((double *)x); 
}

CEquiEnergy_TimeSeries::CEquiEnergy_TimeSeries() : 
CEquiEnergyModel(), target_model(NULL)
{}

CEquiEnergy_TimeSeries::CEquiEnergy_TimeSeries(bool _if_bounded, int eL, double _t, const CSampleIDWeight &_x, time_t _time, CMetropolis *_metropolis, CEESParameter *_parameter, CStorageHead *_storage, TTimeSeries* _model) : 
CEquiEnergyModel(_if_bounded, eL, _t, _x, _time, _metropolis, _parameter, _storage), target_model(_model)
{}

CEquiEnergy_TimeSeries::~CEquiEnergy_TimeSeries()
{}

bool CEquiEnergy_TimeSeries::DrawParametersFromPrior(double *x) const
{
	return target_model->DrawParametersFromPrior(x); 
}
