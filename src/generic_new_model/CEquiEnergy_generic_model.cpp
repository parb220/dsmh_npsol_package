#include <ctime>
#include "dw_dense_matrix.hpp"
#include "generic_model.hpp"
#include "CEquiEnergy_generic_model.hpp"

double CEquiEnergy_GenericModel::log_posterior_function(CSampleIDWeight &x)
{
	if (!x.calculated)
	{
		x.reserved = target_model->log_likelihood_function(x.data.vector, x.data.dim); 
		x.weight = x.reserved + target_model->log_prior_function(x.data.vector, x.data.dim); 
		x.calculated = true; 
	}
	double bounded_log_posterior; 
	if (if_bounded)
		bounded_log_posterior = x.reserved*lambda + (x.weight-x.reserved); 
	else 
		bounded_log_posterior = x.weight; 
	return bounded_log_posterior; 
}

double CEquiEnergy_GenericModel::log_likelihood_function(const CSampleIDWeight &x)
{
	return target_model->log_likelihood_function(x.data.vector, x.data.dim);
}

double CEquiEnergy_GenericModel::log_prior_function(const CSampleIDWeight &x)
{
	return target_model->log_prior_function(x.data.vector, x.data.dim); 
}

double CEquiEnergy_GenericModel::log_posterior_function(const double *x, int n)
{
	double original_log_likelihood = target_model->log_likelihood_function(x, n); 
	double original_log_prior = target_model->log_prior_function(x,n); 
	if (if_bounded)
		return original_log_likelihood*lambda + original_log_prior; 
	else 
		return original_log_likelihood + original_log_prior; 
}

double CEquiEnergy_GenericModel::log_likelihood_function(const double *x, int n)
{
	return  target_model->log_likelihood_function(x, n);
}

double CEquiEnergy_GenericModel::log_prior_function(const double *x, int n) 
{
	return target_model->log_prior_function(x, n); 
}

bool CEquiEnergy_GenericModel::DrawParametersFromPrior(double *x) const
{
	return target_model->DrawParametersFromPrior(x, target_model->GetNumberParameters()); 
}

CEquiEnergy_GenericModel::CEquiEnergy_GenericModel() {}

CEquiEnergy_GenericModel::CEquiEnergy_GenericModel(bool _if_bounded, int eL, double _t, const CSampleIDWeight &_x, time_t _time, CMetropolis *_metropolis, CEESParameter *_parameter, CStorageHead *_storage, Generic_Model *_model) :
CEquiEnergyModel(_if_bounded, eL, _t, _x, _time, _metropolis, _parameter, _storage), target_model(_model)
{}

CEquiEnergy_GenericModel::~CEquiEnergy_GenericModel() {}

