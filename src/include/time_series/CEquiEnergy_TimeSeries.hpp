#ifndef _EQUI_ENERGY_TIME_SERIES_
#define _EQUI_ENERGY_TIME_SERIES__

#include "CEquiEnergyModel.hpp"

class CSampleIDWeight; 
class TTimeSeries;  

class CEquiEnergy_TimeSeries : public CEquiEnergyModel
{
protected: 
	CSampleIDWeight original_sample; 
public: 
	TTimeSeries *target_model; 

	virtual double log_posterior_function(CSampleIDWeight &x); 
	virtual double log_likelihood_function(const CSampleIDWeight &x); 
	virtual double log_prior_function(const CSampleIDWeight &x); 
	
	virtual double log_posterior_function(const double *x, int n); 
	virtual double log_likelihood_function(const double *x, int n);
	virtual double log_prior_function(const double *x, int n);  
	virtual bool DrawParametersFromPrior(double *x) const; 

	CEquiEnergy_TimeSeries(); 
	CEquiEnergy_TimeSeries(bool _if_bounded, int eL, double _t, const CSampleIDWeight &_x, time_t _tim, CMetropolis *_metropolis, CEESParameter *_parameter, CStorageHead *_storage, TTimeSeries *_model); 
	virtual ~CEquiEnergy_TimeSeries(); 
};

#endif
