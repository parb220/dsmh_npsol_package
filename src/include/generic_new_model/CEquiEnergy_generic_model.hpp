#ifndef EQUI_ENERGY_GENERIC_MODEL
#define EQUI_ENERGY_GENERIC_MODEL

#include "CEquiEnergyModel.hpp"

class CSampleIDWeight; 
class Generic_Model; 

class CEquiEnergy_GenericModel : public CEquiEnergyModel
{
public:
	Generic_Model *target_model; 
	virtual double log_posterior_function(CSampleIDWeight &x); 
	virtual double log_likelihood_function(const CSampleIDWeight &x); 
	virtual double log_prior_function(const CSampleIDWeight &x); 
	virtual double log_posterior_function(const double *x, int n);
        virtual double log_likelihood_function(const double *x, int n);
        virtual double log_prior_function(const double *x, int n); 
	virtual bool  DrawParametersFromPrior(double *x) const; 
	
	CEquiEnergy_GenericModel(); 
	CEquiEnergy_GenericModel(bool _if_bounded, int eL, double _t, const CSampleIDWeight &_x, time_t _tim, CMetropolis *_metropolis, CEESParameter *_parameter, CStorageHead *_storage, Generic_Model *_model);
	virtual ~CEquiEnergy_GenericModel(); 
};

#endif
