#ifndef GENERIC_NEW_MODEL_EXAMPLE_HEADER
#define GENERIC_NEW_MODEL_EXAMPLE_HEADER

#include "generic_model.hpp"

class Generic_Model_Example : public Generic_Model
{
public:
	///////////////////////////////////////////////////////////////////////////////
	//
	// This example is to check whether the interface of generic model works
	// The functions need to be rewritten
	//
	///////////////////////////////////////////////////////////////////////////////
	virtual double log_posterior_function(const double *x, int n) { return 0.0; }
	virtual double log_likelihood_function(const double *x, int n) { return 0.0; }
	virtual double log_prior_function(const double *x, int n)  { return 0.0; }
	virtual bool DrawParametersFromPrior(double *x, int n) { return true; }
	virtual int GetNumberParameters() const { return 0; } 

	Generic_Model_Example() {}
	virtual ~Generic_Model_Example() {} 
}; 

#endif
