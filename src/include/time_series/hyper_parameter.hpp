#ifndef _HYPER_PARAMETER_
#define _HYPER_PARAMETER_

#include <istream>
#include <cmath>
#include "prcsn.h"
#include "dw_dense_matrix.hpp"
#include "dw_rand.h"
#include "dw_ascii.hpp"

using namespace std; 
class THyperParameter
{
public: 
	THyperParameter() {}; 
	THyperParameter(const THyperParameter &right) {}; 
	virtual ~THyperParameter() {}; 

	virtual bool SetParameters(double *parameter) = 0; 
	virtual bool GetParameters(double *parameter) const = 0; 
	virtual bool GetConstantParameters(double *parameter) const =0; 
	virtual bool GetVariableParameters(double *parameter) const =0; 
	virtual void DefaultParameters()  = 0; 
	virtual double LogPrior() const =0; 
	virtual double LogPrior(double *parameter) 
	{ 
		if (!SetParameters(parameter))
			return MINUS_INFINITY; 
		else 
			return LogPrior(); 
	}; 	
	virtual int NumberConstantParameters() const = 0; 
	virtual int NumberVariableParameters() const = 0; 
	virtual int NumberParameters() const {return NumberConstantParameters()+NumberVariableParameters();}; 

	virtual THyperParameter *Clone() const; 
}; 

class THyperParameter_Uniform : public THyperParameter
{
protected: 
	double a; 
	double b; 
	double x; 
		
public: 
	THyperParameter_Uniform(double _a=0.0, double _b=1.0) : THyperParameter(), a(_a<_b ? _a:_b), b(_b>_a ? _b:_a), x((b-a)*dw_uniform_rnd()+a) {}; 
	THyperParameter_Uniform(const THyperParameter_Uniform &right) : THyperParameter(right), a(right.a), b(right.b), x(right.x) {}; 
	virtual ~THyperParameter_Uniform() {};
	virtual bool SetParameters(double *parameter) 
	{ 
		x = *parameter; 
		if (x <a || x> b)
			return false; 
		else 
			return true; 
	}
        virtual bool GetVariableParameters(double *parameter) const { parameter[0] = x; return true; };

        virtual bool GetParameters(double *parameter) const { return GetVariableParameters(parameter); };
        virtual bool GetConstantParameters(double *parameter) const { return false; };
        virtual void DefaultParameters() { x=(b-a)*dw_uniform_rnd()+a; };

        virtual THyperParameter_Uniform *Clone() const {return new THyperParameter_Uniform(*this); };
        virtual int NumberConstantParameters() const { return 0; };
        virtual int NumberVariableParameters() const { return 1; };
	using THyperParameter::LogPrior; 
        virtual double LogPrior() const 
	{
		if (x < a || x > b) 
			return MINUS_INFINITY; 
		else 
			return -log(b-a); 
	}
};


class THyperParameter_Gamma : public THyperParameter
{
protected: 
	double a; 
	double b; 
	double x; 

public: 
	THyperParameter_Gamma(double _a=1.0, double _b=1.0) : THyperParameter(), a(_a), b(_b), x(_b*dw_gamma_rnd(_a)) {}; 
	THyperParameter_Gamma(const THyperParameter_Gamma &right) : THyperParameter(right), a(right.a), b(right.b), x(right.x) {}; 
	virtual ~THyperParameter_Gamma() {};

	virtual bool SetParameters(double *parameter) 
	{
		x = fabs(*parameter); 
		return true;
	};   
	virtual bool GetVariableParameters(double *parameter) const { parameter[0] = x; return true; }; 
	virtual bool GetParameters(double *parameter) const { return GetVariableParameters(parameter); }; 
	virtual bool GetConstantParameters(double *parameter) const { return false; }; 
	virtual void DefaultParameters() { x = b*dw_gamma_rnd(a); }; 

	virtual THyperParameter_Gamma *Clone() const {return new THyperParameter_Gamma(*this); }; 
	virtual int NumberConstantParameters() const { return 0; };
	virtual int NumberVariableParameters() const { return 1; };  
	using THyperParameter::LogPrior; 
	virtual double LogPrior() const { return -dw_log_gamma(a) + a*log(x/b) - log(x) - x/b; }
}; 

class THyperParameter_Vector : public THyperParameter
{
protected: 
	int dim; 
	TIndex constant_part; 
	TIndex vary_part; 
	vector<double> ConstantHyperParameter; 
	vector<THyperParameter *> ptrVariableHyperParameter; 

public:
	THyperParameter_Vector(); 
	THyperParameter_Vector(const THyperParameter_Vector &); 
	THyperParameter_Vector(std::istream &input); 
	virtual ~THyperParameter_Vector(); 
	
	virtual bool SetParameters(double *parameter); 
	virtual bool GetParameters(double *parameter) const; 
	virtual bool GetVariableParameters(double *parameter) const; 
	virtual bool GetConstantParameters(double *parameter) const; 
	virtual void DefaultParameters(); 
	
	virtual THyperParameter_Vector *Clone() const { return new THyperParameter_Vector(*this); }; 
	virtual int NumberVariableParameters() const { return vary_part.size; }; 
	virtual int NumberConstantParameters() const { return constant_part.size; }; 
	using THyperParameter::LogPrior; 
	virtual double LogPrior() const;   
};
#endif 
