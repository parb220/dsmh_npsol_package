#include "prcsn.h"
#include "hyper_parameter.hpp"
#include "dw_ascii.hpp"
int ParseRestrictionMatrix(std::vector< std::vector<int> > &R, TDenseMatrix &C, const StringMatrix &M, int pos, int n_rows, int n_cols=-1);
int FindIdentifier(const StringMatrix &M, const std::string &id);
int ParseInteger(int &k, const StringMatrix &M, int pos);

THyperParameter * THyperParameter::Clone() const
{
	throw dw_exception("Cannot creat an instance of the virtual class THyperParameter"); 
	return (THyperParameter *)NULL; 
}

// parameters only contain the variable part
bool THyperParameter_Vector :: SetParameters(double *parameter)
{
	bool return_value = true; 
	for (int i=0; i<vary_part.size; i++)
	{
		if (!ptrVariableHyperParameter[i]->SetParameters(parameter+i))
			return_value = false; 
	}

	return return_value; 
}

// parameters only contain the constant and variable parts
bool THyperParameter_Vector :: GetParameters(double *parameter) const
{
	for (int i=0; i<constant_part.size; i++)
		parameter[constant_part[i]] = ConstantHyperParameter[i]; 
	for (int i=0; i<vary_part.size; i++)
	{
		if ( !ptrVariableHyperParameter[i]->GetParameters(parameter+vary_part[i]))
			return false; 
	}
	return true; 
}

bool THyperParameter_Vector :: GetConstantParameters(double *parameter) const
{
        for (int i=0; i<constant_part.size; i++)
                parameter[constant_part[i]] = ConstantHyperParameter[i];        
	return true;  
}

bool THyperParameter_Vector :: GetVariableParameters(double *parameter) const
{
        for (int i=0; i<vary_part.size; i++)
	{
		if (!ptrVariableHyperParameter[i]->GetParameters(parameter+i))
			return false; 
	}                     
        return true;
}


void THyperParameter_Vector :: DefaultParameters()
{
	for (int i=0; i<vary_part.size; i++)
		ptrVariableHyperParameter[i]->DefaultParameters(); 
}

double THyperParameter_Vector :: LogPrior() const
{
	double log_prior = 0.0; 
	for (int i=0; i<vary_part.size; i++)
	{
		if (ptrVariableHyperParameter[i]->LogPrior() == MINUS_INFINITY)
			return MINUS_INFINITY; 
		else 
			log_prior += ptrVariableHyperParameter[i]->LogPrior(); 
	}
	return log_prior; 
}

THyperParameter_Vector::THyperParameter_Vector() : 
dim(0), 
constant_part(TIndex()), 
vary_part(TIndex()), 
ConstantHyperParameter(vector<double>(0)), 
ptrVariableHyperParameter(vector<THyperParameter*>(0)) 
{}

THyperParameter_Vector::THyperParameter_Vector(const THyperParameter_Vector &right) : 
dim(right.dim), 
constant_part(right.constant_part), 
vary_part(right.vary_part), 
ConstantHyperParameter(vector<double>(right.ConstantHyperParameter)), 
ptrVariableHyperParameter(vector<THyperParameter*>(right.ptrVariableHyperParameter.size()))
{
	for (int i=0; i<vary_part.size; i++)
		ptrVariableHyperParameter[i] = right.ptrVariableHyperParameter[i]->Clone(); 
}

THyperParameter_Vector::~THyperParameter_Vector()
{
	for (int i=0; i<vary_part.size; i++)
		if (ptrVariableHyperParameter[i])
			delete ptrVariableHyperParameter[i]; 
}

THyperParameter_Vector::THyperParameter_Vector(std::istream &input) : 
dim(0), 
constant_part(TIndex()),
vary_part(TIndex()),
ConstantHyperParameter(vector<double>(0)),
ptrVariableHyperParameter(vector<THyperParameter*>(0))
{
	StringMatrix M(input, string("//**")); 
	int pos, _nhyper; 
	if ( (pos=FindIdentifier(M, "NumberHyperParameter")) <0 || !ParseInteger(_nhyper, M, pos+1) || _nhyper <0 )
		throw dw_exception("THyperParameter_Vector:: construction failed on parsing //== NumberHyperParameter"); 
	dim = _nhyper; 

	vector<vector<int> > R; 
	TDenseMatrix C; 
	string id = string("HyperParameter"); 
	if ( (pos=FindIdentifier(M, id)) >= 0 && ParseRestrictionMatrix(R,C,M,pos+1,1,_nhyper) )
	{
		for (int ii=0; ii<_nhyper; ii++)
		{
			if (R[0][ii] == 1) 
				vary_part += ii; 
			else 
			{
				constant_part += ii; 
				ConstantHyperParameter.push_back(C(0, ii)); 
			}
		}
	}
	else 
		throw dw_exception("THyperParameter_Vector:: construction failed on parsing //== " + id); 	
	// variable hyper parameter assuming them to be Uniform, needs to read in gamma parameters
	for (int kk=0; kk<vary_part.size; kk++)
	{
		if( (pos=FindIdentifier(M, id=string("Uniform[")+cluster_to_string(vary_part[kk])+string("]"))) >= 0 && ParseRestrictionMatrix(R,C,M,pos+1,1,2))
			ptrVariableHyperParameter.push_back(new THyperParameter_Uniform(C(0,0), C(0,1)));
		else  	
			throw dw_exception("THyperParameter_Vector:: construction failed on parsing //== " + id); 
	}

	/* variable hyper parameter assuming them to be Gamma, needs to read in gamma parameters
	for (int kk=0; kk<vary_part.size; kk++)
	{
		if( (pos=FindIdentifier(M, id=string("Gamma[")+cluster_to_string(vary_part[kk])+string("]"))) >= 0 && ParseRestrictionMatrix(R,C,M,pos+1,1,2))
			ptrVariableHyperParameter.push_back(new THyperParameter_Gamma(C(0,0), C(0,1)));
		else  	
			throw dw_exception("THyperParameter_Vector:: construction failed on parsing //== " + id); 
	}*/
}
