#include <algorithm>
#include <cmath>

#include "prcsn.h"
#include "dw_ascii.hpp"
#include "tvsbvar.hpp"
#include "dw_math.h"
#include "sbvar.hpp"

string cluster_to_string(int);
int ParseRestrictionMatrix(std::vector< std::vector<int> > &R, TDenseMatrix &C, const StringMatrix &M, int pos, int n_rows, int n_cols=-1);
int FindIdentifier(const StringMatrix &M, const std::string &id);
int ParseInteger(int &k, const StringMatrix &M, int pos); 

//===============================================================================
//=== class TVSBVAR
//===============================================================================
int TVSBVAR::RegimeIndex(int base_index, int mult_index, int add_index) const
{
	return add_index + mult_index*add_restrictions.n_regimes + base_index*add_restrictions.n_regimes*mult_restrictions.n_regimes; 
	//return base_index + mult_index*base_restrictions.n_regimes + add_index*base_restrictions.n_regimes*mult_restrictions.n_regimes; 
}

double TVSBVAR::LogPrior() 
{
	double log_hyper = hyper_parameter->LogPrior(); 
	if (log_hyper <= MINUS_INFINITY)
		return MINUS_INFINITY; 
	double log_regime_process = regime_process->LogPrior(); 
	if (log_regime_process <= MINUS_INFINITY)
		return MINUS_INFINITY; 
	double log_sims_zha = LogPrior_SimsZha(); 
	if (log_sims_zha <= MINUS_INFINITY)
		return MINUS_INFINITY; 
	return log_sims_zha+log_regime_process+log_hyper; 
}

double TVSBVAR::LogPrior_SimsZha_Base(const TDenseVector &parameters) const
{
	// LogPrior_SimsZhaBase.m
	double log_prior = 0.0; 
	TDenseVector aa, bb; 
	if (!sims_zha.base_flat_prior)
	{
		log_prior += sims_zha.base_prior_constant; 
		for (int kk=0; kk<base_restrictions.n_regimes; kk++)
		{
			for (int ii=0; ii<n_vars; ii++)
			{
				aa = parameters.SubVector(sims_zha.select_a[kk][ii])-sims_zha.c_a[kk][ii]; 
				bb = parameters.SubVector(sims_zha.select_aplus[kk][ii])-sims_zha.c_aplus[kk][ii] - Multiply(sims_zha.P[kk][ii], aa); 
				log_prior += -0.5*(InnerProduct(aa,aa,sims_zha.S[kk][ii])+InnerProduct(bb,bb,sims_zha.H[kk][ii])); 
			}
		}
	}
	return log_prior; 
}

double TVSBVAR::LogPrior_SimsZha_Mult(const TDenseVector &parameters) const
{
	// LogPrior_GammaMultiplicative.m
	double log_prior = 0.0; 
	if (!sims_zha.mult_flat_prior)
	{
		log_prior += sims_zha.mult_prior_constant; 
		int idx = mult_restrictions.offset[0]; 
		for (int jj=0; jj<mult_restrictions.n_free; jj++)
		{
		  //if (parameters[idx+jj] <= 0) return MINUS_INFINITY;
		  double x=fabs(parameters[idx+jj]);
			// log_prior += (sims_zha.mult_prior_a[jj]-1.0)*log(parameters[idx+jj])-parameters[idx+jj]/sims_zha.mult_prior_b[jj];  // consistent with TSBVAR_mult.m
		  log_prior += (2*sims_zha.mult_prior_a[jj]-1.0)*log(x) - x*x/sims_zha.mult_prior_b[jj]; // consistent with TSBVAR_mult_square.m 
		}
	}
	return log_prior; 
}

double TVSBVAR::LogPrior_SimsZha_Add(const TDenseVector &parameters) const
{
	// LogPrior_GaussianAdditive.m
	double log_prior = 0.0; 
	if (~sims_zha.add_flat_prior)
	{
		log_prior += sims_zha.add_prior_constant; 
		int idx = add_restrictions.offset[0]; 
		for (int jj=0; jj<add_restrictions.n_free; jj++)
			log_prior += -0.5*(parameters[idx+jj]-sims_zha.add_prior_mean[jj])*(parameters[idx+jj]-sims_zha.add_prior_mean[jj])/sims_zha.add_prior_variance[jj]; 
	}	
	return log_prior; 
}

double TVSBVAR::LogPrior_SimsZha() const
{
	// LogPrior_SimsZha.m
	// TDenseVector parameters = VARToParameters(A0,Aplus,Xi); 
	double log_prior_base = LogPrior_SimsZha_Base(internal_parameters); 
	if (log_prior_base <= MINUS_INFINITY)
		return MINUS_INFINITY; 
	double log_prior_mult = LogPrior_SimsZha_Mult(internal_parameters); 
	if (log_prior_mult <= MINUS_INFINITY)
		return MINUS_INFINITY; 
	double log_prior_add = LogPrior_SimsZha_Add(internal_parameters); 
	if (log_prior_add <= MINUS_INFINITY)
		return  MINUS_INFINITY; 
	return log_prior_base + log_prior_mult + log_prior_add; 
}

bool TVSBVAR::DrawParametersFromPrior_SimsZha_Base(TDenseVector &parameters)
{
  // LogPrior_SimsZhaBase.m
  TDenseVector aa, bb;
  if (!sims_zha.base_flat_prior)
    {
      for (int kk=0; kk < base_restrictions.n_regimes; kk++)
	{
	  for (int ii=0; ii < n_vars; ii++)
	    {
	      aa=sims_zha.SqrtInvS[kk][ii]*RandomNormalVector(sims_zha.S[kk][ii].cols) + sims_zha.c_a[kk][ii];
	      bb=sims_zha.SqrtInvH[kk][ii]*RandomNormalVector(sims_zha.H[kk][ii].rows) + sims_zha.c_aplus[kk][ii] + sims_zha.P[kk][ii]*aa;
	      parameters.Insert(sims_zha.select_a[kk][ii],aa);
	      parameters.Insert(sims_zha.select_aplus[kk][ii],bb);

	      // log prior calculation
	      //aa = parameters.SubVector(sims_zha.select_a[kk][ii])-sims_zha.c_a[kk][ii]; 
	      //bb = parameters.SubVector(sims_zha.select_aplus[kk][ii])-sims_zha.c_aplus[kk][ii] - Multiply(sims_zha.P[kk][ii], aa); 
	      //log_prior += -0.5*(InnerProduct(aa,aa,sims_zha.S[kk][ii])+InnerProduct(bb,bb,sims_zha.H[kk][ii])); 
	    }
	}
      return true;
    }
  return false;
}

bool TVSBVAR::DrawParametersFromPrior_SimsZha_Mult(TDenseVector &parameters)
{
  if (!sims_zha.mult_flat_prior)
    {
      // the square of each parameters is gamma and each parameter can be positive or negative with equal probability  
      int idx = mult_restrictions.offset[0]; 
      for (int jj=0; jj < mult_restrictions.n_free; jj++)
	parameters[idx+jj]=((dw_uniform_rnd() < 0.5) ? -1.0 : 1.0)*sqrt(sims_zha.mult_prior_b[jj]*dw_gamma_rnd(sims_zha.mult_prior_a[jj]));
      return true;
    }
  return (mult_restrictions.n_free > 0) ? false : true;
}

bool TVSBVAR::DrawParametersFromPrior_SimsZha_Add(TDenseVector &parameters)
{
  if (!sims_zha.add_flat_prior)
    {
      // each parameter is normal
      int idx = add_restrictions.offset[0]; 
      for (int jj=0; jj < add_restrictions.n_free; jj++)
	parameters[idx+jj]=sqrt(sims_zha.add_prior_variance[jj])*dw_gaussian_rnd() + sims_zha.add_prior_mean[jj];
      return true;
    }	
  return (add_restrictions.n_free > 0) ? false : true;
}

bool TVSBVAR::DrawParametersFromPrior_SimsZha(TDenseVector &parameters)
{
  if (!DrawParametersFromPrior_SimsZha_Base(parameters)) return false; 
  if (!DrawParametersFromPrior_SimsZha_Mult(parameters)) return false; 
  if (!DrawParametersFromPrior_SimsZha_Add(parameters)) return false; 
  return true;
}

double TVSBVAR::LogLikelihood(void)
{
#define LOG_ZERO -1.0E300;

  int n_regimes=NumberRegimes(), n_obs=NumberObservations();
  TDenseVector log_constant(n_regimes);
  for (int s=0; s < n_regimes; s++)
    log_constant(s)=-0.918938533204673*n_vars + LogAbsDeterminant(Xi[s]*A0[s]);   // -0.5*n_vars*log(2*pi) + log(abs(det(Xi(k)*A(k))))

  double log_likelihood=0.0, scale;
  TDenseVector epsilon, p_tm1(n_regimes), p_t(n_regimes);

  //=== Hamilton Filter ======================================================
  for (int t=0; t < n_obs; t++)
    {
      // compute p_tm1(k) = p( s_t = k | Y(t-1) )
      p_tm1=(t == 0) ? InitialProbabilities() : TransitionMatrix(t-1)*p_t;

      // compute log conditional probabilities and scale
      scale=LOG_ZERO;	// scale = MINUS_INFINITY; 
      for (int s_t=0; s_t < n_regimes; s_t++)
	if (p_tm1(s_t) > 0.0)
	  {
	    epsilon=Xi[s_t]*(A0[s_t]*Data(t) - Aplus[s_t]*((TData_predetermined *)data)->PredeterminedData(t));
	    scale=AddLogs(scale,p_t.vector[s_t]=log(p_tm1.vector[s_t]) + log_constant.vector[s_t] - 0.5*InnerProduct(epsilon,epsilon));
	  }

      // update log likelihood
      log_likelihood=log_likelihood + scale;
    
      // update p_t(t) = p( s_t = k | Y(t) )
      for (int s_t=0; s_t < n_regimes; s_t++)
	p_t(s_t)=(p_tm1(s_t) > 0.0) ? exp(p_t(s_t) - scale) : 0.0;
    }

  return log_likelihood;

#undef LOG_ZERO
}

TVSBVAR::TVSBVAR() : 
TTimeSeries_TV(), 
internal_parameters(TDenseVector(0)), 
n_vars(0), n_lags(0), n_predetermined(0), 
periods_per_year(4), variance_scale(1.0), 
A0(vector<TDenseMatrix>(0)), Aplus(vector<TDenseMatrix>(0)), Xi(vector<TDenseMatrix>(0)),
RA_base(vector<TDenseMatrix>(0)), RAplus_base(vector<TDenseMatrix>(0)), RXi_base(vector<TDenseMatrix>(0)),
RA_mult(vector<TDenseMatrix>(0)), RAplus_mult(vector<TDenseMatrix>(0)), RXi_mult(vector<TDenseMatrix>(0)),
RA_add(vector<TDenseMatrix>(0)), RAplus_add(vector<TDenseMatrix>(0)), RXi_add(vector<TDenseMatrix>(0)),
base_restrictions(), mult_restrictions(), add_restrictions(), 
sims_zha(), index_mapping(vector<vector<int> >(0)), 
hyper_parameter(NULL), hyper_offset(0), 
set_parameter_counter(0)
{
}

TVSBVAR::TVSBVAR(const TVSBVAR &rhs) : 
TTimeSeries_TV(rhs), 
internal_parameters(TDenseVector(rhs.internal_parameters.dim,0.0)), 
n_vars(rhs.n_vars), n_lags(rhs.n_lags), n_predetermined(rhs.n_predetermined), 
periods_per_year(rhs.periods_per_year), 
variance_scale(rhs.variance_scale), 
A0(vector<TDenseMatrix>(rhs.A0.size())),
Aplus(vector<TDenseMatrix>(rhs.Aplus.size())),
Xi(vector<TDenseMatrix>(rhs.Xi.size())),
RA_base(vector<TDenseMatrix>(rhs.RA_base.size())), RAplus_base(vector<TDenseMatrix>(rhs.RAplus_base.size())), RXi_base(vector<TDenseMatrix>(rhs.RXi_base.size())),
RA_mult(vector<TDenseMatrix>(rhs.RA_mult.size())), RAplus_mult(vector<TDenseMatrix>(rhs.RAplus_mult.size())), RXi_mult(vector<TDenseMatrix>(rhs.RXi_mult.size())),
RA_add(vector<TDenseMatrix>(rhs.RA_add.size())), RAplus_add(vector<TDenseMatrix>(rhs.RAplus_add.size())), RXi_add(vector<TDenseMatrix>(rhs.RXi_add.size())),
base_restrictions(rhs.base_restrictions),
mult_restrictions(rhs.mult_restrictions),
add_restrictions(rhs.add_restrictions), 
sims_zha(rhs.sims_zha), 
index_mapping(rhs.index_mapping), 
hyper_parameter(rhs.hyper_parameter->Clone()), 
hyper_offset(rhs.hyper_offset),
set_parameter_counter(rhs.set_parameter_counter)
{
	internal_parameters.CopyContent(rhs.internal_parameters); 
	for (int kk=0; kk<(int)(A0.size()); kk++)
		A0[kk].CopyContent(rhs.A0[kk]); 
	for (int kk=0; kk<(int)(Aplus.size()); kk++)
		Aplus[kk].CopyContent(rhs.Aplus[kk]); 
	for (int kk=0; kk<(int)(Xi.size()); kk++)
		Xi[kk].CopyContent(rhs.Xi[kk]); 

	for (int kk=0; kk<(int)(RA_base.size()); kk++)
		RA_base[kk].CopyContent(rhs.RA_base[kk]); 
	for (int kk=0; kk<(int)(RAplus_base.size()); kk++)
		RAplus_base[kk].CopyContent(rhs.RAplus_base[kk]); 
	for (int kk=0; kk<(int)(RXi_mult.size()); kk++)
		RXi_base[kk].CopyContent(rhs.RXi_base[kk]); 
	
	for (int kk=0; kk<(int)(RA_mult.size()); kk++)
		RA_mult[kk].CopyContent(rhs.RA_mult[kk]); 
	for (int kk=0; kk<(int)(RAplus_mult.size()); kk++)
		RAplus_mult[kk].CopyContent(rhs.RAplus_mult[kk]); 
	for (int kk=0; kk<(int)(RXi_mult.size()); kk++)
		RXi_mult[kk].CopyContent(rhs.RXi_mult[kk]); 
	
	for (int kk=0; kk<(int)(RA_add.size()); kk++)
		RA_add[kk].CopyContent(rhs.RA_add[kk]); 
	for (int kk=0; kk<(int)(RAplus_add.size()); kk++)
		RAplus_add[kk].CopyContent(rhs.RAplus_add[kk]); 
	for (int kk=0; kk<(int)(RXi_add.size()); kk++)
		RXi_add[kk].CopyContent(rhs.RXi_add[kk]); 
}


TVSBVAR::TVSBVAR(istream &restriction_file, TData_predetermined *_data, TRegimeProcess *_regime_process, THyperParameter *_hyper_parameter, int _periods_per_year, double _variance_scale) : TTimeSeries_TV(_data, _regime_process, 0), 
internal_parameters(TDenseVector(0)), 
n_vars(_data->NumberVariables()), n_lags(_data->NumberLags()), n_predetermined(_data->NumberPredeterminedVariables()), periods_per_year(_periods_per_year), variance_scale(_variance_scale), 
A0(vector<TDenseMatrix>(_regime_process->NumberRegimes(), TDenseMatrix(_data->NumberVariables(), _data->NumberVariables(),0.0)) ), 
Aplus(vector<TDenseMatrix>(_regime_process->NumberRegimes(), TDenseMatrix(_data->NumberVariables(), _data->NumberPredeterminedVariables(),0.0))), 
Xi(vector<TDenseMatrix>(_regime_process->NumberRegimes(), TDenseMatrix(_data->NumberVariables(), _data->NumberVariables(),0.0)) ),
RA_base(vector<TDenseMatrix>(0)), RAplus_base(vector<TDenseMatrix>(0)), RXi_base(vector<TDenseMatrix>(0)),
RA_mult(vector<TDenseMatrix>(0)), RAplus_mult(vector<TDenseMatrix>(0)), RXi_mult(vector<TDenseMatrix>(0)),
RA_add(vector<TDenseMatrix>(0)), RAplus_add(vector<TDenseMatrix>(0)), RXi_add(vector<TDenseMatrix>(0)),
base_restrictions(), mult_restrictions(), add_restrictions(), sims_zha(), 
index_mapping(vector<vector<int> >(_regime_process->NumberRegimes(),vector<int>(3,0))), 
hyper_parameter(_hyper_parameter->Clone()), 
hyper_offset(0),
set_parameter_counter(0)
// offset and hyper_offset are yet incorrectly set 
{
	// SetupVAR.m
	// restrictions 
	if (!SetupRestrictions(restriction_file))
		throw dw_exception("TVSBVAR::TVSBVAR() : error in SetupRestrictionMatrices()"); 
	offset = base_restrictions.n_free + mult_restrictions.n_free + add_restrictions.n_free; 
	hyper_offset = offset + NumberParameters_RegimeProcess(); 

	if (base_restrictions.n_regimes * mult_restrictions.n_regimes * add_restrictions.n_regimes != _regime_process->NumberRegimes())
		throw dw_exception("TVSBVAR::TVSBVAR() : number of regimes determined in parameter restriction file does not match that of regime process restriction file."); 
	
	// index_mapping
	for (int kk_base=0; kk_base<base_restrictions.n_regimes; kk_base++)
	{
		for (int kk_mult=0; kk_mult<mult_restrictions.n_regimes; kk_mult++)
		{
			for (int kk_add=0; kk_add<add_restrictions.n_regimes; kk_add++)
			{
				int regime_index=RegimeIndex(kk_base,kk_mult,kk_add); 
				index_mapping[regime_index][0]=kk_base; 		
				index_mapping[regime_index][1]=kk_mult; 
				index_mapping[regime_index][2]=kk_add; 
			}
		}
	}

	// Prior for VAR coefficients
	TDenseVector x_hyper(hyper_parameter->NumberVariableParameters()+hyper_parameter->NumberConstantParameters(),0.0);
	hyper_parameter->GetParameters(x_hyper.vector);  
	// SetupPrior_SimsZha must be callsed during the construction
	bool if_setup_prior_simszha = sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data); 
	if (!hyper_parameter->NumberVariableParameters() && !if_setup_prior_simszha)
		throw dw_exception("TVSBVAR::TVSBVAR() : hyper parameter invalid.");
	while (!if_setup_prior_simszha)
	{
		hyper_parameter->DefaultParameters(); 
		hyper_parameter->GetParameters(x_hyper.vector);
		if_setup_prior_simszha = sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data); 
	}
		
	
	//  Initialize A0, Aplus, Xi
	// (parameters for regime_process should have already been set-up during its construction)
	TDenseVector parameter_initial = InitialParameterValues_SimsZha(); 
	while (!SetParameters_SimsZha(parameter_initial.vector))
		parameter_initial = RandomNormalVector(parameter_initial.dim); 

	// internal parameters	
	internal_parameters.Zeros(NumberParameters()); 
	internal_parameters.Insert(0, parameter_initial); 
	regime_process->GetParameters(internal_parameters.vector+offset); 
	hyper_parameter->GetVariableParameters(internal_parameters.vector+hyper_offset); 
}

bool TVSBVAR::SetupRestrictions_Regime(const StringMatrix &M, const string &prefix, vector<TDenseMatrix> &RA, vector<TDenseMatrix> &RAplus, vector<TDenseMatrix> &RXi)
{	
	string id; 
	int pos, _nregime; 
	// number of regimes
	if ( (pos=FindIdentifier(M, id=prefix+string("::NumberRegimes")) ) < 0 || !ParseInteger(_nregime, M, pos+1) || _nregime <= 0)
		throw dw_exception("SetupRestrictions_Regime(): error parsing //==" + id); 

	RA = vector<TDenseMatrix>(_nregime,TDenseMatrix(n_vars,n_vars,0.0)); 
	RAplus = vector<TDenseMatrix>(_nregime,TDenseMatrix(n_vars,n_predetermined,0.0)); 
	RXi = vector<TDenseMatrix>(_nregime,TDenseMatrix(n_vars,n_vars,0.0)); 

	vector<vector<int> > R; 
	TDenseMatrix C; 
	for (int kk=0; kk<_nregime; kk++)
	{
		if( (pos=FindIdentifier(M, id=prefix+string("::A[")+cluster_to_string(kk)+string("]")) )>= 0 )
		{
			if (!ParseRestrictionMatrix(R,C, M, pos+1, n_vars, n_vars) )
				throw dw_exception("SetupRestrictions_Regime: error parsing //== " + id); 
			for (int ii=0; ii<n_vars; ii++)
			{
				for (int jj=0; jj<n_vars; jj++)
					RA[kk](ii,jj) = R[ii][jj] == 1 ? NAN : C(ii,jj); 
			}
		} 

		if ( (pos=FindIdentifier(M, id=prefix+string("::Aplus[")+cluster_to_string(kk)+string("]")) )>= 0 )
		{
			if (!ParseRestrictionMatrix(R,C, M, pos+1, n_vars, n_predetermined) )
				throw dw_exception("SetupRestrictions_Regime: error parsing //== " + id);
			for (int ii=0; ii<n_vars; ii++)
			{
				for (int jj=0; jj<n_predetermined; jj++)
					RAplus[kk](ii,jj) = R[ii][jj] == 1 ? NAN : C(ii,jj); 
			}
		}
	
		if ( (pos=FindIdentifier(M, id=prefix+string("::Xi[")+cluster_to_string(kk)+string("]")) )>= 0 )
                {
                        if (!ParseRestrictionMatrix(R,C, M, pos+1, n_vars, n_vars) )
                                throw dw_exception("SetupRestrictions_Regime: error parsing //== " + id);
                        for (int ii=0; ii<n_vars; ii++)
                        {       
                                for (int jj=0; jj<n_vars; jj++)
                                        RXi[kk](ii,jj) = R[ii][jj] == 1 ? NAN : C(ii,jj);
                        }
                }
	}
	return true; 
}

bool TVSBVAR::SetupRestrictions(std::istream &input)
{
	// SetupRestrictionsVAR.m
	StringMatrix M(input, string("//**")); 
	int pos, _nvar, _nlag, _nexogenous; 
	if ( (pos=FindIdentifier(M, "NumberVariables")) < 0 || !ParseInteger(_nvar, M, pos+1) || _nvar != n_vars )
		throw dw_exception("SetupRestrictions(): error parsing //== NumberVariables"); 
	if ( (pos=FindIdentifier(M, "NumberLags")) < 0 || !ParseInteger(_nlag, M, pos+1) || _nlag != n_lags)
		throw dw_exception("SetupRestrictions(): error parsing //== NumberLags"); 
	if ( (pos=FindIdentifier(M, "NumberExogenousVariables")) < 0 || !ParseInteger(_nexogenous, M, pos+1) || _nvar*_nlag+_nexogenous != n_predetermined) 
		throw dw_exception("SetupRestrictions(): error parsing //== NumberExogenousVariables"); 

	if (!SetupRestrictions_Regime(M, string("Base"), RA_base,RAplus_base,RXi_base) || !SetupRestrictions_Regime(M,string("Multiplicative"), RA_mult,RAplus_mult,RXi_mult) || !SetupRestrictions_Regime(M, string("Additive"), RA_add,RAplus_add,RXi_add) )
		return false; 
	
	// base regime
	int initial_offset=0; 
	if (!base_restrictions.AffineExclusion(RA_base,RAplus_base,RXi_base,initial_offset))
		throw dw_exception("TVSBVAR::SetupRestrictions() : error occurred in base_restrictions::AffineExclusion()."); 
	// multiplicative regime
	initial_offset += base_restrictions.n_free; 
	if(!mult_restrictions.AffineExclusion(RA_mult,RAplus_mult,RXi_mult,initial_offset) )
		throw dw_exception("TVSBVAR::SetupRestrictions() : error occurred in mult_restrictions::AffineExclusion()."); 
	// additive regime
	initial_offset += mult_restrictions.n_free; 
	if(!add_restrictions.AffineExclusion(RA_add,RAplus_add,RXi_add,initial_offset) )
		throw dw_exception("TVSBVAR::SetupRestrictions() : error occurred in add_restrictions::AffineExclusion()."); 

	return true; 
}

//===============================================================================
TDenseVector TVSBVAR :: InitialParameterValues_SimsZha(const vector<TDenseMatrix> &A_cell, const vector<TDenseMatrix> &Aplus_cell) const
{
	// InitialParameterValues_SimsZha.m
	if (A_cell.empty() || Aplus_cell.empty() || A_cell.size() != Aplus_cell.size() || (int)A_cell.size() != NumberRegimes() )
		return RandomNormalVector(base_restrictions.n_free+mult_restrictions.n_free+add_restrictions.n_free);

	vector<TDenseMatrix>Xi_cell(NumberRegimes(), Identity(n_vars)); 
	return VARToParameters(A_cell, Aplus_cell, Xi_cell); 
}

bool abs_comp(float a, float b) 
{ 
	return fabs(a)<fabs(b); 
}

TDenseVector TVSBVAR :: VARToParameters(const vector<TDenseMatrix> &A_cell, const vector<TDenseMatrix> &Aplus_cell, const vector<TDenseMatrix> &Xi_cell) const
{
	// VARToParameters_cell.m
	TDenseVector parameter(base_restrictions.n_free+mult_restrictions.n_free+add_restrictions.n_free,0.0); 
	
	int n = n_vars*(n_vars+n_predetermined+n_vars); 
	TDenseMatrix x(base_restrictions.n_regimes, n, 0.0); // base_restrictions.n_regimes * n; 
	for (int kk_base=0; kk_base<base_restrictions.n_regimes; kk_base++)
	{
		if (base_restrictions.dim[kk_base] )
		{
			int idx = RegimeIndex(kk_base, 0, 0); 
			// x(kk_base,:) = [vec(A_cell[idx]'); vec(Aplus_cell[idx]'); vec(Xi_cell[idx]')]
			for (int ii=0; ii<n_vars; ii++)
				x.InsertRowMatrix(kk_base, ii*n_vars, RowVector(A_cell[idx],ii) ); 
			for (int ii=0; ii<n_vars; ii++)
				x.InsertRowMatrix(kk_base, n_vars*n_vars+ii*n_predetermined, RowVector(Aplus_cell[idx],ii) ); 
			for (int ii=0; ii<n_vars; ii++)
				x.InsertRowMatrix(kk_base, n_vars*n_vars+n_vars*n_predetermined+ii*n_vars, RowVector(Xi_cell[idx],ii) ); 
			parameter.Insert(base_restrictions.offset[kk_base], RowVector(x, kk_base, base_restrictions.select[kk_base])); 	
		}
	}

	// [m, index] = max(abs(x)); 
	TDenseVector m(n,0.0); 
	TIndex index; 
	for (int ii=0; ii<n; ii++)
	{
		TDenseVector x_column=ColumnVector(x,ii); 
		int max_position = max_element(x_column.vector, x_column.vector+x_column.dim, abs_comp)-x_column.vector; 
		index += max_position; 
		m[ii] = fabs(x_column[max_position]); 
	}

	TDenseMatrix z(base_restrictions.n_regimes,n,0.0); // base_restrictions.n_regimes * n 
	TDenseVector y(n,0.0); 
	for (int kk_mult=0; kk_mult<mult_restrictions.n_regimes; kk_mult++)
	{
		if (mult_restrictions.dim[kk_mult] )
		{
			for (int kk_base=0; kk_base<base_restrictions.n_regimes; kk_base++)
			{
				int idx = RegimeIndex(kk_base,kk_mult,0); 
				// z(kk_base,:) = [vec(A_cell{idx})'; vec(Aplus_cell{idx}'); vec(Xi_cell{idx}')]
				for (int ii=0; ii<n_vars; ii++)
					z.InsertRowMatrix(kk_base, ii*n_vars, RowVector(A_cell[idx], ii)); 
				for (int ii=0; ii<n_vars; ii++)
					z.InsertRowMatrix(kk_base, n_vars*n_vars+ii*n_predetermined, RowVector(Aplus_cell[idx], ii) ); 
				for (int ii=0; ii<n_vars; ii++)
					z.InsertRowMatrix(kk_base, n_vars*n_vars+n_vars*n_predetermined+ii*n_vars, RowVector(Xi_cell[idx], ii)); 
			}
			for (int ii=0; ii<n; ii++)
			{
				if (m[ii] > 0)
					y[ii] = z(index[ii],ii)/x(index[ii],ii); 
				else 
					y[ii] = 1.0; 
			}
			parameter.Insert(mult_restrictions.offset[kk_mult],y.SubVector(mult_restrictions.select[kk_mult])); 
		}
	}

	for (int kk_add=0; kk_add<add_restrictions.n_regimes; kk_add++)
	{
		if (add_restrictions.dim[kk_add] )
		{
			y.Zeros(n);
			int idx = RegimeIndex(0,0,kk_add); 
			for (int ii=0; ii<n_vars; ii++)
				y.InsertRowVector(ii*n_vars, A_cell[idx], ii); 
			for (int ii=0; ii<n_vars; ii++)
				y.InsertRowVector(n_vars*n_vars+ii*n_predetermined, Aplus_cell[idx], ii); 
			for (int ii=0; ii<n_vars; ii++)
				y.InsertRowVector(n_vars*n_vars+n_vars*n_predetermined+ii*n_vars, Xi_cell[idx], ii); 
			y = y - RowVector(x,0); 
			parameter.Insert(add_restrictions.offset[kk_add],y.SubVector(add_restrictions.select[kk_add])); 
		}
	}	

	return parameter; 	
}

//===============================================================================


bool TVSBVAR :: SetParameters_SimsZha(double *_parameters)
{
	// ParametersToVAR_cell.m
	vector<TDenseVector> base_parameters = VectorizedParameters(_parameters,base_restrictions); 
	vector<TDenseVector> mult_parameters = VectorizedParameters(_parameters,mult_restrictions); 
	vector<TDenseVector> add_parameters = VectorizedParameters(_parameters,add_restrictions); 

	int n1 = n_vars*n_vars; 
	int n2 = n1 + n_vars*n_predetermined; 

	TDenseVector x, y; 
	for (int kk_base=0; kk_base<base_restrictions.n_regimes; kk_base++)
	{
		for (int kk_mult=0; kk_mult<mult_restrictions.n_regimes; kk_mult++)
		{
			x = DotMultiply(base_parameters[kk_base], mult_parameters[kk_mult]); 
			for (int kk_add=0; kk_add<add_restrictions.n_regimes; kk_add++)
			{
				y = x + add_parameters[kk_add]; 
				int idx = RegimeIndex(kk_base,kk_mult,kk_add); 
				// A0[idx] = reshape(y(1:n1), n_vars, n_vars)'
				for (int jj=0; jj<n_vars; jj++)
					A0[idx].InsertRowMatrix(jj,0, y, jj*n_vars, (jj+1)*n_vars-1); 
				// Aplus[idx] = reshape(y(n1+1:n2), n_prdetermined, n_vars)'
				for (int jj=0; jj<n_vars; jj++)
					Aplus[idx].InsertRowMatrix(jj,0, y, n1+jj*n_predetermined, n1+(jj+1)*n_predetermined-1);
				// Xi[idx] = reshape(y(n2+1:n3), n_vars, n_vars)'
				for (int jj=0; jj<n_vars; jj++)
					Xi[idx].InsertRowMatrix(jj,0, y, n2+jj*n_vars, n2+(jj+1)*n_vars-1);  
			}
		}
	}
	return true; 
}

bool TVSBVAR::SetParameters(double *_parameters)
{
	bool return_value = true; 
	if (!regime_process->SetParameters(_parameters+offset))
		return_value =false; 
	
	if (!SetParameters_SimsZha(_parameters))
		return_value = false; 
	// Only when hyper_parameter changes, does SetupPrior_SimsZha need to be called.
	if (set_parameter_counter == 0 || hyper_parameter->NumberVariableParameters())
	{
		if (!hyper_parameter->SetParameters(_parameters+hyper_offset))
			return_value = false; 
		TDenseVector x_hyper(hyper_parameter->NumberVariableParameters()+hyper_parameter->NumberConstantParameters(),0.0); 
		hyper_parameter->GetParameters(x_hyper.vector);
		if (!sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data) )
			return_value = false; 
	}
	
	// also need to take _parameter as internal_parameters
	// internal parameter only contains variable part.
	memcpy(internal_parameters.vector, _parameters, sizeof(double)*internal_parameters.dim); 

	set_parameter_counter ++; 
	return return_value;  
}

vector<TDenseVector> TVSBVAR:: VectorizedParameters(double *parameters, const Restriction &restrictions) const
{
	// contained in ParametersToVAR_cell.m
	vector<TDenseVector> x(restrictions.n_regimes); 
	TDenseVector parameter_select; 
	for (int kk=0; kk<restrictions.n_regimes; kk++)
	{
		x[kk] = restrictions.d[kk]; 
		if (restrictions.dim[kk])
		{
			parameter_select.Zeros(restrictions.dim[kk]); 
			for (int jj=0; jj<parameter_select.dim; jj++)
				parameter_select[jj] = parameters[restrictions.offset[kk]+jj]; 
			x[kk].Insert(restrictions.select[kk],parameter_select); 
		}
	}

	return x; 
}

bool TVSBVAR::GetParameters(double *parameters) const
{
	// parameter contains variable part.
	memcpy(parameters, internal_parameters.vector, internal_parameters.dim*sizeof(double)); 
	return true; 
}

void TVSBVAR::DefaultParameters(void)
{
  // DW 4/9/15 - I think there is a serious bug in the code below.  I do not believe that 
  // the vector interal_parameters is ever set.  I think The only place internal_parameters 
  // is set is in the constructors and in the SetParameters() routine. This means that
  // after a call to this routine, there will be an inconsistancy in the internal representation
  // of the parameters.

	regime_process->DefaultParameters(); 
	regime_process->GetParameters(internal_parameters.vector+offset);                            // added 4/9/15
 	
	TDenseVector par_vec = RandomNormalVector(NumberParameters_VAR());
	while (!SetParameters_SimsZha(par_vec.vector))
		par_vec = RandomNormalVector(NumberParameters_VAR());
	memcpy(internal_parameters.vector,par_vec.vector,NumberParameters_VAR());                    // added 4/9/15
	
	hyper_parameter->DefaultParameters(); 
	if (hyper_parameter->NumberVariableParameters())
	{
		TDenseVector x_hyper(hyper_parameter->NumberVariableParameters()+hyper_parameter->NumberConstantParameters(),0.0); 
		hyper_parameter->GetParameters(x_hyper.vector);
		while (!sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data))
		{
			hyper_parameter->DefaultParameters();
			hyper_parameter->GetParameters(x_hyper.vector);
		}
		hyper_parameter->GetVariableParameters(internal_parameters.vector+hyper_offset);     // added 4/9/15
	}

	return; 
}

bool TVSBVAR::DrawParametersFromPrior(double *_parameters)
{
	regime_process->SimulatePrior(); 
	regime_process->GetParameters(_parameters+offset);

	TDenseVector par_vec(NumberParameters_VAR());
	if (!DrawParametersFromPrior_SimsZha(par_vec)) return false;
	memcpy(_parameters,par_vec.vector,NumberParameters_VAR()*sizeof(double));

        if (hyper_parameter->NumberVariableParameters())
        {
                hyper_parameter->DefaultParameters();
                TDenseVector x_hyper(hyper_parameter->NumberVariableParameters()+hyper_parameter->NumberConstantParameters(),0.0);
                hyper_parameter->GetParameters(x_hyper.vector);
                while (!sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data))
                {
                        hyper_parameter->DefaultParameters();
                        hyper_parameter->GetParameters(x_hyper.vector);
                }
		hyper_parameter->GetVariableParameters(_parameters+hyper_offset);
        }

	memcpy(internal_parameters.vector,_parameters,NumberParameters()*sizeof(double));	

	return true; 

  // DW 4/9/15 - I think there are two serious bugs in the code below.  First, I do not believe that 
  // the vector interal_parameters is ever set.  I think The only place internal_parameters 
  // is set is in the constructors and in the SetParameters() routine. This means that
  // after a call to this routine, there will be an inconsistancy in the internal representation
  // of the parameters.
  // Second, the vector autoregression parameters are simple set to standard normal instead of
  // drawing from the prior. This can cause problems.
  // The code above attempts to fix this.
  //
	// regime_process->SimulatePrior(); 

	// TDenseVector par_vec = RandomNormalVector(NumberParameters_VAR());
        // while (!SetParameters_SimsZha(par_vec.vector))
        //         par_vec = RandomNormalVector(NumberParameters_VAR());

        // hyper_parameter->DefaultParameters();
        // if (hyper_parameter->NumberVariableParameters())
        // {
        //         TDenseVector x_hyper(hyper_parameter->NumberVariableParameters()+hyper_parameter->NumberConstantParameters(),0.0);
        //         hyper_parameter->GetParameters(x_hyper.vector);
        //         while (!sims_zha.SetupPrior_SimsZha(x_hyper, periods_per_year, variance_scale, base_restrictions, mult_restrictions, add_restrictions, (TData_predetermined *)data))
        //         {
        //                 hyper_parameter->DefaultParameters();
        //                 hyper_parameter->GetParameters(x_hyper.vector);
        //         }
        // }
		
	// // DefaultParameters(); 
	// GetParameters(_parameters); 
	// return true; 
}


//===============================================================================


//=== Added by DW - 10/21/2013 ===
// Returns the maximum of the square of the norm of the eigenvalues of the 
// companion matrix for each regime.
double TVSBVAR::MaxSquareNormEigenvalues(const double *parameter)
{

	if (!SetParameters((double*)parameter))
		return PLUS_INFINITY; 

  int k=n_vars*n_lags, n=n_vars;
  double max=0.0, tmp;

  TDenseMatrix X(k,k,0.0);
  for (int i=n; i < k; i++) X(i,i-n)=1.0;

  TDenseVector re_eig(k), im_eig(k);

  for (int i=A0.size()-1; i >= 0; i--)
    {
      X.Insert(0,0,InverseMultiply(A0[i],SubMatrix(Aplus[i],0,n-1,0,k-1)));
      Eig(re_eig,im_eig,X);
      for (int j=k-1; j >= 0; j--)
	if ((tmp=re_eig(j)*re_eig(j)+im_eig(j)*im_eig(j)) > max) max=tmp;
    }
  return max;
}

//=== Added by HW - 10/02/2014 ---
// Only works for 1 base regime, no additive regimes, because multiplicative regime parameters do not need to flipped signs.
TDenseVector TVSBVAR::GetNormalizedParameter(const TDenseVector &parameter)
{
	if (parameter.dim !=  NumberParameters() || base_restrictions.n_regimes != 1 || add_restrictions.n_regimes !=1 )
		return TDenseVector(0); 

	TDenseVector normalized(parameter.dim,0.0); 
	normalized.CopyContent(parameter); 

	TIndex selected_index = base_restrictions.select[0]; 
	for (int i_eqn=0; i_eqn<n_vars; i_eqn++)
	{
		// find base_restrictions.select[0](??) == i_eqn*n_vars+i_eqn and check whether parameter[select[0](??)] <0
		bool found_flag = false; 
		int l = base_restrictions.offset_a[0][i_eqn];
		while (!found_flag && l < base_restrictions.offset_a[0][i_eqn]+base_restrictions.dim_a[0][i_eqn])
		{
			if (selected_index[l] == i_eqn*n_vars+i_eqn)
				found_flag = true; 
			else 
				l++; 
		}
		if (found_flag && parameter[base_restrictions.offset[0]+selected_index[l]] < 0)
		{
			normalized.Insert(base_restrictions.offset_a[0][i_eqn], -1.0*parameter.SubVector(base_restrictions.offset_a[0][i_eqn], base_restrictions.offset_a[0][i_eqn]+base_restrictions.dim_a[0][i_eqn]-1)); 
			normalized.Insert(base_restrictions.offset_aplus[0][i_eqn], -1.0*parameter.SubVector(base_restrictions.offset_aplus[0][i_eqn], base_restrictions.offset_aplus[0][i_eqn]+base_restrictions.dim_aplus[0][i_eqn]-1)); 	
		}
	}
	return normalized;
}

TDenseMatrix TVSBVAR::GetA0(const TDenseVector &parameters, int t, int regime)
{
  if (!SetParameters(parameters.vector)) throw dw_exception("RTVSBVAR::GetA0() - unable to set free parameters");
  return A0[regime];
}
TDenseMatrix TVSBVAR::GetAplus(const TDenseVector &parameters, int t, int regime)
{
  if (!SetParameters(parameters.vector)) throw dw_exception("RTVSBVAR::GetAplus() - unable to set free parameters");
  return Aplus[regime];
}
TDenseMatrix TVSBVAR::GetXi(const TDenseVector &parameters, int t, int regime)
{
  if (!SetParameters(parameters.vector)) throw dw_exception("RTVSBVAR::GetXi() - unable to set free parameters");
  return Xi[regime];
}

// Sets matrices of filtered probabilities.  Note that
//   
//      Pt(k,t) = p(s(t) = k | Y(t))
//
//      Ptm1(k,t) = p(s(t) = k | Y(t-1))
// 
// where Y(t) = (y(t), y(t-1), y(t-2), ... ) 
//
// Assumes that SetParameters() has been successfully called.
void TVSBVAR::FilteredProbabilities(TDenseMatrix &Pt, TDenseMatrix &Ptm1)
{
	#define LOG_ZERO -1.0E300;
	int n_regimes=NumberRegimes(), n_obs=NumberObservations(); 
	Pt.UniqueMemory(n_regimes,n_obs,false);
	Ptm1.UniqueMemory(n_regimes,n_obs,false);

	TDenseVector log_constant(n_regimes);
	for (int s=0; s < n_regimes; s++)
	  log_constant(s)=-0.918938533204673*n_vars + LogAbsDeterminant(Xi[s]*A0[s]);   // -0.5*n_vars*log(2*pi) + log(abs(det(Xi(k)*A(k))))

	
	double scale; 
	TDenseVector epsilon, p_tm1(n_regimes), p_t(n_regimes);
	
	// =======  Hamilton filter ==============
	for (int t=0; t < n_obs; t++)
	{		
		// compute p_tm1(k) = p( s_t = k | Y(t-1) )
		p_tm1=(t == 0) ? InitialProbabilities() : TransitionMatrix(t-1)*p_t;

		// compute log conditional probabilities and scale
		scale = LOG_ZERO; 
		for (int s_t=0; s_t<n_regimes; s_t++)
		{
			if (p_tm1(s_t) > 0.0)
          		{
            			epsilon=Xi[s_t]*(A0[s_t]*Data(t) - Aplus[s_t]*((TData_predetermined *)data)->PredeterminedData(t));
            			scale=AddLogs(scale,p_t[s_t]=log(p_tm1[s_t]) + log_constant[s_t] - 0.5*InnerProduct(epsilon,epsilon));
          		}
		}
		
		// update p_t(k) = p( s_t = k | Y(t) )
		for (int s_t=0; s_t<n_regimes; s_t++)
        		p_t(s_t)=(p_tm1(s_t) > 0.0) ? exp(p_t(s_t) - scale) : 0.0;

		// save p_t and p_tm1
		Ptm1.InsertColumnMatrix(0,t,p_tm1);
		Pt.InsertColumnMatrix(0,t,p_t);
	}
	#undef LOG_ZERO 
}

// Returns n_regimes x n_obs matrix P of smoothed probabilities.  Note that
//   
//      P(k,t) = p(s(t) = k | Y),
// 
// where Y = (y(n_obs-1), y(n_obs-2), y(n_obs-3), ... )
//
// Assumes that SetParameters() has been successfully called.
TDenseMatrix TVSBVAR::SmoothedProbabilities(void)
{
  TDenseMatrix Pt, Ptm1;
  FilteredProbabilities(Pt,Ptm1);
  int n_obs=NumberObservations(), n_regimes=NumberRegimes();
  TDenseMatrix SP(n_regimes,n_obs,0.0);
  
  SP.Insert(0,n_obs-1,Pt,0,n_regimes-1,n_obs-1,n_obs-1);
  for (int t=n_obs-2; t >= 0; t--)
    {
      TDenseMatrix Q=TransitionMatrix(t+1);

      // s(t)=k and s(t+1)=i
      for (int k=n_regimes-1; k >= 0; k--)
	{
	  for (int i=n_regimes-1; i >= 0; i--)
	    if (Ptm1(i,t+1) > 0.0)
	      SP(k,t)+=SP(i,t+1)*Q(i,k)/Ptm1(i,t+1);
	  SP(k,t)*=Pt(k,t);
	}
    }

  return SP; 
}

TDenseMatrix TVSBVAR::TrendCycle(const TDenseVector &parameters, int n_paths, double cutoff)
{
  TDenseMatrix Y=data->Data(), X=((TData_predetermined *)data)->PredeterminedData();
  int n_regimes=NumberRegimes(), n_obs=NumberObservations();

  if (!SetParameters(parameters.vector)) throw dw_exception("RTVSBVAR::ImpulseResponseTrendCycle() - unable to set free parameters");

  // companion form
  TDenseMatrix B(n_lags*n_vars,n_lags*n_vars,0.0);
  B.Insert(0,0,InverseMultiply(A0[0],Aplus[0].SubMatrix(0,n_vars-1,0,n_lags*n_vars-1)));
  B.Insert(n_vars,0,Identity((n_lags-1)*n_vars));

  // Schur decomposition of companion form
  TDenseMatrix T, Z, Ordered_T, Ordered_Z;
  TDenseVector E_re, E_im;
  Schur(T,E_re,E_im,Z,B,true);
  int *select=new int[n_lags*n_vars];
  int trend_dim=0;
  //double cutoff = 1.0 - 1.0e-6;
  for (int i=0; i < n_lags*n_vars; i++) 
    if (sqrt(E_re[i]*E_re[i] + E_im[i]*E_im[i]) < cutoff)
      select[i]=1;
    else
      {
	select[i]=0;
	trend_dim++;
      }
  OrderSchur(Ordered_T, E_re, E_im, Ordered_Z, T, Z, select, true);
  // if (trend_dim > n_vars) throw dw_exception("RTVSBVAR_China::TrendCycle() - not enough stable roots");
  
  TDenseMatrix W1=Ordered_Z.SubMatrix(0,n_lags*n_vars-1,0,n_lags*n_vars-trend_dim-1);
  TDenseMatrix W2=Ordered_Z.SubMatrix(0,n_lags*n_vars-1,n_lags*n_vars-trend_dim,n_lags*n_vars-1);
  TDenseMatrix T11=Ordered_T.SubMatrix(0,n_lags*n_vars-trend_dim-1,0,n_lags*n_vars-trend_dim-1);
  TDenseMatrix T12=Ordered_T.SubMatrix(0,n_lags*n_vars-trend_dim-1,n_lags*n_vars-trend_dim,n_lags*n_vars-1);
  TDenseMatrix T22=Ordered_T.SubMatrix(n_lags*n_vars-trend_dim,n_lags*n_vars-1,n_lags*n_vars-trend_dim,n_lags*n_vars-1);
  TDenseMatrix beta=InverseMultiply(Identity(n_lags*n_vars-trend_dim) - T11,T12);
  TDenseMatrix K=Identity(n_lags*n_vars,n_vars);

  // // Check T22 == Identity
  // if (Norm(Identity(trend_dim) - T22) > 0.00001)
  //   throw dw_exception("RTVSBVAR::TrendCycle() - model not integrated");

  std::vector<TDenseMatrix> UUprime(n_regimes);
  for (int i=0; i < n_regimes; i++)
    {
      TDenseMatrix U;
      NullSpace(U,MultiplyInverse(Transpose(W2)*K,Xi[i]*A0[i]));
      UUprime[i]=MultiplyTranspose(U,U);
    }

  // filtered probabilities
  TDenseMatrix Pt, Ptm1;
  FilteredProbabilities(Pt,Ptm1);

  // trend/cycle decomposition
  TDenseMatrix decomposition(n_paths,3*n_obs*n_vars);
  TDenseVector z_s(n_lags*n_vars), z_p, z;
  for (int i=0; i < n_paths; i++)
    {
      std::vector<int> s=DrawRegimes(Pt,Ptm1);
      z_s.Initialize(0.0);
      z_p=((TData_predetermined *)data)->PredeterminedData(0).SubVector(0,n_lags*n_vars-1);
      for (int t=0; t < n_obs; t++)
	{
	  TDenseVector y_t=Data(t);
	  TDenseVector x_t=((TData_predetermined *)data)->PredeterminedData(t);
	  TDenseVector epsilon=Xi[s[t]]*(A0[s[t]]*y_t - Aplus[s[t]]*x_t);
	  TDenseVector epsilon_s=UUprime[s[t]]*epsilon;
	  TDenseVector epsilon_p=epsilon - epsilon_s;
	  TDenseVector c_t=ColumnVector(Aplus[s[t]],n_lags*n_vars);

	  z_s=K*InverseMultiply(Xi[s[t]]*A0[s[t]],epsilon_s) + B*z_s;
	  z_p=K*(InverseMultiply(A0[s[t]],c_t) + InverseMultiply(Xi[s[t]]*A0[s[t]],epsilon_p)) + B*z_p;

	  for (int j=0; j < n_vars; j++)
	    {
	      decomposition(i,3*j*n_obs+t)=y_t(j);
	      decomposition(i,(3*j+1)*n_obs+t)=z_p(j);
	      decomposition(i,(3*j+2)*n_obs+t)=z_s(j);
	    }

	  // // check
          // if (Norm(data->Data(t) - z_s.SubVector(0,n_vars-1) - z_p.SubVector(0,n_vars-1)) > 0.000001)
	  //   throw dw_exception("RTVSBVAR::TrendCycle() - trend + cycle != data");
	}
    }
  return decomposition;
}


// Using backwards recursion, draws a path of regimes the given the parameters and 
// the data.  The arguments are
//
//   Pt(k,t) = p(s(t)=k | Y(t))
//
//   Ptm1(k,t) = p(s(t)=k | Y(t-1))
//
// and can be obtained via a call to FilteredProbabilities().
std::vector<int> TVSBVAR::DrawRegimes(const TDenseMatrix &Pt, const TDenseMatrix &Ptm1)
{
  int n_obs=NumberObservations(), n_regimes=NumberRegimes();

  // Backward recursion
  std::vector<int> regimes(n_obs);
  double u, s, scale;
  int i, j, t;

  if ((u=dw_uniform_rnd()) >= (s=Pt(i=n_regimes-1,t=n_obs-1)))
    while (--i > 0)
      if (u < (s+=Pt(i,t))) break;
  regimes[t]=i;

  for (t--; t >= 0; t--)
    {
      TDenseMatrix Q=TransitionMatrix(t+1);

      scale=1.0/Ptm1(j=i,t+1);
      i=n_regimes-1;
      if ((u=dw_uniform_rnd()) >= (s=Pt(i,t)*Q(j,i)*scale))
	while (--i > 0)
	  if (u < (s+=Pt(i,t)*Q(j,i)*scale)) break;
      regimes[t]=i;
    }

  return regimes;
}

// Impuse Responses /////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////
// The element in position (k,i+j*n_vars) of IR is the response of the ith 
// variable to the jth shock at horizon k.  Horizon 0 is the contemporaneous
// response.
/////////////////////////////////////////////////////////////////////////////////
TDenseMatrix TVSBVAR::ImpulseResponse(int horizon, const TDenseVector &parameters, int regime)
{
  if (horizon <= 0) throw dw_exception("RTVSBVAR::ImpulseResponse() - horizon must be positive");

  if (!SetParameters(parameters.vector))
    throw dw_exception("RTVSBVAR::ImpulseResponse() - unable to set free parameters");

  TDenseMatrix IR(horizon,n_vars*n_vars);
  TDenseMatrix B=InverseMultiply(A0[regime],Aplus[regime]);

  TDenseMatrix Y_t=Inverse(Xi[regime]*A0[regime]), X_t(n_predetermined,n_vars,0.0);

  // contemporaneous impulse response
  for (int i=n_vars-1; i >= 0; i--)
    for (int j=n_vars-1; j >= 0; j--)
      IR(0,i+n_vars*j)=Y_t(i,j);

  // horizon t impulse response
  for (int t=1; t < horizon; t++)
    {
      if (n_lags > 0) X_t.Insert(n_vars,0,X_t,0,(n_lags-1)*n_vars-1,0,n_vars-1);
      X_t.Insert(0,0,Y_t);
      Y_t=B*X_t;
      for (int i=n_vars-1; i >= 0; i--)
	for (int j=n_vars-1; j >= 0; j--)
	  IR(t,i+n_vars*j)=Y_t(i,j);
    }

  return IR;
}

/////////////////////////////////////////////////////////////////////////////////
// The element in position (k,i+j*n_vars) of IR is the response of the ith 
// variable to the jth shock at horizon k.  Horizon 0 is the contemporaneous
// response.  
//
// The first trend_dim shocks are the trends and remaining shocks are the cycles.
/////////////////////////////////////////////////////////////////////////////////
TDenseMatrix TVSBVAR::ImpulseResponseTrendCycle(int horizon, const TDenseVector &parameters, int regime, int &trend_dim, double cutoff)
{
  if (horizon <= 0) throw dw_exception("RTVSBVAR::ImpulseResponseTrendCycle() - horizon must be positive");

  if ((regime < 0) || (regime >= NumberRegimes())) throw dw_exception("RTVSBVAR::ImpulseResponseTrendCycle() - invalid regime index");

  if (!SetParameters(parameters.vector)) throw dw_exception("RTVSBVAR::ImpulseResponseTrendCycle() - unable to set free parameters");

  // companion form
  TDenseMatrix B(n_lags*n_vars,n_lags*n_vars,0.0);
  B.Insert(0,0,InverseMultiply(A0[regime],Aplus[regime].SubMatrix(0,n_vars-1,0,n_lags*n_vars-1)));
  B.Insert(n_vars,0,Identity((n_lags-1)*n_vars));

  // Schur decomposition
  TDenseMatrix T, Z, Ordered_T, Ordered_Z;
  TDenseVector E_re, E_im;
  Schur(T,E_re,E_im,Z,B,true);
  int *select=new int[n_lags*n_vars];
  trend_dim=0;
  for (int i=0; i < n_lags*n_vars; i++) 
    if (sqrt(E_re[i]*E_re[i] + E_im[i]*E_im[i]) < cutoff)
      select[i]=1;
    else
      {
	select[i]=0;
	trend_dim++;
      }
  //cout << "T:\n" << T << endl;
  //cout << "E_re:\n" << E_re << endl;
  //cout << "E_im:\n" << E_im << endl;
  OrderSchur(Ordered_T, E_re, E_im, Ordered_Z, T, Z, select, true);
  //if (trend_dim > n_vars) throw dw_exception("RTVSBVAR::ImpulseResponseTrendCycle() - too few stable roots");

  // all trend or cycle
  //if ((trend_dim == 0) || (trend_dim == n_vars)) 
  if ((trend_dim == 0) || (trend_dim >= n_vars))
    return ImpulseResponse(horizon,parameters,regime);

  // not all trend or cycle
  TDenseMatrix Q, R;
  QR(Q,R,InverseMultiply(Transpose(A0[regime])*Xi[regime],Identity(n_vars,n_lags*n_vars)*Ordered_Z.SubMatrix(0,n_lags*n_vars-1,n_lags*n_vars - trend_dim,n_lags*n_vars-1)),0);

  // impulse responses
  TDenseMatrix IR(horizon,n_vars*n_vars);
  B=InverseMultiply(A0[regime],Aplus[regime]);
  TDenseMatrix Y_t=InverseMultiply(Xi[regime]*A0[regime],Q), X_t(n_predetermined,n_vars,0.0);

  // contemporaneous impulse response
  for (int i=n_vars-1; i >= 0; i--)
    for (int j=n_vars-1; j >= 0; j--)
      IR(0,i+n_vars*j)=Y_t(i,j);

  // horizon t impulse response
  for (int t=1; t < horizon; t++)
    {
      if (n_lags > 0) X_t.Insert(n_vars,0,X_t,0,(n_lags-1)*n_vars-1,0,n_vars-1);
      X_t.Insert(0,0,Y_t);
      Y_t=B*X_t;
      for (int i=n_vars-1; i >= 0; i--)
	for (int j=n_vars-1; j >= 0; j--)
	  IR(t,i+n_vars*j)=Y_t(i,j);
    }

  return IR;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// 
//  Regime switching BVAR that can be normalized
//
//  Note that only a subset of tvsbvar models can be normalized by using Dan and Tao's method reported 
//  in the paper "Likelihood preserving normalization in multiple equation models"
//
//  For example, if A0, Aplus or Xi contains non-zero constrained elements, then their signs cannot be
//  changed.
//
//  Currently we do not check if a tvsbvar model can be normalized. Insteady, we just apply the method
//
//
//  Also, the current method for normalization only applies when there is only ONE basic regime, multiple
//  multiplicative regimes, and ONE additvie regime. When the sign has to be changed, we only change the
//  sign of A0 and Aplus. Note that Xi's sign does not change, because the shock is modeled as a Gaussian
//  random vector, and changing Xi's sign or not does not matter.
/////////////////////////////////////////////////////////////////////////////////////////////////////


TVSBVAR_normalized::TVSBVAR_normalized(void) :
TVSBVAR(),
Ahat(0,0)
{}

TVSBVAR_normalized::TVSBVAR_normalized(const TVSBVAR_normalized &rhs) :
TVSBVAR(rhs),
Ahat(rhs.Ahat)
{
	Ahat.CopyContent(rhs.Ahat); 
}

TVSBVAR_normalized::TVSBVAR_normalized(istream &restriction_file, TData_predetermined *_data, TRegimeProcess *_regime_process, THyperParameter *_hyper_parameter, int _periods_per_year, double _variance_scale):
TVSBVAR(restriction_file, _data, _regime_process, _hyper_parameter, _periods_per_year, _variance_scale),
Ahat(n_vars, n_vars, 0.0)
{
	std::vector<TDenseMatrix> U(n_vars), V(n_vars); 
	double x; 
	for (int i=0; i<n_vars; i++)
	{
		int k=0;
      		for (int j=n_vars-1; j >= 0; j--)
		{
        		if (std::isnan(RA_base[0](i,j)) ) 
				k++;
		}
      		U[i].Zeros(n_vars,k);
      		k=0;
      		for (int j=0; j < n_vars; j++)
		{
        		if (std::isnan(RA_base[0](i,j)) )
          			U[i](j,k++)=1.0;
		}
	}
	for (int i=0; i < n_vars; i++)
    	{
      		int k=0;
      		for (int j=n_predetermined-1; j >= 0; j--)
		{
        		if (std::isnan(RAplus_base[0](i,j))) k++;
		}
      		V[i].Zeros(n_predetermined,k);
      		k=0;
      		for (int j=0; j < n_predetermined; j++)
		{
        		if (std::isnan(RAplus_base[0](i,j)))
          			V[i](j,k++)=1.0;
		}
    }

	SBVAR_symmetric_linear *temp_sbvar = new SBVAR_symmetric_linear(_data, U, V); 
	temp_sbvar->MaximizePosterior();
	Ahat.CopyContent(temp_sbvar->A0);
	delete temp_sbvar; 
}	

double TVSBVAR_normalized::LogPrior()
{
	TDenseMatrix A0_Inverse; 
	TDenseVector e(n_vars,0.0); 
	for (int i=0; i<NumberRegimes(); i++)
	{
		A0_Inverse = Inverse(A0[i]); 
		for (int j=0; j<n_vars; j++)
		{
			e.Zeros();
			e(i) = 1.0; 
			if (InnerProduct(e, Ahat.RowVector(i), Transpose(A0_Inverse)) < 0)
				return MINUS_INFINITY;
		}		
	}	
	return TVSBVAR::LogPrior() + 0.693147180559945*n_vars; // ? Question: number of symmetric solutions?
}

bool TVSBVAR_normalized::DrawParametersFromPrior(double *parameters)
{
	TVSBVAR::DrawParametersFromPrior(parameters); 
	TVSBVAR::SetParameters(parameters); 

	vector<TDenseMatrix> A0_normalized(NumberRegimes()), Aplus_normalized(NumberRegimes()), Xi_normalized(NumberRegimes());  
	TDenseMatrix A0_normalized_inverse; 
	TDenseVector e(n_vars,0.0); 
	for (int i=0; i<NumberRegimes(); i++)
	{
		A0_normalized[i].CopyContent(A0[i]); 
		Aplus_normalized[i].CopyContent(Aplus[i]);
		Xi_normalized[i].CopyContent(Xi[i]); 
	
		A0_normalized_inverse = Inverse(A0_normalized[i]);
                for (int j=0; j<n_vars; j++)
                {
                        e.Zeros();
                        e(i) = 1.0;
                        if (InnerProduct(e, Ahat.RowVector(i), Transpose(A0_normalized_inverse)) < 0)
                        {
				A0_normalized[i].InsertRowMatrix(i,0, -1.0*A0_normalized[i].RowVector(i)); 
				Aplus_normalized[i].InsertRowMatrix(i,0,-1.0*Aplus_normalized[i].RowVector(i));
			}
                }
	}

	TDenseVector normalized_p = VARToParameters(A0_normalized, Aplus_normalized, Xi_normalized);
	memcpy(parameters, normalized_p.vector, sizeof(double)*normalized_p.dim); 
	SetParameters(parameters);
	return true; 
}
