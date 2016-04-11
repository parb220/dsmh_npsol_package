#ifndef _TVSBVAR_HEADER_
#define _TVSBVAR_HEADER_

#include <vector>
#include <string>
#include <istream>

#include "dw_ascii.hpp"
#include "dw_dense_matrix.hpp"
#include "dw_time_series.hpp"
#include "hyper_parameter.hpp"

using namespace std; 

class Restriction; 
class Sims_Zha; 
class TVSBVAR; 
class THyperParameter; 

class Restriction
{
public:
	int n_regimes; 
	int n_free; 
	vector<int> offset; 
	vector<TDenseVector> d; 
	vector<TIndex> select; 
	vector<int> dim; 
	vector<vector<int> > offset_a; 
	vector<vector<int> > offset_aplus; 
	vector<vector<int> > offset_xi; 
	vector<vector<int> > dim_a; 
	vector<vector<int> > dim_aplus; 
	vector<vector<int> > dim_xi; 

public:
	Restriction();
	Restriction(const Restriction &rhs); 
	bool AffineExclusion(const std::vector<TDenseMatrix> &RA, const std::vector<TDenseMatrix> &RAplus, const std::vector<TDenseMatrix> &RXi, int offset); 
}; 

class Sims_Zha
{
protected:// used in GetDummyObservations
	bool if_precalculated; 
	bool if_preallocated; 
	TDenseVector mean; 
	TDenseVector s; 
	TDenseMatrix Sbar; 
	TDenseMatrix YY; 
	TDenseMatrix XY; 
	TDenseMatrix XX; 
	void GetDummyObservations_Precalculations(TData_predetermined *data, int n_lags); 
	void MakeHPSc_PreAllocation(const Restriction &restriction); 
public:
	TDenseVector hyper; 
	int periods_per_year; 
	double variance_scale; 
	
	bool base_flat_prior; 
	TDenseMatrix X_dummy; 
	TDenseMatrix Y_dummy; 
	vector<vector<TIndex > >select_a; 
	vector<vector<TIndex > >select_aplus; 
	vector<vector<TDenseMatrix> >H; 
	vector<vector<TDenseMatrix> >P; 
	vector<vector<TDenseMatrix> >S; 
	vector<vector<TDenseVector> >c_a; 
	vector<vector<TDenseVector> >c_aplus; 
	double base_prior_constant; 
	double base_restriction_constant; 
	
	bool mult_flat_prior; 
	TDenseVector mult_prior_a; 
	TDenseVector mult_prior_b; 
	double mult_prior_constant; 

	bool add_flat_prior; 
	TDenseVector add_prior_mean; 
	TDenseVector add_prior_variance; 
	double add_prior_constant;	

  // added by DW 4/9/15 for simulating from prior
  vector<vector<TDenseMatrix> > SqrtInvS;
  vector<vector<TDenseMatrix> > SqrtInvH;
	
	void MakeSelect(int n_vars, int n_predetermined, const Restriction &restrictions); 
	void GetDummyObservations(TData_predetermined *data, int n_lags); 
	bool MakeHPSc(const Restriction &restrictions); 

public:
	static double log2pi;
	Sims_Zha(); 
	Sims_Zha(const Sims_Zha &); 	
	bool SetupPrior_SimsZha(const TDenseVector &_hyper, int _periods_per_year, double _variance_scale, const Restriction &base_restriction, const Restriction &mult_restriction, const Restriction &add_restriction, TData_predetermined *data); 
}; 

class TVSBVAR : public TTimeSeries_TV
{
private:
	TDenseVector internal_parameters; 
protected:
  int n_vars;
  int n_lags;
  int n_predetermined;
	int periods_per_year; 
	double variance_scale; 

  std::vector<TDenseMatrix> A0;
  std::vector<TDenseMatrix> Aplus;
  std::vector<TDenseMatrix> Xi;

	std::vector<TDenseMatrix> RA_base, RAplus_base, RXi_base;
        std::vector<TDenseMatrix> RA_mult, RAplus_mult, RXi_mult;
        std::vector<TDenseMatrix> RA_add, RAplus_add, RXi_add;

	Restriction base_restrictions; 
	Restriction mult_restrictions; 
	Restriction add_restrictions; 	

	Sims_Zha sims_zha; 

	vector<vector<int> >index_mapping; 
	THyperParameter *hyper_parameter; 
	int hyper_offset;  
 
	virtual double LogLikelihood(void);
  	virtual double LogPrior(void);  

private: 	
	int set_parameter_counter; 

protected:
	// Setup functions, called by TVSBVAR::TVSBVAR(std::istream &input)
  	bool SetupRestrictions(std::istream &input); 
	bool SetupRestrictions_Regime(const StringMatrix &M, const string &prefix, vector<TDenseMatrix> &RA, vector<TDenseMatrix> &RAplus, vector<TDenseMatrix> &RXi); 
	TDenseVector InitialParameterValues_SimsZha(const vector<TDenseMatrix> &A_cell=vector<TDenseMatrix>(0), const vector<TDenseMatrix> &Aplus_cell=vector<TDenseMatrix>(0)) const;
	TDenseVector VARToParameters(const vector<TDenseMatrix> &A_cell, const vector<TDenseMatrix> &Aplus_cell, const vector<TDenseMatrix> &Xi_cell) const; 
	vector<TDenseVector> VectorizedParameters(double *parameters, const Restriction &restrictions) const; 
	double LogPrior_SimsZha() const; 
	double LogPrior_SimsZha_Base(const TDenseVector &parameters) const; 
	double LogPrior_SimsZha_Mult(const TDenseVector &parameters) const; 
	double LogPrior_SimsZha_Add(const TDenseVector &parameters) const;
        double MaxSquareNormEigenvalues(const double *);

  bool DrawParametersFromPrior_SimsZha(TDenseVector &parameters);
  bool DrawParametersFromPrior_SimsZha_Base(TDenseVector &parameters);
  bool DrawParametersFromPrior_SimsZha_Mult(TDenseVector &parameters);
  bool DrawParametersFromPrior_SimsZha_Add(TDenseVector &parameters);

public:
  // constructors
  TVSBVAR(void); 
  TVSBVAR(const TVSBVAR &TimeSeries);
  TVSBVAR(istream &restriction_file, TData_predetermined *_data, TRegimeProcess *_regime_process, THyperParameter *_hyper_parameter, int _periods_per_year, double _variance_scale);

  // destructor
  virtual ~TVSBVAR() { };

  // cloning an object
  virtual TVSBVAR* Clone(void) { return new TVSBVAR(*this); };

	bool SetParameters_SimsZha(double *parameters); 
  	virtual bool SetParameters(double *parameters);
	virtual bool GetParameters(double *parameters) const; 
	virtual void DefaultParameters(void); 
	virtual bool DrawParametersFromPrior(double *parameters); 

  
	
  	// information
 	int AddIndex(int k) const {return index_mapping[k][2]; }
 	int MultIndex(int k) const {return index_mapping[k][1]; }
	int BaseIndex(int k) const {return index_mapping[k][0]; }
	int RegimeIndex(int base_index, int mult_index, int add_index) const; 
	virtual int NumberParameters(void) const{ return base_restrictions.n_free+mult_restrictions.n_free+add_restrictions.n_free+regime_process->NumberParameters()+hyper_parameter->NumberVariableParameters();} ;
	int NumberParameters_VAR() const { return  base_restrictions.n_free+mult_restrictions.n_free+add_restrictions.n_free; }
	int NumberParameters_RegimeProcess() const { return regime_process->NumberParameters(); }; 
	int Offset_HyperParameters() const { return hyper_offset; }
	int NumberParameters_HyperParameter() const { return hyper_parameter->NumberVariableParameters(); }; 
	const Restriction & BaseRestriction() const { return base_restrictions; }; 
	const Restriction & MultiplicativeRestriction() const { return mult_restrictions; }; 
	const Restriction & AdditiveRestriction() const { return add_restrictions; }; 
	int NumberVariables() const { return n_vars; }
	int NumberPredeterminedVariables() const {return n_predetermined; }
	int NumberLags() const { return n_lags; }

	// Maximization
	bool Maximize_LogPosterior_NPSOL(double &log_posterior_optimal, TDenseVector &x_optimal, const TDenseVector &x_initial, const vector<TIndex> &blocks, double perturbation_scale=0, int pertubation_iteration=0, int block_iteration=1, int optimazation_iteration=1); 
        bool Maximize_LogPosterior_Constrained_NPSOL(double &log_posterior_optimal, TDenseVector &x_optimal, const TDenseVector &x_initial, const vector<TIndex> &blocks, 
						     double perturbation_scale=0, int perturbation_iterations=0, int block_iterations=1, int optimization_iterations=1);
	bool Maximize_LogPosterior_CSMINWEL(double &log_posterior_optimal, TDenseVector &x_optimal, const TDenseVector &x_initial, const vector<TIndex> &blocks, double perturbation_scale=0, int pertubation_iteration=0, int block_iteration=1, int optimazation_iteration=1); 

	bool Optimize_ConstantRegime_NPSOL(double &log_posterior_optimal, TDenseVector &x_optimal, const TDenseVector &x_initial, const double perturbation_scale=0, int pertubation_iteration=0, int optimization_iteration=1); 
	bool Optimize_ConstantRegime_CSMINWEL(double &log_posterior_optimal, TDenseVector &x_optimal, const TDenseVector &x_initial, double perturbation_scale=0, int pertubation_iteration=0, int optimization_iteration=1);

	double MaxSquareNormEigenvalues(const TDenseVector & p) { return MaxSquareNormEigenvalues(p.vector);}
	using TTimeSeries::LogLikelihood;
	using TTimeSeries::LogPrior;
	
	// Blocing scheme
	vector<TIndex> ConstructBlocks(int ); 

	// Normalization
	TDenseVector GetNormalizedParameter(const TDenseVector &parameter); 

  // added 4/13/15 DW
  // info
  TDenseMatrix GetA0(const TDenseVector &parameters, int t, int regime);
  TDenseMatrix GetAplus(const TDenseVector &parameters, int t, int regime);
  TDenseMatrix GetXi(const TDenseVector &parameters, int t, int regime);

  // Impulse responses
  TDenseMatrix ImpulseResponse(int horizon, const TDenseVector &parameters, int regime);
  //TDenseMatrix ImpulseResponse(int horizon, const TDenseVector &percentiles, const std::string &draws_file, int offset, int length, int n_draws);
  TDenseMatrix ImpulseResponseTrendCycle(int horizon, const TDenseVector &parameters, int regime, int &trend_dim, double cutoff);
  TDenseMatrix TrendCycle(const TDenseVector &parameters, int n_paths, double cutoff);

  // probabilities
  void FilteredProbabilities(TDenseMatrix &Pt, TDenseMatrix &Ptm1);
  TDenseMatrix SmoothedProbabilities(void);
  std::vector<int> DrawRegimes(const TDenseMatrix &Pt, const TDenseMatrix &Ptm1);

};

class TVSBVAR_MinusLogPosterior_NPSOL
{
public:
        static TVSBVAR *model;
        static void *function(int *mode, int *n, double *x_array, double *f, double *g, int *n_state);
        static TDenseVector x_complete;
        static TIndex vary_part;

        static void ExcludeExplosive(int* mode, int* ncnln, int* n, int* ldJ, int* needc, double* x, double* c, double* cJac, int* nstate);
};

class TVSBVAR_MinusLogPosterior_CSMINWEL
{
public:
        static TVSBVAR *model;
        static double function(double *x, int n, double **args, int *dims);

        static TDenseVector x_complete;
        static TIndex vary_part;
};

// Added by HWu on 04/06/2016

class TVSBVAR_normalized : public TVSBVAR
{
protected:
	TDenseMatrix Ahat;
	virtual double LogPrior(void);
public:
	// constructors
	TVSBVAR_normalized(void);
	TVSBVAR_normalized(const TVSBVAR_normalized &TimeSeries);
	TVSBVAR_normalized(istream &restriction_file, TData_predetermined *_data, TRegimeProcess *_regime_process, THyperParameter *_hyper_parameter, int _periods_per_year, double _variance_scale);
	
	// destructor
	~TVSBVAR_normalized() { };
	
	using TVSBVAR::LogPrior; 	
	virtual bool DrawParametersFromPrior(double *parameters);
}; 
#endif
