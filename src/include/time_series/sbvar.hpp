#ifndef _SBVAR_HEADER_
#define _SBVAR_HEADER_
// Structural Bayesian Vector Auto-Regression Class

#include <string>
#include <vector>

#include "dw_dense_matrix.hpp"
#include "dw_time_series.hpp"

/*
   The SBVAR class is designed to be the base class for structural Bayesian 
   vector autoregressions.  There are no restrictions and the prior is flat.
*/
class SBVAR : public TTimeSeries
{
protected:
public:
  // basic info
  int n_vars;
  int n_lags;
  int n_exogenous;
  int n_predetermined;

  // data
  TDenseMatrix YY;      // YY = Y' * Y
  TDenseMatrix XX;      // XX = X' * X
  TDenseMatrix XY;      // XY = X' * Y

  // constants
  double log_likelihood_constant;

  // SBVAR representation
  TDenseMatrix A0;
  TDenseMatrix Aplus;

  // temperature level
  double lambda;
  double lambda_T;
  double lambda_bar;
  
  // log likelihood
  virtual double LogLikelihood(void);

  // log prior
  virtual double LogPrior(void) { return 0.0; };

  // setup
  void SetupSBVAR(void);

public:
  using TTimeSeries::LogLikelihood;
  using TTimeSeries::LogPrior;

  // constructors
  SBVAR(const SBVAR &model);
  SBVAR(TData_predetermined *Data);
  
  // destructor
  virtual ~SBVAR() { };

  // set current parameters values
  virtual bool SetParameters(double *Parameters);
  virtual bool SetParameters(const TDenseMatrix &A0, const TDenseMatrix &Aplus);

  // return current parameter values
  virtual bool GetParameters(double *Parameters) const;

  // draw parameters from prior distribution
  using TTimeSeries::DrawParametersFromPrior; 

  // default parameter values
  virtual void DefaultParameters(void);

  // number of free parameters
  virtual int NumberParameters(void) const { return n_vars*(n_vars+n_predetermined); };

  // sbvar info
  int NumberVariables(void) const { return n_vars; };
  int NumberLags(void) const { return n_lags; };
  int NumberExogeneous(void) const { return n_exogenous; };
  int NumberPredetermined(void) const { return n_predetermined; };

  TDenseMatrix PredeterminedData(void) const { return static_cast<TData_predetermined*>(data)->PredeterminedData(); };
  TDenseVector PredeterminedData(int t) const { return static_cast<TData_predetermined*>(data)->PredeterminedData(t); };

  TDenseMatrix GetA0(void) const { return A0; };
  TDenseMatrix GetA0(double *Parameters) { SetParameters(Parameters); return A0; };
  TDenseMatrix GetAplus(void) const { return Aplus; };
  TDenseMatrix GetAplus(double *Parameters) { SetParameters(Parameters); return Aplus; };

  // reduced form maximum likelihood estimate
  void OLSReducedFormEstimate(TDenseMatrix &B, TDenseMatrix &Sigma);

  // blocking scheme
  virtual std::vector<TIndex> ConstructBlocks(int) const; 

  // normalization
  virtual TDenseVector GetNormalizedParameter(const TDenseVector &parameter); 

  // Set temperature
  virtual void SetTemperature(double Lambda, double LambdaBar);

  // simulate data
  virtual TDenseMatrix SimulateData(const TDenseVector &parameters, int n_obs);
};


/*
   The SBVAR_symmetric class will support either the flat prior or any prior that
   can be expressed in terms of dummy observations.  The class does not support
   restrictions.
*/
class SBVAR_symmetric : public SBVAR
{
  //protected:
public:
  // prior
  bool flat_prior;
  TDenseMatrix prior_Y;
  TDenseMatrix prior_X;
  TDenseMatrix prior_YY;
  TDenseMatrix prior_XX;
  TDenseMatrix prior_XY;

  double log_prior_constant;

  // log prior
  virtual double LogPrior(void);

  // setup
  void SetupSBVAR_symmetric(void);

public:
  using SBVAR::LogPrior;
  using SBVAR::GetNormalizedParameter; 
  //using SBVAR::ConstructBlocks; 

  // constructors
  SBVAR_symmetric(const SBVAR_symmetric &model);
  SBVAR_symmetric(TData_predetermined *Data);
  SBVAR_symmetric(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX);
  SBVAR_symmetric(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale);

  // destructor
  virtual ~SBVAR_symmetric() { };

  // Sims-Zha prior
  void SimsZhaDummyObservations(const TDenseVector &Mu, double PeriodsPerYear, double VarianceScale);

  // draw parameters from prior distribution
  virtual bool DrawParametersFromPrior(double *parameters); 

  // Set temperature
  virtual void SetTemperature(double Lambda, double LambdaBar);
};


/*
   Allows for restrictions of the form
     
     A0(i,:)' = U(i) * b(i)
     Aplus(i,:)' = V(i) * g(i)

   U and V must have orthonormal columns.
*/
class SBVAR_symmetric_linear : public SBVAR_symmetric
{
protected:
  // Restriction Matrices
  std::vector<TDenseMatrix> U;
  std::vector<TDenseMatrix> V;

  // free parameters layout
  std::vector<int> begin_b, dim_b, begin_g, dim_g;
  TDenseVector parameters;

  // simulation information
  bool simulation_info_set;
  std::vector<TDenseMatrix> Simulate_SqrtH;
  std::vector<TDenseMatrix> Simulate_P;
  std::vector<TDenseMatrix> Simulate_SqrtS;
  std::vector<TDenseMatrix> Simulate_USqrtS;

  // simulatiion information for prior
  bool prior_simulation_info_set;
  std::vector<TDenseMatrix> PriorSimulate_SqrtVariance;

  // protected helpers for constructors
  void SetupRestrictions(void);
  void SetLogPriorConstant(void);
  virtual void SetSimulationInfo(void);
  virtual void SetPriorSimulationInfo(void);

  // protected functions for marginal data density computation
  double LogConditionalA0_gibbs(const TDenseVector &p, int i, int ndraws, int thin, int burn_in);
  double LogConditionalA0_kernel(const TDenseMatrix &A, const TDenseVector &b, int i);

public:
  using SBVAR_symmetric::LogPosterior;
  using SBVAR_symmetric::GetNormalizedParameter;

  // constructors
  SBVAR_symmetric_linear(const SBVAR_symmetric_linear &model);
  SBVAR_symmetric_linear(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
  SBVAR_symmetric_linear(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
  SBVAR_symmetric_linear(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);

  // destructor
  virtual ~SBVAR_symmetric_linear() { };

  // set current parameter values
  virtual bool SetParameters(double *parameters);
  virtual bool SetParameters(const TDenseMatrix &A0, const TDenseMatrix &Aplus);

  // return current parameter values
  virtual bool GetParameters(double *parameters) const;

  // draw parameters from prior distribution
  virtual bool DrawParametersFromPrior(double *parameters); 

  // number of free parameters
  virtual int NumberParameters(void) const { return parameters.dim; };

  // simulation routines
  virtual void SimulateA0(int idx=0);
  virtual void SimulateAplus(void);
  virtual void DrawInitialA0(void);
  virtual double MaximizePosterior(double tolerance=1.0e-6, bool verbose=false);
  TDenseMatrix Simulate(int n, int burn_in=10);
  TDenseMatrix Simulate_serial_correlation(int n, int burn_in=100, int thin=1);
  TDenseMatrix Simulate_serial_correlation(double *Parameters, int n, int burn_in=100, int thin=1) { SetParameters(Parameters); return Simulate_serial_correlation(n,burn_in,thin); };

  // marginal data density
  double LogMarginalDataDensity(int ndraws=10000, int thin=10, int burn_in=100) { return LogPosteriorIntegral(ndraws,thin,burn_in); };
  double LogMarginalDataDensity(TDenseVector p, int ndraws=10000, int thin=10, int burn_in=100) { return LogPosteriorIntegral(p,ndraws,thin,burn_in); };
  double LogPosteriorIntegral(int ndraws=10000, int thin=10, int burn_in=100) { MaximizePosterior(); return LogPosteriorIntegral(parameters,ndraws,thin,burn_in); };
  double LogPosteriorIntegral(TDenseVector p, int ndraws=10000, int thin=10, int burn_in=100);

  // blocking scheme
  virtual std::vector<TIndex> ConstructBlocks(int) const;  

  // Set temperature
  virtual void SetTemperature(double Lambda, double LambdaBar);
};

/*
   Normalizes by forcing the diagonal of A0 to be positive
*/
class SBVAR_symmetric_linear_normalized : public SBVAR_symmetric_linear
{
public:
  using TTimeSeries::LogPrior;

  // constructors
  SBVAR_symmetric_linear_normalized(const SBVAR_symmetric_linear_normalized &model);
  SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
  SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
  SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);

  // destructor
  virtual ~SBVAR_symmetric_linear_normalized() { };

  // log prior
  virtual double LogPrior(void);

  // draw parameters from prior distribution
  virtual bool DrawParametersFromPrior(double *parameters); 
};


// Added by HWu on 04/06/2015
/*
 * Normalizes using the method in "Likelihood preserving normalization in multiple equation models." 
 * Keep a_k if innerproduct(e_k, a^hat_k, A_inverse) > 0 and replace a_k with -a_k if innerproduct (e_k, a^hat_k, A_inverse) < 0
 */
class SBVAR_symmetric_linear_normalized_generalization : public SBVAR_symmetric_linear
{
protected:
	TDenseMatrix Ahat; 
public:
	using TTimeSeries::LogPrior;
	
	// constructors
	SBVAR_symmetric_linear_normalized_generalization(const SBVAR_symmetric_linear_normalized_generalization &model); 
	SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
	SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);
	SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV);

	// destructor
	virtual ~SBVAR_symmetric_linear_normalized_generalization() {}; 
	
	// log prior
	virtual double LogPrior(); 
	
	// draw parameters from prior distribution
	virtual bool DrawParametersFromPrior(double *parameters); 
}; 



//===============================================================================
//=== Auxiliary routines
//===============================================================================
// Function for simulation routine
TDenseVector GetNullVector(const TDenseMatrix &X, int j);

// Compute number of lags and check dimensions
int NumberLags(const TDenseMatrix &A0, const TDenseMatrix &Aplus, int n_exogenous);
int NumberLags(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant);

// Unconditional moments
TDenseMatrix ReducedForm(const TDenseMatrix &A0, const TDenseMatrix &Aplus);
TDenseMatrix ConditionalVariance(const TDenseMatrix &A0);
//void ReducedForm(TDenseMatrix &B, TDenseMatrix &Sigma, const TDenseMatrix &A0, const TDenseMatrix &Aplus);
TDenseMatrix CompanionMatrix(const TDenseMatrix &B, int n_lags);
TDenseMatrix CompanionMatrix(const TDenseMatrix &A0, const TDenseMatrix &Aplus, int n_lags);
TDenseMatrix UnconditionalVariance(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant);
TDenseVector UnconditionalMean(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant);

// Simulate artificial data
TDenseMatrix SimulateData(int T, const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant, const TDenseVector &initial_value, int burn_in=0);
TDenseMatrix SimulateData(int T, const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant, int burn_in=0);

// Reading restriction matrices and initial values before an instance of SBVAR_symmetric_linear is created.
void SetupRestrictionMatrices(std::vector<TDenseMatrix> &U, std::vector<TDenseMatrix> &V, std::istream &input);
void SetupRestrictionMatrices(std::vector<TDenseMatrix> &U, std::vector<TDenseMatrix> &V, TDenseMatrix &A0, TDenseMatrix &Aplus, std::istream &input);

#endif
