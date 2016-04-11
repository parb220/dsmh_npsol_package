#include "sbvar.hpp"
#include "dw_rand.h"
#include "dw_math.h"
#include <math.h>

//===============================================================================
// BVAR class
//===============================================================================
SBVAR::SBVAR(const SBVAR &model) 
  : TTimeSeries(model), 
    n_vars(model.n_vars), 
    n_lags(model.n_lags), 
    n_exogenous(model.n_exogenous),
    n_predetermined(model.n_predetermined), 
    YY(model.YY), 
    XX(model.XX), 
    XY(model.XY),
    log_likelihood_constant(model.log_likelihood_constant), 
    A0(model.A0), 
    Aplus(model.Aplus),
    lambda(model.lambda),
    lambda_T(model.lambda_T),
    lambda_bar(model.lambda_bar)
{ }

SBVAR::SBVAR(TData_predetermined *Data)
  : TTimeSeries(Data), 
    n_vars(Data->NumberVariables()), 
    n_lags(Data->NumberLags()), 
    n_exogenous(Data->NumberPredeterminedVariables()-n_vars*n_lags),
    n_predetermined(Data->NumberPredeterminedVariables()), 
    YY(0,0), 
    XX(0,0),
    XY(0,0), 
    log_likelihood_constant(0),
    A0(n_vars,n_vars,false),
    Aplus(n_vars,n_predetermined,false),
    lambda(1.0),
    lambda_T(Data->NumberObservations()),
    lambda_bar(1.0)
{ 
  SetupSBVAR();
}

void SBVAR::SetupSBVAR(void)
{
  TData_predetermined *data=(TData_predetermined*)pData();
  YY=lambda*TransposeMultiply(data->Data(),data->Data());
  XX=lambda*TransposeMultiply(data->PredeterminedData(),data->PredeterminedData());
  XY=lambda*TransposeMultiply(data->PredeterminedData(),data->Data());
  log_likelihood_constant=-lambda_T*n_vars*0.918938533204673;      // 0.918938533204673 = 0.5*ln(2*pi)
  delete data;
}


bool SBVAR::SetParameters(double *Parameters)
{
  A0.UniqueMemory(n_vars,n_vars,false);
  memcpy(A0.matrix,Parameters,n_vars*n_vars*sizeof(double));
  Aplus.UniqueMemory(n_vars,n_predetermined,false);
  memcpy(Aplus.matrix,Parameters+n_vars*n_vars,n_vars*n_predetermined*sizeof(double));
  return true;
}

bool SBVAR::SetParameters(const TDenseMatrix &A_0, const TDenseMatrix &A_plus)
{
  if ((n_vars != A_0.rows) || (n_vars != A_0.cols) || (n_vars != A_plus.rows) || (n_predetermined != A_plus.cols)) 
    throw dw_exception("VARToParameters() - invalid matrix dimensions");
  A0=A_0;
  Aplus=A_plus;
  return true;
}

bool SBVAR::GetParameters(double *Parameters) const
{
  TDenseMatrix A0_copy, Aplus_copy;  
  A0_copy.CopyContent(A0); 
  Aplus_copy.CopyContent(Aplus); 
  A0_copy.ForceRowMajor();
  memcpy(Parameters,A0_copy.matrix,n_vars*n_vars*sizeof(double));
  Aplus_copy.ForceRowMajor();
  memcpy(Parameters+n_vars*n_vars,Aplus_copy.matrix,n_vars*n_predetermined*sizeof(double));
  return true;
}

void SBVAR::DefaultParameters(void)
{
  A0.RandomNormal(n_vars,n_vars);
  Aplus.Zeros(n_predetermined,n_vars);
}

double SBVAR::LogLikelihood(void)
{
  double log_likelihood=log_likelihood_constant + lambda_T*LogAbsDeterminant(A0);
  TDenseVector a0(n_vars), aplus(n_predetermined);

  for (int i=n_vars-1; i >= 0; i--)
    {
      a0.RowVector(A0,i);
      aplus.RowVector(Aplus,i);
      log_likelihood+=-0.5*(InnerProduct(a0,a0,YY) - 2.0*InnerProduct(aplus,a0,XY) + InnerProduct(aplus,aplus,XX));
    }

  return log_likelihood;
}

void SBVAR::OLSReducedFormEstimate(TDenseMatrix &B, TDenseMatrix &Sigma)
{
  // Y = Data()
  // X = PredeterminedData()

  TDenseMatrix U, V;
  TDenseVector d;

  // X = U*D*V'
  SVD(U,d,V,PredeterminedData(),1);

  // Compute the generalize inverse of D.
  d.UniqueMemory();
  for (int i=d.dim-1; i > 0; i--)
    d.vector[i]=(d.vector[i] > d.vector[0]*MACHINE_EPSILON) ? 1.0/d.vector[i] : 0.0;
  d.vector[0]=(d.vector[0] > 0.0) ? 1.0/d.vector[0] : 0.0;

  // B = inv(X'*X)*X'*Y = V*D^(-2)*V'*V*D*U'*Y' = V*D^(-1)*U'*Y.
  TDenseMatrix Z=Transpose(U)*Data();
  B=Transpose(V*(DiagonalMatrix(d)*Z));

  // Sigma = (Y'*Y -  Y'*X*inv(X'*X)*X'*Y)/NumberObservatons() 
  //       = (Y'*Y - Y'*U*U*Y)/NumberObservations()
  Sigma=(1.0/(double)NumberObservations())*(YY - Transpose(Z)*Z);
}

void SBVAR::SetTemperature(double Lambda, double LambdaBar)
{
  if (Lambda < 0)
    throw dw_exception("SBVAR::SetTemperature(): lambda must be non-negative");
  if (LambdaBar <= 0.0)
    throw dw_exception("SBVAR::SetTemperature(): lambda bar must be positive");
  lambda=Lambda;
  lambda_T=lambda*NumberObservations();
  lambda_bar=LambdaBar;
  SetupSBVAR();
}

TDenseMatrix SBVAR::SimulateData(const TDenseVector &parameters, int n_obs)
{
  TData_predetermined *data=(TData_predetermined*)pData();

  int n_exogenous=data->NumberPredeterminedVariables() - data->NumberLags()*data->NumberVariables();
  if ((n_exogenous > 1) || ((n_exogenous == 1) && !(data->IsConstant())))
    throw dw_exception("SBVAR::SimulateData() - the only exogenous variable permissible is a constant");

  if (parameters.dim != NumberParameters())
    throw dw_exception("SBVAR::SimulationData() - invalid number of parameters");
 
 
  SetParameters(parameters.vector);
  TDenseMatrix A0=GetA0(), Aplus=GetAplus();
  return ::SimulateData(n_obs+n_lags,A0,Aplus,data->IsConstant(),n_obs);
}

//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// BVAR_symmetric class
//===============================================================================
SBVAR_symmetric::SBVAR_symmetric(const SBVAR_symmetric &model)
  : SBVAR(model), flat_prior(model.flat_prior), prior_Y(model.prior_Y), prior_X(model.prior_X),
    prior_YY(model.prior_YY), prior_XX(model.prior_XX), prior_XY(model.prior_XY), 
    log_prior_constant(model.log_prior_constant)
{ }

// Dummy observation prior
SBVAR_symmetric::SBVAR_symmetric(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX)
  : SBVAR(Data), flat_prior(false), prior_Y(PriorY), prior_X(PriorX), 
    prior_YY(), prior_XX(), prior_XY(), log_prior_constant(0.0)
{ 
  SetupSBVAR_symmetric();
}

// Flat prior
SBVAR_symmetric::SBVAR_symmetric(TData_predetermined *Data)
  : SBVAR(Data), flat_prior(true), prior_Y(0,n_vars), prior_X(0,n_predetermined),
    prior_YY(), prior_XX(), prior_XY(), log_prior_constant(0.0)
{ 
  SetupSBVAR_symmetric();
}

// Sims-Zha prior
SBVAR_symmetric::SBVAR_symmetric(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale)
  : SBVAR(Data), flat_prior(false), prior_Y(), prior_X(),
    prior_YY(), prior_XX(), prior_XY(), log_prior_constant(0.0)
{
  SimsZhaDummyObservations(Hyperparameters,PeriodsPerYear,VarianceScale);
  SetupSBVAR_symmetric();
}

void SBVAR_symmetric::SetupSBVAR_symmetric(void)
{
  prior_YY=TransposeMultiply(prior_Y,prior_Y);
  prior_XX=TransposeMultiply(prior_X,prior_X);
  prior_XY=TransposeMultiply(prior_X,prior_Y);

  if (flat_prior)
    log_prior_constant=0.0;
  else
    {
      TDenseMatrix S(n_vars+n_predetermined,n_vars+n_predetermined);
      S.Insert(0,0,prior_YY);
      S.Insert(n_vars,0,-prior_XY);
      S.Insert(0,n_vars,-Transpose(prior_XY));
      S.Insert(n_vars,n_vars,prior_XX);
      log_prior_constant=n_vars*(-0.918938533204673*(n_vars+n_predetermined) + 0.5*LogAbsDeterminant(S));  // 0.918938533204673 = 0.5*ln(2*pi)
    }

  prior_YY*=lambda_bar;
  prior_XX*=lambda_bar;
  prior_XY*=lambda_bar;
  log_prior_constant*=lambda_bar;
}

double SBVAR_symmetric::LogPrior(void)
{
  if (flat_prior) return 0.0;

  double log_prior=log_prior_constant;
  TDenseVector a0(n_vars), aplus(n_predetermined);

  for (int i=n_vars-1; i >= 0; i--)
    {
      a0.RowVector(A0,i);
      aplus.RowVector(Aplus,i);
      log_prior+=-0.5*(InnerProduct(a0,a0,prior_YY) - 2.0*InnerProduct(aplus,a0,prior_XY) + InnerProduct(aplus,aplus,prior_XX));
    }

  return log_prior;
}

bool SBVAR_symmetric::DrawParametersFromPrior(double *Parameters)
{
  if (!flat_prior)
    {
      TDenseVector a0_aplus(n_vars+n_predetermined,0.0), x(n_vars+n_predetermined,0.0); 
      TDenseMatrix covariance(n_vars+n_predetermined, n_vars+n_predetermined,0.0);
      covariance.Insert(0,0,prior_YY); 
      covariance.Insert(0,n_vars,-Transpose(prior_XY)); 
      covariance.Insert(n_vars,0,-prior_XY); 
      covariance.Insert(n_vars,n_vars,prior_XX);
      covariance = 0.5*(covariance+Transpose(covariance)); 

      // eig analysis of covariance
      TDenseVector EigValue(prior_YY.rows+prior_XX.rows,0.0); 
      TDenseMatrix EigVector(prior_YY.rows+prior_XX.rows, prior_YY.cols+prior_YY.cols, 0.0);
      Eig(EigValue, EigVector, covariance); 
		 
      for (int i=n_vars-1; i>=0; i--)
      	{
      	  x.RandomNormal(); 
      	  a0_aplus.Insert(0, EigVector*(DotMultiply(EigValue,x))); 
      	  A0.InsertRowMatrix(i,0,a0_aplus,0,n_vars-1); 
      	  Aplus.InsertRowMatrix(0,0,a0_aplus,n_vars,a0_aplus.dim-1); 
      	}		
      GetParameters(Parameters); 
      return true; 
    }
  else
    {
      TDenseVector a0_aplus(n_vars+n_predetermined,0.0); 
      for (int i=n_vars-1; i>=0; i--)
	{
	  a0_aplus.RandomNormal(); 
	  A0.InsertRowMatrix(i,0,a0_aplus,0,n_vars-1);
	  Aplus.InsertRowMatrix(0,0,a0_aplus,n_vars,a0_aplus.dim-1);
	}
      GetParameters(Parameters); 
      return true; 
    }
}

void SBVAR_symmetric::SetTemperature(double Lambda, double LambdaBar)
{
  SBVAR::SetTemperature(Lambda,LambdaBar);
  SetupSBVAR_symmetric();
}

/*
   Hyperparameters - vector of length 7 (if of length 6, then default value of 1.0 is used for Hyperparameters(3))
   
   Sets prior_X and prior_Y
*/
void SBVAR_symmetric::SimsZhaDummyObservations(const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale)
{
  // check hyperparameters
  TDenseVector Mu(7);
  if (Hyperparameters.dim == 6) 
    {
      for (int i=0; i < 7; i++)
	if (i < 3)
	  Mu(i)=Hyperparameters(i);
	else if (i == 3)
	  Mu(i)=1.0;
	else
	  Mu(i)=Hyperparameters(i-1);
    }
  else if (Hyperparameters.dim == 7)
    Mu=Hyperparameters;
  else
    throw dw_exception("Sims-Zha prior requires seven hyperparameters");
  
  if (PeriodsPerYear <= 0)
    throw dw_exception("Periods per year must be positive");

  if (VarianceScale <= 0)
    throw dw_exception("variance scale must be positive");

  for (int i=6; i >= 0; i--)
    if (Mu(i) <= 0) throw dw_exception("Sims-Zha hyperparameters must be positive");

  // data info
  int n_obs=NumberObservations();
  TDenseMatrix Y=Data();
  TDenseMatrix X=PredeterminedData();

  // compute mean of initial data contained in the first observaton of the predetermined data
  TDenseVector m(n_vars,0.0);
  if (n_lags > 0)
    {
      for (int i=n_lags-1; i >= 0; i--)
	for (int j=n_vars-1; j >= 0; j--)
	  m(j)+=X(0,i*n_vars+j);
      m=(1.0/(double)n_lags)*m;
    }

  // compute variance of univariate AR residuals
  TDenseVector s(n_vars,0.0), y(n_obs), e(n_obs);
  TDenseMatrix Q, R, M(n_obs,n_lags+1);
  for (int i=n_vars-1; i >= 0; i--)
    {
      for (int j=n_lags-1; j >= 0; j--)
	M.Insert(0,j,X,0,n_obs-1,j*n_vars+i,j*n_vars+i);
      M.Insert(0,n_lags,X,0,n_obs-1,n_predetermined-1,n_predetermined-1);
      QR(Q,R,M);
      y.ColumnVector(Y,i);
      e=y - Q*(Transpose(Q)*y);
      s(i)=Norm(e)/sqrt((double)n_obs);
    }

  // dummy observations
  prior_Y.Zeros(2+n_vars*(n_lags+2),n_vars);
  prior_X.Zeros(2+n_vars*(n_lags+2),n_predetermined);

  // prior on a(k,i): dummy observations of the form
  //   y=s(i)/mu(1) * e(i,n_vars)
  //   x=0
  int row=0;
  for (int i=0; i < n_vars; i++)
    prior_Y(row+i,i)=s(i)/Mu(0);
  row+=n_vars;

  // prior on constant: dummy observation of the form
  //   y=0
  //   x=1.0/(mu(1)*mu(3)) * e(n_predermined,n_predetermined)
  prior_X(row,n_predetermined-1)=1.0/(Mu(0)*Mu(2));
  row+=1;

  if (n_lags > 0)
    {
      // random walk prior: dummy observations of the form
      //   y=s(i)/(mu(2)*mu(1)) * e(i,n_vars)
      //   x=s(i)/(mu(2)*mu(1)) * e(i,n_predetermined)
      for (int i=0; i < n_vars; i++)
	{
          prior_Y(row+i,i)=s(i)/(Mu(0)*Mu(1));
          prior_X(row+i,i)=s(i)/(Mu(0)*Mu(1));
	}
      row+=n_vars;
        
      // lag decay prior: dummy observations of the form
      //   y=0
      //   x=s(i)*(4*j/PeriodsPerYear+1)^mu(4)/(mu(1)*mu(2)) * e(j)*nvars+i,n_pred)
      for (int j=1; j < n_lags; j++)
	{
	  for (int i=0; i < n_vars; i++)
	    prior_X(row+i,j*n_vars+i)=s(i)*pow(4.0*Mu(3)*(double)j/PeriodsPerYear+1.0,Mu(4))/(Mu(0)*Mu(1));
	  row+=n_vars;
	}
    
      // sums-of-coefficients prior: dummy observations of the form
      //   y=mu(5)*m(i) * e(i,n_vars)
      //   x=sum_j { mu(5)*m(i) * e((j-1)*n_vars+i,n_pred) }
      for (int i=0; i < n_vars; i++)
	{
	  prior_Y(row+i,i)=Mu(5)*m(i);
	  for (int j=0; j < n_lags; j++)
	    prior_X(row+i,j*n_vars+i)=Mu(5)*m(i);
	}
      row+=n_vars;
    
      // co-persistence prior: dummy observation of the form
      //   y=sum_i { mu(6)*m(i) * e(i,n_vars) }
      //   x=sum_i,j { mu(6)*m(i) * e((j-1)*n_vars+i,n_pred) } + mu(6) * e(n_pred,n_pred)
      for (int i=0; i < n_vars; i++)
	{
	  prior_Y(row,i)=Mu(6)*m(i);
	  for (int j=0; j < n_lags; j++)
	    prior_X(row,j*n_vars+i)=Mu(6)*m(i);
	}
      prior_X(row,n_predetermined-1)=Mu(6);
    }

  // variance scale
  prior_Y=sqrt(1.0/VarianceScale)*prior_Y;
  prior_X=sqrt(1.0/VarianceScale)*prior_X;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// BVAR_symmetric_linear Class
//===============================================================================
SBVAR_symmetric_linear::SBVAR_symmetric_linear(const SBVAR_symmetric_linear &model)
  : SBVAR_symmetric(model), U(model.U), V(model.V), begin_b(model.begin_b), dim_b(model.begin_b),
    begin_g(model.begin_g), dim_g(model.dim_g), parameters(model.parameters),
    simulation_info_set(model.simulation_info_set), Simulate_SqrtH(model.Simulate_SqrtH), 
    Simulate_P(model.Simulate_P), Simulate_SqrtS(model.Simulate_SqrtS),Simulate_USqrtS(model.Simulate_USqrtS),
    prior_simulation_info_set(model.prior_simulation_info_set), PriorSimulate_SqrtVariance(model.PriorSimulate_SqrtVariance)
{ }

// symmetric prior
SBVAR_symmetric_linear::SBVAR_symmetric_linear(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric(Data,PriorY,PriorX), U(iU), V(iV), simulation_info_set(false), prior_simulation_info_set(false)
{
  SetupRestrictions();
  SetLogPriorConstant();
}

// Flat prior
SBVAR_symmetric_linear::SBVAR_symmetric_linear(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric(Data), U(iU), V(iV), simulation_info_set(false), prior_simulation_info_set(false)
{
  SetupRestrictions();
  SetLogPriorConstant();
}

// Sims-Zha prior
SBVAR_symmetric_linear::SBVAR_symmetric_linear(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric(Data,Hyperparameters,PeriodsPerYear,VarianceScale), U(iU), V(iV), simulation_info_set(false), prior_simulation_info_set(false)
{
  SetupRestrictions();
  SetLogPriorConstant();
}

void SBVAR_symmetric_linear::SetupRestrictions(void)
{
  // consistancy checks
  if ((n_vars != (int)U.size()) || (n_vars != (int)V.size())) 
    throw dw_exception("BVAR_symmetric_linear(): invalid number of restriction matrices");

  for (int i=n_vars-1; i >= 0; i--)
    if ((n_vars != U[i].rows) || (n_predetermined != V[i].rows))
      throw dw_exception("BVAR_symmetric_linear(): invalid dimensions of restriction matrices");

  // setup begin_b and dim_b
  begin_b.resize(n_vars);
  dim_b.resize(n_vars);
  begin_b[0]=0;
  dim_b[0]=U[0].cols;
  for (int i=1; i < n_vars; i++)
    {
      begin_b[i]=begin_b[i-1]+dim_b[i-1];
      dim_b[i]=U[i].cols;
    }

  // setup begin_g and dim_g
  begin_g.resize(n_vars);
  dim_g.resize(n_vars);
  begin_g[0]=begin_b[n_vars-1]+dim_b[n_vars-1];
  dim_g[0]=V[0].cols;
  for (int i=1; i < n_vars; i++)
    {
      begin_g[i]=begin_g[i-1]+dim_g[i-1];
      dim_g[i]=V[i].cols;
    }

  // setup parameters
  int n_parameters=0;
  for (int i=n_vars-1; i >= 0; i--) n_parameters+=dim_b[i]+dim_g[i];
  parameters.Resize(n_parameters);
}

void SBVAR_symmetric_linear::SetLogPriorConstant(void)
{
  TDenseMatrix S(n_vars+n_predetermined,n_vars+n_predetermined);
  S.Insert(0,0,prior_YY);
  S.Insert(n_vars,0,-prior_XY);
  S.Insert(0,n_vars,-Transpose(prior_XY));
  S.Insert(n_vars,n_vars,prior_XX);
  S*=1.0/lambda_bar;
  log_prior_constant=-0.918938533204673*parameters.dim;      // 0.918938533204673 = 0.5*ln(2*pi)
  for (int i=n_vars-1; i >= 0; i--)
    {
      TDenseMatrix T(n_vars+n_predetermined,dim_b[i]+dim_g[i],0.0);
      T.Insert(0,0,U[i]);
      T.Insert(n_vars,dim_b[i],V[i]);
      log_prior_constant+=0.5*LogAbsDeterminant(Transpose(T)*(S*T));
    }
  log_prior_constant*=lambda_bar;
}

/*
   See Waggoner and Zha, "A Gibbs sampler for structural vector autoregressions", 
   JEDC 2003, for discription of notations.  We take the square root of a 
   symmetric and positive definite X to be any matrix Y such that Y*Y'=X.  Note 
   that this is not the usual definition because we do not require Y to be 
   symmetric and positive definite.
*/
void SBVAR_symmetric_linear::SetSimulationInfo(void)
{
  if (NumberObservations() == 0)
    throw dw_exception("SetSimulationInfo(): cannot simulate if no observations");

  TDenseMatrix all_YY, all_XY, all_XX;
  if (flat_prior)
    {
      all_YY=YY;
      all_XY=XY;
      all_XX=XX;
    }
  else
    {
      TDenseMatrix all_Y, all_X;
      all_Y=VCat(sqrt(lambda)*Data(),sqrt(lambda_bar)*prior_Y);
      all_X=VCat(sqrt(lambda)*PredeterminedData(),sqrt(lambda_bar)*prior_X);
      all_YY=Transpose(all_Y)*all_Y;
      all_XY=Transpose(all_X)*all_Y;
      all_XX=Transpose(all_X)*all_X;
    }

  Simulate_SqrtH.resize(n_vars);
  Simulate_P.resize(n_vars);
  Simulate_SqrtS.resize(n_vars);
  Simulate_USqrtS.resize(n_vars);

  for (int i=n_vars-1; i >= 0; i--)
    {
      TDenseMatrix invH=Transpose(V[i])*(all_XX*V[i]);
      Simulate_SqrtH[i]=Inverse(Cholesky(invH,CHOLESKY_UPPER_TRIANGULAR),SOLVE_UPPER_TRIANGULAR);
      Simulate_P[i]=Simulate_SqrtH[i]*(Transpose(Simulate_SqrtH[i])*(Transpose(V[i])*(all_XY*U[i])));
      Simulate_SqrtS[i]=sqrt(lambda_T)*Inverse(Cholesky(Transpose(U[i])*(all_YY*U[i]) - Transpose(Simulate_P[i])*(invH*Simulate_P[i]),CHOLESKY_UPPER_TRIANGULAR),SOLVE_UPPER_TRIANGULAR);
      Simulate_USqrtS[i]=U[i]*Simulate_SqrtS[i];
    }

  simulation_info_set=true;
}

/*
   See Waggoner and Zha, "A Gibbs sampler for structural vector autoregressions", 
   JEDC 2003, for discription of notations.  We take the square root of a 
   symmetric and positive definite X to be any matrix Y such that Y*Y'=X.  Note 
   that this is not the usual definition because we do not require Y to be 
   symmetric and positive definite.
*/
void SBVAR_symmetric_linear::SetPriorSimulationInfo(void)
{
  if (flat_prior)
    throw dw_exception("flat prior not allowed if simulating from prior");

  PriorSimulate_SqrtVariance.resize(n_vars);
  TDenseMatrix X;
  for (int i=n_vars-1; i >= 0; i--)
    {
      TDenseMatrix S(dim_b[i]+dim_g[i],dim_b[i]+dim_g[i]);
      S.Insert(0,0,TransposeMultiply(U[i],prior_YY*U[i]));
      S.Insert(dim_b[i],0,X=-TransposeMultiply(V[i],prior_XY*U[i]));
      S.Insert(0,dim_b[i],Transpose(X));
      S.Insert(dim_b[i],dim_b[i],TransposeMultiply(V[i],prior_XX*V[i]));
      PriorSimulate_SqrtVariance[i]=Inverse(Cholesky(S,CHOLESKY_UPPER_TRIANGULAR),SOLVE_UPPER_TRIANGULAR);
    }

  prior_simulation_info_set=true;
}

void SBVAR_symmetric_linear::SetTemperature(double Lambda, double LambdaBar)
{
  SBVAR_symmetric::SetTemperature(Lambda,LambdaBar);
  SetLogPriorConstant();
  simulation_info_set=false;
}

bool SBVAR_symmetric_linear::SetParameters(double *Parameters)
{
  parameters.UniqueMemory();
  memcpy(parameters.vector,Parameters,parameters.dim*sizeof(double));

  A0.UniqueMemory(n_vars,n_vars,false);
  Aplus.UniqueMemory(n_vars,n_predetermined,false);

  for (int i=n_vars-1; i >= 0; i--)
    {
      A0.InsertRowMatrix(i,0,U[i]*parameters.SubVector(begin_b[i],begin_b[i]+dim_b[i]-1));
      Aplus.InsertRowMatrix(i,0,V[i]*parameters.SubVector(begin_g[i],begin_g[i]+dim_g[i]-1));
    }

  return true;
}

bool SBVAR_symmetric_linear::SetParameters(const TDenseMatrix &A_0, const TDenseMatrix &A_plus)
{
  if ((n_vars != A_0.rows) || (n_vars != A_0.cols) || (n_vars != A_plus.rows) || (n_predetermined != A_plus.cols)) 
    throw dw_exception("VARToParameters() - invalid matrix dimensions");

  parameters.UniqueMemory();
  for (int i=n_vars-1; i >= 0; i--)
    {
      parameters.Insert(begin_b[i],Transpose(U[i])*RowVector(A_0,i));
      parameters.Insert(begin_g[i],Transpose(V[i])*RowVector(A_plus,i));
    }

  A0.UniqueMemory(n_vars,n_vars,false);
  Aplus.UniqueMemory(n_vars,n_predetermined,false);
  for (int i=n_vars-1; i >= 0; i--)
    {
      A0.InsertRowMatrix(i,0,U[i]*parameters.SubVector(begin_b[i],begin_b[i]+dim_b[i]-1));
      Aplus.InsertRowMatrix(i,0,V[i]*parameters.SubVector(begin_g[i],begin_g[i]+dim_g[i]-1));
    }

  return true;
}

bool SBVAR_symmetric_linear::GetParameters(double *Parameters) const
{
  memcpy(Parameters,parameters.vector,parameters.dim*sizeof(double));
  return true;
}

bool SBVAR_symmetric_linear::DrawParametersFromPrior(double *Parameters)
{
  if (!prior_simulation_info_set) SetPriorSimulationInfo();
  TDenseVector bg;
  for (int i=n_vars-1; i >= 0; i--)
    {
      bg=PriorSimulate_SqrtVariance[i]*RandomNormalVector(dim_b[i]+dim_g[i]);
      memcpy(Parameters+begin_b[i],bg.vector,dim_b[i]*sizeof(double));
      memcpy(Parameters+begin_g[i],bg.vector+dim_b[i],dim_g[i]*sizeof(double));
    }
  return true; 
}

/*
  Draws from the distribution

              p( b(idx), ... b(n-1) | b(0), ... b(idx-1), Y ).

  See Waggoner and Zha (JEDC, 2003) for notation and details.

  For idx <= j < n, parameters(begin_b[j]:begin_b[j]+dim_b[j]-1) is set to b(j) 
  and A0(j,:) is set to U[j]*b(j).
  
  Notes:
   Only the value of A0 is used.  Both A0 and parameters are modified.  If 
   idx = 0, the default value, then the draw is from the distribution p(b|Y).
*/
void SBVAR_symmetric_linear::SimulateA0(int idx)
{
  if ((idx < 0) || (idx >= n_vars))
    throw dw_exception("SimulateA0():  index out of range");

  int m;
  TDenseVector w1, b, g;
  double c0, c1, scale;

  if (!simulation_info_set) SetSimulationInfo();

  for (int k=idx; k < n_vars; k++)
    {
      // w is a non-zero vector that is perpendicular to all columns of A0 except the kth
      // w1 = T'(k)*U'(k)*w / ||T'(k)*U'(k)*w||
      w1=GetNullVector(A0,k)*Simulate_USqrtS[k];
      w1=(1.0/w1.Norm())*w1;

      // Draw univariate Wishard component
      b=dw_univariate_wishard_rnd(lambda_T,0.0)*w1;

      // Find largest element of w1
      scale=fabs(w1(m=0));
      for (int i=dim_b[k]-1; i > 0; i--)
	if (fabs(w1.vector[i]) > scale) scale=fabs(w1.vector[m=i]);
      c0=scale*scale;

      // Draw Gaussian component
      b.UniqueMemory();
      for (int j=0; j < m; j++)
	{
	  c1=c0+w1.vector[j]*w1.vector[j];
	  scale=dw_gaussian_rnd()/sqrt(lambda_T*c0*c1);
	  b.vector[j]-=scale*c0;
	  scale*=w1.vector[j];
	  for (int i=0; i < j; i++) b.vector[i]+=scale*w1.vector[i];
	  b.vector[m]+=scale*w1.vector[m];
	  c0=c1;
	}
      for (int j=m+1; j < dim_b[k]; j++)
	{
	  c1=c0+w1.vector[j]*w1.vector[j];
	  scale=dw_gaussian_rnd()/sqrt(lambda_T*c0*c1);
	  b.vector[j]-=scale*c0;
	  scale*=w1.vector[j];
	  for (int i=0; i < j; i++) b.vector[i]+=scale*w1.vector[i];
	  c0=c1;
	}

      // A(k,:) = U[k]*T(k)*b
      A0.InsertRowMatrix(k,0,Simulate_USqrtS[k]*b);

      /// Insert into parameters
      parameters.Insert(begin_b[k],Simulate_SqrtS[k]*b);
    }
}

/*
  For 0 <= i < n, draws g(i) conditional on b(i) and Y, which is Gaussian with 
  mean P(i)*b(i) and variance H(i). 

  See Waggoner and Zha (JEDC, 2003) for notation and details.

  For 0 <= i < n, parameters(begin_g[i]:begin_g[i]+dim_g[i]-1) is set to g(i) and
  Aplus(:,i) is set to V[i]*g(i). 

  Notes:
    Usually a call to SimulateA0() preceeds a call to SimulateAplus().  Only the
    values of parameters(begin_b[i],begin_b[i]+dim_b[i]-1) are used.  Both Aplus 
    and parameters are modified. 
*/
void SBVAR_symmetric_linear::SimulateAplus(void)
{
  TDenseVector b, g;
  if (!simulation_info_set) SetSimulationInfo();
  for (int k=n_vars-1; k >= 0; k--)
    {
      // b = T(k)*b
      b.SubVector(parameters,begin_b[k],begin_b[k]+dim_b[k]-1);

      // Draw g
      g=Simulate_P[k]*b + Simulate_SqrtH[k]*RandomNormalVector(dim_g[k]);

      // Insert into Aplus
      Aplus.InsertRowMatrix(k,0,V[k]*g);

      // Insert into parameters
      parameters.Insert(begin_g[k],g);
    }
}

/*
  Draws b(i) from the Gaussian distribution with mean zero and variance
  S(i)/T.  Sets A0(:,i) to U[i]*b(i).  Upon exit, parameters and 
  A0 are in sync, but Aplus is not.

  See Waggoner and Zha (JEDC, 2003) for notation and details.
*/
void SBVAR_symmetric_linear::DrawInitialA0(void)
{
  if (!simulation_info_set) SetSimulationInfo();
  for (int i=n_vars-1; i >= 0; i--)
    A0.InsertRowMatrix(i,0,sqrt(1.0/lambda_T)*(Simulate_USqrtS[i]*RandomNormalVector(dim_b[i])));
}

double SBVAR_symmetric_linear::MaximizePosterior(double tolerance, bool verbose)
{
  if (tolerance <= 0.0) throw dw_exception("MaximizePosterior(): tolerance must be positive");

  if (!simulation_info_set) SetSimulationInfo();

  double old_posterior=LogPosterior((parameters.RandomNormal()).vector), new_posterior, diff, normdiff;
  TDenseVector w1, b, g, old_parameters=parameters;
  int max_iteration=1000;
  for (int i=1; i <= max_iteration; i++)
    {
      for (int k=n_vars-1; k >= 0; k--)
	{
	  // w is non-zero vector w perpendicular to all columns of A0 except the kth
	  // w1 = T'(k)*U'(k)*w / ||T'(k)*U'(k)*w||
	  w1=GetNullVector(A0,k)*Simulate_USqrtS[k];
	  w1=(1.0/w1.Norm())*w1;

	  // A(k,:) = U[k]*T(k)*b
	  A0.InsertRowMatrix(k,0,Simulate_USqrtS[k]*w1);

	  // b = T(k)*w1 and g = P(k)*b
	  b=Simulate_SqrtS[k]*w1;
	  g=Simulate_P[k]*b;

	  // Insert into parameters and compute log posterior
	  parameters.Insert(begin_b[k],b);
	  parameters.Insert(begin_g[k],g);
	}

      diff=(new_posterior=LogPosterior(parameters.vector)) - old_posterior;
      normdiff=Norm(parameters - old_parameters);

      if (verbose)
	{
	  std::cout << "posterior=" << new_posterior << "  posterior improvement=" << diff << "  change in parameters=" << normdiff;
	  if (diff < 0.0) std::cout << " - Decreasing log posterior!";
	  std::cout << std::endl;
	}

      // exit?
      if ((0.0 <= diff) && (diff <= tolerance) && (normdiff < parameters.dim*tolerance)) return diff+normdiff;

      old_posterior=new_posterior;
      old_parameters=parameters;
    }
  return -diff-normdiff;
}

TDenseMatrix SBVAR_symmetric_linear::Simulate(int n, int burn_in)
{
  if ((n <= 0) || (burn_in < 0))
    throw dw_exception("Simulate(): invalid arguments");
  TDenseMatrix draws(n,NumberParameters(),false);
  for (int i=0; i < n; i++)
    {
      DrawInitialA0();
      for (int j=0; j < burn_in; j++) SimulateA0();
      SimulateAplus();
      draws.InsertRowMatrix(i,0,parameters);
    } 
  return draws;
}

TDenseMatrix SBVAR_symmetric_linear::Simulate_serial_correlation(int n, int burn_in, int thin)
{
  if ((n <= 0) || (burn_in < 0) || (thin <= 0))
    throw dw_exception("Simulate(): invalid arguments");
  TDenseMatrix draws(n,NumberParameters(),false);
  for (int i=0; i < burn_in; i++) SimulateA0();
  for (int i=0; i < n; i++)
    {
      for (int j=0; j < thin; j++) SimulateA0();
      SimulateAplus();
      draws.InsertRowMatrix(i,0,parameters);
    } 
  return draws;
}

/*
   Computes the log of the integral of the tempered posterior.

         Integrate( (p(Y|Theta)*p(Theta))^(1/K) dTheta )
*/
double SBVAR_symmetric_linear::LogPosteriorIntegral(TDenseVector p, int ndraws, int thin, int burn_in)
{
  if (ndraws <= 0) throw dw_exception("PosteriorIntegral(): number of draws must be postive");
  if (thin <= 0) throw dw_exception("PosteriorIntegral(): thinning factor must be positive");
  if (p.dim != NumberParameters()) throw dw_exception("PosteriorIntegral(): Incorrect number of parameters");

  if (!simulation_info_set) SetSimulationInfo();

  SetParameters(p.vector);

  double integral=log_likelihood_constant + log_prior_constant;
  for (int i=0; i < n_vars; i++)
    integral+=0.5*dim_g[i]*1.837877066409345 + LogAbsDeterminant(Simulate_SqrtH[i]);  // 1.837877066409345 = log(2*pi)

  integral+=lambda_T*LogAbsDeterminant(A0);
  for (int i=0; i < n_vars; i++)
    {
      TDenseVector x=InverseMultiply(Simulate_SqrtS[i],p.SubVector(begin_b[i],begin_b[i]+dim_b[i]-1));
      integral+=-0.5*lambda_T*InnerProduct(x,x);
    }

  for (int i=0; i < n_vars; i++)
    integral-=LogConditionalA0_gibbs(p,i,ndraws,thin,burn_in);

  return integral;
}

/*
  Returns the properly scalled log density:

       log(p(b(i) | b(i+1), ... b(n), Y))

  This is computed via Gibbs simulation using Chib's method, since

        p(b(i) | b(1), ... b(i-1), b(i+1), ... b(n), Y)  

  is known.
*/
double SBVAR_symmetric_linear::LogConditionalA0_gibbs(const TDenseVector &p, int i, int ndraws, int thin, int burn_in)
{
  TDenseVector b=SubVector(p,begin_b[i],begin_b[i]+dim_b[i]-1);
  SetParameters(p.vector);

  double constant=0.5*(lambda_T+1)*log(0.5*lambda_T) + 0.5*(dim_b[i]-1)*log(lambda_T*0.159154943091895)
    - dw_log_gamma(0.5*(lambda_T+1)) - LogAbsDeterminant(Simulate_SqrtS[i]);   // 0.159154943091895 = 1/(2*pi)

  if (i == n_vars-1)
    return constant + LogConditionalA0_kernel(A0,b,i);
  else
    {
      for (int k=burn_in; k > 0; k--) SimulateA0(i);

      double sum=-1.0e300;
      for (int k=ndraws; k > 0; k--)
	{
	  for (int j=thin; j > 0; j--) SimulateA0(i);
	  sum=AddLogs(sum,LogConditionalA0_kernel(A0,b,i));
	}
      return constant + sum - log((double)ndraws);
    }
}

/*
   Returns
    
     log( p( b(i) | b(1), ... b(i-1), b(i+1), ... b(n), Y ) )

*/
double SBVAR_symmetric_linear::LogConditionalA0_kernel(const TDenseMatrix &A, const TDenseVector &b, int i)
{
  // Get orthornormal basis for R^dim_b[i] such that first element is
  // T'(k)*U'(k)*w / ||T'(k)*U'(k)*w||
  TDenseMatrix Q, R;
  QR(Q,R,ColumnMatrix(GetNullVector(A,i)*Simulate_USqrtS[i]),0);
  
  // get beta
  TDenseVector beta=Transpose(Q)*InverseMultiply(Simulate_SqrtS[i],b);
  
  // compute log kernel
  double kernel=beta[0]*beta[0];
  for (int k=dim_b[i]-1; k > 0; k--) kernel+=beta[k]*beta[k];

  return lambda_T*(log(fabs(beta[0])) - 0.5*kernel);
}

//===============================================================================
//===============================================================================
//===============================================================================

//===============================================================================
// BVAR_symmetric_linear_normalized Class
//===============================================================================
//
//
// Note BVAR_symmetric_linear_normalized and BVAR_symmetric_linear_normalized_generalization
// can only be applied to BVAR models that can be normalized. 
// For example, If A or Aplus contains any non-zero restricted element, then they cannot be
// normalized through changing the sign. 
// Currently we do not check whether this conditions satisfies or not. 
//
SBVAR_symmetric_linear_normalized::SBVAR_symmetric_linear_normalized(const SBVAR_symmetric_linear_normalized &model)
  : SBVAR_symmetric_linear(model)
{ }

// symmetric prior
SBVAR_symmetric_linear_normalized::SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric_linear(Data,PriorY,PriorX,iU,iV)
{ }

// Flat prior
SBVAR_symmetric_linear_normalized::SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric_linear(Data,iU,iV)
{ }

// Sims-Zha prior
SBVAR_symmetric_linear_normalized::SBVAR_symmetric_linear_normalized(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV)
  : SBVAR_symmetric_linear(Data,Hyperparameters,PeriodsPerYear,VarianceScale,iU,iV)
{ }

double SBVAR_symmetric_linear_normalized::LogPrior(void)
{
  for (int i=0; i < n_vars; i++)
    if (A0(i,i) < 0.0)
      return -1.0E300;
  return SBVAR_symmetric_linear::LogPrior() + 0.693147180559945*n_vars;    // 0.693147180559945 = log(2)
}

bool SBVAR_symmetric_linear_normalized::DrawParametersFromPrior(double *parameters)
{
  SBVAR_symmetric_linear::DrawParametersFromPrior(parameters);
  TDenseMatrix A0_normalized=GetA0(parameters), Aplus_normalized=GetAplus(parameters);
  for (int i=0; i < n_vars; i++)
    if (A0_normalized(i,i) < 0.0)
      {
	for (int j=0; j < n_vars; j++) A0_normalized(i,j)=-A0_normalized(i,j);
	for (int j=0; j < n_predetermined; j++) Aplus_normalized(i,j)=-Aplus_normalized(i,j);
      }
  SetParameters(A0_normalized,Aplus_normalized);
  GetParameters(parameters);
  return true;
}

//===============================================================================
// BVAR_symmetric_linear_normalized_generalization Class
//
//
//Note BVAR_symmetric_linear_normalized and BVAR_symmetric_linear_normalized_generalization
// can only be applied to BVAR models that can be normalized. 
// or example, If A or Aplus contains any non-zero restricted element, then they cannot be
// normalized through changing the sign. 
// We do not check whether this conditions satisfies or not. 
//
//===============================================================================
SBVAR_symmetric_linear_normalized_generalization::SBVAR_symmetric_linear_normalized_generalization(const SBVAR_symmetric_linear_normalized_generalization &model) :
SBVAR_symmetric_linear(model),
Ahat(model.Ahat)
{
	Ahat.CopyContent(model.Ahat); 
}

SBVAR_symmetric_linear_normalized_generalization::SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV) :
SBVAR_symmetric_linear(Data,iU,iV), 
Ahat(n_vars, n_vars, 0.0)
{
	SBVAR_symmetric_linear *temp = new SBVAR_symmetric_linear(Data,iU,iV);
        temp->MaximizePosterior();
        Ahat.CopyContent(temp->A0);
        delete temp;
}

SBVAR_symmetric_linear_normalized_generalization::SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const TDenseMatrix &PriorY, const TDenseMatrix &PriorX, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV) : 
SBVAR_symmetric_linear(Data,PriorY,PriorX,iU,iV),
Ahat(n_vars, n_vars, 0.0)
{
	SBVAR_symmetric_linear *temp = new SBVAR_symmetric_linear(Data,PriorY,PriorX,iU,iV);
        temp->MaximizePosterior();
        Ahat.CopyContent(temp->A0);
        delete temp;
}

SBVAR_symmetric_linear_normalized_generalization::SBVAR_symmetric_linear_normalized_generalization(TData_predetermined *Data, const TDenseVector &Hyperparameters, double PeriodsPerYear, double VarianceScale, const std::vector<TDenseMatrix> &iU, const std::vector<TDenseMatrix> &iV) :
SBVAR_symmetric_linear(Data,Hyperparameters,PeriodsPerYear,VarianceScale,iU,iV),
Ahat(n_vars, n_vars, 0.0)
{
	SBVAR_symmetric_linear *temp = new SBVAR_symmetric_linear(Data,Hyperparameters,PeriodsPerYear,VarianceScale,iU,iV);
        temp->MaximizePosterior();
        Ahat.CopyContent(temp->A0);
        delete temp;
}

double SBVAR_symmetric_linear_normalized_generalization::LogPrior()
{
	TDenseMatrix A0_Inverse=Inverse(A0); 
	TDenseVector e(n_vars,0.0); 
	for (int i=0; i < n_vars; i++)
	{
		e.Zeros(); 
		e(i) = 1.0; 
		if (InnerProduct(e, Ahat.RowVector(i), Transpose(A0_Inverse)) < 0)
			return MINUS_INFINITY; 
	}
  	return SBVAR_symmetric_linear::LogPrior() + 0.693147180559945*n_vars;    // 0.693147180559945 = log(2)
}

bool SBVAR_symmetric_linear_normalized_generalization::DrawParametersFromPrior(double *parameters)
{
  	SBVAR_symmetric_linear::DrawParametersFromPrior(parameters);
  	TDenseMatrix A0_normalized=GetA0(parameters), Aplus_normalized=GetAplus(parameters);
	TDenseMatrix A0_normalized_inverse = Inverse(A0_normalized); 
	TDenseVector e(n_vars,0.0); 
  	for (int i=0; i < n_vars; i++)
	{
		e.Zeros(); 
		e(i) = 1.0; 
		if (InnerProduct(e, Ahat.RowVector(i), Transpose(A0_normalized_inverse)) < 0)
		{
			A0_normalized.InsertRowMatrix(i,0,-1.0*A0_normalized.RowVector(i)); 
			Aplus_normalized.InsertRowMatrix(i,0,-1.0*Aplus_normalized.RowVector(i)); 
		}		
	}
  	SetParameters(A0_normalized,Aplus_normalized);
  	GetParameters(parameters);
  	return true;
}


//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// Auxiliary routines
//===============================================================================
/*
   Returns a non-zero vector v such that v is perpendicular to every row of X, 
   except for the ith row.  Use the LU decomposition of X.  The vector is NOT
   normalized to have length one.
*/
TDenseVector GetNullVector(const TDenseMatrix &X, int i)
{
  int n=X.rows;
  if ((i < 0) || (i >= n))
    throw dw_exception("GetNullVector(): row index out of range");
  if (n != X.cols)
    throw dw_exception("GetNullVector(): matrix must be square");
  TDenseVector v=Ones(n);
  if (n > 1)
    {
      int j, k, m;
      TDenseMatrix Y(X);
      Y.UniqueMemory();
      if (Y.column_major)
      	for (k=n*(n-1)+i; k >= 0; k-=n) Y.matrix[k]=0.0;
      else
      	for (m=i*n, k=m+n-1; k >= m; k--) Y.matrix[k]=0.0;
      TLapackLU LU(Y);
      double sum, diag, scale;
      for (m=n-2; m >= 0; m--)
      	{
	  diag=LU.LU[k=n*m+m];
	  if (diag == 0.0)
	    for (j=m+1; j < n; j++) v.vector[j]=0.0;
	  else
	    {
	      for (sum=LU.LU[k+=n]*v.vector[m+1], j=m+2; j < n; j++) sum+=LU.LU[k+=n]*v.vector[j];
	      if (fabs(diag) >= fabs(sum))
		v.vector[m]=-sum/diag;
	      else
		for (scale=-diag/sum, j=n-1; j > m; j--) v.vector[j]*=scale;
	    }
      	}
    }
  return v;
}

/*
   Extracts the number of lags from A0 and Aplus.  Throws exception if the dimensions
   are not consistant.
*/
int NumberLags(const TDenseMatrix &A0, const TDenseMatrix &Aplus, int n_exogenous)
{
  int n_vars=A0.cols, n_predetermined=Aplus.cols, n_lags=(n_predetermined - n_exogenous)/n_vars;
  if ((n_predetermined != n_vars*n_lags+n_exogenous) || (n_vars != A0.rows) || (n_vars != Aplus.rows))
    throw dw_exception("NumberLags(): arguments dimensions invald");
  return n_lags;
}

int NumberLags(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant)
{
  return NumberLags(A0,Aplus,IsConstant ? 1 : 0);
}

TDenseMatrix CompanionMatrix(const TDenseMatrix &B, int n_lags)
{
  if (n_lags == 0) throw dw_exception("CompanionMatrix(): companion matrix form requires positive number of lags");
  int n_vars=B.rows;
  TDenseMatrix C(n_vars*n_lags,n_vars*n_lags,0.0);
  C.Insert(0,0,B,0,n_vars-1,0,n_vars*n_lags-1);
  C.Insert(n_vars,0,Identity(n_vars*(n_lags-1)));
  return C;
}

TDenseMatrix CompanionMatrix(const TDenseMatrix &A0, const TDenseMatrix &Aplus, int n_lags)
{
  return CompanionMatrix(InverseMultiply(A0,Aplus),n_lags);
}

TDenseMatrix ReducedForm(const TDenseMatrix &A0, const TDenseMatrix &Aplus)
{
  try
    {
      return InverseMultiply(A0,Aplus);
    }
  catch (dw_exception &e)
    {
      throw dw_exception("Reduced Form(): A0 is singular");
    }
}

// TDenseMatrix ReducedForm(TDenseMatrix &B, TDenseMatrix &Sigma, const TDenseMatrix &A0, const TDenseMatrix &Aplus)
// {
//   try
//     {
//       B=InverseMultiply(A0,Aplus);
//       Sigma=Inverse(TransposeMultiply(A0,A0),SOLVE_CHOLESKY);
//     }
//   catch (dw_exception &e)
//     {
//       throw dw_exception("Reduced Form(): A0 is singular");
//     }
// }

TDenseMatrix ConditionalVariance(const TDenseMatrix &A0)
{
  try
    {
      return Inverse(TransposeMultiply(A0,A0));
    }
  catch (dw_exception &e)
    {
      throw dw_exception("Reduced Form(): A0 is singular");
    }
}

// Assumes that there is either no exogenous variables or there is a single
// exogenous variable that is constant and equal to one.  The unconditional
// mean is obtained from the reduced form companion matrix.
TDenseMatrix UnconditionalVariance(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant)
{
  int n_lags=NumberLags(A0,Aplus,IsConstant), n_vars=A0.cols;
  TDenseMatrix B=ReducedForm(A0,Aplus);
  if (n_lags == 0) return ConditionalVariance(A0);
  TDenseMatrix C=CompanionMatrix(B,n_lags), V=BlockDiagonalMatrix(A0,n_lags),
    X=V*(Identity(n_vars*n_lags) - C);
  try
    {
      return SubMatrix(Inverse(TransposeMultiply(X,X)),0,n_vars-1,0,n_vars-1);
    }
  catch (dw_exception &e)
    {
      throw dw_exception("UnconditionalMean(): Unconditional mean does not exist");
    }
}

// Assumes that there is either no exogenous variables or there is a single
// exogenous variable that is constant and equal to one.  The unconditional
// mean is obtained from the reduced form companion matrix.
TDenseVector UnconditionalMean(const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant)
{
  int n_lags=NumberLags(A0,Aplus,IsConstant), n_vars=A0.cols;
  if (!IsConstant) return TDenseVector(n_vars,0.0);
  TDenseMatrix B=ReducedForm(A0,Aplus);
  if (n_lags == 0) return ColumnVector(B,0);
  TDenseMatrix C=CompanionMatrix(B,n_lags);
  TDenseVector b(n_vars*n_lags,0.0);
  b.InsertColumnVector(0,B,n_vars*n_lags,0,n_vars-1);
  try
    {
      return SubVector(InverseMultiply(Identity(n_vars*n_lags) - C,b),0,n_vars-1);
    }
  catch (dw_exception &e)
    {
      throw dw_exception("UnconditionalMean(): Unconditional mean does not exist");
    }
}

/*
   Simulates artificial from using the given parameters.  It must be the case that there are
   one or zero exogenous variables and if there is one exogenous variable, then it is assumed
   that it is constant.
*/
TDenseMatrix SimulateData(int T, const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant, const TDenseVector &initial_value, int burn_in)
{
  int n_lags=NumberLags(A0,Aplus,IsConstant), n_vars=A0.cols, n_predetermined=Aplus.cols;

  if (initial_value.dim != n_predetermined)
    throw dw_exception("initial_value not correct length");

  if (IsConstant && (initial_value(n_predetermined-1) != 1.0))
    throw dw_exception("constant term in initial_value not equal to one");

  TDenseVector x=initial_value, y(n_vars), epsilon(n_vars);
  TDenseMatrix A0_inverse=Inverse(A0);

  for (int t=0; t < burn_in; t++)
    {
      y=A0_inverse*(Aplus*x + epsilon.RandomNormal());
      if (n_lags > 0)
  	{
  	  if (n_lags > 1)
  	    memmove(x.vector+n_vars,x.vector,n_vars*(n_lags-1)*sizeof(double));
  	  memcpy(x.vector,y.vector,n_vars*sizeof(double));
  	}
    }

  TDenseMatrix simulated_data(T,n_vars,false);
  for (int t=0; t < T; t++)
    {
      y=A0_inverse*(Aplus*x + epsilon.RandomNormal());
      simulated_data.InsertRowMatrix(t,0,y);
      if (n_lags > 0)
  	{
  	  if (n_lags > 1)
  	    memmove(x.vector+n_vars,x.vector,n_vars*(n_lags-1)*sizeof(double));
  	  memcpy(x.vector,y.vector,n_vars*sizeof(double));
  	}
    }

  return simulated_data;
}

TDenseMatrix SimulateData(int T, const TDenseMatrix &A0, const TDenseMatrix &Aplus, bool IsConstant, int burn_in)
{
  int n_vars=A0.cols, n_lags=NumberLags(A0,Aplus,IsConstant);
  TDenseVector initial_value(Aplus.cols), mean;
  try
    {
      mean=UnconditionalMean(A0,Aplus,IsConstant);
    }
  catch (dw_exception &e)
    {
      mean.Initialize(n_vars,0.0);
    }
  for (int i=0; i < n_lags; i++) initial_value.Insert(i*n_vars,mean);
  if (IsConstant) initial_value(initial_value.dim-1)=1.0;
  return SimulateData(T,A0,Aplus,IsConstant,initial_value,burn_in);
}

/*
   Assumes restrictions file is of the form:

     3

     X  0  0    
     X  X  0    
     X  X  X

     4
     1

     X  X  X     X  X  X     X  X  X     X  X  X     X
     X  X  X     X  X  X     X  X  X     X  X  X     X    
     X  X  X     X  X  X     X  X  X     X  X  X     X

   where
     first number = number of variables
     first matrix = contemporaneous exclusion restrictions, X is a free parameter and 0 is an exclusion restriction
     second number = number of lags
     third number = number or exogenous variables
     second matrix = predetermined exclusion restrictions, X is a free parameter and 0 is an exclusion restriction
*/
void SetupRestrictionMatrices(std::vector<TDenseMatrix> &U, std::vector<TDenseMatrix> &V, std::istream &input)
{
  // skip whitespace
  input >> std::skipws;

  // read number of variables
  int n_vars;
  input >> n_vars;
  if (!input.good()) throw dw_exception("SetupRestrictionMatrices(): error reading number of variables");

  // resize restriction arrays
  U.resize(n_vars);
  V.resize(n_vars);

  // read contemporaneous exclusion restrictions
  TDenseMatrix R(n_vars,n_vars);
  char ch;
  for (int i=0; i < n_vars; i++)
    for (int j=0; j < n_vars; j++)
      {
  	input >> ch;
	if (!input.good()) throw dw_exception("SetupRestrictionMatrices(): error reading predetermined exclusion restrictions");
  	if (ch == '0')
	  R(i,j)=0.0;
	else if ((ch == 'X') || (ch == 'x'))
	  R(i,j)=1.0;
	else
	  throw dw_exception("SetupRestrictionMatrices(): error reading contemporaneous exclusion restrictions");
      }

  // setup contemporaneous restrictions
  for (int i=0; i < n_vars; i++)
    {
      int k=0;
      for (int j=n_vars-1; j >= 0; j--) 
	if (R(i,j) > 0.0) k++;
      U[i].Zeros(n_vars,k);
      k=0;
      for (int j=0; j < n_vars; j++) 
	if (R(i,j) > 0.0)
	  U[i](j,k++)=1.0;
    }

  // read number of lags
  int n_lags;
  input >> n_lags;
  if (!input.good()) throw dw_exception("SetupRestrictionMatrices(): error reading number of lags");

  // read number of exogenous variables
  int n_exogenous;
  input >> n_exogenous;
  if (!input.good()) throw dw_exception("SetupRestrictionMatrices(): error reading number of exogenous variables");

  // read predetermined exclusion restrictions
  int n_predetermined=n_lags*n_vars+n_exogenous;
  R.Resize(n_vars,n_predetermined);
  for (int i=0; i < n_vars; i++)
    for (int j=0; j < n_predetermined; j++)
      {
  	input >> ch;
	if (!input.good()) throw dw_exception("SetupRestrictionMatrices(): error reading predetermined exclusion restrictions");
  	if (ch == '0')
	  R(i,j)=0.0;
	else if ((ch == 'X') || (ch == 'x'))
	  R(i,j)=1.0;
	else
	  throw dw_exception("SetupRestrictionMatrices(): error reading predetermined exclusion restrictions");
      }

  // setup predetermined restrictions
  for (int i=0; i < n_vars; i++)
    {
      int k=0;
      for (int j=n_predetermined-1; j >= 0; j--) 
	if (R(i,j) > 0.0) k++;
      V[i].Zeros(n_predetermined,k);
      k=0;
      for (int j=0; j < n_predetermined; j++) 
	if (R(i,j) > 0.0)
	  V[i](j,k++)=1.0;
    }
}

void SetupRestrictionMatrices(std::vector<TDenseMatrix> &U, std::vector<TDenseMatrix> &V, TDenseMatrix &A0, std::istream &input)
{
  SetupRestrictionMatrices(U,V,input);
  int n_vars=U[0].rows;
  A0.Resize(n_vars,n_vars);
  input >> A0;
}

void SetupRestrictionMatrices(std::vector<TDenseMatrix> &U, std::vector<TDenseMatrix> &V, TDenseMatrix &A0, TDenseMatrix &Aplus, std::istream &input)
{
  SetupRestrictionMatrices(U,V,input);
  int n_vars=U[0].rows;
  int n_predetermined=V[0].rows;
  A0.Resize(n_vars,n_vars);
  input >> A0;
  Aplus.Resize(n_vars,n_predetermined);
  input >> Aplus;
}
//===============================================================================
//===============================================================================
//===============================================================================

TDenseVector SBVAR::GetNormalizedParameter(const TDenseVector &parameter)
{
	if (parameter.dim != NumberParameters())
		return TDenseVector(0); 

	// A0: n_vars * n_vars, Aplus: n_vars * n_predetermined
	TDenseMatrix A0 = GetA0(parameter.vector), A0_normalized(n_vars, n_vars,0.0); 
	TDenseMatrix Aplus = GetAplus(parameter.vector), Aplus_normalized(n_vars, n_predetermined,0.0);

	for (int i_eqns=0; i_eqns<n_vars; i_eqns++)
	{
		if (A0(i_eqns, i_eqns) < 0)
		{
			A0_normalized.InsertRowMatrix(i_eqns, 0, -1.0*A0.RowVector(i_eqns)); 
			Aplus_normalized.InsertRowMatrix(i_eqns, 0, -1.0*Aplus.RowVector(i_eqns));
		}
		else 
		{
			A0_normalized.InsertRowMatrix(i_eqns, 0, A0.RowVector(i_eqns));
			Aplus_normalized.InsertRowMatrix(i_eqns, 0, Aplus.RowVector(i_eqns));
		}
	}
	SetParameters(A0_normalized, Aplus_normalized); 
	TDenseVector new_parameter(NumberParameters(),0.0);
	GetParameters(new_parameter.vector);
	return new_parameter; 
}
