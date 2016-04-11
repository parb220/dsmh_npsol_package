
#include <fstream>
#include <math.h>

#include "prcsn.h"
#include "dw_rand.h"
#include "dw_ascii.hpp"
#include "regime_processes.hpp"

//===============================================================================
// class TRegimeProcessArray
//===============================================================================
TRegimeProcessArray& TRegimeProcessArray::operator=(const TRegimeProcessArray &processes)
{
  for (int i=this->size()-1; i >= 0; i--) 
    {
      delete this->operator[](i);
      this->operator[](i)=(TRegimeProcess*)NULL;
    }
  resize(processes.size());
  for (int i=this->size()-1; i >= 0; i--)
    this->operator[](i)=processes[i] ? processes[i]->Clone() : (TRegimeProcess*)NULL; 
  return *this;
}

TRegimeProcessArray& TRegimeProcessArray::operator=(const std::vector<TRegimeProcess*> &processes)
{
  for (int i=this->size()-1; i >= 0; i--) 
    {
      delete this->operator[](i);
      this->operator[](i)=(TRegimeProcess*)NULL;
    }
  resize(processes.size());
  for (int i=this->size()-1; i >= 0; i--)
    this->operator[](i)=processes[i] ? processes[i]->Clone() : (TRegimeProcess*)NULL; 
  return *this;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess
//===============================================================================
std::string RegimeProcessName(int id)
{
  switch(id)
    {
    case ID_RP_DETERMINISTIC: return std::string("deterministic");
    case ID_RP_TRIVIAL: return std::string("trivial");
    case ID_RP_MARKOV: return std::string("markov");
    case ID_RP_INDEPENDENT: return std::string("independent");
    case ID_RP_INDEPENDENT_INVARIANT: return std::string("independent/invariant");
    case ID_RP_INDEPENDENT_MARKOV: return std::string("independent/markov");
    default: return std::string("unknown");
    }
}

TRegimeProcess::TRegimeProcess(const TRegimeProcess &Process) 
  : id(Process.id), n_regimes(Process.n_regimes), n_parameters(Process.n_parameters), initial_probabilities(Process.initial_probabilities),
    parameter_invariant_initial_probabilities(Process.parameter_invariant_initial_probabilities)
{
}

TRegimeProcess::TRegimeProcess(int Id, int NumberRegimes, int NumberParameters, const TDenseVector &InitialProbabilities)
  : id(Id), n_regimes(NumberRegimes), n_parameters(NumberParameters), initial_probabilities(InitialProbabilities),
    parameter_invariant_initial_probabilities(true)
{ 
  Check();
}

TRegimeProcess::TRegimeProcess(int Id, int NumberRegimes, int NumberParameters)
  : id(Id), n_regimes(NumberRegimes), n_parameters(NumberParameters), initial_probabilities(0),
    parameter_invariant_initial_probabilities(false)
{
  Check(); 
}

void TRegimeProcess::Check(void)
{
  if (n_regimes <= 0) 
    throw dw_exception("TRegimeProcess(): number of regimes must be positive");
  if (n_parameters < 0) 
    throw dw_exception("TRegimeProcess(): number of parameters must be non-negative");
  if (parameter_invariant_initial_probabilities)
    {
      if (n_regimes != initial_probabilities.dim)
	throw dw_exception("TRegimeProcess(): number of initial probabilities must be equal to the number of regimes");
      double sum=0.0, max=1.0+n_regimes*MACHINE_EPSILON;
      for (int i=n_regimes-1; i >= 0; i--)
	if ((initial_probabilities(i) < 0) || ((sum+=initial_probabilities(i)) > max))
	  throw dw_exception("TRegimeProcess(): invalid initial probabilities");
    }	    
}

TRegimeProcess* TRegimeProcess::Clone(void) 
{ 
  throw dw_exception("Cannot create an instance of the virtual class TRegimeProcess");
  return (TRegimeProcess*)NULL;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess_invariant
//===============================================================================
TRegimeProcess_invariant* TRegimeProcess_invariant::Clone(void) 
{ 
  throw dw_exception("Cannot create an instance of the virtual class TRegimeProcess_invariant");
  return (TRegimeProcess_invariant*)NULL;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
//=== TRegimeProcess_deterministic class
//===============================================================================
TRegimeProcess_deterministic::TRegimeProcess_deterministic(int NumberRegimes, const std::vector<int> &Regimes)
  : TRegimeProcess(ID_RP_DETERMINISTIC,NumberRegimes,0), regimes(Regimes)
{
  for (int t=regimes.size()-1; t >= 0; t--)
    if ((regimes[t] < 0) || (regimes[t] >= NumberRegimes))
      throw dw_exception("TRegimeProcess_deterministic(): invalid vector of regimes");
  parameter_invariant_initial_probabilities=true;
  initial_probabilities=Zeros(n_regimes);
  if (Regimes.size() == 0)
    initial_probabilities(0)=1.0;
  else
    initial_probabilities(regimes[0])=1.0;
}

TDenseMatrix TRegimeProcess_deterministic::TransitionMatrix(int t)
{
  TDenseMatrix transition_matrix(n_regimes,n_regimes,0.0);
  int row;
  if (regimes.size() == 0)
    row=0;
  else if (t >= (int)regimes.size()-1)
    row=regimes[regimes.size()-1];
  else if (t < 0)
    row=regimes[0];
  else
    row=regimes[t+1];
  transition_matrix.InsertRowMatrix(row,0,TDenseVector(n_regimes,1.0));
  return transition_matrix;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess_markov
//===============================================================================
TRegimeProcess_markov::TRegimeProcess_markov(const TRegimeProcess_markov &Process)
  : TRegimeProcess_invariant(Process),
    DirichletDim(Process.DirichletDim),
    NonZeroIndex(Process.NonZeroIndex),
    MQ(Process.MQ),
    total_dirichlet_parameters(Process.total_dirichlet_parameters),
    transition_matrix_computed(Process.transition_matrix_computed),
    transition_matrix(Process.transition_matrix),
    initial_probabilities_computed(Process.initial_probabilities_computed),
    log_prior_computed(Process.log_prior_computed),
    log_prior(Process.log_prior),
    log_prior_constant(Process.log_prior_constant),
    scale(Process.scale)
{
  AllocateMemory();

  //=== Set Prior_B and B
  memcpy(Prior_B,Process.Prior_B,total_dirichlet_parameters*sizeof(double));
  memcpy(B,Process.B,total_dirichlet_parameters*sizeof(double));
}

// constant initial probabilites
TRegimeProcess_markov::TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
					     const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers, const TDenseVector &InitialProbabilities)
  : TRegimeProcess_invariant(ID_RP_MARKOV,NumberRegimes,DirichletPrior.dim-DirichletDimensions.size(),InitialProbabilities), DirichletDim(DirichletDimensions), NonZeroIndex(DirichletIndices), 
    MQ(DirichletMultipliers), total_dirichlet_parameters(DirichletPrior.dim), scale(1.0)
{
  Setup(DirichletPrior);
}

// constant and equal initial probabilites
TRegimeProcess_markov::TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
					     const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers)
  : TRegimeProcess_invariant(ID_RP_MARKOV,NumberRegimes,DirichletPrior.dim-DirichletDimensions.size(),DefaultInitialProbabilities(NumberRegimes)), DirichletDim(DirichletDimensions), 
    NonZeroIndex(DirichletIndices), MQ(DirichletMultipliers), total_dirichlet_parameters(DirichletPrior.dim), scale(1.0)
{
  Setup(DirichletPrior);
}

// ergodic initial probabilites
TRegimeProcess_markov::TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
					     const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers, double Scale)
  : TRegimeProcess_invariant(ID_RP_MARKOV,NumberRegimes,DirichletPrior.dim-DirichletDimensions.size()), DirichletDim(DirichletDimensions), NonZeroIndex(DirichletIndices), 
    MQ(DirichletMultipliers), total_dirichlet_parameters(DirichletPrior.dim), scale(Scale)
{
  Setup(DirichletPrior);
}

TRegimeProcess_markov::~TRegimeProcess_markov()
{
  delete[] Prior_B;
  delete[] Prior_b;
  delete[] B;
  delete[] b;
}

void TRegimeProcess_markov::AllocateMemory(void)
{
  // allocate memory
  if (Prior_B=new(std::nothrow) double[total_dirichlet_parameters])
    {
      if (Prior_b=new(std::nothrow) double*[DirichletDim.size()])
	{
	  if (B=new(std::nothrow) double[total_dirichlet_parameters])
	    {
	      if (b=new(std::nothrow) double*[DirichletDim.size()])
		{
		  // set b and Prior_b
		  for (unsigned int q=0, k=0; k < DirichletDim.size(); k++)
		    {
		      b[k]=B + q;
		      Prior_b[k]=Prior_B + q;
		      q+=DirichletDim[k];
		    }
		  return;
		}
	      delete[] B;
	    }
	  delete[] Prior_b;
	}
      delete[] Prior_B;
    }
  throw std::bad_alloc();
}

void TRegimeProcess_markov::Setup(const TDenseVector &DirichletPrior)
{
  //=== Check sizes 
  if ((n_regimes <= 0) || (DirichletPrior.dim != total_dirichlet_parameters) || (MQ.rows != n_regimes) || (MQ.cols != n_regimes) || ((int)NonZeroIndex.size() != n_regimes))
    throw dw_exception("TRegimeProcess_markov(): invalid dimensions");

  //=== Check dimension of NonZeroIndex[i]
  for (int i=n_regimes-1; i >= 0; i--)
    if ((int)NonZeroIndex[i].size() != n_regimes) throw dw_exception("TRegimeProcess_markov(): invalid dimension of Dirichlet indices");

  //=== Check DirichletDim[i] > 0
  //=== Check DircheletDim[0] + ... + DirichletDim[DirichletDim.size()-1] = total_dirichlet_parameters
  int total=0;
  for (int i=DirichletDim.size()-1; i >= 0; i--)
    if (DirichletDim[i] <= 0)
      throw dw_exception("TRegimeProcess_markov(): Dirichlet dimensions must be positive");
    else
      total+=DirichletDim[i];
  if (total != total_dirichlet_parameters) throw dw_exception("TRegimeProcess_markov(): incorrect Dirichlet dimensions sum");

  //=== Check -1 <= NonZeroIndex[i][j] < total_dirichlet_parameters.
  //=== Check NonZeroIndex[i][j] >= 0, ==> MQ[i][j] > 0.
  //=== Check NonZeroIndex[i][j] = -1, ==> MQ[i][j] = 0.
  for (int j=0; j < n_regimes; j++)
    for (int i=0; i < n_regimes; i++)
      if ((NonZeroIndex[i][j] < -1) || (total_dirichlet_parameters <= NonZeroIndex[i][j]))
	throw dw_exception("TRegimeProcess_markov(): Invalid Dirichlet indices");
      else
        if (NonZeroIndex[i][j] >= 0)
          {
            if (MQ(i,j) <= 0.0) throw dw_exception("TRegimeProcess_markov(): Dirichlet multiplier not positive");
          }
        else
          {
            if (MQ(i,j) != 0.0) throw dw_exception("TRegimeProcess_markov(): Dirichlet multiplier not zero");
          }

  //=== Check that column sums are correct 
  TDenseVector v(total_dirichlet_parameters); // total_dirichlet_parameters
  for (int j=n_regimes-1; j >= 0; j--)
    {
      v.Zeros();
      for (int i=n_regimes-1; i >= 0; i--)
	if (NonZeroIndex[i][j] >= 0)
	  v(NonZeroIndex[i][j])+=MQ(i,j);

      int offset=0;
      double sum=0.0;
      for (unsigned int i=0; i < DirichletDim.size(); offset+=DirichletDim[i++])
	{
	  sum+=v(offset);
	  for (int k=DirichletDim[i]-1; k > 0; k--)
	    if (fabs(v(offset) - v(offset+k)) > SQRT_MACHINE_EPSILON) 
	      throw dw_exception("TRegimeProcess_markov(): Invalid Dirichlet multiplier sums (code 1)");
	}

      if (fabs(sum - 1.0) > SQRT_MACHINE_EPSILON)
	throw dw_exception("TRegimeProcess_markov(): Invalid Dirichlet multiplier sums (code 2)");
    }

  //=== Check that prior elements are positive
  for (int i=total_dirichlet_parameters-1; i >= 0; i--)
    if (DirichletPrior(i) <= 0) throw dw_exception("TRegimeProcess_markov(): Dirichlet prior elements must be positive");

  //=== Check that scale is positive
  if (scale <= 0)
    throw dw_exception("TRegimeProcess_markov(): scale must be positive");

  //=== Allocate memory
  AllocateMemory();

  //=== Set Prior_B
  memcpy(Prior_B,DirichletPrior.vector,total_dirichlet_parameters*sizeof(double));

  //=== Set log prior constant
  log_prior_constant=0.0;
  for (int i=DirichletDim.size()-1; i >= 0; i--)
    {
      double sum=0.0;
      for (int j=DirichletDim[i]-1; j >= 0; j--)
	{
	  sum+=Prior_b[i][j];
	  log_prior_constant-=dw_log_gamma(Prior_b[i][j]);
	}
      log_prior_constant+=dw_log_gamma(sum);
    }

  //=== Set flags
  transition_matrix_computed=false;
  log_prior_computed=false;
  initial_probabilities_computed=parameter_invariant_initial_probabilities;
}

/*
  The free parameters map into the Dirichlet probabilities by

  p(i) = (1.0 - p(0) - ... - p(i-1))/(1 + exp(-f(i)))

  for 0 <= i < n_parameters-1 and

  p(n_parameters-1) = 1.0 - p(0) - ... - p(n_parameters-1)
    
*/
bool TRegimeProcess_markov::SetParameters(double *parameters)
{
  for (int n=0, i=0; i < (int)DirichletDim.size(); i++)
    {
      double p=1.0, q;
      int k=DirichletDim[i]-1;
      for (int j=0; j < k; j++)
	if (parameters[n] >= 0.0)
	  p-=(b[i][j]=p/(1.0 + exp(-parameters[n++])));     // p becomes numerically zero at about 36.7368006
	else
	  {
	    q=exp(parameters[n++]);
	    p-=(b[i][j]=p*q/(q+1));                         // b[i][j] becomes numerically zero at about -745.1332192
	  }
      b[i][k]=p;	
    }
  transition_matrix_computed=log_prior_computed=false;
  initial_probabilities_computed=parameter_invariant_initial_probabilities;
  return true;
}

bool TRegimeProcess_markov::GetParameters(double *parameters) const
{
  for (int n=0, i=0; i < (int)DirichletDim.size(); i++)
    {
      double p=1.0, q;
      int k=DirichletDim[i]-1;
      for (int j=0; j < k; j++)
	{
	  parameters[n++]=(b[i][j] > 0.0) ? (((q=p/b[i][j] - 1.0) > 0.0) ? -log(q) : 40.0) : -750.0;    // see SetupParameters() for these bounds
	  p-=b[i][j];
	}
    }
  return true;
}

TDenseMatrix TRegimeProcess_markov::TransitionMatrix(void)
{
  if (!transition_matrix_computed)
    {
      transition_matrix.UniqueMemory(n_regimes,n_regimes,true);
      for (int j=n_regimes-1; j >= 0; j--)
	for (int i=n_regimes-1; i >= 0; i--)
	  transition_matrix(i,j)=(NonZeroIndex[i][j] >= 0) ? MQ(i,j)*B[NonZeroIndex[i][j]] : 0.0;
      transition_matrix_computed=true;
    }
  return transition_matrix;
}

TDenseVector TRegimeProcess_markov::InitialProbabilities(void)
{
  if (!initial_probabilities_computed)
    {
      throw dw_exception("egodic initial probabilites not yet implemented");

      initial_probabilities_computed=true;
    }
  return initial_probabilities;
}

double TRegimeProcess_markov::LogPrior(void)
{
  if (!log_prior_computed)
    {
      log_prior_computed=true;
      log_prior=log_prior_constant;

      // Accumulate Dirichlet kernel
      for (int k=total_dirichlet_parameters-1; k >= 0; k--)
	if (B[k] > 0)
	  log_prior+=Prior_B[k]*log(B[k]);
	else
	  return log_prior=MINUS_INFINITY;
    }
  return log_prior;
}

void TRegimeProcess_markov::PriorMean(void)
{
  memcpy(B,Prior_B,total_dirichlet_parameters*sizeof(double));
  for (int i=DirichletDim.size()-1; i >= 0; i--)
    {
      double sum=0.0;
      for (int j=DirichletDim[i]-1; j >= 0; j--) sum+=b[i][j];
      sum=1.0/sum;
      for (int j=DirichletDim[i]-1; j >= 0; j--) b[i][j]*=sum;
    }
  transition_matrix_computed=log_prior_computed=false;
  initial_probabilities_computed=parameter_invariant_initial_probabilities;
}

// draw from the Dirichlet distribuion
void TRegimeProcess_markov::Draw(void)
{   
  double scale;
  int i;
  for (int k=DirichletDim.size()-1; k >= 0; k--)
    {
      for (scale=0.0, i=DirichletDim[k]-1; i >= 0; i--) scale+=(b[k][i]=dw_gamma_rnd(b[k][i]));
      for (scale=1.0/scale, i=DirichletDim[k]-1; i >= 0; i--) b[k][i]*=scale;
    }
}

void TRegimeProcess_markov::SimulatePrior(void)
{
  memcpy(B,Prior_B,total_dirichlet_parameters*sizeof(double));
  Draw();
  transition_matrix_computed=log_prior_computed=false;
  initial_probabilities_computed=parameter_invariant_initial_probabilities;
}

void TRegimeProcess_markov::SimulateConditionalPath(const std::vector<int> &path)
{
  if (parameter_invariant_initial_probabilities)
    {
      memcpy(B,Prior_B,total_dirichlet_parameters*sizeof(double));
      int k, t=path.size()-1;
      for ( ; t > 0; t--)
	if ((k=NonZeroIndex[path[t]][path[t-1]]) >= 0) B[k]+=1.0;
      Draw();
      transition_matrix_computed=log_prior_computed=false;
      initial_probabilities_computed=parameter_invariant_initial_probabilities;
    }
  else
    {
      // make a independent Metropolis-Hasting draw when the initial probabilities are the egodic distribution
      throw dw_exception("Draw(): egodic initial probabilities not yet implemented");
    }
}

//-------------------------------------------------------------------------------
//--- Auxiliary routines
//-------------------------------------------------------------------------------
int FindIdentifier(const StringMatrix &M, const std::string &id)
{
  for (unsigned int i=0; i < M.size(); i++)
    if ((M[i][0] == "//==") && (M[i][1] == id)) return i;
  return -1;
}

int ParseMatrix(TDenseMatrix &m, const StringMatrix &M, int pos, int n_rows, int n_cols=-1)
{
  if ((n_rows < 0) || (pos < 0) || (pos+n_rows > (int)M.size())) return 0;
  if (n_cols < 0) n_cols=M[pos].size();
  for (int i=n_rows-1; i >= 0; i--) if ((int)M[pos+i].size() != n_cols) return 0;
  m.Resize(n_rows,n_cols);
  try
    {
      for (int i=n_rows-1; i >= 0; i--)
	for (int j=n_cols-1; j >= 0; j--)
	  m(i,j)=_stod(M[pos+i][j]);
    }
  catch (std::exception)
    {
      return 0;
    }
  return 1;
}

int ParseVector(TDenseVector &v, const StringMatrix &M, int pos, int n_cols=-1)
{
  if ((pos < 0) || (pos >= (int)M.size())) return 0;
  if (n_cols < 0) 
    n_cols=M[pos].size();
  else
    if (n_cols != (int)M[pos].size()) return 0;
  v.Resize(n_cols);
  try
    {
      for (int j=n_cols-1; j >= 0; j--) v(j)=_stod(M[pos][j]);
    }
  catch (std::exception)
    {
      return 0;
    }
  return 1;
}

int ParseInteger(int &k, const StringMatrix &M, int pos)
{
  if ((pos < 0) || (pos >= (int)M.size()) || ((int)M[pos].size() != 1)) return 0;
  try
    {
      k=_stoi(M[pos][0]);
    }
  catch (std::exception)
    {
      return 0;
    }
  return 1;
}

// The lines between pos and pos+n_rows must all have the same length.  Furthermore, if
// n_cols >= 0, then the lengths must all equal n_cols.  Each element of M must be an
// "x", "X", or convertable to a double.  Upon a sucessful return, the elements of R
// will be one if the corresponding element of M was "x" or "X" and zero otherwise. The
// elements of C will equal to zero if the corresponding element of M was "x" or "X"
// and the element of M converted to a double otherwise.
int ParseRestrictionMatrix(std::vector< std::vector<int> > &R, TDenseMatrix &C, const StringMatrix &M, int pos, int n_rows, int n_cols=-1)
{
  if ((n_rows < 0) || (pos < 0) || (pos+n_rows > (int)M.size())) return 0;
  if (n_cols < 0) n_cols=M[pos].size();
  for (int i=n_rows-1; i >= 0; i--) if ((int)M[pos+i].size() != n_cols) return 0;
  C.Resize(n_rows,n_cols);
  R.resize(n_rows);
  try
    {
      for (int i=n_rows-1; i >= 0; i--)
	{
	  R[i].resize(n_cols);
	  for (int j=n_cols-1; j >= 0; j--)
	    if ((M[pos+i][j] == "x") || (M[pos+i][j] == "X"))
	      {
		R[i][j]=1;
		C(i,j)=0.0;
	      }
	    else
	      {
		R[i][j]=0;
		C(i,j)=_stod(M[pos+i][j]);
	      }
	}
    }
  catch (std::exception)
    {
      return 0;
    }
  return 1;
}

// Each element of M must be of the form double or double*x(integer,integer) or x(integer,integer).
int ParseGeneralRestrictionMatrix(std::vector< std::vector<int> > &I1, std::vector< std::vector<int> > &I2, TDenseMatrix &C, const StringMatrix &M, int pos, int n_rows, int n_cols=-1)
{
  if ((n_rows < 0) || (pos < 0) || (pos+n_rows > (int)M.size())) return 0;
  if (n_cols < 0) n_cols=M[pos].size();
  for (int i=n_rows-1; i >= 0; i--) if ((int)M[pos+i].size() != n_cols) return 0;
  C.Resize(n_rows,n_cols);
  I1.resize(n_rows);
  I2.resize(n_rows);
  try
    {
      for (int i=n_rows-1; i >= 0; i--)
	{
	  I1[i].resize(n_cols);
	  I2[i].resize(n_cols);
	  for (int j=n_cols-1; j >= 0; j--)
	    {
	      std::string::size_type length=M[pos+i][j].length(), k=0, n;
	      if (length == 0) return 0;

	      if ((M[pos+i][j][0] != 'x') && (M[pos+i][j][0] != 'X'))
		C(i,j)=_stod(M[pos+i][j],&k);
	      else 
		C(i,j)=1.0;

	      if (k < length)
		{
		  if ((k > 0) && (M[pos+i][j][k++] != '*')) return 0;

		  if ((k >= length) || (tolower(M[pos+i][j][k++]) != 'x')) return 0;
		  if ((k >= length) || (M[pos+i][j][k++] != '(')) return 0;
		  if ((I1[i][j]=_stoi(M[pos+i][j].substr(k),&n)) < 0) return 0;

		  if (((k+=n) >= length) || (M[pos+i][j][k++] != ',')) return 0;
		  if ((I2[i][j]=_stoi(M[pos+i][j].substr(k),&n)) < 0) return 0;

		  if (((k+=n) >= length) || (M[pos+i][j][k++] != ')')) return 0;
		  if (k < length) return 0;
		}
	      else
		I1[i][j]=I2[i][j]=-1;
	    }
	}
    }
  catch (std::exception)
    {
      return 0;
    }
  return 1;
}

TRegimeProcessArray SetupMarkovArrayRegimeProcess(const char *filename)
{
  std::ifstream input(filename,std::ifstream::in);
  if (!input.is_open()) throw dw_exception("SetupMarkovArrayRegimeProcess(): unable to open input file");
  StringMatrix M(input,"//**");
  input.close();

  int pos, NumberProcesses;
  if (((pos=FindIdentifier(M,"NumberProcesses")) < 0) || !ParseInteger(NumberProcesses,M,pos+1) || (NumberProcesses <= 0))
    throw dw_exception("SetupMarkovArrayRegimeProcess(): error parsing //== NumberProcesses");

TRegimeProcessArray Processes(NumberProcesses);
  for (int i=0; i < NumberProcesses; i++) Processes[i]=SetupMarkovRegimeProcess(i,M);

  return Processes;
}

TRegimeProcess_markov* SetupMarkovRegimeProcess(int id_number, const StringMatrix &M)
{
  std::string id;
  int pos, NumberRegimes;
  std::vector<int> DirichletDimensions;
  std::vector< std::vector<int> > DirichletIndices;
  TDenseMatrix DirichletMultipliers;
  TDenseVector DirichletPrior;

  // get number of regimes
  if (((pos=FindIdentifier(M,id="NumberRegimes[" + cluster_to_string(id_number) + "]")) < 0) || !ParseInteger(NumberRegimes,M,pos+1) || (NumberRegimes <= 0))
    throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);

  // get restrictions matrix
  if ((pos=FindIdentifier(M,id="SimpleRestrictions[" + cluster_to_string(id_number) + "]")) >= 0)
    {
      if (!ParseRestrictionMatrix(DirichletIndices,DirichletMultipliers,M,pos+1,NumberRegimes,NumberRegimes))
	throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);

      DirichletDimensions.resize(NumberRegimes);
      for (int k=0, j=0; j < NumberRegimes; j++)
	{
	  DirichletDimensions[j]=0;
	  for (int i=0; i < NumberRegimes; i++)
	    if (DirichletIndices[i][j])
	      {
		DirichletDimensions[j]++;
		DirichletIndices[i][j]=k++;
		DirichletMultipliers(i,j)=1.0;
	      }
	    else
	      {
		DirichletIndices[i][j]=-1;
		if (DirichletMultipliers(i,j) != 0.0) throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);
	      }
	}
    }
  else if ((pos=FindIdentifier(M,id="GeneralRestrictions[" + cluster_to_string(id_number) + "]")) >= 0)
    {
      std::vector< std::vector<int> > I1, I2;
      if (!ParseGeneralRestrictionMatrix(I1,I2,DirichletMultipliers,M,pos+1,NumberRegimes,NumberRegimes))
	throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);

      // counts number of Dirichlet processes
      int NumberDirichlet=0;
      for (int i=NumberRegimes-1; i >= 0; i--)
	for (int j=NumberRegimes-1; j >= 0; j--)
	  if (I1[i][j] >= NumberDirichlet) NumberDirichlet=I1[i][j]+1; 

      // gets the size of each Dirichlet processes
      DirichletDimensions.assign(NumberDirichlet,0);
      for (int i=NumberRegimes-1; i >= 0; i--)
	for (int j=NumberRegimes-1; j >= 0; j--)
	  if (I2[i][j] >= DirichletDimensions[I1[i][j]]) DirichletDimensions[I1[i][j]]=I2[i][j]+1;

      // handles fixed and positive probabilities if necessary
      int k=NumberDirichlet;
      for (int i=NumberRegimes-1; i >= 0; i--)
	for (int j=NumberRegimes-1; j >= 0; j--)
	  if ((I1[i][j] < 0) && (DirichletMultipliers(i,j) > 0))
	    {
	      if (k == NumberDirichlet)
		{
		  DirichletDimensions.push_back(1);
		  NumberDirichlet++;
		}
	      I1[i][j]=k;
	      I2[i][j]=0;
	    }


      int offsets[NumberDirichlet];
      offsets[0]=0;
      for (int i=1; i < NumberDirichlet; i++) offsets[i]=offsets[i-1]+DirichletDimensions[i-1];
 
      DirichletIndices.resize(NumberRegimes);
      for (int i=NumberRegimes-1; i >= 0; i--) DirichletIndices[i].assign(NumberRegimes,-1);

      for (int i=NumberRegimes-1; i >= 0; i--)
	for (int j=NumberRegimes-1; j >= 0; j--)
	  if (I1[i][j] >= 0)
	    DirichletIndices[i][j]=offsets[I1[i][j]]+I2[i][j];

      // debugging code
      // std::cout << "r matrix =\n";
      // for (int i=0; i < NumberRegimes; i++)
      // 	{
      // 	  for (int j=0; j < NumberRegimes; j++)
      // 	    std::cout << r[i][j] << "  ";
      // 	  std::cout << std::endl;
      // 	}
      // std::cout << "s matrix =\n";
      // for (int i=0; i < NumberRegimes; i++)
      // 	{
      // 	  for (int j=0; j < NumberRegimes; j++)
      // 	    std::cout << s[i][j] << "  ";
      // 	  std::cout << std::endl;
      // 	}
      // std::cout << "offsets =\n";
      // for (int i=0; i < NumberDirichlet; i++) std::cout << offsets[i] << " ";
      // std::cout << std::endl;
    }
  else 
    throw dw_exception("SetupMarkovRegimeProcess(): //== " + id + " not found");

  int total_dirichlet_dimensions=0;
  for (int i=DirichletDimensions.size()-1; i >= 0; i--) total_dirichlet_dimensions+=DirichletDimensions[i];

  // get prior
  if ((pos=FindIdentifier(M,id="DurationPrior[" + cluster_to_string(id_number) + "]")) >= 0)
    {
      TDenseVector durations;
      if (!ParseVector(durations,M,pos+1,NumberRegimes))
	throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 1)");

      DirichletPrior.Initialize(-1.0,total_dirichlet_dimensions);

      for (int i=0; i < NumberRegimes; i++)
	if (durations[i] <= 0)
	  throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 2)");
	else
	  if (DirichletIndices[i][i] < 0)
	    throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 3)");
	  else
	    if (DirichletPrior(DirichletIndices[i][i]) > 0)
	      throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 4)");
	    else
	      DirichletPrior(DirichletIndices[i][i])=durations(i);

      for (int k=0, i=0; i < (int)DirichletDimensions.size(); i++)
	for (int non_zero=0, j=0; j < DirichletDimensions[i]; k++, j++)
	  if (DirichletPrior(k) < 0)
	    DirichletPrior(k)=1.0;
	  else
	    if (++non_zero == 1)
	      if (DirichletDimensions[i] > 1)
		DirichletPrior(k)*=DirichletDimensions[i] - 1.0;
	      else
		throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 5)");
	    else
	      throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id + " (code 6)");
    }
  else if ((pos=FindIdentifier(M,id="MatrixPrior[" + cluster_to_string(id_number) + "]")) >= 0)
    {
      TDenseMatrix PriorMatrix;
      if (!ParseMatrix(PriorMatrix,M,pos+1,NumberRegimes,NumberRegimes))
	throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);

	// prior conditional on the restrictions
      DirichletPrior.Ones(total_dirichlet_dimensions);
      for (int j=NumberRegimes-1; j >= 0; j--)
	for (int i=NumberRegimes-1; i >= 0; i--)
	  if ((DirichletIndices[i][j]) >= 0)
	    DirichletPrior(DirichletIndices[i][j])+=PriorMatrix(i,j)-1;
    }
  else if ((pos=FindIdentifier(M,id="GeneralDirichletPrior[" + cluster_to_string(id_number) + "]")) >= 0)
    {
      if (!ParseVector(DirichletPrior,M,pos+1,total_dirichlet_dimensions))
	throw dw_exception("SetupMarkovRegimeProcess(): error parsing //== " + id);
    }
  else 
    throw dw_exception("SetupMarkovRegimeProcess(): //== " + id + " not found");

  // debugging code
  // std::cout << "number regimes = " << NumberRegimes << std::endl;
  // std::cout << "Dirichlet dimensions =\n";
  // for (int i=0; i < (int)DirichletDimensions.size(); i++)
  //   std::cout << DirichletDimensions[i] << " ";
  // std::cout << "\nDirichlet indices =\n";
  // for (int r=0; r < NumberRegimes; r++)
  //   {
  //     for (int c=0; c < NumberRegimes; c++) 
  // 	std::cout << DirichletIndices[r][c] << " ";
  //     std::cout << std::endl;
  //   }
  // std::cout << "Dirichlet multipliers =\n" << DirichletMultipliers;
  // std::cout << "Dirichlet prior =\n" << DirichletPrior << std::endl;
  // char ch;
  // std::cout << "enter a character to continue\n"; std::cin >> ch; 

  return new TRegimeProcess_markov(NumberRegimes,DirichletPrior,DirichletDimensions,DirichletIndices,DirichletMultipliers);
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess_independent
//===============================================================================
static int NumberRegimes_independent(const std::vector<TRegimeProcess*> &Processes)
{
  int n_regimes=1;
  for (int i=Processes.size()-1; i >= 0; i--) n_regimes*=Processes[i]->NumberRegimes();
  return n_regimes;
}

static int NumberParameters_independent(const std::vector<TRegimeProcess*> &Processes)
{
  int n_parameters=0;
  for (int i=Processes.size()-1; i >= 0; i--) n_parameters+=Processes[i]->NumberParameters();
  return n_parameters;
}

TRegimeProcess_independent::TRegimeProcess_independent(const TRegimeProcess_independent &Process)
  : TRegimeProcess(Process), 
    offsets(Process.offsets), 
    dims(Process.dims), 
    processes(Process.processes),
    time_invariant_transition_matrix(Process.time_invariant_transition_matrix),
    transition_matrix(Process.transition_matrix), 
    transition_matrix_computed(Process.transition_matrix_computed),
    initial_probabilities_computed(Process.initial_probabilities_computed),
    log_prior(Process.log_prior), 
    log_prior_computed(Process.log_prior_computed),
    translation_table(Process.translation_table)
{ }

TRegimeProcess_independent::TRegimeProcess_independent(const std::vector<TRegimeProcess*> &Processes)
  : TRegimeProcess(ID_RP_INDEPENDENT,NumberRegimes_independent(Processes),NumberParameters_independent(Processes)), 
    offsets(Processes.size()), dims(Processes.size()), processes(Processes)
{
  int n_processes=Processes.size();
  if (n_processes > 0)
    {
      // build offsets and dims
      if (!processes[0]) throw dw_exception("TRegimeProcess_independent(): null regime process not allowed");
      parameter_invariant_initial_probabilities=processes[0]->ParameterInvariantInitialProbabilities();
      offsets[0]=0;
      dims[0]=processes[0]->NumberParameters();
      for (int i=1; i < n_processes; i++)
	{
	  if (!processes[i]) throw dw_exception("TRegimeProcess_independent(): null regime process not allowed");
	  if (!processes[i]->ParameterInvariantInitialProbabilities()) parameter_invariant_initial_probabilities=false;
	  offsets[i]=offsets[i-1]+dims[i-1];
	  dims[i]=processes[i]->NumberParameters();
	}

      transition_matrix_computed=initial_probabilities_computed=log_prior_computed=false;

      // build translation table
      translation_table.resize(n_regimes);
      translation_table[0].assign(n_processes,0);
      for (int k=1; k < n_regimes; k++)
	{
	  translation_table[k].resize(n_processes);
	  memcpy(translation_table[k].data(),translation_table[k-1].data(),n_processes*sizeof(int));
	  for (int i=n_processes-1; i >= 0; i--)
	    if (++translation_table[k][i] >= processes[i]->NumberRegimes())
	      translation_table[k][i]=0;
	    else
	      break;
	} 

      // invariant transition matrices and initial probabilities
      time_invariant_transition_matrix=parameter_invariant_initial_probabilities=true;
      for (int i=n_processes-1; i >= 0; i--)
	{
	  switch (processes[i]->Id())
	    {
	    case ID_RP_DETERMINISTIC: time_invariant_transition_matrix=false; break;
	    case ID_RP_INDEPENDENT:
	      if (static_cast<TRegimeProcess_independent*>(processes[i])->time_invariant_transition_matrix)
		time_invariant_transition_matrix=false;
	      break;
	    }
	  if (!processes[i]->ParameterInvariantInitialProbabilities()) parameter_invariant_initial_probabilities=false;
	}
    }
  else
    throw dw_exception("TRegimeProcess_independent(): number of processes must be positive");
}

bool TRegimeProcess_independent::SetParameters(double *parameters)
{
  bool rtrn=true;
  for (int k=processes.size()-1; k >= 0; k--) 
    if (!processes[k]->SetParameters(parameters+offsets[k])) rtrn=false;

  if (!parameter_invariant_initial_probabilities) initial_probabilities_computed=false;
  transition_matrix_computed=log_prior_computed=false;

  return rtrn;
}

bool TRegimeProcess_independent::GetParameters(double *parameters) const
{
  bool rtrn=true;
  for (int k=processes.size()-1; k >= 0; k--)
    if (!processes[k]->GetParameters(parameters+offsets[k])) rtrn=false;
  return rtrn;
}

void TRegimeProcess_independent::DefaultParameters(void)
{
  for (int k=processes.size()-1; k >= 0; k--) processes[k]->DefaultParameters();

  if (!parameter_invariant_initial_probabilities) initial_probabilities_computed=false;
  transition_matrix_computed=log_prior_computed=false;
}

TDenseMatrix TRegimeProcess_independent::TransitionMatrix(int t)
{
  if (!transition_matrix_computed)
    {
      transition_matrix=processes[0]->TransitionMatrix(t);
      for (int k=1; k < (int)processes.size(); k++)
	transition_matrix=Kron(transition_matrix,processes[k]->TransitionMatrix(t));
      if (time_invariant_transition_matrix) transition_matrix_computed=true;
    }
  return transition_matrix;
}

TDenseMatrix TRegimeProcess_independent::BaseTransitionMatrix(int t, int idx)
{
  if ((idx < 0) || (idx >= (int)processes.size()))
    throw dw_exception("TRegimeProcess_independent::TransitionMatrix(): process index out of range");
  return processes[idx]->TransitionMatrix(t);
}

TDenseVector TRegimeProcess_independent::InitialProbabilities(void)
{
  if (!initial_probabilities_computed)
    {
      initial_probabilities=processes[0]->InitialProbabilities();
      for (int k=1; k < (int)processes.size(); k++)
	initial_probabilities=Kron(initial_probabilities,processes[k]->InitialProbabilities());
      initial_probabilities_computed=true;
    }
  return initial_probabilities;
}

TDenseVector TRegimeProcess_independent::BaseInitialProbabilities(int idx)
{
  if ((idx < 0) || (idx >= (int)processes.size()))
    throw dw_exception("TRegimeProcess_independent::InitialProbabilities(): process index out of range");
  return processes[idx]->InitialProbabilities();
}

double TRegimeProcess_independent::LogPrior(void)
{
  if (!log_prior_computed)
    {
      log_prior=0.0;
      for (int k=processes.size()-1; k >= 0; k--) log_prior+=processes[k]->LogPrior();
      log_prior_computed=true;
    }
  return log_prior;
}

void TRegimeProcess_independent::PriorMean(void)
{
  for (int k=processes.size()-1; k >= 0; k--) 
	processes[k]->PriorMean();
}

void TRegimeProcess_independent::SimulatePrior(void)
{
  for (int k=processes.size()-1; k >= 0; k--) 
	processes[k]->SimulatePrior();
}

double TRegimeProcess_independent::BaseLogPrior(int idx)
{
  if ((idx < 0) || (idx >= (int)processes.size()))
    throw dw_exception("TRegimeProcess_independent::LogPrior(): process index out of range");
  return processes[idx]->LogPrior();
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess_independent_invariant
//===============================================================================
TRegimeProcess_independent_invariant::TRegimeProcess_independent_invariant(const std::vector<TRegimeProcess*> &Processes)
    : TRegimeProcess_independent(Processes)
{
  id=ID_RP_INDEPENDENT_INVARIANT;
  for (int i=processes.size()-1; i >= 0; i--)
    if (!(processes[i]->Id() & ID_RP_INVARIANT_TYPE))
      throw dw_exception("TRegimeProcess_independent_invariant(): base regime not time invariant");
}

TDenseMatrix TRegimeProcess_independent_invariant::TransitionMatrix(void)
{
  if (!transition_matrix_computed)
    {
      switch (processes[0]->Id())
	{
	case ID_RP_TRIVIAL:
	case ID_RP_MARKOV:
	  transition_matrix=static_cast<TRegimeProcess_invariant*>(processes[0])->TransitionMatrix();
	  break;
	case ID_RP_INDEPENDENT_INVARIANT:
	case ID_RP_INDEPENDENT_MARKOV:
	  transition_matrix=static_cast<TRegimeProcess_independent_invariant*>(processes[0])->TransitionMatrix();
	  break;
	default: throw dw_exception("TransitionMatrix(): base regime not time invariant");
	}  
      for (int k=1; k < (int)processes.size(); k++)
	switch (processes[k]->Id())
	  {
	  case ID_RP_TRIVIAL:
	  case ID_RP_MARKOV:
	    transition_matrix=Kron(transition_matrix,static_cast<TRegimeProcess_invariant*>(processes[k])->TransitionMatrix());
	    break;
	  case ID_RP_INDEPENDENT_INVARIANT:
	  case ID_RP_INDEPENDENT_MARKOV:
	    transition_matrix=Kron(transition_matrix,static_cast<TRegimeProcess_independent_invariant*>(processes[k])->TransitionMatrix());
	    break;
	  default: throw dw_exception("TransitionMatrix(): base regime not time invariant");
	  }
      transition_matrix_computed=true;
    }
  return transition_matrix;
}

TDenseMatrix TRegimeProcess_independent_invariant::BaseTransitionMatrix(int idx)
{
  if ((idx < 0) || (idx >= (int)processes.size()))
    throw dw_exception("TRegimeProcess_independent::TransitionMatrix(): process index out of range");
  switch (processes[idx]->Id())
    {
    case ID_RP_TRIVIAL:
    case ID_RP_MARKOV:
      return static_cast<TRegimeProcess_invariant*>(processes[idx])->TransitionMatrix();
      break;
    case ID_RP_INDEPENDENT_INVARIANT:
    case ID_RP_INDEPENDENT_MARKOV:
      return static_cast<TRegimeProcess_independent_invariant*>(processes[idx])->TransitionMatrix();
      break;
    default: throw dw_exception("BaseTransitionMatrix(): base regime not time invariant");
    }
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
// class TRegimeProcess_independent_markov
//===============================================================================
TRegimeProcess_independent_markov::TRegimeProcess_independent_markov(const std::vector<TRegimeProcess*> &Processes)
    : TRegimeProcess_independent_invariant(Processes)
{
  id=ID_RP_INDEPENDENT_MARKOV;
  for (int i=processes.size()-1; i >= 0; i--)
    if (!(processes[i]->Id() & ID_RP_MARKOV_TYPE))
      throw dw_exception("TRegimeProcess_independent_invariant(): base regime not Markov");
}

void TRegimeProcess_independent_markov::PriorMean(void)
{
  for (int i=processes.size()-1; i >= 0; i--)
    switch (processes[i]->Id())
      {
      case ID_RP_TRIVIAL:
	static_cast<TRegimeProcess_trivial*>(processes[i])->PriorMean();
	break;
      case ID_RP_MARKOV:
	static_cast<TRegimeProcess_markov*>(processes[i])->PriorMean();
	break;
      case ID_RP_INDEPENDENT_MARKOV:
	static_cast<TRegimeProcess_independent_markov*>(processes[i])->PriorMean();
	break;
      default: throw dw_exception("PriorMean(): base regime not Markov");
      }

  if (!parameter_invariant_initial_probabilities) initial_probabilities_computed=false;
  transition_matrix_computed=log_prior_computed=false;
}


void TRegimeProcess_independent_markov::SimulatePrior(void)
{
  for (int i=processes.size()-1; i >= 0; i--)
    switch (processes[i]->Id())
      {
      case ID_RP_TRIVIAL:
	static_cast<TRegimeProcess_trivial*>(processes[i])->SimulatePrior();
	break;
      case ID_RP_MARKOV:
	static_cast<TRegimeProcess_markov*>(processes[i])->SimulatePrior();
	break;
      case ID_RP_INDEPENDENT_MARKOV:
	static_cast<TRegimeProcess_independent_markov*>(processes[i])->SimulatePrior();
	break;
      default: throw dw_exception("SimulatePrior(): base regime not Markov");
      }

  if (!parameter_invariant_initial_probabilities) initial_probabilities_computed=false;
  transition_matrix_computed=log_prior_computed=false;
}

void TRegimeProcess_independent_markov::SimulateConditionalPath(const std::vector<int> &path)
{
  std::vector<int> restricted_path(path.size());
  for (int i=processes.size()-1; i >= 0; i--)
    {
      for (int j=path.size()-1; j >= 0; j--) restricted_path[j]=translation_table[path[j]][i];

      switch (processes[i]->Id())
	{
	case ID_RP_TRIVIAL:
	  static_cast<TRegimeProcess_trivial*>(processes[i])->SimulateConditionalPath(restricted_path);
	  break;
	case ID_RP_MARKOV:
	  static_cast<TRegimeProcess_markov*>(processes[i])->SimulateConditionalPath(restricted_path);
	  break;
	case ID_RP_INDEPENDENT_MARKOV:
	  static_cast<TRegimeProcess_independent_markov*>(processes[i])->SimulateConditionalPath(restricted_path);
	  break;
	default: throw dw_exception("SimulateCondtionalPath(): base regime not Markov");
	}
    }

  if (!parameter_invariant_initial_probabilities) initial_probabilities_computed=false;
  transition_matrix_computed=log_prior_computed=false;
}
//===============================================================================
//===============================================================================
//===============================================================================


