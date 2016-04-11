#ifndef _REGIME_PROCESS_HEADER_
#define _REGIME_PROCESS_HEADER_
 
#include <vector>
#include <string>

#include "dw_dense_matrix.hpp"
#include "dw_ascii.hpp"
#include "prcsn.h"

// id defines
#define ID_RP_TYPE                      0x00000000
#define ID_RP_INVARIANT_TYPE            0x00000001
#define ID_RP_MARKOV_TYPE               0x00000002
#define ID_RP_INDEPENDENT_TYPE          0x00000004
#define ID_RP_DETERMINISTIC             0x00100000
#define ID_RP_TRIVIAL                   0x00100003
#define ID_RP_MARKOV                    0x00200003
#define ID_RP_INDEPENDENT               0x00100004
#define ID_RP_INDEPENDENT_INVARIANT     0x00100005
#define ID_RP_INDEPENDENT_MARKOV        0x00100007

std::string RegimeProcessName(int id);
inline TDenseVector DefaultInitialProbabilities(int NumberRegimes) { return (NumberRegimes > 0) ? TDenseVector(NumberRegimes,1.0/(double)NumberRegimes) : TDenseVector(0); };

std::string _to_string(int i); 
//===============================================================================
//=== TRegimeProcess class
//===============================================================================
class TRegimeProcess
{
protected:
  // basic info
  int id;
  int n_regimes;
  int n_parameters;

  // initial probabilities
  TDenseVector initial_probabilities;
  bool parameter_invariant_initial_probabilities;

  void Check(void);

public:
  // constructors
  // TRegimeProcess(void) : id(0), n_regimes(1), n_parameters(0), initial_probabilities(0), parameter_invariant_initial_probabilities(false) { Check(); };
  TRegimeProcess(const TRegimeProcess &Process);
  TRegimeProcess(int Id, int NumberRegimes, int NumberParameters, const TDenseVector &InitialProbabilities);
  TRegimeProcess(int Id, int NumberRegimes, int NumberParameters);

  // destructors
  virtual ~TRegimeProcess() { };

  // cloning an object
  virtual TRegimeProcess* Clone(void);

  // setup from vector of parameters

  virtual bool SetParameters(double *parameters) = 0;

  // return parameter values
  virtual bool GetParameters(double *parameters) const = 0;

  // default parameter values
  virtual void DefaultParameters(void) { PriorMean(); };

  // transistion matrices
  virtual TDenseMatrix TransitionMatrix(int t) = 0;
  TDenseMatrix TransitionMatrix(int t, double* parameters) { SetParameters(parameters); return TransitionMatrix(t); };
  
 // initial probabilites
  virtual TDenseVector InitialProbabilities(void) { return initial_probabilities; };
  TDenseVector InitialProbabilities(double *parameters) { SetParameters(parameters); return InitialProbabilities(); };

  // prior
  virtual double LogPrior() = 0;
  double LogPrior(double *parameters) 
  { 
	if (!SetParameters(parameters))
		return MINUS_INFINITY; 
	else 
		return LogPrior(); 
  };
  virtual void PriorMean(void) = 0;
  virtual void SimulatePrior(void) = 0;

  // info
  int Id(void) const { return id; };
  std::string Name(void) const { return RegimeProcessName(id); };
  int NumberRegimes(void) const { return n_regimes; };
  int NumberParameters(void) const { return n_parameters; };
  bool ParameterInvariantInitialProbabilities(void) const { return parameter_invariant_initial_probabilities; }
};


//===============================================================================
//=== TRegimeProcessArray class
//===============================================================================
class TRegimeProcessArray : public std::vector<TRegimeProcess*>
{
public:
  TRegimeProcessArray(const TRegimeProcessArray &processes) : std::vector<TRegimeProcess*>(processes.size()) { for (int i=this->size()-1; i >= 0; i--) this->operator[](i)=processes[i] ? processes[i]->Clone() : (TRegimeProcess*)NULL; };
  TRegimeProcessArray(const std::vector<TRegimeProcess*> &processes) : std::vector<TRegimeProcess*>(processes.size()) { for (int i=this->size()-1; i >= 0; i--) this->operator[](i)=processes[i] ? processes[i]->Clone() : (TRegimeProcess*)NULL; };
  explicit TRegimeProcessArray(int i=0) : std::vector<TRegimeProcess*>(i) { };
  ~TRegimeProcessArray() { for (int i=this->size()-1; i >= 0; i--) delete this->operator[](i); };
  TRegimeProcessArray& operator=(const TRegimeProcessArray &processes);
  TRegimeProcessArray& operator=(const std::vector<TRegimeProcess*> &processes);
};


//===============================================================================
//=== TRegimeProcess_invariant class
//===============================================================================
class TRegimeProcess_invariant : public TRegimeProcess
{
public:
  // constructors
  // TRegimeProcess_invariant() : TRegimeProcess() {}; 
  TRegimeProcess_invariant(const TRegimeProcess &Process) : TRegimeProcess(Process) { };
  TRegimeProcess_invariant(int Id, int NumberRegimes, int NumberParameters, const TDenseVector &InitialProbabilities) : TRegimeProcess(Id,NumberRegimes,NumberParameters,InitialProbabilities) { };
  TRegimeProcess_invariant(int Id, int NumberRegimes, int NumberParameters) : TRegimeProcess(Id,NumberRegimes,NumberParameters) { };

  // destructors
  virtual ~TRegimeProcess_invariant() { };

  // cloning an object
  virtual TRegimeProcess_invariant* Clone(void);

  // transistion matrices
  using TRegimeProcess::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(void) = 0;
  TDenseMatrix TransitionMatrix(double* parameters) { SetParameters(parameters); return TransitionMatrix(); };
  virtual TDenseMatrix TransitionMatrix(int t) { return TransitionMatrix(); };
};


//===============================================================================
//=== TRegimeProcess_trivial class
//===============================================================================
class TRegimeProcess_trivial : public TRegimeProcess_invariant
{
public:
  // constructors
  TRegimeProcess_trivial(const TRegimeProcess &Process) : TRegimeProcess_invariant(Process) { };
  TRegimeProcess_trivial(void) : TRegimeProcess_invariant(ID_RP_TRIVIAL,1,0) { };

  // destructors
  virtual ~TRegimeProcess_trivial() { };

  // cloning an object
  virtual TRegimeProcess_trivial* Clone(void) { return new TRegimeProcess_trivial(*this); };

  // setup from vector of parameters
  virtual bool SetParameters(double *parameters) { return true; };

  // return parameter values
  virtual bool GetParameters(double *parameters) const { return true; };

  // transistion matrices
  using TRegimeProcess_invariant::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(void) { return TDenseMatrix(1,1,1.0); };

  // prior
  using TRegimeProcess_invariant::LogPrior;
  virtual double LogPrior() { return 0; };
  virtual void PriorMean(void) { };
  virtual void SimulatePrior(void) { };

  // markov specific routines 
  void SimulateConditionalPath(const std::vector<int> &path) { };
};

//===============================================================================
//=== TRegimeProcess_deterministic class
//===============================================================================
class TRegimeProcess_deterministic : public TRegimeProcess
{
protected:
  std::vector<int> regimes;

public:
  // constructors
  TRegimeProcess_deterministic(const TRegimeProcess_deterministic &Process) : TRegimeProcess(Process), regimes(Process.regimes) { };
  TRegimeProcess_deterministic(int NumberRegimes, const std::vector<int> &Regimes);

  // destructors
  virtual ~TRegimeProcess_deterministic() { };

  // cloning an object
  virtual TRegimeProcess_deterministic* Clone(void) { return new TRegimeProcess_deterministic(*this); };

  // setup from vector of parameters
  virtual bool SetParameters(double *parameters) { return true; };

  // return parameter values
  virtual bool GetParameters(double *parameters) const { return true; };

  // default parameter values
  virtual void DefaultParameters(void) { };

  // prior
  using TRegimeProcess::LogPrior;
  virtual double LogPrior() { return 0; };
  virtual void PriorMean(void) { };
  virtual void SimulatePrior(void) { };
  
 // transistion matrices
  using TRegimeProcess::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(int t);

};


//===============================================================================
//=== TRegimeProcess_markov class
//===============================================================================
class TRegimeProcess_markov : public TRegimeProcess_invariant
{
private:
  //=== Restrictions ===
  std::vector<int> DirichletDim;                   // DirichletDim[k] = number Dirichlet parameters for the kth Dirichlet random variable
  std::vector< std::vector<int> > NonZeroIndex;    // n_regimes x n_regimes : Q(i,j) = (NonZeroIndex[i][j] >= 0) ? B[NonZeroIndex[i][j]]*MQ(i,j) : 0.0; 
  TDenseMatrix MQ;                                 // n_regimes x n_regimes : Q(i,j) = (NonZeroIndex[i][j] >= 0) ? B[NonZeroIndex[i][j]]*MQ(i,j) : 0.0;

  //=== Number quasi-free variables ===
  int total_dirichlet_parameters;                  // total_dirichlet_parameters = n_parameters + DirichletDim.size()

  //=== Transition matrix ===
  bool transition_matrix_computed;
  TDenseMatrix transition_matrix;

  //=== initial probabiliites ===
  bool initial_probabilities_computed;

  //=== Prior information ===
  double *Prior_B;                                 // Workspace for prior on the Dirichlet parameters.  Dimension of Prior_B is total_dirichlet_parameters.
  double **Prior_b;                                // Dirichlet prior parameters.  Points to the buffer Prior_B.  Dimension of Prior_b is the number of Dirichlet random variables.
  bool log_prior_computed;
  double log_prior;                                

  //=== Workspace for Dirichlet parameters ===
  double *B;                                       // Workspace for the dirichlet parameters. Dimension of B is total_dirichlet_parameters.
  double **b;                                      // Dirichlet parameters.  Points to the buffer B.  Dimension of b is the number of Dirichlet random variables.

  //=== Constants ===
  double log_prior_constant;
  double scale;

  void Setup(const TDenseVector &DirichletPrior);
  void AllocateMemory(void);
  void Draw(void);

public:
  // constructors
  // TRegimeProcess_markov() : TRegimeProcess_invariant(), DirichletDim(std::vector<int>(0)), NonZeroIndex(std::vector<std::vector<int> >(0)), MQ(TDenseMatrix(0,0)), 
  // 			    total_dirichlet_parameters(0), transition_matrix_computed(false), transition_matrix(TDenseMatrix(0,0)), initial_probabilities_computed(false), 
  // 			    Prior_B(NULL), Prior_b(NULL), log_prior_computed(false), log_prior(0.0), B(NULL), b(NULL), log_prior_constant(0.0), scale(0.0) {};
  TRegimeProcess_markov(const TRegimeProcess_markov &Processes);
  TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
			const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers, const TDenseVector &InitialProbabilities);
  TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
			const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers);
  TRegimeProcess_markov(int NumberRegimes, const TDenseVector &DirichletPrior, const std::vector<int> &DirichletDimensions, 
			const std::vector< std::vector<int> > &DirichletIndices, const TDenseMatrix &DirichletMultipliers, double Scale);

  // destructors
  virtual ~TRegimeProcess_markov();

  // cloning an object
  virtual TRegimeProcess_markov* Clone(void) { return new TRegimeProcess_markov(*this); };

  // setup from vector of parameters
  virtual bool SetParameters(double *parameters);

  // return parameter values
  virtual bool GetParameters(double *parameters) const;

  // default parameter values
  virtual void DefaultParameters(void) { PriorMean(); };

  // transistion matrices
  using TRegimeProcess_invariant::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(void);

  // prior
  using TRegimeProcess_invariant::LogPrior;
  virtual double LogPrior(void); 
  virtual void PriorMean(void);
  virtual void SimulatePrior(void);
  
  // initial probabilities
  using TRegimeProcess_invariant::InitialProbabilities;
  virtual TDenseVector InitialProbabilities(void);
 
  // markov specific routines 
  void SimulateConditionalPath(const std::vector<int> &path);
};

TRegimeProcessArray SetupMarkovArrayRegimeProcess(const char *filename);
TRegimeProcess_markov* SetupMarkovRegimeProcess(int id_number, const StringMatrix &M);


//===============================================================================
//=== TRegimeProcess_independent class
//===============================================================================
class TRegimeProcess_independent : public TRegimeProcess
{
public:
  std::vector<int> offsets;
  std::vector<int> dims;
  TRegimeProcessArray processes;

  bool time_invariant_transition_matrix;
  TDenseMatrix transition_matrix;
  bool transition_matrix_computed;

  bool initial_probabilities_computed;

  double log_prior;
  bool log_prior_computed;

  std::vector< std::vector<int> > translation_table;       // translation_table[k][i] is the value of the ith regime process when the overall regime is k

public:
  // constructors
  // TRegimeProcess_independent() : TRegimeProcess(), offsets(std::vector<int>(0)), dims(std::vector<int>(0)), processes(), time_invariant_transition_matrix(false), 
  // 				 transition_matrix(TDenseMatrix(0,0)), transition_matrix_computed(false), parameter_invariant_initial_probabilities(false), 
  // 				 initial_probabilities_computed(false), log_prior(0.0), log_prior_computed(false), translation_table(std::vector<std::vector<int> >(0)) {};
  TRegimeProcess_independent(const TRegimeProcess_independent &Process);
  TRegimeProcess_independent(const std::vector<TRegimeProcess*> &Processes);

  // destructors
  virtual ~TRegimeProcess_independent() { };

  // cloning an object
  virtual TRegimeProcess_independent* Clone(void) { return new TRegimeProcess_independent(*this); };

  // setup from vector of parameters
  virtual bool SetParameters(double *parameters);

  // return parameter values
  virtual bool GetParameters(double *parameters) const;

  // default parameter values
  virtual void DefaultParameters(void);

  // transistion matrices
  using TRegimeProcess::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(int t);

  // initial probabilities
  using TRegimeProcess::InitialProbabilities;
  virtual TDenseVector InitialProbabilities(void);

  // prior
  using TRegimeProcess::LogPrior;
  virtual double LogPrior(void);
  virtual void PriorMean(void);
  virtual void SimulatePrior(void);
  double BaseLogPrior(int idx);

  // info
  int NumberProcesses(void) { return processes.size(); };

  // base transition matices
  TDenseMatrix BaseTransitionMatrix(int t, int idx);
  TDenseMatrix BaseTransitionMatrix(int t, int idx, double *parameters) { SetParameters(parameters); return BaseTransitionMatrix(t,idx); };

  // base initial probabilities
  TDenseVector BaseInitialProbabilities(int idx);
  TDenseVector BaseInitialProbabilities(int idx, double *parameters) { SetParameters(parameters); return BaseInitialProbabilities(idx); };

  // base priors
  double BaseLogPrior(int idx, double *parameters) { SetParameters(parameters); return BaseLogPrior(idx); };

  // base info
  int BaseId(int idx) const;
  std::string BaseName(int idx) const;
  int BaseNumberRegimes(int idx) const;
  int BaseNumberParameters(int idx) const;
  bool BaseParameterInvariantInitialProbabilities(int idx) const;
  int BaseRegime(int regime, int idx) { return translation_table[regime][idx]; };
};

//===============================================================================
//=== TRegimeProcess_independent_invariant class
//===============================================================================
class TRegimeProcess_independent_invariant : public TRegimeProcess_independent
{
public:
  // constructors
  // TRegimeProcess_independent_invariant() : TRegimeProcess_independent() {}; 
  TRegimeProcess_independent_invariant(const TRegimeProcess_independent_invariant &Process) : TRegimeProcess_independent(Process) { };
  TRegimeProcess_independent_invariant(const std::vector<TRegimeProcess*> &Processes);

  // destructors
  virtual ~TRegimeProcess_independent_invariant() { };

  // cloning an object
  virtual TRegimeProcess_independent_invariant* Clone(void) { return new TRegimeProcess_independent_invariant(*this); };

  // transistion matrices
  using TRegimeProcess_independent::TransitionMatrix;
  virtual TDenseMatrix TransitionMatrix(void);

  // base transition matrices
  using TRegimeProcess_independent::BaseTransitionMatrix;
  TDenseMatrix BaseTransitionMatrix(int idx);
  TDenseMatrix BaseTransitionMatrix(int idx, double *parameters) { SetParameters(parameters); return BaseTransitionMatrix(idx); };
};

//===============================================================================
//=== TRegimeProcess_independent_markov class
//===============================================================================
class TRegimeProcess_independent_markov : public TRegimeProcess_independent_invariant
{
public:
  // constructors
  // TRegimeProcess_independent_markov() : TRegimeProcess_independent_invariant() {}; 
  TRegimeProcess_independent_markov(const TRegimeProcess_independent_markov &Process) : TRegimeProcess_independent_invariant(Process) { };
  TRegimeProcess_independent_markov(const std::vector<TRegimeProcess*> &Processes);

  // destructors
  virtual ~TRegimeProcess_independent_markov() { };

  // cloning an object
  virtual TRegimeProcess_independent_markov* Clone(void) { return new TRegimeProcess_independent_markov(*this); };

  // markov specific routines 
  void PriorMean(void);
  void SimulatePrior(void);
  void SimulateConditionalPath(const std::vector<int> &path);
};

#endif
