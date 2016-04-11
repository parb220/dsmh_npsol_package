#ifndef _DW_TIME_SERIES_HEADER_
#define _DW_TIME_SERIES_HEADER_

#include <vector>
#include <string>

#include "dw_dense_matrix.hpp"
#include "dw_data.hpp"
#include "regime_processes.hpp"
#include "prcsn.h"

class TTimeSeries
{
protected:
  TData *data;

  virtual double LogLikelihood(void) = 0;
  virtual double LogPrior(void) = 0;
  virtual double LogPosterior(void)
  { 
    double log_prior= LogPrior(); 
    if (log_prior <= MINUS_INFINITY) return MINUS_INFINITY; 
    double log_likelihood = LogLikelihood(); 
    if (log_likelihood <= MINUS_INFINITY) return MINUS_INFINITY; 
    return log_prior+log_likelihood;
  };

public:
  // constructors
  TTimeSeries(): data(NULL) {}; 
  TTimeSeries(const TTimeSeries &TimeSeries) : data(TimeSeries.data->Clone()) { };

  TTimeSeries(TData *Data) : data(Data->Clone()) { };

  // destructors
  virtual ~TTimeSeries() { delete data; };

  // cloning an object
  virtual TTimeSeries* Clone(void) { throw dw_exception("Cannot create an instance of the virtual class TTimeSeries"); return (TTimeSeries*)NULL; };

  // set current parameter values
  virtual bool SetParameters(double *parameters) = 0;

  // return current parameter values
  virtual bool GetParameters(double *parameters) const= 0;

  // default parameter values
  virtual void DefaultParameters(void) = 0;

  // draw parameters from the prior distribution
  virtual bool DrawParametersFromPrior(double *parameters) = 0; 

  // log likelihood 
  double LogLikelihood(double *parameters) 
  { 
    if (!SetParameters(parameters)) 
      return MINUS_INFINITY; 
    else 
      return LogLikelihood(); 
  };
  double LogLikelihood(const TDenseVector &parameter) { return LogLikelihood(parameter.vector); }; 

  // log prior
  double LogPrior(double *parameters) 
  { 
    if (!SetParameters(parameters))
      return MINUS_INFINITY; 
    else 
      return LogPrior(); 
  };
  double LogPrior(const TDenseVector &parameter) { return LogPrior(parameter.vector); }; 

  // log posterior
  double LogPosterior(double *parameters) 
  { 
    if (!SetParameters(parameters)) 
      return MINUS_INFINITY; 
    else 
      return LogPosterior(); 
  };
  double LogPosterior(const TDenseVector &parameter) { return LogPosterior(parameter.vector); }; 

  // number of free parameters
  virtual int NumberParameters(void) const= 0;

  // data info
  TData* pData(void) const { return data->Clone(); };
  TDenseMatrix Data(void) const { return data->Data(); };
  TDenseVector Data(int t) const { return data->Data(t); };
  int NumberVariables(void) const { return data->NumberVariables(); };
  int NumberObservations(void) const { return data->NumberObservations(); };
  TDenseVector Dates(void) const { return data->Dates(); };
  std::string VariableName(int i) const { return data->VariableName(i); };
};

class TTimeSeries_TV : public TTimeSeries
{
protected:
  int offset;
  TRegimeProcess *regime_process;

public:
  // constructors
  TTimeSeries_TV() : TTimeSeries(), offset(0), regime_process(NULL) {}; 
  TTimeSeries_TV(const TTimeSeries_TV &model) 
    : TTimeSeries(model), offset(model.offset), regime_process(model.regime_process->Clone()) { };

  TTimeSeries_TV(TData* Data, TRegimeProcess* RegimeProcess, int RegimeParametersOffset) 
    : TTimeSeries(Data), offset(RegimeParametersOffset), regime_process(RegimeProcess->Clone()) { };

  // destructors
  virtual ~TTimeSeries_TV() { delete regime_process; };

  // transition matrices
  TDenseMatrix TransitionMatrix(int t) { return regime_process->TransitionMatrix(t); };
  TDenseMatrix TransitionMatrix(int t, double* parameters) { return regime_process->TransitionMatrix(t,parameters+offset); };

  // initial probabilities
  TDenseVector InitialProbabilities(void) { return regime_process->InitialProbabilities(); };
  TDenseVector InitialProbabilities(double *parameters) { return regime_process->InitialProbabilities(parameters+offset); };

  // regime process info
  TRegimeProcess* pRegimeProcess(void) const { return regime_process->Clone(); };
  int NumberRegimes(void) const { return regime_process->NumberRegimes(); };
  virtual int NumberParameters(void) const { return regime_process->NumberParameters(); };
  int Offset_RegimeProcessParameters() const {return offset; };
};


// Imposing additional restrictions on the class T.  The class T must be derived 
// from the class TTimeSeries. 
template<class T> class TTimeSeriesRestricted : public T
{
protected:
  double log_prior_constant;
  TIndex free_indices;
  TDenseVector constant;

public:
  TTimeSeriesRestricted(const TTimeSeriesRestricted<T> &Model)
    : T(Model), log_prior_constant(Model.log_prior_constant), free_indices(Model.free_indices), constant(Model.constant) { };

  TTimeSeriesRestricted(const T &Model, double LogPriorConstant, const TIndex &FreeIndices, const TDenseVector &Constant)
    : T(Model), log_prior_constant(LogPriorConstant), free_indices(FreeIndices), constant(Constant) { };

  virtual bool SetParameters(double *parameters);
  virtual bool GetParameters(double *parameters) const;

  virtual double LogPrior(void) { return T::LogPrior() + log_prior_constant; }

  virtual int NumberParameters(void) const { return free_indices.size; };
};

// Implementation of SetParameters() and GetParameters()
template<class T> bool TTimeSeriesRestricted<T>::SetParameters(double *parameters)
{
  TDenseVector p(constant);
  p.UniqueMemory();
  for (int i=free_indices.size-1; i >= 0; i--) p.vector[free_indices.index[i]]=parameters[i];
  return T::SetParameters(p.vector);
}

template<class T> bool TTimeSeriesRestricted<T>::GetParameters(double *parameters) const
{
  TDenseVector p(constant.dim);
  bool rtrn=T::GetParameters(p.vector);
  for (int i=free_indices.size-1; i >= 0; i--) parameters[i]=p.vector[free_indices.index[i]];
  return rtrn;
}
#endif
