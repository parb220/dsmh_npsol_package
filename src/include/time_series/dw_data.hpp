#ifndef _DW_DATA_HEADER
#define _DW_DATA_HEADER
#include <vector>
#include <string>

#include "dw_dense_matrix.hpp"

class TData
{
protected:
  TDenseMatrix data;
  double date0;
  double date_inc;
  TDenseVector* dates;
  std::vector<std::string > *names;

  // protected constructor helper
  void ConstructData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex);

public:
  // constructors
  TData() : data(TDenseMatrix(0,0)), dates(NULL), names(NULL) {}; 
  TData(const TData &Data);
  explicit TData(int NumberVariables);
  TData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex);
  TData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear);
  TData(const TDenseMatrix &RawData, const TIndex &Columns, int DateColumn, double BeginDate, double EndDate);
  TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int BeginIndex, int EndIndex);
  TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear);
  TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int DateColumn, double BeginDate, double EndDate);

  // destructors
  virtual ~TData() { delete dates; delete names; };

  // cloning an object
  virtual TData* Clone(void) { return new TData(*this); };

  // data info
  TDenseMatrix Data(void) { return data; };
  TDenseVector Data(int t) { return data.RowVector(t); };

  // size info
  int NumberVariables(void) { return data.cols; };
  int NumberObservations(void) { return data.rows; };

  // date info
  TDenseVector Dates(void);
  double Date(int t);
  int FindDate(double Date);

  // variable info
  virtual std::string VariableName(int i);

  // descriptive statistics
  TDenseVector Mean(void) { return (data.rows > 0) ? ConstantVector(1.0/(double)data.rows,data.rows)*data : ZeroVector(data.cols); };
  TDenseMatrix Covariance(void) { return (data.rows > 0) ? (1.0/(double)data.rows)*TransposeMultiply(data,data)-OuterProduct(Mean(), Mean()) : TDenseMatrix(data.cols,data.cols,0.0); };
};

class TData_predetermined : public TData
{
protected:
  TDenseMatrix predetermined_data;
  int n_lags;
  bool is_constant;
  std::vector<std::string> *exogenous_names;

  void ConstructPredeterminedData(int NumberLags, bool IsConstant, const TDenseMatrix &InitialData, const TDenseMatrix &ExogenousData);

public:
  // constructors
	TData_predetermined() : TData(), predetermined_data(TDenseMatrix(0,0)), n_lags(0), is_constant(false), exogenous_names(NULL) {}; 
  TData_predetermined(const TData_predetermined &Data);
  TData_predetermined(int NumberLags, bool IsConstant, int NumberVariables, int NumberExogenous);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
		      const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
		      const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
		      const TIndex &ExogenousDataColumns, int DateColumn, double BeginDate, double EndDate);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
		      const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
		      const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear);
  TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
		      const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int DateColumn, double BeginDate, double EndDate);

  // destructors
  virtual ~TData_predetermined() { delete exogenous_names; }

  // cloning an object
  virtual TData_predetermined* Clone(void) { return new TData_predetermined(*this); };

  // data info
  TDenseMatrix PredeterminedData(void) { return predetermined_data; };
  TDenseVector PredeterminedData(int t) { return RowVector(predetermined_data,t); };
  int NumberLags(void) { return n_lags; };
  bool IsConstant(void) { return is_constant; };

  // info
  int NumberPredeterminedVariables(void) { return predetermined_data.cols; };
  virtual std::string PredeterminedVariableName(int i);
};

int FindDate(double date, const TDenseMatrix &RawData, int DateColumn);

#endif
