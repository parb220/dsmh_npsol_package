#include <sstream>
#include <cmath>
#include <string>
#include <cfloat>

#include "dw_data.hpp"
#include "dw_ascii.hpp"

//===============================================================================
//=== class TData
//===============================================================================
TData::TData(const TData &Data)
  : data(Data.data),
    date0(Data.date0),
    date_inc(Data.date_inc),
    dates(Data.dates ? new TDenseVector(*(Data.dates)) : (TDenseVector*)NULL),
    names(Data.names ? new std::vector<std::string >(*(Data.names)) : (std::vector<std::string >*)NULL)
{ }

// Can be used to construct data set with no observations but correct number of
// variables.  Caution must be used as some routines may expect positive number 
// of observations.
TData::TData(int NumberVariables)
  : data(0,NumberVariables,false),
    date0(0.0),
    date_inc(1.0),
    dates((TDenseVector*)NULL),
    names((std::vector<std::string >*)NULL)
{ }

TData::TData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex)
{
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  date_inc=1.0;
  date0=BeginIndex;
  dates=(TDenseVector*)NULL;

  // set name info
  names=(std::vector<std::string >*)NULL;
}

TData::TData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear)
{
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  if (PeriodsPerYear <= 0.0) throw dw_exception("TData(): periods per year must be positive");
  date_inc=1.0/PeriodsPerYear;
  date0=BeginDate + BeginIndex*date_inc;
  dates=(TDenseVector*)NULL;

  // set name info
  names=(std::vector<std::string >*)NULL;
}

TData::TData(const TDenseMatrix &RawData, const TIndex &Columns, int DateColumn, double BeginDate, double EndDate)
{
  int BeginIndex=::FindDate(BeginDate,RawData,DateColumn), EndIndex=::FindDate(EndDate,RawData,DateColumn);
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  dates=new TDenseVector(EndIndex-BeginIndex);
  dates->ColumnVector(RawData,DateColumn,BeginIndex,EndIndex);
  date0=dates->operator()(0);
  date_inc=1.0;

  // set name info
  names=(std::vector<std::string >*)NULL;
}

TData::TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int BeginIndex, int EndIndex)
{
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  date_inc=1.0;
  date0=BeginIndex;
  dates=(TDenseVector*)NULL;

  // set name info
  if ((int)RawNames.size() != RawData.NumberColumns()) throw dw_exception("TData(): incorrect number of variable names");
  names=new std::vector<std::string >(Columns.Size());
  for (int i=Columns.Size()-1; i >= 0; i--) (*names)[i]=RawNames[Columns[i]];
}

TData::TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear)
{
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  if (PeriodsPerYear <= 0.0) throw dw_exception("TData(): periods per year must be positive");
  date_inc=1.0/PeriodsPerYear;
  date0=BeginDate + BeginIndex*date_inc;
  dates=(TDenseVector*)NULL;

  // set name info
  if ((int)RawNames.size() != RawData.NumberColumns()) throw dw_exception("TData(): incorrect number of variable names");
  names=new std::vector<std::string >(Columns.Size());
  for (int i=Columns.Size()-1; i >= 0; i--) (*names)[i]=RawNames[Columns[i]];
}

TData::TData(const TDenseMatrix &RawData, const std::vector<std::string> RawNames, const TIndex &Columns, int DateColumn, double BeginDate, double EndDate)
{
  int BeginIndex=::FindDate(BeginDate,RawData,DateColumn), EndIndex=::FindDate(EndDate,RawData,DateColumn);
  ConstructData(RawData,Columns,BeginIndex,EndIndex);

  // set date info
  dates=new TDenseVector(EndIndex-BeginIndex);
  dates->ColumnVector(RawData,DateColumn,BeginIndex,EndIndex);
  date0=dates->operator()(0);
  date_inc=1.0;

  // set name info
  if ((int)RawNames.size() != RawData.NumberColumns()) throw dw_exception("TData(): incorrect number of variable names");
  names=new std::vector<std::string >(Columns.Size());
  for (int i=Columns.Size()-1; i >= 0; i--) (*names)[i]=RawNames[Columns[i]];
}

void TData::ConstructData(const TDenseMatrix &RawData, const TIndex &Columns, int BeginIndex, int EndIndex)
{
  // checks
  if (Columns.Size() == 0) throw dw_exception("TData(): must have at least one endogenous variable");
  if (EndIndex < BeginIndex) throw dw_exception("TData(): ending index less than beginning index");
  if ((BeginIndex < 0) || (EndIndex >= RawData.rows)) throw dw_exception("TData(): date indices out of range");

  // set data
  data=RawData.SubMatrix(BeginIndex,EndIndex,Columns);
  data.ForceRowMajor();
}

TDenseVector TData::Dates(void)
{
  if (dates) 
    return *dates;
  else
    {
      TDenseVector d(data.NumberRows());
      for (int t=d.Dimension()-1; t >= 0; t--) d.vector[t]=date0+t*date_inc;
      return d;
    }
}

// Dates will be returned even if it is not the case that 0 <= t < data.NumberRows().  However,
// this may not make sense if there is a date vector.
double TData::Date(int t)
{
  if (!dates)
    return date0+t*date_inc;
  else  if (t < 0)
    return dates->vector[0]-t;
  else if (t >= dates->dim)
    return dates->vector[dates->dim-1]+(t-dates->dim+1);
  else
    return dates->vector[t];
}

// Gets the index for Date.  Throws exception if Date is not found in the proper range.
int TData::FindDate(double Date)
{
  if (dates)
    {
      for (int t=dates->Dimension()-1; t >= 0; t--) if (dates->vector[t] == Date) return t;
      throw dw_exception("FindDate(): date not found");
    }
  else
    { 
      double t=(Date - date0)/date_inc;
      int idx=(int)floor(t + sqrt(DBL_EPSILON));
      if ((idx < 0) || (data.NumberRows() <= idx)) throw dw_exception("FindDate(): date out of range");
      return idx;
    }
}

std::string TData::VariableName(int i)
{
  if ((i < 0) || (data.NumberColumns() <= i)) throw dw_exception("VariableName(): index out of range");
  return (names)  ? (*names)[i] : "variable_" + cluster_to_string(i);
}

//===============================================================================
//=== Class TData_predetermined
//===============================================================================
TData_predetermined::TData_predetermined(const TData_predetermined &Data) 
  : TData(Data),
    predetermined_data(Data.predetermined_data), 
    n_lags(Data.n_lags),
    is_constant(Data.is_constant),
    exogenous_names(Data.exogenous_names ? new std::vector<std::string >(*(Data.exogenous_names)) : (std::vector<std::string >*)NULL)
{ }

// Can be used to construct data set with no observations but correct number of
// variables.  Caution must be used as some routines may expect positive number 
// of observations.
//
// Note that NumberExogenous does not include the constant term.
TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, int NumberVariables, int NumberExogenous)
  : TData(NumberVariables),
    predetermined_data(0,NumberVariables*NumberLags+NumberExogenous+(IsConstant ? 1 : 0)),
    n_lags(NumberLags),
    is_constant(IsConstant),
    exogenous_names((std::vector<std::string >*)NULL)
{
  if (NumberLags < 0) throw dw_exception("TData_predetermined(): number of lags must be non-negative");
  if (NumberExogenous < 0) throw dw_exception("TData_predetermined(): number of exogenous variables  must be non-negative");
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
					 const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex)
  : TData(RawData,DataColumns,BeginIndex,EndIndex)
{
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));
  exogenous_names=(std::vector<std::string>*)NULL;
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
					 const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear)
  : TData(RawData,DataColumns,BeginIndex,EndIndex,BeginDate,PeriodsPerYear)
{
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));
  exogenous_names=(std::vector<std::string>*)NULL;
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const TIndex &DataColumns, 
					 const TIndex &ExogenousDataColumns, int DateColumn, double BeginDate, double EndDate)
  : TData(RawData,DataColumns,DateColumn,BeginDate,EndDate)
{
  int BeginIndex=::FindDate(BeginDate,RawData,DateColumn), EndIndex=::FindDate(EndDate,RawData,DateColumn);
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));
  exogenous_names=(std::vector<std::string>*)NULL;
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
					 const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex)
  : TData(RawData,DataColumns,BeginIndex,EndIndex)
{
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));

  exogenous_names=new std::vector<std::string >(ExogenousDataColumns.Size());
  for (int i=ExogenousDataColumns.Size()-1; i >= 0; i--) (*exogenous_names)[i]=RawNames[ExogenousDataColumns[i]];
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
					 const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int BeginIndex, int EndIndex, double BeginDate, double PeriodsPerYear)
  : TData(RawData,DataColumns,BeginIndex,EndIndex,BeginDate,PeriodsPerYear)
{
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));

  exogenous_names=new std::vector<std::string >(ExogenousDataColumns.Size());
  for (int i=ExogenousDataColumns.Size()-1; i >= 0; i--) (*exogenous_names)[i]=RawNames[ExogenousDataColumns[i]];
}

TData_predetermined::TData_predetermined(int NumberLags, bool IsConstant, const TDenseMatrix &RawData, const std::vector<std::string> RawNames, 
					 const TIndex &DataColumns, const TIndex &ExogenousDataColumns, int DateColumn, double BeginDate, double EndDate)
  : TData(RawData,RawNames,DataColumns,DateColumn,BeginDate,EndDate)
{
  int BeginIndex=::FindDate(BeginDate,RawData,DateColumn), EndIndex=::FindDate(EndDate,RawData,DateColumn);
  ConstructPredeterminedData(NumberLags,IsConstant,RawData.SubMatrix(BeginIndex-NumberLags,BeginIndex-1,DataColumns),RawData.SubMatrix(BeginIndex,EndIndex,ExogenousDataColumns));

  exogenous_names=new std::vector<std::string >(ExogenousDataColumns.Size());
  for (int i=ExogenousDataColumns.Size()-1; i >= 0; i--) (*exogenous_names)[i]=RawNames[ExogenousDataColumns[i]];
}

void TData_predetermined::ConstructPredeterminedData(int NumberLags, bool IsConstant, const TDenseMatrix &InitialData, const TDenseMatrix &ExogenousData)
{
  if (NumberLags < 0) throw dw_exception("TData_predetermined(): number of lags must be non-negative");
  int NumberVariables=InitialData.NumberColumns();
  int NumberObservations=ExogenousData.NumberRows();
  TDenseMatrix lags(NumberObservations,NumberLags*NumberVariables);
  if (NumberLags > 0)
    {
      for (int i=1; i <= NumberLags; i++)
	lags.Insert(0,(i-1)*NumberVariables,InitialData,NumberLags-i,NumberLags-i,0,NumberVariables-1);
      for (int t=1; t < NumberObservations; t++)
	{
	  if (NumberLags > 1) lags.Insert(t,NumberVariables,lags,t-1,t-1,0,(NumberLags-1)*NumberVariables-1);
	  lags.Insert(t,0,data,t-1,t-1,0,NumberVariables-1);
	}
    }
  predetermined_data=HCat(lags,ExogenousData);
  if (IsConstant) predetermined_data=HCat(predetermined_data,TDenseMatrix(NumberObservations,1,1.0));
  predetermined_data.ForceRowMajor();

  n_lags=NumberLags;
  is_constant=IsConstant;
}

std::string TData_predetermined::PredeterminedVariableName(int i)
{
  if ((i < 0) || (i >= predetermined_data.cols))
    throw dw_exception("PredeterminedVariableName(): index out of range");

  if (i < n_lags*data.cols)
    {
      std::string str=names ? (*names)[i % data.cols] : "variable_" + cluster_to_string(i % data.cols);
      return str + "(t-" + cluster_to_string(i/data.cols + 1) + ")";
    }
  else if (is_constant && (i == predetermined_data.cols-1))
    return std::string("constant");
  else if (exogenous_names)
    return (*exogenous_names)[i-n_lags*data.cols];
  else
    return "exogenous_variable_" + cluster_to_string(i);
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
//=== auxiliary routines
//===============================================================================
int FindDate(double Date, const TDenseMatrix &RawData, int DateColumn)
{
  if ((DateColumn < 0) || (DateColumn >= RawData.NumberColumns()))
    throw dw_exception("TData::FindDate(): date column out of range");

  for (int t=RawData.NumberRows()-1; t >= 0; t--)
    if (RawData(t,DateColumn) == Date) return t;

  throw dw_exception("TData::FindDate(): date not found");
}
//===============================================================================
//===============================================================================
//===============================================================================
