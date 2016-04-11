#include <cmath>
////  #include <ctgmath>
#include <gsl/gsl_sf_gamma.h>
#include "tvsbvar.hpp"
// #include "specialfunctions.h" // invincompletegammac, /home/f1hxw01/alglib/alglib-3.8.0/cpp/src

using namespace std; 
double Sims_Zha::log2pi; 
void MakeSelectRows(vector<TIndex> &select_a, vector<TIndex> &select_b, const TIndex &select, int n_vars, int n_predetermined);

Sims_Zha::Sims_Zha() :
if_precalculated(false), 
if_preallocated(false),
mean(TDenseVector(0)), 
s(TDenseVector(0)), 
Sbar(TDenseMatrix(0,0)), 
YY(TDenseMatrix(0,0)), 
XY(TDenseMatrix(0,0)), 
XX(TDenseMatrix(0,0)),
hyper(TDenseVector(0)), periods_per_year(0), variance_scale(0), 
base_flat_prior(true), X_dummy(TDenseMatrix(0,0)), Y_dummy(TDenseMatrix(0,0)),
select_a(vector<vector<TIndex> >(0)), select_aplus(vector<vector<TIndex> >(0)), 
H(vector<vector<TDenseMatrix> >(0)), P(vector<vector<TDenseMatrix> >(0)), S(vector<vector<TDenseMatrix> >(0)),
c_a(vector<vector<TDenseVector> >(0)), c_aplus(vector<vector<TDenseVector> >(0)), 
base_prior_constant(0.0), base_restriction_constant(0.0),
mult_flat_prior(true), mult_prior_a(TDenseVector(0)), mult_prior_b(TDenseVector(0)), mult_prior_constant(0),
add_flat_prior(true), add_prior_mean(TDenseVector(0)), add_prior_variance(TDenseVector(0)), add_prior_constant(0.0)
{
}

Sims_Zha::Sims_Zha(const Sims_Zha &right) :
if_precalculated(right.if_precalculated), 
if_preallocated(right.if_preallocated), 
mean(TDenseVector(right.mean.dim)), 
s(TDenseVector(right.s.dim)),
Sbar(TDenseMatrix(right.Sbar.rows,right.Sbar.cols)),
YY(TDenseMatrix(right.YY.rows,right.YY.cols)),
XY(TDenseMatrix(right.XY.rows,right.XY.cols)),
XX(TDenseMatrix(right.XX.rows,right.XX.cols)),
hyper(TDenseVector(right.hyper.dim)), 
periods_per_year(right.periods_per_year), 
variance_scale(right.variance_scale),
base_flat_prior(right.base_flat_prior), 
X_dummy(TDenseMatrix(right.X_dummy.rows,right.X_dummy.cols)),
Y_dummy(TDenseMatrix(right.Y_dummy.rows,right.Y_dummy.cols)),
select_a(right.select_a), 
select_aplus(right.select_aplus),
H(right.H), 
P(right.P), 
S(right.S), 
c_a(right.c_a), 
c_aplus(right.c_aplus), 
base_prior_constant(right.base_prior_constant), 
base_restriction_constant(right.base_restriction_constant),
mult_flat_prior(right.mult_flat_prior), 
mult_prior_a(TDenseVector(right.mult_prior_a.dim)), 
mult_prior_b(TDenseVector(right.mult_prior_b.dim)),
mult_prior_constant(right.mult_prior_constant),
add_flat_prior(right.add_flat_prior), 
add_prior_mean(TDenseVector(right.add_prior_mean.dim)), 
add_prior_variance(TDenseVector(right.add_prior_variance.dim)), 
  add_prior_constant(right.add_prior_constant),
  // added by DW 4/9/15
  SqrtInvS(right.SqrtInvS),
  SqrtInvH(right.SqrtInvH)
{
	mean.CopyContent(right.mean); 
	s.CopyContent(right.s);
	Sbar.CopyContent(right.Sbar); 
	YY.CopyContent(right.YY); 
	XY.CopyContent(right.XY); 
	XX.CopyContent(right.XX); 

	hyper.CopyContent(right.hyper); 
	X_dummy.CopyContent(right.X_dummy); 
	Y_dummy.CopyContent(right.Y_dummy); 
	for (int kk=0; kk<(int)right.H.size(); kk++)
	{
		H[kk]=vector<TDenseMatrix>(right.H[kk].size()); 
		for(int ii=0; ii<(int)right.H[kk].size(); ii++)
			H[kk][ii].CopyContent(right.H[kk][ii]); 
	}
	for (int kk=0; kk<(int)right.P.size(); kk++)
        {
                P[kk]=vector<TDenseMatrix>(right.P[kk].size());
                for(int ii=0; ii<(int)right.P[kk].size(); ii++)
                        P[kk][ii].CopyContent(right.P[kk][ii]);
        }
	for (int kk=0; kk<(int)right.S.size(); kk++)
        {
                S[kk]=vector<TDenseMatrix>(right.S[kk].size());
                for(int ii=0; ii<(int)right.S[kk].size(); ii++)
                        S[kk][ii].CopyContent(right.S[kk][ii]);
        }
	for (int kk=0; kk<(int)right.c_a.size(); kk++)
        {
                c_a[kk]=vector<TDenseVector>(right.c_a[kk].size());
                for(int ii=0; ii<(int)right.c_a[kk].size(); ii++)
                        c_a[kk][ii].CopyContent(right.c_a[kk][ii]);
        }
	for (int kk=0; kk<(int)right.c_aplus.size(); kk++)
        {
                c_aplus[kk]=vector<TDenseVector>(right.c_aplus[kk].size());
                for(int ii=0; ii<(int)right.c_aplus[kk].size(); ii++)
                        c_aplus[kk][ii].CopyContent(right.c_aplus[kk][ii]);
        }
	mult_prior_a.CopyContent(right.mult_prior_a); 
	mult_prior_b.CopyContent(right.mult_prior_b);
	add_prior_mean.CopyContent(right.add_prior_mean);
	add_prior_variance.CopyContent(right.add_prior_variance);  
}

bool Sims_Zha::SetupPrior_SimsZha(const TDenseVector &_hyper, int _periods_per_year, double _variance_scale, const Restriction &base_restriction, const Restriction &mult_restriction, const Restriction &add_restriction, TData_predetermined *data)  
{
	bool return_value = true; 
	//SetupPrior_SimsZha.m
	int n_vars = data->NumberVariables(); 
	int n_predetermined = data->NumberPredeterminedVariables(); 
	int n_lags = (n_predetermined-1)/n_vars; 	

	hyper = _hyper; 
	periods_per_year = _periods_per_year; 
	variance_scale = _variance_scale; 

	if (!std::isnan(hyper[0]) ) // if hyper[0] is finite
	{
		// not using flat prior
		base_flat_prior = false; 
		
		// dummy observations
		GetDummyObservations(data,n_lags);  
		
		MakeSelect(n_vars,n_predetermined,base_restriction); 

		if (!MakeHPSc(base_restriction))
			return_value = false;  
	}
	else 
		base_flat_prior = true; 

	// multiplicative parameter gamma prior
	if (mult_restriction.n_free && !std::isnan(hyper[7]) )
	{
		/* The following is consistent with TSBVAR_mult.m
		double a = 1.0/hyper[7]; 
		// invincompletegammac is a function provided by alglib
		// It is the inverse function of the upper incomplete gammma function. 
		// Here upper incomplete gamma function refers to 
		//     1/Gamma(a) * integral from x to infinity of t^(a-1)exp(-t) dt
		// and the lower (default) incomplete gamma function refers to 
		//     1/Gamma(a) * integral from 0 to x of t^(a-1)exp(-t) dt
		// The matlab function gammaincinv is the inverse lower incomplete function. 
		// So to get an equivalent of gammaincinv(0.5,a) as in Line 140 of SetupPrior_SimsZha.m
		// using invincompletegammac, we used the following relationship
		// 	gammaincinv(y, a, 'upper') = gamma(1-y, a, 'lower'); 
		//
		double b = 1.0/alglib::invincompletegammac(a,1.0-0.5);
		mult_flat_prior = false; 
		mult_prior_a=TDenseVector(mult_restriction.n_free,a); 
		mult_prior_b = TDenseVector(mult_restriction.n_free,b); 
		mult_prior_constant = mult_restriction.n_free*(-a*log(b)-gsl_sf_lngamma(a));
		// consistent with TSBVAR_mult.m*/
		/* The following is consistent with TSBVAR_mult_square.m */
		double a = 0.5; 
		double b = hyper[7]*hyper[7]/a; 
		mult_flat_prior = false; 
		mult_prior_a=TDenseVector(mult_restriction.n_free,a);
		mult_prior_b = TDenseVector(mult_restriction.n_free,b); 
		// changed by DW 4/9/15: if u is gamma with parameters a and b and x is the actual free parameters,
		// then x^2 = u.  Thus the Jacobian is 2*x, but since we allow x to be either positive or negative
		// with equal probability, then the Jacobian must be halved. So:
		mult_prior_constant = mult_restriction.n_free*(-a*log(b)-gsl_sf_lngamma(a));
		// mult_prior_constant = mult_restriction.n_free*(-a*log(b)-gsl_sf_lngamma(a))+log(2.0)*mult_restriction.n_free; 
		// for (int ii=0; ii<mult_restriction.n_free; ii++)
		//	mult_prior_constant += -mult_prior_a[ii]*log(mult_prior_b[ii])-gsl_sf_lngamma(mult_prior_a[ii]); 
		// consistent with TSBVAR_mult_square.m*/
	}	
	else 
		mult_flat_prior = true;

	// additive parameter Gaussian prior
	if (add_restriction.n_free && !std::isnan(hyper[8]) )
	{
		double variance = hyper[8]; 
		add_flat_prior = false; 
		add_prior_mean = TDenseVector(add_restriction.n_free,0.0); 
		add_prior_variance = TDenseVector(add_restriction.n_free,1.0)*variance; 
		add_prior_constant = -add_restriction.n_free*0.5*(log2pi+log(variance)); 
	}
	else 
		add_flat_prior = true;

	// added by DW 4/9/15 for simulating prior
	SqrtInvS.resize(S.size());
	for (int kk=0; kk < SqrtInvS.size(); kk++)
	  {
	    SqrtInvS[kk].resize(S[kk].size());
	    for (int ii=0; ii < SqrtInvS[kk].size(); ii++)
	      SqrtInvS[kk][ii].Inverse(Cholesky(S[kk][ii],CHOLESKY_UPPER_TRIANGULAR),SOLVE_UPPER_TRIANGULAR);
	  }
	SqrtInvH.resize(H.size());
	for (int kk=0; kk < SqrtInvH.size(); kk++)
	  {
	    SqrtInvH[kk].resize(H[kk].size());
	    for (int ii=0; ii < SqrtInvH[kk].size(); ii++)
	      SqrtInvH[kk][ii].Inverse(Cholesky(H[kk][ii],CHOLESKY_UPPER_TRIANGULAR),SOLVE_UPPER_TRIANGULAR);
	  }

	return return_value; 
}

void Sims_Zha::MakeSelect(int n_vars, int n_predetermined, const Restriction &restriction)
{
	int n_regimes = restriction.n_regimes; 
	select_a=vector<vector<TIndex> >(n_regimes,vector<TIndex>(n_vars) ); 
	select_aplus=vector<vector<TIndex> >(n_regimes,vector<TIndex>(n_vars) ); 

	int offset, dim, idx, max_select, nn; 
	for (int kk=0; kk<n_regimes; kk++)
	{
		offset = restriction.offset[kk]; 
		dim = restriction.dim[kk];
 
		idx = 0; 
		max_select = 0; 
		// selection for A(kk)
		for (int ii=0; ii<n_vars; ii++)
		{	
			nn=0; 
			max_select += n_vars; 
			while (idx < dim && restriction.select[kk][idx] < max_select)
			{
				nn++; 
				idx++; 
			}
			select_a[kk][ii] = TIndex(offset,offset+nn-1); 
			offset += nn; 
		}
		// selection for Aplus(kk)
		for (int ii=0; ii<n_vars; ii++)
		{
			nn=0; 
			max_select += n_predetermined; 
			while (idx <dim && restriction.select[kk][idx] < max_select)
			{
				nn ++; 
				idx ++; 
			}
			select_aplus[kk][ii] = TIndex(offset,offset+nn-1); 
			offset += nn; 
		}
	}
}

void Sims_Zha::GetDummyObservations_Precalculations(TData_predetermined *data, int n_lags)
{
	int n_vars = data->NumberVariables();
        int T = data->NumberObservations();
        int n_predetermined = data->NumberPredeterminedVariables();	
	
	// mean of reshape(data->PredeterminedData(0,1:end-1), n_vars, n_lags) along
	// row directions
	// mean(reshape(predetermined_data(1:end-1,1),n_vars,n_lags),2)
	TDenseVector row_predetermined = data->PredeterminedData(0); 
	TDenseVector selected_data;
	mean=TDenseVector(n_vars,0.0); 
	for (int ii=0; ii<n_vars; ii++)
	{
		selected_data = row_predetermined.SubVector(TIndex(ii,n_vars,n_predetermined-2)); 
		for (int jj=0; jj<selected_data.dim; jj++)
			mean[ii] += selected_data[jj]; 
		mean[ii] = mean[ii]/selected_data.dim; 
	}

	// variance of univariate AR residuals
	TDenseVector y, e;  
	TDenseMatrix X, Q, R; 
	TDenseMatrix predetermined = data->PredeterminedData(); 
	s=TDenseVector(n_vars,0.0); 
	for (int ii=0; ii<n_vars; ii++)
	{
		X=predetermined.SubMatrix(0,T-1,TIndex(ii,n_vars,n_predetermined-2)(n_predetermined-1) ); 
		QR(Q, R, X); 
		y = (data->Data()).ColumnVector(ii); 
		e=y-Multiply(Q, TransposeMultiply(Q,y) ); 
		s[ii] = sqrt(InnerProduct(e,e)/T);
	}
	
	// dummy observations
	Y_dummy=TDenseMatrix(2+n_vars*(n_lags+2),n_vars,0.0); 
	X_dummy=TDenseMatrix(2+n_vars*(n_lags+2),n_predetermined,0.0); 
	
	if_precalculated = true; 
}

void Sims_Zha::GetDummyObservations(TData_predetermined *data, int n_lags) 
{
	int n_vars = data->NumberVariables(); 
	int n_predetermined = data->NumberPredeterminedVariables(); 
	
	if (!if_precalculated)
		GetDummyObservations_Precalculations(data, n_lags); 
		
	// Prior on a(k,i): dummy observations of the form	
	int start_row = 0; 
	for (int ii=0; ii<n_vars; ii++)
		Y_dummy(start_row+ii,ii) = s(ii)/hyper[0]; 
	start_row += n_vars; 

	// Prior on constant: dummy observations of the form
	X_dummy(start_row, n_predetermined-1) = 1.0/(hyper[0]*hyper[2]); 
	start_row ++; 

	if (n_lags > 0)
	{
		// random walk prior: 
		for (int ii=0; ii<n_vars; ii++)
		{
			Y_dummy(start_row+ii,ii) = s(ii)/(hyper[0]*hyper[1]); 
			X_dummy(start_row+ii,ii) = s(ii)/(hyper[0]*hyper[1]); 
		}
		start_row += n_vars; 

		// lag decay prior
		for (int jj=1; jj<n_lags; jj++)
		{
			for (int ii=0; ii<n_vars; ii++)
				// hyper[3]: hyper_{4a}
				// hyper[4]: hyper_{4b}
				X_dummy(start_row+ii, jj*n_vars+ii) = s(ii)*pow((4.0*hyper[3]*jj/periods_per_year+1.0),hyper[4])/(hyper[0]*hyper[1]);

			start_row += n_vars; 
		}

		// sums-of-coefficients prior
		for (int ii=0; ii<n_vars; ii++)
		{
			Y_dummy(start_row+ii,ii)=hyper[5]*mean[ii];
			for (int jj=0; jj<n_lags; jj++)
				X_dummy(start_row+ii, jj*n_vars+ii)=hyper[5]*mean[ii];  
		}
		start_row += n_vars; 

		// co-persistence prior
		for (int ii=0; ii<n_vars; ii++)
		{
			Y_dummy(start_row,ii) = hyper[6]*mean[ii]; 
			for (int jj=0; jj<n_lags; jj++)
				X_dummy(start_row, jj*n_vars+ii) = hyper[6]*mean[ii]; 
		}
		X_dummy(start_row, n_predetermined-1) = hyper[6]; 
	}
	
	Y_dummy = Y_dummy * (1.0/sqrt(variance_scale)); 
	X_dummy = X_dummy * (1.0/sqrt(variance_scale)); 
}

void Sims_Zha::MakeHPSc_PreAllocation(const Restriction &restriction)
{
	int n_vars = Y_dummy.cols; 
	int n_predetermined = X_dummy.cols; 
	Sbar = TDenseMatrix(n_vars+n_predetermined, n_vars+n_predetermined, 0.0); 
	YY = TDenseMatrix(n_vars, n_vars, 0.0); 
	XY = TDenseMatrix(n_vars, n_predetermined, 0.0); 
	XX = TDenseMatrix(n_predetermined, n_predetermined, 0.0); 
	H=vector<vector<TDenseMatrix> >(restriction.n_regimes,vector<TDenseMatrix>(n_vars)); 
	P=vector<vector<TDenseMatrix> >(restriction.n_regimes,vector<TDenseMatrix>(n_vars));
	S=vector<vector<TDenseMatrix> >(restriction.n_regimes,vector<TDenseMatrix>(n_vars));
	c_a=vector<vector<TDenseVector> >(restriction.n_regimes,vector<TDenseVector>(n_vars) ); 
	c_aplus=vector<vector<TDenseVector> >(restriction.n_regimes,vector<TDenseVector>(n_vars) ); 
	if_preallocated = true; 
}

bool Sims_Zha::MakeHPSc(const Restriction &restriction)
{
	// MakeHPSc.m
	if (!if_preallocated)
		MakeHPSc_PreAllocation(restriction); 
	
	// sizes; 
	int n_vars = Y_dummy.cols; 
	int n_predetermined = X_dummy.cols; 

	// Sbar
	YY = TransposeMultiply(Y_dummy,Y_dummy); 
	YY = (YY+Transpose(YY)) * 0.5; 
	XY = TransposeMultiply(X_dummy,Y_dummy); 
	XX = TransposeMultiply(X_dummy,X_dummy); 
	XX = (XX+Transpose(XX)) * 0.5; 
	// Sbar=[YY, -XY'; -XY, XX]
	Sbar.Insert(0,0,YY); 
	Sbar.Insert(0,n_vars,Transpose(XY)*(-1.0)); 
	Sbar.Insert(n_vars,0,XY*(-1.0)); 
	Sbar.Insert(n_vars,n_vars,XX); 
	Sbar = (Sbar+Transpose(Sbar)) * 0.5; 

	// Setup H, P, S, c_a, c_aplus
	// temporary variables
	vector<TDenseMatrix>DSD(n_vars); 
	vector<TDenseVector>Pd(n_vars);

	// initial constants
	base_prior_constant = 0.0; 
 	base_restriction_constant = 0.0;  

	vector<TIndex>select_a_kk, select_aplus_kk; 
	for (int kk=0; kk<restriction.n_regimes; kk++)
	{
		MakeSelectRows(select_a_kk, select_aplus_kk, restriction.select[kk], n_vars, n_predetermined); 

		for (int ii=0; ii<n_vars; ii++)
		{
			H[kk][ii] = XX.SubMatrix(select_aplus_kk[ii], select_aplus_kk[ii]); 
			try {
				P[kk][ii] = InverseMultiply(H[kk][ii], XY.SubMatrix(select_aplus_kk[ii], select_a_kk[ii]) ); 
			}
			catch(...){
				return false; 
			}	
			S[kk][ii] = YY.SubMatrix(select_a_kk[ii],select_a_kk[ii]) - TransposeMultiply(P[kk][ii], Multiply(H[kk][ii],P[kk][ii]) ); 
			S[kk][ii] = (S[kk][ii]+Transpose(S[kk][ii])) * 0.5; 

			int dim_a = select_a_kk[ii].Size(); 
			int dim_aplus = select_aplus_kk[ii].Size(); 
			
			base_prior_constant += -0.5*(dim_a+dim_aplus)*log2pi + 0.5*LogAbsDeterminant(S[kk][ii]) + 0.5*LogAbsDeterminant(H[kk][ii]); 

			// select_ab=[select_a{ii}; select_b{ii}+n_vars];
			TIndex select_a_aplus(select_a_kk[ii]); 
			for (int jj=0; jj<select_aplus_kk[ii].Size(); jj++)
				select_a_aplus += (select_aplus_kk[ii][jj] + n_vars); 

			TDenseVector d=restriction.d[kk](TIndex(ii*n_vars,(ii+1)*n_vars-1)(n_vars*n_vars+ii*n_predetermined,n_vars*n_vars+(ii+1)*n_predetermined-1)); 
			TDenseVector Sd = Sbar *d; 
			TDenseVector DSd= Sd.SubVector(select_a_aplus); 
			DSD[ii] = Sbar.SubMatrix(select_a_aplus, select_a_aplus);
			DSD[ii] = (DSD[ii]+Transpose(DSD[ii]))*0.5; 
			try {
				Pd[ii] = InverseMultiply(DSD[ii], DSd); 
			}
			catch (...){
				return false; 
			}
			c_a[kk][ii] = Pd[ii].SubVector(0, dim_a-1) * (-1.0); 
			c_aplus[kk][ii] = Pd[ii].SubVector(dim_a, dim_a+dim_aplus-1) * (-1.0);  
		 
			base_restriction_constant += -0.5*InnerProduct(d,Sd)+0.5*InnerProduct(DSd, Pd[ii]); 
		}
	}
	return true; 
}

// MakeSelectRows is a stand-alone function and is called by MakeHSPc
void MakeSelectRows(vector<TIndex> &select_a, vector<TIndex> &select_aplus, const TIndex &select, int n_vars, int n_predetermined)
{
	// MakeSelectRows.m
	select_a.resize(n_vars); 
	select_aplus.resize(n_vars); 
	int dim = select.Size(); 
	int idx=0, begin_select=0, end_select, nn; 
	for (int ii=0; ii<n_vars; ii++)
	{
		end_select = begin_select + n_vars; 
		nn = 0; 
		while (idx+nn < dim && select[idx+nn] < end_select)
			nn ++; 
		select_a[ii].Clear(); 
		if (nn > 0 )
		{
		//	select_a_rows{ii}=select(idx:idx+nn-1) - begin_select
			for (int jj=0; jj<nn; jj++)
				select_a[ii] += (select[idx+jj] - begin_select); 
		}
		idx += nn; 
		begin_select = end_select; 
	}
	for (int ii=0; ii<n_vars; ii++)
	{
		end_select = begin_select + n_predetermined; 
		nn = 0; 
		while (idx+nn < dim && select[idx+nn] < end_select)
			nn ++; 
		select_aplus[ii].Clear();  
		if (nn > 0)
		{
			for (int jj=0; jj<nn; jj++)
				select_aplus[ii] += (select[idx+jj]- begin_select); 
		}
		idx += nn; 
		begin_select = end_select; 
	}
}
