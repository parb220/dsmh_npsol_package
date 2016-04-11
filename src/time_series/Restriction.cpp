#include <cmath>
#include <iostream>
#include "dw_exception.hpp"
#include "tvsbvar.hpp"

using namespace std; 

Restriction::Restriction() : 
n_regimes(0), 
n_free(0), 
offset(vector<int>(0)), 
d(vector<TDenseVector>(0) ), 
select(vector<TIndex>(0) ), 
dim(vector<int>(0))/*,
offset_a(vector<vector<int> >(0) ), 
offset_aplus(vector<vector<int> >(0) ),
offset_xi(vector<vector<int> >(0) ), 
dim_a(vector<vector<int> >(0) ), 
dim_aplus(vector<vector<int> >(0) ), 
dim_xi(vector<vector<int> >(0) )*/
{
}

Restriction::Restriction(const Restriction &rhs) :
n_regimes(rhs.n_regimes), n_free(rhs.n_free),
offset(rhs.offset), d(vector<TDenseVector>(rhs.d.size())), select(rhs.select), dim(rhs.dim)/*, 
offset_a(rhs.offset_a), offset_aplus(rhs.offset_aplus), offset_xi(rhs.offset_xi),
dim_a(rhs.dim_a), dim_aplus(rhs.dim_aplus), dim_xi(rhs.dim_xi) */
{
	for (int i=0; i<(int)d.size(); i++)
		d[i].CopyContent(rhs.d[i]); 
}

bool Restriction::AffineExclusion(const std::vector<TDenseMatrix> &RA, const std::vector<TDenseMatrix> &RAplus, const std::vector<TDenseMatrix> &RXi, int initial_offset)
{
	if (RA.empty() || RAplus.empty() || RXi.empty() || initial_offset < 0)
		throw dw_exception("Restriction::AffineExclusion : number of regimes cannot be zero and offset cannot be negative\n"); 
	
	for (int i=0; i<(int)(RA.size()); i++)
	{
		if (RA[i].rows != RA[i].cols || RA[i].rows != RAplus[i].rows || RA[i].rows != RXi[i].rows || RXi[i].rows != RXi[i].cols) 
			throw dw_exception("Restriction::AffineExclusion : dimensions of A, Aplus and Xi must match.\n"); 
	}

	// number of regimes
	n_regimes=(int)RAplus.size(); 
	int n_vars = RAplus[0].rows; 
	int n_predetermined = RAplus[0].cols; 
 
	// create and fill cells
	select=vector<TIndex>(n_regimes); 
	d.resize(n_regimes); 
	dim.resize(n_regimes); 
	offset.resize(n_regimes); 
	n_free = 0; 

	for (int kk=0; kk<n_regimes; kk++)
	{
		// select[kk] records those positions whose values are unrestricted
		// (corresponding to 'X' or 'x')
		d[kk].Zeros(n_vars*(n_vars+n_predetermined+n_vars) ); 
		select[kk].Clear();  
		// RA(ii,jj): finite number, restricted
		// RA(ii,jj): nan, unrestricted and (ii,jj) selected
		for (int ii=0; ii<n_vars; ii++)
		{
			for (int jj=0; jj<n_vars; jj++)
			{
				if (std::isnan(RA[kk](ii,jj)) )
				{
					select[kk]+=ii*n_vars+jj; 
					d[kk][ii*n_vars+jj] = 0.0; 
				}
				else 
					d[kk][ii*n_vars+jj]=RA[kk](ii,jj); 
			}
		}
		// RAplus(ii,jj): finite number, restricted
		// RAplus(ii,jj): nan, unrestricted and (ii,jj) selected
		for (int ii=0; ii<n_vars; ii++)
		{
			for (int jj=0; jj<n_predetermined; jj++)
			{
				if (std::isnan(RAplus[kk](ii,jj) ) ) 
				{
					select[kk]+=n_vars*n_vars + ii*n_predetermined+jj;
					d[kk][n_vars*n_vars + ii*n_predetermined+jj] = 0.0;  
				}
				else 
					d[kk][n_vars*n_vars + ii*n_predetermined+jj]=RAplus[kk](ii,jj); 
			}
		}
		// RXi(ii,jj): finite number, restricted
		// RXi(ii,jj): nan, unrestricted and (ii,jj) selected
		for (int ii=0; ii<n_vars; ii++)
		{
			for (int jj=0; jj<n_vars; jj++)
			{
				if (std::isnan(RXi[kk](ii,jj) ) )
				{
					select[kk]+=n_vars*n_vars + n_vars*n_predetermined + ii*n_vars+jj; 
					d[kk][n_vars*n_vars + n_vars*n_predetermined + ii*n_vars+jj] = 0.0; 
				}
				else
					d[kk][n_vars*n_vars + n_vars*n_predetermined + ii*n_vars+jj] = RXi[kk](ii,jj); 
			}
		}
		dim[kk] =select[kk].Size(); 

		if (kk==0)
			offset[kk] = initial_offset; 
		else 
			offset[kk] = offset[kk-1]+dim[kk-1]; 
		n_free += dim[kk]; 
	}

	offset_a=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 
	offset_aplus=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 
	offset_xi=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 
	dim_a=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 
	dim_aplus=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 
	dim_xi=vector<vector<int> >(n_regimes,vector<int>(n_vars,0) ); 

	int begin_select=initial_offset, end_select, idx, nn; 
	for (int kk=0; kk<n_regimes; kk++)
	{
		idx=0;
		// RA 
		for (int ii=0; ii<n_vars; ii++)
		{
			end_select = begin_select + n_vars; 
			nn = 0; 
			while(idx+nn<dim[kk] && select[kk][idx+nn] < end_select)
				nn++; 
			offset_a[kk][ii] = idx; 
			dim_a[kk][ii] = nn; 
			idx += nn; 
			begin_select = end_select; 
		}
		// RAplus
		for (int ii=0; ii<n_vars; ii++)
		{
			end_select = begin_select + n_predetermined; 
			nn = 0; 
			while(idx+nn<dim[kk] && select[kk][idx+nn] < end_select)
				nn++; 
			offset_aplus[kk][ii] = idx; 
			dim_aplus[kk][ii] = nn; 
			idx += nn; 
			begin_select = end_select; 
		}
		// RXi
		for (int ii=0; ii<n_vars; ii++)
		{
			end_select=begin_select + n_vars; 
			nn = 0; 
			while(idx+nn<dim[kk] && select[kk][idx+nn] < end_select)
				nn++; 
			offset_xi[kk][ii] = idx; 
			dim_xi[kk][ii] = nn; 
			idx += nn; 
			begin_select = end_select;  
		}
	}
	return true; 	
}
