/*
 * Copyright (C) 1996-2011 Daniel Waggoner
 *
 * This free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * It is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * If you did not received a copy of the GNU General Public License
 * with this software, see <http://www.gnu.org/licenses/>.
 */
  
#include "bmatrix.h"
#include "dw_error.h"
#include "blas_lapack.h"
#include "dw_std.h"

#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

/********************************************************************************
  An (n x m) matrix x is in row major format if x(i,j)=x[i*n+j] and is in column 
  major format if x(i,j)=x[i+j*m].
********************************************************************************/
    
/******************************************************************************/
/***************************** Uniary Operations ******************************/
/******************************************************************************/
/*
  Assumes:
   x : n-vector
   y : n-vector
   n : positive

  Results:
   x[i] = -y[i] for 0 <= i < n

  Returns:
   0 upon success

  Notes:
   x and y do not have to be distinct
*/
int bNegative(PRECISION *x, PRECISION *y, int n)
{
 while (--n >= 0) x[n]=-y[n];
 return NO_ERR;
}

/*
  Assumes:
   x : n-vector
   y : n-vector
   n : positive

  Results:
   x[i] = fabs(y[i]) for 0 <= i < n

  Returns:
   0 upon success

  Notes:
   x and y do not have to be distinct
*/
int bAbs(PRECISION *x, PRECISION *y, int n)
{
 while (--n >= 0) x[n]=fabs(y[n]);
 return NO_ERR;
}

/*
  Assumes:
   x : n x m matrix
   y : m x n matrix
   m : positive
   n : positive
   t : 0 or 1

  Results:
    x = y'
    
  Returns:
   0 upon success

  Notes:
   It t is one, both x and y are in column major format.  If t is zero, both x
   y are in row major format.
*/
int bTranspose(PRECISION *x, PRECISION *y, int m, int n, int t)
{
 int i, j, k;
 if (t)
   for (i=k=m*n-1; k >= 0; i--)
     for (j=i; j >= 0; j-=m)
       x[k--]=y[j];
 else
   for (i=k=m*n-1; k >= 0; i--)
     for (j=i; j >= 0; j-=n)
       x[k--]=y[j];
 return NO_ERR;
}

/*
  Assumes:
   x : m x m matrix
   m : positive

  Results:
   x = x'

  Returns:
   0 upon success.

  Notes:
   The major format (row or column) does not matter.  
*/
int bTransposeInPlace(PRECISION *x, int m)
{
 PRECISION tmp;
 int i, j;
 for (j=m*m-2; j > 0; j+=i-1)
   for (i=j-m+1; i >= 0; j--, i-=m)
     {
       tmp=x[i];
       x[i]=x[j];
       x[j]=tmp;
     }
 return NO_ERR;
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/******************************************************************************/
/***************************** Addition Routines ******************************/
/******************************************************************************/
/*
  Assumes:
   x : n vector
   y : n vector
   z : n vector
   n : positive

  Results:
   x = y + z

  Returns:
   0 upon success

  Notes:
   The vectors x, y and z do not have to be distinct
*/
int bAddV(PRECISION *x, PRECISION *y, PRECISION *z, int n)
{
 while (--n >= 0) x[n]=y[n]+z[n];
 return NO_ERR;
}

/*
  Assumes:
   x : n-vector
   y : n-vector
   z : n-vector
   n : positive

  Results
   x = y - z

  Returns:
   0 upon success

  Notes:
   The vectors x, y and z do not have to be distinct
*/
int bSubtractV(PRECISION *x, PRECISION *y, PRECISION *z, int n)
{
 while (--n >= 0) x[n]=y[n]-z[n];
 return NO_ERR;
}

/*
  Assumes:
   x : m x n matrix
   y : m x n matrix
   z : m x n matrix
   m : positive
   n : positive
   xt: 0 or 1
   yt: 0 or 1
   zt: 0 or 1

  Results:
   x = y + z

  Returns:
   0 upon success

  Notes:
   The matrices y and z do not have to be distinct.  If xt, yt, or zt is 0 then 
   the matrix is in row major format.  If xt, yt, or zt is 1 then the matrix is 
   in column major format.  
*/
int bAddM(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int xt, int yt, int zt)
{
  int i, j, k, s;
  if (xt == yt)
    if (yt == zt)
      for (k=m*n-1; k >= 0; k--) x[k]=y[k]+z[k];
    else
      for (s=zt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[k]+z[j];
  else
    if (yt == zt)
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[j]+z[j];
    else
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[j]+z[k];
 return NO_ERR;
}

/*
  Assumes:
   x : m x n matrix
   y : m x n matrix
   z : m x n matrix
   m : positive
   n : positive
   xt: 0 or 1
   yt: 0 or 1
   zt: 0 or 1

  Results:
   x = y - z

  Returns:
   0 upon success

  Notes:
   The matrices y and z do not have to be distinct.  If xt, yt, or zt is 0 then 
   x, y, or z is in row major format.  If xt, yt, or zt is 1 then x, y, or z is 
   in column major format.  
*/
int bSubtractM(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int xt, int yt, int zt)
{
  int i, j, k, s;
  if (xt == yt)
    if (yt == zt)
      for (k=m*n-1; k >= 0; k--) x[k]=y[k]-z[k];
    else
      for (s=zt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[k]-z[j];
  else
    if (yt == zt)
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[j]-z[j];
    else
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=y[j]-z[k];
 return NO_ERR;
}

/*
  Assumes:
   x : m vector
   a : scalar
   y : m vector
   b : scalar
   z : m vector
   m : positive

  Results:
   x = a*y + b*z

  Returns:
   0 upon success

  Notes:
   The vectors x, y and z do not have to be distinct
*/
int bLinearCombinationV(PRECISION *x, PRECISION a, PRECISION *y, PRECISION b, PRECISION *z, int m)
{
  while (--m >= 0) x[m]=a*y[m]+b*z[m];
  return NO_ERR;
}

/*
  Assumes:
   x : m x n matrix
   a : scalar
   y : m x n matrix
   b : scalar
   z : m x n matrix
   m : positive
   n : positive
   xt: 0 or 1
   yt: 0 or 1
   zt: 0 or 1

  Results:
   x = a*y + b*z

  Returns:
   0 upon success

  Notes:
   The vector y and z do not have to be distinct.  If xt, yt, or zt is 0 then x,
   y, or z is in row major format.  If xt, yt, or zt is 1 then x, y, or z is in 
   column major format.  
*/
int bLinearCombinationM(PRECISION *x, PRECISION a, PRECISION *y, PRECISION b, PRECISION *z, int m, int n, int xt, int yt, int zt)
{
  int i, j, k, s;
  if (xt == yt)
    if (yt == zt)
      for (k=m*n-1; k >= 0; k--) x[k]=a*y[k]+b*z[k];
    else
      for (s=zt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=a*y[k]+b*z[j];
  else
    if (yt == zt)
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=a*y[j]+b*z[j];
    else
      for (s=yt ? m : n, k=i=m*n-1; k >= 0; i--)
	for (j=i; j >= 0; k--, j-=s)
	  x[k]=a*y[j]+b*z[k];
 return NO_ERR;
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/******************************************************************************/
/******************************************************************************/
/******************************************************************************/
/*
  Assumes:
   x : n vector
   y : n vector
   n : positive

  Results:
   x[i] = s * y[i] for 0 <= i < n

  Returns:
   0 upon success

  Notes:
  x and y do not have to be distinct
*/
int bMultiply(PRECISION *x, PRECISION *y, PRECISION s, int n)
{
  while (--n >= 0) x[n]=s*y[n];
  return NO_ERR;
}

/*
  Assumes:
   x : m x n matrix
   y : m x p matrix
   z : p x n matrix 
   a : scalar
   b : scalar 
   m : positive integer
   n : positive integer
   p : positive integer
   xt: 0 or 1
   yt: 0 or 1
   zt: 0 or 1

  Results:
    x = a*x + b*y*z

  Returns:
   0 upon success

  Notes:
   If xt, yt, or zt is one, then x, y, or z is in column major format.  If xt, 
   yt, or zt is zero, then x, y, or z is in row major format.  The matrices
   y and z do not have to be distinct.
*/
int bProductMM_Update(PRECISION *x, PRECISION *y, PRECISION *z, PRECISION a, PRECISION b, int m, int n, int p, int xt, int yt, int zt)
{
#if PRECISION_SIZE == 4
  #define gemm sgemm
#else
  #define gemm dgemm
#endif
  char transy, transz;
  blas_int dy, dz, bm=m, bn=n, bp=p;
  if (xt) 
    {
      if (yt) {transy='N'; dy=bm;} else {transy='T'; dy=bp;}   
      if (zt) {transz='N'; dz=bp;} else {transz='T'; dz=bn;}   
      gemm(&transy,&transz,&bm,&bn,&bp,&b,y,&dy,z,&dz,&a,x,&bm); 
    }
  else
    {
      if (yt) {transy='T'; dy=bm;} else {transy='N'; dy=bp;}   
      if (zt) {transz='T'; dz=bp;} else {transz='N'; dz=bn;}   
      gemm(&transz,&transy,&bn,&bm,&bp,&b,z,&dz,y,&dy,&a,x,&bn);  
    }
  return NO_ERR;
#undef gemm
}

/*
  Assumes:
   x : m vector
   y : n vector
   z : m x n matrix 
   a : scalar
   b : scalar 
   m : positive integer
   n : positive integer
   zt: 0 or 1

  Results:
    x = a*x + b*z*y

  Returns:
   0 upon success

  Notes:
   If zt is one, then z is in column major format.  If zt is zero, then z is in 
   row major format.
*/
int bProductMV_Update(PRECISION *x, PRECISION *y, PRECISION *z, PRECISION a, PRECISION b, int m, int n, int zt)
{
#if PRECISION_SIZE == 4
  #define gemv sgemv
#else
  #define gemv dgemv
#endif
  char trans;
  blas_int inc=1, bm=m, bn=n;
  if (zt) 
    {
      trans='N';
      gemv(&trans,&bm,&bn,&b,z,&bm,y,&inc,&a,x,&inc);
    }
  else
    {
      trans='T'; 
      gemv(&trans,&bn,&bm,&b,z,&bn,y,&inc,&a,x,&inc);  
    }
  return NO_ERR;
#undef gemv
}

/*
  Assumes:
   x : m x n matrix 
   y : n x n triangular matrix
   m : positive integer
   n : positive integer
   u : 0 or 1
   xt: 0 or 1
   yt: 0 or 1

  Results:

      x = x*y

   x is overwritten with x*y.

  Returns:
   0 upon success

  Notes:
   If u is one, then y is upper triangular.  If u is zero, then y is lower 
   triangular.  If xt, yt, or zt is one, then x, y, or z is in column major 
   format.  If xt, yt, or zt is zero, then x, y, or z is in row major format.
*/
int bProductMT(PRECISION *x, PRECISION *y, int m, int n, int u, int xt, int yt)
{
#if PRECISION_SIZE == 4
  #define trmm strmm
#else
  #define trmm dtrmm
#endif

  char side, uplo, transa, diag='N';
  PRECISION alpha=1.0;
  blas_int bm=m, bn=n;

  if (xt) 
    {
      side='R';
      if (yt) {transa='N'; uplo=u ? 'U' : 'L';} else {transa='T'; uplo=u ? 'L' : 'U';}   
      trmm(&side,&uplo,&transa,&diag,&bm,&bn,&alpha,y,&bn,x,&bm);
    }
  else
    {
      side='L';
      if (yt) {transa='T'; uplo=u ? 'U' : 'L';} else {transa='N'; uplo=u ? 'L' : 'U';}
      trmm(&side,&uplo,&transa,&diag,&bn,&bm,&alpha,y,&bn,x,&bn);
    }
  return NO_ERR;

#undef trmm
}

/*
  Assumes:
   x : m x n matrix 
   y : m x m triangular matrix
   m : positive integer
   n : positive integer
   u : 0 or 1
   xt: 0 or 1
   yt: 0 or 1

  Results:

      x = y*x

   x is overwritten with y*x.

  Returns:
   0 upon success

  Notes:
   If u is one, then y is upper triangular.  If u is zero, then y is lower 
   triangular.  If xt, yt, or zt is one, then x, y, or z is in column major 
   format.  If xt, yt, or zt is zero, then x, y, or z is in row major format.
*/
int bProductTM(PRECISION *x, PRECISION *y, int m, int n, int u, int xt, int yt)
{
#if PRECISION_SIZE == 4
  #define trmm strmm
#else
  #define trmm dtrmm
#endif

  char side, uplo, transa, diag='N';
  PRECISION alpha=1.0;
  blas_int bm=m, bn=n;

  if (xt) 
    {
      side='L';
      if (yt) {transa='N'; uplo=u ? 'U' : 'L';} else {transa='T'; uplo=u ? 'L' : 'U';}   
      trmm(&side,&uplo,&transa,&diag,&bm,&bn,&alpha,y,&bm,x,&bm);
    }
  else
    {
      side='R';
      if (yt) {transa='T'; uplo=u ? 'U' : 'L';} else {transa='N'; uplo=u ? 'L' : 'U';}
      trmm(&side,&uplo,&transa,&diag,&bn,&bm,&alpha,y,&bm,x,&bn);
    }
  return NO_ERR;

#undef trmm
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/***************************** LU Decompositions ******************************/
/******************************************************************************/
/*
   Assumes
    b  : m x n matrix
    a  : m x m matrix 
    m  : positive
    n  : positive
    bt : 0 or 1
    at : 0 or 1

  Results:
    Solves

       a*x = b

    x over writes b.

   Returns
     NO_ERR   - success
     SING_ERR - a was singular to machine precision.

   Note:
     If bt or at is one, then b or a is in column major format.  If bt or at is
     zero, then b or a is in row major format.  If m is equal to n and bt is
     equal to at, then b and a do not have to distinct. 
*/
int bSolveLU(PRECISION *b, PRECISION *a, int m, int n, int bt, int at)
{
#if PRECISION_SIZE == 4
  #define gesv sgesv
#else
  #define gesv dgesv
#endif

  PRECISION *pa, *pb;
  lapack_int info, *p, bm=m, bn=n;
  if (bt)
    pb=b;
  else
    {
      if (!(pb=(PRECISION*)dw_malloc(m*n*sizeof(PRECISION)))) return MEM_ERR;
      bTranspose(pb,b,m,n,bt);
    }
  pa=(PRECISION*)dw_malloc(m*m*sizeof(PRECISION));
  p=(lapack_int*)dw_malloc(m*sizeof(lapack_int));
  if (pa && p)
    {
      if (at)
	memcpy(pa,a,m*m*sizeof(PRECISION));
      else
	bTranspose(pa,a,m,m,at);
      gesv(&bm,&bn,pa,&bm,p,pb,&bm,&info);
      if (!bt)
	{
	  bTranspose(b,pb,m,n,1);
	  dw_free(pb);
	}
      dw_free(pa);
      dw_free(p);
      return (info < 0) ? BLAS_LAPACK_ERR : (info > 0) ? SING_ERR : NO_ERR;
    }
  else
    {
      if (pa) dw_free(pa);
      if (p) dw_free(p);
      if (b != pb) dw_free(pb);
      return MEM_ERR;
    }

#undef gesv
}

/*
   Assumes
    p  : integer array of length at least q=min(m,n)
    x  : mn matrix 
    m  : positive
    n  : positive
    xt : 0 or 1

  Results:
    Computes the LU decomposition of A with partial pivoting. The LU
    decomposition of a matrix A is

                                 A = P * L * U

    where P is a (m x m) permutation matrix, L is a (m x q) lower triangular
    matrix with ones on the diagonal, U is a (q x n) upper triangular matrix,
    and q=min(m,n).  These matrices are stored as follows.

      U is stored in the upper part of x, including the diagonal.

      L is stored in the lower part of x.  The diagonal of L is not stored.

      The matrix P is defined by

                   P = P(0,p[0])*P(1,p[1])*...*P(q-1,p[q-1])

      where P(r,s) is the (m x m) matrix obtained by permuting the rth and sth
      rows of the (m x m) identity matrix.  It is assumed that i <= p[i] < m.

   Returns
     NO_ERR - success
     BLAS_LAPACK_ERR - error in one of the arguments passed to getrf.
     SING_ERR - x was singular to machine precision.  LU decomposition is still
                computed and returned.

   Notes
     Only q elements of p are set.  The conventions of LAPACK uses base 1 arrays.  
     Thus to convert the permutation returned by getrf, 1 must be subtracted from 
     each element of p.  If xt is one, then x is in row major format as is the 
     returned matrix.  If xt is zero, then x is in column major format as is the 
     returned matrix.
*/
int bLU(int *p, PRECISION *x, int m, int n, int xt)
{
#if PRECISION_SIZE == 4
  #define getrf sgetrf
#else
  #define getrf dgetrf
#endif

   PRECISION *y;
   int i, q=(m < n) ? m : n;
   lapack_int *pp, info, bm=m, bn=n;
   if (!(pp=(lapack_int*)dw_malloc(q*sizeof(lapack_int)))) return MEM_ERR;

   if (xt)
     getrf(&bm,&bn,x,&bm,pp,&info);
   else
     {
       if (!( y=(PRECISION*)dw_malloc(m*n*sizeof(PRECISION)))) return MEM_ERR;
       bTranspose(y,x,m,n,0); 
       getrf(&bm,&bn,y,&bm,pp,&info);
       bTranspose(x,y,m,n,1);
       dw_free(y);
     }
   for (i=q-1; i >= 0; i--) p[i]=pp[i]-1;
   dw_free(pp);
   return (info < 0) ? BLAS_LAPACK_ERR : (info > 0) ? SING_ERR : NO_ERR;

#undef getrf
}

/*
   Assumes
     a  : m x m triangular matrix.
     b  : m x n matrix
     m  : positive integer
     n  : positive integer
     u  : 0 or 1
     at : 0 or 1
     bt : 0 or 1

   Results:
     Solves the system
 
        a*x = b

     The solution x is stored in b.

   Returns
     0 (NO_ERR) - success
     SING_ERR   - a is singular

   Notes
     The matrix a is assumed to be triangular.  If u is 1, then a is upper 
     triangular and if u is 0 then a is lower triangular.  The elements of a that
     should be zero are not accessed.

     If at or bt is 1, then a or b is in column major format.  If at or bt is 0,
     then a or b is in row major format.
*/
int bSolveTriangular(PRECISION *a, PRECISION *b, int m, int n, int u, int at, int bt)
{
#if PRECISION_SIZE == 4
  #define trsm strsm
#else
  #define trsm dtrsm
#endif

  PRECISION alpha=1.0;
  char diag='N', side, transa, uplo;
  int j;
  blas_int bi, bm=m, bn=n;

  // Is exactly singular - trsm does not check
  for (j=m*m-1; j >= 0; j-=m+1)
    if (a[j] == 0)
      return SING_ERR;

  if (bt)
    {
      bi=bm; side='L';
      if (at) { transa='N'; uplo=u ? 'U' : 'L'; } else { transa='T'; uplo=u ? 'L' : 'U';}
    }
  else
    {
      bi=bn; bn=bm; side='R';
      if (at) { transa='T'; uplo=u ? 'U' : 'L'; } else { transa='N'; uplo=u ? 'L' : 'U'; }
    }

  trsm(&side,&uplo,&transa,&diag,&bi,&bn,&alpha,a,&bm,b,&bi);

  return NO_ERR;

#undef trsm
}

/*
   Assumes
     a  : m x m triangular matrix with ones along the diagonal.
     b  : m x n matrix
     m  : positive integer
     n  : positive integer
     u  : 0 or 1
     at : 0 or 1
     bt : 0 or 1

   Results:
     Solves the system
 
        a*x = b

     The solution x is stored in b.

   Returns
     0 (NO_ERR) - success
     SING_ERR   - a is singular

   Notes
     The matrix a is assumed to be triangular.  If u is 1, then a is upper 
     triangular and if u is 0 then a is lower triangular.  The elements of a that
     should be zero or one are not accessed.

     If at or bt is 1, then a or b is in column major format.  If at or bt is 0,
     then a or b is in row major format.
*/
int bSolveUnitTriangular(PRECISION *a, PRECISION *b, int m, int n, int u, int at, int bt)
{
#if PRECISION_SIZE == 4
  #define trsm strsm
#else
  #define trsm dtrsm
#endif

  PRECISION alpha=1.0;
  char diag='U', side, transa, uplo;
  blas_int bi, bm=m, bn=n;

  if (bt)
    {
      bi=bm; side='L';
      if (at) { transa='N'; uplo=u ? 'U' : 'L'; } else { transa='T'; uplo=u ? 'L' : 'U';}
    }
  else
    {
      bi=bn; bn=bm; side='R';
      if (at) { transa='T'; uplo=u ? 'U' : 'L'; } else { transa='N'; uplo=u ? 'L' : 'U'; }
    }

  trsm(&side,&uplo,&transa,&diag,&bi,&bn,&alpha,a,&bm,b,&bi);

  return NO_ERR;

#undef trsm
}

/*
   Assumes
     x : m x n matrix
     y : m x n matrix
     p : integer array of length n representing a permutation.
     m : positive
     n : positive
     yt: 0 or 1
     pt: 0 or 1

   Results
     x = y * p  (pt == 0)
     x = y * p' (pt == 1)

   Returns
     NO_ERR

   Notes:
     An n-dimensional integer array x defines the permutation that maps i to
     x[i].  It must be the case that 0 <= x[i] < n for 0 <= i < n and that
     x[i] != x[j] for 0 <= i < j < n.  Note that (p*q)(i)=p(q(i)).

     If yt is one, then both x and y are in column major format.  If yt is zero
     then both x and y are in row major format.
*/
int bPermuteMatrix(PRECISION *x, PRECISION *y, int *p, int m, int n, int yt, int pt)
{
  int j, kx, ky;
  if (m == 1)
    if (pt)
      for (j=n-1; j >= 0; j--) x[p[j]]=y[j];
    else
      for (j=n-1; j >= 0; j--) x[j]=y[p[j]];
  else
    if (yt)
      if (pt)
	for (j=n-1; j >= 0; j--) 
	  memcpy(x+p[j]*m,y+j*m,m*sizeof(PRECISION));
      else
	for (j=n-1; j >= 0; j--) 
	  memcpy(x+j*m,y+p[j]*m,m*sizeof(PRECISION));
    else
      if (pt)
	for (m=(m-1)*n, j=n-1; j >= 0; j--)
	  for (ky=m+j, kx=m+p[j]; kx >= 0; ky-=n, kx-=n) x[kx]=y[ky];
      else
	for (m=(m-1)*n, j=n-1; j >= 0; j--)
	  for (ky=m+p[j], kx=m+j; kx >= 0; ky-=n, kx-=n) x[kx]=y[ky];
  return NO_ERR;
}

/*
   Assumes
     r : integer array of length n representing a permutation.
     p : integer array of length n representing a permutation.
     q : integer array of length n representing a permutation.
     n : positive
     pt: 0 or 1
     qt: 0 or 1

   Results
     r = p * q  (pt == 0 and qt == 0)
     r = p'* q  (pt == 1 and qt == 0)
     r = p * q' (pt == 0 and qt == 1)
     r = p'* q' (pt == 1 and qt == 1)

   Returns
     NO_ERR or MEM_ERR.

   Notes:
     An n-dimensional integer array x defines a permutation that maps i to x[i].
     It must be the case that 0 <= x[i] < n for 0 <= i < n and that x[i] != x[j]
     for 0 <= i < j < n.  Note that (p*q)(i)=p(q(i)).
*/
int bPermuteMultiply(int *r, int *p, int *q, int n, int pt, int qt)
{
  int i, *s;
  if (n == 1)
    r[0]=0;
  else
    if (pt)
      if (qt)
	for (i=n-1; i >= 0; i--) r[q[p[i]]]=i;
      else
	{
	  if (!(s=(int*)dw_malloc(n*sizeof(int)))) return MEM_ERR;
	  for (i=n-1; i >= 0; i--) s[p[i]]=i;
	  for (i=n-1; i >= 0; i--) r[i]=s[q[i]];
	  dw_free(s);
	}
    else
      if (qt)
	for (i=n-1; i >= 0; i--) r[q[i]]=p[i];
      else
	for (i=n-1; i >= 0; i--) r[i]=p[q[i]];
  return NO_ERR;
}

/*
   Assumes
     r : integer array of length n representing a permutation.
     p : integer array of length n representing a permutation.
     n : positive

   Results
     r = inverse(p).

   Returns
     NO_ERR

   Notes:
     An n-dimensional integer array x defines a permutation that maps i to x[i].  
     It must be the case that 0 <= x[i] < n for 0 <= i < n and that x[i] != x[j] 
     for 0 <= i < j < n.  Note that (p*q)(i)=p(q(i)).
*/
int bPermuteInverse(int *r, int *p, int n)
{
  for (--n; n >= 0; --n) r[p[n]]=n;
  return NO_ERR;
}

/*
   Assumes
     p : integer array of length q with 0 <= p[i] < m for all 0 <= i < q.
     y : m x n matrix
     m : positive
     n : positive
     q : positive
     pt: 0 or 1
     yt: 0 or 1

   Results
     (1) y = p * y  or  (2) y = p' * y

   Returns
     NO_ERR

   Notes:
     The array p defines a permutation matrix by

                    P = P(0,p[0])*P(1,p[1])*...*P(q-1,p[q-1])

     where P(r,s) is the m x m matrix obtained by permuting the rth and sth rows
     of the m x m identity matrix.  Note that 

                    P' = P(q-1,p[q-1])*...*P(0,p[0])*P(1,p[1])

     If yt is one, y is in column major format and if yt is zero, y is in row
     major format.  If pt is one, then y = p * y and if pt is zero then 
     y = p' * y.  
*/
int bPermutationMultiply(int *p, PRECISION *y, int m, int n, int q, int pt, int yt)
{
 int i, j, k, pk;
 PRECISION tmp;
 if (yt)
   if (pt)
     for (j=0; j < q; j++)
      {
       if (j != p[j])
        for (i=(n-1)*m; i >= 0; i-=m)
         {
          tmp=y[i+j];
          y[i+j]=y[i+p[j]];
          y[i+p[j]]=tmp;
         }
      }
    else
     for (j=q-1; j >= 0; j--)
      {
       if (j != p[j])
        for (i=(n-1)*m; i >= 0; i-=m)
         {
          tmp=y[i+j];
          y[i+j]=y[i+p[j]];
          y[i+p[j]]=tmp;
         }
      }
  else
   if (pt)
     for (i=0; i < q; i++)
      {
       if (i != p[i])
        {
         k=i*n;
         pk=p[i]*n;
         for (j=n-1; j >= 0; j--)
          {
           tmp=y[k+j];
           y[k+j]=y[pk+j];
           y[pk+j]=tmp;
          }
        }
      }
    else
     for (i=q-1; i >= 0; i--)
      {
       if (i != p[i])
        {
         k=i*n;
         pk=p[i]*n;
         for (j=n-1; j >= 0; j--) 
          {
           tmp=y[k+j];
           y[k+j]=y[pk+j];
           y[pk+j]=tmp;
          }
        }
      }
 return NO_ERR;
}

/*
   Assumes
     x : m x m matrix
     p : integer array of length q with i <= p[i] < m for all 0 <= i < m.
     m : positive
     q : positive
     xt: 0 or 1

   Results
     Computes the permutation matrix associated with the array p.

   Returns
     NO_ERR

   Notes
     The array p defines a permutation matrix by

                    P = P(0,p[0])*P(1,p[1])*...*P(q-1,p[q-1])

     where P(r,s) is the m x m matrix obtained by permuting the rth and sth rows
     of the m x m identity matrix.  If xt is one, x is in column major format and 
     if xt is zero, x is in row major format.
*/
int bPermutation(PRECISION *x, int *p, int m, int q, int xt)
{
  int i, j, k;
  for (k=m*m-1; k >= 0; k--) x[k]=0.0;
  if (xt)
    for (j=m-1; j >= 0; j--)
      {
	if (j < q)
	  {
	    k=j-1;
	    i=p[j];
	  }
	else
	  {
	    k=q-1;
	    i=j;
	  }
	for ( ; k >= 0; k--) if (i == p[k]) i=k;
	x[i+j*m]=1.0;
      }
  else
    for (j=m-1; j >= 0; j--)
      {
	if (j < q)
	  {
	    k=j-1;
	    i=p[j];
	  }
	else
	  {
	    k=q-1;
	    i=j;
	  }
	for ( ; k >= 0; k--) if (i == p[k]) i=k;
	x[i*m+j]=1.0;
      }
  return NO_ERR;
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/


/******************************************************************************/
/************************ Singular Value Decomposition ************************/
/******************************************************************************/
/*
   Assumes
    U       : array of length m*m (compact=0) or m*q (compact=1) or null 
    d       : array of length q=min(m,n)
    V       : array of length n*n (compact=0) or n*q (compact=1) or null 
    A       : array of length m*n
    m       : positive
    n       : positive
    ut      : 0 or 1
    vt      : 0 or 1
    at      : 0 or 1
    compact : 0 or 1

   Returns
     NO_ERR     : success
     MEM_ERR    : out of memory

   Results
     Finds matrices U and V with orthonormal columns and a diagonal matrix 
     D=diag(d) with non-negative diagonal such that A = U*D*V'.  The matrix D is 
     m x n if compact = 0 and is q x q if compact = 1.  The elements of d are in
     descending order.  The flags ut, vt, and at determine the format of U, V, 
     and A.  A value of 1 indicates column major format and a value of 0 
     indicates row major format.  If either U or V is null, then it is not 
     computed.

   Notes
     If A=U, U and A must be of the same size and ut=at.  If A=V, then V and A 
     must be of the same size and vt=at.  It cannot be the case that U=V unless
     both are null.

     New memory will always be allocated for A, so the contents of this array
     will not change.  A minimum amount of memory copying will occur if ut == 1
     and vt == 0, so that U is in column major format and V is in row major 
     format.
*/
int bSVD(PRECISION *U, PRECISION *d, PRECISION *V, PRECISION *A, int m, int n, int ut, int vt, int at, int compact)
{
#if (PRECISION_SIZE == 4) 
  #define gesvd sgesvd
#else
  #define gesvd dgesvd
#endif

  char jobu, jobv;
  int qu, qv, err=NO_ERR;
  lapack_int k, info, bqv, bm=m, bn=n;
  PRECISION  *A_, *U_, *V_, *work, opt_size;

  if (!(A_=(PRECISION*)dw_malloc(m*n*sizeof(PRECISION)))) return MEM_ERR;
  if (at)
    memcpy(A_,A,m*n*sizeof(PRECISION));
  else
    bTranspose(A_,A,m,n,at);

  if (!U)
    {
      jobu='N';
      qu=0;
      U_=(PRECISION*)NULL;
    }
  else
    {
      if (compact && (m > n))
	{
	  jobu='S';
	  qu=n;
	}
      else
	{
	  jobu='A';
	  qu=m;
	}
      if (ut)
	U_=U;
      else
	if (!(U_=(PRECISION*)dw_malloc(m*qu*sizeof(PRECISION))))
	  {
	    dw_free(A_);
	    return MEM_ERR;
	  }
    }

  if (!V)
    {
      qv=1;
      jobv='N';
      V_=(PRECISION*)NULL;
    }
  else
    {
      if (compact && (m < n))
	{
	  jobv='S';
	  qv=m;
	}
      else
	{
	  jobv='A';
	  qv=n;
	}
      if (!vt)
	V_=V;
      else
	if (!(V_=(PRECISION*)dw_malloc(n*qv*sizeof(PRECISION))))
	  {
	    dw_free(A_);
	    if (U_ && (U_ != U)) dw_free(U_); 
	    return MEM_ERR;
	  }
    }

  // compute singular value decomposition
  k=-1;
  bqv=qv;
  gesvd(&jobu,&jobv,&bm,&bn,A_,&bm,d,U_,&bm,V_,&bqv,&opt_size,&k,&info);
  if (info)
    err=BLAS_LAPACK_ERR;
  else
    if (!(work=(PRECISION*)dw_malloc((k=(lapack_int)opt_size)*sizeof(PRECISION))))
      err=MEM_ERR;
    else
      {
	gesvd(&jobu,&jobv,&bm,&bn,A_,&bm,d,U_,&bm,V_,&bqv,work,&k,&info);
	dw_free(work);
	if (info)
	  err=BLAS_LAPACK_ERR;
	else
	  {
	    if (U_ != U) bTranspose(U,U_,m,qu,1);
	    if (V_ != V) bTranspose(V,V_,qv,n,1);
	    err=NO_ERR;
	  }
      }

  dw_free(A_);
  if (U_ && (U_ != U)) dw_free(U_);
  if (V_ && (V_ != V)) dw_free(V_);
  return err;
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/***************************** QR Decompositions ******************************/
/******************************************************************************/
/*
   Assumes
     Q : m x q matrix or null pointer
     R : q x n
     X : m x n matrix 
     n : positive
     m : positive
     q : m or min(m,n)
    qt : 0 or 1
    rt : 0 or 1
    xt : 0 or 1

   Returns
     NO_ERR   : Success
     MEM_ERR  : Out of memory

   Results
     Finds an orthogonal matrix Q and an upper triangular matrix R
     such that

                                X = Q * R
     
     The matrix Q is computed only if it is not null.  

   Notes
     The matrices X and R do not have to be distinct.  If X == R, then it must
     be the case that m == q and rt == xt. 
*/
int bQR(PRECISION *Q, PRECISION *R, PRECISION *X, int m, int n, int q, int qt, int rt, int xt)
{
#if (PRECISION_SIZE == 4)
  #define geqrf sgeqrf
  #define orgqr sorgqr
  #define gelqf sgelqf
  #define orglq sorglq
#else
  #define geqrf dgeqrf
  #define orgqr dorgqr
  #define gelqf dgelqf
  #define orglq dorglq
#endif

  int i, j, k, l;
  lapack_int lwork, info, p=(m < n) ? m : n, bm=m, bn=n, bq=q;
  PRECISION *tau, *work, *ptr, opt_size;
  if (!(tau=(PRECISION*)dw_malloc(p*sizeof(PRECISION)))) return MEM_ERR;
  if (xt)
    {
      lwork=-1;

      geqrf(&bm,&bn,X,&bm,tau,&opt_size,&lwork,&info);

      if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)opt_size)*sizeof(PRECISION))))
      	{
      	  dw_free(tau);
      	  return MEM_ERR;
      	}

      geqrf(&bm,&bn,X,&bm,tau,work,&lwork,&info);

      /*********************************************/
      /*** MKL Problem *****************************/
      /*** does not seem to be in latest version ***
      if (info)
      	{
      	  printf("Unexpected error: info= %d  lwork= %d  m = %d  n = %d",info,lwork,m,n);
      	  dw_free(work);
      	  work=(PRECISION*)dw_malloc((lwork=(m > n) ? m*m : n*n)*64*sizeof(PRECISION));
      	  geqrf(&m,&n,X,&m,tau,work,&lwork,&info);
      	  printf("  minimal = %d\n",(int)work[0]);
      	}
      //else
      //  printf("lwork = %d  optimal= %d  m %d  n %d\n",lwork,(int)work[0],m,n);
      *********************************************/

      dw_free(work);
      if (info)
      	{
      	  dw_free(tau);
      	  return ARG_ERR;
      	}
      if (Q)
      	{
      	  if (qt)
      	    ptr=Q;
      	  else
      	    if (!(ptr=(PRECISION*)dw_malloc(m*q*sizeof(PRECISION))))
      	      {
      		dw_free(tau);
      		return MEM_ERR;
      	      }
      	  memcpy(ptr,X,m*p*sizeof(PRECISION));
      	  lwork=-1;

      	  orgqr(&bm,&bq,&p,ptr,&bm,tau,&opt_size,&lwork,&info);

      	  if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)opt_size)*sizeof(PRECISION))))
      	    {
      	      if (!qt) dw_free(ptr);
      	      dw_free(tau);
      	      return MEM_ERR;
      	    }

      	  orgqr(&bm,&bq,&p,ptr,&bm,tau,work,&lwork,&info);

      	  dw_free(work);
      	  if (!qt)
      	    {
      	      bTranspose(Q,ptr,m,q,1);
      	      dw_free(ptr);
      	    }
      	  dw_free(tau);
      	  if (info) return ARG_ERR;
      	}
      else
      	dw_free(tau);
      if (R != X)
      	if (rt)
      	  for (k=q*n, j=n-1; j >= 0; j--)
      	    {
      	      for (i=q-1; i > j; i--) R[--k]=0.0;
      	      for (l=i+j*m; i >= 0; i--) R[--k]=X[l--];
      	    }
      	else
      	  for (k=q*n, i=q-1; i >= 0; i--)
      	    {
      	      for (l=i+n*m, j=n-1; j >= i; j--) R[--k]=X[l-=m];
      	      for ( ; j >= 0; j--) R[--k]=0.0;
      	    }
      else
      	{
      	  for (j=p-1; j >= 0; j--)
      	    for (k=m*(j+1), i=m-1; i > j; i--) X[--k]=0.0;
      	}
    }
  else
    {
      lwork=-1;

      gelqf(&bn,&bm,X,&bn,tau,&opt_size,&lwork,&info);

      if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)opt_size)*sizeof(PRECISION))))
  	{
  	  dw_free(tau);
  	  return MEM_ERR;
  	}

      gelqf(&bn,&bm,X,&bn,tau,work,&lwork,&info);

      dw_free(work);
      if (info)
  	{
  	  dw_free(tau);
  	  return ARG_ERR;
  	}
      if (Q)
      	{
      	  if (!qt)
      	    ptr=Q;
      	  else
      	    if (!(ptr=(PRECISION*)dw_malloc(m*q*sizeof(PRECISION))))
      	      {
      		dw_free(tau);
      		return MEM_ERR;
      	      }
      	  if (q == n)
      	    memcpy(ptr,X,m*n*sizeof(PRECISION));
      	  else
      	    if (m < n)
      	      for (k=q*m, j=m-1; j >= 0; j--)
      		for (l=p+j*n, i=p-1; i >= 0; i--)
      		  ptr[--k]=X[--l];
      	    else
      	      for (l=n*m, j=m-1; j >= 0; j--)
      		for (k=p+j*q, i=p-1; i >= 0; i--)
      		  ptr[--k]=X[--l];
      	  lwork=-1;

      	  orglq(&bq,&bm,&p,ptr,&bq,tau,&opt_size,&lwork,&info);

      	  if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)opt_size)*sizeof(PRECISION))))
      	    {
      	      if (!qt) dw_free(ptr);
      	      dw_free(tau);
      	      return MEM_ERR;
      	    }

      	  orglq(&bq,&bm,&p,ptr,&bq,tau,work,&lwork,&info);

      	  dw_free(work);
      	  if (qt)
      	    {
      	      bTranspose(Q,ptr,q,m,1);
      	      dw_free(ptr);
      	    }
      	  dw_free(tau);
      	  if (info) return ARG_ERR;
      	}
      else
  	dw_free(tau);
      if (R != X)
  	if (rt)
  	  for (k=n*q, i=n-1; i >= 0; i--)
  	    {
  	      for (j=q-1; j > i; j--) R[--k]=0.0;
  	      for (l=i+j*n; j >= 0; l-=n, j--) R[--k]=X[l];
  	    }
  	else
          for (k=n*q-1, j=q-1; j >= 0; j--)
  	    {
  	      for (i=n-1; i >= j; k--, i--) R[k]=X[k];
  	      for ( ; i >= 0; k--, i--) R[k]=0.0;
  	    }
      else
  	{
  	  for (i=p-1; i >= 0; i--)
  	    for (k=i+m*n, j=m-1; j > i; j--) X[k-=n]=0.0;
  	}
    }
  return NO_ERR;

#undef geqrf 
#undef orgqr 
#undef gelqf 
#undef orglq 
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/********************** Generalized Schur Decompositions **********************/
/******************************************************************************/
/*
   Assumes
    Q       : array of length n*n or null
    Z       : array of length n*n or null
    S       : array of length n*n
    T       : array of length n*n
    A       : array of length n*n
    B       : array of length n*n
    n       : positive integer
    qt      : 0 or 1
    zt      : 0 or 1
    st      : 0 or 1
    tt      : 0 or 1
    at      : 0 or 1
    bt      : 0 or 1
    alpha_i : array of length n or null
    alpha_i : array of length n or null
    beta    : array of length n or null

   Returns
     NO_ERR          : success
     MEM_ERR         : out of memory
     BLAS_LAPACK_ERR : blas or lapack error

   Results
     Finds orthogonal matrices Q and Z, an block upper triangular matrix S with 
     1 x 1 or 2 x 2 blocks along the diagonal, an upper triangular matrix T such
     that

         A = Q*S*Z'   and    B = Q*T*Z
 
     If either Q or Z is null, then it is not returned.

   Notes
     The flags qt, zt, st, tt, at, and bt control the format of the matrices Q, 
     Z, S, T, A, and B.  A value of 1 indicate column major format and a value of
     0 indictes row major format.  All matrices must be distinct with the 
     exception A=S or B=T.
*/
int bGeneralizedSchur_real(PRECISION *Q, PRECISION *Z, PRECISION *S, PRECISION *T, PRECISION *A, PRECISION *B, 
			   int n, int qt, int zt, int st, int tt, int at, int bt,
			   PRECISION *alpha_r, PRECISION *alpha_i, PRECISION *beta)
{
#if (PRECISION_SIZE == 4) 
  #define gges sgges
#else
  #define gges dgges
#endif

  // not yet converted to lapack_int
  char jobvsl, jobvsr, sort='N';
  lapack_int lwork, simd, info, bn=n;
  int rtrn;
  PRECISION *work, size, *palpha_r, *palpha_i, *pbeta;

  jobvsl=Q ? 'V' : 'N';
  jobvsr=Z ? 'V' : 'N';
  palpha_r=alpha_r ? alpha_r : (PRECISION*)dw_malloc(n*sizeof(PRECISION)); 
  palpha_i=alpha_i ? alpha_i : (PRECISION*)dw_malloc(n*sizeof(PRECISION));
  pbeta=beta ? beta : (PRECISION*)dw_malloc(n*sizeof(PRECISION)); 

  if (palpha_r && palpha_i && pbeta)
    {
      if (S != A) 
	if (at)
	  memcpy(S,A,n*n*sizeof(PRECISION));
	else
	  bTranspose(S,A,n,n,0);
      else
	if (!at) bTransposeInPlace(A,n);

      if (T != B) 
	if (bt)
	  memcpy(T,B,n*n*sizeof(PRECISION));
	else
	  bTranspose(T,B,n,n,0);
      else
	if (!bt) bTransposeInPlace(B,n);

      lwork=-1;
      gges(&jobvsl,&jobvsr,&sort,(void*)NULL,&bn,S,&bn,T,&bn,&simd,palpha_r,palpha_i,pbeta,Q,&bn,Z,&bn,&size,&lwork,(void*)NULL,&info);
      if (!info)
	if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)size)*sizeof(PRECISION)))) 
	  rtrn=MEM_ERR;
	else
	  {
	    gges(&jobvsl,&jobvsr,&sort,(void*)NULL,&bn,S,&bn,T,&bn,&simd,palpha_r,palpha_i,pbeta,Q,&bn,Z,&bn,work,&lwork,(void*)NULL,&info);
	    if (!info)
	      {
		if (Q && !qt) bTransposeInPlace(Q,n);
		if (Z && !zt) bTransposeInPlace(Z,n);
		if (!st) bTransposeInPlace(S,n);
		if (!tt) bTransposeInPlace(T,n);
		rtrn=NO_ERR;
	      }
	    else
	      rtrn=BLAS_LAPACK_ERR;
	    dw_free(work);
	  }
      else
	rtrn=BLAS_LAPACK_ERR;
    }
  else
    rtrn=MEM_ERR;

  if (!alpha_r && palpha_r) dw_free(palpha_r);
  if (!alpha_i && palpha_i) dw_free(palpha_i);
  if (!beta && pbeta) dw_free(pbeta);

  return rtrn;

#undef gges
}

/*
   Assumes
    select  : array of length n
    QQ      : array of length n*n or null
    ZZ      : array of length n*n or null
    SS      : array of length n*n
    TT      : array of length n*n
    Q       : array of length n*n or null
    Z       : array of length n*n or null
    S       : array of length n*n
    T       : array of length n*n
    n       : positive integer 
    qqt     : 0 or 1
    zzt     : 0 or 1
    sst     : 0 or 1
    ttt     : 0 or 1
    qt      : 0 or 1
    zt      : 0 or 1
    st      : 0 or 1
    tt      : 0 or 1
    alpha_i : array of length n or null
    alpha_i : array of length n or null
    beta    : array of length n or null

   Returns
     NO_ERR          : success
     MEM_ERR         : out of memory
     BLAS_LAPACK_ERR : blas or lapack error

   Results
     Finds orthogonal matrices QQ and ZZ, an block upper triangular matrix SS 
     with 1 x 1 or 2 x 2 blocks along the diagonal, an upper triangular matrix TT 
     such that

              Q*S*Z' = QQ*SS*ZZ'   and    Q*T*Z' = QQ*TT*ZZ'
 
     If either Q or QQ are null, then QQ is not computed and if either Z or ZZ is
     null, then ZZ is not computed.  The matrices S and T are multiplied by  
     orthogonal matrices in such a manner that their block triangular structure
     is retained and the generalized eigenvalues corresponding to value of select
     equal to one are transformed to the upper part of SS and TT.   

   Notes
     The flags qqt, zzt, sst, ttt, qt, zt, st, and tt control the format of the 
     matrices QQ, ZZ, SS, TT, Q, Z, S, and T.  A value of 1 indicate column major 
     format and a value of 0 indictes row major format.

     There is a bug in the netlib version of lapack that occasionally requries 
     the matrix QQ passed to tgsen() to be an n*n matrix even when wantq is zero.
     Thus we always ensure that QQ is a n*n matrix.
*/
int bReorderGeneralizedSchur_real(int *select, PRECISION *QQ, PRECISION *ZZ, PRECISION *SS, PRECISION *TT, 
		    PRECISION *Q, PRECISION *Z, PRECISION *S, PRECISION *T, int n, int qqt, int zzt, int sst, int ttt, 
		    int qt, int zt, int st, int tt, PRECISION *alpha_r, PRECISION *alpha_i, PRECISION *beta)
{
#if (PRECISION_SIZE == 4) 
  #define tgsen stgsen
#else
  #define tgsen dtgsen
#endif

  int i, rtrn=NO_ERR;
  lapack_int ijob=0, wantq, wantz, lwork=-1, liwork=-1, isize, info, *iwork, *bselect, bm, bn=n;
  PRECISION size, *palpha_r, *palpha_i, *pbeta, *work, pl, pr, dif, *pQQ=QQ;

  wantq=(QQ && Q) ? 1 : 0;
  wantz=(ZZ && Z) ? 1 : 0;

  palpha_r=alpha_r ? alpha_r : (PRECISION*)dw_malloc(n*sizeof(PRECISION)); 
  palpha_i=alpha_i ? alpha_i : (PRECISION*)dw_malloc(n*sizeof(PRECISION));
  pbeta=beta ? beta : (PRECISION*)dw_malloc(n*sizeof(PRECISION)); 
  bselect=(lapack_int*)dw_malloc(n*sizeof(lapack_int));

  if (palpha_r && palpha_i && pbeta && bselect)
    {
      if (SS != S) 
	if (st)
	  memcpy(SS,S,n*n*sizeof(PRECISION));
	else
	  bTranspose(SS,S,n,n,0);
      else
	if (!st) bTransposeInPlace(S,n);

      if (TT != T) 
	if (tt)
	  memcpy(TT,T,n*n*sizeof(PRECISION));
	else
	  bTranspose(TT,T,n,n,0);
      else
	if (!tt) bTransposeInPlace(T,n);

      if (wantq)
	{
	  if (QQ != Q)
	    if (qt)
	      memcpy(QQ,Q,n*n*sizeof(PRECISION));
	    else
	      bTranspose(QQ,Q,n,n,0);
	  else
	    if (!qt) bTransposeInPlace(Q,n);
	}
      else
	if (!QQ) pQQ=(PRECISION*)dw_malloc(n*n*sizeof(PRECISION));

      if (wantz)
	if (ZZ != Z)
	  if (zt)
	    memcpy(ZZ,Z,n*n*sizeof(PRECISION));
	  else
	    bTranspose(ZZ,Z,n,n,0);
	else
	  if (!zt) bTransposeInPlace(Z,n);

      for (i=n-1; i >= 0; i--) bselect[i]=select[i];

      tgsen(&ijob,&wantq,&wantz,bselect,&bn,SS,&bn,TT,&bn,palpha_r,palpha_i,pbeta,pQQ,&bn,ZZ,&bn,&bm,
	    &pl,&pr,&dif,&size,&lwork,&isize,&liwork,&info);
      if (!info)
	if (!(work=(PRECISION*)dw_malloc((lwork=(int)size)*sizeof(PRECISION)))) 
	  rtrn=MEM_ERR;
	else
	  {
	    if (!(iwork=(lapack_int*)dw_malloc((liwork=isize)*sizeof(lapack_int))))
	      rtrn=MEM_ERR;
	    else
	      {
		tgsen(&ijob,&wantq,&wantz,bselect,&bn,SS,&bn,TT,&bn,palpha_r,palpha_i,pbeta,pQQ,&bn,ZZ,&bn,&bm,
		      &pl,&pr,&dif,work,&lwork,iwork,&liwork,&info);
		if (info)
		  rtrn=BLAS_LAPACK_ERR;
		dw_free(iwork);
	      }
	    dw_free(work);
	  }
      else
	rtrn=BLAS_LAPACK_ERR;

      if (wantq && !qqt) bTransposeInPlace(QQ,n);
      if (wantz && !zzt) bTransposeInPlace(ZZ,n);
      if (!sst) bTransposeInPlace(SS,n);
      if (!ttt) bTransposeInPlace(TT,n);

      if ((pQQ != QQ) && pQQ) dw_free(pQQ);
    }
  else
    rtrn=MEM_ERR;

  if (!alpha_r && palpha_r) dw_free(palpha_r);
  if (!alpha_i && palpha_i) dw_free(palpha_i);
  if (!beta && pbeta) dw_free(pbeta);
  if (bselect) dw_free(bselect);

  return rtrn;

#undef tgsen
}

/*
   Assumes
    QQ      : array of length n*n or null
    ZZ      : array of length n*n or null
    SS      : array of length n*n
    TT      : array of length n*n
    Q       : array of length n*n or null
    Z       : array of length n*n or null
    S       : array of length n*n
    T       : array of length n*n
    n       : positive integer
    qqt     : 0 or 1
    zzt     : 0 or 1
    sst     : 0 or 1
    ttt     : 0 or 1
    qt      : 0 or 1
    zt      : 0 or 1
    st      : 0 or 1
    tt      : 0 or 1
    alpha_i : array of length n
    alpha_i : array of length n
    beta    : array of length n

   Returns
     NO_ERR          : success
     MEM_ERR         : out of memory
     BLAS_LAPACK_ERR : blas or lapack error

   Results
     Finds orthogonal matrices QQ and ZZ, an block upper triangular matrix SS 
     with 1 x 1 or 2 x 2 blocks along the diagonal, an upper triangular matrix TT 
     such that

              Q*S*Z' = QQ*SS*ZZ'   and    Q*T*Z' = QQ*TT*ZZ'
 
     If either Q or QQ are null, then QQ is not computed and if either Z or ZZ is
     null, then ZZ is not computed.  The matrices S and T are multiplied by  
     orthogonal matrices in such a manner that their block triangular structure 
     retained and the generalized eigenvalues are sorted in descending order if 
     descend is one and ascending order if descend is zero.  So if descend is 
     one, then upon exit

           sqrt(alpha_r[i]^2 + alpha_i[i]^2)/fabs(beta[i]) 
                        >= sqrt(alpha_r[i+1]^2 + alpha_i[i+1]^2)/fabs(beta[i+1]) 

   Notes
     The flags qqt, zzt, sst, ttt, qt, zt, st, and tt control the format of the 
     matrices QQ, ZZ, SS, TT, Q, Z, S, and T.  A value of 1 indicate column major 
     format and a value of 0 indictes row major format.  All the matrices should be 
     distinct except that S=SS, T=TT, Q=QQ, or Z=ZZ.
*/
int bSortGeneralizedSchur_real(PRECISION *QQ, PRECISION *ZZ, PRECISION *SS, PRECISION *TT, PRECISION *Q, 
			       PRECISION *Z, PRECISION *S, PRECISION *T, int n, int qqt, int zzt, int sst, 
			       int ttt, int qt, int zt, int st, int tt, PRECISION *alpha_r, PRECISION *alpha_i, 
			       PRECISION *beta, int descend)
{
#if (PRECISION_SIZE == 4) 
  #define tgexc stgexc
#else
  #define tgexc dtgexc
#endif

  int rtrn=NO_ERR, k, i, j;
  lapack_int wantq, wantz, lwork=-1, info, ii=2, jj=1, bn=n;
  PRECISION tmp1, tmp2, size, *work;

  wantq=(QQ && Q) ? 1 : 0;
  wantz=(ZZ && Z) ? 1 : 0;

  if (SS != S) 
    if (st)
      memcpy(SS,S,n*n*sizeof(PRECISION));
    else
      bTranspose(SS,S,n,n,0);
  else
    if (!st) bTransposeInPlace(S,n);

  if (TT != T) 
    if (tt)
      memcpy(TT,T,n*n*sizeof(PRECISION));
    else
      bTranspose(TT,T,n,n,0);
  else
    if (!tt) bTransposeInPlace(T,n);

  if (wantq)
    if (QQ != Q)
      if (qt)
	memcpy(QQ,Q,n*n*sizeof(PRECISION));
      else
	bTranspose(QQ,Q,n,n,0);
    else
      if (!qt) bTransposeInPlace(Q,n);

  if (wantz)
    if (ZZ != Z)
      if (zt)
	memcpy(ZZ,Z,n*n*sizeof(PRECISION));
      else
	bTranspose(ZZ,Z,n,n,0);
    else
      if (!zt) bTransposeInPlace(Z,n);

  if (n > 1)
    {
      tgexc(&wantq,&wantz,&bn,SS,&bn,TT,&bn,QQ,&bn,ZZ,&bn,&ii,&jj,&size,&lwork,&info);
      if (!info)
	if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)size)*sizeof(PRECISION)))) 
	  rtrn=MEM_ERR;
	else
	  {
	    for (i=(SS[1] == 0.0) ? 1 : 2; i < n; )
	      for (j=i-1; j >= -1; )
		{
		  tmp1=descend ? fabs(beta[i])*sqrt(alpha_r[j]*alpha_r[j] + alpha_i[j]*alpha_i[j])
		    - fabs(beta[j])*sqrt(alpha_r[i]*alpha_r[i] + alpha_i[i]*alpha_i[i]) 
		    : fabs(beta[j])*sqrt(alpha_r[i]*alpha_r[i] + alpha_i[i]*alpha_i[i]) 
		    - fabs(beta[i])*sqrt(alpha_r[j]*alpha_r[j] + alpha_i[j]*alpha_i[j]);
		  if ((j == -1) || (0 <= tmp1))
		    {
		      ii=i+1;
		      jj=(++j)+1;
		      k=((i == n-1) || (SS[i*n+i+1] == 0.0)) ? 1 : 2;
		      if (jj < ii)
			{
			  tgexc(&wantq,&wantz,&bn,SS,&bn,TT,&bn,QQ,&bn,ZZ,&bn,&ii,&jj,work,&lwork,&info);
			  if (info)
			    {
			      dw_free(work);
			      rtrn=BLAS_LAPACK_ERR;
			      goto EXIT;
			    }
			  if (k == 1)
			    {
			      tmp1=alpha_r[i]; memmove(alpha_r+j+1,alpha_r+j,(i-j)*sizeof(PRECISION)); alpha_r[j]=tmp1;
			      tmp1=alpha_i[i]; memmove(alpha_i+j+1,alpha_i+j,(i-j)*sizeof(PRECISION)); alpha_i[j]=tmp1;
			      tmp1=beta[i]; memmove(beta+j+1,beta+j,(i-j)*sizeof(PRECISION)); beta[j]=tmp1;
			    }
			  else
			    {
			      tmp1=alpha_r[i]; tmp2=alpha_r[i+1]; memmove(alpha_r+j+2,alpha_r+j,(i-j)*sizeof(PRECISION)); 
			      alpha_r[j]=tmp1; alpha_r[j+1]=tmp2;
			      tmp1=alpha_i[i]; tmp2=alpha_i[i+1]; memmove(alpha_i+j+2,alpha_i+j,(i-j)*sizeof(PRECISION)); 
			      alpha_i[j]=tmp1; alpha_i[j+1]=tmp2;
			      tmp1=beta[i]; tmp2=beta[i+1]; memmove(beta+j+2,beta+j,(i-j)*sizeof(PRECISION)); 
			      beta[j]=tmp1; beta[j+1]=tmp2;
			    }
			}
		      i+=k;
		      break;
		    }
		  else
		    j-=((j == 0) || (SS[j*n+j-n] == 0.0)) ? 1 : 2;
		}
	    dw_free(work);
	  }
      else
	rtrn=BLAS_LAPACK_ERR;
    }

 EXIT:
  if (wantq && !qqt) bTransposeInPlace(QQ,n);
  if (wantz && !zzt) bTransposeInPlace(ZZ,n);
  if (!sst) bTransposeInPlace(SS,n);
  if (!ttt) bTransposeInPlace(TT,n);

  return rtrn;

#undef tgexc
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/************************** Cholesky Decompositions ***************************/
/******************************************************************************/
/*
   Assumes
     X : m x m symmetric positive definite matrix
     m : positive integer
     u : 0 or 1
     t : 0 or 1

   Returns
     0 (NO_ERR)  : success
     POS_DEF_ERR : X not positive definite

   Results
     Finds a triangular matrix T such that 
   
        X = T' * T

     If u is 1, T is upper triangular.  If u is 0, T is lower triangular.  
     T overwrites X.  It t is 1, X and T are in column major format.  If t 
     is 0, X and T are in row major format.

   Notes
     Failure usually indicates X is not positive definite.  Only half of X
     is accessed.  Currently, makes lapack calls only if u is one.
*/
int bCholesky(PRECISION *X, int m, int u, int t)
{
#if (PRECISION_SIZE == 4) 
  #define potrf spotrf
#else
  #define potrf dpotrf
#endif

  int i, j, k;
  lapack_int info, bm=m;
  PRECISION scale, *pX, *pXi, *pXj;

  if (u)
    {
      char uplo=t ? 'U' : 'L';
      potrf(&uplo,&bm,X,&bm,&info);
      if (!info)
	{
	  if (t)
	    for (i=m-1; i > 0; i--)
	      for (k=(i-1)*m+i; k >= 0; k-=m)
		X[k]=0.0;
	  else
	    for (i=m-1; i > 0; i--)
	      for (k=i*m+(j=i-1); j >= 0; j--)
		X[k--]=0.0;
	  return NO_ERR;
	} 
      else
	return (info < 0) ? BLAS_LAPACK_ERR : POSDEF_ERR;
    }
  else
    {
      if (!t)
	for (i=0, pXi=X; i < m; pXi++, i++)
	  {
	    for (j=0, pX=X+i*m; j < i; pX++, j++) *pX=0.0;

	    for (k=(i-1)*m; k >= 0; k-=m) *pX-=pXi[k]*pXi[k];

	    if (*pX <= 0.0) return POSDEF_ERR;
	    scale=1.0/(*pX=sqrt(*pX));

	    pXj=pXi;
	    for (j++; j < m; j++)
	      {
		pXj++;
		pX++;
		for (k=(i-1)*m; k >= 0; k-=m) *pX-=pXi[k]*pXj[k];
		*pX*=scale;
	      }
	  }
      else
	for (i=0, pXi=X; i < m; pXi+=m, i++)
	  {
	    for (j=0, pX=X+i; j < i; pX+=m, j++) *pX=0.0;
       
	    for (k=i-1; k >= 0; k--) *pX-=pXi[k]*pXi[k];

	    if (*pX <= 0.0) return POSDEF_ERR;
	    scale=1.0/(*pX=sqrt(*pX));

	    pXj=pXi;
	    for (j++; j < m; j++)
	      {
		pX+=m;
		pXj+=m;
		for (k=i-1; k >= 0; k--) *pX-=pXi[k]*pXj[k];
		*pX*=scale;
	      }
	  }
      return NO_ERR;
    }
#undef potrf
}


/******************************************************************************/
/************************** Eigen Values and Vectors **************************/ 
/******************************************************************************/
/*
   Assumes
     re : array of length n
     im : array of length n
     x  : n x n matrix
     n  : positive integer
     xt : 0 or 1

   Results
     re[k] + i*im[k] will be an eigenvalue of x

   Returns
     0 upon success and non-zero otherwise.
*/
int bEigenvalues(PRECISION *re, PRECISION *im, PRECISION *x, int n, int xt)
{
#if (PRECISION_SIZE == 4) 
  #define geev sgeev
#else
  #define geev dgeev
#endif
  char job='N';
  int rtrn;
  lapack_int bm=1, lwork=-1, info, bn=n;
  PRECISION size, *work, *tmp;

  if (!(tmp=(PRECISION*)dw_malloc(n*n*sizeof(PRECISION))))
    rtrn=MEM_ERR;
  else
    {
      if (xt)
	memcpy(tmp,x,n*n*sizeof(PRECISION));
      else
	bTranspose(tmp,x,n,n,0);
      geev(&job,&job,&bn,tmp,&bn,re,im,(PRECISION*)NULL,&bm,(PRECISION*)NULL,&bm,&size,&lwork,&info);
      if (info)
	rtrn=BLAS_LAPACK_ERR;
      else
	{
	  if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)size)*sizeof(PRECISION))))
	    rtrn=MEM_ERR;
	  else
	    {
	      geev(&job,&job,&bn,tmp,&bn,re,im,(PRECISION*)NULL,&bm,(PRECISION*)NULL,&bm,work,&lwork,&info);
	      if (info)
		rtrn=BLAS_LAPACK_ERR;
	      else
		rtrn=NO_ERR;
	      dw_free(work);
	    }
	}
      dw_free(tmp);
    }
  return rtrn;

#undef geev
}

/*
   Assumes
     reval : array of length n
     imval : array of length n
     revec : array of length n*n
     imvec : array of length n*n
     x     : array of length n*n
     n     : positive integer
     x_t   : 0 or 1
     vec_t : 0 or 1

   Results
     reval[k] + i*imval[k] will be an eigenvalue of x with eigenvector 
     (revec + i*imvec)*e(k), where e(k) is the kth column of the n x n identity 
     matrix.

   Returns
     0 upon success and non-zero otherwise.

   Notes
     If xt, revect, or imvect is 1, then x, revec, or imvec is in column major 
     form.  If xt , revect, or imvect is 0, then x, revec, or imvec is in row 
     major form.
*/
int bEigen(PRECISION *reval, PRECISION *imval, PRECISION *revec, PRECISION *imvec, PRECISION *x, int n, int xt, int revect, int imvect)
{
#if (PRECISION_SIZE == 4) 
  #define geev sgeev
#else
  #define geev dgeev
#endif

  char jobr='V', jobl='N';
  int i, j, k, rtrn=0;
  lapack_int bm=1, lwork=-1, info, bn=n;
  PRECISION *work, size, *tmp;

  if (!(tmp=(PRECISION*)dw_malloc(n*n*sizeof(PRECISION))))
    rtrn=MEM_ERR;
  else
    {
      if (xt)
	memcpy(tmp,x,n*n*sizeof(PRECISION));
      else
	bTranspose(tmp,x,n,n,0);
      geev(&jobl,&jobr,&bn,tmp,&bn,reval,imval,(PRECISION*)NULL,&bm,revec,&bn,&size,&lwork,&info);
      if (info)
	rtrn=BLAS_LAPACK_ERR;
      else
	{
	  if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)size)*sizeof(PRECISION))))
	    rtrn=MEM_ERR;
	  else
	    {
	      geev(&jobl,&jobr,&bn,tmp,&bn,reval,imval,(PRECISION*)NULL,&bm,revec,&bn,work,&lwork,&info);
	      if (info)
		rtrn=BLAS_LAPACK_ERR;
	      else
		{
		  for (j=0; j < n; j++)
		    if (imval[j] == 0.0)
		      for (k=j*n, i=n-1; i >= 0; i--) imvec[k+i]=0.0;
		    else
		      {
			k=(++j)*n;
			memcpy(imvec+k-n,revec+k,n*sizeof(PRECISION));
			for (i=n-1; i >= 0; i--) imvec[k+i]=-revec[k+i];
			memcpy(revec+k,revec+k-n,n*sizeof(PRECISION));
		      }
		  if (!revect) bTransposeInPlace(revec,n);
		  if (!imvect) bTransposeInPlace(imvec,n);
		  rtrn=NO_ERR;
		}
	    }
	  dw_free(work);
	}
      dw_free(tmp);
    }
  return rtrn;

#undef geev
}

/*
   Assumes
     val  : array of length n
     vec  : array of length n*n
     x    : array of length n*n
     n    : positive integer
     vect : 0 or 1

   Results
     val[k] will be an eigenvalue of x with eigenvector vec*e(k), where e(k) is 
     the kth column of the n x n identity matrix.

   Returns
     0 upon success and non-zero otherwise.

   Notes
     If vect is 1, then vec is in column major form.  If vect is 0, then vec is 
     in row major form.
*/
int bEigen_symmetric(PRECISION *val, PRECISION *vec, PRECISION *x, int n, int vect)
{
#if (PRECISION_SIZE == 4) 
  #define syevr ssyevr
#else
  #define syevr dsyevr
#endif

  char jobz=vec ? 'V' : 'N', range='A', uplo='U';
  int rtrn=0;
  lapack_int i, m, lwork=-1, *iwork, liwork=-1, isize, info, isuppz, bn=n;
  PRECISION *work, size, *A, v, abstol=-1.0;

  if (!(A=(PRECISION*)dw_malloc(n*n*sizeof(PRECISION))))
    rtrn=MEM_ERR;
  else
    {
      memcpy(A,x,n*n*sizeof(PRECISION));
      syevr(&jobz,&range,&uplo,&bn,A,&bn,&v,&v,&i,&i,&abstol,&m,val,vec,&bn,&isuppz,&size,&lwork,&isize,&liwork,&info);
      if (info)
	rtrn=BLAS_LAPACK_ERR;
      else
	{
	  if (!(work=(PRECISION*)dw_malloc((lwork=(lapack_int)size)*sizeof(PRECISION))))
	    rtrn=MEM_ERR;
	  else
	    {
	      if (!(iwork=(lapack_int*)dw_malloc((liwork=isize)*sizeof(lapack_int))))
		rtrn=MEM_ERR;
	      else
		{
		  syevr(&jobz,&range,&uplo,&bn,A,&bn,&v,&v,&i,&i,&abstol,&m,val,vec,&bn,&isuppz,work,&lwork,iwork,&liwork,&info);
		  if (info)
		    rtrn=BLAS_LAPACK_ERR;
		  else
		    {
		      if (!vect) bTransposeInPlace(vec,n);
		      rtrn=NO_ERR;
		    }
		  dw_free(iwork);
		}
	      dw_free(work);
	    }
	}
      dw_free(A);
    }
  return rtrn;
#undef syevr
}


/******************************************************************************/
/****************************** Tensor Products *******************************/
/******************************************************************************/
/*
   Assumes
       x     : array of length m*r*n*s
       y     : array of length m*n
       z     : array of length r*s
    m,n,r,s  : positive
    xt,yt,zt : 0 or 1

   Returns
     NO_ERR     : success

   Results
                        x          y          z       
       xt  yt  zt   (mr x ns)   (m x n)    (r x s)      computes
       ---------------------------------------------------------------------
        0   0   0   row major  row major  row major   x = y tensor z
        1   0   0   col major  row major  row major   x = y tensor z
        1   1   0   col major  col major  row major   x = y tensor z
        0   1   0   row major  col major  row major   x = y tensor z
        0   0   1   row major  row major  col major   x = y tensor z
        1   0   1   col major  row major  col major   x = y tensor z
        1   1   1   col major  col major  col major   x = y tensor z
        0   1   1   row major  col major  col major   x = y tensor z
*/
int bMatrixTensor(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int r, int s, int xt, int yt, int zt)
{
  int iy, jy, iz, jz, k, l, stride;
  PRECISION t, *pz=z+r*s-1;
  if (xt)
    if (zt)
      {
	stride=m*r;
	for (iy=m-1; iy >= 0; iy--)
	  for (jy=n-1; jy >= 0; jy--)
	    {
	      t=y[yt ? iy+m*jy : n*iy+jy];
	      l=(iy+1)*r-1 + ((jy+1)*s-1)*stride;
	      z=pz;
	      for (jz=s-1; jz >= 0; l-=stride, jz--)
		for (iz=r-1, k=l; iz >= 0; z--, k--, iz--)
		  x[k]=t*(*z);
	    }
      }
    else
      {
	stride=m*r;
	for (iy=m-1; iy >= 0; iy--)
	  for (jy=n-1; jy >= 0; jy--)
	    {
	      t=y[yt ? iy+m*jy : n*iy+jy];
	      l=(iy+1)*r-1 + ((jy+1)*s-1)*stride;
	      z=pz;
	      for (iz=r-1; iz >= 0; l--, iz--)
		for (jz=s-1, k=l; jz >= 0; z--, k-=stride, jz--)
		  x[k]=t*(*z);
	    }
      }
  else
    if (zt)
      {
	stride=n*s;
	for (iy=m-1; iy >= 0; iy--)
	  for (jy=n-1; jy >= 0; jy--)
	    {
	      t=y[yt ? iy+m*jy : n*iy+jy];
	      l=((iy+1)*r-1)*stride + (jy+1)*s-1;
	      z=pz;
	      for (jz=s-1; jz >= 0; l--, jz--)
		for (iz=r-1, k=l; iz >= 0; z--, k-=stride, iz--)
		  x[k]=t*(*z);

	    }
      }
    else
      {
	stride=n*s;
	for (iy=m-1; iy >= 0; iy--)
	  for (jy=n-1; jy >= 0; jy--)
	    {
	      t=y[yt ? iy+m*jy : n*iy+jy];
	      l=((iy+1)*r-1)*stride + (jy+1)*s-1;
	      z=pz;
	      for (iz=r-1; iz >= 0; l-=stride, iz--)
		for (jz=s-1, k=l; jz >= 0; z--, k--, jz--)
		  x[k]=t*(*z);

	    }
      }
  return NO_ERR;
}

int bVectorTensor(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n)
{
  int j, k;
  PRECISION s;
  for (x+=m*n-1, j=m-1; j >= 0; j--)
    for (s=y[j], k=n-1; k >= 0; x--, k--)
      *x=s*z[k];
  return NO_ERR;
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/******************************* Inner Product ********************************/ 
/******************************************************************************/
/*
  Assumes
   x - array of length n
   y - array of length n
   S - array of length n*n containing a real symmetric matrix
   n - positive integer

  Returns
   x'*S*y
*/
PRECISION bInnerProduct(PRECISION *x, PRECISION *y, PRECISION *S, int n)
{
  int i, j;
  PRECISION u, v, *px, *pS;
  if (x == y)
    for (u=(*x)*(*x)*(*S), i=n-1; i > 0; i--)
      {
	//for (v=(*((px=x)++))*(*((pS=S+i*n)++)), j=i-2; j >= 0; v+=(*(px++))*(*(pS++)), j--);
	for (v=(*(px=x))*(*(pS=S+i*n)), px++, pS++, j=i-2; j >= 0; v+=(*(px++))*(*(pS++)), j--);
	u+=(*px)*(2.0*v + (*px)*(*pS));
      }
  else
    for (pS=S+n*n-1, u=0.0, i=n-1; i >= 0; i--)
      {
	//for (v=(*((px=x+(n-1))--))*(*(pS--)), j=n-2; j >= 0; v+=(*(px--))*(*(pS--)), j--);
	for (v=(*(px=x+(n-1)))*(*pS), px--, pS--, j=n-2; j >= 0; v+=(*(px--))*(*(pS--)), j--);
	u+=y[i]*v;
      }
  return u;
}
