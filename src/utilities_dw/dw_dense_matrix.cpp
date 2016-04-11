#include "dw_dense_matrix.hpp"

#include "dw_rand.h"
#include "blas_lapack.h"

#include <string.h>
#include <math.h>
#include <climits>
#include <algorithm>

/*
  Upon entry:
   x : (c x r) matrix with major format determined by column_major
   y : (r x c) matrix with major format determined by column_major

   x and y must point to non-overlapping memory.

  Upon return:
   x = y'
*/
void TransposeMatrix(double *x, double *y, int r, int c, int column_major)
{
  int i, j, k;
  if (column_major)
    for (i=k=r*c-1; k >= 0; i--)
      for (j=i; j >= 0; k--, j-=r)
	x[k]=y[j];
  else
    for (i=k=r*c-1; k >= 0; i--)
      for (j=i; j >= 0; k--, j-=c)
	x[k]=y[j];
}

/*
  Upon entry:
   x : (cols x rows) complex matrix with major format determined by column_major
   y : (rows x cols) complex matrix with major format determined by column_major

   x and y must point to non-overlapping memory.

  Upon return:
   x = y'
*/
static void TransposeComplexMatrix(double *x, double *y, int r, int c, int column_major)
{
  int i, j, k;
  if (column_major)
    for (r*=2, i=k=r*c-2; k >= 0; i-=2)
      for (j=i; j >= 0; k-=2, j-=r)
	memcpy(x+k,y+j,2*sizeof(double));
  else
    for (c*=2, i=k=r*c-2; k >= 0; i-=2)
      for (j=i; j >= 0; k-=2, j-=c)
	memcpy(x+k,y+j,2*sizeof(double));;
}

/*
  Upon entry:
   x : (n x n) matrix in either row or column major format
   
  Upon return:
   x = x' with the same major format as upon entry.
*/
static void TransposeMatrix(double *x, int n)
{
  int i, j;
  double tmp;
  for (j=n*n-2; j > 0; j+=i-1)
    for (i=j-n+1; i >= 0; j--, i-=n)
      {
	tmp=x[i];
	x[i]=x[j];
	x[j]=tmp;
      }
}

/*
  Upon entry:
   x : (n x n) complex matrix in either row or column major format
   
  Upon return:
   x = x' with the same major format as upon entry.
*/
static void TransposeComplexMatrix(double *x, int n)
{
  int i, j;
  double tmp[2];
  for (j=2*n*n-4; j > 0; j+=2*(i-1))
    for (i=j-2*n+2; i >= 0; j-=2, i-=2*n)
      {
	memcpy(tmp,x+i,2*sizeof(double));
	memcpy(x+i,x+j,2*sizeof(double)); 
	memcpy(x+j,tmp,2*sizeof(double));
      }
}

/*
  Upon entry:
   d : array of length r
   M : array of length r*c
   r : positive integer
   c : positive integer
   cm: 1 => M is in column major format   0 => M is in row major format
   X : array of length r*c

  Upon exit:
   X = diag(d) * M with the same major format as M

  Notes:
   X and M can be equal.
*/
// static void DiagonalMatrixMultiply(double *d, double *M, int r, int c, int cm, double *X)
// {
//   int i, j, k;
//   if (cm)
//     {
//       for (k=r*r-1, i=r-1; i >= 0; k--, i--)
// 	for (j=k; j >= 0; j-=r) X[j]*=d[i];
//     }
//   else
//     {
//       for (k=r*r, i=r-1; i >= 0; i--)
// 	for (j=c; j > 0; j--) X[--k]*=d[i];
//     }
// }

/*
  Upon entry:
   d : array of length at least stride*(r-1)+1 with non-zero elements
   M : array of length at least r*c
   r : positive integer
   c : positive integer
   cm: 1 => M is in column major format   0 => M is in row major format
   X : array of length at least r*c

  Upon exit:
   X = inv(diag(d)) * M with the same major format as M

  Notes:
   X and M can be equal.  The elements of d MUST be non-zero.
*/
static void InverseDiagonalMatrixMultiply(double *d, int stride, double *M, int r, int c, int cm, double *X)
{
  double s;
  int i, j, k;
  if (cm)
    {
      for (k=r*c-1, i=stride*(r-1); i >= 0; k--, i-=stride)
	for (s=1.0/d[i], j=k; j >= 0; j-=r) X[j]=M[j]*s;
    }
  else
    {
      for (k=r*c-1, i=stride*(r-1); i >= 0; i-=stride)
	for (s=1.0/d[i], j=c; j > 0; k--, j--) X[k]=M[k]*s;
    }
}

/*******************************************************************************/
/******************************** Base Routines ********************************/
/*******************************************************************************/
// double* BaseMultiply(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)    matrix * matrix
// double* BaseMultiply(double s, double *v, int d, double *buffer)                                                  scalar * vector or scalar*matrix
// double* BaseMultiply(double *M, int r, int c, int cm, int *P, int d, int T, double *buffer)                       matix * permutation
// int* BaseMultiply(const TPermutationMatrix &P1, int T1, const TPermutationMatrix &P2, int T2, int *buffer)        permutation * permutation

// double* BaseAdd(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)         matrix + matrix
// double* BaseSubtract(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)    matrix - matrix
// double* BaseAdd(double *v1, int d1, double *v2, int d2, double *buffer)                                           vector + vector
// double* BaseSubtract(double *v1, int d1, double *v2, int d2, double *buffer)                                      vector - vector

// double* BaseDiagonalMatrix(int r, int c, double *v, int d, double *buffer)
// double* BaseDiagonalMatrix(int r, int c, double s, double *buffer)     

// double* BaseMinus(double *v, int d, double *buffer)
// double* BaseAbsoluteValue(double *v, int d, double *buffer)    

// void BaseSVD(double *M, int r, int c, int cm, double *U, double *d, double *V, int compact)
// void BaseQR(double *M, int r, int c, int cm, double *Q, double *R, int compact)      


//===============================================================================
//= The following block of routines are the base routines for copying and
//= inserting subvectors and submatrices.  There are two low level routines for
//= copying subblocks of matrices.  The following five routines are overloaded to
//= handle all possible methods for specifying indices.
//=   IndexArgumentsOK()
//=   BaseCopyVector()
//=   BaseInsertVector()
//=   BaseCopyMatrix()
//=   BaseInsertMatrix()
//===============================================================================
/*
   target : double array pointing to beginning of target block
   target_stride : number of rows if column major format and number of columns othewise
   target_column_major : true if target in column major format
   source : double array pointing to beginning of source block 
   source_stride : number of rows if column major format and number of columns othewise
   source_column_major : true if target in row major format
   r : number of rows to copy
   c : number of columns to copy

     target[target_column_major ? j*target_stride+i : i*target_stride+j] 
           =source[target_column_major ? j*target_stride+i : i*target_stride+j] 

   for 0 <= i < r and 0 <= j < c
*/
static void copy_matrix(double *target, int target_stride, bool target_column_major, double *source, int source_stride, bool source_column_major, int r, int c) 
{
  if ((target_stride == 1) && (source_stride == 1))
    memcpy(target,source,r*c*sizeof(double));
  else
    {
      if (!source_column_major) { int x=r; r=c; c=x; }
      if (target_column_major == source_column_major)
	{
	  if ((source_stride == r) && (target_stride == r))
	    memcpy(target,source,r*c*sizeof(double));
	  else
	    if (r > 1)
	      for (int j=c-1; j >= 0; j--) memcpy(target+j*target_stride,source+j*source_stride,r*sizeof(double));
	    else
	      if (c == 1)
		*target=*source;
	      else
		dcopy(&c,source,&source_stride,target,&target_stride);
	}
      else
	if ((target_stride == 1) || (source_stride == 1))
	  memcpy(target,source,r*c*sizeof(double));
	else
	  if (r > c)
	    for (int inc=1, j=c-1; j >= 0; j--)
	      dcopy(&r,source+j*source_stride,&inc,target+j,&target_stride);
	  else
	    for (int inc=1, i=r-1; i >= 0; i--)
	      dcopy(&c,source+i,&source_stride,target+i*target_stride,&inc);
    }
}

// Copies source vector into target vector
//         source[i*source_stride]=target[i*target_stride] for 0 <= i < n.
static void copy_vector(double *target, int target_stride, double *source, int source_stride, int n) 
{
  if ((target_stride == 1) && (source_stride == 1))
    memcpy(target,source,n*sizeof(double));
  else
    dcopy(&n,source,&source_stride,target,&target_stride);
}

// Returns 1 if arguments are valid and 0 otherwise
static int IndexArgumentsOK(int *idx, int n, int size)
{
  if (n < 0) return 0;
  while (--n >= 0) if ((idx[n] < 0) || (idx[n] >= size)) return 0;
  return 1;
}
static int IndexArgumentsOK(int b, int e, int size)
{
  if ((b <= e) && ((b < 0) || (e >= size))) return 0;
  return 1;
}
static int IndexArgumentsOK(int b1, int e1, int size1, int b2, int e2, int size2)
{
  if (e1 < b1) return (e2 < b2) ? 1 : 0;
  if ((e2 < b2) || (e1-b1 != e2-b2) || (b1 < 0) || (e1 >= size1) || (b2 < 0) || (e2 >= size2)) return 0;
  return 1;
}
static int IndexArgumentsOK(int *idx1, int n1, int size1, int b2, int e2, int size2)
{
  if (e2 < b2) return n1 ? 0 : 1;
  if ((n1 != e2-b2+1) || (b2 < 0) || (e2 >= size2)) return 0;
  while (--n1 >= 0) if ((idx1[n1] < 0) || (idx1[n1] >= size1)) return 0;
  return 1;
}
static int IndexArgumentsOK(int b1, int e1, int size1, int *idx2, int n2, int size2)
{
  if (e1 < b1) return n2 ? 0 : 1;
  if ((e1-b1+1 != n2) || (b1 < 0) || (e1 >= size1)) return 0;
  while (--n2 >= 0) if ((idx2[n2] < 0) || (idx2[n2] >= size2)) return 0;
  return 1;
}
static int IndexArgumentsOK(int *idx1, int n1, int size1, int *idx2, int n2, int size2)
{
  if ((n1 != n2) || (n1 < 0)) return 0;
  while (--n1 >= 0) if ((idx1[n1] < 0) || (idx1[n1] >= size1) || (idx2[n1] < 0) || (idx2[n1] >= size2)) return 0;
  return 1;
}

double* BaseCopyVector(int b, int e, double *input, int size, double *output)
{
  if (!IndexArgumentsOK(b,e,size)) throw dw_exception("BaseCopyVector(): invalid arguments");

  int n=e-b+1;
  if (n <= 0) return (double*)NULL;

  if ((SharedMemoryDimension_double(output) != n) || (SharedMemoryCounter_double(output) > 1))
    output=AllocateSharedMemory_double(n);

  if (output != input) memcpy(output,input+b,n*sizeof(double));
 
  return output;
}

double* BaseCopyVector(int *idx, int n, double *input, int size, double *output)
{
  if (!IndexArgumentsOK(idx,n,size)) throw dw_exception("BaseCopyVector(): invalid arguments");

  if (n == 0) return (double*)NULL;

  if ((SharedMemoryDimension_double(output) != n) || (SharedMemoryCounter_double(output) > 1) || (input == output))
    output=AllocateSharedMemory_double(n);

  for (int i=n-1; i >= 0; i--) output[i]=input[idx[i]];

  return output;
}

double* BaseInsertVector(int b, int e, double *input, int size, int _b, int _e, double *output)
{
  if (!IndexArgumentsOK(b,e,size,_b,_e,SharedMemoryDimension_double(output))) 
    throw dw_exception("BaseInsertVector(): invalid arguments");

  int n=e-b+1;
  if (n <= 0) return output;

  if (SharedMemoryCounter_double(output) > 1)
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  if (output != input)
    memcpy(output+_b,input+b,n*sizeof(double));
  else
    if (b != _b) memmove(output+_b,input+b,n*sizeof(double));

  return output;
}

double* BaseInsertVector(int *idx, int n, double *input, int size, int _b, int _e, double *output)
{
  if (!IndexArgumentsOK(idx,n,size,_b,_e,SharedMemoryDimension_double(output)))
    throw dw_exception("BaseInsertVector(): invalid arguments");

  if (n == 0) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=n-1; i >= 0; i--) output[_b+i]=input[idx[i]];

  return output;
}

double* BaseInsertVector(int b, int e, double *input, int size, int *_idx, int _n, double *output)
{
  if (!IndexArgumentsOK(b,e,size,_idx,_n,SharedMemoryDimension_double(output)))
    throw dw_exception("BaseInsertVector(): invalid arguments");

  if (_n == 0) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=_n-1; i >= 0; i--) output[_idx[i]]=input[b+i];

  return output;
}

double* BaseInsertVector(int *idx, int n, double *input, int size, int *_idx, int _n, double *output)
{
  if (!IndexArgumentsOK(idx,n,size,_idx,_n,SharedMemoryDimension_double(output)))
    throw dw_exception("BaseInsertVector(): invalid arguments");

  if (n == 0) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=n-1; i >= 0; i--) output[_idx[i]]=input[idx[i]];

  return output;
}

double* BaseCopyMatrix(int br, int er, int bc, int ec, double *input, int r, int c, bool column_major, double *output)
{
  if (!IndexArgumentsOK(br,er,r) || !IndexArgumentsOK(bc,ec,c))
    throw dw_exception("BaseCopyMatrix(): invalid arguments");

  int nr=er-br+1, nc=ec-bc+1;
  if ((nr <= 0) || (nc <= 0)) return (double*)NULL;

  if ((SharedMemoryDimension_double(output) != nc*nr) || (SharedMemoryCounter_double(output) > 1))
    output=AllocateSharedMemory_double(nc*nr);

  if (input == output) return output;

  copy_matrix(output,column_major ? nr : nc,column_major,input+(column_major ? bc*r+br : br*c+bc),column_major ? r : c,column_major,nr,nc);

  return output;
}

double* BaseCopyMatrix(int *idxr, int nr, int bc, int ec, double *input, int r, int c, bool column_major, double *output)
{
  if (!IndexArgumentsOK(idxr,nr,r) || !IndexArgumentsOK(bc,ec,c))
    throw dw_exception("BaseCopyMatrix(): invalid arguments");

  int nc=ec-bc+1;
  if ((nr <= 0) || (nc <= 0)) return (double*)NULL;

 if ((SharedMemoryDimension_double(output) != nc*nr) || (SharedMemoryCounter_double(output) > 1))
    output=AllocateSharedMemory_double(nc*nr);

  if (input == output) return output;

  if (column_major)
    for (int i=nr-1; i >= 0; i--)
      copy_vector(output+i,nr,input+bc*r+idxr[i],r,nc);
  else
    for (int i=nr-1; i >= 0; i--)
      copy_vector(output+i*nc,1,input+idxr[i]*c+bc,1,nc);

  return output;
}

double* BaseCopyMatrix(int br, int er, int *idxc, int nc, double *input, int r, int c, bool column_major, double *output)
{
  if (!IndexArgumentsOK(br,er,r) || !IndexArgumentsOK(idxc,nc,c))
    throw dw_exception("BaseCopyMatrix(): invalid arguments");

  int nr=er-br+1;
  if ((nr <= 0) || (nc <= 0)) return (double*)NULL;

  if ((SharedMemoryDimension_double(output) != nc*nr) || (SharedMemoryCounter_double(output) > 1))
    output=AllocateSharedMemory_double(nc*nr);

  if (input == output) return output;

  if (column_major)
    for (int j=nc-1; j >= 0; j--)
      copy_vector(output+j*nr,1,input+idxc[j]*r+br,1,nr);
  else
    for (int j=nc-1; j >= 0; j--)
      copy_vector(output+j,nc,input+br*c+idxc[j],c,nr);

  return output;
}

double* BaseCopyMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, double *output)
{
  if (!IndexArgumentsOK(idxr,nr,r) || !IndexArgumentsOK(idxc,nc,c))
    throw dw_exception("BaseCopyMatrix(): invalid arguments");

  if ((nr <= 0) || (nc <= 0)) return (double*)NULL;

  if ((SharedMemoryDimension_double(output) != nc*nr) || (SharedMemoryCounter_double(output) > 1))
    output=AllocateSharedMemory_double(nc*nr);

  if (input == output) return output;

  if (column_major)
    for (int k=nr*nc, j=nc-1; j >= 0; j--)
      for (int i=nr-1; i >=0; i--)
	output[--k]=input[idxc[j]*r+idxr[i]];
  else
    for (int k=nr*nc, i=nr-1; i >=0; i--)
      for (int j=nc-1; j >= 0; j--)
	output[--k]=input[idxr[i]*c+idxc[j]];

  return output;
}

// 0: * * * *
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_idxr,_nr,_r) || !IndexArgumentsOK(idxc,nc,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+_idxr[i] : _idxr[i]*_c+_idxc[j]]=input[column_major ? idxc[j]*r+idxr[i] : idxr[i]*c+idxc[j]];

  return output;
}

// 1: * * * i
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_idxr,_nr,_r) || !IndexArgumentsOK(idxc,nc,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? (_bc+j)*_r+_idxr[i] : _idxr[i]*_c+(_bc+j)]=input[column_major ? idxc[j]*r+idxr[i] : idxr[i]*c+idxc[j]];

  return output;
}

// 2: * * i *
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_br,_er,_r) || !IndexArgumentsOK(idxc,nc,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+(_br+i) : (_br+i)*_c+_idxc[j]]=input[column_major ? idxc[j]*r+idxr[i] : idxr[i]*c+idxc[j]];

  return output;
}

// 3: * * i i
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_br,_er,_r) || !IndexArgumentsOK(idxc,nc,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? (_bc+j)*_r+(_br+i) : (_br+i)*_c+(_bc+j)]=input[column_major ? idxc[j]*r+idxr[i] : idxr[i]*c+idxc[j]];
 
  return output;
}

// 4: * i * *
double* BaseInsertMatrix(int *idxr, int nr, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_idxr,_nr,_r) || !IndexArgumentsOK(bc,ec,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (_nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=_nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+_idxr[i] : _idxr[i]*_c+_idxc[j]]=input[column_major ? (bc+j)*r+idxr[i] : idxr[i]*c+(bc+j)];

  return output;
}

// 5: * i * i
double* BaseInsertMatrix(int *idxr, int nr, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_idxr,_nr,_r) || !IndexArgumentsOK(bc,ec,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nc=ec-bc+1;
  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >= 0; i--)
    copy_vector(output+(_column_major ? _bc*_r+_idxr[i] : _idxr[i]*_c+_bc), _column_major ? _r : 1, 
		input+(column_major ? bc*r+idxr[i] : idxr[i]*c+bc), column_major ? r : 1, nc);

  return output;
}

// 6: * i i *
double* BaseInsertMatrix(int *idxr, int nr, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_br,_er,_r) || !IndexArgumentsOK(bc,ec,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((nr <= 0) || (_nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >=0; i--)
    for (int j=_nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+(_br+i) : (_br+i)*_c+_idxc[j]]=input[column_major ? (bc+j)*r+idxr[i] : idxr[i]*c+(bc+j)];

  return output;
}

// 7: * i i i 
double* BaseInsertMatrix(int *idxr, int nr, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(idxr,nr,r,_br,_er,_r) || !IndexArgumentsOK(bc,ec,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nc=ec-bc+1;
  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=nr-1; i >= 0; i--)
    copy_vector(output+(_column_major ? _bc*_r+(_br+i) : (_br+i)*_c+_bc), _column_major ? _r : 1, 
		input+(column_major ? bc*r+idxr[i] : idxr[i]*c+bc), column_major ? r : 1, nc);

  return output;
}

// 8: i * * *
double* BaseInsertMatrix(int br, int er, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_idxr,_nr,_r) || !IndexArgumentsOK(idxc,nc,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((_nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=_nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+_idxr[i] : _idxr[i]*_c+_idxc[j]]=input[column_major ? idxc[j]*r+(br+i) : (br+i)*c+idxc[j]];

  return output;
}

// 9: i * * i
double* BaseInsertMatrix(int br, int er, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_idxr,_nr,_r) || !IndexArgumentsOK(idxc,nc,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((_nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=_nr-1; i >=0; i--)
    for (int j=nc-1; j >= 0; j--)
      output[_column_major ? (_bc+j)*_r+_idxr[i] : _idxr[i]*_c+(_bc+j)]=input[column_major ? idxc[j]*r+(br+i) : (br+i)*c+idxc[j]];

  return output;
}

// 10: i * i *
double* BaseInsertMatrix(int br, int er, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_br,_er,_r) || !IndexArgumentsOK(idxc,nc,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nr=er-br+1;
  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int j=nc-1; j >= 0; j--)
    copy_vector(output+(_column_major ? _idxc[j]*_r+_br : _br*_c+_idxc[j]), _column_major ? 1 : _c,
		input+(column_major ? idxc[j]*r+br : br*c+idxc[j]), column_major ? 1 : c, nr);

  return output;
}

// 11: i * i i
double* BaseInsertMatrix(int br, int er, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_br,_er,_r) || !IndexArgumentsOK(idxc,nc,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nr=er-br+1;
  if ((nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int j=nc-1; j >= 0; j--)
    copy_vector(output+(_column_major ? (_bc+j)*_r+_br : _br*_c+(_bc+j)), _column_major ? 1 : _c,
		input+(column_major ? idxc[j]*r+br : br*c+idxc[j]), column_major ? 1 : c, nr);

  return output;
}

// 12: i i * *
double* BaseInsertMatrix(int br, int er, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_idxr,_nr,_r) || !IndexArgumentsOK(bc,ec,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  if ((_nr <= 0) || (_nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=_nr-1; i >=0; i--)
    for (int j=_nc-1; j >= 0; j--)
      output[_column_major ? _idxc[j]*_r+_idxr[i] : _idxr[i]*_c+_idxc[j]]=input[column_major ? (bc+j)*r+(br+i) : (br+i)*c+(bc+j)];

  return output;
}

// 13: i i * i
double* BaseInsertMatrix(int br, int er, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_idxr,_nr,_r) || !IndexArgumentsOK(bc,ec,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nc=ec-bc+1;
  if ((_nr <= 0) || (nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int i=_nr-1; i >= 0; i--)
    copy_vector(output+(_column_major ? _bc*_r+_idxr[i] : _idxr[i]*_c+_bc), _column_major ? _r : 1,
		input+(column_major ? bc*r+(br+i) : (br+i)*c+bc), column_major ? r : 1, nc);

  return output;
}

// 14: i i i *
double* BaseInsertMatrix(int br, int er, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_br,_er,_r) || !IndexArgumentsOK(bc,ec,c,_idxc,_nc,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nr=er-br+1;
  if ((nr <= 0) || (_nc <= 0)) return output;

  if ((SharedMemoryCounter_double(output) > 1) || (output == input)) 
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }

  for (int j=_nc-1; j >= 0; j--)
    copy_vector(output+(_column_major ? _idxc[j]*_r+_br : _br*_c+_idxc[j]), _column_major ? 1 : _c,
		input+(column_major ? (bc+j)*r+br : br*c+(bc+j)), column_major ? 1 : c, nr);

  return output;
}

// 15: i i i i
double* BaseInsertMatrix(int br, int er, int bc, int ec, double *input, int r, int c, bool column_major, 
			 int _br, int _er, int _bc, int _ec, double *output, int _r, int _c, bool _column_major)
{
  if (!IndexArgumentsOK(br,er,r,_br,_er,_r) || !IndexArgumentsOK(bc,ec,c,_bc,_ec,_c))
    throw dw_exception("BaseInsertMatrix(): invalid arguments");

  int nr=er-br+1, nc=ec-bc+1;
  if ((nr <= 0) || (nc <= 0)) return output;

  if (SharedMemoryCounter_double(output) > 1)
    {
      double *original_output=output;
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,original_output,SharedMemoryDimension_double(output)*sizeof(double));
    }
  else if ((output == input) && (_br <= er) && (br <= _er) && (_bc <= ec) && (bc <= _ec))
    {
      output=AllocateSharedMemory_double(SharedMemoryDimension_double(output));
      memcpy(output,input,SharedMemoryDimension_double(output)*sizeof(double));
    }

  copy_matrix(output+(_column_major ? _bc*_r+_br : _br*_c+_bc), _column_major ? _r : _c, _column_major, 
	      input+(column_major ? bc*r+br : br*c+bc), column_major ? r : c, column_major, nr, nc);

  return output;
}
//===============================================================================
//===============================================================================
//===============================================================================

                       
// =================================== Added by HWu ==================================
double *BaseMultiply(int cm, double *v1, int d1, double *v2, int d2, double *buffer) //v1*v2';
//	cm=1 if data in buffer are column-majored; cm=0 otherwise
// 	v1: null pointer if d1==0 and array of length at least d1 otherwise
// 	d1: number of elements in v1 - must be non-negative
// 	v2: null pointer if d2==0 and array of length at least d2 otherwise
// 	d2: number of elements in v2 - must be non-negative
// 	buffer: member of the shared memory system 
{
	if ((SharedMemoryDimension_double(buffer) != d1*d2) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == v1) || (buffer == v2))
		buffer=AllocateSharedMemory_double(d1*d2);
	if (cm == 1)	// column major
	{
		for (int j=0; j<d2; j++)
			for (int i=0; i<d1; i++)
				buffer[i+j*d1] = v1[i]*v2[j]; 	
	}
	else 	// row major
	{
		for (int i=0; i<d1; i++)
			for (int j=0; j<d2; j++)
				buffer[i*d2+j] = v1[i]*v2[j]; 
	}
	return buffer;
}

double *BaseDotMultiply(double *v1, double *v2, int d, double *buffer) // buffer = v1 .* v2;
// 	v1: null pointer if d==0 or array of length at least d otherwise
// 	v2: null pointer if d==0 or array of length at least d otherwise
// 	buffer: member of the sared memmory system
{
	if ( (SharedMemoryDimension_double(buffer) != d) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == v1) || (buffer == v2) )
		buffer = AllocateSharedMemory_double(d); 
	for (int j=0; j<d; j++)
		buffer[j] = v1[j]*v2[j]; 
	return buffer; 
}

TDenseMatrix& TDenseMatrix::Multiply(const TDenseVector &v1, const TDenseVector &v2)
{
        ShareMemory(BaseMultiply(1,v1.vector,v1.dim,v2.vector,v2.dim,matrix),v1.dim, v2.dim, 1);
        return *this;
}
TDenseMatrix operator*(const TDenseVector &v1, const TDenseVector &v2)
{
	TDenseMatrix product; 
	product.Multiply(v1,v2); 
	return product; 
}
TDenseMatrix Multiply(const TDenseVector &v1, const TDenseVector &v2)
{
        TDenseMatrix product; 
	product.Multiply(v1,v2); 
	return product; 
}

TDenseVector & TDenseVector::DotMultiply(const TDenseVector &v1, const TDenseVector &v2)
{
	int dim = v1.dim <= v2.dim ? v1.dim : v2.dim; 
	ShareMemory(BaseDotMultiply(v1.vector,v2.vector,dim,vector),dim); 
	return *this; 
}

TDenseVector DotMultiply(const TDenseVector &v1, const TDenseVector &v2)
{
	TDenseVector dot_product; 
	dot_product.DotMultiply(v1,v2); 
	return dot_product; 
}

// ===================================================================================

/*
  Upon entry:
   M1  : null pointer if r1*c1 == 0 and array of length at least r1*c1 otherwise 
   r1  : number of rows - must be non-negative
   c1  : number of columns - must be non-negative
   cm1 : 1 => M1 is in column major format   0 => M1 is in row major format
   M2  : null pointer if r2*c2 == 0 and array of length at least r2*c2 otherwise 
   r2  : number of rows - must be non-negative
   c2  : number of columns - must be non-negative
   cm2 : 1 => M2 is in column major format  -  0 => M2 is in row major format
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the r1 by c2
   matrix M1*M2 in column major format.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if c1 != r2

  Notes:
   If buffer == M1 or buffer == M2, then buffer will be reallocated and the
   contents of M1 and M2 will not be changed.
*/
double* BaseMultiply(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)
{
  if (c1 != r2) 
    throw dw_exception("BaseMultiply(matrix,matrix): matrices not conformable");

  if ((SharedMemoryDimension_double(buffer) != r1*c2) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == M1) || (buffer == M2))
    buffer=AllocateSharedMemory_double(r1*c2);

  char transpose1, transpose2;
  int ld1, ld2, increment=1;
  double alpha=1.0, beta=0.0;
  if (M1 && M2) 
    if (r1 == 1)
      if (c2 == 1)
	{
	  //buffer[0]=M1[0]*M2[0];
	  //for (int i=c1-1; i > 0; i--) buffer[0]+=M1[i]*M2[i];
	  int inc=1;
	  buffer[0]=ddot(&c1,M1,&inc,M2,&inc);
	}
      else
	if (cm2) 
	  {
	    transpose2='T';
	    dgemv(&transpose2,&r2,&c2,&alpha,M2,&r2,M1,&increment,&beta,buffer,&increment);
	  }
	else
	  {
	    transpose2='N';
	    dgemv(&transpose2,&c2,&r2,&alpha,M2,&c2,M1,&increment,&beta,buffer,&increment);
	  }
    else
      if (c2 == 1)
	if (cm1) 
	  {
	    transpose1='N';
	    dgemv(&transpose1,&r1,&c1,&alpha,M1,&r1,M2,&increment,&beta,buffer,&increment);
	  }
	else
	  {
	    transpose1='T';
	    dgemv(&transpose1,&c1,&r1,&alpha,M1,&c1,M2,&increment,&beta,buffer,&increment);
	  }
      else
	{
	  if (cm1) { transpose1='N'; ld1=r1; } else { transpose1='T'; ld1=c1; }
	  if (cm2) { transpose2='N'; ld2=r2; } else { transpose2='T'; ld2=c2; }
	  dgemm(&transpose1,&transpose2,&r1,&c2,&c1,&alpha,M1,&ld1,M2,&ld2,&beta,buffer,&r1);  
	}
  else
    for (int i=r1*c2-1; i >= 0; i--) buffer[i]=0.0;

  return buffer;
}

/*
  Upon entry:
   M1  : null pointer if r1*c1 == 0 and array of length at least 2*r1*c1 otherwise 
   r1  : number of rows - must be non-negative
   c1  : number of columns - must be non-negative
   cm1 : 1 => M1 is in column major format   0 => M1 is in row major format
   M2  : null pointer if r2*c2 == 0 and array of length at least 2*r2*c2 otherwise 
   r2  : number of rows - must be non-negative
   c2  : number of columns - must be non-negative
   cm2 : 1 => M2 is in column major format  -  0 => M2 is in row major format
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the r1 by c2
   matrix M1*M2 in column major format.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if c1 != r2

  Notes:
   If buffer == M1 or buffer == M2, then buffer will be reallocated and the
   contents of M1 and M2 will not be changed.
*/
double* BaseMultiplyComplexComplex(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)
{
  if (c1 != r2) 
    throw dw_exception("BaseMultiply(matrix,matrix): matrices not conformable");

  if ((SharedMemoryDimension_double(buffer) != 2*r1*c2) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == M1) || (buffer == M2))
    buffer=AllocateSharedMemory_double(2*r1*c2);

  char transpose1, transpose2;
  int ld1, ld2, increment=1;
  double alpha=1.0, beta=0.0;
  if (M1 && M2) 
    if (r1 == 1)
      if (c2 == 1)
	{
	  int inc=1;
	  zdotu(buffer,&c1,M1,&inc,M2,&inc);
	}
      else
	if (cm2) 
	  {
	    transpose2='T';
	    zgemv(&transpose2,&r2,&c2,&alpha,M2,&r2,M1,&increment,&beta,buffer,&increment);
	  }
	else
	  {
	    transpose2='N';
	    zgemv(&transpose2,&c2,&r2,&alpha,M2,&c2,M1,&increment,&beta,buffer,&increment);
	  }
    else
      if (c2 == 1)
	if (cm1) 
	  {
	    transpose1='N';
	    zgemv(&transpose1,&r1,&c1,&alpha,M1,&r1,M2,&increment,&beta,buffer,&increment);
	  }
	else
	  {
	    transpose1='T';
	    zgemv(&transpose1,&c1,&r1,&alpha,M1,&c1,M2,&increment,&beta,buffer,&increment);
	  }
      else
	{
	  if (cm1) { transpose1='N'; ld1=r1; } else { transpose1='T'; ld1=c1; }
	  if (cm2) { transpose2='N'; ld2=r2; } else { transpose2='T'; ld2=c2; }
	  zgemm(&transpose1,&transpose2,&r1,&c2,&c1,&alpha,M1,&ld1,M2,&ld2,&beta,buffer,&r1);  
	}
  else
    for (int i=2*r1*c2-1; i >= 0; i--) buffer[i]=0.0;

  return buffer;
}

/*
  Upon entry:
   s      : double
   v      : array of length at least d or null pointer if d = 0
   d      : non-negative integer
   buffer : member of shared memory system

  Upon sucessful exit:
   The return value is a member of shared memory system of length d with 
   buffer[i] = s*v[i] for 0 <= i < d.

  Throws:
   std::bad_alloc if unable to allocate required memory

  Notes:
   If buffer == v, then the contents of v will be overwritten.
*/
double* BaseMultiply(double s, double *v, int d, double *buffer)
{
  if ((SharedMemoryDimension_double(buffer) != d) || (SharedMemoryCounter_double(buffer) > 1)) 
    buffer=AllocateSharedMemory_double(d);

  while (--d >= 0) buffer[d]=s*v[d];

  return buffer;
}

/*
  Upon entry:
   M  : null pointer if r*c == 0 and array of length at least r*c otherwise 
   r  : number of rows - must be non-negative.
   c  : number of columns - must be non-negative.
   cm : 1 => M is in column major format  -  0 => M is in row major format
   P  : null pointer if d == 0 and permutation of {0,...,d-1} otherwise
   d  : dimension - must be non-negative
   T  : 0 or 1
   buffer : member if shared memory system

  Upon successful exit:
   The return value is a member of shared memory system containing the r by c
   matrix M*P if T == 0 and M*P' if T == 1.  The return value will have the same 
   major format as M.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if c != d

  Notes:
   If buffer == M, then buffer will be reallocated and the contents of M will not 
   be changed.
*/
double* BaseMultiply(double *M, int r, int c, int cm, int *P, int d, int T, double *buffer)
{
  if (c != d) 
    throw dw_exception("BaseMultiply(matrix,permutation): matrices not conformable");

  if ((SharedMemoryDimension_double(buffer) != r*c) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == M))
    buffer=AllocateSharedMemory_double(r*c);

  if (buffer) 
    {
      int j, k, kM;
      if (r == 1)
	if (T)
	  for (j=c-1; j >= 0; j--) buffer[P[j]]=M[j];
	else
	  for (j=c-1; j >= 0; j--) buffer[j]=M[P[j]];
      else
	if (cm)
	  if (T)
	    for (j=c-1; j >= 0; j--) 
	      memcpy(buffer+P[j]*r,M+j*r,r*sizeof(double));
	  else
	    for (j=c-1; j >= 0; j--) 
	      memcpy(buffer+j*r,M+P[j]*r,r*sizeof(double));
	else
	  if (T)
	    for (r=(r-1)*c, j=c-1; j >= 0; j--)
	      for (kM=r+j, k=r+P[j]; k >= 0; kM-=c, k-=c) buffer[k]=M[kM];
	  else
	    for (r=(r-1)*c, j=c-1; j >= 0; j--)
	      for (kM=r+P[j], k=r+j; k >= 0; kM-=c, k-=c) buffer[k]=M[kM];
    }

  return buffer;
}

/*
  Upon entry:
   P1  : null pointer if d1 == 0 and permutation of {0,...,d1-1} otherwise
   d1  : dimension - must be non-negative
   T1  : 0 or 1
   P2  : null pointer if d2 == 0 and permutation of {0,...,d2-1} otherwise
   d2  : dimension - must be non-negative
   T2  : 0 or 1
   buffer : member if shared memory system

  Upon successful exit
   buffer = P1 * P2  if T1 == 0 and T2 == 0 
   buffer = P1'* P2  if T1 == 1 and T2 == 0 
   buffer = P1 * P2' if T1 == 0 and T2 == 1 
   buffer = P1'* P2' if T1 == 1 and T2 == 1 

  Throws:
   std:bad_alloc if unable to allocate required memory
   dw_exception if d1 != d2

  Notes:
   If buffer == P1 or buffer == p2, then buffer will be reallocated and the 
   contents of P1 and P2 will not be changed.
*/
int* BaseMultiply(int *P1, int d1, int T1, int *P2, int d2, int T2, int *buffer)
{
  if (d1 != d2) 
    throw dw_exception("BaseMultiply(permutation,permutation): matrices not conformable");

  int *permutation=buffer;

  if ((SharedMemoryDimension_int(buffer) != d1) || (SharedMemoryCounter_int(buffer) > 1) || (buffer == P1) || (buffer == P2))
    buffer=AllocateSharedMemory_int(d1);

  // Perform multipliation
  if (d1 == 1)
    buffer[0]=0;
  else
    if (T1)
      if (T2)
	for (int i=d1-1; i >= 0; i--) buffer[P2[P1[i]]]=i;
      else
	{
	  int *s;
	  if (!(s=new (std::nothrow) int[d1]))
	    {
	      if (buffer != permutation) FreeSharedMemory_int(buffer);
	      throw std::bad_alloc();
	    }
	  for (int i=d1-1; i >= 0; i--) s[P1[i]]=i;
	  for (int i=d1-1; i >= 0; i--) buffer[i]=s[P2[i]];
	  delete[] s;
	}
    else
      if (T2)
	for (int i=d1-1; i >= 0; i--) buffer[P2[i]]=P1[i];
      else
	for (int i=d1-1; i >= 0; i--) buffer[i]=P1[P2[i]];

  return buffer;
}

/*
  Upon entry:
   M1  : null pointer if r1*c1 == 0 and array of length at least r1*c1 otherwise 
   r1  : number of rows - must be non-negative
   c1  : number of columns - must be non-negative
   cm1 : true if M1 is in column major format
   M2  : null pointer if r2*c2 == 0 and array of length at least r2*c2 otherwise 
   r2  : number of rows - must be non-negative
   c2  : number of columns - must be non-negative
   cm2 : true if M2 is in column major format
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the r1*r2 x c1*c2
   matrix that is the Kronecker product of M1 and M2, with the same major format 
   as M2.

  Throws:
   std::bad_alloc if unable to allocate required memory
*/
double* BaseKron(double *M1, int r1, int c1, bool cm1, double *M2, int r2, int c2, bool cm2, double *buffer)
{
  if ((r1 == 0) || (r2 == 0) || (c1 == 0) || (c2 == 0)) return (double*)NULL;

  if ((SharedMemoryDimension_double(buffer) != r1*r2*c1*c2) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == M1) || (buffer == M2))
    buffer=AllocateSharedMemory_double(r1*r2*c1*c2);

  int m, stride;
  double t, *pM2=M2+r2*c2-1;
  if (cm2)
    {
      stride=r1*r2;
      for (int i1=r1-1; i1 >= 0; i1--)
	for (int j1=c1-1; j1 >= 0; j1--)
	  {
	    t=M1[cm1 ? i1+r1*j1 : c1*i1+j1];
	    m=(i1+1)*r2-1 + ((j1+1)*c2-1)*stride;
	    M2=pM2;
	    for (int j2=c2-1; j2 >= 0; m-=stride, j2--)
	      for (int i2=r2-1, k=m; i2 >= 0; M2--, k--, i2--)
		buffer[k]=t*(*M2);
	  }
    }
  else
    {
      stride=c1*c2;
      for (int i1=r1-1; i1 >= 0; i1--)
	for (int j1=c1-1; j1 >= 0; j1--)
	  {
	    t=M1[cm1 ? i1+r1*j1 : c1*i1+j1];
	    m=((i1+1)*r2-1)*stride + (j1+1)*c2-1;
	    M2=pM2;
	    for (int i2=r2-1; i2 >= 0; m-=stride, i2--)
	      for (int j2=c2-1, k=m; j2 >= 0; M2--, k--, j2--)
		buffer[k]=t*(*M2);
	  }
    }
  return buffer;
}

/*
  Upon entry:
   M1  : null pointer if r1*c1 == 0 and array of length at least r1*c1 otherwise 
   r1  : number of rows - must be non-negative
   c1  : number of columns - must be non-negative
   cm1 : 1 => M1 is in column major format   0 => M1 is in row major format
   M2  : null pointer if r2*c2 == 0 and array of length at least r2*c2 otherwise 
   r2  : number of rows - must be non-negative
   c2  : number of columns - must be non-negative
   cm2 : 1 => M2 is in column major format  -  0 => M2 is in row major format
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the r1 x r2
   matrix M1 + M2 with the same major format as M1.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if r1 != r2 or c1 != c2

  Notes:
   The routine is faster if M1 and M2 have the same major format.
*/
double* BaseAdd(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)
{
  if ((r1 != r2) || (c1 != c2)) 
    throw dw_exception("TDenseMatrix::BaseSubtract(): matrices not conformable");
  if (cm1 == cm2)
    {
      if ((SharedMemoryDimension_double(buffer) != r1*c1) || (SharedMemoryCounter_double(buffer) > 1))
	buffer=AllocateSharedMemory_double(r1*c1);
      for (int k=r1*c1-1; k >= 0; k--) buffer[k]=M1[k]+M2[k];
    }
  else
    {
      if ((SharedMemoryDimension_double(buffer) != r1*c1) || (SharedMemoryCounter_double(buffer) > 1) || ((buffer == M1) && (buffer == M2)))
	buffer=AllocateSharedMemory_double(r1*c1);
      if (buffer)
	{
	  int  stride, i, j, k;
	  for (stride=cm2 ? r2 : c2, k=i=r1*c1-1; k >= 0; i--)
	    for (j=i; j >= 0; k--, j-=stride)
	      buffer[k]=M1[k]+M2[j];
	}
    }
  return buffer;
}

/*
  Upon entry:
   M1  : null pointer if r1*c1 == 0 and array of length at least r1*c1 otherwise 
   r1  : number of rows - must be non-negative
   c1  : number of columns - must be non-negative
   cm1 : 1 => M1 is in column major format   0 => M1 is in row major format
   M2  : null pointer if r2*c2 == 0 and array of length at least r2*c2 otherwise 
   r2  : number of rows - must be non-negative
   c2  : number of columns - must be non-negative
   cm2 : 1 => M2 is in column major format  -  0 => M2 is in row major format
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the r1 x c1
   matrix M1 - M2 with the same major format as M1.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if r1 != r2 or c1 != c2

  Notes:
   The routine is faster if M1 and M2 have the same major format.
*/
double* BaseSubtract(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer)
{
  if ((r1 != r2) || (c1 != c2)) 
    throw dw_exception("TDenseMatrix::BaseSubtract(): matrices not conformable");
  if (cm1 == cm2)
    {
      if ((SharedMemoryDimension_double(buffer) != r1*c1) || (SharedMemoryCounter_double(buffer) > 1))
	buffer=AllocateSharedMemory_double(r1*c1);
      for (int k=r1*c1-1; k >= 0; k--) buffer[k]=M1[k]-M2[k];
    }
  else
    {
      if ((SharedMemoryDimension_double(buffer) != r1*c1) || (SharedMemoryCounter_double(buffer) > 1) || ((buffer == M1) && (buffer == M2)))
	buffer=AllocateSharedMemory_double(r1*c1);
      if (buffer)
	{
	  int  stride, i, j, k;
	  for (stride=cm2 ? r2 : c2, k=i=r1*c1-1; k >= 0; i--)
	    for (j=i; j >= 0; k--, j-=stride)
	      buffer[k]=M1[k]-M2[j];
	}
    }
  return buffer;
}

/*
  Upon entry:
   v1     : null pointer if d1 = 0 and array of length at least d1 otherwise
   d1     : non-negative integer
   v2     : null pointer if d2 = 0 and array of length at least d2 otherwise
   d2     : non-negative integer
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the d1
   dimensional vector v1 + v2.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if d1 != d2
*/
double* BaseAdd(double *v1, int d1, double *v2, int d2, double *buffer)
{
  if (d1 != d2) 
    throw dw_exception("TDenseVector::BaseAdd(): vectors not conformable");

  if ((SharedMemoryDimension_double(buffer) != d1) || (SharedMemoryCounter_double(buffer) > 1))  
    buffer=AllocateSharedMemory_double(d1);

  if (buffer) 
    for (int i=d1-1; i >= 0; i--) buffer[i]=v1[i]+v2[i];

  return buffer;
}

/*
  Upon entry:
   v1     : null pointer if d1 = 0 and array of length at least d1 otherwise
   d1     : non-negative integer
   v2     : null pointer if d2 = 0 and array of length at least d2 otherwise
   d2     : non-negative integer
   buffer : member of the shared memory system.

  Upon sucessful exit:
   The return value is a member of shared memory system containing the d1
   dimensional vector v1 - v2.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if d1 != d2
*/
double* BaseSubtract(double *v1, int d1, double *v2, int d2, double *buffer)
{
  if (d1 != d2) 
    throw dw_exception("TDenseVector::BaseAdd(): vectors not conformable");

  if ((SharedMemoryDimension_double(buffer) != d1) || (SharedMemoryCounter_double(buffer) > 1))  
    buffer=AllocateSharedMemory_double(d1);

  if (buffer) 
    for (int i=d1-1; i >= 0; i--) buffer[i]=v1[i]-v2[i];

  return buffer;
}

/*
  Upon entry:
   r      : non-negative integer
   c      : non-negative integer
   v      : null pointer if d1 = 0 and array of length at least d1 otherwise
   d      : non-negative integer
   buffer : member of the shared memory system.

  Upon success exit:
    The return value is a member of the shared memory system containing the r x c
    matrix, in column major format, with the vector v along the diagonal and 
    zeros off diagonal.  If v is shorter than minimum of r and c, then it is 
    padded with zeros.  If it is longer, then the last elements are ignored.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if r < 0 or c < 0
*/
double* BaseDiagonalMatrix(int r, int c, double *v, int d, double *buffer)
{
  if ((r < 0) || (c < 0)) 
    throw dw_exception("BaseDiagonalMatrix(): negative index");

  if ((SharedMemoryDimension_double(buffer) != r*c) || (SharedMemoryCounter_double(buffer) > 1) || (buffer == v))
    buffer=AllocateSharedMemory_double(r*c);

  if (buffer)
    {
      int i, stride=r+1, j=((r < c) ? ((r < d) ? r : d) : ((c < d) ? c : d)) - 1;
      for (i=r*c-1; i >= 0; i--) buffer[i]=0.0;
      for (i=j*stride; i >= 0; j--, i-=stride) buffer[i]=v[j];
    }

  return buffer;
}

/*
   Upon success, the returned array is r x c in column major format, with ones
   along the diagonal and zeros off diagonal.
*/
double* BaseDiagonalMatrix(int r, int c, double s, double *buffer)
{
  if ((r < 0) || (c < 0)) 
    throw dw_exception("BaseIdentitylMatrix(): negative index");

  if ((SharedMemoryDimension_double(buffer) != r*c) || (SharedMemoryCounter_double(buffer) > 1))
    buffer=AllocateSharedMemory_double(r*c);

  if (buffer)
    {
      int i, stride=r+1, j=((r < c) ? r : c) - 1;
      for (i=r*c-1; i >= 0; i--) buffer[i]=0.0;
      for (i=j*stride; i >= 0; j--, i-=stride) buffer[i]=s;
    }

  return buffer;
}

/*
  Upon entry:
   v      : array of length at least d or null pointer if d = 0
   d      : non-negative integer
   buffer : member of shared memory system

  Upon sucessful exit:
   The return value is a member of shared memory system of length d with 
   buffer[i] = -v[i] for 0 <= i < d.

  Throws:
   std::bad_alloc if unable to allocate required memory
*/
double* BaseMinus(double *v, int d, double *buffer)
{
  if ((SharedMemoryDimension_double(buffer) != d) || (SharedMemoryCounter_double(buffer) > 1)) buffer=AllocateSharedMemory_double(d);
  while (--d >= 0) buffer[d]=-v[d];
  return buffer;
}

/*
  Upon entry:
   v      : array of length at least d or null pointer if d = 0
   d      : non-negative integer
   buffer : member of shared memory system

  Upon sucessful exit:
   The return value is a member of shared memory system of length d with 
   buffer[i] = |v[i]| for 0 <= i < d.

  Throws:
   std::bad_alloc if unable to allocate required memory
*/
double* BaseAbsoluteValue(double *v, int d, double *buffer)
{
  if ((SharedMemoryDimension_double(buffer) != d) || (SharedMemoryCounter_double(buffer) > 1)) buffer=AllocateSharedMemory_double(d);
  while (--d >= 0) buffer[d]=fabs(v[d]);
  return buffer;
}

/*
   Assumes:
    M  : null pointer if r*c == 0 and array of length at least r*c otherwise 
    r  : number of rows - must be non-negative
    c  : number of columns - must be non-negative
    cm : 1 => M is in column major format   0 => M is in row major format
    U  : member shared memory system
    d  : member shared memory system
    V  : member shared memory system
    compact : zero or one

   Results:
    X = U * DiagonalMatrix(d,r,c) * V'  (compact == 0)
    X = U * DiagonalMatrix(d,p,p) * V'  (compact == 1)

   Notes:
    The vector d will be p-dimensional with non-negative elements in decending 
    order, where p is the minimum of r and c.  

    (1) The columns of U and V will be orthonormal.
    (2) If compact is one, then U will be (r x p) in column major format and V
        will be (c x p) in row major format.
    (3) If compact is zero, then U will be (r x r) in column major format and V 
        will be (c x c) in row major format.

    U, V, and d must be of the correct size and unique.    
*/
void BaseSVD(double *M, int r, int c, int cm, double *U, double *d, double *V, int compact)
{
  int p=(r < c) ? r : c, uc=compact ? p : r, vc=compact ? p : c;

  if ((SharedMemoryDimension_double(d) != p) || (SharedMemoryCounter_double(d) > 1)) goto ERROR;

  if (p)
    {
      char job;
      int *iwork, worksize, info=-15;
      double *M_, *work, opt_size;
      if (U && V)
	{
	  job=compact ? 'S' : 'A';
	  if ((SharedMemoryDimension_double(U) != r*uc) || (SharedMemoryCounter_double(U) > 1) || (SharedMemoryDimension_double(V) != c*vc) || (SharedMemoryCounter_double(V) > 1))
	    goto ERROR;
	}
      else if (!U && !V)
	job='N';
      else
	goto ERROR;
      if ((M_=new (std::nothrow) double[r*c]))
	{
	  if (cm)
	    memcpy(M_,M,r*c*sizeof(PRECISION));
	  else
	    TransposeMatrix(M_,M,r,c,0);
	  if ((iwork=new (std::nothrow) int[8*p]))
	    {
	      worksize=-1;
	      dgesdd(&job,&r,&c,M_,&r,d,U,&r,V,&vc,&opt_size,&worksize,iwork,&info);
	      if (!info)
		{
		  if ((work=new (std::nothrow) double[worksize=opt_size]))
		    {
		      dgesdd(&job,&r,&c,M_,&r,d,U,&r,V,&vc,work,&worksize,iwork,&info);
		      delete[] work;
		    }
		  else
		    info=-15;
		}
	      delete[] iwork;
	    }
	  delete[] M_;
	}
      if (info == -15)
	throw std::bad_alloc();
      else if (info < 0)
	throw dw_exception("BaseSVD(): BLAS/LAPACK argument error");
      else if (info > 0)
	throw dw_exception("BaseSVD(): Unable to compute SVD");
    }
  else
    {
      if (U)
	{
	  if ((SharedMemoryDimension_double(U) != r*uc) || (SharedMemoryCounter_double(U) > 1)) 
	    goto ERROR;
	  else
	    BaseDiagonalMatrix(r,r,1.0,U);
	}
      if (V)
	{
	  if ((SharedMemoryDimension_double(V) != c*vc) || (SharedMemoryCounter_double(V) > 1))
	    goto ERROR;
	  else
	    BaseDiagonalMatrix(c,c,1.0,V);
	}
    }

  return;

 ERROR:
  throw dw_exception("BaseSVD(): invalid arguments");
}

/*
   Assumes:
    M  : null pointer if r*c == 0 and array of length at least r*c otherwise 
    r  : number of rows - must be non-negative
    c  : number of columns - must be non-negative
    cm : 1 => M is in column major format   0 => M is in row major format
    Q  : member shared memory system
    R  : member shared memory system
    compact : zero or one

   Results:

           M = Q * R

    where Q has orthonormal columns and R is upper triangular.  If compact is
    one, then Q is r x min(r,c) and R is min(r,c) x c.  If compact is zero, then
    Q is r x r and R is r x c.  Both Q and R are will be column major format.

   Notes:
    Both Q and R must be of the correct size and unique.  If Q is null, it is not
    computed.    
*/
void BaseQR(double *M, int r, int c, int cm, double *Q, double *R, int compact)
{
  int i, j, k, m;
  int lwork, info=0, p=(r < c) ? r : c, q=compact ? p : r;
  PRECISION *X, *tau, *work, opt_size;

  // check sizes and uniqueness
  if ((SharedMemoryDimension_double(R) != q*c) || (SharedMemoryCounter_double(R) > 1) 
      || (Q && ((SharedMemoryDimension_double(Q) != r*q) || (SharedMemoryCounter_double(Q) > 1)))) 
    throw dw_exception("BaseQR(): invalid arguments");

  // if r or c is zero, then nothing to do
  if (!M) return;

  // allocate X if necessary and set elements
  if ((M == R) && (q == r) && (cm || (r == c)))
    {
      X=R;
      if (!cm) TransposeMatrix(X,r);
    }
  else
    {
      X=new double[r*c];
      if (cm)
	memcpy(X,M,r*c*sizeof(double));
      else
	TransposeMatrix(X,M,r,c,cm);
    }

  // allocate tau
  if (!(tau=new (std::nothrow) double[p])) goto ERROR_FREE_X;

  // get optimal workspace size
  lwork=-1;
  dgeqrf(&r,&c,X,&r,tau,&opt_size,&lwork,&info);
  if (info || !(work=new (std::nothrow) double[lwork=(int)opt_size])) goto ERROR_FREE_TAU;

  // compute QR decomposition
  dgeqrf(&r,&c,X,&r,tau,work,&lwork,&info);
  delete[] work;
  if (info) goto ERROR_FREE_TAU;

  // compute Q if necessary
  if (Q)
    {
      memcpy(Q,X,r*q*sizeof(PRECISION));

      // get optimal workspace size
      lwork=-1;
      dorgqr(&r,&q,&p,Q,&r,tau,&opt_size,&lwork,&info);
      if (info || !(work=new double[lwork=(int)opt_size])) goto ERROR_FREE_TAU;

      // compute Q
      dorgqr(&r,&q,&p,Q,&r,tau,work,&lwork,&info);
      delete[] work;
      delete[] tau;
      if (info) goto ERROR;
    }
  else
    delete[] tau;

  // fill R
  if (R != X)
    {
      for (k=q*c, j=c-1; j >= 0; j--)
	{
	  for (i=q-1; i > j; i--) R[--k]=0.0;
	  for (m=j*r+i; i >= 0; i--) R[--k]=X[m--];
	}
      delete[] X;
    }
  else
    {
      for (j=c-1; j >= 0; j--)
	for (i=q-1, k=j*q+i; i > j; i--) R[k--]=0.0;
    }
  return;

 ERROR_FREE_TAU:
  delete[] tau;
 ERROR_FREE_X:
  if (X != R) delete[] X;
 ERROR:
  if (info)
    throw dw_exception("BaseQR(): BLAS/LAPACK argument error");
  else
    throw std::bad_alloc();
}

/*
  Upon entry:
   M  : null pointer if r*c == 0 and array of length at least r*c otherwise 
   r  : number of rows - must be non-negative
   c  : number of columns - must be non-negative
   cm : 1 => M is in column major format   0 => M is in row major format
   buffer : member of the shared memory system.
   options : CHOLESKY_UPPER_TRIANGULAR or CHOLESKY_LOWER_TRIANGULAR

  Upon sucessful exit:
   The return value is a member of shared memory system such that

       M = buffer' * buffer

   If CHOLESKY_UPPER_TRIANGULAR is passed then buffer is upper triangular in 
   column major format.  If CHOLESKY_LOWER_TRIANGULAR is passed, then buffer is
   lower triangular in column major format.

  Throws:
   std::bad_alloc if unable to allocate required memory.
   dw_exception if r != c or M is not symmetric and positive definite.

  Notes:
   Uses LAPACK if CHOLESKY_UPPER_TRIANGULAR is passed.  The LAPACK call is much
   more efficient than the C++ code.
*/
double* BaseCholesky(double* M, int r, int c, int cm, double *buffer, int options)
{
  if (r != c)
    throw dw_exception("BaseCholesky(): number of rows and columns must be equal");

  double *original_buffer=buffer;

  if ((SharedMemoryDimension_double(buffer) != r*r) || (SharedMemoryCounter_double(buffer) > 1)) 
    buffer=AllocateSharedMemory_double(r*r);

  if (buffer)
    {
      if (options == CHOLESKY_UPPER_TRIANGULAR)
	{
	  char uplo='U';
	  int info;
	  if (buffer != M) memcpy(buffer,M,r*r*sizeof(double));
	  dpotrf(&uplo,&r,buffer,&r,&info);
	  if (info)
	    {
	      if (buffer != original_buffer) FreeSharedMemory_double(buffer);
	      if (info > 0)
		throw dw_exception("BaseCholesky(): matrix not positive definite");
	      else
		throw dw_exception("BaseCholesky(): lapack error");
	    }
	  // force zeros below diagonal (column major format)
	  for (int i=r*r-r-1; i >= 0; i-=r+1)
	    for (int j=i; j >= 0; j-=r) buffer[j]=0.0;
	}
      else
	{
	  int i, j, k;
	  double scale, *pX, *pXi, *pXj;
	  if (buffer !=M) memcpy(buffer,M,r*r*sizeof(double));
	  for (i=0, pXi=buffer; i < r; pXi+=r, i++)
	    {
	      for (j=0, pX=buffer+i; j < i; pX+=r, j++) *pX=0.0;     
	      for (k=i-1; k >= 0; k--) *pX-=pXi[k]*pXi[k];
	      if (*pX <= 0.0) 
		throw dw_exception("BaseCholesky(): matrix not positive definite");
	      scale=1.0/(*pX=sqrt(*pX));
	      pXj=pXi;
	      for (j++; j < r; j++)
		{
		  pX+=r;
		  pXj+=r;
		  for (k=i-1; k >= 0; k--) *pX-=pXi[k]*pXj[k];
		  *pX*=scale;
		}
	    }
	}
    }

  return buffer;
}

/*
   Assumes:
     x - array of length m

   Results:
     x is sorted in ascending order if ascending is true and in descending order
     otherwise.

   Notes:
     Uses the quick sort mean algorithm.  Switches to insertion sort when the
     size of the list is 10 or less.
*/
void BaseQuickSortArray(double *x, int m, bool ascending)
{
  double y, c;
  int j, k;
  if (ascending)  // ascending order
    {
      if (m > 10)
	{
	  // quick sort
	  m--;

	  if (x[0] == x[m])
	    c=x[0];
	  else
	    {
	      if (x[0] > x[m])
		{ y=x[m]; x[m]=x[0]; x[0]=y; }
	      c=0.5*(x[0] + x[m]);
	    }

	  for (j=1; (j < m) && (x[j] <= c); j++);
	  for (k=m-1; (k > 0) && (x[k] >= c); k--);
	  while (j < k)
	    {
	      y=x[j]; x[j]=x[k]; x[k]=y;
	      while (x[j] <= c) j++;
	      while (x[k] >= c) k--;
	    }
	  if (k > 0)
	    BaseQuickSortArray(x,k+1,ascending);
	  if (j < m)
	    BaseQuickSortArray(x+j,m-j+1,ascending);
	}
      else
	{
	  // insertion sort
	  for (j=1; j < m; j++)
	    {
	      y=x[j];
	      for (k=j-1; k >= 0; k--)
		if (x[k] <= y)
		  break;
		else
		  x[k+1]=x[k];
	      x[k+1]=y;
	    }
	}
    }
  else   // descending order
    {
      if (m > 10)
	{
	  // quick sort
	  m--;

	  if (x[0] == x[m])
	    c=x[0];
	  else
	    {
	      if (x[0] < x[m])
		{ y=x[m]; x[m]=x[0]; x[0]=y; }
	      c=0.5*(x[0] + x[m]);
	    }

	  for (j=1; (j < m) && (x[j] >= c); j++);
	  for (k=m-1; (k > 0) && (x[k] <= c); k--);
	  while (j < k)
	    {
	      y=x[j]; x[j]=x[k]; x[k]=y;
	      while (x[j] >= c) j++;
	      while (x[k] <= c) k--;
	    }
	  if (k > 0)
	    BaseQuickSortArray(x,k+1,ascending);
	  if (j < m)
	    BaseQuickSortArray(x+j,m-j+1,ascending);
	}
      else
	{
	  // insertion sort
	  for (j=1; j < m; j++)
	    {
	      y=x[j];
	      for (k=j-1; k >= 0; k--)
		if (x[k] >= y)
		  break;
		else
		  x[k+1]=x[k];
	      x[k+1]=y;
	    }
	}
    }
}

/*
   Assumes:
     x - array of length m*n in colum major format.
     m - number of rows
     n - number of columns
     idx - row on which to sort
     y - array of length m or null pointer.

   Results:
     The columns of x are sorted in ascending order on row idx.

   Notes:
     Uses the quick sort mean algorithm.  Switches to insertion sort when the
     size of the list is 10 or less.

     This is equivalent to sorting the rows of an n x m matrix in row major
     format on column idx.
*/
void BaseQuickSortMatrix(double *x, int m, int n, int idx, double *z, bool ascending)
{
  double *y=z ? z : new double[m], c;
  int j, k, p, s=m*sizeof(double);
  if (ascending)    // ascending order
    if (n > 10)
      {
	// quick sort
	p=(n-1)*m;
	k=p+idx;

	if (x[idx] == x[k])
	  c=x[idx];
	else
	  {
	    if (x[idx] > x[k])
	      { memcpy(y,x+p,s); memcpy(x+p,x,s); memcpy(x,y,s); }
	    c=0.5*(x[idx] + x[k]);
	  }

	for (j=m+idx; (j < p) && (x[j] <= c); j+=m);
	for (k-=m; (k > idx) && (x[k] >= c); k-=m);
	while (j < k)
	  {
	    memcpy(y,x+j-idx,s); memcpy(x+j-idx,x+k-idx,s); memcpy(x+k-idx,y,s);
	    while (x[j] <= c) j+=m;
	    while (x[k] >= c) k-=m;
	  }
	if (k > idx)
	  BaseQuickSortMatrix(x,m,(k-idx)/m+1,idx,y,ascending);
	if (j < p)
	  BaseQuickSortMatrix(x+j-idx,m,n-(j-idx)/m,idx,y,ascending);
      }
    else
      {
	// insertion sort
	p=n*m;
	for (j=m+idx; j < p; j+=m)
	  if (x[j-m] > x[j])
	    {
	      memcpy(y,x+j-idx,s);
	      memcpy(x+j-idx,x+j-m-idx,s);
	      for (k=j-m-m; k >= 0; k-=m)
		if (x[k] <= y[idx])
		  break;
		else
		  memcpy(x+k+m-idx,x+k-idx,s);
	      memcpy(x+k+m-idx,y,s);
	    }
      }
  else   // decending order
    if (n > 10)
      {
	// quick sort
	p=(n-1)*m;
	k=p+idx;

	if (x[idx] == x[k])
	  c=x[idx];
	else
	  {
	    if (x[idx] < x[k])
	      { memcpy(y,x+p,s); memcpy(x+p,x,s); memcpy(x,y,s); }
	    c=0.5*(x[idx] + x[k]);
	  }

	for (j=m+idx; (j < p) && (x[j] >= c); j+=m);
	for (k-=m; (k > idx) && (x[k] <= c); k-=m);
	while (j < k)
	  {
	    memcpy(y,x+j-idx,s); memcpy(x+j-idx,x+k-idx,s); memcpy(x+k-idx,y,s);
	    while (x[j] >= c) j+=m;
	    while (x[k] <= c) k-=m;
	  }
	if (k > idx)
	  BaseQuickSortMatrix(x,m,(k-idx)/m+1,idx,y,ascending);
	if (j < p)
	  BaseQuickSortMatrix(x+j-idx,m,n-(j-idx)/m,idx,y,ascending);
      }
    else
      {
	// insertion sort
	p=n*m;
	for (j=m+idx; j < p; j+=m)
	  if (x[j-m] < x[j])
	    {
	      memcpy(y,x+j-idx,s);
	      memcpy(x+j-idx,x+j-m-idx,s);
	      for (k=j-m-m; k >= 0; k-=m)
		if (x[k] >= y[idx])
		  break;
		else
		  memcpy(x+k+m-idx,x+k-idx,s);
	      memcpy(x+k+m-idx,y,s);
	    }
      }

  if (!z) delete[] y;
}

//============================ Added by HW =====================================
/* TDenseVector LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseVector &x)
 * Assumes:
 * 	T: Cholesky decomposition of a matrix M
 * 	x: a vector
 * Results 
 * 	y = LeftSolve(M, x)
 */
TDenseVector LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseVector &x)
{
	if (T.rows != T.cols)
		throw dw_exception("LeftSolve_Cholesky(): T must be square"); 
	if (T.rows != x.dim)
		throw dw_exception("LeftSolve_Cholesky(): T and x must have compatible dimensions"); 
	int cx=1; 
	
	TDenseVector y(x.dim); 
	for (int i=0; i<x.dim; i++)
		y[i] = x[i]; 
	char uplo='U', trans='T', diag='N'; 
	int info; 
	// y = T'\x;
	dtrtrs(&uplo, &trans, &diag, (int *)&(T.rows), &cx, T.matrix, (int *)&(T.rows), y.vector, (int *)&(y.dim), &info); 
	if (info > 0)
		throw dw_exception("LeftSolve_Cholesky(): T singular"); 
	else if (info < 0)
		throw dw_exception("LeftSolve_Cholesky(): BLAS/LAPACK argument error"); 

	trans='N';
	// y = T\y; 
	dtrtrs(&uplo, &trans, &diag, (int *)&(T.rows), &cx, T.matrix, (int *)&(T.rows), y.vector, (int *)&(y.dim), &info); 
	if (info > 0)
		throw dw_exception("LeftSolve_Cholesky(): T singular"); 
	else if (info < 0)
		throw dw_exception("LeftSolve_Cholesky(): BLAS/LAPACK argument error"); 
	return y; 
}

/* TDenseMatrix LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseVector &B)
 * Assumes:
 * 	T: Cholesky decomposition of a matrix M
 * 	B: a matrix
 * Results
 * 	A = LeftSolve(T, B)
 */ 	
TDenseMatrix LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseMatrix &B)
{
	if (T.rows != T.cols)
		throw dw_exception("LeftSolve_Cholesky(): T must be square"); 
	if (T.rows != B.rows)
		throw dw_exception("LeftSolve_Cholesky(): T and B must have compatible dimensions"); 
	
	TDenseMatrix A(B.rows, B.cols, true);	// A must be column major 
	for (int j=0; j<A.cols; j++)
	  for (int i=0; i<A.rows; i++)
			A(i,j) = B(i,j); 
	char uplo='U', trans='T', diag='N'; 
	int info; 
	// A = T'\B
	dtrtrs(&uplo, &trans, &diag, (int *)&(T.rows), (int *)&(B.cols), T.matrix, (int *)&(T.rows), A.matrix, (int *)&(A.rows), &info); 
	if (info > 0)
		throw dw_exception("LeftSolve_Cholesky(): T singular"); 
	else if (info < 0)
		throw dw_exception("LeftSolve_Cholesky(): BLAS/LAPACK argument error"); 
	trans = 'N'; 
	// A = T\A
	dtrtrs(&uplo, &trans, &diag, (int *)&(T.rows), (int *)&(A.cols), T.matrix, (int *)&(T.rows), A.matrix, (int *)&(A.rows), &info); 
	if (info > 0)
		throw dw_exception("LeftSolve_Cholesky(): T singular"); 
	else if (info < 0)
		throw dw_exception("LeftSolve_Cholesky(), BLAS/LAPACK argument error");  	

	return A; 
}

//============================ Added by HW =======================================

/*
  Assumes:
   A   : null pointer if ra*ca == 0 and array of length at least ra*ca otherwise 
   ra  : number of rows - must be non-negative
   ca  : number of columns - must be non-negative
   cma : 1 => A is in column major format   0 => A is in row major format
   B   : null pointer if rb*cb == 0 and array of length at least rb*cb otherwise 
   rb  : number of rows - must be non-negative
   cb  : number of columns - must be non-negative
   cmb : 1 => M is in column major format   0 => M is in row major format
   X   : member shared memory system
   method

  Results:
   The return value is a member of shared memory system containing the ra by cb
   matrix X in column major format which solves the system:

                                     A*X = B

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if ra != ca or ra != rb.

  Notes:
   If X == B, the contents of B may be overwritten.  
   If X == A, the contents of A will not be overwritten.
*/
double* BaseSolve(double *A, int ra, int ca, int cma, double *B, int rb, int cb, int cmb, double *X, int method)
{
  if ((ra != ca) || (ra != rb)) 
    throw dw_exception("BaseSolve(): matrices not comformable");

  if (!B) return (double*)NULL;

  double *original_X=X;
  switch (method)
    {
    case SOLVE_DIAGONAL:
      int i;
      double max, min, s;
      for (min=max=fabs(A[0]), i=ra*ra-1; i > 0; i-=ra+1)
      	if ((s=fabs(A[i])) > max) 
      	  max=s;
      	else if (s < min)
      	  min=s;
      if (min <= max*MACHINE_EPSILON) throw dw_exception("BaseSolve(): matrix singular");
      if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
      	X=AllocateSharedMemory_double(rb*cb);
      if (!cmb)
      	{
      	  TransposeMatrix(X,B,rb,cb,0);
      	  InverseDiagonalMatrixMultiply(A,ra+1,X,rb,cb,1,X);
      	}
      else
      	InverseDiagonalMatrixMultiply(A,ra+1,B,rb,cb,1,X);
      break;
    case SOLVE_UPPER_TRIANGULAR:
    case SOLVE_LOWER_TRIANGULAR:
      try
	{
	  char diag='N', uplo, trans;
	  int info;
	  if (cma)
	    { uplo=(method == SOLVE_UPPER_TRIANGULAR) ? 'U' : 'L'; trans='N'; }
	  else
	    { uplo=(method == SOLVE_UPPER_TRIANGULAR) ? 'L' : 'U'; trans='T'; }
	  if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
	    X=AllocateSharedMemory_double(rb*cb);
	  if (X != B)
	    {
	      if (cmb)
		memcpy(X,B,rb*cb*sizeof(double));
	      else
		TransposeMatrix(X,B,rb,cb,0);
	    }
	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,A,&ra,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	}
      catch (...)
	{
	  if (X != original_X) FreeSharedMemory_double(X);
	  throw;
	}
      break;
    case SOLVE_CHOLESKY:
      double *T;
      T=(double*)NULL;
      try
	{
	  T=BaseCholesky(A,ra,ra,cma,(double*)NULL,CHOLESKY_UPPER_TRIANGULAR);
          if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
	    X=AllocateSharedMemory_double(rb*cb);
          if (X != B)
	    {
	      if (cmb)
		memcpy(X,B,rb*cb*sizeof(double));
	      else
		TransposeMatrix(X,B,rb,cb,0);
	    }
	  char uplo='U', trans='T', diag='N';
	  int info;
	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,T,&ra,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else  if (info < 0)
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	  trans='N';
	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,T,&ra,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	  FreeSharedMemory_double(T);
	}
      catch (...)
	{
	  FreeSharedMemory_double(T);
	  if (X != original_X) FreeSharedMemory_double(X);
	  throw;
	}
      break;
    case SOLVE_QR:
      double *Q, *R;
      Q=R=(double*)NULL;
      try
	{
	  Q=AllocateSharedMemory_double(ra*ra);
	  R=AllocateSharedMemory_double(ra*ra);
	  BaseQR(A,ra,ra,cma,Q,R,1);
	  X=BaseMultiply(Q,ra,ra,0,B,rb,cb,cmb,X);
	  char uplo='U', trans='N', diag='N';
	  int info;
	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,R,&ra,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else if (info < 0) 
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	  FreeSharedMemory_double(Q);
	  FreeSharedMemory_double(R);
	}
      catch (...)
	{
	  FreeSharedMemory_double(Q);
	  FreeSharedMemory_double(R);
	  if (X != original_X) FreeSharedMemory_double(X);
	  throw;
	}
      break;
    case SOLVE_SVD:
      double *U, *V, *d, *Y;
      U=V=d=Y=(double*)NULL;
      try
    	{
    	  U=AllocateSharedMemory_double(ra*ra);
    	  V=AllocateSharedMemory_double(ra*ra);
    	  d=AllocateSharedMemory_double(ra);
    	  BaseSVD(A,ra,ra,cma,U,d,V,1);
    	  Y=BaseMultiply(U,ra,ra,0,B,rb,cb,cmb,(double*)NULL);
    	  if (d[ra-1] <= d[0]*MACHINE_EPSILON) throw dw_exception("BaseSolve(): matrix singular");
	  InverseDiagonalMatrixMultiply(d,1,Y,rb,cb,1,Y);
    	  X=BaseMultiply(V,ra,ra,0,Y,rb,cb,1,X);
    	  FreeSharedMemory_double(U);
    	  FreeSharedMemory_double(V);
    	  FreeSharedMemory_double(d);
    	  FreeSharedMemory_double(Y);
    	}
      catch (...)
    	{
    	  FreeSharedMemory_double(U);
    	  FreeSharedMemory_double(V);
    	  FreeSharedMemory_double(d);
    	  FreeSharedMemory_double(Y);
    	  if (X != original_X) FreeSharedMemory_double(X);
    	  throw;
    	}
      break;
    case SOLVE_LU:
      double *AA; int *pivot;
      AA=(double*)NULL; pivot=(int*)NULL;
      try
	{
	  int info;
	  AA=new double[ra*ra];
	  if (cma)
	    memcpy(AA,A,ra*ra*sizeof(double));
	  else
	    TransposeMatrix(AA,A,ra,ra,0);
	  if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || ((X == B) && !cmb))
	    X=AllocateSharedMemory_double(rb*cb);
	  if (X != B)
	    {
	      if (cmb)
		memcpy(X,B,rb*cb*sizeof(double));
	      else
		TransposeMatrix(X,B,rb,cb,0);
	    }
	  pivot=new int[ra];
	  dgesv(&ra,&cb,AA,&ra,pivot,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	  delete[] pivot;
	  delete[] AA;
	}
      catch (...)
	{
	  if (pivot) delete[] pivot;
	  if (AA) delete[] AA;
	  if (X != original_X) FreeSharedMemory_double(X);
	  throw;
	}
      break;
    default:
      throw dw_exception("BaseSolve: unknown solution technique");
    }

  return X;
}

/*
  Assumes:
   A   : null pointer if ra*ca == 0 and array of length at least 2*ra*ca otherwise 
   ra  : number of rows - must be non-negative
   ca  : number of columns - must be non-negative
   cma : 1 => A is in column major format   0 => A is in row major format
   B   : null pointer if rb*cb == 0 and array of length at least 2*rb*cb otherwise 
   rb  : number of rows - must be non-negative
   cb  : number of columns - must be non-negative
   cmb : 1 => M is in column major format   0 => M is in row major format
   X   : member shared memory system
   method

  Results:
   The return value is a member of shared memory system containing the ra by cb
   matrix X in column major format which solves the system:

                                     A*X = B

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if ra != ca or ra != rb.

  Notes:
   If X == B, the contents of B may be overwritten.  
   If X == A, the contents of A will not be overwritten.
*/
double* BaseSolveComplex(double *A, int ra, int ca, int cma, double *B, int rb, int cb, int cmb, double *X, int method)
{
  if ((ra != ca) || (ra != rb)) 
    throw dw_exception("BaseSolve(): matrices not comformable");

  if (!B) return (double*)NULL;

  double *original_X=X;
  switch (method)
    {
    case SOLVE_DIAGONAL:
      throw dw_exception("BaseSolveComplex(): not yet implemented");
      // int i;
      // double max, min, s;
      // for (min=max=fabs(A[0]), i=ra*ra-1; i > 0; i-=ra+1)
      // 	if ((s=fabs(A[i])) > max) 
      // 	  max=s;
      // 	else if (s < min)
      // 	  min=s;
      // if (min <= max*MACHINE_EPSILON) throw dw_exception("BaseSolve(): matrix singular");
      // if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
      // 	X=AllocateSharedMemory_double(rb*cb);
      // if (!cmb)
      // 	{
      // 	  TransposeMatrix(X,B,rb,cb,0);
      // 	  InverseDiagonalMatrixMultiply(A,ra+1,X,rb,cb,1,X);
      // 	}
      // else
      // 	InverseDiagonalMatrixMultiply(A,ra+1,B,rb,cb,1,X);
      break;
    case SOLVE_UPPER_TRIANGULAR:
    case SOLVE_LOWER_TRIANGULAR:
      throw dw_exception("BaseSolveComplex(): not yet implemented");
      // try
      // 	{
      // 	  char diag='N', uplo, trans;
      // 	  int info;
      // 	  if (cma)
      // 	    { uplo=(method == SOLVE_UPPER_TRIANGULAR) ? 'U' : 'L'; trans='N'; }
      // 	  else
      // 	    { uplo=(method == SOLVE_UPPER_TRIANGULAR) ? 'L' : 'U'; trans='T'; }
      // 	  if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
      // 	    X=AllocateSharedMemory_double(rb*cb);
      // 	  if (X != B)
      // 	    if (cmb)
      // 	      memcpy(X,B,rb*cb*sizeof(double));
      // 	    else
      // 	      TransposeMatrix(X,B,rb,cb,0);
      // 	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,A,&ra,X,&rb,&info);
      // 	  if (info > 0)
      // 	    throw dw_exception("BaseSolve(): matrix singular");
      // 	  else if (info < 0)
      // 	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
      // 	}
      // catch (...)
      // 	{
      // 	  if (X != original_X) FreeSharedMemory_double(X);
      // 	  throw;
      // 	}
      break;
    case SOLVE_CHOLESKY:
      throw dw_exception("BaseSolveComplex(): not yet implemented");
      // double *T;
      // T=(double*)NULL;
      // try
      // 	{
      // 	  T=BaseCholesky(A,ra,ra,cma,(double*)NULL,CHOLESKY_UPPER_TRIANGULAR);
      //     if ((SharedMemoryDimension_double(X) != rb*cb) || (SharedMemoryCounter_double(X) > 1) || (X == A) || ((X == B) && !cmb))
      // 	    X=AllocateSharedMemory_double(rb*cb);
      //     if (X != B)
      // 	    if (cmb)
      // 	      memcpy(X,B,rb*cb*sizeof(double));
      // 	    else
      // 	      TransposeMatrix(X,B,rb,cb,0);
      // 	  char uplo='U', trans='T', diag='N';
      // 	  int info;
      // 	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,T,&ra,X,&rb,&info);
      // 	  if (info > 0)
      // 	    throw dw_exception("BaseSolve(): matrix singular");
      // 	  else  if (info < 0)
      // 	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
      // 	  trans='N';
      // 	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,T,&ra,X,&rb,&info);
      // 	  if (info > 0)
      // 	    throw dw_exception("BaseSolve(): matrix singular");
      // 	  else if (info < 0)
      // 	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
      // 	  FreeSharedMemory_double(T);
      // 	}
      // catch (...)
      // 	{
      // 	  FreeSharedMemory_double(T);
      // 	  if (X != original_X) FreeSharedMemory_double(X);
      // 	  throw;
      // 	}
      break;
    case SOLVE_QR:
      throw dw_exception("BaseSolveComplex(): not yet implemented");
      // double *Q, *R;
      // Q=R=(double*)NULL;
      // try
      // 	{
      // 	  Q=AllocateSharedMemory_double(ra*ra);
      // 	  R=AllocateSharedMemory_double(ra*ra);
      // 	  BaseQR(A,ra,ra,cma,Q,R,1);
      // 	  X=BaseMultiply(Q,ra,ra,0,B,rb,cb,cmb,X);
      // 	  char uplo='U', trans='N', diag='N';
      // 	  int info;
      // 	  dtrtrs(&uplo,&trans,&diag,&ra,&cb,R,&ra,X,&rb,&info);
      // 	  if (info > 0)
      // 	    throw dw_exception("BaseSolve(): matrix singular");
      // 	  else if (info < 0) 
      // 	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
      // 	  FreeSharedMemory_double(Q);
      // 	  FreeSharedMemory_double(R);
      // 	}
      // catch (...)
      // 	{
      // 	  FreeSharedMemory_double(Q);
      // 	  FreeSharedMemory_double(R);
      // 	  if (X != original_X) FreeSharedMemory_double(X);
      // 	  throw;
      // 	}
      break;
    case SOLVE_SVD:
      throw dw_exception("BaseSolveComplex(): not yet implemented");
      // double *U, *V, *d, *Y;
      // U=V=d=Y=(double*)NULL;
      // try
      // 	{
      // 	  U=AllocateSharedMemory_double(ra*ra);
      // 	  V=AllocateSharedMemory_double(ra*ra);
      // 	  d=AllocateSharedMemory_double(ra);
      // 	  BaseSVD(A,ra,ra,cma,U,d,V,1);
      // 	  Y=BaseMultiply(U,ra,ra,0,B,rb,cb,cmb,(double*)NULL);
      // 	  if (d[ra-1] <= d[0]*MACHINE_EPSILON) throw dw_exception("BaseSolve(): matrix singular");
      // 	  InverseDiagonalMatrixMultiply(d,1,Y,rb,cb,1,Y);
      // 	  X=BaseMultiply(V,ra,ra,0,Y,rb,cb,1,X);
      // 	  FreeSharedMemory_double(U);
      // 	  FreeSharedMemory_double(V);
      // 	  FreeSharedMemory_double(d);
      // 	  FreeSharedMemory_double(Y);
      // 	}
      // catch (...)
      // 	{
      // 	  FreeSharedMemory_double(U);
      // 	  FreeSharedMemory_double(V);
      // 	  FreeSharedMemory_double(d);
      // 	  FreeSharedMemory_double(Y);
      // 	  if (X != original_X) FreeSharedMemory_double(X);
      // 	  throw;
      // 	}
      break;
    case SOLVE_LU:
      double *AA; int *pivot;
      AA=(double*)NULL; pivot=(int*)NULL;
      try
	{
	  int info;
	  AA=new double[2*ra*ra];
	  if (cma)
	    memcpy(AA,A,2*ra*ra*sizeof(double));
	  else
	    TransposeComplexMatrix(AA,A,ra,ra,0);
	  if ((SharedMemoryDimension_double(X) != 2*rb*cb) || (SharedMemoryCounter_double(X) > 1) || ((X == B) && !cmb))
	    X=AllocateSharedMemory_double(2*rb*cb);
	  if (X != B)
	    {
	      if (cmb)
		memcpy(X,B,2*rb*cb*sizeof(double));
	      else
		TransposeComplexMatrix(X,B,rb,cb,0);
	    }
	  pivot=new int[ra];
	  zgesv(&ra,&cb,AA,&ra,pivot,X,&rb,&info);
	  if (info > 0)
	    throw dw_exception("BaseSolve(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseSolve(): BLAS/LAPACK argument error");
	  delete[] pivot;
	  delete[] AA;
	}
      catch (...)
	{
	  if (pivot) delete[] pivot;
	  if (AA) delete[] AA;
	  if (X != original_X) FreeSharedMemory_double(X);
	  throw;
	}
      break;
    default:
      throw dw_exception("BaseSolve: unknown solution technique");
    }

  return X;
}

/*
   buffer = Inverse(M)
   The array buffer will always be in column major format.
*/
double* BaseInverse(double *M, int r, int c, int cm, double *buffer, int method)
{
  if (r != c)
    throw dw_exception("BaseInverse(): matrix must be square");

  if (!M) return (double*)NULL;
  double *original_buffer=buffer;

  switch (method)
    {
    case SOLVE_DIAGONAL:
      int i;
      double max, min, s;
      for (max=min=fabs(M[0]), i=r*r-1; i > 0; i-=r+1)
	if ((s=fabs(M[i])) > max) 
	  max=s;
	else if (s < min) min=s;
      if (min <= max*MACHINE_EPSILON) throw dw_exception("BaseInverse(): matrix singular");
      if ((SharedMemoryDimension_double(buffer) != r*r) || (SharedMemoryCounter_double(buffer) > 1))
	buffer=AllocateSharedMemory_double(r*r);
      if (buffer != M) memcpy(buffer,M,r*r*sizeof(double));
      for (i=r*r-1; i >= 0; i-=r+1) buffer[i]=1.0/buffer[i];
      break;
    case SOLVE_UPPER_TRIANGULAR:
    case SOLVE_LOWER_TRIANGULAR:
      try
	{
	  char diag='N', uplo=(method == SOLVE_UPPER_TRIANGULAR) ? 'U' : 'L';
	  int info;
	  if ((SharedMemoryDimension_double(buffer) != r*r) || (SharedMemoryCounter_double(buffer) > 1))
	    buffer=AllocateSharedMemory_double(r*r);
	  if (buffer != M)
	    if (cm)
	      memcpy(buffer,M,r*r*sizeof(double));
	    else
	      TransposeMatrix(buffer,M,r,r,0);
	  else
	    if (!cm) TransposeMatrix(buffer,r);
	  dtrtri(&uplo,&diag,&r,buffer,&r,&info);
	  if (info > 0)
	    throw dw_exception("BaseInverse(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	}
      catch (...)
	{
	  if (buffer != original_buffer) FreeSharedMemory_double(buffer);
	  throw;
	}
      break;
    case SOLVE_CHOLESKY:
      double *T;
      T=(double*)NULL;
      try
	{
	  T=BaseCholesky(M,r,r,cm,(double*)NULL,CHOLESKY_UPPER_TRIANGULAR);
	  char uplo='U', diag='N';
	  int info;
	  dtrtri(&uplo,&diag,&r,T,&r,&info);
	  if (info > 0)
	    throw dw_exception("BaseInverse(): matrix singular");
	  else  if (info < 0)
	    throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	  buffer=BaseMultiply(T,r,r,1,T,r,r,0,buffer);
	  FreeSharedMemory_double(T);
	}
      catch (...)
	{
	  FreeSharedMemory_double(T);
	  if (buffer != original_buffer) FreeSharedMemory_double(buffer);
	  throw;
	}
      break;
    case SOLVE_QR:
      double *Q, *R;
      Q=R=(double*)NULL;
      try
	{
	  Q=AllocateSharedMemory_double(r*r);
	  R=AllocateSharedMemory_double(r*r);
	  BaseQR(M,r,r,cm,Q,R,1);
	  char uplo='U', diag='N';
	  int info;
	  dtrtri(&uplo,&diag,&r,R,&r,&info);
	  if (info > 0)
	    throw dw_exception("BaseInverse(): matrix singular");
	  else if (info < 0) 
	    throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	  buffer=BaseMultiply(R,r,r,1,Q,r,r,0,buffer);
	  FreeSharedMemory_double(Q);
	  FreeSharedMemory_double(R);
	}
      catch (...)
	{
	  FreeSharedMemory_double(Q);
	  FreeSharedMemory_double(R);
	  if (buffer != original_buffer) FreeSharedMemory_double(buffer);
	  throw;
	}
      break;
    case SOLVE_SVD:
      double *U, *V, *d;
      U=V=d=(double*)NULL;
      try
    	{
    	  U=AllocateSharedMemory_double(r*r);
    	  V=AllocateSharedMemory_double(r*r);
    	  d=AllocateSharedMemory_double(r);
    	  BaseSVD(M,r,r,cm,U,d,V,1);
    	  if (d[r-1] <= d[0]*MACHINE_EPSILON) throw dw_exception("BaseInverse(): matrix singular");
	  InverseDiagonalMatrixMultiply(d,1,U,r,r,0,U);
    	  buffer=BaseMultiply(V,r,r,0,U,r,r,0,buffer);
    	  FreeSharedMemory_double(U);
    	  FreeSharedMemory_double(V);
    	  FreeSharedMemory_double(d);
    	}
      catch (...)
    	{
    	  FreeSharedMemory_double(U);
    	  FreeSharedMemory_double(V);
    	  FreeSharedMemory_double(d);
    	  if (buffer != original_buffer) FreeSharedMemory_double(buffer);
    	  throw;
    	}
      break;
    case SOLVE_LU:
      double *work; int *pivot;
      work=(double*)NULL, pivot=(int*)NULL;
      try
	{
	  int info, lwork=-1;
	  double size;
	  if ((SharedMemoryDimension_double(buffer) != r*r) || (SharedMemoryCounter_double(buffer) > 1))
	    buffer=AllocateSharedMemory_double(r*r);
	  if (buffer != M)
	    if (cm)
	      memcpy(buffer,M,r*r*sizeof(double));
	    else
	      TransposeMatrix(buffer,M,r,r,0);
	  else 
	    if (!cm)
	      TransposeMatrix(buffer,r);
	  pivot=new int[r];
	  dgetrf(&r,&r,buffer,&r,pivot,&info);
	  if (info > 0)
	    throw dw_exception("BaseInverse(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	  dgetri(&r,buffer,&r,pivot,&size,&lwork,&info);
	  if (info < 0) throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	  work=new double[lwork=(int)size];
	  dgetri(&r,buffer,&r,pivot,work,&lwork,&info);
	  if (info > 0)
	    throw dw_exception("BaseInverse(): matrix singular");
	  else if (info < 0)
	    throw dw_exception("BaseInverse(): BLAS/LAPACK argument error");
	  delete[] pivot;
	  delete[] work;
	}
      catch (...)
	{
	  if (pivot) delete[] pivot;
	  if (work) delete[] work;
	  if (buffer != original_buffer) FreeSharedMemory_double(buffer);
	  throw;
	}
      break;
    default:
      throw dw_exception("BaseInverse: unknown solution technique");
    }

  return buffer;
}

//====== Revised by HWu ======
// Calculates the generalized inverse of M
// Assume
// 	M:	array of r*c 
// 	r:	number of rows
// 	c:	number of columns
// 	cm:	1 column major; 0 row major
// 	s:	stride
// 	buffer:	array containing the generalized inverse of M
//
// Returns:
// 	buffer = U * inv(D) * V 
// 	where U, D and V are obtained through SVD of M
//
// Results:
// 	return pointer of the array containing the generalized inverse of M upon success
// 	otherwise throw exceptions

double* BaseGeneralizedInverse(const double * M, int r, int c, int cm, int s, double *buffer)
{
	if (!M)
                return NULL;
	
	double *original_buffer = buffer;
	int q = r < c ? r : c;
	double *d=NULL, *u=NULL, *v=NULL; 

	if ( SharedMemoryDimension_double(buffer) != r*c || SharedMemoryCounter_double(buffer) > 1 )
               	buffer = AllocateSharedMemory_double(r*c);

       	d = AllocateSharedMemory_double(q);          
       	u = AllocateSharedMemory_double(r*r);           
       	v = AllocateSharedMemory_double(c*c);
	try
	{
		BaseSVD((double*)M, r, c, cm, u, d, v, 0); 
	}
	catch(...)
	{
		if (buffer != original_buffer)
			FreeSharedMemory_double(buffer); 
		if (d)
			FreeSharedMemory_double(d); 
		if (u)
			FreeSharedMemory_double(u); 
		if (v) 
			FreeSharedMemory_double(v); 
		
		throw dw_exception("BaseGeneralizedInverse(): error occurred during BaseSVD"); 
	}

	// Inverse d
	const double EPSILON_TOLERANCE = 1.06E-08; 
        for (int i=0; i<q; i++)
                if (d[i] > EPSILON_TOLERANCE )
                        d[i] = 1.0/d[i]; 
                else
                        d[i] = 0;

	// svd: M = U D V'
	// 	U: column major
	// 	V': column majory	
	// generalized inverse = V inv(D) U'
	//
	// buffer (column major) = diagnoal matrix made from transpose and inverse of (d)
	buffer = BaseDiagonalMatrix(c, r, d, q, buffer);
	// buffer (column major) = V (row major) * buffer (column major)
	buffer = BaseMultiply(v, c, c, 0, buffer, c, r, 1, buffer); 
	// buffer (column major) = buffer (column major) * U' (row major)
	buffer = BaseMultiply(buffer, c, r, 1, u, r, r, 0, buffer);

	// Need to transpose buffer if cm=row major
	if (!cm)
        {
                double *temp = AllocateSharedMemory_double(r*c);
                memcpy(temp, buffer, r*c*sizeof(double));
                TransposeMatrix(buffer, temp, r, c, 1);
                FreeSharedMemory_double(temp);
        }
        FreeSharedMemory_double(d); 
	FreeSharedMemory_double(u); 
	FreeSharedMemory_double(v); 

	return buffer;
}
//============================

//====== Added by HWu ======
// Schur decomposition (simplified version) of real matrix without select function being specified
//  
// buffer = schur(M); 
//
// Assumes
// 	M:	array of length r*c
// 	r:	number of rows
// 	c:	number of columns
// 	cm:	column major or row major
// 	T:	array of length r*c containing the real schur form of M
// 	R:	array of length r containing the real parts of the eigen-values
// 	I:	array of length r containing the imaginary parts of the eigen-values
// 	Z:	array of length r*c or null, containing the schur vectors
//
// Returns
// 	return to the calling function upon success
// 	otherwise threw exception
//
// Results
// 	Computes for an r*c (r==c) real non-symmetric matrix M, the eighenvalues, 
// 	the real Schur form T, and the matrix of Schur vectors Z
//
// 		M = Z*T*U'
//
void BaseSchur(const double *M, int r, int c, int cm, double *T, double *eR, double *eI, double *Z)
{
        if(r != c)
                throw dw_exception("BaseSchur(): number of rows and columns must be equal");

	if (r == 0)
		return; 

        if ( (SharedMemoryDimension_double(T) != r*r) || ( SharedMemoryCounter_double(T)>1 ) )
		throw dw_exception("BaseSchur(): invalid arguments"); 
	if ( (SharedMemoryDimension_double(eR) != r) || ( SharedMemoryCounter_double(eR)>1 ) )
		throw dw_exception("BaseSchur(): invalid arguments"); 
	if ( (SharedMemoryDimension_double(eI) != r) || (SharedMemoryCounter_double(eI)>1 ) )
                throw dw_exception("BaseSchur(): invalid arguments");
	if ( Z && (SharedMemoryDimension_double(Z) != r*r || SharedMemoryCounter_double(Z)>1 ) )
		throw dw_exception("BaseSchur(): invalid arguments");  

	// T is a copy of M in column major
	if (cm)	
       		memcpy(T, M, sizeof(double)*r*r);
	else
		TransposeMatrix(T, (double*)M, r, c, cm); 	

        char jobvs = Z ? 'V' : 'N';  // Schur vector not computed if Z==NULL;    
        char sort = 'N';        // Eigenvalues not ordered 
	int ldvs= (jobvs == 'V' ? r : 1), lwork=3*r, sdim, info;
        double *work = new double[lwork]; 
        int *bwork = new int[r]; 
                
	dgees(&jobvs,&sort,0,&r,T,&r,&sdim,eR,eI,Z,&ldvs,work,&lwork,bwork,&info);
	
	delete [] work; 
	delete [] bwork;
	if (info)
		throw dw_exception("BaseSchur(): lapack error");
	
	if (!cm)
	{
		TransposeMatrix(T, r); 
		if (Z) TransposeMatrix(Z, r); 
	}
}
// Reorders the real Schur factorization of a real matrix so that a selected cluster of eigen-values
// appears in the leading diagonal blocks of the upper quasi-triangular matrix, and the leading columns
// of vector matrix form an orthonormal basis of the corresponding right invariant subspace
//
// Assumes
//	T:	array of length r*r containing the real schur form of some real matrix
//	r:	rows of T
//	c:	columns of T
//	cm:	column major (if cm=1) or row major (if cm=0)
//	Z:	array of length r*r or NULL, containing the schur vectors of some real matrix
//	select:	array of length r, select[i]=1 means the i-th eigen value selected; select[i]=0 
//		means the i-th eigen value not selected
//	OrdT:	array of length r*r containining reordered shur form 
//	OrderER:	array of length r, containing the real parts of the reordered eigen values
//	OrderEI:	array of length r, containing the imaginary parts of the reordered eigen
//			values
//	OrderZ:	array of length r*r or NULL, containing the reordered schur vectors		
// 
// Returns
//	returns to the calling function upon success
//	otherwise throw exception
//
// Results
//	orders the schur factorization of a ream matrix (characterized by T and Z) so that a
//	selected sub-group of eigenvalues appears in the leading diagonal blocks of the upper
//	quasi-triangular matrix, and the leading columns of the schur vectors form an orthonormal
//	basis of the corresponding right invariant subspace

void BaseOrderSchur(const double *T, int r, int c, int cm, const double *Z, const int *select, double *OrdT, double *OrdER, double *OrdEI, double *OrdZ)
{
	if (r != c)
		throw dw_exception("BaseOrderSchur(): number of rows and columns must be equal");
	
	if ( (SharedMemoryDimension_double(OrdT) != r*r) || (SharedMemoryCounter_double(OrdT)>1) )
		throw dw_exception("BaseOrderSchur(): invalid arguments"); 
	if ( (SharedMemoryDimension_double(OrdER) != r) || (SharedMemoryCounter_double(OrdER)>1) )
		throw dw_exception("BaseOrderSchur(): invalid arguments"); 
	if ( (SharedMemoryDimension_double(OrdEI) != r) || (SharedMemoryCounter_double(OrdEI)>1) )
		throw dw_exception("BaseOrderSchur(): invalid arguments"); 
	if ( Z && ( (SharedMemoryDimension_double(OrdZ) != r*r) || (SharedMemoryCounter_double(OrdZ)>1) ) )
		throw dw_exception("BaseOrderSchur(): invalid arguments"); 

	char job = 'N'; 	// condition numbers not calculated
	char compq = Z ? 'V' : 'N'; 	// schur vectors not updated if Z=NULL; 
	if (cm)
		memcpy(OrdT, T, sizeof(double)*r*r); 
	else 
		TransposeMatrix(OrdT, (double*)T, r, c, cm); 	

	if (Z)
	{
		if (cm)
			memcpy(OrdZ, Z, sizeof(double)*r*r); 
		else 
			TransposeMatrix(OrdZ, (double*)Z, r, c, cm); 
	}
	int ldq = (compq == 'V' ? r : 1);	// leading dimension of OrdZ
	int lwork=r;	// size of work 
	int liwork=1;	// size of iwork 
	int invariantM;	// dimension of specified invariant subspace
	int info;  
	double eCondition, sCondition;	//  condition number for the selected eigen values and the invariant subspace
	double *work = new double[lwork]; 
	int *iwork = new int[liwork]; 

	dtrsen(&job,&compq,(int*)select,&r,OrdT,&r,OrdZ,&ldq,OrdER,OrdEI,&invariantM,&eCondition,&sCondition,work,&lwork,iwork,&liwork,&info); 
	
	delete []work; 
	delete []iwork; 
	if (info)
		throw dw_exception("BaseOrderSchur(): lapack error"); 
	if (!cm)
	{
		TransposeMatrix(OrdT,r); 
		if (Z) TransposeMatrix(OrdZ,r); 
	}
} 

//==========================


/*******************************************************************************/
/************************** Constructors/Destructors ***************************/
/*******************************************************************************/
TDenseMatrix::TDenseMatrix(const TPermutationMatrix &P)
{
  IncrementSharedMemory_double(matrix=AllocateSharedMemory_double(P.dim*P.dim));
  rows=cols=P.dim;
  column_major=1;
  if (matrix)
    {
      Zeros();
      int i, j;
      for (i=(j=cols-1)*rows; j >= 0; i-=rows, j--) matrix[i+P.permutation[j]]=1.0;
    }
}


/*******************************************************************************/
/******************************* Size and Shape ********************************/
/*******************************************************************************/
/*
   Guarantees the array vector is unique and of dimension d.  The values of the
   shared memory are undefined.  The integer d MUST be non-negative.
*/
void TDenseVector::UniqueMemory(int d)
{
  if (dim != d)
    {
      if (d < 0) throw dw_exception("UniqueMemory(): dimension must be non-negative");
      double *buffer=AllocateSharedMemory_double(d);
      FreeSharedMemory_double(vector);
      IncrementSharedMemory_double(buffer);
      vector=buffer;
      dim=d;
    }
  else
    vector=UniqueSharedMemory_double(vector);
}

/*
   Guarantees the array matrix is unique and of dimension r*c.  The values of the
   shared memory are undefined.  Both r and c MUST be non-negative.
*/
void TDenseMatrix::UniqueMemory(int r, int c, bool col_major)
{
  if ((r < 0) || (c < 0)) throw dw_exception("UniqueMemory(): number of rows or columns must be non-negative");
  if (rows*cols != r*c)
    {
      double *buffer=AllocateSharedMemory_double(r*c);
      FreeSharedMemory_double(matrix);    
      IncrementSharedMemory_double(buffer);
      matrix=buffer;
    }
  else
    matrix=UniqueSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
}

/*
   Guarantees the array permutation is unique and of dimension d.  The contents 
   of permutation are undefined, but guaranteed to represent a valid permutation.
   The integer d MUST be non-negative.
*/
void TPermutationMatrix::UniqueMemory(int d)
{
  if (dim != d)
    {
      if (d < 0) throw dw_exception("UniqueMemory(): dimension must be non-negative");
      int *buffer=AllocateSharedMemory_int(d);
      FreeSharedMemory_int(permutation);
      IncrementSharedMemory_int(buffer);
      permutation=buffer;
      dim=d;
    }
  else
    permutation=UniqueSharedMemory_int(permutation);
}

void TDenseVector::Resize(int d) 
{ 
  if (dim != d)
    {
      if (d < 0) throw dw_exception("TDenseVector::Resize(int): negative index");
      FreeSharedMemory_double(vector);
      try { vector=AllocateSharedMemory_double(dim=d); }
      catch (std::bad_alloc)
	{
	  vector=(double*)NULL;
	  dim=0;
	  throw;
	}
      IncrementSharedMemory_double(vector);
    }
}

void TDenseMatrix::Resize(int r, int c)
{
  if ((rows != r) || (cols != c))
    {
      if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrix::Resize(int,int): negative index");
      if (rows*cols != r*c)
	{
	  FreeSharedMemory_double(matrix);
	  try { matrix=AllocateSharedMemory_double(r*c); }
	  catch (std::bad_alloc)
	    {
	      matrix=(double*)NULL;
	      r=c=0;
	      throw;
	    }
	  IncrementSharedMemory_double(matrix);
	}
      rows=r;
      cols=c;
    }
}

void TPermutationMatrix::Resize(int d) 
{ 
  if (dim != d) 
    {
      if (d < 0) throw dw_exception("TPermutationMatrix::Resize(int): negative index");
      FreeSharedMemory_int(permutation);
      try { permutation=AllocateSharedMemory_int(dim=d); }
      catch (std::bad_alloc)
	{
	  permutation=(int*)NULL;
	  dim=0;
	  throw;
	}
      IncrementSharedMemory_int(permutation);
    }
}

void TDenseMatrix::ForceColumnMajor(void)
{
  if (!column_major)
    {
      if (rows == cols)
	{
	  UniqueMemory();
	  TransposeMatrix(matrix,rows);
	  column_major=1;
	}
      else
	{
	  double *buffer=AllocateSharedMemory_double(rows*cols);
	  TransposeMatrix(buffer,matrix,rows,cols,0);
	  ShareMemory(buffer,rows,cols,1);
	}
    }
}

void TDenseMatrix::ForceRowMajor(void)
{
  if (column_major)
    {
      if (rows == cols)
	{
	  UniqueMemory();
	  TransposeMatrix(matrix,rows);
	  column_major=0;
	}
      else
	{
	  double *buffer=AllocateSharedMemory_double(rows*cols);
	  TransposeMatrix(buffer,matrix,rows,cols,1);
	  ShareMemory(buffer,rows,cols,0);
	}
    }
}

/*******************************************************************************/
/********************************* Logical *************************************/
/*******************************************************************************/
bool TDenseVector::operator == (const TDenseVector &right)
{
	if (dim == right.dim)
	{
		for (int i=0; i<dim; i++)
			if (vector[i] != right.vector[i]) 
				return false; 
		return true; 
	}
	else 
		return false; 
}


/*******************************************************************************/
/********************************* Assignmment *********************************/
/*******************************************************************************/
TDenseMatrix& TDenseMatrix::operator=(const TPermutationMatrix &P)
{
  UniqueMemory(P.dim,P.dim,1);
  if (P.dim > 0)
    {
      Zeros();
      int i, j;
      for (i=(j=cols-1)*rows; j >= 0; i-=rows, j--) matrix[i+P.permutation[j]]=1.0;
    }
  return *this;
}

/*******************************************************************************/
/******************************* Initialization ********************************/
/*******************************************************************************/
TDenseVector& TDenseVector::Initialize(double s)
{
  if (vector)
    {
      UniqueMemory();
      for (int i=dim-1; i >= 0; i--) vector[i]=s;
    }
  return *this;
}

TDenseVector& TDenseVector::Initialize(double s, int d)
{
  if (d < 0) throw dw_exception("TDenseVector::Initialize(double,int): negative index");
  UniqueMemory(d);
  for (int i=dim-1; i >= 0; i--) vector[i]=s;
  return *this;
}

//============= Added by HWu =======================
double & TDenseMatrix::operator() (int r, int c)
{
	if (r<0 || r>=rows || c<0 || c>=cols)
		throw dw_exception("TDenseMatrix::operator(int, int): index out of range"); 
	UniqueMemory(); 
	return matrix[Index(r,c)]; 
}

TDenseMatrix & TDenseMatrix::CopyContent(const TDenseMatrix &M)
{
	UniqueMemory(M.rows, M.cols, M.column_major); 
	for (int m=0; m<M.rows; m++)
		for (int n=0; n<M.cols; n++)
			matrix[Index(m,n)] = M(m,n); 
	return *this; 
}

//==================================================

void TDenseMatrix::SetElement(double s, int r, int c)
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrix::SetElement(double,int,int): index out of range");
  UniqueMemory();
  matrix[Index(r,c)]=s;
}

//======== Added by HWu ===============
void TDenseVector::SetElement(double s, int i)
{
	if (i<0 || i>=dim)
		throw dw_exception("TDenseVector::SetElement(double, int): index out of range"); 
	UniqueMemory(); 
	vector[i] = s; 
}

// vector[locs] = x
void TDenseVector::SetSubVector(const std::vector<int> &locs, const TDenseVector &x)
{
	if ((int)locs.size() != x.dim)
		throw dw_exception("TDenseVector::SetSubVector(): subvector locations and values do not match"); 
	UniqueMemory(); 
	for (unsigned int i=0; i<locs.size(); i++)
	{
		if (locs[i]<0 || locs[i] >= dim)
			throw dw_exception("TDenseVector::SetSubVector(): subvector location exceeds boundary");
		this->vector[locs[i]] = x[i]; 
	}
}

// Subvector
TDenseVector TDenseVector::SubVector(const std::vector<int> &index) const
{
        TDenseVector v(index.size());
        for (unsigned int i=0; i<index.size(); i++)
        {
                if (index[i] <0 || index[i] >= dim)
                {
                        v=TDenseVector();
                        return v;
                }
                v[i] = this->vector[index[i]];
        }
        return v;
}


TDenseVector& TDenseVector::CopyContent(const TDenseVector &v)
{
	UniqueMemory(v.dim);
	for (int i=0; i<dim; i++)
		vector[i] = v(i); 
	return *this;  	
}

double & TDenseVector::operator() (int i)
{
	if (i<0 || i>=dim)
		throw dw_exception("TDenseVector::operator(int): index out of range"); 
	UniqueMemory(); 
	return vector[i]; 
}

double & TDenseVector::operator[] (int i)
{
	if (i<0 || i>=dim)
                throw dw_exception("TDenseVector::operator(int): index out of range");
        UniqueMemory();
        return vector[i];
}

// return a vector (x) whose elements are re-ordered according to order 
// 	x(i) = vector(order[i])
TDenseVector TDenseVector::SwitchOrder(const std::vector<int> &order)
{
        if ((int)order.size() != dim)
		throw dw_exception("TDenseVector::SwitchOrder(vector<int>): dimension of order is inconsistent with the dimension of vector"); 
	TDenseVector switched(dim); 
	for (int i=0; i<dim; i++)
		switched[i] = vector[order[i]]; 
	return switched; 
}

// ======================================

TDenseMatrix& TDenseMatrix::Initialize(double s)
{
  if (matrix)
    {
      UniqueMemory();
      for (int i=rows*cols-1; i >= 0; i--) matrix[i]=s;
    }
  return *this;
}

TDenseMatrix& TDenseMatrix::Initialize(double s, int r, int c)
{
  if ((r < 0) || (c < 0)) 
    throw dw_exception("TDenseMatrix::Initialize(double,int,int): negative index");
  UniqueMemory(r,c,column_major);
  for (int i=rows*cols-1; i >= 0; i--) matrix[i]=s;
  return *this;
}

TDenseMatrix BlockDiagonalMatrix(const TDenseMatrix &M, int n)
{
  if (n < 0) throw dw_exception("BlockDiagonalMatrix(): number of blocks must be positive");
  if (n == 0) return TDenseMatrix(0,0);
  TDenseMatrix BD(n*M.rows,n*M.cols,M.column_major,0.0);
  for (int i=n-1; i >= 0; i--) BD.Insert(i*M.rows,i*M.cols,M);
  return BD;
}


//=== miscellaneous matrix/vector manipulation
TDenseVector Cat(const TDenseVector &v1, const TDenseVector &v2)
{
  TDenseVector v(v1.dim+v2.dim);
  memcpy(v.vector,v1.vector,v1.dim*sizeof(double));
  memcpy(v.vector+v1.dim,v2.vector,v2.dim*sizeof(double));
  return v;
}

TDenseVector Cat(const TDenseVector &v1, double x2)
{
  TDenseVector v(v1.dim+1);
  memcpy(v.vector,v1.vector,v1.dim*sizeof(double));
  v.vector[v1.dim]=x2;
  return v;
}

TDenseVector Cat(double x1, const TDenseVector &v2)
{
  TDenseVector v(1+v2.dim);
  v.vector[0]=x1;
  memcpy(v.vector+1,v2.vector,v2.dim*sizeof(double));
  return v;
}

TDenseVector Cat(const std::vector<TDenseVector> &v_array)
{
  if (v_array.size() == 0) return TDenseVector(0);
  int dim=v_array[0
].dim;
  for (int i=v_array.size()-1; i > 0; i--)
    dim+=v_array[i].dim;
  TDenseVector v(dim);
  for (int i=dim=0; i < (int)v_array.size(); dim+=v_array[i].dim, i++)
    memcpy(v.vector+dim,v_array[i].vector,v_array[i].dim*sizeof(double));
  return v;
}

TDenseMatrix HCat(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  if (M1.rows != M2.rows) throw dw_exception("HCat(): matrices not conformable");
  TDenseMatrix M(M1.rows,M1.cols+M2.cols,true);
  if (M1.rows > 0)
    {
      if (M1.cols > 0)
	{
	  if (M1.column_major)
	    memcpy(M.matrix,M1.matrix,M1.rows*M1.cols*sizeof(double));
	  else
	    TransposeMatrix(M.matrix,M1.matrix,M1.rows,M1.cols,false);
	}
      if (M2.cols > 0)
	{
	  if (M2.column_major)
	    memcpy(M.matrix+M1.rows*M1.cols,M2.matrix,M2.rows*M2.cols*sizeof(double));
	  else
	    TransposeMatrix(M.matrix+M1.rows*M1.cols,M2.matrix,M2.rows,M2.cols,false);
	}
    }
  return M;
}

TDenseMatrix HCat(const std::vector<TDenseMatrix> &M_array);

TDenseMatrix VCat(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  if (M1.cols != M2.cols) throw dw_exception("HCat(): matrices not conformable");
  TDenseMatrix M(M1.rows+M2.rows,M1.cols,false);
  if (M1.cols)
    {
      if (M1.rows)
	{
	  if (!M1.column_major)
	    memcpy(M.matrix,M1.matrix,M1.rows*M1.cols*sizeof(double));
	  else
	    TransposeMatrix(M.matrix,M1.matrix,M1.rows,M1.cols,true);
	}
      if (M2.rows)
	{
	  if (!M2.column_major)
	    memcpy(M.matrix+M1.rows*M1.cols,M2.matrix,M2.rows*M2.cols*sizeof(double));
	  else
	    TransposeMatrix(M.matrix+M1.rows*M1.cols,M2.matrix,M2.rows,M2.cols,true);
	}
    }
  return M;
}
TDenseMatrix VCat(const std::vector<TDenseMatrix> &M_array);

TDenseVector& TDenseVector::Vec(const TDenseMatrix &M)
{
  if (M.column_major)
    ShareMemory(M.matrix,M.rows*M.cols);
  else
    {
      UniqueMemory(M.rows*M.cols);
      TransposeMatrix(vector,M.matrix,M.rows,M.cols,0);
    }
  return *this;
}

TDenseVector Vec(const TDenseMatrix &M)
{
  if (M.column_major)
    return TDenseVector(M.matrix,M.rows*M.cols);
  else
    {
      double *vector=AllocateSharedMemory_double(M.rows*M.cols);
      TransposeMatrix(vector,M.matrix,M.rows,M.cols,0);
      return TDenseVector(vector,M.rows*M.cols);
    }
}

TDenseMatrix& TDenseMatrix::Reshape(int r, int c)
{
  if ((r < 0) || (c < 0) || (r*c != rows*cols))
    throw dw_exception("Reshape() - invalid number of rows or columns");
  ForceColumnMajor();
  UniqueMemory();
  rows=r;
  cols=c;
  return *this;
}


/*******************************************************************************/
/*************************** Miscellaneous Routines ****************************/
/*******************************************************************************/
double Norm(const TDenseVector &v)
{
  double rtrn=0.0;
  if (v.vector)
    for (int i=v.dim-1; i >= 0; i--) rtrn+=v.vector[i]*v.vector[i];
  return sqrt(rtrn);
}

double Norm(const TDenseMatrix &M)
{
  double rtrn=0.0;
  if (M.matrix)
    for (int i=M.rows*M.cols-1; i >= 0; i--) rtrn+=M.matrix[i]*M.matrix[i];
  return sqrt(rtrn);
}

double MatrixNorm(const TDenseMatrix &M)
{
  int q=M.rows < M.cols ? M.rows : M.cols;
  double rtrn, *d=AllocateSharedMemory_double(q);
  try
    {
      BaseSVD(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,d,(double*)NULL,1);
    }
  catch(...)
    {
      FreeSharedMemory_double(d);
      throw;
    }
  rtrn=d[0];
  FreeSharedMemory_double(d);
  return rtrn;
}

double TDenseVector::Sum(void) const
{
  double sum=0.0;
  for (int i=dim-1; i >= 0; i--) sum+=vector[i];
  return sum;
}

double Sum(const TDenseVector &v)
{
  double sum=0.0;
  for (int i=v.dim-1; i >= 0; i--) sum+=v.vector[i];
  return sum;
}

//====== Added by HWu ======
//======= Changed by DW - blas call seems to be faster =======
double InnerProduct(const TDenseVector &v1, const TDenseVector &v2)
{
	if (v1.dim != v2.dim)
		throw dw_exception("InnerProduct(): v1 and v2 must be of equal dimension"); 
	
        //double result = 0; 
	//for (unsigned int i=0; i<v1.dim; i++)
	//	result += v1[i]*v2[i]; 
	//return result; 

        int inc=1, dim=v1.dim;
        return ddot(&dim,v1.vector,&inc,v2.vector,&inc);
}
//==========================

double InnerProduct(const TDenseVector &v1, const TDenseVector &v2, const TDenseMatrix &QuadraticForm)
{
  if ((v1.dim != QuadraticForm.rows) || (v2.dim != QuadraticForm.cols))
    throw dw_exception("InnerProduct(): invalid matrix or vector dimensions");

  TDenseVector x=QuadraticForm*v2;
  int inc=1, dim=v1.dim;
  return ddot(&dim,v1.vector,&inc,x.vector,&inc);

  // // This code, while using less memory, seems to be less efficient for vectors of dimension greater than 16
  // double x, inner_product=0.0;
  // int k=v1.dim*v1.dim-1;
  // if (QuadraticForm.column_major)
  //   for (int j=v1.dim-1; j >= 0; j--)
  //     {
  // 	x=0.0;
  // 	for (int i=v1.dim-1; i >= 0; k--, i--) x+=v1.vector[i]*QuadraticForm.matrix[k];
  // 	inner_product+=x*v2.vector[j];
  //     }
  // else
  //   for (int i=v1.dim-1; i >= 0; i--)
  //     {
  // 	x=0.0;
  // 	for (int j=v1.dim-1; j >= 0; k--, j--) x+=QuadraticForm.matrix[k]*v2.vector[j];
  // 	inner_product+=v1.vector[i]*x;
  //     }
  // return inner_product;

  // // This code only works for symmetric matrices and v1 = v2
  // double x, inner_product=0.0;
  // int k=v1.dim*v1.dim-1;
  // for (int i=v1.dim-1; i >= 0; i--)
  //   {
  //     x=0.0;
  //     for (int j=v1.dim-1; j > i; k--, j--) x+=QuadraticForm.matrix[k]*v2.vector[j];
  //     inner_product+=v1.vector[i]*(QuadraticForm.matrix[k]*v2.vector[i] + 2.0*x);
  //     k-=i+1;
  //   }
  // return inner_product;
}

TDenseMatrix OuterProduct(const TDenseVector &v1, const TDenseVector &v2)
{
  return TDenseMatrix(v1.dim,v2.dim,true).OuterProduct(v1,v2);
}

TDenseMatrix OuterProduct(const TDenseVector &v)
{
  return TDenseMatrix(v.dim,v.dim,true).OuterProduct(v);
}

TDenseMatrix& TDenseMatrix::OuterProduct(const TDenseVector &v1, const TDenseVector &v2)
{
  UniqueMemory(v1.dim,v2.dim,true);
  for (int k=v1.dim*v2.dim, j=v2.dim-1; j >= 0; j--)
    for (int i=v1.dim-1; i >= 0; i--) matrix[--k]=v1[i]*v2[j];
  return *this;
}

TDenseMatrix& TDenseMatrix::OuterProduct(const TDenseVector &v)
{
  UniqueMemory(v.dim,v.dim,true);
  for (int k=v.dim*v.dim, j=v.dim-1; j >= 0; j--)
    for (int i=v.dim-1; i >= 0; i--) matrix[--k]=v[i]*v[j];
  return *this;
}

//====== Added by HWu ======
double LogAbsDeterminant(const TDenseMatrix &M)
{
	if (M.rows != M.cols)
		throw dw_exception("LogAbsDeterminant(): M must be square");
	
	TLapackLU lu(M); 

	double det=0.0; 
	for (int i=0; i<M.rows; i++)
	{
		if (lu.LU[i+i*M.rows] < 0.0) 
			det += log(-lu.LU[i+i*M.rows]); 
		else if (lu.LU[i+i*M.rows] > 0.0) 
			det += log(lu.LU[i+i*M.rows]); 
		else 
			return MINUS_INFINITY; 
	}
	return det;  	
}
double LogAbsDeterminant_Cholesky(const TDenseMatrix &T)
// T is the cholesky composition of some matrix 
{
	if (T.rows != T.cols)
		throw dw_exception("LogAbsDeterminant_Cholesky(): T must be square"); 
	double det = 0.0; 
	for (int i=0; i<T.rows; i++)
	{
		if (T(i,i) <0.0 || T(i,i) > 0.0)
			det += 2.0*log(T(i,i)); 
		else
			return MINUS_INFINITY; 
	}
	return det; 
}

double RCond(const TDenseMatrix &M)
//  reciprocal of the condition number of a general real matrix A, in the 1-norm
//  replicates rcond in Matlab
{
	// 1-norm of M
	double *workspace = NULL, mnorm, rcond; 
	if (M.column_major)
	{
		workspace = new double[M.rows]; 
		mnorm = dlange("1", &M.rows, &M.cols, M.matrix, &M.rows, workspace); 
	}
	else 
	{
		TDenseMatrix M_transpose=Transpose(M); 
		workspace = new double[M_transpose.rows]; 
		mnorm = dlange("1", &M_transpose.rows, &M_transpose.cols, M_transpose.matrix, &M_transpose.rows, workspace); 
	}

	// Rcond = 1/ (norm(A) * norm(inv(A)));
	// LU decomposition first
	TLapackLU lu(M); // lu is column majored
	if (workspace != NULL)
		delete [] workspace; 
	workspace = new double[4*lu.cols]; 
	int *iworkspace = new int[lu.cols], info; 
	dgecon("1", &lu.cols, lu.LU, &lu.rows, &mnorm, &rcond, workspace, iworkspace, &info); 
	delete [] workspace; 
	delete [] iworkspace; 
	if (info == 0)
		return rcond; 
	else 
		throw dw_exception("RCond(): illegal argument");				
}

//==========================

//====== Added by HWu ======
double Determinant(const TDenseMatrix &M)
{
	if (M.rows != M.cols)
		throw dw_exception("Determinant(): M must be square"); 

	TLapackLU lu(M); 
	
	int sign = 1; 
	for (int i=M.rows-2; i>=0; i--)
		if (lu.p[i]-1 != i)
			sign = -sign; 
	double det=0.0; 
	for (int i=0; i<M.rows; i++)
	{
		if ( lu.LU[i+i*M.rows] < 0.0 )
		{
			det += log( -lu.LU[i+i*M.rows] ); 
			sign = -sign; 
		}
		else if ( lu.LU[i+i*M.rows] >0.0)
			det += log( lu.LU[i+i*M.rows] );
		else 
			return 0.0; 
	}
	return exp(det)*sign; 
}

double Determinant_Cholesky(const TDenseMatrix &T)
// T is the Cholsky decomposition of some matrix
{
	if (T.rows != T.cols)
		throw dw_exception("Determinant_Cholesky(): T must be square"); 

	double det = 0.0; 
	for (int i=0; i<T.rows; i++)
	{
		if (T(i,i) > 0 || T(i,i) < 0)
			det += 2.0*log(T(i,i)); 
		else 
			return 0.0; 
	}
	return exp(det); 
}

int Rank(const TDenseMatrix &M)
{
	TDenseVector d; 
	SVD(d, M); 
	const double EPSILON_TOLERANCE = 1.06E-08;
	double small = d(0)*EPSILON_TOLERANCE; 
	int q = M.cols < M.rows ? M.cols : M.rows; 
	int rank; 
	for (rank=q-1; rank>=0; rank--)
		if (d(rank) > small)
			break; 	
	return rank+1; 		
}

bool IsZeroMatrix(const TDenseMatrix &M)
{
	/*for (int j=0; j<M.cols; j++)
	{
		for (int i=0; i<M.rows; i++)
		{
			if (M(i,j))
				return false; 
		}
	}
	return true; */
	TDenseVector wsp(M.rows, 0.0); 
	for (int j=0; j<M.cols; j++)
		for (int i=0; i<M.rows; i++)
			wsp(i) += fabs(M(i,j)); 
	double hnorm = 0.0; 
	for (int i=0; i<wsp.Dimension(); i++)
		hnorm = hnorm >= wsp(i) ? hnorm : wsp(i); 

	if (hnorm <= MACHINE_EPSILON)
		return true; 
	else 
		return false; 

}
//====== Added by Hwu ======

/*******************************************************************************/
/******************************* Random Matrices *******************************/
/*******************************************************************************/
TDenseVector& TDenseVector::RandomNormal(void)
{
  if (vector)
    {
      UniqueMemory();
      for (int i=dim-1; i >= 0; i--) vector[i]=dw_gaussian_rnd();
    }
  return *this;
}

TDenseVector& TDenseVector::RandomUniform(void)
{
  if (vector)
    {
      UniqueMemory();
      for (int i=dim-1; i >= 0; i--) vector[i]=dw_uniform_rnd();
    }
  return *this;
}

TDenseMatrix& TDenseMatrix::RandomNormal(void)
{
  if (matrix)
    {
      UniqueMemory();
      for (int i=rows*cols-1; i >= 0; i--) matrix[i]=dw_gaussian_rnd();
    }
  return *this;
}

TDenseMatrix& TDenseMatrix::RandomUniform(void)
{
  if (matrix)
    {
      UniqueMemory();
      for (int i=rows*cols-1; i >= 0; i--) matrix[i]=dw_uniform_rnd();
    }
  return *this;
}

TPermutationMatrix& TPermutationMatrix::RandomUniform(void)
{
  int i, j, k;
  if (dim > 1)
    {
      UniqueMemory();
      int *p=new int[dim-1];
      for (i=dim-2; i >= 0; i--) p[i]=floor(i + dw_uniform_rnd()*(dim-i));
      for (k=dim-1, j=dim-2; j >= 0; j--)
      	if (k == p[j]) k=j;
      permutation[dim-1]=k;
      for (i=dim-2; i >= 0; i--)
      	{
      	  for (k=p[i], j=i-1; j >= 0; j--)
      	    if (k == p[j]) k=j;
      	  permutation[i]=k;
      	}
      delete[] p;
    }
  return *this;
}

/*******************************************************************************/
/******************************* Decompositions ********************************/
/*******************************************************************************/
/*
   Assumes:
    compact is zero or one (one is default)
    U, V, and X are distinct matrices

   Results:
    X = U * DiagonalMatrix(d,X.rows,X.cols) * V'  (compact == 0)
    X = U * DiagonalMatrix(d,p,p) * V'            (compact == 1)

   Notes:
    The vector d will be p-dimensional with non-negative elements in decending 
    order, where p is the minimum of X.rows and X.cols.  The columns of U and V 
    will be orthonormal.

    If compact is one, then U will be (X.rows x p) in column major format and V
    will be (X.cols x p) in row major format.

    If compact is zero, then U will be (X.rows x X.rows) in column major format 
    and V will be (X.cols x X.cols) in row major format.
*/
void SVD(TDenseMatrix &U, TDenseVector &d, TDenseMatrix &V, const TDenseMatrix &X, int compact)
{
  if ((&U == &V) || (&U == &X) || (&V == &X)) throw dw_exception("SVD(): U, V, and X must be distinct");
  int p=(X.rows < X.cols) ? X.rows : X.cols;
  U.UniqueMemory(X.rows,compact ? p : X.rows,1);
  V.UniqueMemory(X.cols,compact ? p : X.cols,0);
  d.UniqueMemory(p);
  BaseSVD(X.matrix,X.rows,X.cols,X.column_major,U.matrix,d.vector,V.matrix,compact);
}

// Computes the eigenvalues and eigenvectors of a general matrix M. 
void Eig(TDenseVector &RealEigenValues, TDenseVector &ImaginaryEigenValues, TDenseMatrix &RealEigenVectors, TDenseMatrix &ImaginaryEigenVectors, const TDenseMatrix &M)
{
  if (M.rows != M.cols) throw dw_exception("Eig(): input matrix must be square");
  if ((&RealEigenValues == &ImaginaryEigenValues) || (&RealEigenVectors == &ImaginaryEigenVectors)) throw dw_exception("Eig(): output vectors and matrices must be distinct");
  if (!M.rows)
    {
      RealEigenValues.Resize(0);
      ImaginaryEigenValues.Resize(0);
      RealEigenVectors.Resize(0,0);
      ImaginaryEigenVectors.Resize(0,0);
    }
  else
    {
      char jobr='V', jobl='N';
      int n=M.rows, i, j, k, rtrn=0, bm=1, lwork=-1, info, bn=n;
      PRECISION *work, size, *A;
      dgeev(&jobl,&jobr,&bn,(double*)NULL,&bn,(double*)NULL,(double*)NULL,(double*)NULL,&bm,(double*)NULL,&bn,&size,&lwork,&info);
      if (info)
	rtrn=1;
      else
	{
	  if (!(A=new (std::nothrow) double[n*n]))
	    rtrn=2;
	  else
	    {
	      if (!(work=new (std::nothrow) double[(int)(lwork=size)]))
		rtrn=2;
	      else
		{
		  RealEigenValues.UniqueMemory(n);
		  ImaginaryEigenValues.UniqueMemory(n);
		  if (M.column_major)
		    memcpy(A,M.matrix,n*n*sizeof(PRECISION));
		  else
		    TransposeMatrix(A,M.matrix,n,n,0);
		  RealEigenVectors.UniqueMemory(n,n,true);
		  ImaginaryEigenVectors.UniqueMemory(n,n,true);
		  dgeev(&jobl,&jobr,&bn,A,&bn,RealEigenValues.vector,ImaginaryEigenValues.vector,(PRECISION*)NULL,&bm,RealEigenVectors.matrix,&bn,work,&lwork,&info);
		  if (info)
		    rtrn=1;
		  else
		    {
		      for (j=0; j < n; j++)
			if (ImaginaryEigenValues.vector[j] == 0.0)
			  for (k=j*n, i=n-1; i >= 0; i--) ImaginaryEigenVectors.matrix[k+i]=0.0;
			else
			  {
			    k=(++j)*n;
			    memcpy(ImaginaryEigenVectors.matrix+k-n,RealEigenVectors.matrix+k,n*sizeof(double));
			    for (i=n-1; i >= 0; i--) ImaginaryEigenVectors.matrix[k+i]=-RealEigenVectors.matrix[k+i];
			    memcpy(RealEigenVectors.matrix+k,RealEigenVectors.matrix+k-n,n*sizeof(double));
			  }
		    }
		  delete[] work;
		}
	      delete[] A;
	    }
	}
      if (rtrn == 1) throw dw_exception("Eig(): Blas/Lapack error");
      if (rtrn == 2) throw std::bad_alloc();
    }
}

// Computes the eigenvalues of a general matrix M.
void Eig(TDenseVector &RealEigenValues, TDenseVector &ImaginaryEigenValues, const TDenseMatrix &M)
{
  if (M.rows != M.cols) throw dw_exception("Eig(): input matrix must be square");
  if (&RealEigenValues == &ImaginaryEigenValues) throw dw_exception("Eig(): output vectors must be distinct");
  if (!M.rows)
    {
      RealEigenValues.Resize(0);
      ImaginaryEigenValues.Resize(0);
    }
  else
    {
      int rtrn=0, n=M.rows;
      RealEigenValues.UniqueMemory(n);
      ImaginaryEigenValues.UniqueMemory(n);
      double *A=new (std::nothrow) double[n*n];
      if (!A)
	rtrn=2;
      else
	{
	  char jobr='N', jobl='N';
	  int bm=1, lwork=-1, info;
	  double *work, size;
	  double *left_vector=new double[bm*n], *right_vector=new double[bm*n];
	  dgeev(&jobl,&jobr,&n,A,&n,RealEigenValues.vector,ImaginaryEigenValues.vector,left_vector,&bm,right_vector,&bm,&size,&lwork,&info);
	  if (info)
	    rtrn=1;
	  else
	    if (!(work=new (std::nothrow) double[lwork=(int)size]))
	      rtrn=2;
	    else
	      {
		if (M.column_major)
		  memcpy(A,M.matrix,n*n*sizeof(PRECISION));
		else
		  TransposeMatrix(A,M.matrix,n,n,0);
		dgeev(&jobl,&jobr,&n,A,&n,RealEigenValues.vector,ImaginaryEigenValues.vector,left_vector,&bm,right_vector,&bm,work,&lwork,&info);
		if (info)
		  rtrn=1;
		delete[] work;
	      }
	  delete[] left_vector;
	  delete[] right_vector;
	  delete[] A;
	}
      if (rtrn == 1) throw dw_exception("Eig(): Blas/Lapack error");
      if (rtrn == 2) throw std::bad_alloc();
    }
}

// Computes the eigenvalues and eigenvectors of the symmetric matrix M.  Routine 
// does not check that M is symmetric.
void Eig(TDenseVector &EigenValues, TDenseMatrix &EigenVectors, const TDenseMatrix &M)
{
  if (M.rows != M.cols) throw dw_exception("Eig(): input matrix must be square");
  if (!M.rows)
    {
      EigenValues.Resize(0);
      EigenVectors.Resize(0,0);
    }
  else
    {
      char jobz='V', range='A', uplo='U';
      int i, m, lwork=-1, *iwork, liwork=-1, isize, info, isuppz[2*M.rows], bn=M.rows, error=0;
      double *work, size, *A, v, abstol=-1.0;
      dsyevr(&jobz,&range,&uplo,&bn,(double*)NULL,&bn,&v,&v,&i,&i,&abstol,&m,(double*)NULL,(double*)NULL,&bn,isuppz,&size,&lwork,&isize,&liwork,&info);
      if (info)
	error=1;
      else
	if ((A=new (std::nothrow) double[M.rows*M.cols]))
	  {
	    if ((work=new (std::nothrow) double[(int)(lwork=size)]))
	      {
		if ((iwork=new (std::nothrow) int[liwork=isize]))
		  {
		    EigenValues.UniqueMemory(M.rows);
		    memcpy(A,M.matrix,M.rows*M.cols*sizeof(double));
		    EigenVectors.UniqueMemory(M.rows,M.cols,true);
		    dsyevr(&jobz,&range,&uplo,&bn,A,&bn,&v,&v,&i,&i,&abstol,&m,EigenValues.vector,EigenVectors.matrix,&bn,isuppz,work,&lwork,iwork,&liwork,&info);
		    if (info) error=1;
		    delete[] iwork;
		  }
		else
		  error=2;
		delete[] work;
	      }
	    else
	      error=2;
	    delete[] A;
	  }
	else
	  error=2;
      if (error == 1) throw dw_exception("Eig(): Blas/Lapack error");
      if (error ==2) throw std::bad_alloc();
    }
}

// Computes the eigenvalues of the symmetric matrix M.  Routine does not check
// that M is symmetric.
void Eig(TDenseVector &EigenValues, const TDenseMatrix &M)
{
  if (M.rows != M.cols) throw dw_exception("Eig(): input matrix must be square");
  if (!M.rows)
    EigenValues.Resize(0);
  else
    {
      char jobz='N', range='A', uplo='U';
      int i, m, lwork=-1, *iwork, liwork=-1, isize, info, isuppz, bn=M.rows, error=0;
      double *work, size, *A, v, abstol=-1.0;
      dsyevr(&jobz,&range,&uplo,&bn,(double*)NULL,&bn,&v,&v,&i,&i,&abstol,&m,(double*)NULL,(double*)NULL,&bn,&isuppz,&size,&lwork,&isize,&liwork,&info);
      if (info)
	error=1;
      else
	if ((A=new (std::nothrow) double[M.rows*M.cols]))
	  {
	    if ((work=new (std::nothrow) double[(int)(lwork=size)]))
	      {
		if ((iwork=new (std::nothrow) int[liwork=isize]))
		  {
		    EigenValues.UniqueMemory(M.rows);
		    memcpy(A,M.matrix,M.rows*M.cols*sizeof(double));
		    dsyevr(&jobz,&range,&uplo,&bn,A,&bn,&v,&v,&i,&i,&abstol,&m,EigenValues.vector,(double*)NULL,&bn,&isuppz,work,&lwork,iwork,&liwork,&info);
		    if (info) error=1;
		    delete[] iwork;
		  }
		else
		  error=2;
		delete[] work;
	      }
	    else
	      error=2;
	    delete[] A;
	  }
	else
	  error=2;
      if (error == 1) throw dw_exception("Eig(): Blas/Lapack error");
      if (error == 2) throw std::bad_alloc();
    }
}

//====== Added by HWu ======
void Schur(TDenseMatrix &T, TDenseVector &eR, TDenseVector &eI, TDenseMatrix &Z, const TDenseMatrix &M, bool if_schur_vector)
{
	if ( (&M == &T) || (&M == &Z) || (&T == &Z) )
		throw dw_exception("Schur(): Z, T and M must be distinct"); 
	T.UniqueMemory(M.rows, M.cols, M.column_major); 
	eR.UniqueMemory(M.rows); 
	eI.UniqueMemory(M.rows); 

	if (if_schur_vector)
	{
		Z.UniqueMemory(M.rows, M.cols, M.column_major); 
		BaseSchur(M.matrix,M.rows,M.cols,M.column_major,T.matrix,eR.vector,eI.vector,Z.matrix); 
	}
	else 
		BaseSchur(M.matrix,M.rows,M.cols,M.column_major,T.matrix,eR.vector,eI.vector,NULL); 
}
void OrderSchur(TDenseMatrix &OrdT, TDenseVector &OrdER, TDenseVector &OrdEI, TDenseMatrix &OrdZ, const TDenseMatrix &T, const TDenseMatrix &Z, const int *select, bool if_schur_vector)
{
	if ( (&T == &OrdT) || (&T == &OrdZ) || (&Z == &OrdT) || (&Z == &OrdZ) || (&OrdT == &OrdZ) )
		throw dw_exception("OrderSchur(): T, Z, OrdT and OrdZ must be distinct"); 
	OrdT.UniqueMemory(T.rows,T.cols,T.column_major); 
	OrdER.UniqueMemory(T.rows); 
	OrdEI.UniqueMemory(T.rows); 
	if (if_schur_vector)
	{
		OrdZ.UniqueMemory(T.rows,T.cols, T.column_major); 
		BaseOrderSchur(T.matrix,T.rows,T.cols,T.column_major,Z.matrix,select,OrdT.matrix,OrdER.vector,OrdEI.vector,OrdZ.matrix);
	}
	else 
		BaseOrderSchur(T.matrix,T.rows,T.cols,T.column_major,NULL,select,OrdT.matrix,OrdER.vector,OrdEI.vector,NULL); 
}
// Computes the annihilator of the stable subspace oof X. The stable subspace is the set of all x 
// such that the limit of (X^t)*x as t increases without bound is zero. X must be a square matrix
//
// Assumes
// 	X:	TDenseMatrix a square matrix
// 	Ann:	TDenseMatrix Annihilator matri of X
// 	AbsEig:	vecto<double> sorted absolute value of eigen-values of X
//
// Returns:
// 	returns to the calling function upon success.
// 	If any error occurs during the call of Schur or OrderSchur, exceptions will be thrown.
//
// Results:
//	Obtain the Annihilator matrix and sorted eigen values (in term of magnitude) of X.
//
void Annihilator(TDenseMatrix &Ann, TDenseVector &AbsEig, const TDenseMatrix &X)
{
	TDenseMatrix T, U;	// T: Schur form of X; U: Schur vectors
	TDenseVector eR, eI; 	// eR and eI: real and imaginary parts of the eigen values
	// Schur decomposition with Schur vectors
	Schur(T, eR, eI, U, X, true); 
	
	// Select those eigen values whose norm < 1.0
	AbsEig.Resize(X.rows); 
	int *select = new int[X.rows]; 
	int nStable = 0; 
	for (int i=0; i<eR.dim; i++)
	{	
		AbsEig.SetElement(sqrt(eR(i)*eR(i) + eI(i)*eI(i)), i); 
		if ( AbsEig[i] < 1.0)
		{
			nStable ++; 
			select[i] = 1; 
		}
		else 
			select[i] = 0; 
	}
	
	if (nStable == 0)
		Ann.Identity(X.rows); 
	else if (nStable == X.rows)
		Ann.Resize(0, X.cols); 
	else 
	{
		TDenseMatrix OrderT, OrderU;
		TDenseVector OrderER, OrderEI; 
		OrderSchur(OrderT, OrderER, OrderEI, OrderU, T, U, select, true); 
		// Ann = US(:, n_stable+1:end)'
		Ann.Resize(X.rows-nStable,X.rows); 
		for (int i=nStable; i<X.rows; i++)
			for (int j=0; j<X.rows; j++)
				Ann.SetElement(OrderU(j,i), i-nStable, j); 
	}
	std::sort(AbsEig.vector, AbsEig.vector+AbsEig.dim); 
} 
// Computes the null space of A
//
// Assumes:
// 	Z:	n * nullity(A) matrix containing the orthonormal basis of the null space of A
// 	A:	m * n matrix
// 
// Returns:
// 	nullity of (A) upon success
// 	otherwise exceptions will be thrown
//
// Results:	
//	Null space of (A)
//
//
int NullSpace(TDenseMatrix &Z, const TDenseMatrix &A)
{
        TDenseMatrix U, V;
        TDenseVector d;
        SVD(U, d, V, A, 0);	// compact=0, because we need U to be A.cols*A.cols;
        const double EPSILON_TOLERANCE = 1.06E-08;
        double small = d(0)*EPSILON_TOLERANCE;
        int rank = (A.rows < A.cols ? A.rows : A.cols) -1;
        for (; rank>=0; rank--)
        {
                if ( d(rank) > small)
                        break; 
        } 
        if (++rank < A.cols)
        {
                Z.Resize(A.cols, A.cols-rank); 
                for (int i=0; i<Z.rows; i++)
                        for (int j=0; j<Z.cols; j++)
                                Z.SetElement(V(i,j+rank), i, j);
        } 
        else
                Z.Resize(V.rows, 0);    
        return Z.cols;
}


//======================

/*
   Assumes:
    compact is zero or one (one is default)

   Results:
    X = U * diag(d,p,p) * V'

   Notes:
    In this version, U and V are not computed, which is more efficient.

    Let p be the minimum of X.rows and X.cols.  The vector d will be p-
    dimensional with non-negative elements in decending order.
*/
void SVD(TDenseVector &d, const TDenseMatrix &X)
{
  d.UniqueMemory((X.rows < X.cols) ? X.rows : X.cols);
  BaseSVD(X.matrix,X.rows,X.cols,X.column_major,(double*)NULL,d.vector,(double*)NULL,1);
}

void QR(TDenseMatrix &Q, TDenseMatrix &R, const TDenseMatrix &M, int compact)
{
  if ((&Q == &R) || (&Q == &M) || (&R == &M)) throw dw_exception("QR(): Q, R, and M must be distinct");
  int q=compact ? ((M.rows < M.cols) ? M.rows : M.cols) : M.rows;
  Q.UniqueMemory(M.rows,q,1);
  R.UniqueMemory(q,M.cols,1);
  BaseQR(M.matrix,M.rows,M.cols,M.column_major,Q.matrix,R.matrix,compact);
}

void QR(TDenseMatrix &R, const TDenseMatrix &M, int compact)
{
  if (&R == &M) throw dw_exception("QR(): R and M must be distinct");
  int q=compact ? ((M.rows < M.cols) ? M.rows : M.cols) : M.rows;
  R.UniqueMemory(q,M.cols,1);
  BaseQR(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,R.matrix,compact);
}

/*
   Results:
     Computes (rows x rows) permutation matrix P, (rows x q) lower triangular
     matrix L with ones along the diagonal, and (q x cols) upper triangular 
     matrix U such that

                                *this = P * L * U

    q is the minimum of rows and cols.
*/
void LU(TPermutationMatrix &P, TDenseMatrix &L, TDenseMatrix &U, const TDenseMatrix &X)
{
  if (&L == &U) throw dw_exception("LU(): L and U must be distinct");
  int rows=X.rows, cols=X.cols, dim=rows*cols;
  if (!rows || !cols)
    {
      P.Resize(rows);
      L.Resize(rows,0);
      U.Resize(0,cols);
    }
  else
    {
      int i, k, m;
      TLapackLU lapack(X);
      P.LapackLU(lapack);
      if (cols >= rows)
	{
	  L.UniqueMemory(rows,rows,1);
	  L.Zeros();
	  U.UniqueMemory(rows,cols,1);
	  memcpy(U.matrix,lapack.LU,dim*sizeof(double));
	  for (i=rows-1; i >= 0; i--)
	    for (L.matrix[k=i*rows+i]=1.0, k-=rows; k >= 0; k-=rows) 
	      {
		U.matrix[k]=0.0;
		L.matrix[k]=lapack.LU[k];
	      }
	}
      else
	{
	  U.UniqueMemory(cols,cols,1);
	  U.Zeros();
	  L.UniqueMemory(rows,cols,1);
	  memcpy(L.matrix,lapack.LU,dim*sizeof(double));
	  for (i=cols-1; i >= 0; i--)
	    {
	      L.matrix[k=i*rows+i]=1.0;
	      U.matrix[m=i*cols+i]=lapack.LU[k];
	      for (m+=cols, k+=rows; k < dim; m+=cols, k+=rows)
		{
		  U.matrix[m]=lapack.LU[k];
		  L.matrix[k]=0.0;
		}
	    }
	}
    }
}

/*******************************************************************************/
/******************************** Input/Output *********************************/
/*******************************************************************************/
std::ostream& operator<<(std::ostream &output, const TDenseMatrix &M)
{
  if (M.matrix)
    for (int i=0; i < M.rows; i++)
      {
	for (int j=0; j < M.cols; j++)
	  output << M.matrix[M.Index(i,j)] << ' ';
	output << std::endl;
      }
  else
    output << std::endl;
  return output;
}

std::istream& operator>>(std::istream &input, TDenseMatrix &M)
{
  if (M.matrix)
    {
      M.UniqueMemory();
      for (int i=0; i < M.rows; i++)
	for (int j=0; j < M.cols; j++) 
	  {
	    input >> M.matrix[M.Index(i,j)];
	    if (input.fail()) throw dw_exception("Error reading matrix");
	  }
    }
  return input;
}

std::ostream& operator<<(std::ostream &output, const TDenseVector &v)
{
  for (int i=0; i < v.dim; i++) output << v.vector[i] << ' ';
  output << std::endl;
  return output;
}

std::istream& operator>>(std::istream &input, TDenseVector &v)
{
  if (v.vector)
    {
      v.UniqueMemory();
      for (int i=0; i < v.dim; i++)
	{
	  input >> v.vector[i];
	  if (input.fail()) throw dw_exception("Error reading vector");
	}
    }
  return input;
}

std::ostream& operator<<(std::ostream &output, const TPermutationMatrix &P)
{
  for (int i=0; i < P.dim; i++) output << i << "->" << P(i) << ", ";
  output << std::endl;
  return output;
}

TDenseMatrix& TDenseMatrix::ReadBinary(std::fstream &file)
{
  int rows_file, cols_file;
  bool column_major_file;
  file.read((char*)(&rows_file),sizeof(int));
  file.read((char*)(&cols_file),sizeof(int));
  file.read((char*)(&column_major_file),sizeof(bool));
  UniqueMemory(rows_file,cols_file,column_major_file);
  file.read((char*)(matrix),rows*cols*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseMatrix::ReadBinary(): read failure");
  return *this;
}

void TDenseMatrix::WriteBinary(std::fstream &file)
{
  file.write((char*)(&rows),sizeof(int));
  file.write((char*)(&cols),sizeof(int));
  file.write((char*)(&column_major),sizeof(bool));
  file.write((char*)(matrix),rows*cols*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseMatrix::WriteBinary(): write failure");
}

void TDenseMatrix::WriteBinary(std::fstream &file, int brow, int erow, int bcol, int ecol)
{
  if ((brow < 0) || (brow > erow) || (erow >= rows) || (bcol < 0) || (bcol > ecol) || (ecol >= cols))
    throw dw_exception("TDenseMatrix::WriteBinary(): invalid indices");
  int r=erow-brow+1, c=ecol-bcol+1;
  file.write((char*)(&r),sizeof(int));
  file.write((char*)(&c),sizeof(int));
  file.write((char*)(&column_major),sizeof(bool));
  if (column_major)
    for (int j=bcol; j <= ecol; j++)
      file.write((char*)(matrix+j*rows+brow),r*sizeof(double));
  else
    for (int i=brow; i <= erow; i++)
      file.write((char*)(matrix+i*cols+bcol),c*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseMatrix::WriteBinary(): write failure");
}

TDenseVector& TDenseVector::ReadBinary(std::fstream &file)
{
  int dim_file;
  file.read((char*)(&dim_file),sizeof(int));
  UniqueMemory(dim_file);
  file.read((char*)(vector),dim*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseVector::ReadBinary(): read failure");
  return *this;
}

void TDenseVector::WriteBinary(std::fstream &file)
{
  file.write((char*)(&dim),sizeof(int));
  file.write((char*)(vector),dim*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseVector::WriteBinary(): write failure");
}

void TDenseVector::WriteBinary(std::fstream &file, int b, int e)
{
  if ((b < 0) || (b > e) || (e >= dim)) throw dw_exception("TDenseVector::WriteBinary(): invalid indices");
  int d=e-b+1;
  file.write((char*)(&d),sizeof(int));
  file.write((char*)(vector+b),d*sizeof(double));
  if (file.fail()) throw dw_exception("TDenseVector::WriteBinary(): write failure");
}



/*******************************************************************************/
/********************************* Permutation *********************************/
/*******************************************************************************/
void TPermutationMatrix::LapackLU(TLapackLU &lapack)
{
  Resize(lapack.rows);
  for (int i=lapack.dim-1; i >= 0; i--)
    {
      int k=lapack.p[i]-1;
      for (int j=i-1; j >= 0; j--)
	if (k == lapack.p[j]-1) k=j;
      permutation[i]=k;
    }
  for (int i=lapack.dim; i < dim; i++)
    {
      int k=i;
      for (int j=lapack.dim-1; j >= 0; j--)
	if (k == lapack.p[j]-1) k=j;
      permutation[i]=k;
    }
}

TPermutationMatrix& TPermutationMatrix::Transpose(void)
{
  if (dim > 1)
    {
      int *buffer=AllocateSharedMemory_int(dim);
      for (int i=dim-1; i >= 0; i--) buffer[permutation[i]]=i;
      ShareMemory(buffer,dim);
    }
  return *this;
}

TPermutationMatrix& TPermutationMatrix::Transpose(const TPermutationMatrix &P)
{
  int *buffer=permutation;
  if ((dim != P.dim) || (SharedMemoryCounter_int(buffer) != 1) || (buffer == P.permutation))
    buffer=AllocateSharedMemory_int(P.dim);
  for (int i=P.dim-1; i >= 0; i--) buffer[P.permutation[i]]=i;
  ShareMemory(buffer,P.dim);
  return *this;
}

/*******************************************************************************/
/********************************** TLapackLU **********************************/
/*******************************************************************************/
TLapackLU::~TLapackLU()
{
  delete[] LU;
  delete[] p;
}

/*
    The TLapackLU class is designed to store the output of the Lapack LU 
    decomposition of the matrix Y.  The integers rows and cols are the number of
    rows and columns of the matrix Y and dim is the minimum of rows and cols.  
    The array LU is a rows x cols matrix in column major form.  The lower 
    rows x dim block of LU contains the lower triangular matrix L.  The diagonal 
    of L, which contains ones, is not stored.  The upper dim x cols block 
    contains the upper triangular matrix U.  The array p stores the pivot 
    permutation.  If P(i,j) denotes the matrix obtained by permuting the ith and 
    jth rows of the rows x rows identity matrix, define P by

             P = P(0,p[0]-1)*P(1,p[1]-1)*...*P(dim-1,p[dim-1]-1).

    The LU decomposition is Y = P*L*U.
*/
TLapackLU::TLapackLU(const TDenseMatrix &Y) : X(Y)
{
  rows=Y.rows;
  cols=Y.cols;
  dim=(rows < cols) ? rows : cols;
  singular=0;
  if (!dim)
    {
      LU=(double*)NULL;
      p=(int*)NULL;
    }
  else
    {
      LU=new double[rows*cols];
      try { p=new int[dim]; }
      catch (std::bad_alloc)
	{
	  delete[] LU;
	  throw;
	}
      if (Y.column_major)
	memcpy(LU,X.matrix,rows*cols*sizeof(double));
      else
	TransposeMatrix(LU,X.matrix,rows,cols,0);
      int info;

      // Lapack Call
      dgetrf(&rows,&cols,LU,&rows,p,&info);

      if (info < 0)
	{
	  delete[] LU;
	  delete[] p;
	  throw dw_exception("Error in LAPACK LU decomposition");
	}
      else if (info > 0) singular=1;
    }
}

//===============================================================================
//=== Sort
//===============================================================================
TDenseVector& TDenseVector::Sort(bool ascending)
{
  if (dim > 1)
    {
      UniqueMemory();
      BaseQuickSortArray(vector,dim,ascending);
    }
  return *this;
}

TDenseVector& TDenseVector::Sort(const TDenseVector &v, bool ascending)
{
  *this=v;
  if (dim > 1)
    {
      UniqueMemory();
      BaseQuickSortArray(vector,dim,ascending);
    }
  return *this;
}

TDenseVector Sort(const TDenseVector &v, bool ascending)
{
  if (v.dim > 1)
    {
      TDenseVector w(v);
      w.UniqueMemory();
      BaseQuickSortArray(w.vector,w.dim,ascending);
      return w;
    }
  else
    return v;
}

TDenseMatrix& TDenseMatrix::SortRows(int c, bool ascending)
{
  if ((c < 0) || (c >= cols)) throw dw_exception("SortRows(): column index out of range");
  if (rows > 1)
    {
      ForceRowMajor();
      UniqueMemory();
      BaseQuickSortMatrix(matrix,cols,rows,c,(double*)NULL,ascending);
    }
  return *this;
}

TDenseMatrix& TDenseMatrix::SortRows(const TDenseMatrix &M, int c, bool ascending)
{
  if ((c < 0) || (c >= M.cols)) throw dw_exception("SortRows(): column index out of range");
  *this=M;
  if (rows > 1)
    {
      ForceRowMajor();
      UniqueMemory();
      BaseQuickSortMatrix(matrix,cols,rows,c,(double*)NULL,ascending);
    }
  return *this;
}

TDenseMatrix SortRows(const TDenseMatrix &M, int c, bool ascending)
{
  if ((c < 0) || (c >= M.cols)) throw dw_exception("SortRows(): column index out of range");
  if (M.rows > 1)
    {
      TDenseMatrix N(M);
      N.ForceRowMajor();
      N.UniqueMemory();
      BaseQuickSortMatrix(N.matrix,N.cols,N.rows,c,(double*)NULL,ascending);
      return N;
    }
  else
    return M;
}

TDenseMatrix& TDenseMatrix::SortColumns(int r, bool ascending)
{
  if ((r < 0) || (r >= rows)) throw dw_exception("SortColumns(): row index out of range");
  if (cols > 1)
    {
      ForceColumnMajor();
      UniqueMemory();
      BaseQuickSortMatrix(matrix,rows,cols,r,(double*)NULL,ascending);
    }
  return *this;
}

TDenseMatrix& TDenseMatrix::SortColumns(const TDenseMatrix &M, int r, bool ascending)
{
  if ((r < 0) || (r >= M.rows)) throw dw_exception("SortColumns(): row index out of range");
  *this=M;
  if (cols > 1)
    {
      ForceColumnMajor();
      UniqueMemory();
      BaseQuickSortMatrix(matrix,rows,cols,r,(double*)NULL,ascending);
    }
  return *this;
}

TDenseMatrix SortColumns(const TDenseMatrix &M, int r, bool ascending)
{
  if ((r < 0) || (r >= M.rows)) throw dw_exception("SortColumns(): row index out of range");
  if (M.cols > 1)
    {
      TDenseMatrix N(M);
      N.ForceColumnMajor();
      N.UniqueMemory();
      BaseQuickSortMatrix(N.matrix,N.rows,N.cols,r,(double*)NULL,ascending);
      return N;
    }
  else
    return M;
}
  
//===============================================================================
//=== class TIndex
//===============================================================================
void TIndex::allocate(int n)
{
  if (!(index=(int*)malloc((allocated=n+32)*sizeof(int)))) throw std::bad_alloc();
}

void TIndex::resize(int n)
{
  int *buffer=(int*)realloc(index,(n+32)*sizeof(int));
  if (!buffer) throw std::bad_alloc();
  index=buffer;
  allocated=n+32;
}

TIndex::TIndex(void)
{
  size=0;
  allocate(0);
}

TIndex::TIndex(int idx)
{
  size=1;
  allocate(0);
  index[0]=idx;
}

TIndex::TIndex(int b_idx, int e_idx)
{
  size=0;
  if (b_idx <= e_idx)
    {
      allocate(e_idx-b_idx);
      for (int idx=b_idx; idx <= e_idx; idx++) index[size++]=idx;
    }
  else
    allocate(0);
}

TIndex::TIndex(int b_idx, int inc, int e_idx)
{
  size=0;
  if (inc > 0)
    {
      if (b_idx <= e_idx)
	{
	  allocate((e_idx-b_idx)/inc);
	  for (int idx=b_idx; idx <= e_idx; idx+=inc) index[size++]=idx;
	}
      else
	allocate(0);
    }
  else if (inc < 0)
    {
      if (b_idx >= e_idx)
	{
	  allocate((b_idx-e_idx)/(-inc));
	  for (int idx=b_idx; idx >= e_idx; idx+=inc) index[size++]=idx;
	}
      else
	allocate(0);
    }
  else
    throw dw_exception("TIndex(): increment cannot be zero");
}

TIndex::TIndex(const TIndex &idx)
{
  allocate(size=idx.size);
  memcpy(index,idx.index,size*sizeof(int));
}

int TIndex::operator[](int idx) const
{
  if ((idx < 0) || (idx >= size)) throw dw_exception("TIndex[]: index out of range");
  return index[idx];
}

int& TIndex::operator[](int idx)
{
  if ((idx < 0) || (idx >= size)) throw dw_exception("TIndex[]: index out of range");
  return index[idx];
}

TIndex& TIndex::operator=(int idx)
{
  index[0]=idx;
  size=1;
  return *this;
}

TIndex& TIndex::operator=(const TIndex &idx)
{
  if (allocated < idx.size) resize(idx.size);
  memcpy(index,idx.index,(size=idx.size)*sizeof(int));
  return *this;
}

TIndex& TIndex::operator()(int idx)
{
  if (allocated <= size) resize(size);
  index[size++]=idx;
  return *this;
}

TIndex& TIndex::operator()(int b_idx, int e_idx)
{
  if (b_idx <= e_idx)
    {
      if (allocated <= size+e_idx-b_idx) resize(size+e_idx-b_idx);
      for (int idx=b_idx; idx <= e_idx; idx++) index[size++]=idx;
    }
  return *this;
}

TIndex& TIndex::operator()(int b_idx, int inc, int e_idx)
{
  if (inc > 0)
    {
      if (b_idx <= e_idx)
	{
	  if (allocated <= size+(e_idx-b_idx)/inc) resize(size+(e_idx-b_idx)/inc);
	  for (int idx=b_idx; idx <= e_idx; idx+=inc) index[size++]=idx;
	}
    }
  else if (inc < 0)
    {
      if (b_idx >= e_idx)
	{
	  if (allocated <= size+(b_idx-e_idx)/(-inc)) resize(size+(b_idx-e_idx)/(-inc));
	  for (int idx=b_idx; idx >= e_idx; idx+=inc) index[size++]=idx;
	}
    }
  else
    throw dw_exception("TIndex(): increment cannot be zero");
  return *this;
}

TIndex& TIndex::operator()(const TIndex &idx)
{
  if (allocated < size+idx.size) resize(size+idx.size);
  memcpy(index+size,idx.index,idx.size*sizeof(int));
  size+=idx.size;
  return *this;
}

// Added by HWu to merge/subtract a TIndex object and remove redundant elements
TIndex& TIndex::UniqMerge(const TIndex &right)
{
	TIndex together(*this); 
	together += right; 
	this->Clear(); 

        for (int i=0; i< together.size; i++)
        {
                int j=0; 
                while (j< this->size && together[i] != this->operator[](j))
                        j++; 
                if ( j >= this->size)
                        this->operator += (together[i]); 
        }  
	return *this; 
}

TIndex& TIndex::UniqSubtract(const TIndex &right)
{
	TIndex copy_this(*this); 
	this->Clear(); 
	for (int i=0; i<copy_this.size; i++)
	{
		int j=0; 
		while (j<right.size && copy_this[i] != right[j])
			j++; 
		if ( j>=right.size)
			this->operator += (copy_this[i]); 
	}
	copy_this = *this; 
	this->UniqMerge(copy_this); 
	return *this; 
}


// Returns the largest of the indices.  If the are no indices, then return INT_MIN.
int TIndex::Max(void)
{
  if (size == 0) return INT_MIN;
  int max=index[0];
  for (int i=size-1; i > 0; i--) 
    if (max < index[i]) max=index[i];
  return max;
}

// Returns the largest of the indices.  If the are no indices, then return INT_MAX.
int TIndex::Min(void)
{
  if (size == 0) return INT_MAX;
  int min=index[0];
  for (int i=size-1; i > 0; i--) 
    if (min > index[i]) min=index[i];
  return min;
}
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
//== Complex matrices and vectors
//===============================================================================
inline TDenseMatrixComplex::TDenseMatrixComplex(const TDenseMatrix &Re, const TDenseMatrix &Im)
{
  if ((Re.rows != Im.cols) || (Re.cols != Im.cols)) 
    throw dw_exception("TDenseMatrixComplex(): real and imaginary components must be of the same size");
  matrix=AllocateSharedMemory_double(2*Re.rows*Re.cols);
  IncrementSharedMemory_double(matrix);
  rows=Re.rows;
  cols=Re.cols;
  column_major=Re.column_major;
  if (Re.column_major == Im.column_major)
    for (int i=rows*cols-1, j=2*rows*cols-1; i >= 0; matrix[j--]=Im.matrix[i], matrix[j--]=Re.matrix[i--]);
  else
    {
      for (int stride=Im.Stride(), kk=rows*cols-1, j=2*rows*cols-1, i=kk; i >= 0; kk--)
	for (int k=kk; k >= 0; k-=stride)
	  {
	    matrix[j--]=Im.matrix[k];
	    matrix[j--]=Re.matrix[i--];
	  }
    }
}

void TDenseMatrixComplex::UniqueMemory(int r, int c, bool col_major)
{
  if ((r < 0) || (c < 0)) throw dw_exception("UniqueMemory(): number of rows or columns must be non-negative");
  if (rows*cols != r*c)
    {
      double *buffer=AllocateSharedMemory_double(2*r*c);
      FreeSharedMemory_double(matrix);    
      IncrementSharedMemory_double(buffer);
      matrix=buffer;
    }
  else
    matrix=UniqueSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
}

TDenseMatrixComplex& TDenseMatrixComplex::RandomNormal(void)
{
  if (matrix)
    {
      UniqueMemory();
      for (int i=2*rows*cols-1; i >= 0; i--) matrix[i]=dw_gaussian_rnd();
    }
  return *this;
}

TDenseMatrixComplex& TDenseMatrixComplex::RandomUniform(void)
{
  if (matrix)
    {
      UniqueMemory();
      for (int i=2*rows*cols-1; i >= 0; i--) matrix[i]=dw_uniform_rnd();
    }
  return *this;
}

std::ostream& operator<<(std::ostream &output, const TDenseMatrixComplex &M)
{
  if (M.matrix)
    for (int i=0; i < M.rows; i++)
      {
	for (int j=0; j < M.cols; j++)
	  output << M.matrix[M.Index(i,j)] << '+' << M.matrix[M.Index(i,j)+1] << "i ";
	output << std::endl;
      }
  else
    output << std::endl;
  return output;
}

//===============================================================================
//===============================================================================
//===============================================================================
