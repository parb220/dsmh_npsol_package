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

#include "dw_matrix.h"
#include "bmatrix.h"
#include "dw_error.h"
#include "dw_std.h"

#include <stdlib.h>
#include <memory.h>
#include <math.h>

/*******************************************************************************/
/********************** Allocation/Deallocation Routines ***********************/
/*******************************************************************************/
/*
   TVector CreateVector(int m)
     Allocates memory for a m-dimensional vector.  On success, returns a pointer
     to the vector.  On failure, calls dw_Error() and returns null.  The 
     routine will fail if m <= 0 (SIZE_ERR) or is unable to allocate required 
     memory (MEM_ERR).

   TMatrix CreateMatrix(int m, int n)
     Allocates memory for a (m x n) matrix.  On success, returns a pointer to the
     matrix.  On failure, calls dw_Error() and returns null.  The routine will 
     fail if m <= 0 or n <= 0 (SIZE_ERR) or is unable to allocate required memory 
     (MEM_ERR).

   void FreeVector(TVector *x)
     Assumes
       x : valid vector or null pointer
     Results
       Deallocates memory for vector if x is not null.
     Notes
       The vector x MUST have been previously allocated with a call to 
       CreateVector() or be null.

   void FreeMatrix(TMatrix *X)
     Assumes
       X : valid matrix or null pointer
     Results
       Deallocates memory for matrix if X is not null.
     Notes
       The matrix X MUST have been previously allocated with a call to
       CreateMatrix() or be null.
*/
#if (defined(STANDARD_ROW_MAJOR) || defined(STANDARD_COLUMN_MAJOR))
/**/
TVector CreateVector(int m)
{
 TVector x;
 if (m <= 0) 
  {
   dw_Error(SIZE_ERR); 
   return (TVector)NULL; 
  }
 if (x=(TVector)dw_malloc(sizeof(TVectorStructure) + (m-1)*sizeof(PRECISION)))
   DimV(x)=m;
  else
   dw_Error(MEM_ERR);
 return x;
}
/**/
void FreeVector(TVector x)
{
 if (x) dw_free(x);
}
/**/
TMatrix CreateMatrix(int m, int n)
{
 TMatrix X;
 if ((m <= 0) || (n <= 0))
  { 
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (X=(TMatrix)dw_malloc(sizeof(TMatrixStructure) + (m*n-1)*sizeof(PRECISION)))
   {
    RowM(X)=m;
    ColM(X)=n;
   }
  else
   dw_Error(MEM_ERR);
 return X;
}
/**/
void FreeMatrix(TMatrix X)
{
 if (X) dw_free(X);
}
#endif
/*-----------------------------------------------------------------------------*/
#if (defined(STRUCTURED_ROW_MAJOR) || defined(STRUCTURED_COLUMN_MAJOR) || defined(STRUCTURED_MAJOR_FORM))
/**/
TVector CreateVector(int m)
{
 TVector x;
 if (m <= 0) 
  {
   dw_Error(SIZE_ERR); 
   return (TVector)NULL; 
  }
 if (x=(TVector)dw_malloc(sizeof(TVectorStructure)))
   if (x->x=(PRECISION*)dw_malloc(m*sizeof(PRECISION)))
     DimV(x)=m;
   else
     {
       dw_free(x);
       dw_Error(MEM_ERR);
       return (TVector)NULL;
     }
  else
   dw_Error(MEM_ERR);
 return x;
}
/**/
void FreeVector(TVector x)
{
 if (x) 
   {
     if (x->x) dw_free(x->x);
     dw_free(x);
   }
}
/**/
TMatrix CreateMatrix(int m, int n)
{
 TMatrix X;
 if ((m <= 0) || (n <= 0))
  { 
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (X=(TMatrix)dw_malloc(sizeof(TMatrixStructure)))
   if (X->x=(PRECISION*)dw_malloc(m*n*sizeof(PRECISION)))
     {
       RowM(X)=m;
       ColM(X)=n;
#if defined STRUCTURED_MAJOR_FORM
       SetMajorForm(X,(rand() % 2) ? 0 : 1);
#endif
     }
   else
     {
       dw_free(X);
       dw_Error(MEM_ERR);
       return (TMatrix)NULL;
     }
  else
   dw_Error(MEM_ERR);
 return X;
}
/**/
void FreeMatrix(TMatrix X)
{
 if (X) 
   {
     if (X->x) dw_free(X->x);
     dw_free(X);
   }
}
#endif
/*-----------------------------------------------------------------------------*/
#if (defined(LEGACY_ROW_MAJOR))
/**/
TVector CreateVector(int m)
{
 TVector x;
 if (m <= 0) 
  {
   dw_Error(SIZE_ERR); 
   return (TVector)NULL; 
  }
 if (x=(TVector)((int *)dw_malloc(m*sizeof(PRECISION)+sizeof(int))+1))
   V_DIM(x)=m;
  else
   dw_Error(MEM_ERR);
 return x;
}
/**/
void FreeVector(TVector x)
{
 if (x) dw_free((int *)x-1);
}
/**/
TMatrix CreateMatrix(int m, int n)
{
 TMatrix X;
 int i;
 if ((m <= 0) || (n <= 0))
  { 
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (X=(TMatrix)((int *)dw_malloc(m*(sizeof(PRECISION *)+n*sizeof(PRECISION))+2*sizeof(int))+2))
   {
    M_ROW(X)=m;
    M_COL(X)=n;
    X[0]=(PRECISION *)(X+m);
    for (i=1; i < m; i++) X[i]=X[i-1]+n;
   }
  else
   dw_Error(MEM_ERR);
 return X;
}
/**/
void FreeMatrix(TMatrix X)
{
 if (X) dw_free((int *)X-2);
}
#endif
/*-----------------------------------------------------------------------------*/
#if (defined(TZ_COLUMN_MAJOR))
/**/
TVector CreateVector(int m)
{
 TVector x;
 if (m <= 0) 
  {
   dw_Error(SIZE_ERR); 
   return (TVector)NULL; 
  }
 if (x=(TVector)dw_malloc(sizeof(TSdvector)))
   if (!(pElementV(x)=(PRECISION*)dw_malloc(m*sizeof(PRECISION))))
     {
       dw_free(x);
       dw_Error(MEM_ERR);
       return (TVector)NULL;
     }
   else
     {
       DimV(x)=m;
       x->flag=V_DEF;
     }
  else
   dw_Error(MEM_ERR);
 return x;
}
/**/
void FreeVector(TVector x)
{
 if (x) 
   {
     if (pElementV(x)) dw_free(pElementV(x));
     dw_free(x);
   }
}
/**/
TMatrix CreateMatrix(int m, int n)
{
 TMatrix X;
 if ((m <= 0) || (n <= 0))
  { 
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (X=(TMatrix)dw_malloc(sizeof(TSdmatrix)))
   if (!(pElementM(X)=(PRECISION*)dw_malloc(m*n*sizeof(PRECISION))))
     {
       dw_free(X);
       dw_Error(MEM_ERR);
       return (TMatrix)NULL;
     }
   else
     {
       RowM(X)=m;
       ColM(X)=n;
       X->flag=M_GE;
     }
  else
   dw_Error(MEM_ERR);
 return X;
}
/**/
void FreeMatrix(TMatrix X)
{
 if (X) 
   {
     if (pElementM(X)) dw_free(pElementM(X));
     dw_free(X);
   }
}
#endif
/*-----------------------------------------------------------------------------*/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/************************** Initialization Routines ****************************/
/*******************************************************************************/
/*
   Assumes
     x : m vector

   Results
     x(i) = c

   Returns
     x
*/
TVector InitializeVector(TVector x, PRECISION c)
{
 int i;
 if (!x)
   dw_Error(NULL_ERR);
  else
   for (i=DimV(x); --i >= 0; ElementV(x,i)=c);
 return x;
}

/*
   Assumes
     X : m x n matrix

   Results
     X(i,j) = c

   Returns
     X
*/
TMatrix InitializeMatrix(TMatrix X, PRECISION c)
{
 int i;
 PRECISION *pX;
 if (!X)
   dw_Error(NULL_ERR);
  else
   for (pX=pElementM(X), i=RowM(X)*ColM(X)-1; i >= 0; i--) pX[i]=c;
 return X;
}
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/**************************** Assignment Routines ******************************/
/*******************************************************************************/
/*
   Assumes
     x : m-vector or null pointer
     y : m-vector

   Results
     x = y.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call dw_GetError() to
     determine the cause of failure.
*/
TVector EquateVector(TVector x, TVector y)
{
 if (!y) 
  {
   dw_Error(NULL_ERR); 
   return (TVector)NULL; 
  }
 if (!x)
   { 
    if (!(x=CreateVector(DimV(y)))) 
     return (TVector)NULL; 
   }
 else
   if (x == y)
     return x;
   else
     if (DimV(x) != DimV(y))
       { 
	 dw_Error(SIZE_ERR); 
	 return (TVector)NULL; 
       }
 memcpy(pElementV(x),pElementV(y),DimV(y)*sizeof(PRECISION));
 return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix

   Results
     X = Y.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix EquateMatrix(TMatrix X, TMatrix Y)
{
  if (!Y) 
    {
      dw_Error(NULL_ERR); 
      return (TMatrix)NULL; 
    }
  if (!X)
    { 
      if (!(X=CreateMatrix(RowM(Y),ColM(Y)))) 
	return (TMatrix)NULL; 
    }
  else
    if (X == Y) 
      return X;
    else
      if ((RowM(Y) != RowM(X)) || (ColM(Y) != ColM(X)))
	{
	  dw_Error(SIZE_ERR); 
	  return (TMatrix)NULL; 
	}
  memcpy(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
  SetMajorForm(X,MajorForm(Y));
  return X;
}

/*
   Assumes
     X : m x m matrix or null pointer

   Results
     X is set to the m x m identity matrix.  If X is null pointer, X 
     is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix IdentityMatrix(TMatrix X, int m)
{
 int i;
 PRECISION *pX;
 if (!X)
   { 
    if (!(X=CreateMatrix(m,m))) 
     return (TMatrix)NULL; 
   }
  else
   if ((m != RowM(X)) || (m != ColM(X)))
    { 
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 for (pX=pElementM(X), i=m*m-1; i >= 0; i--) pX[i]=0.0;
 for (i=m*m-1; i >= 0; i-=m+1) pX[i]=1.0;
 return X;
}

/*
   Assumes
     X : m x n matrix or null pointer

   Results
     X is set to the m x n zero matrix.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix ZeroMatrix(TMatrix X, int m, int n)
{
 int i;
 PRECISION *pX;
 if (!X)
   { 
    if (!(X=CreateMatrix(m,n))) 
     return (TMatrix)NULL; 
   }
  else
   if ((m != RowM(X)) || (n != ColM(X)))
    { 
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 for (pX=pElementM(X), i=m*m-1; i >= 0; i--) pX[i]=0.0;
 return X;
}

/*
   Assumes
     X : a null pointer or r x s matrix, where r = m and s >= m or r >= m and s = m.
     y : m-vector

   Results
     X = diag(y).  If X is null pointer, a square matrix X is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix DiagonalMatrix(TMatrix X, TVector y)
{
 int i, j, k;
 PRECISION *pX;
 if (!y) 
   { 
     dw_Error(NULL_ERR); 
     return (TMatrix)NULL; 
   }
 if (!X)
   { 
     if (!(X=CreateMatrix(DimV(y),DimV(y)))) 
       return (TMatrix)NULL; 
   }
 else
   if (DimV(y) != ((RowM(X) < ColM(X)) ? RowM(X) : ColM(X)))
     { 
       dw_Error(SIZE_ERR); 
       return (TMatrix)NULL; 
     }
 for (pX=pElementM(X), i=RowM(X)*ColM(X)-1; i >= 0; i--) pX[i]=0.0;
 for (k=(MajorForm(X) ? RowM(X) : ColM(X))+1, j=DimV(y)-1, pX=&ElementM(X,j,j); j >= 0; pX-=k, j--) *pX=ElementV(y,j);
 return X;
}

/*
   Assumes
     X : m x 1 matrix or null pointer
     y : m-vector

   Results
     X is equal to the column vector y.  If X is null pointer, X is 
     created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix ColumnMatrix(TMatrix X, TVector y)
{
 if (!y) 
  { 
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   { 
    if (!(X=CreateMatrix(DimV(y),1))) 
     return (TMatrix)NULL; 
   }
  else
   if ((DimV(y) != RowM(X)) || (1 != ColM(X)))
    { 
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 memcpy(pElementM(X),pElementV(y),DimV(y)*sizeof(PRECISION));
 return X;
}

/*
   Assumes
     X : 1 x m matrix or null pointer
     y : m-vector

   Results
     X is equal to the row vector y.  If X is null pointer, X is 
     created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix RowMatrix(TMatrix X, TVector y)
{
 if (!y) 
  { 
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   { 
    if (!(X=CreateMatrix(1,DimV(y)))) 
     return (TMatrix)NULL; 
   }
  else
   if ((1 != RowM(X)) || (DimV(y) != ColM(X)))
    { 
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 memcpy(pElementM(X),pElementV(y),DimV(y)*sizeof(PRECISION));
 return X;
}

/*
   Assumes
     x : m vector or null pointer
     y : m vector

   Results
     x(i) = abs(y(i)).  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x and y do not have to be distinct vectors.
*/
TVector AbsV(TVector x, TVector y)
{
 if (!y) 
  { 
   dw_Error(NULL_ERR); 
   return (TVector)NULL; 
  }
 if (!x)
   { 
    if (!(x=CreateVector(DimV(y)))) 
     return (TVector)NULL; 
   }
  else
   if (DimV(x) != DimV(y)) 
    { 
     dw_Error(SIZE_ERR); 
     return (TVector)NULL; 
    }
 bAbs(pElementV(x),pElementV(y),DimV(y));
 return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix

   Results
     X(i,j) = abs(Y(i,j)).  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X and Y do not have to be distinct matrices
*/
TMatrix AbsM(TMatrix X, TMatrix Y)
{
 if (!Y) 
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   { 
    if (!(X=CreateMatrix(RowM(Y),ColM(Y))))
     return (TMatrix)NULL; 
   }
  else
   if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Y)))
    {
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 bAbs(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y));
 SetMajorForm(X,MajorForm(Y));
 return X;
}

/*
   Assumes
     x : m vector or null pointer
     y : m vector

   Results
     x = -x.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x and y do not have to be distinct vectors.
*/
TVector MinusV(TVector x, TVector y)
{
 if (!y) 
  {
   dw_Error(NULL_ERR); 
   return (TVector)NULL; 
  }
 if (!x)
   {
    if (!(x=CreateVector(DimV(y))))
     return (TVector)NULL; 
   }
  else
   if (DimV(x) != DimV(y))
    {
     dw_Error(SIZE_ERR); 
     return (TVector)NULL; 
    }
 bNegative(pElementV(x),pElementV(y),DimV(y));
 return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix

   Results
     X = -Y.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X and Y do not have to be distinct matrices.
*/
TMatrix MinusM(TMatrix X, TMatrix Y)
{
 if (!Y)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   {
    if (!(X=CreateMatrix(RowM(Y),ColM(Y))))
     return (TMatrix)NULL; 
   }
  else
   if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Y)))
    {
     dw_Error(SIZE_ERR); 
     return (TMatrix)NULL; 
    }
 bNegative(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y));
 SetMajorForm(X,MajorForm(Y));
 return X;
}

/*
   Assumes
     X : n x m matrix or null pointer
     Y : m x n matrix

   Results
     X = Y'.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     If Y is square, X and Y do not have to be distinct matrices
*/
TMatrix Transpose(TMatrix X, TMatrix Y)
{
 if (!Y)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (X == Y)
   {
    if (RowM(Y) != ColM(Y))
     {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
     }
    bTransposeInPlace(pElementM(Y),RowM(Y));
   }
  else
   {
    if (!X)
      {
       if (!(X=CreateMatrix(ColM(Y),RowM(Y)))) 
        return (TMatrix)NULL;
      }
     else
      if ((RowM(X) != ColM(Y)) || (ColM(X) != RowM(Y)))
       {
        dw_Error(SIZE_ERR); 
        return (TMatrix)NULL; 
       }
    if (MajorForm(X) == MajorForm(Y))
      bTranspose(pElementM(X),pElementM(Y),RowM(Y),ColM(Y),MajorForm(Y));
    else
      memcpy(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
   }
 return X;
}

/*
   Assumes
     X : (rows x cols) matrix or null pointer
     Y : (r x c) matrix, where r >= brow+rows and c >= bcol+cols

   Results
     X(i,j) = Y(brow+i,bcol+j), for 0 <= i < rows and 0 <= j < cols.  If
     X is null pointer, then the matrix X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix SubMatrix(TMatrix X, TMatrix Y, int brow, int bcol, int rows, int cols)
{
  int i, j, k, s, r, m;
  PRECISION *pX, *pY;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((brow+rows > RowM(Y)) || (bcol+cols > ColM(Y)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
 if (Y == X)
    {
      dw_Error(ARG_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(rows,cols)))
	return (TMatrix)NULL;
    }
  else
    if ((rows != RowM(X)) || (cols != ColM(X)))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
 
  if (MajorForm(Y) == MajorForm(X))
    if (MajorForm(Y))
      {
	if (rows == RowM(Y))
	  memcpy(pElementM(X),&ElementM(Y,0,bcol),rows*cols*sizeof(PRECISION));
	else
	  for (s=rows*sizeof(PRECISION), j=cols-1; j >= 0; j--)
	    memcpy(&ElementM(X,0,j),&ElementM(Y,brow,bcol+j),s);
      }
    else
      {
	if (cols == ColM(Y))
	  memcpy(pElementM(X),&ElementM(Y,brow,0),rows*cols*sizeof(PRECISION));
	else
	  for (s=cols*sizeof(PRECISION), j=rows-1; j >= 0; j--)
	    memcpy(&ElementM(X,j,0),&ElementM(Y,brow+j,bcol),s);
      }
  else
    if (MajorForm(X))
      {
	s=ColM(Y), m=brow*s, r=(brow+rows-1)*s, pX=pElementM(X), pY=pElementM(Y);
	for (k=rows*cols-1, j=bcol+cols-1; j >= bcol; j--)
	  for (i=r+j; i >= m ; k--, i-=s)
	    pX[k]=pY[i];
      }
    else
      {
	s=RowM(Y), m=bcol*s, r=(bcol+cols-1)*s, pX=pElementM(X), pY=pElementM(Y), k=rows*cols-1;
	for (i=brow+rows-1; i >= brow; i--)
	  for (j=r+i; j >= m ; k--, j-=s)
	    pX[k]=pY[j];
      }
 
  return X;
}

/*
   Assumes
     x : d-dimensional vector or null pointer
     y : n-dimensional vector, where n >= b+d

   Results
     x(i) = y(b+i), for 0 <= i < d.  If x is a null pointer, then the vector x is
     created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to determine 
     the cause of failure.
*/
TVector SubVector(TVector x, TVector y, int b, int d)
{
  if (!y)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if (b+d > DimV(y))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(d)))
	return (TVector)NULL;
    }
  else
    if (d != DimV(x))
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }
  if (x != y)
    memcpy(pElementV(x),pElementV(y)+b,d*sizeof(PRECISION));
  return x;
}

/*
   Assumes
     X : (rX x cX) matrix, where rX >= brow_X+rows and cX >= bcol_X+cols
     Y : (rY x cY) matrix, where rY >= brow_Y+rows and cY >= bcol_Y+cols

   Results
     X(brow_X+i,bcol_X+j) = Y(brow_Y+i,bcol_Y+j), for 0 <= i < rows and 
     0 <= j < cols.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix InsertSubMatrix(TMatrix X, TMatrix Y, int brow_X, int bcol_X, int brow_Y, int bcol_Y, int rows, int cols)
{
  int i, j, s;
  if (!X || !Y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((brow_X < 0) || (bcol_X < 0) || (brow_Y < 0) || (bcol_Y < 0) || (rows <= 0) || (cols <= 0)
      || (brow_Y+rows > RowM(Y)) || (bcol_Y+cols > ColM(Y)) || (brow_X+rows > RowM(X)) || (bcol_X+cols > ColM(X)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (Y == X)
    {
      dw_Error(ARG_ERR);
      return (TMatrix)NULL;
    } 
  if (MajorForm(Y) == MajorForm(X))
    if (MajorForm(Y))
      {
	if ((rows == RowM(Y)) && (rows == RowM(X)))
	  memcpy(&ElementM(X,0,bcol_X),&ElementM(Y,0,bcol_Y),rows*cols*sizeof(PRECISION));
	else
	  for (s=rows*sizeof(PRECISION), j=cols-1; j >= 0; j--)
	    memcpy(&ElementM(X,brow_X,bcol_X+j),&ElementM(Y,brow_Y,bcol_Y+j),s);
      }
    else
      {
	if ((cols == ColM(Y)) && (cols == ColM(X)))
	  memcpy(&ElementM(X,brow_X,0),&ElementM(Y,brow_Y,0),rows*cols*sizeof(PRECISION));
	else
	  for (s=cols*sizeof(PRECISION), i=rows-1; i >= 0; i--)
	    memcpy(&ElementM(X,brow_X+i,bcol_X),&ElementM(Y,brow_Y+i,bcol_Y),s);
      }
  else
    for (i=rows-1; i >= 0; i--)
      for (j=cols-1; j >= 0; j--)
	ElementM(X,brow_X+i,bcol_X+j)=ElementM(Y,brow_Y+i,bcol_Y+j);
    
  return X;
}

/*
   Assumes
     X   : m x n matrix
     y   : m-vector
     col : 0 <= col < n

   Results
     X(i,col) = y(i), for 0 <= i < m.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix InsertColumnVector(TMatrix X, TVector y, int col)
{
  int i, n;
  PRECISION *pX;
  if (!X || !y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL; 
    }
  if ((DimV(y) != RowM(X)) || (col < 0) || (ColM(X) <= col))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (MajorForm(X))
    memcpy(pElementM(X)+col*DimV(y),pElementV(y),DimV(y)*sizeof(PRECISION));
  else
    for (pX=pElementM(X)+(n=ColM(X))*(i=DimV(y)-1)+col; i >= 0; pX-=n, i--) *pX=ElementV(y,i);
  return X;
}

/*
   Assumes
     X   : m x n matrix
     y   : n-vector
     row : 0 <= row < n

   Results
     X(row,j) = y(j), for 0 <= j < n.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix InsertRowVector(TMatrix X, TVector y, int row)
{
  int j, m;
  PRECISION *pX;
  if (!X || !y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL; 
    }
  if ((DimV(y) != ColM(X)) || (row < 0) || (RowM(X) <= row))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (MajorForm(X))
    for (pX=pElementM(X)+(m=RowM(X))*(j=DimV(y)-1)+row; j >= 0; pX-=m, j--) *pX=ElementV(y,j);
  else
    memcpy(pElementM(X)+row*DimV(y),pElementV(y),DimV(y)*sizeof(PRECISION));
  return X;
}

/*
   Assumes
     x   : m-vector or null pointer
     Y   : m x n matrix
     col : 0 <= col < n

   Results
     x(i) = Y(i,col), for 0 <= i < m.  If x is null pointer, then the vector x
     is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TVector ColumnVector(TVector x, TMatrix Y, int col)
{
  int i;
  PRECISION *pY;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL; 
    }
  if ((col < 0) || (ColM(Y) <= col))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(RowM(Y))))
	return (TVector)NULL;
    }
  else
    if (DimV(x) != RowM(Y))
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }
  if (MajorForm(Y))
    memcpy(pElementV(x),&ElementM(Y,0,col),DimV(x)*sizeof(PRECISION));
  else
    for (pY=&ElementM(Y,i=DimV(x)-1,col); i >= 0; pY-=ColM(Y), i--) ElementV(x,i)=*pY;
  return x;
}

/*
   Assumes
     x   : n-vector or null pointer
     Y   : m x n matrix
     row : 0 <= row < m

   Results
     x(j) = Y(row,j), for 0 <= j < n.  If x is null pointer, then the vector x
     is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.  
*/
TVector RowVector(TVector x, TMatrix Y, int row)
{
  int j, m;
  PRECISION *pY;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if ((row < 0) || (RowM(Y) <= row))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(ColM(Y))))
	return (TVector)NULL;
    }
  else
    if (DimV(x) != ColM(Y))
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }
  if (MajorForm(Y))
    for (pY=pElementM(Y)+row+(m=RowM(Y))*(j=DimV(x)-1); j >= 0; pY-=m, j--) ElementV(x,j)=*pY;
  else
    memcpy(pElementV(x),pElementM(Y)+row*DimV(x),DimV(x)*sizeof(PRECISION));
  return x;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/***************************** Addition Routines *******************************/
/*******************************************************************************/
/*
   Assumes
     x : m-vector or null pointer
     y : m-vector
     z : m-vector

   Results
     x = y + z.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x, y, and z do not have to be distinct vectors
*/
TVector AddVV(TVector x, TVector y, TVector z)
{
 if (!y || !z)
  {
   dw_Error(NULL_ERR);
   return (TVector)NULL;
  }
 if (DimV(y) != DimV(z))
  {
   dw_Error(SIZE_ERR);
   return (TVector)NULL;
  }
 if (!x)
   {
    if (!(x=CreateVector(DimV(y))))
     return (TVector)NULL;
   }
  else
   if (DimV(x) != DimV(y))
    {
     dw_Error(SIZE_ERR);
     return (TVector)NULL;
    }
 bAddV(pElementV(x),pElementV(y),pElementV(z),DimV(y));
 return x;
}

/*
   Assumes
     x : m-vector or null pointer
     y : m-vector
     z : m-vector

   Results
     x = y - z.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x, y, and z do not have to be distinct vectors
*/
TVector SubtractVV(TVector x, TVector y, TVector z)
{
 if (!y || !z)
  {
   dw_Error(NULL_ERR);
   return (TVector)NULL;
  }
 if (DimV(y) != DimV(z))
  {
   dw_Error(SIZE_ERR);
   return (TVector)NULL;
  }
 if (!x)
   {
    if (!(x=CreateVector(DimV(y))))
     return (TVector)NULL;
   }
  else
   if (DimV(x) != DimV(y))
    {
     dw_Error(SIZE_ERR);
     return (TVector)NULL;
    }
 bSubtractV(pElementV(x),pElementV(y),pElementV(z),DimV(y));
 return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix
     Z : m x n matrix

   Results
     X = Y + Z.  If X is null pointer, x is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X, Y, and Z do not have to be distinct matrices
*/
TMatrix AddMM(TMatrix X, TMatrix Y, TMatrix Z)
{
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((RowM(Y) != RowM(Z)) || (ColM(Y) != ColM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Z),ColM(Z))))
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Z)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
  bAddM(pElementM(X),pElementM(Y),pElementM(Z),RowM(Z),ColM(Z),MajorForm(X),MajorForm(Y),MajorForm(Z));
  return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix
     Z : m x n matrix

   Results
     X = Y - Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X, Y, and Z do not have to be distinct matrices
*/
TMatrix SubtractMM(TMatrix X, TMatrix Y, TMatrix Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR);
   return (TMatrix)NULL;
  }
 if ((RowM(Y) != RowM(Z)) || (ColM(Y) != ColM(Z)))
  {
   dw_Error(SIZE_ERR);
   return (TMatrix)NULL;
  }
 if (!X)
   {
    if (!(X=CreateMatrix(RowM(Z),ColM(Z))))
     return (TMatrix)NULL;
   }
  else
   if ((RowM(X) != RowM(Z)) || (ColM(X) != ColM(Z)))
    {
     dw_Error(SIZE_ERR);
     return (TMatrix)NULL;
    }
  bSubtractM(pElementM(X),pElementM(Y),pElementM(Z),RowM(Z),ColM(Z),MajorForm(X),MajorForm(Y),MajorForm(Z));
  return X;
}

/*
   Assumes
     x : m vector or null pointer
     a : scalar
     y : m vector
     b : scalar
     z : m vector

   Results
     x = a*y + b*z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x, y, and z do not have to be distinct vectors
*/
TVector LinearCombinationV(TVector x, PRECISION a, TVector y, PRECISION b, TVector z)
{
  if (!y || !z)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if (DimV(y) != DimV(z))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(DimV(z))))
	return (TVector)NULL;
    }
  else
    if (DimV(x) != DimV(z))
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }
  bLinearCombinationV(pElementV(x),a,pElementV(y),b,pElementV(z),DimV(z));
  return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     a : scalar
     Y : m x n matrix
     b : scalar
     Z : m x n matrix

   Results
     X = a*Y + b*Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X, Y, and Z do not have to be distinct matrices
*/
TMatrix LinearCombinationM(TMatrix X, PRECISION a, TMatrix Y, PRECISION b, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((RowM(Y) != RowM(Z)) || (ColM(Y) != ColM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Z),ColM(Z))))
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Z)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
  if ((X == Z) || (X == Y))
    {
      if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR);
	  return (TMatrix)NULL;
	}
      memcpy(ptr,pElementM(X),RowM(X)*ColM(X)*sizeof(PRECISION));
      bLinearCombinationM(pElementM(X),a,(Y == X) ? ptr : pElementM(Y),b,(Z == X) ? ptr : pElementM(Z),
			  RowM(X),ColM(X),MajorForm(X),MajorForm(Y),MajorForm(Z));
      dw_free(ptr);
    }
  else
    bLinearCombinationM(pElementM(X),a,pElementM(Y),b,pElementM(Z),RowM(X),ColM(X),MajorForm(X),MajorForm(Y),MajorForm(Z));
  return X;
}

/*
   Assumes
     a : scalar
     x : m-vector
     b : scalar
     Y : m x n matrix
     z : n-vector

   Results
     x = a*x + b*Y*z

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x and z do not have to be distinct vectors.
*/
TVector UpdateProductMV(PRECISION a, TVector x, PRECISION b, TMatrix Y, TVector z)
{
  PRECISION *ptr;
  if (!x || !Y || ! z)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if ((DimV(x) != RowM(Y)) || (ColM(Y) != DimV(z)))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (x == z)
    {
      if (!(ptr=(PRECISION*)dw_malloc(DimV(x)*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR); 
	  return (TVector)NULL; 
	}
      memcpy(ptr,pElementV(x),DimV(x)*sizeof(PRECISION));
      bProductMV_Update(pElementV(x),ptr,pElementM(Y),a,b,DimV(x),DimV(x),MajorForm(Y));
      dw_free(ptr);
    }
  else
    bProductMV_Update(pElementV(x),pElementV(z),pElementM(Y),a,b,DimV(x),DimV(z),MajorForm(Y));
  return x;
}

/*
   Assumes
     a : scalar
     x : m-vector
     b : scalar
     y : n-vector 
     Z : n x m matrix

   Results
     x = a*x + b*y*Z

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x and y do not have to be distinct vectors.
*/
TVector UpdateProductVM(PRECISION a, TVector x, PRECISION b, TVector y, TMatrix Z)
{
  PRECISION *ptr;
  if (!x || !y || ! Z)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if ((DimV(x) != ColM(Z)) || (RowM(Z) != DimV(y)))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (x == y)
    {
      if (!(ptr=(PRECISION*)dw_malloc(DimV(x)*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR); 
	  return (TVector)NULL; 
	}
      memcpy(ptr,pElementV(x),DimV(x)*sizeof(PRECISION));
      bProductMV_Update(pElementV(x),ptr,pElementM(Z),a,b,DimV(x),DimV(x),1^MajorForm(Y));
      dw_free(ptr);
    }
  else
    bProductMV_Update(pElementV(x),pElementV(y),pElementM(Z),a,b,DimV(x),DimV(y),1^MajorForm(Y));
  return x;
}

/*
   Assumes
     a : scalar
     X : m x n matrix
     b : scalar
     Y : m x k matrix
     Z : k x n matrix

   Results
     X = a*X + b*Y*Z

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X, Y, and Z do not have to be distinct matrices.
*/
TMatrix UpdateProductMM(PRECISION a, TMatrix X, PRECISION b, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!X || !Y || ! Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)) || (ColM(Y) != RowM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if ((X == Y) || (X == Z))
    {
      if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR); 
	  return (TMatrix)NULL; 
	}
      memcpy(ptr,pElementM(X),RowM(X)*ColM(X)*sizeof(PRECISION));
      bProductMM_Update(pElementM(X),(X == Y) ? ptr : pElementM(Y),(X == Z) ? ptr : pElementM(Z),a,b,RowM(X),
			    ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),MajorForm(Z));
      dw_free(ptr);
    }
  else
    bProductMM_Update(pElementM(X),pElementM(Y),pElementM(Z),a,b,RowM(X),ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),MajorForm(Z));
  return X;
}

/*
   Assumes
     a : scalar
     X : m x n matrix
     b : scalar
     Y : m x k matrix
     Z : k x n matrix

   Results
     X = a*X + b*Y'*Z

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X, Y, and Z do not have to be distinct matrices.
*/
TMatrix UpdateTransposeProductMM(PRECISION a, TMatrix X, PRECISION b, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!X || !Y || ! Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)) || (ColM(Y) != RowM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if ((X == Y) || (X == Z))
    {
      if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR); 
	  return (TMatrix)NULL; 
	}
      memcpy(ptr,pElementM(X),RowM(X)*ColM(X)*sizeof(PRECISION));
      bProductMM_Update(pElementM(X),(X == Y) ? ptr : pElementM(Y),(X == Z) ? ptr : pElementM(Z),a,b,RowM(X),
			    ColM(X),ColM(Y),MajorForm(X),1^MajorForm(Y),MajorForm(Z));
      dw_free(ptr);
    }
  else
    bProductMM_Update(pElementM(X),pElementM(Y),pElementM(Z),a,b,RowM(X),ColM(X),ColM(Y),MajorForm(X),1^MajorForm(Y),MajorForm(Z));
  return X;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/************************** Multiplication Routines ****************************/
/*******************************************************************************/
/*
   Assumes
     x : m-vector or null pointer
     y : m-vector
     s : scalar

   Results
     x = s*y.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     x and y do not have to be distinct
*/
TVector ProductSV(TVector x, PRECISION s, TVector y)
{
 if (!y)
  {
   dw_Error(NULL_ERR);
   return (TVector)NULL;
  }
 if (!x)
   {
    if (!(x=CreateVector(DimV(y))))
     return (TVector)NULL;
   }
  else
   if (DimV(x) != DimV(y))
    {
     dw_Error(SIZE_ERR);
     return (TVector)NULL;
    }
 bMultiply(pElementV(x),pElementV(y),s,DimV(y));
 return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     y : m x n matix
     s : scalar

   Results
     X = s*Y.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     X and Y do not have to be distinct
*/
TMatrix ProductSM(TMatrix X, PRECISION s, TMatrix Y)
{
 if (!Y)
  {
   dw_Error(NULL_ERR);
   return (TMatrix)NULL;
  }
 if (!X)
   {
    if (!(X=CreateMatrix(RowM(Y),ColM(Y))))
     return (TMatrix)NULL;
   }
  else
   if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Y)))
    {
     dw_Error(SIZE_ERR);
     return (TMatrix)NULL;
    }
 bMultiply(pElementM(X),pElementM(Y),s,RowM(Y)*ColM(Y));
 SetMajorForm(X,MajorForm(Y));
 return X;
}

/*
   Assumes
     x : n-vector or null pointer
     y : m-vector
     Z : m x n matrix

   Results
     x = y * Z.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     If Z is square, x and y do not have to be distinct
*/
TVector ProductVM(TVector x, TVector y, TMatrix Z)
{
 PRECISION *ptr;
 if (!y || !Z)
  {
   dw_Error(NULL_ERR);
   return (TVector)NULL;
  }
 if (DimV(y) != RowM(Z))
  {
   dw_Error(SIZE_ERR);
   return (TVector)NULL;
  }
 if (!x)
   {
    if (!(x=CreateVector(ColM(Z))))
     return (TVector)NULL;
   }
  else
   {
    if (DimV(x) != ColM(Z))
     {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
     }
    if (x == y)
      {
       if (!(ptr=(PRECISION*)dw_malloc(DimV(x)*sizeof(PRECISION))))
        {
         dw_Error(MEM_ERR);
         return (TVector)NULL;
        }
       //bMatrixMultiply(ptr,pElementV(y),pElementM(Z),1,DimV(x),DimV(y),0,0,MajorForm(Z));
       bProductVM(ptr,pElementV(y),pElementM(Z),RowM(Z),ColM(Z),MajorForm(Z));
       memcpy(pElementV(x),ptr,DimV(x)*sizeof(PRECISION));
       dw_free(ptr);
       return x;
      }
   }
 //bMatrixMultiply(pElementV(x),pElementV(y),pElementM(Z),1,DimV(x),DimV(y),0,0,MajorForm(Z));
 bProductVM(pElementV(x),pElementV(y),pElementM(Z),RowM(Z),ColM(Z),MajorForm(Z));
 return x;
}

/*
   Assumes
     x : m-vector or null pointer
     Y : m x n matrix
     z : n-vector

   Results
     x = Y * z.  If x is null pointer, x is created.

   Returns
     Returns x upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     If Y is square, x and z do not have to be distinct
*/
TVector ProductMV(TVector x, TMatrix Y, TVector z)
{
  PRECISION *ptr;
  if (!Y || !z)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if (ColM(Y) != DimV(z))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(RowM(Y))))
	return (TVector)NULL;
    }
  else
    {
      if (DimV(x) != RowM(Y))
	{
	  dw_Error(SIZE_ERR);
	  return (TVector)NULL;
	}
      if (x == z)
	{
	  if (!(ptr=(PRECISION*)dw_malloc(DimV(x)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TVector)NULL;
	    }
	  bProductMV(ptr,pElementV(z),pElementM(Y),RowM(Y),ColM(Y),MajorForm(Y));
	  memcpy(pElementV(x),ptr,DimV(x)*sizeof(PRECISION));
	  dw_free(ptr);
	  return x;
	}
    }
  bProductMV(pElementV(x),pElementV(z),pElementM(Y),RowM(Y),ColM(Y),MajorForm(Y));
  return x;
}


/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x r matrix
     Z : r x n matrix

   Results
     X = Y * Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Size permitting, X, Y and Z do not have to be distinct.
*/
TMatrix ProductMM(TMatrix X, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if (ColM(Y) != RowM(Z))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Y),ColM(Z)))) 
	return (TMatrix)NULL;
    }
  else
    {
      if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)))
	{
	  dw_Error(SIZE_ERR); 
	  return (TMatrix)NULL; 
	}
      if ((X == Y) || (X == Z))
	{
	  if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR); 
	      return (TMatrix)NULL; 
	    }
	  bProductMM(ptr,pElementM(Y),pElementM(Z),RowM(X),ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),MajorForm(Z));
	  memcpy(pElementM(X),ptr,RowM(X)*ColM(X)*sizeof(PRECISION));
	  dw_free(ptr);
	  return X;
	}
    }
  bProductMM(pElementM(X),pElementM(Y),pElementM(Z),RowM(X),ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),MajorForm(Z));
  return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix
     Z : n x n upper triangular matrix

   Results
     X = Y * Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Size permitting, X, Y and Z do not have to be distinct.

     When using MKL, for large matrices (m and n more than 50) it is may be more 
     efficient to use ProductMM().
*/
TMatrix ProductMU(TMatrix X, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((ColM(Y) != RowM(Z)) || (RowM(Z) != ColM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Y),ColM(Z)))) 
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR); 
	return (TMatrix)NULL; 
      }
  if (X == Z)
    if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
      {
	dw_Error(MEM_ERR); 
	return (TMatrix)NULL; 
      }
    else
      memcpy(ptr,pElementM(Z),RowM(Z)*ColM(Z)*sizeof(PRECISION));
  else
    ptr=pElementM(Z);
  if (X != Y)
    if (MajorForm(X) == MajorForm(Y))
      memcpy(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
    else
      bTranspose(pElementM(X),pElementM(Y),RowM(Y),ColM(Y),MajorForm(Y));
  bProductMT(pElementM(X),ptr,RowM(X),ColM(X),1,MajorForm(X),MajorForm(Z));
  if (ptr != pElementM(Z)) dw_free(ptr);
  return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x n matrix
     Z : n x n upper triangular matrix

   Results
     X = Y * Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Size permitting, X, Y and Z do not have to be distinct.

     When using MKL, for large matrices (m and n more than 50) it is may be more 
     efficient to use ProductMM().
*/
TMatrix ProductML(TMatrix X, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((ColM(Y) != RowM(Z)) || (RowM(Z) != ColM(Z)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Y),ColM(Z)))) 
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR); 
	return (TMatrix)NULL; 
      }
  if (X == Z)
    if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
      {
	dw_Error(MEM_ERR); 
	return (TMatrix)NULL; 
      }
    else
      memcpy(ptr,pElementM(Z),RowM(Z)*ColM(Z)*sizeof(PRECISION));
  else
    ptr=pElementM(Z);
  if (X != Y)
    if (MajorForm(X) == MajorForm(Y))
      memcpy(pElementM(X),pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
    else
      bTranspose(pElementM(X),pElementM(Y),RowM(Y),ColM(Y),MajorForm(Y));
  bProductMT(pElementM(X),ptr,RowM(X),ColM(X),0,MajorForm(X),MajorForm(Z));
  if (ptr != pElementM(Z)) dw_free(ptr);
  return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x m lower triangular matrix
     Z : m x n matrix

   Results
     X = Y * Z.  If X is a null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Size permitting, X, Y and Z do not have to be distinct.

     When using MKL, for large matrices (m and n more than 50) it is may be more 
     efficient to use ProductMM().
*/
TMatrix ProductUM(TMatrix X, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((ColM(Y) != RowM(Z)) || (RowM(Y) != ColM(Y)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Y),ColM(Z)))) 
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR); 
	return (TMatrix)NULL; 
      }
  if (X == Y)
    if (!(ptr=(PRECISION*)dw_malloc(RowM(Y)*ColM(Y)*sizeof(PRECISION))))
      {
	dw_Error(MEM_ERR); 
	return (TMatrix)NULL; 
      }
    else
      memcpy(ptr,pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
  else
    ptr=pElementM(Y);
  if (X != Z)
    if (MajorForm(X) == MajorForm(Z))
      memcpy(pElementM(X),pElementM(Z),RowM(Z)*ColM(Z)*sizeof(PRECISION));
    else
      bTranspose(pElementM(X),pElementM(Z),RowM(Z),ColM(Z),MajorForm(Z));
  bProductTM(pElementM(X),ptr,RowM(X),ColM(X),1,MajorForm(X),MajorForm(Y));
  if (ptr != pElementM(Y)) dw_free(ptr);
  return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x m lower triangular matrix
     Z : m x n matrix

   Results
     X = Y * Z.  If X is a null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Size permitting, X, Y and Z do not have to be distinct.

     When using MKL, for large matrices (m and n more than 50) it is may be more 
     efficient to use ProductMM().
*/
TMatrix ProductLM(TMatrix X, TMatrix Y, TMatrix Z)
{
  PRECISION *ptr;
  if (!Y || !Z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((ColM(Y) != RowM(Z)) || (RowM(Y) != ColM(Y)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(Y),ColM(Z)))) 
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(Y)) || (ColM(X) != ColM(Z)))
      {
	dw_Error(SIZE_ERR); 
	return (TMatrix)NULL; 
      }
  if (X == Y)
    if (!(ptr=(PRECISION*)dw_malloc(RowM(Y)*ColM(Y)*sizeof(PRECISION))))
      {
	dw_Error(MEM_ERR); 
	return (TMatrix)NULL; 
      }
    else
      memcpy(ptr,pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
  else
    ptr=pElementM(Y);
  if (X != Z)
    if (MajorForm(X) == MajorForm(Z))
      memcpy(pElementM(X),pElementM(Z),RowM(Z)*ColM(Z)*sizeof(PRECISION));
    else
      bTranspose(pElementM(X),pElementM(Z),RowM(Z),ColM(Z),MajorForm(Z));
  bProductTM(pElementM(X),ptr,RowM(X),ColM(X),0,MajorForm(X),MajorForm(Y));
  if (ptr != pElementM(Y)) dw_free(ptr);
  return X;
}
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/********************* Multiplication/Transpose Routines ***********************/
/*******************************************************************************/
/*
   Assumes
     X : m x n matrix or null pointer
     Y : r x m matrix
     Z : r x n matrix

   Results
     X = Y'*Z.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Y and Z do not have to be distinct matrices.
     If Y is square, X and Z do not have to be distinct matrices.
     If Z is square, X and Y do not have to be distinct matrices.
*/
TMatrix TransposeProductMM(TMatrix X, TMatrix Y, TMatrix Z)
{
 PRECISION *ptr;
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (RowM(Y) != RowM(Z))
  { 
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   {
    if (!(X=CreateMatrix(ColM(Y),ColM(Z)))) 
     return (TMatrix)NULL;
   }
  else
   {
    if ((RowM(X) != ColM(Y)) || (ColM(X) != ColM(Z)))
     {
      dw_Error(SIZE_ERR); 
      return (TMatrix)NULL; 
     }
    if ((X == Y) || (X == Z))
      {
       if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
        {
         dw_Error(MEM_ERR); 
         return (TMatrix)NULL; 
        }
       bProductMM(ptr,pElementM(Y),pElementM(Z),RowM(X),ColM(X),RowM(Y),MajorForm(X),1^MajorForm(Y),MajorForm(Z));
       memcpy(pElementM(X),ptr,RowM(X)*ColM(X)*sizeof(PRECISION));
       dw_free(ptr);
       return X;
      }
   }
 bProductMM(pElementM(X),pElementM(Y),pElementM(Z),RowM(X),ColM(X),RowM(Y),MajorForm(X),1^MajorForm(Y),MajorForm(Z));
 return X;
}

/*
   Assumes
     X : m x n matrix or null pointer
     Y : m x p matrix
     Z : n x p matrix

   Results
     X = Y*Z'.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.

   Notes
     Y and Z do not have to be distinct matrices.
     If Y is square, X and Z do not have to be distinct matrices.
     If Z is square, X and Y do not have to be distinct matrices.
*/
TMatrix ProductTransposeMM(TMatrix X, TMatrix Y, TMatrix Z)
{
 PRECISION *ptr;
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (ColM(Y) != ColM(Z))
  {
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   {
    if (!(X=CreateMatrix(RowM(Y),RowM(Z))))
     return (TMatrix)NULL;
   }
  else
   {
    if ((RowM(X) != RowM(Y)) || (ColM(X) != RowM(Z)))
     {
      dw_Error(SIZE_ERR); 
      return (TMatrix)NULL; 
     }
    if ((X == Y) || (X == Z))
      {
       if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
        {
         dw_Error(MEM_ERR); 
         return (TMatrix)NULL; 
        }
       bProductMM(ptr,pElementM(Y),pElementM(Z),RowM(X),ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),1^MajorForm(Z));
       memcpy(pElementM(X),ptr,RowM(X)*ColM(X)*sizeof(PRECISION));
       dw_free(ptr);
       return X;
      }
   }
 bProductMM(pElementM(X),pElementM(Y),pElementM(Z),RowM(X),ColM(X),ColM(Y),MajorForm(X),MajorForm(Y),1^MajorForm(Z));
 return X;
}
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/********************** Multiplication/Inverse Routines ************************/
/*******************************************************************************/
/*
   Assumes
     x : m vector or null
     Y : m x m matrix
     z : m vector

   Results
     x = Inverse(Y) * z.

   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and z do not have to be distinct. Uses Gaussian elimination 
     with partial pivoting and back substitution.  A null return indicates 
     that either Y or z was null, Y was singular, the matrices were not of the 
     required size, or the routine was unable to allocate needed memory.  Call 
     GetError() to determine which of these occured.
*/
TVector InverseProductMV(TVector x, TMatrix Y, TVector z)
{
  int err;
  TVector rtrn=(TVector)NULL;
  if (!Y || !z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Y) != ColM(Y)) || (ColM(Y) != DimV(z)))
      dw_Error(SIZE_ERR);
    else
      {
	if (rtrn=((x == z) ? x : EquateVector(x,z)))
	  if (err=bSolveLU(pElementV(rtrn),pElementM(Y),DimV(rtrn),1,1,MajorForm(Y)))
	    {
	      if (x != rtrn) FreeVector(rtrn);
	      rtrn=(TVector)NULL;
	      dw_Error(err);
	    }
      }
  return rtrn;
}

/*
   Assumes
     x : m vector or null
     Y : m x m matrix upper triangular matrix
     z : m vector

   Results
     x = Inverse(Y) * z.

   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and z do not have to be distinct.  A null return indicates 
     that either Y or z was null, Y was singular, or the matrices were not of the
     required size.  Call GetError() to determine which of these occured.
*/
TVector InverseProductUV(TVector x, TMatrix Y, TVector z)
{
  TVector rtrn=(TVector)NULL;
  int err;
  if (!Y || !z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Y) != ColM(Y)) || (ColM(Y) != DimV(z)))
      dw_Error(SIZE_ERR);
    else
      if (rtrn=((x == z) ? x : EquateVector(x,z)))
	if (err=bSolveTriangular(pElementM(Y),pElementV(rtrn),DimV(rtrn),1,1,MajorForm(Y),1))
	  {
	    if (rtrn != x) FreeVector(rtrn);
	    rtrn=(TVector)NULL;
	    dw_Error(err);
	  }
  return rtrn;
}

/*
   Assumes
     x : m vector or null
     Y : m x m matrix lower triangular matrix
     z : m vector

   Results
     x = Inverse(Y) * z.

   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and z do not have to be distinct. Uses back substitution.  
     A null return indicates that either Y or z was null, Y was singular, or
     the matrices were not of the required size.  Call GetError() to 
     determine which of these occured.
*/
TVector InverseProductLV(TVector x, TMatrix Y, TVector z)
{
 TVector rtrn=(TVector)NULL;
 int err;
 if (!Y || !z) 
   dw_Error(NULL_ERR); 
  else
   if ((RowM(Y) != ColM(Y)) || (ColM(Y) != DimV(z)))
     dw_Error(SIZE_ERR); 
    else
     if (rtrn=((x == z) ? x : EquateVector(x,z)))
      if (err=bSolveTriangular(pElementM(Y),pElementV(rtrn),DimV(rtrn),1,0,MajorForm(Y),1))
       {
        if (rtrn != x) FreeVector(rtrn);
        rtrn=(TVector)NULL;
        dw_Error(err);
       }   
 return rtrn;
}

/*
   Assumes
     X : m x n matrix or null
     Y : m x m matrix
     Z : m x n matrix

   Results
     X = Inverse(Y) * Z.

   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses Gaussian 
     elimination with partial pivoting and back substitution.  A null return 
     indicates that either Y or Z was null, Y was singular, the matrices were
     not of the required size, or the routine was unable to allocate needed 
     memory.  Call GetError() to determine which of these occured.
*/
TMatrix InverseProductMM(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, yt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pY;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Y) != ColM(Y)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	yt=MajorForm(Y);
	if (X != Y)
	  pY=pElementM(Y);
	else
	  if (!(pY=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pY,pElementM(Y),RowM(Y)*RowM(Y)*sizeof(PRECISION));
	if (rtrn=((X == Z) ? X : EquateMatrix(X,Z)))
	  if (err=bSolveLU(pElementM(rtrn),pY,RowM(rtrn),ColM(rtrn),MajorForm(rtrn),yt))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pY != pElementM(Y)) dw_free(pY);
      }
  return rtrn;
}

/*
   Assumes
     X : m x n matrix or null
     Y : m x m upper triangular matrix
     Z : m x n matrix

   Results
     X = Inverse(Y) * Z.

   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses back 
     substitution.  A null return indicates that either Y or Z was null, 
     Y was singular, or the matrices were not of the required size.  Call 
     GetError() to determine which of these occured.
*/
TMatrix InverseProductUM(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, yt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pY;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Y) != ColM(Y)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	yt=MajorForm(Y);
	if (X != Y)
	  pY=pElementM(Y);
	else
	  if (!(pY=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pY,pElementM(Y),RowM(Y)*RowM(Y)*sizeof(PRECISION));
	if (rtrn=((X == Z) ? X : EquateMatrix(X,Z)))
	  if (err=bSolveTriangular(pY,pElementM(rtrn),RowM(rtrn),ColM(rtrn),1,yt,MajorForm(rtrn)))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pY != pElementM(Y)) dw_free(pY);
      }
  return rtrn;
}

/*
   Assumes
     X : m x n matrix or null
     Y : m x m lower triangular matrix
     Z : m x n matrix

   Results
     X = Inverse(Y) * Z.

   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses back 
     substitution.  A null return indicates that either Y or Z was null, 
     Y was singular, or the matrices were not of the required size.  Call 
     GetError() to determine which of these occured.
*/
TMatrix InverseProductLM(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, yt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pY;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Y) != ColM(Y)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	yt=MajorForm(Y);
	if (X != Y)
	  pY=pElementM(Y);
	else
	  if (!(pY=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pY,pElementM(Y),RowM(Y)*RowM(Y)*sizeof(PRECISION));
	if (rtrn=((X == Z) ? X : EquateMatrix(X,Z)))
	  if (err=bSolveTriangular(pY,pElementM(rtrn),RowM(rtrn),ColM(rtrn),0,yt,MajorForm(rtrn)))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pY != pElementM(Y)) dw_free(pY);
      }
  return rtrn;
}

/*
   Assumes
     x : n vector or null pointer
     y : n vector
     Z : n x n invertible matrix

   Results
     x = y * Inverse(Z).  
    
   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and y do not have to be distinct. Uses Gaussian elimination 
     with partial pivoting and back substitution.  A null return indicates 
     that either y or Z was null, Z was singular, the matrices were not of the 
     required size, or the routine was unable to allocate needed memory.  Call 
     GetError() to determine which of these occured.
*/
TVector ProductInverseVM(TVector x, TVector y, TMatrix Z)
{
  int err;
  TVector rtrn=(TVector)NULL;
  if (!y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Z) != ColM(Z)) || (DimV(y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	if (rtrn=((x == y) ? x : EquateVector(x,y)))
	  if (err=bSolveLU(pElementV(rtrn),pElementM(Z),DimV(rtrn),1,1,1^MajorForm(Z)))
	    {
	      if (x != rtrn) FreeVector(rtrn);
	      rtrn=(TVector)NULL;
	      dw_Error(err);
	    }
      }
  return rtrn;
}

/*
   Assumes
     x : n vector or null pointer
     y : n vector
     Z : n x n upper triangular matrix

   Results
     x = y * Inverse(Z).  
    
   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and y do not have to be distinct. Uses back substitution.  
     A null return indicates that either y or Z was null, Z was singular, or
     the matrices were not of the  required size.  Call GetError() to 
     determine which of these occured.
*/
TVector ProductInverseVU(TVector x, TVector y, TMatrix Z)
{
 TVector rtrn=(TVector)NULL;
 int err;
 if (!y || !Z) 
   dw_Error(NULL_ERR); 
  else
   if ((RowM(Z) != ColM(Z)) || (DimV(y) != RowM(Z)))
     dw_Error(SIZE_ERR); 
    else
     if (rtrn=((x == y) ? x : EquateVector(x,y)))
      if (err=bSolveTriangular(pElementM(Z),pElementV(rtrn),DimV(rtrn),1,0,1^MajorForm(Z),1))
       {
        if (rtrn != x) FreeVector(rtrn);
        rtrn=(TVector)NULL;
        dw_Error(err);                         
       }
 return rtrn;
}

/*
   Assumes
     x : n vector or null pointer
     y : n vector
     Z : n x n lower triangular matrix

   Results
     x = y * Inverse(Z).  
    
   Returns
     Returns x upon success, null otherwise.

   Notes
     The vectors x and y do not have to be distinct. Uses back substitution.  
     A null return indicates that either y or Z was null, Z was singular, or
     the matrices were not of the  required size.  Call GetError() to 
     determine which of these occured.
*/
TVector ProductInverseVL(TVector x, TVector y, TMatrix Z)
{
 TVector rtrn=(TVector)NULL;
 int err;
 if (!y || !Z) 
   dw_Error(NULL_ERR); 
  else
   if ((RowM(Z) != ColM(Z)) || (DimV(y) != RowM(Z)))
     dw_Error(SIZE_ERR); 
    else
     if (rtrn=((x == y) ? x : EquateVector(x,y)))
      if (err=bSolveTriangular(pElementM(Z),pElementV(rtrn),DimV(rtrn),1,1,1^MajorForm(Z),1))
       {
        if (rtrn != x) FreeVector(rtrn);
        rtrn=(TVector)NULL;
        dw_Error(err);                         
       }
 return rtrn;
}

/*
   Assumes
     X : m x n matrix
     Y : m x n matrix
     Z : n x n matrix

   Results
     X = Y * Inverse(Z).  
    
   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses Gaussian 
     elimination with partial pivoting and back substitution.  A null return 
     indicates that either Y or Z was null, Z was singular, the matrices were
     not of the required size, or the routine was unable to allocate needed 
     memory.  Call GetError() to determine which of these occured.
*/
TMatrix ProductInverseMM(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, zt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pZ;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Z) != ColM(Z)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	zt=1^MajorForm(Z);
	if (X != Z)
	  pZ=pElementM(Z);
	else
	  if (!(pZ=(PRECISION*)dw_malloc(RowM(Z)*RowM(Z)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pZ,pElementM(Z),RowM(Z)*RowM(Z)*sizeof(PRECISION));
	if (rtrn=((X == Y) ? X : EquateMatrix(X,Y)))
	  if (err=bSolveLU(pElementM(rtrn),pZ,ColM(rtrn),RowM(rtrn),1^MajorForm(rtrn),zt))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pZ != pElementM(Z)) dw_free(pZ);
      }
  return rtrn;
}

/*
   Assumes
     X : m x n matrix
     Y : m x n matrix
     Z : n x n upper triangular matrix

   Results
     X = Y * Inverse(Z).  
    
   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses back 
     substitution.  A null return indicates that either Y or Z was null, 
     Z was singular, or the matrices were not of the required size.  Call 
     Getdw_Error() to determine which of these occured.
*/
TMatrix ProductInverseMU(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, zt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pZ;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Z) != ColM(Z)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	zt=1^MajorForm(Z);
	if (X != Z)
	  pZ=pElementM(Z);
	else
	  if (!(pZ=(PRECISION*)dw_malloc(RowM(Z)*RowM(Z)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pZ,pElementM(Z),RowM(Z)*RowM(Z)*sizeof(PRECISION));
	if (rtrn=((X == Y) ? X : EquateMatrix(X,Y)))
	  if (err=bSolveTriangular(pZ,pElementM(rtrn),ColM(rtrn),RowM(rtrn),0,zt,1^MajorForm(rtrn)))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pZ != pElementM(Z)) dw_free(pZ);
      }
  return rtrn;
}

/*
   Assumes
     X : m x n matrix
     Y : m x n matrix
     Z : n x n upper triangular matrix

   Results
     X = Y * Inverse(Z).  
    
   Returns
     Returns X upon success, null otherwise.

   Notes
     Size permitting, X, Y and Z do not have to be distinct. Uses back 
     substitution.  A null return indicates that either Y or Z was null, 
     Z was singular, or the matrices were not of the required size.  Call 
     GetError() to determine which of these occured.
*/
TMatrix ProductInverseML(TMatrix X, TMatrix Y, TMatrix Z)
{
  int err, zt;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION *pZ;
  if (!Y || !Z)
    dw_Error(NULL_ERR);
  else
    if ((RowM(Z) != ColM(Z)) || (ColM(Y) != RowM(Z)))
      dw_Error(SIZE_ERR);
    else
      {
	zt=1^MajorForm(Z);
	if (X != Z)
	  pZ=pElementM(Z);
	else
	  if (!(pZ=(PRECISION*)dw_malloc(RowM(Z)*RowM(Z)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return (TMatrix)NULL;
	    }
	  else
	    memcpy(pZ,pElementM(Z),RowM(Z)*RowM(Z)*sizeof(PRECISION));
	if (rtrn=((X == Y) ? X : EquateMatrix(X,Y)))
	  if (err=bSolveTriangular(pZ,pElementM(rtrn),ColM(rtrn),RowM(rtrn),1,zt,1^MajorForm(rtrn)))
	    {
	      if (X != rtrn) FreeMatrix(rtrn);
	      rtrn=(TMatrix)NULL;
	      dw_Error(err);
	    }
	if (pZ != pElementM(Z)) dw_free(pZ);
      }
  return rtrn;
}
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/************************** Matrix Inverse Routines ****************************/
/*******************************************************************************/
/*
   Assumes
     X : m x m matrix or null pointer
     Y : m x m matrix

   Results
     X = inverse(Y)

   Returns
    Returns X upon success, null otherwise

   Notes
     The matrices X and Y do not have to be distinct.  Uses Gaussian
     elimination with partial pivoting and back substitution to form the
     inverse.  A null return indicates that either Y or Z was null, Y was
     singular, or unable to allocate required memory.  Call GetError()
     to determine which of these occured.
*/
TMatrix Inverse_LU(TMatrix X, TMatrix Y)
{
 PRECISION *LU;
 int *p;
 TMatrix rtrn=(TMatrix)NULL;
 if (!Y) 
   dw_Error(NULL_ERR); 
  else
   if (RowM(Y) != ColM(Y))
     dw_Error(SIZE_ERR); 
    else
     if (!(p=(int*)dw_malloc(RowM(Y)*sizeof(int))))
       dw_Error(MEM_ERR); 
      else
       {
        if (!(LU=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
          dw_Error(MEM_ERR);
         else
          {  
           memcpy(LU,pElementM(Y),RowM(Y)*RowM(Y)*sizeof(PRECISION));
           if (bLU(p,LU,RowM(Y),RowM(Y),MajorForm(Y)))
             dw_Error(SING_ERR);
            else
             if (rtrn=IdentityMatrix(X,RowM(Y)))
              {
               bPermutationMultiply(p,pElementM(rtrn),RowM(rtrn),ColM(rtrn),RowM(Y),1,MajorForm(rtrn));
               bSolveUnitTriangular(LU,pElementM(rtrn),RowM(rtrn),ColM(rtrn),0,MajorForm(Y),MajorForm(rtrn));
               bSolveTriangular(LU,pElementM(rtrn),RowM(rtrn),ColM(rtrn),1,MajorForm(Y),MajorForm(rtrn));                        
              }
           dw_free(LU);
          }
        dw_free(p);
       }
 return rtrn;
}

/*
   Assumes
     X : m x m matrix or null pointer
     Y : m x m matrix

   Results
     X = inverse(Y)

   Returns
    Returns X upon success, null otherwise

   Notes
    The matrices X and Y do not have to be distinct.  Uses singular value
    decomposition to form the inverse.  A null return indicates that either
    Y or Z was null, Y was singular, or unable to allocate required memory.
    Call GetError() to determine which of these occured.
*/
TMatrix Inverse_SVD(TMatrix X, TMatrix Y)
{
  int i, j, k, err;
  TMatrix rtrn=(TMatrix)NULL;
  PRECISION scale, tolerance, *U, *V, *d;
  if (!Y)
    dw_Error(NULL_ERR);
  else
    if (RowM(Y) != ColM(Y))
      dw_Error(SIZE_ERR);
    else
      if (!(U=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
	dw_Error(MEM_ERR);
      else
	{
	  if (!(V=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
	    dw_Error(MEM_ERR);
	  else
	    {
	      if (!(d=(PRECISION*)dw_malloc(RowM(Y)*sizeof(PRECISION))))
		dw_Error(MEM_ERR);
	      else
		{
		  if (err=bSVD(U,d,V,pElementM(Y),RowM(Y),ColM(Y),COLUMN_MAJOR,ROW_MAJOR,MajorForm(Y),1))
		    dw_Error(err);
		  else
		    {
		      tolerance=d[0]*MACHINE_EPSILON*RowM(Y);
		      for (k=RowM(Y)*RowM(Y), j=RowM(Y)-1; j >= 0; j--)
			{
			  if (d[j] < tolerance)
			    {
			      dw_Error(SING_ERR);
			      dw_free(U);
			      dw_free(V);
			      dw_free(d);
			      return (TMatrix)NULL;
			    }
			  scale=1.0/d[j];
			  for (i=--k; i >= 0; i-=RowM(Y)) V[i]*=scale;
			    //for (i=RowM(Y)-1; i >= 0; i--) V[--k]*=scale;
			}
		      if (X)
			if ((RowM(X) != RowM(Y)) || (ColM(X) != RowM(Y)))
			  dw_Error(SIZE_ERR);
			else
			  bProductMM(pElementM(rtrn=X),V,U,RowM(Y),RowM(Y),RowM(Y),MajorForm(X),ROW_MAJOR,ROW_MAJOR);
		      else
			if (rtrn=CreateMatrix(RowM(Y),RowM(Y)))
			  bProductMM(pElementM(rtrn),V,U,RowM(Y),RowM(Y),RowM(Y),MajorForm(rtrn),ROW_MAJOR,ROW_MAJOR);
		    }
		  dw_free(d);
		}
	      dw_free(V);
	    }
	  dw_free(U);
	}
  return rtrn;
}

/*
   Assumes
     X : m x m matrix or null pointer
     Y : m x m symmetric matrix

   Results
     X = inverse(Y)

   Returns
    Returns X upon success, null otherwise

   Notes
    The matrices X and Y do not have to be distinct.  Uses Cholesky
    decomposition to form the inverse.  Only the upper half of Y is used.
    A null return indicates that either Y or Z was null, Y was singular,
    or unable to allocate required memory.  Call GetError() to
    determine which of these occured.
*/
TMatrix Inverse_Cholesky(TMatrix X, TMatrix Y)
{
 TMatrix rtrn=(TMatrix)NULL;
 PRECISION *ptr;
 if (!Y)
   dw_Error(NULL_ERR);
  else
   if (RowM(Y) != ColM(Y))
     dw_Error(SIZE_ERR);
    else
     if (!(ptr=(PRECISION*)dw_malloc(RowM(Y)*RowM(Y)*sizeof(PRECISION))))
       dw_Error(MEM_ERR);
      else
       {
        memcpy(ptr,pElementM(Y),RowM(Y)*RowM(Y)*sizeof(PRECISION));
        if (bCholesky(ptr,RowM(Y),1,MajorForm(Y)))
          dw_Error(SING_ERR);
         else
          {
           if (rtrn=IdentityMatrix(X,RowM(Y)))
            {
             bSolveTriangular(ptr,pElementM(rtrn),RowM(Y),RowM(Y),1,MajorForm(Y),MajorForm(rtrn));
             bProductMM(ptr,pElementM(rtrn),pElementM(rtrn),RowM(Y),RowM(Y),RowM(Y),MajorForm(rtrn),MajorForm(rtrn),1^MajorForm(rtrn));
             memcpy(pElementM(rtrn),ptr,RowM(Y)*RowM(Y)*sizeof(PRECISION));
            }
          }
        dw_free(ptr);
       }
 return rtrn;
}

/*
   Assumes
     X : m x m matrix or null pointer
     T : m x m upper triangular matrix

   Results
     X = inverse(T)

   Returns
     Returns X upon success, null otherwise

   Notes
     The matrices X and T do not have to be distinct.  Uses back substitution
     to form the inverse.  A null return indicates that T was null or
     singular, or unable to allocate required memory.  Call
     GetError() to determine which of these occured.
*/
TMatrix Inverse_UT(TMatrix X, TMatrix T)
{
 PRECISION *ptr;
 int err;
 if (!T)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (RowM(T) != ColM(T))
  {
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   {
    if (X=IdentityMatrix(X,RowM(T)))
     if (err=bSolveTriangular(pElementM(T),pElementM(X),RowM(T),RowM(T),1,MajorForm(T),MajorForm(X)))
      {
       dw_Error(err); 
       FreeMatrix(X); 
       return (TMatrix)NULL; 
      }
   }
  else
   if (X == T)
     {
      if (!(ptr=(PRECISION*)dw_malloc(RowM(T)*RowM(T)*sizeof(PRECISION))))
       {
        dw_Error(MEM_ERR); 
        return (TMatrix)NULL; 
       }
      memcpy(ptr,pElementM(T),RowM(T)*RowM(T)*sizeof(PRECISION));
      if (X=IdentityMatrix(X,RowM(T))) 
       if (err=bSolveTriangular(ptr,pElementM(X),RowM(T),RowM(T),1,MajorForm(T),MajorForm(X)))
        {
         dw_Error(err); 
         dw_free(ptr); 
         return (TMatrix)NULL; 
        }     
      dw_free(ptr);
     }
    else
     if (X=IdentityMatrix(X,RowM(T))) 
      if (err=bSolveTriangular(pElementM(T),pElementM(X),RowM(T),RowM(T),1,MajorForm(T),MajorForm(X)))
       {
        dw_Error(err); 
        return (TMatrix)NULL; 
       }
 return X;
}

/*
   Assumes
     X : m x m matrix or null pointer
     T : m x m lower triangular matrix

   Results
     X = inverse(T)

   Returns
     Returns X upon success, null otherwise

   Notes
     The matrices X and T do not have to be distinct.  Uses back substitution
     to form the inverse.  A null return indicates that either T was null,
     Y was singular, or unable to allocate required memory.  Call
     GetError() to determine which of these occured.
*/
TMatrix Inverse_LT(TMatrix X, TMatrix T)
{
 PRECISION *ptr;
 int err;
 if (!T)
  {
   dw_Error(NULL_ERR);
   return (TMatrix)NULL; 
  }
 if (RowM(T) != ColM(T))
  {
   dw_Error(SIZE_ERR); 
   return (TMatrix)NULL; 
  }
 if (!X)
   {
    if (X=IdentityMatrix(X,RowM(T)))
     if (err=bSolveTriangular(pElementM(T),pElementM(X),RowM(T),RowM(T),0,MajorForm(T),MajorForm(X)))
      {
       dw_Error(err); 
       FreeMatrix(X); 
       return (TMatrix)NULL; 
      }
   }
  else
   if (X == T)
     {
      if (!(ptr=(PRECISION*)dw_malloc(RowM(T)*RowM(T)*sizeof(PRECISION))))
       {
        dw_Error(MEM_ERR); 
        return (TMatrix)NULL; 
       }
      memcpy(ptr,pElementM(T),RowM(T)*RowM(T)*sizeof(PRECISION));
      if (X=IdentityMatrix(X,RowM(T)))
       if (err=bSolveTriangular(ptr,pElementM(X),RowM(T),RowM(T),0,MajorForm(T),MajorForm(X)))
        {
         dw_Error(err); 
         dw_free(ptr); 
         return (TMatrix)NULL; 
        }     
      dw_free(ptr);
     }
    else
     if (X=IdentityMatrix(X,RowM(T)))
      if (err=bSolveTriangular(pElementM(T),pElementM(X),RowM(T),RowM(T),0,MajorForm(T),MajorForm(X)))
       {
        dw_Error(err); 
        return (TMatrix)NULL; 
       }
 return X;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/****************************** Complex Routines *******************************/
/*******************************************************************************/
/*
   Assumes
     ReX : m x n matrix
     ImX : m x n matrix
     ReY : m x k matrix
     ImY : m x k matrix
     ReZ : k x n matrix
     ImZ : k x n matrix

   Results
     Computes X = Y*Z, where X = ReX + i*ImX, Y = ReY + i*ImY, and 
     Z = ReZ + i*ImZ are complex matrices. 

   Returns
     Returns 1 upon success, 0 otherwise

   Notes
     ReX and ImX must be distinct, but none of the other matrices must be so.
*/
int ComplexProductMM(TMatrix ReX, TMatrix ImX, TMatrix ReY, TMatrix ImY, TMatrix ReZ, TMatrix ImZ)
{
  PRECISION *ptr1=(PRECISION*)NULL, *ptr2=(PRECISION*)NULL;
  if (!ReX || !ImX || !ReY || !ImY || !ReZ || !ImZ)
    {
      dw_Error(NULL_ERR);
      return 0;
    }
  if (ReX == ImX)
    {
      dw_Error(ARG_ERR);
      return 0;
    }
  if ((RowM(ReX) != RowM(ImX)) || (RowM(ReX) != RowM(ReY)) || (RowM(ReX) != RowM(ImY)) ||
      (ColM(ReX) != ColM(ImX)) || (ColM(ReX) != ColM(ReZ)) || (ColM(ReX) != ColM(ImZ)) ||
      (ColM(ReY) != ColM(ImY)) || (ColM(ReY) != RowM(ReZ)) || (ColM(ReY) != RowM(ImZ)))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }
 
  if ((ReX != ReY) && (ReX != ImY) && (ReX != ReZ) && (ReX != ImZ))
    { 
      bProductMM(pElementM(ReX),pElementM(ReY),pElementM(ReZ),RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ReX),MajorForm(ReY),MajorForm(ReZ));
      bProductMM_Update(pElementM(ReX),pElementM(ImY),pElementM(ImZ),1.0,-1.0,RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ReX),MajorForm(ImY),MajorForm(ImZ));
    }
  else
    {
      ptr1=(PRECISION*)dw_malloc(RowM(ReX)*ColM(ReX)*sizeof(PRECISION));
      bProductMM(ptr1,pElementM(ReY),pElementM(ReZ),RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ReX),MajorForm(ReY),MajorForm(ReZ));
      bProductMM_Update(ptr1,pElementM(ImY),pElementM(ImZ),1.0,-1.0,RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ReX),MajorForm(ImY),MajorForm(ImZ));
    }

  if ((ImX != ReY) && (ImX != ImY) && (ImX != ReZ) && (ImX != ImZ))
    { 
      bProductMM(pElementM(ImX),pElementM(ReY),pElementM(ImZ),RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ImX),MajorForm(ReY),MajorForm(ImZ));
      bProductMM_Update(pElementM(ImX),pElementM(ImY),pElementM(ReZ),1.0,1.0,RowM(ReX),ColM(ReX),ColM(ReY),
			    MajorForm(ImX),MajorForm(ImY),MajorForm(ReZ));
    }
  else
    {
      ptr2=(PRECISION*)dw_malloc(RowM(ReX)*ColM(ReX)*sizeof(PRECISION));
      bProductMM(ptr2,pElementM(ReY),pElementM(ImZ),RowM(ReX),ColM(ReX),ColM(ReY),
		      MajorForm(ImX),MajorForm(ReY),MajorForm(ImZ));
      bProductMM_Update(ptr2,pElementM(ImY),pElementM(ReZ),1.0,1.0,RowM(ReX),ColM(ReX),ColM(ReY),
			    MajorForm(ImX),MajorForm(ImY),MajorForm(ReZ));
    }
  if (ptr1)
    {
      memcpy(pElementM(ReX),ptr1,RowM(ReX)*ColM(ReX)*sizeof(PRECISION));
      dw_free(ptr1);
    }
  if (ptr2)
    {
      memcpy(pElementM(ImX),ptr2,RowM(ReX)*ColM(ReX)*sizeof(PRECISION));
      dw_free(ptr2);
    }
 return 1;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/*************************** Miscellaneous Routines ****************************/
/*******************************************************************************/
/*
   Assumes
     x : m-vector

   Results
     Computes the norm of x.

   Returns 
     Returns the norm of x upon success and -1.0 upon failure.
*/
PRECISION Norm(TVector x)
{
 int i;
 PRECISION result;
 if (!x)
  {
   dw_Error(NULL_ERR);
   return -1.0;
   }
 dw_ClearError();
 for (result=ElementV(x,0)*ElementV(x,0), i=DimV(x)-1; i > 0; i--)
  result+=ElementV(x,i)*ElementV(x,i);
 return sqrt(result);
}

/*
   Assumes
     X : m x n matrix

   Results
     Returns the Euclidean norm of x.

   Notes
     The Euclidean norm is the square root of the sum of the squares of the 
     elements of X
*/
PRECISION MatrixNormEuclidean(TMatrix X)
{
 int i;
 PRECISION result, *p;
 if (!X)
  {
   dw_Error(NULL_ERR);
   return -1.0;
   }
 dw_ClearError();
 p=pElementM(X);
 for (result=p[0]*p[0], i=RowM(X)*ColM(X)-1; i > 0; i--)
   result+=p[i]*p[i];
 return sqrt(result);
}

/*
   Assumes
     X : m x n matrix

   Results
     Returns the Euclidean norm of x.

   Notes
     The matrix norm of X is the max of the norm of X*v over all vectors of unit
     length.  It will be equal to the largest of the singular values of X.
*/
PRECISION MatrixNorm(TMatrix X)
{
  int err;
  PRECISION result, *d;
  if (!X)
    {
      dw_Error(NULL_ERR);
      return -1.0;
    }
  if (!(d=(PRECISION*)dw_malloc(((RowM(X) < ColM(X)) ? RowM(X) : ColM(X))*sizeof(PRECISION))))
    {
      dw_Error(MEM_ERR);
      return -1.0;
    }
  dw_ClearError();
  if (err=bSVD((PRECISION*)NULL,d,(PRECISION*)NULL,pElementM(X),RowM(X),ColM(X),1,1,MajorForm(X),1))
    {
      dw_free(d);
      dw_Error(err);
      return -1.0;
    }
  result=d[0];
  dw_free(d);
  return result;
}

/*
   Assumes
     x : m-vector
     y : m-vector

   Results
     returns dot product of x and y.

   Notes
     On error returns 0.0.  Call GetError() to determine the type of error.
*/
PRECISION DotProduct(TVector x, TVector y)
{
 PRECISION result;
 int i;
 if (!x || !y)
  {
   dw_Error(NULL_ERR);
   return 0.0;
  }
 if (DimV(x) != DimV(y))
  {
   dw_Error(SIZE_ERR);
   return 0.0;
  }
 dw_ClearError();
 for (result=ElementV(x,0)*ElementV(y,0), i=DimV(x)-1; i > 0; i--) result+=ElementV(x,i)*ElementV(y,i);
 return result;
}

/*
   Assumes
     x : m-vector
     y : m-vector
     S : m x m matrix

   Results
     returns x'*S*y

   Notes
     In order for S to be a inner product, S must be positive definite
     and symmetric.  A zero return could indicate a error condition.  
     Call GetError() to determine if an error has occured.
*/
PRECISION InnerProduct(TVector x, TVector y, TMatrix S)
{
 if (!x || !y || !S)
  {
   dw_Error(NULL_ERR);
   return 0.0;
  }
 if ((DimV(x) != RowM(S)) || (DimV(y) != ColM(S)) || (DimV(x) != DimV(y)))
  {
   dw_Error(SIZE_ERR);
   return 0.0;
  }
 dw_ClearError();
 return bInnerProduct(pElementV(x),pElementV(y),pElementM(S),DimV(x));
}

/*
   Assumes
     X : m x n matrix
     y : m-vector
     z : n-vector

   Results
     Returns X = y * z' upon success and a null pointer on failure.
*/
TMatrix OuterProduct(TMatrix X, TVector y, TVector z)
{
  int i, j;
  if (!y || !z)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    { if (!(X=CreateMatrix(DimV(y),DimV(z)))) return (TMatrix)NULL; }
  else
    if ((RowM(X) != DimV(y)) || (ColM(X) != DimV(z)))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
  for (i=DimV(y)-1; i >= 0; i--)
    for (j=DimV(z)-1; j >= 0; j--)
      ElementM(X,i,j)=ElementV(y,i)*ElementV(z,j);
  return X;
}

/*
   Assumes
     X : m x m matrix

   Results
     Returns the determinate of X upon success.  Returns
     0 upon failure.  Call GetError() to determine
     failure.

   Notes
     Uses the LU decomposition to compute the determinate.
*/
PRECISION Determinant_LU(TMatrix X)
{
  PRECISION rtrn=0.0, *LU;
  int i, sgn, *p;
  if (!X)
    dw_Error(NULL_ERR);
  else
    if (RowM(X) != ColM(X))
      dw_Error(SIZE_ERR);
    else
      if (!(p=(int*)dw_malloc(RowM(X)*sizeof(int))))
	dw_Error(MEM_ERR); 
      else
	{
	  if (!(LU=(PRECISION*)dw_malloc(RowM(X)*RowM(X)*sizeof(PRECISION))))
	    dw_Error(MEM_ERR);
	  else
	    {  
	      memcpy(LU,pElementM(X),RowM(X)*RowM(X)*sizeof(PRECISION));
	      if (!bLU(p,LU,RowM(X),RowM(X),1))
		{ 
		  for (sgn=1, i=RowM(X)-2; i >= 0; i--) 
		    if (p[i] != i) sgn=-sgn;
		  for (rtrn=0.0, i=RowM(X)*RowM(X)-1; i >= 0; i-=RowM(X)+1)
		    if (LU[i] < 0.0)
		      {
			rtrn+=log(-LU[i]);
			sgn=-sgn;
		      }
		    else
		      if (LU[i] > 0.0)
			rtrn+=log(LU[i]);
		      else
			break;
		  rtrn=(i >= 0) ? 0.0 : sgn*exp(rtrn);
		}   
	      dw_ClearError();       
	      dw_free(LU);
	    } 
	  dw_free(p);
	}
  return rtrn;
}

/*
   Assumes
     X : m x m matrix

   Results
     Returns the determinate of X upon success.  Returns
     0 upon failure.  Call GetError() to determine
     failure.

   Notes
     Uses the LU decomposition to compute the determinate.
*/
PRECISION LogAbsDeterminant_LU(TMatrix X)
{
 PRECISION rtrn=0.0, *LU;
 int i, *p;
 if (!X)
   dw_Error(NULL_ERR);
  else
   if (RowM(X) != ColM(X))
     dw_Error(SIZE_ERR);
    else
     if (!(p=(int*)dw_malloc(RowM(X)*sizeof(int))))
       dw_Error(MEM_ERR); 
      else
       {
        if (!(LU=(PRECISION*)dw_malloc(RowM(X)*RowM(X)*sizeof(PRECISION))))
          dw_Error(MEM_ERR);
         else
          {  
           memcpy(LU,pElementM(X),RowM(X)*RowM(X)*sizeof(PRECISION));
           if (!bLU(p,LU,RowM(X),RowM(X),1))
            { 
             for (i=RowM(X)*RowM(X)-1; i >= 0; i-=RowM(X)+1)
              if (LU[i] < 0.0)
                 rtrn+=log(-LU[i]);
               else
                if (LU[i] > 0.0)
                  rtrn+=log(LU[i]);
		else
		  {
		    rtrn=MINUS_INFINITY;
		    break;
		  }
	     dw_ClearError();
            }       
           dw_free(LU);
          } 
        dw_free(p);
       }
 return rtrn;
}

/*
   Assumes
     X : m x m matrix

   Results
     Returns the trace of X.
*/
PRECISION Trace(TMatrix X)
{
 PRECISION trace=0.0;
 int i;
 if (!X)
   dw_Error(NULL_ERR);
  else
   if (RowM(X) != ColM(X)) 
     dw_Error(SIZE_ERR);
    else
     for (trace=ElementM(X,0,0), i=RowM(X)-1; i > 0; i--) trace+=ElementM(X,i,i);
 return trace;
}

/*
   Assumes
     X : m x n matrix

   Results
     Returns the rank of X.  Return -1 upon error.  Use dw_GetError()
     to determine error type.

   Notes
     Uses the singular value decomposition to compute the rank.  A singular value
     is considered to be zero if it is smaller than the square root of machine
     epsilon times the largest singular value.
*/
int Rank_SVD(TMatrix X)
{
  PRECISION *d, small;
  int rank, q, err;
  if (!X)
    {
      dw_Error(NULL_ERR);
      return -1;
    }
  q=(ColM(X) < RowM(X)) ? ColM(X) : RowM(X);
  if (!(d=(PRECISION*)dw_malloc(q*sizeof(PRECISION))))
    {
      dw_Error(MEM_ERR);
      return -1;
    }
  if (err=bSVD((PRECISION*)NULL,d,(PRECISION*)NULL,pElementM(X),RowM(X),ColM(X),COLUMN_MAJOR,ROW_MAJOR,MajorForm(X),1))
    {
      dw_free(d);
      dw_Error(err);
      return -1;
    }
  for (small=d[0]*SQRT_MACHINE_EPSILON, rank=q-1; rank >= 0; rank--)
    if (d[rank] > small) break;
  dw_free(d); 
  return rank+1;
}

/*
   Assumes
     x : n-vector or null pointer
     Y : n x (n-1) matrix

   Results
     The vector x is set to the cross product of the columns of Y.

   Notes
     The cross product of the columns of Y is the vector such that

      (1) Y'x = 0
      (2) det([Y x]) >= 0
      (3) norm(x) = volume of the parallelpiped spanned by the columns of Y
     
     Uses the LU decomposition to compute the cross product.
*/

TVector CrossProduct_LU(TVector x, TMatrix Y)
{
  int i, j, sgn;
  PRECISION s;
  TMatrix X;
  TPermutation P;
  TVector z=(TVector)NULL;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if (RowM(Y) != ColM(Y)+1)
    { 
      dw_Error(SIZE_ERR); 
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(z=x=CreateVector(RowM(Y))))
	return (TVector)NULL;
    }
  else
    if (RowM(Y) != DimV(x)) 
      { 
	dw_Error(SIZE_ERR); 
	return (TVector)NULL; 
      }
  if (X=CreateMatrix(RowM(Y),ColM(Y)))
    {
      if (P=CreatePermutation(RowM(Y)))
	{
	  if (LU(P,X,Y))
	    {
	      ElementV(x,RowM(Y)-1)=1.0;
	      for (i=RowM(Y)-2; i >= 0; i--)
		{
		  ElementV(x,i)=0.0;
		  for (j=RowM(Y)-1; j > i; j--)
		    ElementV(x,i)-=ElementV(x,j)*ElementM(X,j,i); 
		}
	      ProductTransposeVP(x,x,P);
	      for (sgn=1, i=RowM(X)-2; i >= 0; i--) 
		if (ElementP(P,i) != i) sgn=-sgn;
	      for (s=0.0, i=RowM(X)-2; i >= 0; i--)
		if (ElementM(X,i,i) < 0.0)
		  {
		    s+=log(-ElementM(X,i,i));
		    sgn=-sgn;
		  }
		else
		  if (ElementM(X,i,i) > 0.0)
		    s+=log(ElementM(X,i,i));
		  else
		    break;
	      s=(i >= 0) ? 0.0 : sgn*exp(s);
	      ProductSV(x,s,x);
	    }
	  else
	    {
	      FreePermutation(P); 
	      FreeMatrix(X);         
	      if (z) FreeVector(z);
	      return (TVector)NULL;
	    }
	  FreePermutation(P);      
	}
      FreeMatrix(X);
    }
  return x;
}

/*
   Assumes
     Y : m x n matrix

   Returns
     Upon success, the columns of the returned matrix form an orthonormal basis 
     for the null space of Y.  A null return either indicates the null space is
     {0} or an error condition.  Call dw_GetError() to determine if a failure 
     occured.

   Notes
     Use the singular value decomposition to compute the null space.  

     If the largest singular value of Y is less than or equal to the square root 
     of machine epsilon, then the matrix is assumed to be the zero matrix  and 
     the null space is all of n-dimensional Euclidean space.  For this reason, 
     care must be taken with the scale of Y.  

     A singular value is assumed to be zero if it is smaller than the square root 
     of the minimum of m and n times machine epsilon.
*/
TMatrix NullSpace(TMatrix X)
{
  PRECISION *d, *v, small;
  int rank, q, err;
  TMatrix rtrn=(TMatrix)NULL;
  if (!X)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  q=(ColM(X) < RowM(X)) ? ColM(X) : RowM(X);
  d=(PRECISION*)dw_malloc(q*sizeof(PRECISION));
  v=(PRECISION*)dw_malloc(ColM(X)*ColM(X)*sizeof(PRECISION));
  if (!d || !v)
    dw_Error(MEM_ERR);
  else
    if (err=bSVD((PRECISION*)NULL,d,v,pElementM(X),RowM(X),ColM(X),COLUMN_MAJOR,COLUMN_MAJOR,MajorForm(X),0))
      dw_Error(err);
    else
      {
	dw_ClearError();
	for (small=d[0]*SQRT_MACHINE_EPSILON, rank=q-1; rank >= 0; rank--)
	  if (d[rank] > small) break;
	if (++rank < ColM(X))
	  {
	    rtrn=CreateMatrix(ColM(X),ColM(X)-rank);
	    if (MajorForm(rtrn))
	      memcpy(pElementM(rtrn),v+(ColM(X)*rank),ColM(X)*(ColM(X)-rank)*sizeof(PRECISION));
	    else
	      bTranspose(pElementM(rtrn),v+(ColM(X)*rank),ColM(X),ColM(X)-rank,COLUMN_MAJOR);
	  }
      }
  if (v) dw_free(v);
  if (d) dw_free(d);
  return rtrn;
}

/*
   Assumes
     X : n x m matrix or null pointer
     Y : m x n matrix

   Returns
     The generalized inverse upon success and null upon failure.  If X is null,
     then it is created.  The generalized inverse of Y is the unique matrix X
     such that
      (a) X*Y*X = X
      (b) Y*X*Y = Y
      (c) X*Y and Y*X are symmetric

   Notes
     Use the singular value decomposition to compute the generalized inverse.  

     A singular value is assumed to be zero if it is smaller than the square root
     of machine epsilon times the largest singular value.
*/
TMatrix GeneralizedInverse(TMatrix X, TMatrix Y)
{
  PRECISION *d, *u, *v, small, scale;
  int err, q, i, j, k;
  TMatrix rtrn=(TMatrix)NULL;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if (X && ((RowM(X) != ColM(Y)) || (ColM(X) != RowM(Y))))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  q=(RowM(Y) < ColM(Y)) ? RowM(Y) : ColM(Y);
  d=(PRECISION*)dw_malloc(q*sizeof(PRECISION));
  u=(PRECISION*)dw_malloc(RowM(Y)*q*sizeof(PRECISION));
  v=(PRECISION*)dw_malloc(ColM(Y)*q*sizeof(PRECISION));
  if (!d || !u || !v)
    dw_Error(NULL_ERR);
  else
    if (err=bSVD(u,d,v,pElementM(Y),RowM(Y),ColM(Y),COLUMN_MAJOR,ROW_MAJOR,MajorForm(Y),1))
      dw_Error(err);
    else
      {
	small=d[0]*SQRT_MACHINE_EPSILON;
	for (k=ColM(Y)*q-1, i=q-1; i >= 0; k--, i--)
	  for (scale=(d[i] < small) ? 0.0 : 1.0/d[i], j=k; j >= 0; j-=q)
	    v[j]*=scale;
	rtrn=X ? X : CreateMatrix(ColM(Y),RowM(Y));
	bProductMM(pElementM(rtrn),v,u,RowM(rtrn),ColM(rtrn),q,MajorForm(rtrn),ROW_MAJOR,ROW_MAJOR);
      }
  if (v) dw_free(v);
  if (u) dw_free(u);
  if (d) dw_free(d);
  return rtrn;
}

/*
   Assumes:
     X : m x m matrix

   Return:
     X upon success and null upon failure.
   Results:
     X = 0.5*(X + X'), which forces X to symmetric.
*/
TMatrix ForceSymmetric(TMatrix X)
{
  int j, k, m;
  PRECISION *pX;
  if (!X || (RowM(X) != ColM(X))) return (TMatrix)NULL;
  for (m=RowM(X), pX=pElementM(X), j=m*m-2; j > 0; j+=k-1)
    for (k=j-m+1; k >= 0; j--, k-=m)
      pX[j]=pX[k]=0.5*(pX[j]+pX[k]);
  return X;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/***************************** Kronecker Routines ******************************/
/*******************************************************************************/
/*
   Assumes
     x : mn vector or null pointer
     Y : m x n matrix

   Results
     The vector x is filled by stacking the columns of Y.

   Returns
     The vector x upon success and null upon failure.  If x is null, it is 
     created.
*/
TVector Vec(TVector x, TMatrix Y)
{
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  if (!x)
    {
      if (!(x=CreateVector(RowM(Y)*ColM(Y))))
	return (TVector)NULL;
    }
  else
    if (RowM(Y)*ColM(Y) != DimV(x))
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }
  if (MajorForm(Y))
    memcpy(pElementV(x),pElementM(Y),RowM(Y)*ColM(Y)*sizeof(PRECISION));
  else
    bTranspose(pElementV(x),pElementM(Y),RowM(Y),ColM(Y),0);
  return x;
}

/*
   Assumes
     X : m x n matrix or null pointer
     y : mn vector

   Results
     The matrix is filled by reshaping the vector v in column order.  

   Returns
     The Matrix X upon success and null upon failure.  If X is null, it is 
     created.
*/
TMatrix ReshapeV(TMatrix X, TVector y, int rows, int cols)
{
  if (!y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if (DimV(y) != rows*cols)
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(rows,cols)))
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != rows) || (ColM(X) != cols))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
  if (MajorForm(X))
    memcpy(pElementM(X),pElementV(y),rows*cols*sizeof(PRECISION));
  else
    bTranspose(pElementM(X),pElementV(y),rows,cols,1);
  return X;
}

/*
   Assumes
     X : mr x ns vector or null pointer
     Y : m x n matrix
     Z : r x s matrix

   Results
     The matrix X is is filled with the Kronecker product of Y and Z.

   Returns
     The matrix X upon success and null upon failure.  If X is null, it is 
     created.
*/
TMatrix KroneckerProduct(TMatrix X, TMatrix Y, TMatrix Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR);
   return (TMatrix)NULL;
  }
 if (!X)
   {
    if (!(X=CreateMatrix(RowM(Y)*RowM(Z),ColM(Y)*ColM(Z))))
    return (TMatrix)NULL;
   }
  else
   if ((RowM(Y)*RowM(Z) != RowM(X)) || (ColM(Y)*ColM(Z) != ColM(X)))
    {
     dw_Error(SIZE_ERR);
     return (TMatrix)NULL;
    }
 if (X == Y)
   bMultiply(pElementM(X),pElementM(X),*pElementM(Z),RowM(X)*ColM(X));
 else
   if (X == Z)
     bMultiply(pElementM(X),pElementM(X),*pElementM(Y),RowM(X)*ColM(X));
   else
     bMatrixTensor(pElementM(X),pElementM(Y),pElementM(Z),RowM(Y),ColM(Y),RowM(Z),ColM(Z),MajorForm(X),MajorForm(Y),MajorForm(Z));
 return X;
}
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/****************************** Output Routines ********************************/
/*******************************************************************************/
/*
   Assumes
     x : m-vector

   Returns
     One upon success and zero otherwise.

   Results
     Prints x to f using format.

   Notes
     If format is the NULL pointer, uses default formating.  The elements of x 
     are output as double independently of the setting of PRECISION.  Currently,
     the function does not check for failure.
*/
int dw_PrintVector(FILE *f, TVector x, const char *format)
{
 int i, m;
 if (!x) return 0;
 m=DimV(x);
 if (!format) format="%7.3lf\t";
 for (i=0; i < m; i++) fprintf(f,format,(double)(ElementV(x,i)));
 fprintf(f,"\n");
 return 1;
}

/*
   Assumes
     X : m x n matrix

   Returns
     One upon success and zero otherwise.

   Results
     Prints X to f using format.

   Notes
     If format is the NULL pointer, uses default formating.  The elements of X
     are output as double independently of the setting of PRECISION.
*/
int dw_PrintMatrix(FILE *f, TMatrix X, const char *format)
{
 int i, j, m, n;
 if (!X) return 0;
 m=RowM(X); n=ColM(X);
 if (!format) format="%7.3lf\t";
 for (i=0; i < m; i++)
  {
   for (j=0; j < n; j++) fprintf(f,format,(double)(ElementM(X,i,j)));
   fprintf(f,"\n");
  }
 return 1;
}

/*
   Assumes
     x : m vector
     f : an pointer to an open file

   Results
     Reads in vector x from f.  Return 1 on success, 0 otherwise.
*/
int dw_ReadVector(FILE *f, TVector x)
{
 int i, m;
 PRECISION *px;
 if (!x) return 0;
 for (px=pElementV(x), i=0, m=DimV(x); i < m; i++)
#if (PRECISION_SIZE == 8)
 if (fscanf(f," %lf ",px+i) != 1) return 0; 
#else
 if (fscanf(f," %f ",px+i) != 1) return 0; 
#endif
 return 1;
}

/*
   Assumes
     X : m x n matrix
     f : an pointer to an open file

   Results
     Reads in matrix X from f.  Return 1 on success, 0 otherwise.
*/
int dw_ReadMatrix(FILE *f, TMatrix X)
{
 int i, j, m, n;
 if (!X) return 0;
 m=RowM(X); n=ColM(X);
 for (i=0; i < m; i++)
  for (j=0; j < n; j++)
#if (PRECISION_SIZE == 8)
   if (fscanf(f," %lf ",&ElementM(X,i,j)) != 1) return 0;
#else
  if (fscanf(f," %f ",&Element(X,i,j))) != 1) return 0; 
#endif
 return 1;
}

/*
   Assumes
     f : pointer to open binary file
     x : m matrix

   Results
     Outputs x in binary format.  

   Returns
     1 upon success, 0 upon failure

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   rows if matrix, dimension if vector
       int   columns if matrix, absent if vector
       data  m binary floating point numbers

*/
int OutVectorFloat(FILE *f, TVector x)
{
 float y;
 int i, format_code=0x1000+sizeof(float);
 if (!x) return 0;
 if (fwrite(&format_code,sizeof(int),1,f) != 1) return 0;
 if (fwrite(&DimV(x),sizeof(int),1,f) != 1) return 0;
#if (PRECISION_SIZE == 8)
 for (i=DimV(x)-1; i >= 0; i--) 
  {
   y=(float)ElementV(x,i);
   if (fwrite(&y,sizeof(float),1,f) != 1) return 0;
  }
#else
 if (fwrite(pElementV(x),DimV(x)*sizeof(float),1,f) != 1) return 0;
#endif
 return 1;
}

/*
   Assumes
     f : pointer to open binary file
     x : m matrix

   Results
     Outputs x in binary format.  Returns 1 on success and 0 otherwise.

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   rows if matrix, dimension if vector
       int   columns if matrix, absent if vector
       data  m binary floating point numbers

*/
int OutVectorDouble(FILE *f, TVector x)
{
 double y;
 int i, format_code=0x1000+sizeof(double);
 if (!x) return 0;
 if (fwrite(&format_code,sizeof(int),1,f) == 1) return 0;
 if (fwrite(&DimV(x),sizeof(int),1,f) != 1) return 0;
#if (PRECISION_SIZE == 8) 
  for (i=DimV(x)-1; i >= 0; i--) 
   {
    y=(double)ElementV(x,i);
    if (fwrite(&y,sizeof(double),1,f) != 1) return 0;
   }
#else
 if (fwrite(pElementV(x),DimV(x)*sizeof(double),1,f) != 1) return 0; 
#endif
 return 1;
}


/*
   Assumes
     f : pointer to open binary file
     X : m x n matrix

   Results
     Outputs X in binary format.  Returns 1 on success and 0 otherwise.

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   rows if matrix, dimension if vector
       int   columns if matrix, absent if vector
       data  m x n binary floating point numbers
*/
int OutMatrixFloat(FILE *f, TMatrix X)
{
 float y;
 int i, j, format_code=0x2000+sizeof(float);
 if (!X) return 0;
 if (fwrite(&format_code,sizeof(int),1,f) != 1) return 0;
 if (fwrite(&RowM(X),sizeof(int),2,f) != 1) return 0;
 if (fwrite(&ColM(X),sizeof(int),1,f) != 1) return 0;
#if (PRECISION_SIZE == 8)
  for (j=ColM(X)-1; j >= 0; j--)
   for (i=RowM(X)-1; i >= 0; i--)
    {
     y=(float)ElementM(X,i,j);
     if (fwrite(&y,sizeof(float),1,f) != 1) return 0;
    } 
#else
 if (fwrite(pElementM(X),RowM(X)*ColM(X)*sizeof(float),1,f) != 1) return 0;
#endif
 return 1;
}

/*
   Assumes
     f : pointer to open binary file
     X : m x n matrix

   Results
     Outputs X in binary format.  Returns 1 on success and 0 otherwise.

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   rows if matrix, dimension if vector
       int   columns if matrix, absent if vector
       data  m x n binary floating point numbers
*/
int OutMatrixDouble(FILE *f, TMatrix X)
{
 double y;
 int i, j, format_code=0x2000+sizeof(double);
 if (!X) return 0;
 if (fwrite(&format_code,sizeof(int),1,f) != 1) return 0;
 if (fwrite(&RowM(X),sizeof(int),2,f) != 1) return 0;
 if (fwrite(&ColM(X),sizeof(int),1,f) != 1) return 0;
#if (PRECISION_SIZE == 8)
 if (fwrite(((int *)X)-2,sizeof(int),2,f) != 2) return 0;
 for (j=ColM(X)-1; j >= 0; j--)
  for (i=RowM(X)-1; i >= 0; i--)
    {
     y=ElementM(X,i,j);
     if (fwrite(&y,sizeof(double),1,f) != 1) return 0;
    }
#else
 if (fwrite(pElementM(X),RowM(X)*ColM(X)*sizeof(double),1,f) != 1) return 0; 
#endif
 return 1;
}


/*
   Assumes
     f : pointer to open binary file
     x : m matrix or null pointer

   Results
     Reads x from binary format.  Returns 1 on success and 0 otherwise.

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   number rows if matrix, dimension if vector
       int   number columns if matrix, absent if vector
       data  m binary floating point numbers
*/
TVector InVector(FILE *f, TVector x)
{
 int i, d[2], precision, del=0, position=ftell(f);
 void *y=(void*)NULL;

 if (fread(d,sizeof(int),2,f) != 2) goto EXIT_ERROR;

 switch(d[0])
  {
   case 0x1000+sizeof(float): precision=sizeof(float); break;
   case 0x1000+sizeof(double): precision=sizeof(double); break;
   default: goto EXIT_ERROR;
  }

 i=d[1];

 if (!x)
   {
    x=CreateVector(i);
    del=1;
   }
  else
    if (DimV(x) != i) goto EXIT_ERROR;

 if (precision != sizeof(PRECISION))
   {
    if (!(y=dw_malloc(i*precision))) dw_Error(MEM_ERR);
    if (fread(y,i*precision,1,f) != 1) goto EXIT_ERROR;
    if (precision == sizeof(float))
      while (--i >= 0) ElementV(x,i)=((float*)y)[i];
     else
      while (--i >= 0) ElementV(x,i)=((double*)y)[i];
    dw_free(y);
   }
  else
   if (fread(pElementV(x),i*sizeof(PRECISION),1,f) != 1) goto EXIT_ERROR;

 return x;

EXIT_ERROR:
 fseek(f,position,SEEK_SET);
 if (del) FreeVector(x);
 if (y) dw_free(y);
 return (TVector)NULL;
}

/*
   Assumes
     f : pointer to open binary file
     X : m x n matrix or null pointer

   Results
     Reads x from binary format.  Returns 1 on success and 0 otherwise.

   Notes
     Format
       int
             0x1004 - vector float
             0x2004 - matrix float
             0x1008 - vector double
             0x2008 - mattrix double

       int   number rows if matrix, dimension if vector
       int   number columns if matrix, absent if vector
       data  m x n binary floating point numbers
*/
TMatrix InMatrix(FILE *f, TMatrix X)
{
 int i, d[3], precision, del=0, position=ftell(f);
 void *Y=(void*)NULL;

 if (fread(d,sizeof(int),3,f) != 3) goto EXIT_ERROR;

 switch(d[0])
  {
   case 0x2000+sizeof(float): precision=sizeof(float); break;
   case 0x2000+sizeof(double): precision=sizeof(double); break;
   default: goto EXIT_ERROR;
  }

 if (!X)
   {
    X=CreateMatrix(d[1],d[2]);
    del=1;
   }
  else
   if ((RowM(X) != d[1]) || (ColM(X) != d[2])) goto EXIT_ERROR;

 i=d[1]*d[2];

 if (precision != sizeof(PRECISION))
   {
    if (!(Y=dw_malloc(i*precision))) dw_Error(MEM_ERR);
    if (fread(Y,i*precision,1,f) != 1) goto EXIT_ERROR;
    if (precision == sizeof(float))
      while (--i >= 0) pElementM(X)[i]=((float*)Y)[i];
     else
      while (--i >= 0) pElementM(X)[i]=((double*)Y)[i];
    dw_free(Y);
   }
  else
   if (fread(pElementM(X),i*sizeof(PRECISION),1,f) != 1) goto EXIT_ERROR;

 return X;

EXIT_ERROR:
 fseek(f,position,SEEK_SET);
 if (del) FreeMatrix(X);
 if (Y) dw_free(Y);
 return (TMatrix)NULL;
}
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/*************************** Matrix Decompositions *****************************/
/*******************************************************************************/
/*
   Assumes
     U : m x qu matrix where qu is either m or min(m,n) or NULL pointer
     d : min(m,n)-vector
     V : n x qv matrix where qv is either n or min(m,n) or NULL pointer
     A : m x n matrix

   Returns
     1 upon success, and 0 on failure.  

   Results
     Finds U, V and d such that A = U * diag(d) * V'.  The matrices U and V have 
     orthonormal columns and the elemets of d are non-negative.  Here, diag(d) 
     denotes is the qu x qv diagonal matrix with the elements of d along the 
     diagonal.  The elements of d are non-negative and in descending order.

   Notes
     If either U or V is null, it is not computed.
*/
int SVD(TMatrix U, TVector d, TMatrix V, TMatrix A)
{
 int err, compact=1;
 if (!d || !A)
  {
   dw_Error(NULL_ERR);
   return 0;
  }
 if ((DimV(d) != ((RowM(A) < ColM(A)) ? RowM(A) : ColM(A))))
  {
   dw_Error(SIZE_ERR);
   return 0;
  }
 if (U)
   {
     if (U == V)
       {
	 dw_Error(ARG_ERR);
	 return 0;
       }
     if (RowM(U) != RowM(A))
       {
	 dw_Error(SIZE_ERR);
	 return 0;
       }
     if (ColM(U) != DimV(d))
       {
	 compact=0;
	 if (ColM(U) != RowM(U)) 
	   {
	     dw_Error(SIZE_ERR);
	     return 0;
	   }
       }
   }
 if (V)
   {
     if (RowM(V) != ColM(A))
       {
	 dw_Error(SIZE_ERR);
	 return 0;
       }
     if (ColM(V) != DimV(d))
       {
	 compact=0;
	 if (ColM(V) != RowM(V)) 
	   {
	     dw_Error(SIZE_ERR);
	     return 0;
	   }
       }
   }

 if (err=bSVD(U ? pElementM(U) : (PRECISION*)NULL,pElementV(d),V ? pElementM(V) : (PRECISION*)NULL,
	      pElementM(A),RowM(A),ColM(A),U ? MajorForm(U) : 1,V ? MajorForm(V) : 1,MajorForm(A),compact))
  {
    dw_Error(err);
    return 0;
  } 
 return 1;
}

/*
   Assumes
     S       : n x n matrix
     T       : n x n matrix
     Q       : n x n matrix or null pointer
     Z       : n x n matrix or null pointer
     A       : n x n matrix
     B       : n x n matrix
     alpha_r : n vector or null pointer
     alpha_i : n vector or null pointer
     beta    : n vector or null pointer

   Returns
     1 upon success, and 0 on failure.  

   Results
     Finds orthogonal matrices Q and Z, an block upper triangular matrix S with 
     1 x 1 or 2 x 2 blocks along the diagonal, and an upper triangular matrix T 
     such that

                        A = Q*S*Z'   and    B = Q*T*Z'

     the vectors alpha_r, alpha_i, and beta contain the generalized eigenvalues
     of A and B.  alpha_r contains the real part of alpha and alpha_i contains
     the imginary part.
 
     If Q, Z, alpha_r, alpha_i, or beta is null, then it is not returned.

   Notes
     The generalized eigenvalue is (alpha_r + i*alpha_i)/beta, but because beta 
     can be zero, alpha and beta are returned separately.  

     All the matrices and vectors should be distinct with the exception that A=S
     or B=T, but this is not checked. 
*/
int GeneralizedSchur_Real(TMatrix S, TMatrix T, TMatrix Q, TMatrix Z, TMatrix A, TMatrix B, 
			  TVector alpha_r, TVector alpha_i, TVector beta)
{
  int n, err;
  if (!S || !T || !A || !B) 
    {
      dw_Error(NULL_ERR);
      return 0;
    }
  n=RowM(A);
  if ((ColM(A) != n) || (RowM(B) != n) || (ColM(B) != n) || (RowM(S) != n) || (ColM(S) != n) || (RowM(T) != n) || (ColM(T) != n)
      || (Q && ((RowM(Q) != n) || (ColM(Q) != n))) || (Z && ((RowM(Z) != n) || (ColM(Z) != n))) 
      || (alpha_r && (DimV(alpha_r) != n)) || (alpha_i && (DimV(alpha_i) != n)) || (beta && (DimV(beta) != n)))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }

  err=bGeneralizedSchur_real(Q ? pElementM(Q) : (PRECISION*)NULL,Z ? pElementM(Z) : (PRECISION*)NULL,pElementM(S),pElementM(T),
	       pElementM(A),pElementM(B),n,Q ? MajorForm(Q) : 1,Z ? MajorForm(Z) : 1,MajorForm(S),MajorForm(T),
	       MajorForm(A),MajorForm(B),alpha_r ? pElementV(alpha_r) : (PRECISION*)NULL,
	       alpha_i ? pElementV(alpha_i) : (PRECISION*)NULL,beta ? pElementV(beta) : (PRECISION*)NULL);
 
  if (err == NO_ERR)
    return 1;
  else
    {
      dw_Error(err);
      return 0;
    }
}

/*
   Assumes
    SS      : n x n matrix
    TT      : n x n matrix
    QQ      : n x n matrix or null
    ZZ      : n x n matrix or null
    S       : n x n matrix.  S is block upper triangular with 1x1 or 2x2 blocks along the diagonal.
    T       : n x n matrix.  T is block upper triangular with positive diagonal. 
    Q       : n x n matrix or null.  Q is orthogonal if it is not null.
    Z       : n x n matrix or null.  Z is orghogonal if it is not null.
    select  : integer array of length n
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
     null, then ZZ is not computed.  The generalized eigenvalues of S and T 
     corresponding to the elements of select that are equal to one are 
     transformed to the first block of SS and TT.

   Notes
     Q, Z, S, and T should be the results of a call to GeneralizedSchur_Real(), SortGeneralizedSchur_Real(), 
     or ReorderGeneralizedSchur_Real().

     All the matrices and vectors should be distinct with the exception that 
     S=SS, T=TT, Q=QQ, or Z=ZZ, but this is not checked. 
*/
int ReorderGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
				  TMatrix Z, int *select, TVector alpha_r, TVector alpha_i, TVector beta)
{
  int n, err;
  if (!SS || !TT || !S || !T || !select) 
    {
      dw_Error(NULL_ERR);
      return 0;
    }
  n=RowM(SS);
  if ((ColM(SS) != n) || (RowM(TT) != n) || (ColM(TT) != n) || (RowM(S) != n) || (ColM(S) != n) || (RowM(T) != n) || (ColM(T) != n)
      || (QQ && ((RowM(QQ) != n) || (ColM(QQ) != n))) || (ZZ && ((RowM(ZZ) != n) || (ColM(ZZ) != n)))
      || (Q && ((RowM(Q) != n) || (ColM(Q) != n))) || (Z && ((RowM(Z) != n) || (ColM(Z) != n)))
      || (alpha_r && (DimV(alpha_r) != n)) || (alpha_i && (DimV(alpha_i) != n)) || (beta && (DimV(beta) != n)))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }
  err=bReorderGeneralizedSchur_real(select,QQ ? pElementM(QQ) : (PRECISION*)NULL,
		      ZZ ? pElementM(ZZ) : (PRECISION*)NULL,pElementM(SS),pElementM(TT),
		      Q ? pElementM(Q) : (PRECISION*)NULL,Z ? pElementM(Z) : (PRECISION*)NULL,pElementM(S),pElementM(T),n,
		      QQ ? MajorForm(QQ) : 1,ZZ ? MajorForm(ZZ) : 1,MajorForm(SS),MajorForm(TT),Q ? MajorForm(Q) : 1,
		      Z ? MajorForm(Z) : 1,MajorForm(S),MajorForm(T),alpha_r ? pElementV(alpha_r) : (PRECISION*)NULL,
		      alpha_i ? pElementV(alpha_i) : (PRECISION*)NULL,beta ? pElementV(beta) : (PRECISION*)NULL);
  
  if (err == NO_ERR)
    return 1;
  else
    {
      dw_Error(err);
      return 0;
    }
}

/*
   Assumes
    SS      : n x n matrix
    TT      : n x n matrix
    QQ      : n x n matrix or null
    ZZ      : n x n matrix or null
    S       : n x n matrix.  S is block upper triangular with 1x1 or 2x2 blocks 
              along the diagonal.
    T       : n x n matrix.  T is block upper triangular with positive diagonal.
    Q       : n x n matrix or null.  Q is orthogonal if it is not null.
    Z       : n x n matrix or null.  Z is orghogonal if it is not null.
    alpha_r : array of length n 
    alpha_i : array of length n
    beta    : array of length n

   Returns
     1 upon success and zero otherwise.

   Results
     Finds orthogonal matrices QQ and ZZ, an block upper triangular matrix SS 
     with 1 x 1 or 2 x 2 blocks along the diagonal, an upper triangular matrix TT 
     such that

              Q*S*Z' = QQ*SS*ZZ'   and    Q*T*Z' = QQ*TT*ZZ'
 
     If either Q or QQ are null, then QQ is not computed and if either Z or ZZ is
     null, then ZZ is not computed.  The generalized eigenvalues will be sorted 
     in absolute value according to the value of descend.

   Notes
     Q, Z, S, T, alpha_r, alpha_i, and beta should be the results of a call to
     GeneralizedSchur_Real() or ReorderGeneralizedSchur_Real().  All the matrices
     and vectors should be distinct except that S=SS, T=TT, Q=QQ, or Z=ZZ.
*/
int SortGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
			      TMatrix Z, TVector alpha_r, TVector alpha_i, TVector beta, int descend)
{
  int n, err;
  if (!SS || !TT || !S || !T || !alpha_r || !alpha_i || !beta)
    {
      dw_Error(NULL_ERR);
      return 0;
    }
  n=RowM(SS);
  if ((ColM(SS) != n) || (RowM(TT) != n) || (ColM(TT) != n) || (RowM(S) != n) || (ColM(S) != n) || (RowM(T) != n) || (ColM(T) != n)
      || (QQ && ((RowM(QQ) != n) || (ColM(QQ) != n))) || (ZZ && ((RowM(ZZ) != n) || (ColM(ZZ) != n)))
      || (Q && ((RowM(Q) != n) || (ColM(Q) != n))) || (Z && ((RowM(Z) != n) || (ColM(Z) != n)))
      || (DimV(alpha_r) != n) || (DimV(alpha_i) != n) || (DimV(beta) != n))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }

  err=bSortGeneralizedSchur_real(QQ ? pElementM(QQ) : (PRECISION*)NULL,ZZ ? pElementM(ZZ) : (PRECISION*)NULL,pElementM(SS),pElementM(TT),
		   Q ? pElementM(Q) : (PRECISION*)NULL,Z ? pElementM(Z) : (PRECISION*)NULL,pElementM(S),pElementM(T),n,
		   QQ ? MajorForm(QQ) : 1,ZZ ? MajorForm(ZZ) : 1,MajorForm(SS),MajorForm(TT),Q ? MajorForm(Q) : 1,Z ? MajorForm(Z) : 1,
				 MajorForm(S),MajorForm(T),pElementV(alpha_r),pElementV(alpha_i),pElementV(beta),descend);
  
  if (err == NO_ERR)
    return 1;
  else
    {
      dw_Error(err);
      return 0;
    }
}


/*
   Assumes
     U : m x m matrix or null pointer
     X : m x m symmetric positive definite matrix

   Results
     X = U' * U, where U is a upper triangular matrix with positive diagonal.

   Returns
     The matrix U is returned upon success and a null pointer is return upon 
     failure.  It the matrix U is null, it is created.

   Notes
     Failure usually indicates X is not positive definite.  Only the upper
     half of X is used.  The matrices U and X need not be distinct.
*/
TMatrix CholeskyUT(TMatrix U, TMatrix X)
{
  int err;
  if (!X) 
    {
      dw_Error(NULL_ERR); 
      return (TMatrix)NULL; 
    }
  if (RowM(X) != ColM(X))
    {
      dw_Error(SIZE_ERR); 
      return (TMatrix)NULL; 
    }
  if (!U)
    {
      if (!(U=EquateMatrix((TMatrix)NULL,X)))
	return (TMatrix)NULL;
      if (err=bCholesky(pElementM(U),RowM(U),1,MajorForm(U)))
	{
	  FreeMatrix(U);
	  dw_Error(err); 
	  return (TMatrix)NULL; 
	}
    }
  else
    {
      if (U != X) 
	{
	  if ((RowM(X) != RowM(U)) || (RowM(X) != ColM(U)))
	    {
	      dw_Error(SIZE_ERR);
	      return (TMatrix)NULL;
	    }
	  if (MajorForm(U) == MajorForm(X))
	    memcpy(pElementM(U),pElementM(X),RowM(X)*RowM(X)*sizeof(PRECISION));
	  else
	    bTranspose(pElementM(U),pElementM(X),RowM(X),ColM(X),MajorForm(X));
	}
      if (err=bCholesky(pElementM(U),RowM(X),1,MajorForm(U)))
	{
	  dw_Error(err); 
	  return (TMatrix)NULL; 
	}
    }
  return U; 
}

/*
   Assumes
     L : m x m matrix
     X : m x m symmetric positive definite matrix

   Results
     X = L' * L, where L is a lower triangular matrix with positive diagonal.

   Returns
     1 success
     0 failure, call GetError() to

   Notes
     Failure usually indicates X is not positive definite.  Only the upper
     half of X is used.  The matrices L and X need not be distinct.
*/
TMatrix CholeskyLT(TMatrix L, TMatrix X)
{
  int err;
  if (!X) 
    {
      dw_Error(NULL_ERR); 
      return (TMatrix)NULL;
    }
  if (RowM(X) != ColM(X))
    {
      dw_Error(SIZE_ERR); 
      return (TMatrix)NULL; 
    }
  if (!L)
    {
      if (!(L=EquateMatrix((TMatrix)NULL,X)))
	return (TMatrix)NULL;
      if (err=bCholesky(pElementM(L),RowM(X),0,MajorForm(L)))
	{
	  FreeMatrix(L);
	  dw_Error(err); 
	  return (TMatrix)NULL; 
	}
    }
  else
    {
      if (L != X) 
	{
	  if ((RowM(X) != RowM(L)) || (RowM(X) != ColM(L)))
	    {
	      dw_Error(SIZE_ERR);
	      return (TMatrix)NULL;
	    }
	  memcpy(pElementM(L),pElementM(X),RowM(X)*RowM(X)*sizeof(PRECISION));
	}
      if (err=bCholesky(pElementM(L),RowM(X),0,MajorForm(L)))
	{
	  dw_Error(err); 
	  return (TMatrix)NULL; 
	}
    }
  return L; 
}

/*
   Assumes
     Q : m x q matrix or null pointer
     R : q x n matrix
     X : m x n matrix

   Results
     Finds an orthogonal matrix Q and an upper triangular matrix R such that

                                   X = Q * R

   Returns
     1 - Success
     0 - Error, call GetError() to determine the type of error made.

   Notes
     The integer q must be equal to either m or the minimum of m and n. The 
     matrices R and X do not have to be distinct.
*/
int QR(TMatrix Q, TMatrix R, TMatrix X)
{
  int err;
  PRECISION *ptr;
  if (!R || !X)
    {
      dw_Error(NULL_ERR);
      return 0;
    }
  if (R != X)
    if (ColM(R) != ColM(X)) 
      {
	dw_Error(SIZE_ERR);
	return 0;
      }
    else
      if (RowM(R) == RowM(X))
	{
	  EquateMatrix(R,X);
	  ptr=pElementM(R);
	}
      else
	if ((RowM(R) == ColM(X)) && (ColM(X) < RowM(X)))
	  if (!(ptr=(PRECISION*)dw_malloc(RowM(X)*ColM(X)*sizeof(PRECISION))))
	    {
	      dw_Error(MEM_ERR);
	      return 0;
	    }
	  else
	    memcpy(ptr,pElementM(X),RowM(X)*ColM(X)*sizeof(PRECISION));
	else
	  {
	    dw_Error(SIZE_ERR);
	    return 0;
	  }
  else
    ptr=pElementM(R);
  if (!Q)
    err=bQR((PRECISION*)NULL,pElementM(R),ptr,RowM(X),ColM(X),RowM(R),0,MajorForm(R),MajorForm(X));
  else
    if (Q == R)
      {
	dw_Error(ARG_ERR);
	return 0;
      }
    else
      if ((RowM(Q) != RowM(X)) || (ColM(Q) != RowM(R)))
	{
	  dw_Error(SIZE_ERR);
	  return 0;
	}
      else
	err=bQR(pElementM(Q),pElementM(R),ptr,RowM(X),ColM(X),RowM(R),MajorForm(Q),MajorForm(R),MajorForm(X));
  if (ptr != pElementM(R)) dw_free(ptr);
  if (!err) return 1;
  dw_Error(err);
  return 0;
}

/*
   Assumes
     P : m x m permutation matrix
     X : m x n matrix
     A : m x n matrix invertible matrix

   Results
     Computes the LU decomposition of A with partial pivoting.  The 
     decomposition is

                                A = P * L * U

     where P is a permutation matrix, L is lower triangular with ones along
     the diagonal, and U is upper triangular.  These matrices are stored as
     follows.

        L is m x k, where k is the smaller of n and m, and is stored in the
        lower half of LU.  The diagonal of L is not stored.

        U is k x n, where k is the smaller of n and m, and is stored in the
        upper half of X, including the diagonal.   

        P is the integer representation of a permutation matrix.  See the
        header file matrix.h for a description of its internal 
        reqresentation.


   Returns
     1 : success
     0 : failure, call GetError() to determine the cause.

   Notes
     The matrices X and A do not have to be distinct.  Uses partial pivoting.
*/
int LU(TPermutation P, TMatrix X, TMatrix A)
{
 if (!P || !X || !A) 
  {
   dw_Error(NULL_ERR); 
   return 0; 
  }
 if (DimP(P) != RowM(A))
  {
   dw_Error(SIZE_ERR); 
   return 0; 
  }
 if ((X != A) && !EquateMatrix(X,A)) return 0; 
 bLU(pElementP(P),pElementM(X),RowM(X),ColM(X),MajorForm(X));
 UseP(P)=(RowM(X) < ColM(X)) ? RowM(X) : ColM(X);
 return 1;
}

/*
   Assumes
     x  : n vector or null pointer
     y  : n vector
     LU : n x n matrix with non-zero diagonal
     P  : Permutation matrix of dimension n

   Results
     Solves P * L * U * x = y, where P is a permutation matrix, L is lower
     triangular with ones along the diagonal, and U is upper triangular.

   Notes
     The vectors x and y do not have to be distinct. LU and P should be generated
     by the function LU(). 

     The matrices L and U are stored as follows:
        U is stored in the upper half of LU, including the diagonal.
        L is stored in the lower half of LU.  The diagonal of L is not stored.
*/
TVector LU_SolveCol(TVector x, TVector y, TMatrix LU, TPermutation P)
{
  TVector rtrn;
  int err;
  if (!y || !LU || !P)
    {
      dw_Error(NULL_ERR); 
      return (TVector)NULL; 
    }
  if ((DimV(y) != RowM(LU)) || (DimV(y) != ColM(LU)) || (DimV(y) != DimP(P))) 
    {
      dw_Error(SIZE_ERR); 
      return (TVector)NULL; 
    }
  if (rtrn=((x == y) ? x : EquateVector(x,y)))
    {
      bPermutationMultiply(pElementP(P),pElementV(rtrn),DimV(rtrn),1,UseP(P),1,1);
      bSolveUnitTriangular(pElementM(LU),pElementV(rtrn),DimV(rtrn),1,0,MajorForm(LU),1);
      if (err=bSolveTriangular(pElementM(LU),pElementV(rtrn),DimV(rtrn),1,1,MajorForm(LU),1))
	{
	  if (rtrn != x) FreeVector(rtrn);
	  rtrn=(TVector)NULL;
	  dw_Error(err);
	}
    }                        
  return rtrn;
}

/*
   Assumes
     x  : n vector
     y  : n vector
     LU : n x n matrix with non-zero diagonal
     P  : integer array of length n

   Results
     Solves x * P * L * U = y, where P is a permutation matrix, L is lower
     triangular with ones along the diagonal, and U is upper triangular.

   Notes
     The vectors x and y do not have to be distinct.

     The matrices P, L, and U are stored as follows.
        U is stored in the upper half of LU, including the diagonal.
        L is stored in the lower half of LU.  The diagonal of L is not stored.
*/
TVector LU_SolveRow(TVector x, TVector y, TMatrix LU, TPermutation P)
{
  TVector rtrn;
  int err;
  if (!y || !LU || !P)
    { 
      dw_Error(NULL_ERR); 
      return (TVector)NULL;
    }
  if ((DimV(y) != RowM(LU)) || (DimV(y) != ColM(LU)) || (DimV(y) != DimP(P))) 
    {
      dw_Error(SIZE_ERR); 
      return (TVector)NULL; 
    }
  if (rtrn=((x == y) ? x : EquateVector(x,y)))
    if (err=bSolveTriangular(pElementM(LU),pElementV(rtrn),DimV(rtrn),1,0,1^MajorForm(LU),0))
      {
	if (rtrn != x) FreeVector(rtrn);
	rtrn=(TVector)NULL;
	dw_Error(err);
      }
    else
      {
	bSolveUnitTriangular(pElementM(LU),pElementV(rtrn),DimV(rtrn),1,1,1^MajorForm(LU),0);
	bPermutationMultiply(pElementP(P),pElementV(rtrn),DimV(rtrn),1,UseP(P),0,0);
      }                        
  return rtrn;
}

/*
   Assumes
     Re : n vector
     Im : n vector
     X  : n x n matrix

   Results
     Re[i] + i*Im[i] is a eigenvalue of X.

   Returns
     1 upon success and 0 otherwise.
*/
int Eigenvalues(TVector Re, TVector Im, TMatrix X)
{
  int n;
  if (!Re || !Im || !X) 
    {
      dw_Error(NULL_ERR); 
      return 0;
    }
  if (Re == Im)
    {
      dw_Error(ARG_ERR);
      return 0;
    }
  n=RowM(X);
  if ((ColM(X) != n) || (DimV(Re) != n) || (DimV(Im) != n))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }
  return bEigenvalues(pElementV(Re),pElementV(Im),pElementM(X),n,MajorForm(X)) ? 0 : 1;
}

/*
   Assumes
     ReVal : n vector
     ImVal : n vector
     ReVec : n x n matrix
     ImVec : n x n matrix
     X     : n x n matrix

   Results
     Computes the eigenvalues and eigenvectors of X.  Note that
      
      (ReVec + i*ImVec)*(DiagonalMatrix(ReVal) + i*DiagonalMatrix(ImVal)) = X*(ReVec + i*ImVec)

   Returns
     1 upon success and 0 otherwise.
*/
int Eigen(TVector ReVal, TVector ImVal, TMatrix ReVec, TMatrix ImVec, TMatrix X)
{
  int n, rtrn;
  PRECISION *x;
  if (!ReVal || !ImVal || !ReVec || !ImVec || !X) 
    {
      dw_Error(NULL_ERR); 
      return 0;
    }
  if ((ReVal == ImVal) || (ReVec == ImVec))
    {
      dw_Error(ARG_ERR);
      return 0;
    }
  n=RowM(X);
  if ((ColM(X) != n) || (DimV(ReVal) != n) || (DimV(ImVal) != n) || (RowM(ReVec) != n) 
      || (ColM(ReVec) != n) || (RowM(ImVec) != n) || (ColM(ImVec) != n))
    {
      dw_Error(SIZE_ERR);
      return 0;
    }
  if ((X == ReVec) || (X == ImVec))
    {
      if (!(x=(PRECISION*)dw_malloc(n*n*sizeof(PRECISION))))
	{
	  dw_Error(MEM_ERR);
	  return 0;
	}
      memcpy(x,pElementM(X),n*n*sizeof(PRECISION));
    }
  else
    x=pElementM(X);
	     
  rtrn=bEigen(pElementV(ReVal),pElementV(ImVal),pElementM(ReVec),pElementM(ImVec),x,n,MajorForm(X),MajorForm(ReVec),MajorForm(ImVec));

  if (x != pElementM(X)) dw_free(x);

  return rtrn ? 0 : 1;
}

/*
   Assumes:
    X : matrix of the same size as Y or null pointer
    Y : symmetric and positive semi-definite matrix

   Results:
    Computes symmetric matrix X such that X*X = Y.

   Returns:
    X upon success and null upon failure.  If X is null, it is allocated.

   Notes:
    Uses the singular value decomposition to compute the square root.  Does not
    check if Y is symmetric or positive semi-definite.  
*/
TMatrix MatrixSquareRoot(TMatrix X, TMatrix Y)
{
  int i, j, n;
  TMatrix U, V;
  TVector d;
  PRECISION x;
  if (!Y)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  n=RowM(Y);
  if (n != ColM(Y))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (X && ((RowM(X) != n)|| (ColM(X) != n)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!SVD(U=CreateMatrix(n,n),d=CreateVector(n),V=CreateMatrix(n,n),Y))
    X=(TMatrix)NULL;
  else
    {
      // U=U*sqrt(d)
      for (j=n-1; j >= 0; j--)
	{
	  x=sqrt(ElementV(d,j));
	  for (i=n-1; i >= 0; i--)
	    ElementM(U,i,j)*=x;
	}
      // X = U*sqrt(d)*V'
      X=ProductTransposeMM(X,U,V);
    }
  FreeMatrix(V);
  FreeVector(d);
  FreeMatrix(U);
  return X;
}

/*******************************************************************************/
/**************************** Permutation Matrices *****************************/
/*******************************************************************************/
/*
   Assumes
     m : positive integer

   Results
     Creates TPermutation stucture of dimension m.

   Returns
     A pointer to the TPermutation structure upon success and null upon failure.
*/
TPermutation CreatePermutation(int m)
{
 TPermutation X;
 if (m <= 0) 
  {
   dw_Error(SIZE_ERR); 
   return (TPermutation)NULL; 
  }
 if (X=(TPermutation)dw_malloc((m+2)*sizeof(int)))
   {
    X[0]=m;
    X[1]=0;
   }
  else
   dw_Error(MEM_ERR);
 return (TPermutation)(X+2);
}

/*
   If X is not null, frees the TPermutation structure.
*/
void FreePermutation(TPermutation X)
{
 if (X) dw_free(X-2);
}

/*
   Assumes
     X : m-permutation or null pointer.
     i : 0 <= i < m
     j : 0 <= j < m
     m : positive integer

   Results
     X is initialized to be the permutation which is the transposition (i,j)
*/
TPermutation TranspositionPermutation(TPermutation X, int i, int j, int m)
{
  if ((m <= 0) || (i < 0) || (i >= m) || (j < 0) || (j >= m))
    {
      dw_Error(SIZE_ERR);
      return (TPermutation)NULL;
    }
  if (!X)
    {
      if (!(X=CreatePermutation(m)))
	return (TPermutation)NULL;
    }
  else
    if (DimP(X) != m)
      {
	dw_Error(SIZE_ERR);
	return (TPermutation)NULL;
      }
  if (j > i)
    {
      UseP(X)=i+1;
      ElementP(X,i)=j;
      for (i--; i >= 0; i--) ElementP(X,i)=i;
    }
  else
    if (i > j)
      {
	UseP(X)=j+1;
	ElementP(X,j)=i;
	for (j--; j >= 0; j--) ElementP(X,j)=j;
      }
    else
      UseP(X)=0;
  return X;
}

/*
   Assumes
     X : m-permutation or null pointer.
     p : integer array of length m.
     m : postive integer

   Results
     X is initialized to be the permutation which is the mapping i -> p[i].  If p
     does not represent a permutation, an error will not occur, but the results 
     will not be correct.
*/
TPermutation InitializePermutation(TPermutation X, int *p, int m)
{
  int i, j;
  if (!p)
    {
      dw_Error(NULL_ERR);
      return (TPermutation)NULL;
    }
  if (!X)
    {
      if (!(X=CreatePermutation(m)))
	return (TPermutation)NULL;
    }
  else
    if (DimP(X) != m)
      {
	dw_Error(SIZE_ERR);
	return (TPermutation)NULL;
      }
  ElementP(X,0)=p[0];
  for (i=1; i < m-1; i++)
    for (ElementP(X,i)=p[i], j=0; j < i; j++)
      if (ElementP(X,i) < j)
	break;
      else
	if (ElementP(X,i) == j)
	  ElementP(X,i)=ElementP(X,j);
	else
	  if (ElementP(X,i) == ElementP(X,j))
	    {
	      ElementP(X,i)=j;
	      break;
	    }
  for (i=m-2; i >= 0; i--)
    if (ElementP(X,i) != i) break;
  UseP(X)=i+1;
  return X;
}

/*
   Assumes:
     p : integer array of length m or null pointer
     X : m-permutation

   Results:
     Allocates the array p if it was null.  Fills p with the permutation which
     maps i to p[i].
*/
int* ExtractPermutation(int *p, TPermutation X)
{
  int i, j;
  if (!X) 
    { 
      dw_Error(NULL_ERR);
      return (int*)NULL;
    }
  if (!p && !(p=dw_malloc(DimP(X)*sizeof(int))))
    {
      dw_Error(MEM_ERR);
      return (int*)NULL;
    }
  for (i=DimP(X)-1; i >= 0; i--) p[i]=i;
  for (i=UseP(X)-1; i >= 0; i--)
    if (X[i] != i)
      {
	j=p[i];
	p[i]=p[X[i]];
	p[X[i]]=j;
      }
  return p;
}

/*
   Assumes
     X : m-permutation or null pointer
     Y : m-permutation

   Results
     X = Y.  If X is null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TPermutation EquatePermutation(TPermutation X, TPermutation Y)
{
 if (!Y) 
  {
   dw_Error(NULL_ERR); 
   return (TPermutation)NULL; 
  }
 if (X != Y)
   {
     if (!X)
       { 
	 if (!(X=CreatePermutation(DimP(Y)))) 
	   return (TPermutation)NULL; 
       }
     else
       if (DimP(X) != DimP(Y))
	 { 
	   dw_Error(SIZE_ERR); 
	   return (TPermutation)NULL; 
	 }
     UseP(X)=UseP(Y);
     memcpy(pElementP(X),pElementP(Y),UseP(Y)*sizeof(int));
   }
 return X;
}

/*
   Assumes
     X : m x m matrix pointer
     Y : m-permutation

   Results
     The matrix X is the matrix representation of the permutation Y.  If X is 
     the null pointer, X is created.

   Returns
     Returns X upon success and null on failure.  Call GetError() to
     determine the cause of failure.
*/
TMatrix PermutationMatrix(TMatrix X, TPermutation Y)
{
 if (!Y)
  {
   dw_Error(NULL_ERR);
   return (TMatrix)NULL;
  }
 if (!X)
   {
    if (!(X=CreateMatrix(DimP(Y),DimP(Y))))
     {
      dw_Error(MEM_ERR);
      return (TMatrix)NULL;
     }
   }
  else
   if ((RowM(X) != DimP(Y)) || (ColM(X) != DimP(Y)))
    {
     dw_Error(SIZE_ERR);
     return (TMatrix)NULL;
    }
 bPermutation(pElementM(X),pElementP(Y),DimP(Y),UseP(Y),MajorForm(X));
 return X;
}

TMatrix ProductPM(TMatrix X, TPermutation Y, TMatrix Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (DimP(Y) != RowM(Z))
  {
   dw_Error(SIZE_ERR);
   return (TMatrix)NULL; 
  }
 if ((X != Z) && !(X=EquateMatrix(X,Z))) return (TMatrix)NULL;
 bPermutationMultiply(pElementP(Y),pElementM(X),RowM(X),ColM(X),UseP(Y),0,MajorForm(X));
 return X;
}

TMatrix ProductMP(TMatrix X, TMatrix Y, TPermutation Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (ColM(Y) != DimP(Z))
  {
   dw_Error(SIZE_ERR);
   return (TMatrix)NULL; 
  }
 if ((X != Y) && !(X=EquateMatrix(X,Y))) return (TMatrix)NULL;
 bPermutationMultiply(pElementP(Z),pElementM(X),ColM(X),RowM(X),UseP(Z),1,1^MajorForm(X));
 return X;
}

TVector ProductPV(TVector x, TPermutation Y, TVector z)
{
 if (!Y || !z)
  {
   dw_Error(NULL_ERR); 
   return (TVector)NULL; 
  }
 if (DimP(Y) != DimV(z))
  {
   dw_Error(SIZE_ERR);
   return (TVector)NULL; 
  }
 if ((x != z) && !(x=EquateVector(x,z))) return (TVector)NULL;
 bPermutationMultiply(pElementP(Y),pElementV(x),DimV(x),1,UseP(Y),0,1);
 return x;
}

TVector ProductVP(TVector x, TVector y, TPermutation Z)
{
 if (!y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TVector)NULL; 
  }
 if (DimV(y) != DimP(Z))
  {
   dw_Error(SIZE_ERR);
   return (TVector)NULL; 
  }
 if ((x != y) && !(x=EquateVector(x,y))) return (TVector)NULL;
 bPermutationMultiply(pElementP(Z),pElementV(x),DimV(x),1,UseP(Z),1,1);
 return x;
}

TMatrix TransposeProductPM(TMatrix X, TPermutation Y, TMatrix Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (DimP(Y) != RowM(Z))
  {
   dw_Error(SIZE_ERR);
   return (TMatrix)NULL; 
  }
 if ((X != Z) && !(X=EquateMatrix(X,Z))) return (TMatrix)NULL;
 bPermutationMultiply(pElementP(Y),pElementM(X),RowM(X),ColM(X),UseP(Y),1,MajorForm(X));
 return X;
}

TMatrix ProductTransposeMP(TMatrix X, TMatrix Y, TPermutation Z)
{
 if (!Y || !Z)
  {
   dw_Error(NULL_ERR); 
   return (TMatrix)NULL; 
  }
 if (ColM(Y) != DimP(Z))
  {
   dw_Error(SIZE_ERR);
   return (TMatrix)NULL; 
  }
 if ((X != Y) && !(X=EquateMatrix(X,Y))) return (TMatrix)NULL;
 bPermutationMultiply(pElementP(Z),pElementM(X),ColM(X),RowM(X),UseP(Z),0,1^MajorForm(X));
 return X;
}


void PrintPermutation(FILE *f, TPermutation X)
{
 int i, j, k;
 for (i=0; i < DimP(X); i++)
  fprintf(f,"%3d ",i);
 fprintf(f,"\n");
 for (i=0; i < DimP(X); i++)
  {
    if (i < UseP(X))
      {
	k=ElementP(X,i);
	j=i-1;
      }
    else
      {
	k=i;
	j=UseP(X)-1;
      }
   for ( ; j >= 0; j--)
    if (k == ElementP(X,j)) k=j;
   fprintf(f,"%3d ",k); 
  }
 fprintf(f,"\n");
}
/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

