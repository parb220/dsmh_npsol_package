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

#include "dw_rand.h"
#include "dw_matrix_rand.h"
#include "dw_error.h"
#include "dw_std.h"

#include <math.h>


/******************************************************************************/
/************************ Random Matrices and Vectors *************************/
/******************************************************************************/
/*
   Assumes
     x : m-vector

   Results
     Fills x with deviates drawn from the uniform distribution on [0,1]
*/
TVector dw_UniformVector(TVector x)
{
 int i;
 if (!x) { dw_Error(NULL_ERR); return (TVector)NULL; }
 for (i=DimV(x)-1; i >= 0; i--) ElementV(x,i)=dw_uniform_rnd();
 return x;
}

/*
   Assumes
     X : m x n matrix

   Results
     Fills X with deviates drawn from the uniform distribution on [0,1]
*/
TMatrix dw_UniformMatrix(TMatrix X)
{
 int i;
 PRECISION *pX;
 if (!X) { dw_Error(NULL_ERR); return (TMatrix)NULL; } 
 for (pX=pElementM(X), i=RowM(X)*ColM(X)-1; i >= 0; i--) pX[i]=dw_uniform_rnd();
 return X;
}

/*
   Assumes
     x : m-vector

   Results
     Fills x with independent standard normal deviates
*/
TVector dw_NormalVector(TVector x)
{
 int i;
 if (!x) { dw_Error(NULL_ERR); return (TVector)NULL; }
 for (i=DimV(x)-1; i >= 0; i--) ElementV(x,i)=dw_gaussian_rnd();
 return x;
}

/*
   Assumes
     X : m x n matrix

   Results
     Fills X with independent standard normal deviates
*/
TMatrix dw_NormalMatrix(TMatrix X)
{
 int i;
 PRECISION *pX;
 if (!X) { dw_Error(NULL_ERR); return (TMatrix)NULL; } 
 for (pX=pElementM(X), i=RowM(X)*ColM(X)-1; i >= 0; i--) pX[i]=dw_gaussian_rnd();
 return X;
}

/*
   Assumes:
    x - n-vector or null pointer
    R - n x m upper triangular matrix with non-zero elements along the diagonal
    a - m-vector with a(i) <= b(i)
    b - m-vector with a(i) <= b(i)

   Returns:
    Returns x upon success and null otherwise.  If x is null, then it is created.

   Results:
    The vector x is filled with a truncated normal draw.  The truncation is of
    the form:

        a' <= gamma'*R <= b'

    where gamma is standard normal.

   Notes:
    Since R is upper triangular, the restrictions are of the form

         a(i) <= {gamma(0)*R(i,0) + ... + gamma(i-1)*R(i,i-1)} 
                                           + gamma(i)*R(i,i) <= b(i)

    Suppose the restrictions are of the form

        a' <= epsilon'*S <= b'

    where epsilon is normal with mean mu and variance Sigma.  Let 
    
        Sigma=T*T'

    be the Cholesky decomposition of Sigma and 
  
        T'*S=Q*R

    be the QR-decomposition of T'*S.  Then the above restrictions can be written 
    as

        a' - mu'*S <= gamma'*R <= b' - mu'*S

    where gamma'=(epsilon' - mu')*inv(T')*Q or epsilon'=gamma'*Q'*T' + mu' and
    gamma is standard normal.    
*/
TVector dw_TruncatedNormalVector(TVector x, TMatrix R, TVector a, TVector b)
{
  int i, j, n, m;
  PRECISION dot, d;
  // Checks
  if (!R || !a || !b)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  n=RowM(R);
  m=ColM(R);
  if ((m > n) || (DimV(a) != m) || (DimV(b) != m))
    {
      dw_Error(SIZE_ERR);
      return (TVector)NULL;
    }
  if (!x)
    { if (!(x=CreateVector(n))) return (TVector)NULL; }
  else
    if (DimV(x) != n)
      {
	dw_Error(SIZE_ERR);
	return (TVector)NULL;
      }

  // draw truncated normals
  // Note: dw_truncated_gaussian_rnd() makes no assumption about the ordering of the arguments.
  d=ElementM(R,0,0);
  ElementV(x,0)=dw_truncated_gaussian_rnd(ElementV(a,0)/d,ElementV(b,0)/d);
  for (i=1; i < m; i++)
    {
      for (dot=ElementV(x,0)*ElementM(R,0,i), j=i-1; j > 0; j--) dot+=ElementV(x,j)*ElementM(R,j,i);
      d=ElementM(R,i,i);
      ElementV(x,i)=dw_truncated_gaussian_rnd((ElementV(a,i)-dot)/d,(ElementV(b,i)-dot)/d);
    }

  // draw unrestricted normals
  for ( ; i < n; i++)
    ElementV(x,i)=dw_gaussian_rnd();

  return x;
}

/*
   Assumes
     x : m-vector

   Results
     Fills x with independent log normal deviates.  The mean and standard 
     deviation of the underlying normal distribution are passed.
*/
TVector dw_LogNormalVector(TVector x, PRECISION mean, PRECISION standard_deviation)
{
  int i;
  if (!x) { dw_Error(NULL_ERR); return (TVector)NULL; }
  for (i=DimV(x)-1; i >= 0; i--) ElementV(x,i)=dw_lognormal_rnd(mean,standard_deviation);
  return x;
}

/*
   Computes a matrix of gamma deviates.  If x, a, and b represent X(i,j), 
   A(i,j), and B(i,j), then density of x is

                                 x^(a-1) exp(-x/b)
                                ------------------ 
                                   gamma(a) b^a
*/
TMatrix dw_GammaMatrix(TMatrix X, TMatrix A, TMatrix B)
{
  int i;
  PRECISION *pX, *pA, *pB;
  if (!A || !B)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if ((RowM(A) != RowM(B)) || (ColM(A) != ColM(B)))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }
  if (!X)
    {
      if (!(X=CreateMatrix(RowM(A),ColM(A))))
	return (TMatrix)NULL;
    }
  else
    if ((RowM(X) != RowM(A)) || (ColM(X) != ColM(A)))
      {
	dw_Error(SIZE_ERR);
	return (TMatrix)NULL;
      }
  for (pX=pElementM(X), pA=pElementM(A), pB=pElementM(B), i=RowM(X)*ColM(X)-1; i >= 0; i--)
    pX[i]=pB[i]*dw_gamma_rnd(pA[i]);
  return X;
}


/*
   Assumes
     X : m x m matrix
     S : m x m non-singular matrix

   Results
     X is drawn from the Wishart distribution with parameters sigma, nu, and m,
     where sigma=Inverse(S'S).  The pdf of X is proportional to

       |det(X)|^(0.5*(nu - m - 1))
      ---------------------------- * exp(-0.5*tr(Inverse(sigma*X))
         |det(sigma)|^(0.5*nu)

        = |det(X)|^(0.5*(nu - m - 1)) |det(S)|^(nu) exp(-0.5*tr(S'*X*S))
*/
TMatrix dw_Wishart(TMatrix X, TMatrix S, int nu)
{
 int m=RowM(S);
 TMatrix Z;

 if ((m != ColM(S)) || (m != RowM(X)) || (m != ColM(X))) dw_Error(SIZE_ERR);

 Z=dw_NormalMatrix(CreateMatrix(m,nu));
 ProductMM(Z,S,Z);
 ProductTransposeMM(X,Z,Z);
 FreeMatrix(Z);
 return X;
}

/*
   Assumes
     x : n x n matrix
     T : n x n upper triangular matrix

   Results
     x is drawn from the multivariate student-t distribution with parameters.
     The pdf of x is given by
*/
TVector dw_StudentT(TVector x, TMatrix T, int nu)
{
 PRECISION r=0.0, s;
 int i, n=DimV(x);
 if ((n != ColM(T)) || (n != RowM(T))) dw_Error(SIZE_ERR);
 dw_NormalVector(x);
 ProductMV(x,T,x);
 for (i=nu; i > 0; i--)
  {
   s=dw_gaussian_rnd();
   r+=s*s;
  }
 ProductSV(x,sqrt((PRECISION)nu/r),x);
 return x;
}

TMatrix dw_UniformOrthogonal(TMatrix Q)
{
  TMatrix X;
  int i, j, err;

  if (!Q)
    {
      dw_Error(NULL_ERR);
      return (TMatrix)NULL;
    }
  if (RowM(Q) != ColM(Q))
    {
      dw_Error(SIZE_ERR);
      return (TMatrix)NULL;
    }

  /* Uncomment to use IMSL implementation */
  //imsls_d_random_orthogonal_matrix(RowM(Q),IMSLS_RETURN_USER,pElementM(Q),0);
  /**/


  /* Uncomment to use C code implementation */
  X=dw_NormalMatrix(CreateMatrix(RowM(Q),ColM(Q)));
  if (!(err=QR(Q,X,X)))
  for (i=RowM(X)-1; i >= 0; i--)
    if (ElementM(X,i,i) < 0)
      for (j=RowM(Q)-1; j >= 0; j--) ElementM(Q,j,i)=-ElementM(Q,j,i);
  FreeMatrix(X);
  if (err) return (TMatrix)NULL;
  /**/

  return Q;
}

/*
   Assumes:
     x : m-vector

   Results:
     The vector x is filled with a vector drawn from the uniform distribution on 
     the m-1 dimensional unit sphere.

   Returns:
     The vector x.

   Notes:
     The vector is obtained by drawing a m-vector from the standard normal 
     distribution and then normalizing its length to one.
*/
TVector dw_UniformUnitSphere(TVector x)
{
  PRECISION r;
  if (!x)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  do
    dw_NormalVector(x);
  while ((r=Norm(x)) == 0.0);
  return ProductSV(x,1.0/r,x);
}


/*
   Assumes:
     x : m-vector

   Results:
     The vector x is filled with a vector drawn from the uniform distribution on 
     the m dimensional solid unit sphere.

   Returns:
     Upon success, returns the norm of x, upon failure returns -1.0.

   Notes:
     The vector is drawn by drawing a m-vector from the standard normal 
     distribution and a real number u from the uniform distribution on [0,1], and 
     normalizing the vector so its length equal to u^(1/m). 
*/
TVector dw_UniformUnitBall(TVector x)
{
  PRECISION r, s;
  if (!x)
    {
      dw_Error(NULL_ERR);
      return (TVector)NULL;
    }
  do
    dw_NormalVector(x);
  while ((s=Norm(x)) == 0.0);
  ProductSV(x,(r=pow(dw_uniform_rnd(),1.0/DimV(x)))/s,x);
  return x;
}

/******************************************************************************/
/******************************************************************************/

