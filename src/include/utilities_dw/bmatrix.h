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

#ifndef __BMATRIX__
#define __BMATRIX__

#include "prcsn.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Unary Operators */
int bNegative(PRECISION *x, PRECISION *y, int m);
int bAbs(PRECISION *x, PRECISION *y, int m);

/* Transpose */
int bTranspose(PRECISION *x, PRECISION *y, int m, int n, int t);
int bTransposeInPlace(PRECISION *x, int m);

/* Addition */
int bAddV(PRECISION *x, PRECISION *y, PRECISION *z, int m);
int bSubtractV(PRECISION *x, PRECISION *y, PRECISION *z, int m);
int bAddM(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int xt, int yt, int zt);
int bSubtractM(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int xt, int yt, int zt);
int bLinearCombinationV(PRECISION *x, PRECISION a, PRECISION *y, PRECISION b,PRECISION *z, int m);
int bLinearCombinationM(PRECISION *x, PRECISION a, PRECISION *y, PRECISION b,PRECISION *z, int m, int n, int xt, int yt, int zt);

/* Multiplication */
int bMultiply(PRECISION *x, PRECISION *y, PRECISION s, int m);

int bProductMM_Update(PRECISION *x, PRECISION *y, PRECISION *z, PRECISION a, PRECISION b, int m, int n, int p, int xt, int yt, int zt);
int bProductMV_Update(PRECISION *x, PRECISION *y, PRECISION *z, PRECISION a, PRECISION b, int m, int n, int zt);

#define bProductMM(x,y,z,m,n,p,xt,yt,zt) bProductMM_Update(x,y,z,0.0,1.0,m,n,p,xt,yt,zt)
#define bProductMV(x,y,z,m,n,zt) bProductMV_Update(x,y,z,0.0,1.0,m,n,zt)
#define bProductVM(x,y,z,m,n,zt) bProductMV_Update(x,y,z,0.0,1.0,n,m,1^zt)
int bProductMT(PRECISION *x, PRECISION *y, int m, int n, int u, int xt, int yt);
int bProductTM(PRECISION *x, PRECISION *y, int m, int n, int u, int xt, int yt);

/* LU Decomposition */
int bSolveLU(PRECISION *b, PRECISION *a, int m, int n, int bt, int at);
int bLU(int *p, PRECISION *x, int m, int n, int xt);
int bSolveTriangular(PRECISION *x, PRECISION *b, int m, int n, int u, int xt, int bt);
int bSolveUnitTriangular(PRECISION *x, PRECISION *b, int m, int n, int u, int xt, int bt);

/* QR Decompositions */
int bQR(PRECISION *Q, PRECISION *R, PRECISION *X, int m, int n, int q, int qt, int rt, int xt);

/* Singular Value Decomposition */
int bSVD(PRECISION *U, PRECISION *d, PRECISION *V, PRECISION *A, int m, int n, int ut, int vt, int at, int compact);

/* Generalize Schur Decomposition */
int bGeneralizedSchur_real(PRECISION *Q, PRECISION *Z, PRECISION *S, PRECISION *T, PRECISION *A, PRECISION *B, int n, int qt, int zt, 
	     int st, int tt, int at, int bt, PRECISION *alpha_r, PRECISION *alpha_i, PRECISION *beta);
int bReorderGeneralizedSchur_real(int *select, PRECISION *QQ, PRECISION *ZZ, PRECISION *SS, PRECISION *TT, PRECISION *Q, PRECISION *Z, 
				  PRECISION *S, PRECISION *T, int n, int qqt, int zzt, int sst, int ttt, int qt,
				  int zt, int st, int tt, PRECISION *alpha_r, PRECISION *alpha_i, PRECISION *beta);
int bSortGeneralizedSchur_real(PRECISION *QQ, PRECISION *ZZ, PRECISION *SS, PRECISION *TT, PRECISION *Q, PRECISION *Z, 
			       PRECISION *S, PRECISION *T, int n, int qqt, int zzt, int sst, int ttt, int qt, int zt, 
			       int st, int tt, PRECISION *alpha_r, PRECISION *alpha_i, PRECISION *beta, int descend);

/* Cholesky Decompositions */
int bCholesky(PRECISION *X, int m, int u, int t);

/* Eigenvalues and Eigenvectors */
int bEigenvalues(PRECISION *re, PRECISION *im, PRECISION *x, int n, int xt);
int bEigen(PRECISION *reval, PRECISION *imval, PRECISION *revec, PRECISION *imvec, PRECISION *x, int n, int xt, int revect, int imvect);

/* Permutation Routines */
int bPermuteMatrix(PRECISION *x, PRECISION *y, int *p, int m, int n, int yt, int pt);
int bPermuteMultiply(int *r, int *p, int *q, int n, int pt, int qt);
int bPermuteInverse(int *r, int *p, int n);
int bPermutationMultiply(int *p, PRECISION *y, int m, int n, int q, int pt, int yt);
int bPermutation(PRECISION *x, int *p, int m, int q, int t);

/* Tensor Calculus */
int bMatrixTensor(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n, int r, int s, int xt, int yt, int zt);
int bVectorTensor(PRECISION *x, PRECISION *y, PRECISION *z, int m, int n);

/* Inner Product */
PRECISION bInnerProduct(PRECISION *x, PRECISION *y, PRECISION *S, int n);

#ifdef __cplusplus
}
#endif

#endif 


