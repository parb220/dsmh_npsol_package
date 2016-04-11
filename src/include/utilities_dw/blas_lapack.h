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

#ifndef __BLAS_LAPACK__
#define __BLAS_LAPACK__

#if !defined(MATLAB_MEX_FILE) && !defined(OCTAVE_MEX_FILE)

#include "dw_std.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**************** Linux defines because underscore is appended *****************/
#ifdef __linux__

/* BLAS */

#define dcopy     dcopy_     // Blas copy
#define scopy     scopy_

#define ddot      ddot_      // Blas dot product of vectors
#define sdot      sdot_     
#define zdotc     zdotc_
#define zdotu     zdotu_ 

#define dscal     dscal_     // Blas scalar times vector
#define sscal     sscal_

#define daxpy     daxpy_     // Blas vector plus scalar times vector
#define saxpy     saxpy_

#define dgemm     dgemm_     // Blas matrix multiplication
#define sgemm     sgemm_
#define zgemm     zgemm_

#define dtrsm     dtrsm_     // Solves triangular system
#define strsm     strsm_

#define dtrmm     dtrmm_     // Blas matrix multiplication - triangular
#define strmm     strmm_

#define dgemv     dgemv_     // Blas vector matrix multiplication
#define sgemv     sgemv_
#define zgemv     zgemv_


/* LAPACK */ 
#define dgesdd    dgesdd_    // SVD decomposition (divide and conquer)
#define sgesdd    sgesdd_

#define dgesvd    dgesvd_    // SVD decomposition (QR)
#define sgesvd    sgesvd_

#define dgetrf    dgetrf_    // LU decomposition
#define sgetrf    sgetrf_

#define dgetri    dgetri_    // computes inverse from LU decomposition
#define sgetri    sgetri_

#define dgetrs    dgetrs_    // Solves system using previously computed LU decomposition
#define sgetrs    sgetrs_

#define dgesv     dgesv_     // Solves system using LU decomposition
#define sgesv     sgesv_
#define zgesv     zgesv_

#define dgeqrf    dgeqrf_    // QR decomposition
#define sgetrf    sgetrf_

#define dorgqr    dorgqr_    // Forms orthogonal matrix from Housholder matrices created by dgeqrf
#define sorgqr    sorgqr_

#define dgelqf    dgelqf_    // LQ decompostion
#define sgelqf    sgelqf_

#define dorglq    dorglq_    // Forms orthogonal matrix from Housholder matrices created by dgeqlq
#define sorglq    sorglq_

//====== Added by HWu ======
#define dgees     dgees_    	// Schur decomposition
#define dtrsen    dtrsen_	// Reorder the real Schur factorization
//==========================

//====== Added by HWu ======
#define dlange	dlange_		// 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any element of a general rectangular matrix
#define dgecon	dgecon_		// reciprocal of the condition number of a general real matrix
//==========================

#define dgges     dgges_     // Generalized Schur decomposition
#define sgges     sgges_

#define dtgsen    dtgsen_    // Reorders generalized Schur decomposition
#define stgsen    stgsen_

#define dtgexc    dtgexc_    // Reorders generalized Schur decomposition
#define stgexc    stgexc_

#define dsyev     dsyev_
#define ssyev     ssyev_

#define dgeev     dgeev_     // Computes eigenvalues and eigenvectors
#define sgeev     sgeev_

#define dsyevr    dsyevr_    // Computes eigenvalues and eigenvectors of a symmetric matrix
#define ssyevr    ssyevr_

#define dpotrf    dpotrf_    // Computes the Cholesky decompostion
#define spotrf    spotrf_

#define dpotri    dpotri_
#define spotri    spotri_

#define dgeev     dgeev_     
#define sgeev     sgeev_

#define dtrtrs    dtrtrs_   // solves triangular system
#define strtrs    strtrs_

#define dtrtri    dtrtri_   // computes inverse of triangular matrix
#define strtri    strtri_

// Added by HWu
#define dgerfs	dgerfs_	// refine solution of linear equations and estimate errors 
#define dpotrs	dpotrs_ // solving linear systems for symmetric positive definite matrix
#define dporfs	dporfs_ // refine solution of linear equations and estimate errors
#define dsytrf	dsytrf_ // LU decomposition of symmetric indefinite matrix
#define dsytrs	dsytrs_ // solving linear systems for symmetric indefinite matrix
#define dsyrfs  dsyrfs_ // refine solution
#define ilaenv  ilaenv_ // determine optimal blocksize
#define dtrrfs	dtrrfs_ // refine solution of linear equations
// Added by HWu
#endif
/*******************************************************************************/



/*******************************************************************************/
/* BLAS */
double dcopy(blas_int*,double*,blas_int*,double*,blas_int*);
float  scopy(blas_int*,float*,blas_int*,float*,blas_int*);

double ddot(blas_int*,double*,blas_int*,double*,blas_int*);
float  sdot(blas_int*,float*,blas_int*,float*,blas_int*);
void  zdotc(double*,blas_int*,double*,blas_int*,double*,blas_int*);
void   zdotu(double*,blas_int*,double*,blas_int*,double*,blas_int*);

void dscal(blas_int*,double*,double*,blas_int*);
void sscal(blas_int*,float*,float*,blas_int*);

void daxpy(blas_int*,double*,double*,blas_int*,double*,blas_int*);
void saxpy(blas_int*,float*,float*,blas_int*,float*,blas_int*);

void dgemm(char*,char*,blas_int*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*,double*,double*,blas_int*);
void sgemm(char*,char*,blas_int*,blas_int*,blas_int*,float*,float*,blas_int*,float*,blas_int*,float*,float*,blas_int*);
void zgemm(char*,char*,blas_int*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*,double*,double*,blas_int*);

void dtrsm(char*,char*,char*,char*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*);
void strsm(char*,char*,char*,char*,blas_int*,blas_int*,float*,float*,blas_int*,float*,blas_int*);

void dtrmm(char*,char*,char*,char*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*);
void strmm(char*,char*,char*,char*,blas_int*,blas_int*,float*,float*,blas_int*,float*,blas_int*);

void dgemv(char*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*,double*,double*,blas_int*);
void sgemv(char*,blas_int*,blas_int*,float*,float*,blas_int*,float*,blas_int*,float*,float*,blas_int*);
void zgemv(char*,blas_int*,blas_int*,double*,double*,blas_int*,double*,blas_int*,double*,double*,blas_int*);


/* LAPACK */
void dgesdd(char*,lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*,lapack_int*);
void sgesdd(char*,lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*,lapack_int*);

void dgesvd(char*,char*,lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*);
void sgesvd(char*,char*,lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*);

void dgeqrf(lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*);
void sgeqrf(lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*);

void dgetrf(lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*,lapack_int*);
void sgetrf(lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*,lapack_int*);

void dgetri(lapack_int*,double*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*);
void sgetri(lapack_int*,float*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*);

void dgesv(lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*);
void sgesv(lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*);
void zgesv(lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*);

void dgetrs(char*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*);
void sgetrs(char*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*);

void dorgqr(lapack_int*,lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*);
void sorgqr(lapack_int*,lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*);

void dgelqf(lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*);
void sgelqf(lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*);

void dorglq(lapack_int*,lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*);
void sorglq(lapack_int*,lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*);

void dgges(char*,char*,char*,lapack_int*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*,double*,double*,double*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,void*,lapack_int*);
void sgges(char*,char*,char*,lapack_int*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*,float*,float*,float*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,void*,lapack_int*);

void dtgsen(lapack_int*,lapack_int*,lapack_int*,lapack_int*,lapack_int*,double*,lapack_int*,double*,lapack_int*,double*,double*,double*,double*,lapack_int*,double*,lapack_int*,lapack_int*,double*,double*,double*,double*,lapack_int*,lapack_int*,lapack_int*,lapack_int*);
void stgsen(lapack_int*,lapack_int*,lapack_int*,lapack_int*,lapack_int*,float*,lapack_int*,float*,lapack_int*,float*,float*,float*,float*,lapack_int*,float*,lapack_int*,lapack_int*,float*,float*,float*,float*,lapack_int*,lapack_int*,lapack_int*,lapack_int*);

//====== Added by HWu ======
void dgees(char *jobvs, char *sort, int *select, int *n, double *a, int *lda, int *sdim, double *wr, double *wi, double *vs, int *ldvs, double *work, int *lwork, int *bwork, int *info);  
void dtrsen(char *job, char *compq, int *select, int *n, double *t, int *ldt, double *q, int *ldq, double *wr, double *wi, int *m, double *s, double *sep, double *work, int *lwork, int *iwork, int *liwork, int *info); 

void dgerfs(char *trans, int *n, int *nrhs, double *a, int *lda, double *af, int *ldaf, int *ipiv, double *b, int *ldb, double *x, int *ldx, double *ferr, double *berr, double *work, int *iwork, int *inf); 

void dpotrs(char *uplo, int *n, int *nrhs, double *a, int *lda, double *b, int *ra, int *info);

void dporfs(char *uplo, int *n, int *nrhs, double *a, int *lda, double *af, int *ldaf, double *b, int *ldb, double *x, int *ldx, double *ferr, double *berr, double *work, int *iwork, int *info); 

void dsytrf(char *uplo, int *n, double *a, int *lda, int *ipiv, double *work, int *lWork, int *info); 

void dsytrs(char *uplo, int *n, int *nrhs, double *a, int *lda, int *ipiv, double *b, int *ldb, int *info);

void dsyrfs(char *uplo, int *n, int *nrhs, double *a, int *lda, double *fa, int *ldfa, int *ipiv, double *b, int *ldb, double *x, int *ldx, double *ferr, double *berr, double *work, int *iwork, int *info);

int ilaenv(int *ispec, char *name, char *opts, int *n1, int *n2, int *n3, int *n4); 

void dtrrfs(char *uplo, char *trans, char *diag, int *n, int *nrhs, double *a, int *lda, double *b, int *ldb, double *x, int *ldx,double *Ferr, double *Berr, double *dWork, int *iWork, int *info);

void dgecon(char *norm, int *n, double *a, int *lda, double *anorm, double *rcond, double *workspace, int *iworkspace, int *info); 
double dlange(const char *norm, const int *m, const int *n, double *a, const int *lda, double *workspace); 

//==========================

void dtgexc(lapack_int*,lapack_int*,lapack_int*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*);
void stgexc(lapack_int*,lapack_int*,lapack_int*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*);

void dsyev(char*,char*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*);
void ssyev(char*,char*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*);

void dsyevr(char*,char*,char*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*,double*,lapack_int*,double*,double*,lapack_int*,lapack_int*,double*,lapack_int*,lapack_int*,lapack_int*,lapack_int*);
void ssyevr(char*,char*,char*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*,float*,lapack_int*,float*,float*,lapack_int*,lapack_int*,float*,lapack_int*,lapack_int*,lapack_int*,lapack_int*);

void dgeev(char*,char*,lapack_int*,double*,lapack_int*,double*,double*,double*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*);
void sgeev(char*,char*,lapack_int*,float*,lapack_int*,float*,float*,float*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*);

void dpotrf(char*,lapack_int*,double*,lapack_int*,lapack_int*);
void spotrf(char*,lapack_int*,float*,lapack_int*,lapack_int*);

void dpotri(char*,lapack_int*,double*,lapack_int*,lapack_int*);
void spotri(char*,lapack_int*,float*,lapack_int*,lapack_int*);

void dtrtrs(char*,char*,char*,lapack_int*,lapack_int*,double*,lapack_int*,double*,lapack_int*,lapack_int*);
void strtrs(char*,char*,char*,lapack_int*,lapack_int*,float*,lapack_int*,float*,lapack_int*,lapack_int*);

void dtrtri(char*,char*,lapack_int*,double*,lapack_int*,lapack_int*);
void strtri(char*,char*,lapack_int*,float*,lapack_int*,lapack_int*);

/*******************************************************************************/

#ifdef __cplusplus
}
#endif

#endif

#endif
