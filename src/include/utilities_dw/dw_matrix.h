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

/********************************************************************************
 VECTORS AND MATRICES
 A TVector is an array of floating points together with the dimension of the 
 vector.  A TVector implementation can contain additional information.  An
 instance of TVector must be created with calls to CreateVector() and freed with
 calls to FreeVector(). The following macros must be defined.

 DimV(x)       - x      : TVector
                 Returns: int containing the dimension.

 ElementV(x,i) - x      : TVector
                 i      : integer
                 Returns: L-value PRECISION containing the ith element.  The
                          index i is zero-based.

 pElementV(x)  - x      : TVector
                 Returns: pointer to 0th element of the array storing the vector.

 A TMatrix is an array of floating points together with the number of rows and 
 columns in the matrix.  If the storage type (row or column major) is variable,
 then it also must be stored.  A TMatrix implementation can contain additional 
 information. A instance of TMatrix must be created with CreateMatrix() and freed
 with FreeMatrix().  The following macros must be defined.

 RowM(x)         - x      : TMatrix
                   Returns: int containing the number of rows.

 ColM(x)         - x      : TMatrix
                   Returns: int containing the number of columns.

 ElementM(x,i,j) - x      : TVector
                   i      : int
                   j      : int
                   Returns: L-value PRECISION containing the element in the ith 
                            row and jth column.  The indexes i and j are zero
                            based.

 pElementM(x)    - x      : TMatrix
                   Returns: pointer to 0th element of the array storing the 
                            matrix.
 
 MajorForm(x)    - x      : TMatrix
                   Returns: 0 if data stored in row major format and 1 if data
                            stored in column major format.  The data is in row
                            major format if

                               ElementM(x,i,j) = pElementM(x)[i*ColM(x)+j]

                            and is in column major format if 

                               ElementM(x,i,j) = pElementM(x)[i+j*RowM(x)]

 SetMajorForm(x,i) - Sets the MajorForm of the TMatrix x to the int i.  The value
                     of i must be either 0 or 1.  If the implementation allows
                     for only one type, then this can be defined to be blank.  
                     For this reason, it is important that the user be careful in
                     using this macro since it may not have an effect in all 
                     implementations.  It is always permissible to assign the 
                     value of an existing TMatrix, as in 

                                   SetMajorForm(x,MajorForm(y));

                     but in all other cases, it is important to check, via a call
                     to MajorForm(), that the MajorForm has actually been set.
  

 The precision (float or double) is controlled by the define PRECISION 
 contained in the file prcsn.h.


 PERMUTATION MATRICES
 For 0 <= i,j <= m-1, let (i,j) denote the transposition which interchanges i 
 and j leaves the other elements fixed.  Let P(i,j) denote the m x m matrix
 obtained from the m x m identitiy matrix by interchanging the ith and jth rows,
 which for the identity matrix is equivalent to interchanging the ith and jth 
 columns.  If p is a permutation of {0,...,m-1} and is equal to the product of 
 transpositions

                        (i1,j1)*(i2,j2)*...*(iq,jq)
 
 then the permutation matrix associated with the permutation p is

                     P = P(i1,j1)*P(i2,j2)*...*P(iq,jq)

 Note that our convention is that 
  
                     (i1,j1)*(i2,j2)(k) = (i1,j1)((i2,j2)(k))

 Thus (1,2)(2,3) is the permutation that sends 1 to 2, 2 to 3, and 3 to 1.  Note 
 that multiplication on the left by a permutation matrix P associated with the
 permutation p permutes the rows by p.  Multiplication on the right permutes the
 columns by the inverse of p.

 A TPermutation is an integer array together with the length of the array and 
 the number of array elements actually used.  A TPermutation implementation can 
 contain additional information.  A instance of TPermutation must be created
 with CreatePermutation() and freed with FreePermutation().  The following 
 macros must be defined.

 DimP(x)       - x      : TPermutation
                 Returns: int containing the dimension.

 UseP(x)       - x      : TPermutation
                 Returns: L-value int containing the number of array elements
                          used.  This macro can also be used to set this number.
                          It must be the case that 0 <= UseP(x) <= DimP(x).

 ElementP(x,i) - x      : TPermutation
                 i      : int
                 Returns: L-value int containing the ith element.  The index i
                          is zero-based.  It must be case that  i < UseP(x) and

                                    i <= ElementP(x,i) < DimP(x).
                                    
 pElementP(x)  - x      : TPermutation
                 Returns: pointer to 0th element of the array storing the 
                          permutation.

 The representation as a product of transpositions used by TPermutation
     
           (0,ElementP(x,0))*...*(UseP(x)-1,ElementP(x,UseP(x)-1)

 In order to have a unique representation, we require that

           UseP(x) - 1 < ElementP(x,UseP(x) - 1)

*******************************************************************************/

/*******************************************************************************
 Some thoughts on vector and matrix implementation:

   1)  Because vectors are one dimensional and matrices are two dimensional,
       they should have different implementations for reasons of efficiency.

   2)  Can one get efficiency from more general n-dimensional matrix
       representations?

   3)  Should the types be encoded as a pointer to a structure or as a pointer
       to a float or double with the dimension and other infomation hidden.

   4)  There must be an efficient Element() operator.  If (1) is followed, there
       must be efficient ElementV() and ElementM() operators.  These operators
       must be able to return L-values and so probably need to implemented as
       macros.  This has the disadvantage that ElementM() will have side effects
       that must be avoided.

   5)  There must be an efficient Dim() operator.  If (1) is followed, there
       should be efficient DimV(), RowM(), ColM() operators, probably
       implemented as macros.

   6)  Should there be a flag to represent special matrices.  For instance,
       diagonal, upper triangular, lower triagular, and symmetric.  If the
       decision is made to include such a flag, then a decision must be made
       on the storage of special matrices.  In particular, should special
       matrices use a compressed storage, or should they use the general
       storage technique.  If they use the general storage technique, should
       the full matrix be stored, or should the redundant elements be left
       undefined.  This has implications for the ElementM() operator.

   7)  Should a decision be made to always encode matrices as column major, or
       should there be a flag to determine whether the matrix is encoded as
       column major or row major.  This gives added flexibility, but adds an
       extra cost to all matrix functions.  For complex functions which already
       check size, this cost is small, but for functions such as ElementM(),
       the cost may not be so small.  One solution would to be to add the
       operators ElementM_R() and ElementM_C(), which would retrieve the orginal
       efficiency in ElementM(), but would put a burden on the user to ensure
       that the right operator was called.

    8) Many routines will have the form Y = fnct(X, Z1, Z2, ...), where X and Y
       are pointers to the same type.  The characteristics (usually size) of X
       depends on the Z's.  For this reason, X is allowed to be a null pointer. 
       This allows the routine to create and return a pointer with the proper
       characteristics.  On the other hand, errors occur in the routine, then
       a null pointer is returned.  There is a potential for memory leaks.  The
       following syntax must be avoided: X = fnct(X, Z1, Z2, ... ).  If X is null
       or no errors occur, then no harm will result, but if X is not null and an 
       error occurs then a memory leak will exist.  This type of construct must 
       be avoided. Similarly, the following construct must be avoided:
                    
                             X = fnct2(fnct1(...), ...)

       If fnct1() returns a non-null pointer but fnct2 exits because of an error,
       then a memory leak will exist.  The proper construct in this case is
                             fnct1(X = fnct1(...), ...)

********************************************************************************/

#ifndef __MATRIX__
#define __MATRIX__

#include "prcsn.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

/******************************************************************************/
/********************************* Data Types *********************************/
/******************************************************************************/
/*
  The default is TZ_COLUMN_MAJOR.  To use one of the other type, it must be passed
  via the compiler -- generally -d TYPE
*/
#if !defined(STANDARD_COLUMN_MAJOR) && !defined(STANDARD_ROW_MAJOR)  && !defined(STRUCTURED_COLUMN_MAJOR) && !defined(STRUCTURED_ROW_MAJOR) && !defined(STRUCTURED_MAJOR_FORM) \
  && !defined(LEGACY_ROW_MAJOR) && !defined(TZ_COLUMN_MAJOR)  && !defined(CHECK_MACRO_CALLS)

#define TZ_COLUMN_MAJOR

#endif

#define COLUMN_MAJOR   1
#define ROW_MAJOR      0

typedef int *TPermutation;
#define UseP(y)          (((int*)(y))[-1])  
#define DimP(y)          (((int*)(y))[-2])
#define pElementP(y)     (y)
#define ElementP(y,i)    ((y)[(i)])

/*----------------------------------------------------------------------------*/
#if defined CHECK_MACRO_CALLS
typedef PRECISION  *TVector;
typedef PRECISION **TMatrix;
typedef int *TPermutation;

int DimV(TVector x);
PRECISION ElementV(TVector x, int i);
PRECISION* pElementV(TVector x);

int RowM(TMatrix x);
int ColM(TMatrix x);
PRECISION ElementM(TMatrix x, int i, int j);
PRECISION* pElementM(TMatrix x);

int DimP(TPermutation x);
int UseP(TPermutation x);
int ElementP(TPermutation y, int i);
int* pElementP(TPermutation y);
#endif
/*----------------------------------------------------------------------------*/
#if (defined(STANDARD_COLUMN_MAJOR) || defined(STANDARD_ROW_MAJOR))
// Data types
typedef struct
{
  int dim;
  PRECISION x[1];
} TVectorStructure;
typedef TVectorStructure *TVector;

typedef struct
{
  int row;
  int col;
  PRECISION x[1];
} TMatrixStructure;
typedef TMatrixStructure* TMatrix;

// Element access macros
#define DimV(y)          ((y)->dim)
#define pElementV(y)     ((y)->x)
#define ElementV(y,i)    ((y)->x[(i)])

#define RowM(y)          ((y)->row)
#define ColM(y)          ((y)->col)  
#define pElementM(y)     ((y)->x)
#if defined STANDARD_COLUMN_MAJOR
#define ElementM(y,i,j)  ((y)->x[(i)+(j)*((y)->row)])
#else
#define ElementM(y,i,j)  ((y)->x[(i)*((y)->col)+(j)])
#endif

// Major form macros
#define SetMajorForm(x,i)
#if defined STANDARD_COLUMN_MAJOR
#define MajorForm(x)       COLUMN_MAJOR  
#else
#define MajorForm(x)       ROW_MAJOR
#endif

#endif
/*----------------------------------------------------------------------------*/
#if (defined(STRUCTURED_COLUMN_MAJOR) || defined(STRUCTURED_ROW_MAJOR) || defined(STRUCTURED_MAJOR_FORM))
// Data types
typedef struct
{
  int dim;
  PRECISION *x;
} TVectorStructure;
typedef TVectorStructure *TVector;

typedef struct
{
  int row;
  int col;
#if (defined(STRUCTURED_MAJOR_FORM))
  int major;
#endif
  PRECISION *x;
} TMatrixStructure;
typedef TMatrixStructure* TMatrix;

// Element access macros
#define DimV(y)          ((y)->dim)
#define pElementV(y)     ((y)->x)
#define ElementV(y,i)    ((y)->x[(i)])

#define RowM(y)          ((y)->row)
#define ColM(y)          ((y)->col)  
#define pElementM(y)     ((y)->x)
#if defined STRUCTURED_COLUMN_MAJOR
#define ElementM(y,i,j)  ((y)->x[(i)+(j)*((y)->row)])
#elif defined STRUCTURED_ROW_MAJOR
#define ElementM(y,i,j)  ((y)->x[(i)*((y)->col)+(j)])
#elif defined STRUCTURED_MAJOR_FORM
#define ElementM(y,i,j)  ((y)->x[(y)->major ? (i)+(j)*((y)->row) : (i)*((y)->col)+(j)])
#endif

// Major form macros
#if defined STRUCTURED_COLUMN_MAJOR
#define MajorForm(x)       COLUMN_MAJOR
#define SetMajorForm(x,i)
#elif defined STRUCTURED_ROW_MAJOR
#define MajorForm(x)       ROW_MAJOR
#define SetMajorForm(x,i)
#elif defined STRUCTURED_MAJOR_FORM
#define SetMajorForm(x,i)  ((x)->major=(i))
#define MajorForm(x)       ((x)->major)
#endif

#endif
/*----------------------------------------------------------------------------*/
#if defined LEGACY_ROW_MAJOR 
// Data types 
typedef PRECISION  *TVector;
typedef PRECISION **TMatrix;

// Element access macros
#define DimV(y)          (((int*)(y))[-1])
#define pElementV(y)     (y)
#define ElementV(y,i)    ((y)[(i)])

#define RowM(y)          (((int*)(y))[-2])
#define ColM(y)          (((int*)(y))[-1])
#define pElementM(y)     ((y)[0])
#define ElementM(y,i,j)  ((y)[(i)][(j)])

// Legacy element access
#define V_DIM(x) (((int*)(x))[-1])
#define M_ROW(x) (((int*)(x))[-2])
#define M_COL(x) (((int*)(x))[-1])
#define P_USE(x) UseP(x)
#define P_DIM(x) DimP(x)

// Major form macros
#define SetMajorForm(x,i)
#define MajorForm(x)       ROW_MAJOR

#endif
/*----------------------------------------------------------------------------*/
#if defined TZ_COLUMN_MAJOR

/* In prcsn.h, PRECISION must be defined to be double */
#if PRECISION_SIZE != 8
  precision size must be 8
#endif

// Defines
#include "tzmatlab.h"

// Data types
typedef TSdvector* TVector;
typedef TSdmatrix* TMatrix;

// Element access macros
#define DimV(y)          ((y)->n)
#define pElementV(y)     ((y)->v)
#define ElementV(y,i)    ((y)->v[(i)])

#define RowM(y)          ((y)->nrows)
#define ColM(y)          ((y)->ncols)  
#define pElementM(y)     ((y)->M)
#define ElementM(y,i,j)  ((y)->M[(i)+(j)*((y)->nrows)])

// Major form macros
#define SetMajorForm(x,i)
#define MajorForm(x)       COLUMN_MAJOR  

#endif
/*----------------------------------------------------------------------------*/
/******************************************************************************/
/******************************************************************************/

/* Allocation/Deallocation Routines */
TVector CreateVector(int m);
TMatrix CreateMatrix(int m, int n);
void FreeVector(TVector x);
void FreeMatrix(TMatrix X);

/* Initialization Routines */
TVector InitializeVector(TVector x, PRECISION c);
TMatrix InitializeMatrix(TMatrix X, PRECISION c);

/* Assignment Routines */
TVector EquateVector(TVector x, TVector y);
TMatrix EquateMatrix(TMatrix X, TMatrix Y);
TMatrix Transpose(TMatrix X, TMatrix Y);
TMatrix IdentityMatrix(TMatrix X, int m);
TMatrix ZeroMatrix(TMatrix X, int m, int n);
TMatrix DiagonalMatrix(TMatrix X, TVector y);
TVector MinusV(TVector x, TVector y);
TMatrix MinusM(TMatrix X, TMatrix Y);
TVector AbsV(TVector x, TVector y);
TMatrix AbsM(TMatrix X, TMatrix Y);
TMatrix SubMatrix(TMatrix X, TMatrix Y, int brow, int bcol, int rows, int cols);
TVector SubVector(TVector x, TVector y, int b, int d);
TMatrix InsertSubMatrix(TMatrix X, TMatrix Y, int brow_X, int bcol_X, int brow_Y, int bcol_Y, int rows, int cols);
TMatrix InsertColumnVector(TMatrix X, TVector y, int col);
TMatrix InsertRowVector(TMatrix X, TVector y, int row);
TVector ColumnVector(TVector x, TMatrix Y, int col);
TVector RowVector(TVector x, TMatrix Y, int row);
TMatrix ColumnMatrix(TMatrix X, TVector y);
TMatrix RowMatrix(TMatrix X, TVector y);

//=== Addition Routines ===
TVector AddVV(TVector x, TVector y, TVector z);
TMatrix AddMM(TMatrix X, TMatrix Y, TMatrix Z);
TVector SubtractVV(TVector x, TVector y, TVector z);
TMatrix SubtractMM(TMatrix X, TMatrix Y, TMatrix Z);

//=== Multiplication Routines ===
TVector ProductSV(TVector x, PRECISION s, TVector y);
#define ProductVS(x,y,s) ProductSV(x,s,y)
TMatrix ProductSM(TMatrix X, PRECISION s, TMatrix Y);
#define ProductMS(X,Y,s) ProductSM(X,s,Y)
TVector ProductVM(TVector x, TVector y, TMatrix Z);
TVector ProductMV(TVector x, TMatrix Y, TVector z);
TMatrix ProductMM(TMatrix X, TMatrix Y, TMatrix Z);
TMatrix ProductMU(TMatrix X, TMatrix Y, TMatrix Z);
TMatrix ProductML(TMatrix X, TMatrix Y, TMatrix Z);
TMatrix ProductUM(TMatrix X, TMatrix Y, TMatrix Z);
TMatrix ProductLM(TMatrix X, TMatrix Y, TMatrix Z);
TVector ProductInverseVM(TVector x, TVector y, TMatrix Z);
TMatrix ProductInverseMM(TMatrix X, TMatrix Y, TMatrix Z);
TVector ProductInverseVU(TVector x, TVector y, TMatrix Z);
TMatrix ProductInverseMU(TMatrix X, TMatrix Y, TMatrix Z);
TVector ProductInverseVL(TVector x, TVector y, TMatrix Z);
TMatrix ProductInverseML(TMatrix X, TMatrix Y, TMatrix Z);
TVector InverseProductMV(TVector x, TMatrix Y, TVector z);
TMatrix InverseProductMM(TMatrix X, TMatrix Y, TMatrix Z);
TVector InverseProductUV(TVector x, TMatrix Y, TVector z);
TMatrix InverseProductUM(TMatrix X, TMatrix Y, TMatrix Z);
TVector InverseProductLV(TVector x, TMatrix Y, TVector z);
TMatrix InverseProductLM(TMatrix X, TMatrix Y, TMatrix Z);
TMatrix TransposeProductMM(TMatrix X, TMatrix Y, TMatrix Z);
#define TransposeProductMV(x,Y,z) ProductVM(x,z,Y)
TMatrix ProductTransposeMM(TMatrix X, TMatrix Y, TMatrix Z);
#define ProductTransposeVM(x,y,Z) ProductMV(x,Z,y)

//=== Updating ===
TVector LinearCombinationV(TVector x, PRECISION a, TVector y, PRECISION b, TVector z);
TMatrix LinearCombinationM(TMatrix X, PRECISION a, TMatrix Y, PRECISION b, TMatrix Z);
#define UpdateV(a,x,b,y) LinearCombinationV(x,a,x,b,y)
#define UpdateM(a,X,b,Y) LinearCombinationM(X,a,X,b,Y)
TMatrix UpdateProductMM(PRECISION a, TMatrix X, PRECISION b, TMatrix Y, TMatrix Z);
TVector UpdateProductMV(PRECISION a, TVector x, PRECISION b, TMatrix Y, TVector z);
TVector UpdateProductVM(PRECISION a, TVector x, PRECISION b, TVector y, TMatrix Z);

/* Matrix Inverse Routines */
TMatrix Inverse_LU(TMatrix X, TMatrix Y);
TMatrix Inverse_SVD(TMatrix X, TMatrix Y);
TMatrix Inverse_Cholesky(TMatrix X, TMatrix Y);
TMatrix Inverse_UT(TMatrix X, TMatrix Y);
TMatrix Inverse_LT(TMatrix X, TMatrix Y);
TMatrix GeneralizedInverse(TMatrix X, TMatrix Y);

/* Matrix Decompositions */
int SVD(TMatrix U, TVector d, TMatrix V, TMatrix A);
int QR(TMatrix Q, TMatrix R, TMatrix X);

int GeneralizedSchur_Real(TMatrix S, TMatrix T, TMatrix Q, TMatrix Z, TMatrix A, TMatrix B, 
			  TVector alpha_r, TVector alpha_i, TVector beta); ;
int ReorderGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
				 TMatrix Z, int *select, TVector alpha_r, TVector alpha_i, TVector beta);
int SortGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
			      TMatrix Z, TVector alpha_r, TVector alpha_i, TVector beta, int descend);

TMatrix CholeskyUT(TMatrix T, TMatrix X);
TMatrix CholeskyLT(TMatrix T, TMatrix X);

int LU(TPermutation P, TMatrix X, TMatrix A);
TVector LU_SolveCol(TVector x, TVector y, TMatrix LU, TPermutation P);
TVector LU_SolveRow(TVector x, TVector y, TMatrix LU, TPermutation P);

int Eigenvalues(TVector Re, TVector Im, TMatrix X);
int Eigen(TVector ReVal, TVector ImVal, TMatrix ReVec, TMatrix ImVec, TMatrix X);

TMatrix MatrixSquareRoot(TMatrix X, TMatrix Y);

/* Complex Routines */
int ComplexProductMM(TMatrix ReX, TMatrix ImX, TMatrix ReY, TMatrix ImY, TMatrix ReZ, TMatrix ImZ);

/* Miscellaneous Routines */
PRECISION Norm(TVector x);
PRECISION MatrixNormEuclidean(TMatrix X);
PRECISION MatrixNorm(TMatrix X);
PRECISION DotProduct(TVector x, TVector y);
PRECISION InnerProduct(TVector x, TVector y, TMatrix S);
TMatrix OuterProduct(TMatrix X, TVector y, TVector z);
PRECISION Trace(TMatrix X);
TVector CrossProduct_LU(TVector x, TMatrix Y);
PRECISION Determinant_LU(TMatrix X);
PRECISION LogAbsDeterminant_LU(TMatrix X);
int Rank_SVD(TMatrix X);
TMatrix NullSpace(TMatrix Y);
TMatrix ForceSymmetric(TMatrix X);

/* Kronecker Routines */
TVector Vec(TVector x, TMatrix Y);
TMatrix ReshapeV(TMatrix X, TVector y, int rows, int cols);
TMatrix KroneckerProduct(TMatrix X, TMatrix Y, TMatrix Z);

/* Input - Output Routines */
int dw_PrintVector(FILE *f, TVector x, const char *format);
int dw_PrintMatrix(FILE *f, TMatrix X, const char *format);
int dw_ReadMatrix(FILE *f, TMatrix X);
int dw_ReadVector(FILE *f, TVector x);
int OutVectorFloat(FILE *f, TVector x);
int OutMatrixFloat(FILE *f, TMatrix X);
int OutVectorDouble(FILE *f, TVector x);
int OutMatrixDouble(FILE *f, TMatrix X);
TVector InVector(FILE *f, TVector x);
TMatrix InMatrix(FILE *f, TMatrix X);

/* Permutations */
TPermutation CreatePermutation(int m);
void FreePermutation(TPermutation X);
TPermutation InitializePermutation(TPermutation X, int *p, int m);
int* ExtractPermutation(int *p, TPermutation X);
TPermutation TranspositionPermutation(TPermutation X, int i, int j, int m);
TPermutation EquatePermutation(TPermutation X, TPermutation Y);
TMatrix PermutationMatrix(TMatrix X, TPermutation Y);
TMatrix ProductPM(TMatrix X, TPermutation Y, TMatrix Z);
TMatrix ProductMP(TMatrix X, TMatrix Y, TPermutation Z);
TVector ProductPV(TVector x, TPermutation Y, TVector z);
TVector ProductVP(TVector x, TVector y, TPermutation Z);
TMatrix TransposeProductPM(TMatrix X, TPermutation Y, TMatrix Z);
#define TransposeProductPV(x,Y,z) ProductVP(x,z,Y)
TMatrix ProductTransposeMP(TMatrix X, TMatrix Y, TPermutation Z);
#define ProductTransposeVP(x,y,Z) ProductPV(x,Z,y)
void PrintPermutation(FILE *f, TPermutation);

/****** Old Style Syntax ******
//====== Error Routines ======
#define MatrixError(err)             Error(err)
#define ClearMatrixError()           ClearError()
#define GetMatrixError()             GetError()
#define SetMatrixErrorVerbose(err)   SetVerboseErrors(err)
#define SetMatrixErrorTerminate(err) SetTerminalErrors(err)

//#define Inverse(X,Y)                           Inverse_LU(X,Y)
//#define TransposeProduct(X,Y,Z)                TransposeProductMM(X,Y,Z)
//#define ProductTranspose(X,Y,Z)                ProductTransposeMM(X,Y,Z)
//#define InverseProduct(X,Y,Z)                  InverseProductMM(X,Y,Z)
//#define ProductInverse(X,Y,Z)                  ProductInverseMM(X,Y,Z)
//#define VectorProductInverse(x,y,Z)            ProductInverseVM(x,y,Z)
//#define ERROR(i)                               MatrixError(i)

// int Cholesky_U(TMatrix T, TMatrix X)                                           X = T'* T  (T upper triangular)
// int Cholesky_L(TMatrix T, TMatrix X)                                           X = T * T' (T lower triangular)
// int SingularValueDecomposition(TMatrix U, TVector d, TMatrix V, TMatrix A)     A = U * Diagonal(d) * V'
// int* QR_RPivot(TMatrix R, TMatrix X);
// int* QR_QRPivot(TMatrix Q, TMatrix R, TMatrix X);
// InitializePermutationFromIntArray(TPermutation X, int *p, int m);
**/

#ifdef __cplusplus
}
#endif

#endif
