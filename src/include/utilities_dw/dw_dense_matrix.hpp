#ifndef _DW_DENSE_MATRIX_
#define _DW_DENSE_MATRIX_

// See DenseMatrixDocumentation.pdf or DenseMatrixDocumentation.tex

//== Solution Methods ==========================================================
#define SOLVE_LU                 1
#define SOLVE_SVD                2
#define SOLVE_QR                 3
#define SOLVE_CHOLESKY           4
#define SOLVE_LOWER_TRIANGULAR   5
#define SOLVE_UPPER_TRIANGULAR   6
#define SOLVE_DIAGONAL           7

//== Cholesky Decomposition defines =============================================
#define CHOLESKY_UPPER_TRIANGULAR 0
#define CHOLESKY_LOWER_TRIANGULAR 1

//== Refineing the Solution and Estimating its Error
#define SOLVE_REFINE_GENERAL 1
#define SOLVE_REFINE_SPD_UPPER_TRIANGULAR 2
#define SOLVE_REFINE_SPD_LOWER_TRIANGULAR 3
#define SOLVE_REFINE_SI_LOWER_TRIANGULAR 4
#define SOLVE_REFINE_SI_UPPER_TRIANGULAR 5
#define SOLVE_REFINE_LOWER_TRIANGULAR 6
#define SOLVE_REFINE_UPPER_TRIANGULAR 7

#include <vector>
#include <complex>
#include <new>
#include <iostream>
#include <fstream>
#include <cstring>

#include "dw_exception.hpp"

class TDenseVector;
class TDenseMatrix;
class TIndex;

class TPermutationMatrix;
class TLapackLU;

class TDenseVector
{
public:
  // Data
  double *vector;
  int dim;

  // Constructors and Destructors
  ~TDenseVector();
  TDenseVector(void);
  explicit TDenseVector(int d);
  TDenseVector(int d, double s);
  TDenseVector(const TDenseVector &v);

  // Memory Management
  void UniqueMemory(void);
  void UniqueMemory(int d);

  // Size
  int Dimension(void) const { return dim; };
  void Resize(int d);

  // Element access
  double operator()(int i) const;
  double & operator()(int i);  // Added by Hwu to allow direct assignment like A(i)=s; 
  double operator[](int i) const; // Added by Hwu
  double & operator[](int i);  // Added by Hwu to allow direct assignment like A[i]=s; 


  void SetElement(double s, int i); // Added by Hwu
  void SetSubVector(const std::vector<int> &locs, const TDenseVector &x); // Added by HWu, vector(locs) = x

  // Logical -- Added by HWu
  bool operator == (const TDenseVector &v); 

  // Copy content -- Added by HWu
  TDenseVector &CopyContent(const TDenseVector &v); 

  // Copy content -- Swith order
  TDenseVector SwitchOrder(const std::vector<int> &order); 

  // Assignment
  TDenseVector& operator=(const TDenseVector &v);
  TDenseVector& operator+=(const TDenseVector &v);
  TDenseVector& operator-=(const TDenseVector &v);
  TDenseVector& operator*=(const TDenseMatrix &M);
  TDenseVector& operator*=(double);

  // extract subvector from vector
  TDenseVector operator()(int b, int e) const;
  TDenseVector operator()(const TIndex &idx) const;
  TDenseVector SubVector(int b, int e) const;
  TDenseVector SubVector(const TIndex &idx) const;
  TDenseVector& SubVector(const TDenseVector &v, int b, int e);
  TDenseVector& SubVector(const TDenseVector &v, TIndex &idx);

  // extract subvector from matrix
  TDenseVector& ColumnVector(const TDenseMatrix &M, int c);
  TDenseVector& ColumnVector(const TDenseMatrix &M, int c, int br, int er);
  TDenseVector& ColumnVector(const TDenseMatrix &M, int c, const TIndex &idx);
  TDenseVector& RowVector(const TDenseMatrix &M, int r);
  TDenseVector& RowVector(const TDenseMatrix &M, int r, int bc, int ec);
  TDenseVector& RowVector(const TDenseMatrix &M, int r, const TIndex &idx);

  // insert subvector into vector
  TDenseVector& Insert(int b, const TDenseVector &v);
  TDenseVector& Insert(const TIndex &idx, const TDenseVector &v);
  TDenseVector& Insert(int b, const TDenseVector &v, int b_v, int e_v);
  TDenseVector& Insert(const TIndex &idx, const TDenseVector &v, int b_v, int e_v);
  TDenseVector& Insert(int b, const TDenseVector &v, const TIndex &idx_v);
  TDenseVector& Insert(const TIndex &idx, const TDenseVector &v, const TIndex &idx_v);

  // insert subvector from matrix into vector
  TDenseVector& InsertColumnVector(int b, const TDenseMatrix &M, int c);
  TDenseVector& InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c);
  TDenseVector& InsertColumnVector(int b, const TDenseMatrix &M, int c, int br, int er);
  TDenseVector& InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c, int br, int er);
  TDenseVector& InsertColumnVector(int b, const TDenseMatrix &M, int c, const TIndex &idx_r);
  TDenseVector& InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c, const TIndex &idx_r);
  TDenseVector& InsertRowVector(int b, const TDenseMatrix &M, int r);
  TDenseVector& InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r);
  TDenseVector& InsertRowVector(int b, const TDenseMatrix &M, int r, int bc, int ec);
  TDenseVector& InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r, int bc, int ec);
  TDenseVector& InsertRowVector(int b, const TDenseMatrix &M, int r, const TIndex &idx_c);
  TDenseVector& InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r, const TIndex &idx_c);
 
  // convert subvectors to row or column matrices
  TDenseMatrix ColumnMatrix(void) const;
  TDenseMatrix ColumnMatrix(int b, int n) const;
  TDenseMatrix ColumnMatrix(const TIndex &idx) const;
  TDenseMatrix RowMatrix(void) const;
  TDenseMatrix RowMatrix(int b, int n) const;
  TDenseMatrix RowMatrix(const TIndex &idx) const;

  // miscellaneous vector manipulation
  TDenseVector& Cat(double x);                                             // not yet implemented
  TDenseVector& Cat(const TDenseVector &v);                                // not yet implemented
  TDenseVector& Cat(const TDenseVector &v1, const TDenseVector &v2);
  TDenseVector& Cat(const std::vector<TDenseVector> &v_array);             // not yet implemented
  TDenseVector& Vec(const TDenseMatrix &M);

  // sorting routines
  TDenseVector& Sort(bool ascending=true);
  TDenseVector& Sort(const TDenseVector &v, bool ascending=true);
  
  // Added by HWu: Return a Subvector //
  TDenseVector SubVector(const std::vector<int> &index) const; 

  // Initialization
  TDenseVector& Initialize(double s);
  TDenseVector& Initialize(double s, int d);
  TDenseVector& Zeros(void);
  TDenseVector& Zeros(int d);
  TDenseVector& Ones(void);
  TDenseVector& Ones(int d);

  // Unary operators
  TDenseVector& Minus(void);
  TDenseVector& Minus(const TDenseVector &v);
  TDenseVector& Abs(void);
  TDenseVector& Abs(const TDenseVector &v);

  // Addition
  TDenseVector& Add(const TDenseVector &v1, const TDenseVector &v2);
  TDenseVector& Subtract(const TDenseVector &v1, const TDenseVector &v2);

  // Multiplication
  TDenseVector& Multiply(const TDenseMatrix &M, const TDenseVector &v);
  TDenseVector& Multiply(const TDenseVector &v, const TDenseMatrix &M);
  TDenseVector& Multiply(const TDenseVector &v, double s);
  TDenseVector& Multiply(double s, const TDenseVector &v);
  TDenseVector& Multiply(const TPermutationMatrix &P, const TDenseVector &v);
  TDenseVector& Multiply(const TDenseVector &v, const TPermutationMatrix &P);

  TDenseVector& TransposeMultiply(const TDenseMatrix &M, const TDenseVector &v);
  TDenseVector& MultiplyTranspose(const TDenseVector &v, const TDenseMatrix &M);
  TDenseVector& TransposeMultiply(const TPermutationMatrix &P, const TDenseVector &v);
  TDenseVector& MultiplyTranspose(const TDenseVector &v, const TPermutationMatrix &P);

  TDenseVector& InverseMultiply(const TDenseMatrix &M, const TDenseVector &v, int method=SOLVE_LU);
  TDenseVector& MultiplyInverse(const TDenseVector &v, const TDenseMatrix &M, int method=SOLVE_LU);
  TDenseVector& InverseMultiply(const TPermutationMatrix &P, const TDenseVector &v);
  TDenseVector& MultiplyInverse(const TDenseVector &v, const TPermutationMatrix &P);

  TDenseVector & DotMultiply(const TDenseVector &v1, const TDenseVector &v2); // Added by HWu

  TDenseVector& Kron(const TDenseVector &x, const TDenseVector &y);

  // Solving Linear Systems
  TDenseVector& LeftSolve(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_LU);
  TDenseVector& LeftSolve_Refine(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_REFINE_GENERAL); 
  TDenseVector& LeftSolve_Cholesky(const TDenseMatrix &A, const TDenseVector &b); 
  TDenseVector& RightSolve(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_LU);

  // Miscellaneous Routines
  double Norm(void) const;
  double Sum(void) const;

  // Random Vectors
  TDenseVector& RandomNormal(void);
  TDenseVector& RandomNormal(int d);
  TDenseVector& RandomUniform(void);
  TDenseVector& RandomUniform(int d);

  // IO
  TDenseVector& ReadBinary(std::fstream &file);
  void WriteBinary(std::fstream &file);
  void WriteBinary(std::fstream &file, int b, int e);

  // Unsafe Routiones
  TDenseVector(double *buffer, int d);
  void ShareMemory(double *buffer, int d);
};

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

class TDenseMatrix
{
public:
  // Data
  double *matrix;
  int rows;
  int cols;
  bool column_major;

  // Constructors and Destructors
  ~TDenseMatrix();
  TDenseMatrix(void);
  TDenseMatrix(int r, int c);
  TDenseMatrix(int r, int c, double s);
  TDenseMatrix(int r, int c, bool col_major);
  TDenseMatrix(int r, int c, bool col_major, double s);
  TDenseMatrix(const TDenseMatrix &M);
  explicit TDenseMatrix(const TPermutationMatrix &P);

  // Memory Management
  void UniqueMemory(void);
  void UniqueMemory(int r, int c, bool col_major);

  // Size and shape
  int NumberRows(void) const { return rows; };
  int NumberColumns(void) const { return cols; };
  bool IsColumnMajor(void) const { return column_major; };
  void Resize(int r, int c);
  TDenseMatrix& Reshape(int r, int c);
  void ForceColumnMajor(void);
  void ForceRowMajor(void);

  // Element access
  int Index(int r, int c) const;
  int Stride(void) const;
  int RowStride(void) const;
  int ColumnStride(void) const;
  double operator()(int r, int c) const;
  double &operator()(int r, int c);   

  void SetElement(double s, int r, int c); // This form is depreciated, use M(r,c)=s

  // Copy content, Added by HWu
  TDenseMatrix & CopyContent(const TDenseMatrix &M); 

  // Assignment
  TDenseMatrix& operator=(const TDenseMatrix &M);
  TDenseMatrix& operator=(const TPermutationMatrix &P);
  TDenseMatrix& operator+=(const TDenseMatrix &M);
  TDenseMatrix& operator-=(const TDenseMatrix &M);
  TDenseMatrix& operator*=(const TDenseMatrix &M);
  TDenseMatrix& operator*=(double);

  // Initialization
  TDenseMatrix& Initialize(double s);
  TDenseMatrix& Initialize(double s, int r, int c);
  TDenseMatrix& Zeros(void);
  TDenseMatrix& Zeros(int r, int c);
  TDenseMatrix& Ones(void);
  TDenseMatrix& Ones(int r, int c);
  TDenseMatrix& Identity(int d);
  TDenseMatrix& Identity(int r, int c);
  TDenseMatrix& DiagonalMatrix(double s);
  TDenseMatrix& DiagonalMatrix(double s, int r, int c);
  TDenseMatrix& DiagonalMatrix(const TDenseVector &v);
  TDenseMatrix& DiagonalMatrix(const TDenseVector &v, int r, int c);
  TDenseMatrix& BlockDiagonalMatrix(const TDenseMatrix &M, int n);
  TDenseMatrix  BlockDiagonalMatrix(int n);

  // extract subvector from matrix
  TDenseVector ColumnVector(int c) const;
  TDenseVector ColumnVector(int c, int br, int er) const;
  TDenseVector ColumnVector(int c, const TIndex &idx) const;
  TDenseVector RowVector(int r) const;
  TDenseVector RowVector(int r, int bc, int ec) const;
  TDenseVector RowVector(int r, const TIndex &idx) const;

  // extract submatrix from vector
  TDenseMatrix& ColumnMatrix(const TDenseVector &v);
  TDenseMatrix& ColumnMatrix(const TDenseVector &v, int b, int n);
  TDenseMatrix& ColumnMatrix(const TDenseVector &v, const TIndex &idx_v);
  TDenseMatrix& RowMatrix(const TDenseVector &v);
  TDenseMatrix& RowMatrix(const TDenseVector &v, int b, int n);
  TDenseMatrix& RowMatrix(const TDenseVector &v, const TIndex &idx_v);

  // extract submatrix from matrix
  TDenseMatrix operator()(int br, int er, int bc, int ec) const;
  TDenseMatrix operator()(const TIndex &idx_r, int bc, int ec) const;
  TDenseMatrix operator()(int br, int er, const TIndex &idx_c) const;
  TDenseMatrix operator()(const TIndex &idx_r, const TIndex &idx_c) const;
  TDenseMatrix SubMatrix(int br, int er, int bc, int ec) const;
  TDenseMatrix SubMatrix(const TIndex &idx_r, int bc, int ec) const;
  TDenseMatrix SubMatrix(int br, int er, const TIndex &idx_c) const;
  TDenseMatrix SubMatrix(const TIndex &idx_r, const TIndex &idx_c) const;
  TDenseMatrix& SubMatrix(const TDenseMatrix &M, int br, int er, int bc, int ec);
  TDenseMatrix& SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
  TDenseMatrix& SubMatrix(const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
  TDenseMatrix& SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);

  // insert submatrix into matrix
  TDenseMatrix& Insert(int r, int c, const TDenseMatrix &M);
  TDenseMatrix& Insert(int r, const TIndex idxc, const TDenseMatrix &M);
  TDenseMatrix& Insert(const TIndex idxr, int c, const TDenseMatrix &M);
  TDenseMatrix& Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M);
  TDenseMatrix& Insert(int r, int c, const TDenseMatrix &M, int br, int er, int bc, int ec);
  TDenseMatrix& Insert(int r, const TIndex idxc, const TDenseMatrix &M, int br, int er, int bc, int ec);
  TDenseMatrix& Insert(const TIndex idxr, int c, const TDenseMatrix &M, int br, int er, int bc, int ec);
  TDenseMatrix& Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, int br, int er, int bc, int ec);
  TDenseMatrix& Insert(int r, int c, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
  TDenseMatrix& Insert(int r, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
  TDenseMatrix& Insert(const TIndex idxr, int c, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
  TDenseMatrix& Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
  TDenseMatrix& Insert(int r, int c, const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
  TDenseMatrix& Insert(int r, const TIndex idxc, const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
  TDenseMatrix& Insert(const TIndex idxr, int c, const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
  TDenseMatrix& Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
  TDenseMatrix& Insert(int r, int c, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);
  TDenseMatrix& Insert(int r, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);
  TDenseMatrix& Insert(const TIndex idxc, int c, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);
  TDenseMatrix& Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);

  // insert subvector into matrix
  TDenseMatrix& InsertRowMatrix(int r, int c, const TDenseVector &v);
  TDenseMatrix& InsertRowMatrix(int r, int c, const TDenseVector &v, int b, int e);
  TDenseMatrix& InsertRowMatrix(int r, int c, const TDenseVector &v, const TIndex &idx_v);
  TDenseMatrix& InsertRowMatrix(int r, TIndex idx, const TDenseVector &v);
  TDenseMatrix& InsertRowMatrix(int r, TIndex idx, const TDenseVector &v, int b, int e);
  TDenseMatrix& InsertRowMatrix(int r, TIndex idx, const TDenseVector &v, const TIndex &idx_v);
  TDenseMatrix& InsertColumnMatrix(int r, int c, const TDenseVector &v);
  TDenseMatrix& InsertColumnMatrix(int r, int c, const TDenseVector &v, int b, int e);
  TDenseMatrix& InsertColumnMatrix(int r, int c, const TDenseVector &v, const TIndex &idx_v);
  TDenseMatrix& InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v);
  TDenseMatrix& InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v, int b, int e);
  TDenseMatrix& InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v, const TIndex &idx_v);

  // miscellaneous matrix manipulation
  TDenseMatrix& HCat(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& HCat(const std::vector<TDenseMatrix> &M_array);               // not yet implemented
  TDenseMatrix& VCat(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& VCat(const std::vector<TDenseMatrix> &M_array);               // not yet implemented
  TDenseVector Vec(void);

  // Unary operators
  TDenseMatrix& Minus(void);
  TDenseMatrix& Minus(const TDenseMatrix &M);
  TDenseMatrix& Abs(void);
  TDenseMatrix& Abs(const TDenseMatrix &M);
  TDenseMatrix& Transpose(void);
  TDenseMatrix& Transpose(const TDenseMatrix &M);

  // Inverse
  TDenseMatrix& Inverse(int method=SOLVE_LU);
  TDenseMatrix& Inverse(const TDenseMatrix &M, int method=SOLVE_LU);
  TDenseMatrix& GeneralizedInverse(void);
  TDenseMatrix& GeneralizedInverse(const TDenseMatrix &M);

  // Solving Linear Systems
  TDenseMatrix& LeftSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_LU);
  TDenseMatrix& LeftSolve_Refine(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_REFINE_GENERAL);
  TDenseMatrix& LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseMatrix &B); 
  TDenseMatrix& RightSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_LU);

  // Addition
  TDenseMatrix& Add(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& Subtract(const TDenseMatrix &M1, const TDenseMatrix &M2);

  // Multiplication
  TDenseMatrix& Multiply(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& Multiply(const TDenseMatrix &M, const TPermutationMatrix &P);
  TDenseMatrix& Multiply(const TPermutationMatrix &P, const TDenseMatrix &M);
  TDenseMatrix& Multiply(const TDenseMatrix &M, double s);
  TDenseMatrix& Multiply(double s, const TDenseMatrix &M);
  TDenseMatrix& Multiply(const TDenseVector &v1, const TDenseVector &v2);	// Added by HWu 

  TDenseMatrix& MultiplyTranspose(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& TransposeMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2);
  TDenseMatrix& MultiplyTranspose(const TDenseMatrix &M, const TPermutationMatrix &P);
  TDenseMatrix& TransposeMultiply(const TDenseMatrix &M, const TPermutationMatrix &P);
  TDenseMatrix& MultiplyTranspose(const TPermutationMatrix &P, const TDenseMatrix &M);
  TDenseMatrix& TransposeMultiply(const TPermutationMatrix &P, const TDenseMatrix &M);
  TDenseMatrix& TransposeMultiply(const TDenseMatrix &M, double s);
  TDenseMatrix& MultiplyTranspose(double s, const TDenseMatrix &M);

  TDenseMatrix& MultiplyInverse(const TDenseMatrix &M1, const TDenseMatrix &M2, int method=SOLVE_LU);
  TDenseMatrix& InverseMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2, int method=SOLVE_LU);
  TDenseMatrix& MultiplyInverse(const TDenseMatrix &M, const TPermutationMatrix &P);
  TDenseMatrix& InverseMultiply(const TDenseMatrix &M, const TPermutationMatrix &P, int method=SOLVE_LU);
  TDenseMatrix& MultiplyInverse(const TPermutationMatrix &P, const TDenseMatrix &M, int method=SOLVE_LU);
  TDenseMatrix& InverseMultiply(const TPermutationMatrix &P, const TDenseMatrix &M);
  TDenseMatrix& InverseMultiply(const TDenseMatrix &M, double s, int method=SOLVE_LU);
  TDenseMatrix& MultiplyInverse(double s, const TDenseMatrix &M, int method=SOLVE_LU);

  TDenseMatrix& Kron(const TDenseMatrix &X, const TDenseMatrix &Y);

  // Miscellaneous Routines
  double Norm(void) const;
  double MatrixNorm(void) const;
  TDenseMatrix& OuterProduct(const TDenseVector &v1, const TDenseVector &v2);
  TDenseMatrix& OuterProduct(const TDenseVector &v);
  
  //====== Added by Hwu ======
  double Determinant(void) const; 
  double Determinant_Cholesky(const TDenseMatrix &T) const; // T is the Cholesky decomposition of this 
  double LogAbsDeterminant(void) const; 
  double LogAbsDeterminant_Cholesky(const TDenseMatrix &T) const; // T is the Cholesky decomposition of this
  int Rank(void) const; 
	double RCond() const; 
  //==========================

  // sorting routines
  TDenseMatrix& SortRows(int c, bool ascending=true);
  TDenseMatrix& SortRows(const TDenseMatrix &M, int c, bool ascending=true);
  TDenseMatrix& SortColumns(int r, bool ascending=true);
  TDenseMatrix& SortColumns(const TDenseMatrix &M, int r, bool ascending=true);

  // Random Matrices
  TDenseMatrix& RandomNormal(void);
  TDenseMatrix& RandomNormal(int r, int c);
  TDenseMatrix& RandomUniform(void);
  TDenseMatrix& RandomUniform(int r, int c);

  // IO
  TDenseMatrix& ReadBinary(std::fstream &file);
  void WriteBinary(std::fstream &file);
  void WriteBinary(std::fstream &file, int brow, int erow, int bcol, int ecol);

  // Unsafe Routines
  TDenseMatrix(double *buffer, int r, int c, bool col_major);
  void ShareMemory(double *buffer, int r, int c, bool col_major);

  // Added by HWu
  bool IsZeroMatrix() const; 
};


/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
class TIndex
{
public:
  int *index;
  int size, allocated;

  void allocate(int n);
  void resize(int n);

public:
  // destructor
  ~TIndex();

  // constructors
  TIndex(void);
  explicit TIndex(int idx);
  TIndex(int b_idx, int e_idx);
  TIndex(int b_idx, int inc, int e_idx);
  TIndex(const TIndex &idx);

  int Size(void) const;
  int operator[](int i) const;
  int& operator[](int i);  

  TIndex& Clear(void);

  TIndex& operator=(int idx);
  TIndex& operator=(const TIndex &idx);

  TIndex& operator()(int idx);
  TIndex& operator()(int b_idx, int e_idx);
  TIndex& operator()(int b_idx, int inc, int e_idx);
  TIndex& operator()(const TIndex &idx);

  TIndex& operator+=(int idx);
  TIndex& operator+=(const TIndex &idx);

  TIndex& operator,(int idx);
  TIndex& operator,(const TIndex &idx);

  // Added by HWu to merge/subtract a TIndex object and remove redundant elements
  TIndex & UniqMerge(const TIndex &right); 
  TIndex & UniqSubtract(const TIndex &right); 

  int Max(void);
  int Min(void);
};


/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

// The TPermutationMatrix class encodes permutation matrices.  This encoding is
// best understood through transposition matrices M(i,j).  The matrix M(i,j) is 
// the matrix which interchanges the ith and jth columns of the n x n identity 
// matrix, or equivalently, the ith and jth rows of the n x n identity matrix.
//
// Permutations are also bijections from {0,...,n-1} to {0,...,n-1}.  Thus if P 
// is a permutation, then 0 <= P(i) < n for 0 <= i < n and P(i) != P(j) for 
// 0 <= i < j < n.  If P and Q are both permutations, then (P*Q)(i)=P(Q(i)).
//
// Permutation matrices are encoded as an integer array of length n.  If p is an 
// array of integers with 0 <= p[i] < n for 0 <= i < n and p[i] != p[j] for 
// 0 <= i < j < n, then p defines a bijection which maps i to p[i].  Permutation
// matrices arise from pivoting in the LU decomposition, though the permutation 
// representation used by Lapack is different from the one used here.
class TPermutationMatrix
{
public:
  int *permutation;
  int dim;

  // Constructors and Destructors
  ~TPermutationMatrix();
  TPermutationMatrix(void);
  explicit TPermutationMatrix(int d);
  TPermutationMatrix(const TPermutationMatrix &P);

  // Memory Management
  void UniqueMemory(void);
  void Resize(int d);

  // Assignment
  TPermutationMatrix& operator=(const TPermutationMatrix &P);

  // Element Access
  int operator()(int i) const;

  // Conversion to DenseMatrix
  TDenseMatrix DenseMatrix(void);

  // Unary Operators
  TPermutationMatrix& Transpose(void);
  TPermutationMatrix& Transpose(const TPermutationMatrix &P);
  TPermutationMatrix& Inverse(void);
  TPermutationMatrix& Inverse(const TPermutationMatrix &P);

  // Multiplication
  TPermutationMatrix& Multiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);

  TPermutationMatrix& MultiplyTranspose(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
  TPermutationMatrix& TransposeMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
  TPermutationMatrix& MultiplyInverse(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
  TPermutationMatrix& InverseMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);

  // Random Permutation
  TPermutationMatrix& RandomUniform(void);
  TPermutationMatrix& RandomUniform(int d);

  void LapackLU(TLapackLU &LU);

  // Unsafe Routines
  TPermutationMatrix(int *buffer, int d);
  void ShareMemory(int *buffer, int d);
  void UniqueMemory(int d);
};

class TLapackLU
{
public:
  TDenseMatrix X;
  double *LU;
  int *p;
  int dim;
  int rows;
  int cols;
  int singular;

  explicit TLapackLU(const TDenseMatrix &M);
  ~TLapackLU();
};

/*******************************************************************************/
/************************ Complex Matrices and Vectors *************************/
/*******************************************************************************/
class TDenseMatrixComplex
{
public:
  // Data
  double *matrix;
  int rows;
  int cols;
  bool column_major;

  // Constructors and Destructors
  ~TDenseMatrixComplex();
  TDenseMatrixComplex(void);
  TDenseMatrixComplex(int r, int c);
  TDenseMatrixComplex(int r, int c, const std::complex<double> &s);
  TDenseMatrixComplex(int r, int c, bool col_major);
  TDenseMatrixComplex(int r, int c, bool col_major, const std::complex<double> &s);
  TDenseMatrixComplex(const TDenseMatrixComplex &M);
  TDenseMatrixComplex(const TDenseMatrix &Re);  
  TDenseMatrixComplex(const TDenseMatrix &Re, const TDenseMatrix &Im);  

  // Memory Management
  void UniqueMemory(void);
  void UniqueMemory(int r, int c, bool col_major);

  // Element Access
  int Index(int r, int c) const;
  std::complex<double> operator()(int r, int c) const;
  double Real(int r, int c) const;
  double& Real(int r, int c);
  double Imag(int r, int c) const;
  double& Imag(int r, int c);

  // Assignment
  TDenseMatrixComplex& operator=(const TDenseMatrixComplex &M);

  // Multiplication
  TDenseMatrixComplex operator*(const TDenseMatrixComplex &M);

  // Inverses
  TDenseMatrixComplex& InverseMultiply(const TDenseMatrixComplex &M1, const TDenseMatrixComplex &M2, int method=SOLVE_LU);

  // Random Matrices
  TDenseMatrixComplex& RandomNormal(void);
  TDenseMatrixComplex& RandomNormal(int r, int c);
  TDenseMatrixComplex& RandomUniform(void);
  TDenseMatrixComplex& RandomUniform(int r, int c);

  // Unsafe routines
  TDenseMatrixComplex(double* buffer, int r, int c, bool col_major);
  void ShareMemory(double *buffer, int r, int c, bool col_major);
};

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

//=== I/O =======================================================================
std::ostream& operator<<(std::ostream &output, const TDenseVector &v);
std::ostream& operator<<(std::ostream &output, const TDenseMatrix &M);
std::ostream& operator<<(std::ostream &output, const TDenseMatrixComplex &M);

std::istream& operator>>(std::istream &input, TDenseVector &v);
std::istream& operator>>(std::istream &input, TDenseMatrix &M);

//=== sub-vector extraction =====================================================
TDenseVector SubVector(const TDenseVector &v, int b, int n);
TDenseVector SubVector(const TDenseVector &v, const TIndex &idx);
TDenseVector ColumnVector(const TDenseMatrix &M, int c);
TDenseVector ColumnVector(const TDenseMatrix &M, int c, int br, int er);
TDenseVector ColumnVector(const TDenseMatrix &M, int c, const TIndex &idx);
TDenseVector RowVector(const TDenseMatrix &M, int r);
TDenseVector RowVector(const TDenseMatrix &M, int r, int bc, int ec);
TDenseVector RowVector(const TDenseMatrix &M, int r, const TIndex &idx);

//=== sub-matrix extraction =====================================================
TDenseMatrix ColumnMatrix(const TDenseVector &v);
TDenseMatrix ColumnMatrix(const TDenseVector &v, int b, int e);
TDenseMatrix ColumnMatrix(const TDenseVector &v, const TIndex &idx);
TDenseMatrix RowMatrix(const TDenseVector &v);
TDenseMatrix RowMatrix(const TDenseVector &v, int b, int e);
TDenseMatrix RowMatrix(const TDenseVector &v, const TIndex &idx);
TDenseMatrix SubMatrix(const TDenseMatrix &M, int br, int er, int bc, int ec);
TDenseMatrix SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec);
TDenseMatrix SubMatrix(const TDenseMatrix &M, int br, int er, const TIndex &idx_c);
TDenseMatrix SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c);

//=== miscellaneous matrix/vector manipulation
TDenseVector Cat(const TDenseVector &v1, const TDenseVector &v2);
TDenseVector Cat(const TDenseVector &v1, double x2);
TDenseVector Cat(double x1, const TDenseVector &v2);

//TDenseVector Cat(const std::vector<TDenseVector> &v_array);
TDenseMatrix HCat(const TDenseMatrix &M1, const TDenseMatrix &M2);
//TDenseMatrix HCat(const std::vector<TDenseMatrix> &M_array);
TDenseMatrix VCat(const TDenseMatrix &M1, const TDenseMatrix &M2);
//TDenseMatrix VCat(const std::vector<TDenseMatrix> &M_array);
TDenseVector Vec(const TDenseMatrix &M);

//=== Initialization ============================================================
TDenseVector ConstantVector(double y, int d);
TDenseVector Zeros(int d);
TDenseVector Ones(int d);

TDenseMatrix ConstantMatrix(double y, int r, int c);
TDenseMatrix Zeros(int r, int c);
TDenseMatrix Ones(int r, int c);
TDenseMatrix DiagonalMatrix(double s, int r, int c);
TDenseMatrix DiagonalMatrix(const TDenseVector &y);
TDenseMatrix DiagonalMatrix(const TDenseVector &y, int r, int c);
TDenseMatrix Identity(int n);
TDenseMatrix Identity(int r, int c);
TDenseMatrix BlockDiagonalMatrix(const TDenseMatrix &M, int n);

//=== Unary operators ===========================================================
TDenseVector operator-(const TDenseVector &v);
TDenseVector Minus(const TDenseVector &v);
TDenseVector Abs(const TDenseVector &v);

TDenseMatrix operator-(const TDenseMatrix &M);
TDenseMatrix Minus(const TDenseMatrix &M);
TDenseMatrix Abs(const TDenseMatrix &M);
TDenseMatrix Transpose(const TDenseMatrix &M);
TDenseMatrix Inverse(const TDenseMatrix &M, int method=SOLVE_LU);

//=== Binary operators ==========================================================
TDenseMatrix operator+(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector operator+(const TDenseVector &v1, const TDenseVector &v2);
TDenseMatrix operator-(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector operator-(const TDenseVector &v1, const TDenseVector &v2);
TDenseMatrix Add(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector Add(const TDenseVector &v1, const TDenseVector &v2);
TDenseMatrix Subtract(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector Subtract(const TDenseVector &v1, const TDenseVector &v2);

TDenseMatrix operator*(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector operator*(const TDenseMatrix &M, const TDenseVector &v);
TDenseVector operator*(const TDenseVector &v, const TDenseMatrix &M); 
TDenseMatrix operator*(const TDenseMatrix &M, double s);
TDenseMatrix operator*(double s, const TDenseMatrix &M);
TDenseVector operator*(const TDenseVector &v, double s);
TDenseVector operator*(double s, const TDenseVector &v);

TDenseMatrix operator*(const TDenseVector &v1, const TDenseVector &v2); // Added by HWu

TDenseMatrix Multiply(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector Multiply(const TDenseMatrix &M, const TDenseVector &v);
TDenseVector Multiply(const TDenseVector &v, const TDenseMatrix &M); 
TDenseMatrix Multiply(const TDenseMatrix &M, double s);
TDenseMatrix Multiply(double s, const TDenseMatrix &M);
TDenseVector Multiply(const TDenseVector &v, double s);
TDenseVector Multiply(double s, const TDenseVector &v);

TDenseVector DotMultiply(const TDenseVector &v1, const TDenseVector &v2); // Added by HWu

TDenseMatrix Multiply(const TDenseVector &v1, const TDenseVector &v2); // Added by HWu

TDenseMatrix MultiplyTranspose(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector MultiplyTranspose(const TDenseVector &v, const TDenseMatrix &M);
TDenseMatrix MultiplyTranspose(double s, const TDenseMatrix &M);

TDenseMatrix TransposeMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2);
TDenseVector TransposeMultiply(const TDenseMatrix &M, const TDenseVector &v);
TDenseMatrix TransposeMultiply(const TDenseMatrix &M, double s);

TDenseMatrix MultiplyInverse(const TDenseMatrix &M1, const TDenseMatrix &M2, int method=SOLVE_LU);
TDenseVector MultiplyInverse(const TDenseVector &v, const TDenseMatrix &M, int method=SOLVE_LU);
TDenseMatrix MultiplyInverse(double s, const TDenseMatrix &M, int method=SOLVE_LU);

TDenseMatrix InverseMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2, int method=SOLVE_LU);
TDenseVector InverseMultiply(const TDenseMatrix &M, const TDenseVector &v, int method=SOLVE_LU);
TDenseMatrix InverseMultiply(const TDenseMatrix &M, double s, int method=SOLVE_LU);

TDenseVector Kron(const TDenseVector &v, const TDenseVector &w);
TDenseMatrix Kron(const TDenseMatrix &X, const TDenseMatrix &Y);

//=== Solving Linear Systems ====================================================
TDenseMatrix LeftSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_LU);
TDenseMatrix LeftSolve_Refine(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_REFINE_GENERAL); 
TDenseVector LeftSolve(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_LU);
TDenseVector LeftSolve_Refine(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_REFINE_GENERAL); 
TDenseMatrix RightSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method=SOLVE_LU);
TDenseVector RightSolve(const TDenseMatrix &A, const TDenseVector &b, int method=SOLVE_LU);

// === Added by HWu =============================================================
TDenseMatrix LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseMatrix &B); 
TDenseVector LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseVector &x); 
// ===============================================================================
//=== Miscellaneous Routines=====================================================
double Norm(const TDenseVector &v);
double Norm(const TDenseMatrix &M);
double MatrixNorm(const TDenseMatrix &M);
double Sum(const TDenseVector &v);
double InnerProduct(const TDenseVector &v1, const TDenseVector &v2);
double InnerProduct(const TDenseVector &v1, const TDenseVector &v2, const TDenseMatrix &QuadraticForm);
TDenseMatrix OuterProduct(const TDenseVector &v1, const TDenseVector &v2);
TDenseMatrix OuterProduct(const TDenseVector &v);

double Determinant(const TDenseMatrix &M); 
double Determinant_Cholesky(const TDenseMatrix &T); // T here is a Cholesky decomposition of some matrix
double LogAbsDeterminant(const TDenseMatrix &M); 
double LogAbsDeterminant_Cholesky(const TDenseMatrix &T); // T here is a Cholesky decomposition of some matrix 
int Rank(const TDenseMatrix &M); 
double RCond(const TDenseMatrix &M); 
bool IsZeroMatrix(const TDenseMatrix &M); 

//=== Random Matrices ===========================================================
TDenseVector RandomNormalVector(int d);
TDenseVector RandomUniformVector(int d);
TDenseMatrix RandomNormalMatrix(int r, int c);
TDenseMatrix RandomUniformMatrix(int r, int c);
TDenseMatrixComplex RandomNormalComplexMatrix(int r, int c);
TDenseMatrixComplex RandomUniformComplexMatrix(int r, int c);

//=== Sorting ===================================================================
TDenseVector Sort(const TDenseVector &v, bool ascending=true);
TDenseMatrix SortRows(const TDenseMatrix &M, int c, bool ascending=true);
TDenseMatrix SortColumns(const TDenseMatrix &M, int r, bool ascending=true);

//=== Decompositions ============================================================
void LU(TPermutationMatrix &P, TDenseMatrix &L, TDenseMatrix &U, const TDenseMatrix &M);
void SVD(TDenseMatrix &U, TDenseVector &d, TDenseMatrix &V, const TDenseMatrix &M, int compact=1);
void SVD(TDenseVector &d, const TDenseMatrix &M);
void Cholesky(TDenseMatrix &T, const TDenseMatrix &M, int options=CHOLESKY_UPPER_TRIANGULAR);
TDenseMatrix Cholesky(const TDenseMatrix &M, int options=CHOLESKY_UPPER_TRIANGULAR); 
void QR(TDenseMatrix &Q, TDenseMatrix &R, const TDenseMatrix &M, int compact=1);
void QR(TDenseMatrix &R, const TDenseMatrix &M, int compact=1);
void Eig(TDenseVector &RealEigenValues, TDenseVector &ImaginaryEigenValues, TDenseMatrix &RealEigenVectors, TDenseMatrix &ImaginaryEigenVectors, const TDenseMatrix &M);
void Eig(TDenseVector &RealEigenValues, TDenseVector &ImaginaryEigenValues, const TDenseMatrix &M);
void Eig(TDenseVector &EigenValues, TDenseMatrix &EigenVectors, const TDenseMatrix &M);
void Eig(TDenseVector &EigenValues, const TDenseMatrix &M);

//====== Added by HWu ======
void Schur(TDenseMatrix &T, TDenseVector &eR, TDenseVector &eI, TDenseMatrix &Z, const TDenseMatrix &M, bool if_schur_vector=false); 
void OrderSchur(TDenseMatrix &OrdT, TDenseVector &OrdER, TDenseVector &OrdEI, TDenseMatrix &OrdZ, const TDenseMatrix &T, const TDenseMatrix &Z, const int *select, bool if_schur_vector=false);
void Annihilator(TDenseMatrix &Ann, TDenseVector &AbsEig, const TDenseMatrix &X); 
int NullSpace(TDenseMatrix &Z, const TDenseMatrix &A); 
//==========================

//=== Permutation matrices ======================================================
//--- i/o
std::ostream& operator<<(std::ostream &output, const TPermutationMatrix &P);

//--- uniary operators
TPermutationMatrix Transpose(const TPermutationMatrix &P);
TPermutationMatrix Inverse(const TPermutationMatrix &P);

//--- binary operators
TPermutationMatrix operator*(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix operator*(const TDenseMatrix &M, const TPermutationMatrix &P);
TDenseMatrix operator*(const TPermutationMatrix &P, const TDenseMatrix &M);
TDenseVector operator*(const TDenseVector &v, const TPermutationMatrix &P);
TDenseVector operator*(const TPermutationMatrix &P, const TDenseVector &v);

TPermutationMatrix Multiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix Multiply(const TDenseMatrix &M, const TPermutationMatrix &P);
TDenseMatrix Multiply(const TPermutationMatrix &P, const TDenseMatrix &M);
TDenseVector Multiply(const TDenseVector &v, const TPermutationMatrix &P);
TDenseVector Multiply(const TPermutationMatrix &P, const TDenseVector &v);

TPermutationMatrix MultiplyTranspose(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix MultiplyTranspose(const TDenseMatrix &M, const TPermutationMatrix &P);
TDenseMatrix MultiplyTranspose(const TPermutationMatrix &P, const TDenseMatrix &M);
TDenseVector MultiplyTranspose(const TDenseVector &v, const TPermutationMatrix &P);

TPermutationMatrix TransposeMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix TransposeMultiply(const TDenseMatrix &M, const TPermutationMatrix &P);
TDenseMatrix TransposeMultiply(const TPermutationMatrix &P, const TDenseMatrix &M);
TDenseVector TransposeMultiply(const TPermutationMatrix &P, const TDenseVector &v);

TPermutationMatrix MultiplyInverse(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix MultiplyInverse(const TDenseMatrix &M, const TPermutationMatrix &P);
TDenseMatrix MultiplyInverse(const TPermutationMatrix &P, const TDenseMatrix &M, int method=SOLVE_LU);
TDenseVector MultiplyInverse(const TDenseVector &v, const TPermutationMatrix &P);

TPermutationMatrix InverseMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2);
TDenseMatrix InverseMultiply(const TDenseMatrix &M, const TPermutationMatrix &P, int method=SOLVE_LU);
TDenseMatrix InverseMultiply(const TPermutationMatrix &P, const TDenseMatrix &M);
TDenseVector InverseMultiply(const TPermutationMatrix &P, const TDenseVector &v);

//--- random permutation matices
TPermutationMatrix RandomUniformPermutationMatrix(int d);


//=== Base Routines =============================================================
double* BaseCopyVector(int b, int n, double *input, int size, double *output);
double* BaseCopyVector(int *idx, int n, double *input, int size, double *output);
double* BaseInsertVector(int b, int n, double *input, int size, int _b, int _n, double *output);
double* BaseInsertVector(int *idx, int n, double *input, int size, int _b, int _n, double *output);
double* BaseInsertVector(int b, int n, double *input, int size, int *_idx, int _n, double *output);
double* BaseInsertVector(int *idx, int n, double *input, int size, int *_idx, int _n, double *output);
double* BaseCopyMatrix(int br, int nr, int bc, int nc, double *input, int r, int c, bool column_major, double *output);
double* BaseCopyMatrix(int *idxr, int nr, int bc, int nc, double *input, int r, int c, bool column_major, double *output);
double* BaseCopyMatrix(int br, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, double *output);
double* BaseCopyMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, double *output);
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int *idxr, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int *idxc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int *_idxr, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int *_idxc, int _nc, double *output, int _r, int _c, bool _column_major);
double* BaseInsertMatrix(int br, int nr, int bc, int nc, double *input, int r, int c, bool column_major, 
			 int _br, int _nr, int _bc, int _nc, double *output, int _r, int _c, bool _column_major);

double* BaseMultiply(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double*buffer);
double* BaseMultiplyComplexComplex(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double*buffer);
double* BaseMultiply(double s, double *source, int n, double *destination);
double* BaseMultiply(double *M, int r, int c, int cm, int *P, int d, int T, double *buffer);
double* BaseMultiply(int *P, int d, int T, double *M, int r, int c, int cm, double *buffer);
int*    BaseMultiply(int *P1, int d1, int T1, int *P2, int d2, int T2, int *buffer);
double* BaseMultiply(int, double *, int, double *, int); 
double* BaseKron(double *M1, int r1, int c1, bool cm1, double *M2, int r2, int c2, bool cm2, double *buffer);

double* BaseAdd(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer);
double* BaseSubtract(double *M1, int r1, int c1, int cm1, double *M2, int r2, int c2, int cm2, double *buffer);
double* BaseAdd(double *v1, int d1, double *v2, int d2, double *buffer);
double* BaseSubtract(double *v1, int d1, double *v2, int d2, double *buffer);

double* BaseDiagonalMatrix(int r, int c, double *v, int d, double *buffer);
double* BaseDiagonalMatrix(int r, int c, double s, double *buffer);

double* BaseMinus(double *source, int n, double *destination);
double* BaseAbsoluteValue(double *source, int n, double *destination);
int BaseSVD(double *M, int r, int c, int cm, int s, double* U, double* d, double* V, int compact);
double* BaseCholesky(double* M, int r, int c, int cm, double *buffer, int options);
double* BaseGeneralizedInverse(const double* M, int r, int c, int cm, int s, double *buffer);
double* BaseInverse(double *M, int r, int c, int cm, double *buffer, int method);
double* BaseSolve(double *A, int ra, int ca, int cma, double *B, int rb, int cb, int cmb, double *X, int method);
double* BaseSolveComplex(double *A, int ra, int ca, int cma, double *B, int rb, int cb, int cmb, double *X, int method);
double* BaseSolve_Refine(double *A, int ra, int ca, int cma, double *B, int rb, int cb, int cmb, double *X, int method); 

void BaseQuickSortArray(double *x, int m, bool ascending);
void BaseQuickSortMatrix(double *x, int m, int n, int idx, double *z, bool ascending);

//=== Depreciated routines
TDenseMatrix ZeroMatrix(int r, int c);
TDenseVector ZeroVector(int d);  

//=== Define inline functions
#include "dw_dense_matrix_inline.hpp"



/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
// //=== Multiplication Routines ===

// TMatrix ProductMU(TMatrix X, TMatrix Y, TMatrix Z);
// TMatrix ProductML(TMatrix X, TMatrix Y, TMatrix Z);
// TMatrix ProductUM(TMatrix X, TMatrix Y, TMatrix Z);
// TMatrix ProductLM(TMatrix X, TMatrix Y, TMatrix Z);

// //=== Updating ===
// TVector LinearCombinationV(TVector x, PRECISION a, TVector y, PRECISION b, TVector z);
// TMatrix LinearCombinationM(TMatrix X, PRECISION a, TMatrix Y, PRECISION b, TMatrix Z);
// #define UpdateV(a,x,b,y) LinearCombinationV(x,a,x,b,y)
// #define UpdateM(a,X,b,Y) LinearCombinationM(X,a,X,b,Y)
// TMatrix UpdateProductMM(PRECISION a, TMatrix X, PRECISION b, TMatrix Y, TMatrix Z);
// TVector UpdateProductMV(PRECISION a, TVector x, PRECISION b, TMatrix Y, TVector z);
// TVector UpdateProductVM(PRECISION a, TVector x, PRECISION b, TVector y, TMatrix Z);

// /* Matrix Inverse Routines */
// TMatrix GeneralizedInverse(TMatrix X, TMatrix Y);

// /* Matrix Decompositions */
// int GeneralizedSchur_Real(TMatrix S, TMatrix T, TMatrix Q, TMatrix Z, TMatrix A, TMatrix B, 
// 			  TVector alpha_r, TVector alpha_i, TVector beta); ;
// int ReorderGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
// 				 TMatrix Z, int *select, TVector alpha_r, TVector alpha_i, TVector beta);
// int SortGeneralizedSchur_Real(TMatrix SS, TMatrix TT, TMatrix QQ, TMatrix ZZ, TMatrix S, TMatrix T, TMatrix Q, 
// 			      TMatrix Z, TVector alpha_r, TVector alpha_i, TVector beta, int descend);

// int LU(TPermutation P, TMatrix X, TMatrix A);
// TVector LU_SolveCol(TVector x, TVector y, TMatrix LU, TPermutation P);
// TVector LU_SolveRow(TVector x, TVector y, TMatrix LU, TPermutation P);

// int Eigenvalues(TVector Re, TVector Im, TMatrix X);
// int Eigen(TVector ReVal, TVector ImVal, TMatrix ReVec, TMatrix ImVec, TMatrix X);

// TMatrix MatrixSquareRoot(TMatrix X, TMatrix Y);

// /* Complex Routines */
// int ComplexProductMM(TMatrix ReX, TMatrix ImX, TMatrix ReY, TMatrix ImY, TMatrix ReZ, TMatrix ImZ);

// /* Miscellaneous Routines */


// PRECISION MatrixNorm(TMatrix X);  (done but not tested)
// PRECISION Determinant_LU(TMatrix X); (done but not tested)
// PRECISION LogAbsDeterminant_LU(TMatrix X); (done but not tested)

// PRECISION Trace(TMatrix X);
// TVector CrossProduct_LU(TVector x, TMatrix Y);
// int Rank_SVD(TMatrix X);
// TMatrix NullSpace(TMatrix Y);
// TMatrix ForceSymmetric(TMatrix X);

// TMatrix ReshapeV(TMatrix X, TVector y, int rows, int cols);

// /* Input - Output Routines */
// int OutVectorFloat(FILE *f, TVector x);
// int OutMatrixFloat(FILE *f, TMatrix X);
// int OutVectorDouble(FILE *f, TVector x);
// int OutMatrixDouble(FILE *f, TMatrix X);
// TVector InVector(FILE *f, TVector x);
// TMatrix InMatrix(FILE *f, TMatrix X);


//=== Dense matrix and vector classes ===
// These classes attempt to provide an efficient implementation of dense vectors
// and matrices. Both TDenseVectors and TDenseMatrices are implemented as an 
// array of doubles and matrices can be in either column major or row major 
// format.  TPermutationMatrics are an array of ints.  These classes are designed
// to be used with the blas/lapack libraries.
//
// Efficient memory management uses shared memory so that pointers can be equated
// as opposed to memory being copied.  This can be critical when constructing and
// equating classes.  The scheme uses an intrusive counter.  Coping memory is
// done by equating pointers and incrementing the memory count.  Freeing memory 
// done by decrementing the memory counter until it is non-positive, at which 
// point it is actually freed.  The following six functions are provided for the 
// management of the shared memory system.  Here <type> is either double or int. 
//
//     void FreeSharedMemory_<type>(<type>*)
//     void IncrementSharedMemory_<type>(<type>*)
//     int SharedMemoryCounter_<type>(<type>*)
//     int SharedMemoryDimension_<type>(<type>*)
//     <type>* AllocateSharedMemory_<type>(int)
//     <type>* UniqueSharedMemory_<type>(<type>*) 
//
// Since these classes are for numeric programming and designed to be extremely 
// efficient, all members are public.  Except for the shared memory system, this 
// means that the internal implementation can not be changed.  The downside to 
// this choice is that it is easier for the system to become corrupted.  We 
// describe below the safe practices that will help to ensure that this does not
// happen.  First, we describe the state of an uncorrupted system.
//
// TDenseVector
//  1) The double array was allocated by either AllocateSharedMemory_double() or 
//     UniqueSharedMemory_double().
//  2) The return value of SharedMemoryCounter_double() is equal to the number
//     of instances of TDenseVector or TDenseMatrix that share the double array.
//  3) The return value of SharedMemoryDimension_double() is non-negative and 
//     equal to dim.
//
// TDenseMatrix
//  1) The double array was allocated by either AllocateSharedMemory_double() or 
//     UniqueSharedMemory_double().
//  2) The return value of SharedMemoryCounter_double() is equal to the number
//     of instances of TDenseVector or TDenseMatrix that share the double array.
//  3) The return value of SharedMemoryDimension_double() is equal to rows*cols,
//     both rows and cols are non-negative, and column_major is either zero or 
//     one.
//
// TPermutationMatrix
//  1) The integer array was allocated by either AllocateSharedMemory_int() or 
//     UniqueSharedMemory_int().
//  2) The return value of SharedMemoryCounter_int() is equal to the number of
//     instances of TPermutationMatrix that share the integer array.
//  3) The return value of SharedMemoryDimension_int() is non-negative and equal 
//     to dim.  Also, it must be the case that 0 <= permutation[i] < dim for 
//     0 <= i < dim and that permutation[i] != permutation[j] for 
//     0 <= i < j < dim.
//
// While point (3) can be verified, points (1) and (2) cannot. For this reason,
// care must be taken.  The following points help ensure safe access.
// 
// a) FreeShareMemory_<type>(buffer) should not be directly called except in the 
//    case that buffer was allocated by AllocateSharedMemory_<type>() and it is
//    the case that SharedMemoryCounter_<type>(buffer) returns 0.
// 
// b) UniqueSharedMemory_<type>() and IncrementSharedMemory_<type> should never 
//    be directly called.
//
// c) In the constructors and routines
//
//        TDenseVector::TDenseVector(double *buffer, int d)
//        TDenseVector::ShareMemory(double *buffer, int d)
//        TDenseMatrix::TDenseMatrix(double *buffer, int r, int c, int col_major)
//        TDenseMatrix::ShareMemory(double *buffer, int r, int c, int col_major)
//        TPermutationMatrix::TPermutationMatrix(int *buffer, int d)
//        TPermutationMatrix::ShareMemory(int *buffer, int d)
//    
//    the array buffer should either have been allocated by a direct call to
//    AllocateShareMemory_<type>() or be the shared memory belonging to an 
//    instance of TDenseVector, TDenseMatrix, or TPermutationMatrix.  The other
//    arguments must satisfy point (3) for the appropriate class.
//
// d) The member elements can always be safely read, subject to the usual caution
//    that one must not read past of the end of the array.  The length of the
//    shared memory array is either dim or rows*cols, which is the same as the
//    return value of SharedMemoryDimension_<type>().
//
// e) The pointer to the shared memory should never be directly modified and the
//    array elements can be modified only if SharedMemoryCounter_<type>() is less
//    or equal to one.  A sucessful call to the member routine UniqueMemory() 
//    guarantees this will be true.  However, various member routines can cause 
//    the memory counter to be incremented.
//
// f) The other member elements should only be modified to the extent that the 
//    new values satisfy point (3) for the appropriate class.
//
// General points concerning efficiency
//
// a) For functions which return a vector/matrix, there are often two versions of
//    the routine.  If F(...) is the function, then there will be x.F(...) and 
//    x=F(...).  The difference between these depends on the details of the 
//    implementation and the efficiency of the compiler, but for most compilers 
//    the second call results in an additional one or two base constructor calls, 
//    one or two additional assignment calls, and one or two additional 
//    destructor calls.  However, in general, there are no additional memory 
//    allocations, deletions or memory copying. Thus the overhead associated with
//    the second call is minimal.  However, when these calls appear in a loop in 
//    which the dimensions of x are not changing, the first version can avoid one 
//    memory allocation and deletion per loop iteration.
//      
// b) Suppose F is a function which takes a single vector/matrix argument and 
//    returns a single vector/matrix value.  It may also that some additional 
//    auxiliary arguments.  For such functions, there are often three versions of 
//    the routine, x.F(y,...), x=F(y,...), and x=y.F(...).  The second two will
//    result in identical code.  The comments in (a) also apply in this case.
//
// c) Where ever possible, we overload operators so that one can write formulas 
//    that as opposed to a series of function calls.  For instance, we can write
//    x=y*z as opposed to x=Multiply(y,z).  These two expressions will result in
//    identical code, thus the comments in (a) will apply in this case.
//

#endif
