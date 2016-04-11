
#ifndef _DW_DENSE_MATRIX_INLINE_
#define _DW_DENSE_MATRIX_INLINE_

#include <stdlib.h>

// Inline functions for dw_dense_matrix.hpp

//===============================================================================
//=== Shared Memory Management ==================================================
//===============================================================================
#define D_COUNTER_OFFSET   -2
#define D_DIMENSION_OFFSET -1
#define D_TOTAL_OFFSET      2

// Returns the value of the memory counter
inline int SharedMemoryCounter_double(double *buffer) { return buffer ? *((int*)(buffer+D_COUNTER_OFFSET)) : 1; }

// Returns the dimension of the shared memory available to the user
inline int SharedMemoryDimension_double(double *buffer) { return buffer ? *((int*)(buffer+D_DIMENSION_OFFSET)) : 0; }

// Decrements memory counter and frees memory if necessary.
inline void FreeSharedMemory_double(double *buffer) { if (buffer && (--(*((int*)(buffer+D_COUNTER_OFFSET))) <= 0)) delete[] (buffer-D_TOTAL_OFFSET); }

// Increments memory counter.
inline void IncrementSharedMemory_double(double *buffer) { if (buffer) ++(*((int*)(buffer+D_COUNTER_OFFSET))); }

// Allocates memory and sets memory counter to zero.  The argument dim MUST be 
// non-negative.  Can throw bad_alloc.
inline double* AllocateSharedMemory_double(int dim)
{
  if (!dim) return (double*)NULL; 
  double *buffer=(new double[dim+D_TOTAL_OFFSET])+D_TOTAL_OFFSET;
  *((int*)(buffer+D_COUNTER_OFFSET))=0;
  *((int*)(buffer+D_DIMENSION_OFFSET))=dim; 
  return buffer;
}

// The returned memory is guaranteed to be unique, which is equivalent to the 
// memory counter being one.  The argument buffer MUST be part of the shared 
// memory system with its memory counter greater than or equal to one.  The 
// calling convention of this routine MUST be
//
//             buffer=UniqueSharedMemory_double(buffer);
//
// This routine should never be called directly, use instead the appropriate
// member function UniqueMemory().  Can throw bad_alloc.
inline double* UniqueSharedMemory_double(double *buffer)
{
  if (!buffer || (*((int*)(buffer+D_COUNTER_OFFSET)) == 1)) return buffer;
  double *x=(new double[*((int*)(buffer+D_DIMENSION_OFFSET))+D_TOTAL_OFFSET]) + D_TOTAL_OFFSET;
  --(*((int*)(buffer+D_COUNTER_OFFSET)));  
  memcpy(x,buffer,(*((int*)(buffer+D_DIMENSION_OFFSET)))*sizeof(double)); 
  *((int*)(x+D_COUNTER_OFFSET))=1;  
  *((int*)(x+D_DIMENSION_OFFSET))=*((int*)(buffer+D_DIMENSION_OFFSET));
  return x;
}

//-------------------------------------------------------------------------------
#define I_COUNTER_OFFSET   -2
#define I_DIMENSION_OFFSET -1
#define I_TOTAL_OFFSET      2

// Returns the value of the memory counter
inline int SharedMemoryCounter_int(int *buffer) { return (buffer ? (buffer)[I_COUNTER_OFFSET] : 1); }

// Returns the dimension of the shared memory available to the user
inline int SharedMemoryDimension_int(int *buffer) { return (buffer ? buffer[I_DIMENSION_OFFSET] : 0); }

// Decrements memory counter and frees memory if necessary.
inline void FreeSharedMemory_int(int *buffer)  { if (buffer && (--((buffer)[I_COUNTER_OFFSET]) <= 0)) delete[] ((buffer)-I_TOTAL_OFFSET); }

// Increments memory counter.
inline void IncrementSharedMemory_int(int *buffer) { if (buffer) ++((buffer)[I_COUNTER_OFFSET]); }

// Allocates memory and sets memory counter to zero.  The argument dim MUST be 
// non-negative.  Can throw bad_alloc.
inline int* AllocateSharedMemory_int(int dim)
{
  if (!dim) return (int*)NULL;
  int *buffer=(new int[dim+I_TOTAL_OFFSET]) + I_TOTAL_OFFSET;
  buffer[I_COUNTER_OFFSET]=0;
  buffer[I_DIMENSION_OFFSET]=dim; 
  for (int i=dim-1; i >=0; i--) buffer[i]=i;
  return buffer;
}

// The returned memory is guaranteed to be unique, which is equivalent to the
// memory counter being one.  The argument buffer MUST be part of the shared
// memory system with its memory counter greater than or equal to one.  The
// calling convention of this routine MUST be
//
//             buffer=UniqueSharedMemory_double(buffer);
//
// This routine should never be called directly.  Instead, use the member 
// function UniqueMemory().  Can throw bad_alloc.
inline int* UniqueSharedMemory_int(int *buffer)
{
  if (!buffer || (buffer[I_COUNTER_OFFSET] == 1)) return buffer;
  int *x=(new int[buffer[I_DIMENSION_OFFSET]+I_TOTAL_OFFSET]) + I_TOTAL_OFFSET;
  --buffer[I_COUNTER_OFFSET];
  memcpy(x,buffer,buffer[I_DIMENSION_OFFSET]*sizeof(int));
  x[I_COUNTER_OFFSET]=1;
  x[I_DIMENSION_OFFSET]=buffer[I_DIMENSION_OFFSET];
  return x;
}

//===============================================================================
//=== class TIndex
//===============================================================================
inline TIndex::~TIndex() { free(index); }
inline int TIndex::Size(void) const { return size; }
inline TIndex& TIndex::Clear(void) { size=0; return *this; }
inline TIndex& TIndex::operator,(int idx) { return this->operator()(idx); }
inline TIndex& TIndex::operator,(const TIndex &idx) { return this->operator()(idx); }
inline TIndex& TIndex::operator+=(int idx) { return this->operator()(idx); }
inline TIndex& TIndex::operator+=(const TIndex &idx) { return this->operator()(idx); }
//===============================================================================
//===============================================================================
//===============================================================================


//===============================================================================
//===============================================================================
//===============================================================================
/*
   Frees existing memory, sets vector and dim and increments memory counter.  The
   array buffer MUST be part of the shared memory system and dimension of the 
   shared memory MUST be equal to d.  If buffer is equal to vector, then nothing 
   needs to be done.
*/
inline void TDenseVector::ShareMemory(double *buffer, int d) 
{ 
  if (buffer != vector)
    {
      FreeSharedMemory_double(vector);
      IncrementSharedMemory_double(buffer);
      vector=buffer;
      dim=d;
    } 
}

/*
   Guarantees the array vector is unique -- that its memory counter is one.
*/
inline void TDenseVector::UniqueMemory(void)
{
  vector=UniqueSharedMemory_double(vector);
}

/*
   Frees existing memory, sets matrix, rows, cols, and column_major, and
   increments the memory counter.  The array buffer MUST be part of the shared 
   memory system and the dimension of the shared memory MUST be r*c.  If buffer 
   is equal to matrix, only rows, cols, and column_major need to be set.
*/
inline void TDenseMatrix::ShareMemory(double *buffer, int r, int c, bool col_major) 
{
  if (buffer != matrix)
    {
      FreeSharedMemory_double(matrix);
      IncrementSharedMemory_double(buffer);
      matrix=buffer;
    }
  rows=r;
  cols=c;
  column_major=col_major;
}

/*
   Guarantees the array matrix is unique -- that its memory counter is one.
*/
inline void TDenseMatrix::UniqueMemory(void)
{
  matrix=UniqueSharedMemory_double(matrix);
}

/*
   Frees existing memory, sets permutation and dim, and increments memory 
   counter.  The array buffer MUST be part of the shared memory system and the 
   dimension of the shared memory MUST be d.  If buffer is equal to permutation,
   then nothing needs to be done.
*/
inline void TPermutationMatrix::ShareMemory(int *buffer, int d) 
{ 
  if (buffer != permutation)
    {
      FreeSharedMemory_int(permutation);
      IncrementSharedMemory_int(buffer);
      permutation=buffer;
      dim=d;
    }
}

/*
   Guarantees the array permutation is unique -- that its memory counter is one.
*/
inline void TPermutationMatrix::UniqueMemory(void) 
{ 
  permutation=UniqueSharedMemory_int(permutation);
}

//=== Constructors and Destructors ==============================================
inline TDenseVector::~TDenseVector() 
{ 
  FreeSharedMemory_double(vector);
}

/*
   The array buffer MUST be part of the shared memory system and dimension of the 
   shared memory MUST be equal to d.
*/
inline TDenseVector::TDenseVector(double *buffer, int d)
{
  IncrementSharedMemory_double(buffer);
  vector=buffer;
  dim=d;
}

inline TDenseVector::TDenseVector(void)
{
  vector=(double*)NULL;
  dim=0;
}

inline TDenseVector::TDenseVector(int d)
{
  if (d < 0) throw dw_exception("TDenseVector::TDenseVector(int): negative index");
  vector=AllocateSharedMemory_double(d);
  IncrementSharedMemory_double(vector);
  dim=d;
}

inline TDenseVector::TDenseVector(int d, double s)
{
  if (d < 0) throw dw_exception("TDenseVector::TDenseVector(int,double): negative index");
  vector=AllocateSharedMemory_double(d);
  IncrementSharedMemory_double(vector);
  for (int i=(dim=d)-1; i >= 0; i--) vector[i]=s;
}

inline TDenseVector::TDenseVector(const TDenseVector &v)
{
  IncrementSharedMemory_double(v.vector);
  vector=v.vector;
  dim=v.dim;
}

//-- TDenseMatrix Constructors/Destructors --------------------------------------
inline TDenseMatrix::~TDenseMatrix() 
{
  FreeSharedMemory_double(matrix);
}

/*
   The array buffer MUST be part of the shared memory system, the dimension of
   the shared memory is r*c with both r and c non-negative.
*/
inline TDenseMatrix::TDenseMatrix(double *buffer, int r, int c, bool col_major)
{
  IncrementSharedMemory_double(buffer);
  matrix=buffer;
  rows=r;
  cols=c;
  column_major=col_major;
}

inline TDenseMatrix::TDenseMatrix(void)
{
  matrix=(double*)NULL;
  rows=cols=0;
  column_major=1;
}

inline TDenseMatrix::TDenseMatrix(int r, int c)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrix::TDenseMatrix(int,int): negative index");
  matrix=AllocateSharedMemory_double(r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=1;
}

inline TDenseMatrix::TDenseMatrix(int r, int c, bool col_major)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrix::TDenseMatrix::TDenseMatrix(int,int,int): negative index");
  matrix=AllocateSharedMemory_double(r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
}

inline TDenseMatrix::TDenseMatrix(int r, int c, double s)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrix::TDenseMatrix(int,int,double): negative index");
  matrix=AllocateSharedMemory_double(r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=1;
  for (int i=rows*cols-1; i >= 0; i--) matrix[i]=s;
}

inline TDenseMatrix::TDenseMatrix(int r, int c, bool col_major, double s)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrix::TDenseMatrix(int,int,double): negative index");
  matrix=AllocateSharedMemory_double(r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
  for (int i=rows*cols-1; i >= 0; i--) matrix[i]=s;
}

inline TDenseMatrix::TDenseMatrix(const TDenseMatrix &M)
{
  IncrementSharedMemory_double(M.matrix);
  matrix=M.matrix;
  rows=M.rows;
  cols=M.cols;
  column_major=M.column_major;
};

//--- TPermutationMatrix Constructors/Destructors -------------------------------
inline TPermutationMatrix::~TPermutationMatrix()
{
  FreeSharedMemory_int(permutation);
}

/*
   It MUST be the case that buffer is part of the shared memory system and the 
   dimension of the shared memory is d.
*/
inline TPermutationMatrix::TPermutationMatrix(int *buffer, int d)
{
  IncrementSharedMemory_int(buffer);
  permutation=buffer;
  dim=d;
}

inline TPermutationMatrix::TPermutationMatrix(void)
{
  dim=0;
  permutation=(int*)NULL;
}

inline TPermutationMatrix::TPermutationMatrix(int d) 
{
  if (d < 0) throw dw_exception("TPermutationMatrix::TPermutationMatrix(int): negative index");
  permutation=AllocateSharedMemory_int(d);
  IncrementSharedMemory_int(permutation);
  dim=d;
}

inline TPermutationMatrix::TPermutationMatrix(const TPermutationMatrix &P)
{
  IncrementSharedMemory_int(P.permutation);
  permutation=P.permutation;
  dim=P.dim;
}

//===============================================================================
//=== Inline Base Routines ======================================================
//===============================================================================
/*
  Upon entry:
   P  : null pointer if d == 0 and permutation of {0,...,d-1} otherwise
   d  : dimension - must be non-negative
   T  : 0 or 1
   M  : null pointer if r*c == 0 and array of length at least r*c otherwise 
   r  : number of rows - must be non-negative.
   c  : number of columns - must be non-negative.
   cm : 1 => M is in column major format  -  0 => M is in row major format
   buffer : member if shared memory system

  Upon successful exit:
   The return value is a member of shared memory system containing the r by d
   matrixP*M if T == 0 and P'*M if T == 1.  The return value will have the same 
   major format as M.

  Throws:
   std::bad_alloc if unable to allocate required memory
   dw_exception if c != d
*/
inline double* BaseMultiply(int *P, int d, int T, double *M, int r, int c, int cm, double *buffer)
{
  return BaseMultiply(M,c,r,1-cm,P,d,1-T,buffer);
}

//=== Element Access ============================================================
inline int TDenseMatrix::Index(int r, int c) const
{
  return (column_major) ? c*rows+r : r*cols+c;
}
inline int TDenseMatrix::Stride(void) const
{
  return column_major ? rows : cols;
}
inline int TDenseMatrix::ColumnStride(void) const
{
  return column_major ? 1 : cols;
}
inline int TDenseMatrix::RowStride(void) const
{
  return column_major ? rows : 1;
}
inline double TDenseVector::operator()(int i) const
{
  if ((i < 0) || (i >= dim)) throw dw_exception("TDenseVector::operator() - index out of range");
  return vector[i];
}

//====== Added by Hwu ======
inline double TDenseVector::operator[](int i) const
{
  if ((i < 0) || (i >= dim)) throw dw_exception("TDenseVector::operator[] - index out of range");
  return vector[i];
}
//==========================

inline double TDenseMatrix::operator()(int r, int c) const
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseVector::operator() - index out of range");
  return matrix[Index(r,c)];
}

inline int TPermutationMatrix::operator()(int i) const
{
  if ((i < 0) || (i >= dim)) throw dw_exception("TPermutationMatrix::operator() - index out of range");
  return permutation[i];
}

//=== Assignment ================================================================
inline TDenseVector& TDenseVector::operator=(const TDenseVector &v)
{
  ShareMemory(v.vector,v.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::operator=(const TDenseMatrix &M)
{
  ShareMemory(M.matrix,M.rows,M.cols,M.column_major);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::operator=(const TPermutationMatrix &P)
{
  ShareMemory(P.permutation,P.dim);
  return *this;
}
inline TDenseVector& TDenseVector::operator+=(const TDenseVector &v)
{
  ShareMemory(BaseAdd(vector,dim,v.vector,v.dim,vector),dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::operator+=(const TDenseMatrix &M)
{
  ShareMemory(BaseAdd(matrix,rows,cols,column_major,M.matrix,M.rows,M.cols,M.column_major,matrix),rows,cols,column_major);
  return *this;
}
inline TDenseVector& TDenseVector::operator-=(const TDenseVector &v)
{
  ShareMemory(BaseSubtract(vector,dim,v.vector,v.dim,vector),dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::operator-=(const TDenseMatrix &M)
{
  ShareMemory(BaseSubtract(matrix,rows,cols,column_major,M.matrix,M.rows,M.cols,M.column_major,matrix),rows,cols,column_major);
  return *this;
}
inline TDenseVector& TDenseVector::operator*=(const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(vector,1,dim,1,M.matrix,M.rows,M.cols,M.column_major,vector),M.cols);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::operator*=(const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(matrix,rows,cols,column_major,M.matrix,M.rows,M.cols,M.column_major,matrix),rows,M.cols,1);
  return *this;
}

inline TDenseVector& TDenseVector::operator*=(double s)
{
  ShareMemory(BaseMultiply(s,vector,dim,vector),dim);
  return *this;
}

inline TDenseMatrix& TDenseMatrix::operator*=(double s)
{
  ShareMemory(BaseMultiply(s,matrix,rows*cols,matrix),rows,cols,column_major);
  return *this;
}

//=== extract subvector from vector =============================================
inline TDenseVector TDenseVector::operator()(int b, int e) const
{
  return TDenseVector(BaseCopyVector(b,e,vector,dim,(double*)NULL),(b <= e) ? e-b+1 : 0);
}
inline TDenseVector TDenseVector::operator()(const TIndex &idx) const
{
  return TDenseVector(BaseCopyVector(idx.index,idx.size,vector,dim,(double*)NULL),idx.size);
}
inline TDenseVector TDenseVector::SubVector(int b, int e) const
{
  return TDenseVector(BaseCopyVector(b,e,vector,dim,(double*)NULL),(b <= e) ? e-b+1 : 0);
}
inline TDenseVector TDenseVector::SubVector(const TIndex &idx) const
{
  return TDenseVector(BaseCopyVector(idx.index,idx.size,vector,dim,(double*)NULL),idx.size);
}
inline TDenseVector& TDenseVector::SubVector(const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseCopyVector(b,e,v.vector,v.dim,vector),(b <= e) ? e-b+1 : 0);
  return *this;
}
inline TDenseVector& TDenseVector::SubVector(const TDenseVector &v, TIndex &idx)
{
  ShareMemory(BaseCopyVector(idx.index,idx.size,v.vector,v.dim,vector),idx.size);
  return *this;
}
inline TDenseVector SubVector(const TDenseVector &v, int b, int e)
{
  return TDenseVector(BaseCopyVector(b,e,v.vector,v.dim,(double*)NULL),(b <= e) ? e-b+1 : 0);
}
inline TDenseVector SubVector(const TDenseVector &v, const TIndex &idx)
{
  return TDenseVector(BaseCopyVector(idx.index,idx.size,v.vector,v.dim,(double*)NULL),idx.size);
}

//=== extract subvector from matrix ============================================
inline TDenseVector TDenseMatrix::ColumnVector(int c) const
{
  return TDenseVector(BaseCopyMatrix(0,rows-1,c,c,matrix,rows,cols,column_major,(double*)NULL),rows);
}
inline TDenseVector TDenseMatrix::ColumnVector(int c, int br, int er) const
{
  return TDenseVector(BaseCopyMatrix(br,er,c,c,matrix,rows,cols,column_major,(double*)NULL),(br <= er) ? er-br+1 : 0);
}
inline TDenseVector TDenseMatrix::ColumnVector(int c, const TIndex &idx) const
{
  return TDenseVector(BaseCopyMatrix(idx.index,idx.size,c,c,matrix,rows,cols,column_major,(double*)NULL),idx.size);
}
inline TDenseVector ColumnVector(const TDenseMatrix &M, int c)
{
  return TDenseVector(BaseCopyMatrix(0,M.rows-1,c,c,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),M.rows);
}
inline TDenseVector ColumnVector(const TDenseMatrix &M, int c, int br, int er)
{
  return TDenseVector(BaseCopyMatrix(br,er,c,c,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),(br <= er) ? er-br+1 : 0);
}
inline TDenseVector ColumnVector(const TDenseMatrix &M, int c, const TIndex &idx)
{
  return TDenseVector(BaseCopyMatrix(idx.index,idx.size,c,c,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),idx.size);
}
inline TDenseVector& TDenseVector::ColumnVector(const TDenseMatrix &M, int c)
{
  ShareMemory(BaseCopyMatrix(0,M.rows-1,c,c,M.matrix,M.rows,M.cols,M.column_major,vector),M.rows);
  return *this;
}
inline TDenseVector& TDenseVector::ColumnVector(const TDenseMatrix &M, int c, int br, int er)
{
  ShareMemory(BaseCopyMatrix(br,er,c,c,M.matrix,M.rows,M.cols,M.column_major,vector),(br <= er) ? er-br+1 : 0);
  return *this;
}
inline TDenseVector& TDenseVector::ColumnVector(const TDenseMatrix &M, int c, const TIndex &idx)
{
  ShareMemory(BaseCopyMatrix(idx.index,idx.size,c,c,M.matrix,M.rows,M.cols,M.column_major,vector),idx.size);
  return *this;
}
inline TDenseVector TDenseMatrix::RowVector(int r) const
{
  return TDenseVector(BaseCopyMatrix(r,r,0,cols-1,matrix,rows,cols,column_major,(double*)NULL),cols);
}
inline TDenseVector TDenseMatrix::RowVector(int r, int bc, int ec) const
{
  return TDenseVector(BaseCopyMatrix(r,r,bc,ec,matrix,rows,cols,column_major,(double*)NULL),(bc <= ec) ? ec-bc+1 : 0);
}
inline TDenseVector TDenseMatrix::RowVector(int r, const TIndex &idx) const
{
  return TDenseVector(BaseCopyMatrix(r,r,idx.index,idx.size,matrix,rows,cols,column_major,(double*)NULL),idx.size);
}
inline TDenseVector RowVector(const TDenseMatrix &M, int r)
{
  return TDenseVector(BaseCopyMatrix(r,r,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),M.cols);
}
inline TDenseVector RowVector(const TDenseMatrix &M, int r, int bc, int ec)
{
  return TDenseVector(BaseCopyMatrix(r,r,bc,ec,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),(bc <= ec) ? ec-bc+1 : 0);
}
inline TDenseVector RowVector(const TDenseMatrix &M, int r, const TIndex &idx)
{
  return TDenseVector(BaseCopyMatrix(r,r,idx.index,idx.size,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),idx.size);
}
inline TDenseVector& TDenseVector::RowVector(const TDenseMatrix &M, int r)
{
  ShareMemory(BaseCopyMatrix(r,r,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,vector),M.cols);
  return *this;
}
inline TDenseVector& TDenseVector::RowVector(const TDenseMatrix &M, int r, int bc, int ec)
{
  ShareMemory(BaseCopyMatrix(r,r,bc,ec,M.matrix,M.rows,M.cols,M.column_major,vector),(bc <= ec) ? ec-bc+1 : 0);
  return *this;
}
inline TDenseVector& TDenseVector::RowVector(const TDenseMatrix &M, int r, const TIndex &idx)
{
  ShareMemory(BaseCopyMatrix(r,r,idx.index,idx.size,M.matrix,M.rows,M.cols,M.column_major,vector),idx.size);
  return *this;
}

//=== insert subvector into vector  =============================================
inline TDenseVector& TDenseVector::Insert(int b, const TDenseVector &v)
{
  ShareMemory(BaseInsertVector(0,v.dim-1,v.vector,v.dim,b,b+v.dim-1,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Insert(const TIndex &idx, const TDenseVector &v)
{
  ShareMemory(BaseInsertVector(0,v.dim-1,v.vector,v.dim,idx.index,idx.size,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Insert(int b, const TDenseVector &v, int b_v, int e_v)
{
  ShareMemory(BaseInsertVector(b_v,e_v,v.vector,v.dim,b,b+e_v-b_v,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Insert(const TIndex &idx, const TDenseVector &v, int b_v, int e_v)
{
  ShareMemory(BaseInsertVector(b_v,e_v,v.vector,v.dim,idx.index,idx.size,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Insert(int b, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertVector(idx_v.index,idx_v.size,v.vector,v.dim,b,b+idx_v.size-1,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Insert(const TIndex &idx, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertVector(idx_v.index,idx_v.size,v.vector,v.dim,idx.index,idx.size,vector),dim);
  return *this;
}//=== insert subvector from matrix into vector
inline TDenseVector& TDenseVector::InsertColumnVector(int b, const TDenseMatrix &M, int c)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,c,c,M.matrix,M.rows,M.cols,M.column_major,b,b+M.rows-1,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,c,c,M.matrix,M.rows,M.cols,M.column_major,idx.index,idx.size,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertColumnVector(int b, const TDenseMatrix &M, int c, int br, int er)
{
  ShareMemory(BaseInsertMatrix(br,er,c,c,M.matrix,M.rows,M.cols,M.column_major,b,b+er-br,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c, int br, int er)
{
  ShareMemory(BaseInsertMatrix(br,er,c,c,M.matrix,M.rows,M.cols,M.column_major,idx.index,idx.size,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertColumnVector(int b, const TDenseMatrix &M, int c, const TIndex &idx_r)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,c,c,M.matrix,M.rows,M.cols,M.column_major,b,b+idx_r.size-1,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertColumnVector(const TIndex &idx, const TDenseMatrix &M, int c, const TIndex &idx_r)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,c,c,M.matrix,M.rows,M.cols,M.column_major,idx.index,idx.size,0,0,vector,dim,1,true),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(int b, const TDenseMatrix &M, int r)
{
  ShareMemory(BaseInsertMatrix(r,r,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,0,0,b,b+M.cols-1,vector,1,dim,false),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r)
{
  ShareMemory(BaseInsertMatrix(r,r,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,0,0,idx.index,idx.size,vector,1,dim,false),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(int b, const TDenseMatrix &M, int r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(r,r,bc,ec,M.matrix,M.rows,M.cols,M.column_major,0,0,b,b+ec-bc,vector,1,dim,false),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(r,r,bc,ec,M.matrix,M.rows,M.cols,M.column_major,0,0,idx.index,idx.size,vector,1,dim,false),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(int b, const TDenseMatrix &M, int r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(r,r,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,0,0,b,b+idx_c.size-1,vector,1,dim,false),dim);
  return *this;
}
inline TDenseVector& TDenseVector::InsertRowVector(const TIndex &idx, const TDenseMatrix &M, int r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(r,r,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,0,0,idx.index,idx.size,vector,1,dim,false),dim);
  return *this;
}

//=== insert submatrix into matrix ==============================================
inline TDenseMatrix& TDenseMatrix::Insert(int r, int c, const TDenseMatrix &M)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,r,r+M.rows-1,c,c+M.cols-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, const TIndex idxc, const TDenseMatrix &M)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,r,r+M.rows-1,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, int c, const TDenseMatrix &M)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,c,c+M.cols-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M)
{
  ShareMemory(BaseInsertMatrix(0,M.rows-1,0,M.cols-1,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, int c, const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,r,r+er-br,c,c+ec-bc,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, const TIndex idxc, const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,r,r+er-br,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, int c, const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,c,c+ec-bc,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, int c, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,r,r+idx_r.size-1,c,c+ec-bc,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,r,r+idx_r.size-1,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, int c, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,c,c+ec-bc,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, int c, const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,r,r+er-br,c,c+idx_c.size-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, const TIndex idxc, const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,r,r+er-br,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, int c, const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,c,c+idx_c.size-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, int c, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,r,r+idx_r.size-1,c,c+idx_c.size-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(int r, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,r,r+idx_r.size-1,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, int c, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,c,c+idx_c.size-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Insert(const TIndex idxr, const TIndex idxc, const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  ShareMemory(BaseInsertMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,idxr.index,idxr.size,idxc.index,idxc.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}

//=== insert subvector into matrix ==============================================
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, int c, const TDenseVector &v)
{
  ShareMemory(BaseInsertMatrix(0,0,0,v.dim-1,v.vector,1,v.dim,false,r,r,c,c+v.dim-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, int c, const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseInsertMatrix(0,0,b,e,v.vector,1,v.dim,false,r,r,c,c+e-b,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, int c, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertMatrix(0,0,idx_v.index,idx_v.size,v.vector,1,v.dim,false,r,r,c,c+idx_v.size-1,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, TIndex idx, const TDenseVector &v)
{
  ShareMemory(BaseInsertMatrix(0,0,0,v.dim-1,v.vector,1,v.dim,false,r,r,idx.index,idx.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, TIndex idx, const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseInsertMatrix(0,0,b,e,v.vector,1,v.dim,false,r,r,idx.index,idx.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertRowMatrix(int r, TIndex idx, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertMatrix(0,0,idx_v.index,idx_v.size,v.vector,1,v.dim,false,r,r,idx.index,idx.size,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(int r, int c, const TDenseVector &v)
{
  ShareMemory(BaseInsertMatrix(0,v.dim-1,0,0,v.vector,v.dim,1,true,r,r+v.dim-1,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(int r, int c, const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseInsertMatrix(b,e,0,0,v.vector,v.dim,1,true,r,r+e-b,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(int r, int c, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertMatrix(idx_v.index,idx_v.size,0,0,v.vector,v.dim,1,true,r,r+idx_v.size-1,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v)
{
  ShareMemory(BaseInsertMatrix(0,v.dim-1,0,0,v.vector,v.dim,1,true,idx.index,idx.size,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseInsertMatrix(b,e,0,0,v.vector,v.dim,1,true,idx.index,idx.size,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InsertColumnMatrix(TIndex idx, int c, const TDenseVector &v, const TIndex &idx_v)
{
  ShareMemory(BaseInsertMatrix(idx_v.index,idx_v.size,0,0,v.vector,v.dim,1,true,idx.index,idx.size,c,c,matrix,rows,cols,column_major),rows,cols,column_major);
  return *this;
}

//=== extract submatrix from matrix =============================================
inline TDenseMatrix TDenseMatrix::operator()(int br, int er, int bc, int ec) const
{
  return TDenseMatrix(BaseCopyMatrix(br,er,bc,ec,matrix,rows,cols,column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,(bc <= ec) ? ec-bc+1 : 0,column_major);
}
inline TDenseMatrix TDenseMatrix::operator()(const TIndex &idx_r, int bc, int ec) const
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,bc,ec,matrix,rows,cols,column_major,(double*)NULL),idx_r.size,(bc <= ec) ? ec-bc+1 : 0,column_major);
}
inline TDenseMatrix TDenseMatrix::operator()(int br, int er, const TIndex &idx_c) const
{
  return TDenseMatrix(BaseCopyMatrix(br,er,idx_c.index,idx_c.size,matrix,rows,cols,column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,idx_c.size,column_major);
}
inline TDenseMatrix TDenseMatrix::operator()(const TIndex &idx_r, const TIndex &idx_c) const
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,matrix,rows,cols,column_major,(double*)NULL),idx_r.size,idx_c.size,column_major);
}
inline TDenseMatrix TDenseMatrix::SubMatrix(int br, int er, int bc, int ec) const
{
  return TDenseMatrix(BaseCopyMatrix(br,er,bc,ec,matrix,rows,cols,column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,(bc <= ec) ? ec-bc+1 : 0,column_major);
}
inline TDenseMatrix TDenseMatrix::SubMatrix(const TIndex &idx_r, int bc, int ec) const
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,bc,ec,matrix,rows,cols,column_major,(double*)NULL),idx_r.size,(bc <= ec) ? ec-bc+1 : 0,column_major);
}
inline TDenseMatrix TDenseMatrix::SubMatrix(int br, int er, const TIndex &idx_c) const
{
  return TDenseMatrix(BaseCopyMatrix(br,er,idx_c.index,idx_c.size,matrix,rows,cols,column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,idx_c.size,column_major);
}
inline TDenseMatrix TDenseMatrix::SubMatrix(const TIndex &idx_r, const TIndex &idx_c) const
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,matrix,rows,cols,column_major,(double*)NULL),idx_r.size,idx_c.size,column_major);
}
inline TDenseMatrix SubMatrix(const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  return TDenseMatrix(BaseCopyMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,(bc <= ec) ? ec-bc+1 : 0,M.column_major);
}
inline TDenseMatrix SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),idx_r.size,(bc <= ec) ? ec-bc+1 : 0,M.column_major);
}
inline TDenseMatrix SubMatrix(const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  return TDenseMatrix(BaseCopyMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),(br <= er) ? er-br+1 : 0,idx_c.size,M.column_major);
}
inline TDenseMatrix SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  return TDenseMatrix(BaseCopyMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),idx_r.size,idx_c.size,M.column_major);
}
inline TDenseMatrix& TDenseMatrix::SubMatrix(const TDenseMatrix &M, int br, int er, int bc, int ec)
{
  ShareMemory(BaseCopyMatrix(br,er,bc,ec,M.matrix,M.rows,M.cols,M.column_major,matrix),(br <= er) ? er-br+1 : 0,(bc <= ec) ? ec-bc+1 : 0,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, int bc, int ec)
{
  ShareMemory(BaseCopyMatrix(idx_r.index,idx_r.size,bc,ec,M.matrix,M.rows,M.cols,M.column_major,matrix),idx_r.size,(bc <= ec) ? ec-bc+1 : 0,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::SubMatrix(const TDenseMatrix &M, int br, int er, const TIndex &idx_c)
{
  ShareMemory(BaseCopyMatrix(br,er,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,matrix),(br <= er) ? er-br+1 : 0,idx_c.size,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::SubMatrix(const TDenseMatrix &M, const TIndex &idx_r, const TIndex &idx_c)
{
  ShareMemory(BaseCopyMatrix(idx_r.index,idx_r.size,idx_c.index,idx_c.size,M.matrix,M.rows,M.cols,M.column_major,matrix),idx_r.size,idx_c.size,M.column_major);
  return *this;
}

//=== convert subvectors to row or column matrices  =============================
inline TDenseMatrix TDenseVector::ColumnMatrix(void) const
{
  return TDenseMatrix(vector,dim,1,true);
}
inline TDenseMatrix TDenseVector::ColumnMatrix(int b, int e) const
{
  return TDenseMatrix(BaseCopyMatrix(b,e,0,0,vector,dim,1,true,(double*)NULL),(b <= e) ? e-b+1 : 0,1,true);
}
inline TDenseMatrix TDenseVector::ColumnMatrix(const TIndex &idx) const
{
  return TDenseMatrix(BaseCopyMatrix(idx.index,idx.size,0,0,vector,dim,1,true,(double*)NULL),idx.size,1,true);
}
inline TDenseMatrix ColumnMatrix(const TDenseVector &v)
{ 
  return TDenseMatrix(v.vector,v.dim,1,true);
}
inline TDenseMatrix ColumnMatrix(const TDenseVector &v, int b, int e)
{
  return TDenseMatrix(BaseCopyMatrix(b,e,0,0,v.vector,v.dim,1,true,(double*)NULL),(b <= e) ? e-b+1 : 0,1,true);
}
inline TDenseMatrix ColumnMatrix(const TDenseVector &v, const TIndex &idx)
{
  return TDenseMatrix(BaseCopyMatrix(idx.index,idx.size,0,0,v.vector,v.dim,1,true,(double*)NULL),idx.size,1,true);
}
inline TDenseMatrix& TDenseMatrix::ColumnMatrix(const TDenseVector &v)
{
  ShareMemory(v.vector,v.dim,1,true);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::ColumnMatrix(const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseCopyMatrix(b,e,0,0,v.vector,v.dim,1,true,matrix),(b <= e) ? e-b+1 : 0,1,true);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::ColumnMatrix(const TDenseVector &v,const TIndex &idx)
{
  ShareMemory(BaseCopyMatrix(idx.index,idx.size,0,0,v.vector,v.dim,1,true,matrix),idx.size,1,true);
  return *this;
}
inline TDenseMatrix TDenseVector::RowMatrix(void) const
{
  return TDenseMatrix(vector,1,dim,false);
}
inline TDenseMatrix TDenseVector::RowMatrix(int b, int e) const
{
  return TDenseMatrix(BaseCopyMatrix(0,0,b,e,vector,1,dim,false,(double*)NULL),1,(b <= e) ? e-b+1 : 0,false);
}
inline TDenseMatrix TDenseVector::RowMatrix(const TIndex &idx) const
{
  return TDenseMatrix(BaseCopyMatrix(0,0,idx.index,idx.size,vector,1,dim,false,(double*)NULL),1,idx.size,false);
}
inline TDenseMatrix RowMatrix(const TDenseVector &v)
{
  return TDenseMatrix(v.vector,1,v.dim,false);
}
inline TDenseMatrix RowMatrix(const TDenseVector &v, int b, int e)
{
  return TDenseMatrix(BaseCopyMatrix(0,0,b,e,v.vector,1,v.dim,false,(double*)NULL),1,(b <= e) ? e-b+1 : 0,false);
}
inline TDenseMatrix RowMatrix(const TDenseVector &v, const TIndex &idx)
{
  return TDenseMatrix(BaseCopyMatrix(0,0,idx.index,idx.size,v.vector,1,v.dim,false,(double*)NULL),1,idx.size,false);
}
inline TDenseMatrix& TDenseMatrix::RowMatrix(const TDenseVector &v)
{
  ShareMemory(v.vector,1,v.dim,false);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::RowMatrix(const TDenseVector &v, int b, int e)
{
  ShareMemory(BaseCopyMatrix(0,0,b,e,v.vector,1,v.dim,false,matrix),1,(b <= e) ? e-b+1 : 0,false);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::RowMatrix(const TDenseVector &v, const TIndex &idx)
{
  ShareMemory(BaseCopyMatrix(0,0,idx.index,idx.size,v.vector,1,v.dim,false,matrix),1,idx.size,false);
  return *this;
}

//=== miscellaneous matrix/vector manipulation
inline TDenseVector TDenseMatrix::Vec(void)
{
  return ::Vec(*this);
}

//=== Initialization ============================================================
inline TDenseVector ConstantVector(double s, int d)
{
  return TDenseVector(d,s);
}
inline TDenseVector& TDenseVector::Zeros(void)
{ 
  return Initialize(0.0);
}
inline TDenseVector& TDenseVector::Zeros(int d)
{
  return Initialize(0.0,d);
}
inline TDenseVector Zeros(int d)
{
  return TDenseVector(d,0.0);
}
inline TDenseVector ZeroVector(int d)
{
  return TDenseVector(d,0.0);
}
inline TDenseVector& TDenseVector::Ones(void)
{
  return Initialize(1.0);
}
inline TDenseVector& TDenseVector::Ones(int d)
{ 
  return Initialize(1.0,d);
}
inline TDenseVector Ones(int d)
{ 
  return TDenseVector(d,1.0);
}
inline TDenseMatrix ConstantMatrix(double y, int r, int c)
{
  return TDenseMatrix(r,c,y);
}
inline TDenseMatrix& TDenseMatrix::Zeros(void)
{
  return Initialize(0.0);
}
inline TDenseMatrix& TDenseMatrix::Zeros(int r, int c)
{ 
  return Initialize(0.0,r,c);
}
inline TDenseMatrix Zeros(int r, int c)
{
  return TDenseMatrix(r,c,0.0);
}
inline TDenseMatrix ZeroMatrix(int r, int c)
{
  return TDenseMatrix(r,c,0.0);
}
inline TDenseMatrix& TDenseMatrix::Ones(void)
{
  return Initialize(1.0);
}
inline TDenseMatrix& TDenseMatrix::Ones(int r, int c)
{
  return Initialize(1.0,r,c);
}
inline TDenseMatrix Ones(int r, int c)
{
  return TDenseMatrix(r,c,1.0);
}
inline TDenseMatrix& TDenseMatrix::DiagonalMatrix(double s)
{
  ShareMemory(BaseDiagonalMatrix(rows,cols,s,matrix),rows,cols,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::DiagonalMatrix(double s, int r, int c)
{
  ShareMemory(BaseDiagonalMatrix(r,c,s,matrix),r,c,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::DiagonalMatrix(const TDenseVector &v)
{
  ShareMemory(BaseDiagonalMatrix(v.dim,v.dim,v.vector,v.dim,matrix),v.dim,v.dim,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::DiagonalMatrix(const TDenseVector &v, int r, int c)
{
  ShareMemory(BaseDiagonalMatrix(r,c,v.vector,v.dim,matrix),r,c,1);
  return *this;
}
inline TDenseMatrix DiagonalMatrix(double s, int r, int c)
{
  return TDenseMatrix(BaseDiagonalMatrix(r,c,s,(double*)NULL),r,c,true);
}
inline TDenseMatrix DiagonalMatrix(const TDenseVector &v) 
{ 
  return TDenseMatrix(BaseDiagonalMatrix(v.dim,v.dim,v.vector,v.dim,(double*)NULL),v.dim,v.dim,1);
}
inline TDenseMatrix DiagonalMatrix(const TDenseVector &v, int r, int c) 
{
  return TDenseMatrix(BaseDiagonalMatrix(r,c,v.vector,v.dim,(double*)NULL),r,c,true);
}
inline TDenseMatrix& TDenseMatrix::Identity(int n)
{
  ShareMemory(BaseDiagonalMatrix(n,n,1.0,matrix),n,n,true);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Identity(int r, int c)
{
  ShareMemory(BaseDiagonalMatrix(r,c,1.0,(double*)NULL),r,c,true);
  return *this;
}
inline TDenseMatrix Identity(int n) 
{ 
  return TDenseMatrix(BaseDiagonalMatrix(n,n,1.0,(double*)NULL),n,n,true);
}
inline TDenseMatrix Identity(int r, int c) 
{
  return TDenseMatrix(BaseDiagonalMatrix(r,c,1.0,(double*)NULL),r,c,true);
}
inline TDenseMatrix TDenseMatrix::BlockDiagonalMatrix(int n)
{
  return ::BlockDiagonalMatrix(*this,n);
}

//== Vector Unary Operators =====================================================
inline TDenseVector& TDenseVector::Minus(void)
{
  ShareMemory(BaseMinus(vector,dim,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Minus(const TDenseVector &v)
{
  ShareMemory(BaseMinus(v.vector,v.dim,vector),v.dim);
  return *this;
}
inline TDenseVector operator-(const TDenseVector &v)
{
  return TDenseVector(BaseMinus(v.vector,v.dim,(double*)NULL),v.dim);
}
inline TDenseVector Minus(const TDenseVector &v)
{
  return TDenseVector(BaseMinus(v.vector,v.dim,(double*)NULL),v.dim);
}
inline TDenseVector& TDenseVector::Abs(void)
{
  ShareMemory(BaseAbsoluteValue(vector,dim,vector),dim);
  return *this;
}
inline TDenseVector& TDenseVector::Abs(const TDenseVector &v)
{
  ShareMemory(BaseAbsoluteValue(v.vector,v.dim,vector),v.dim);
  return *this;
}
inline TDenseVector Abs(const TDenseVector &v)
{
  return TDenseVector(BaseAbsoluteValue(v.vector,v.dim,(double*)NULL),v.dim);
}

//== Matrix Unary Operators =====================================================
inline TDenseMatrix& TDenseMatrix::Minus(void)
{
  ShareMemory(BaseMinus(matrix,rows*cols,matrix),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Minus(const TDenseMatrix &M)
{
  ShareMemory(BaseMinus(M.matrix,M.rows*M.cols,matrix),M.rows,M.cols,M.column_major);
  return *this;
}
inline TDenseMatrix operator-(const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMinus(M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
};
inline TDenseMatrix Minus(const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMinus(M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
};
inline TDenseMatrix& TDenseMatrix::Abs(void)
{
  ShareMemory(BaseAbsoluteValue(matrix,rows*cols,matrix),rows,cols,column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Abs(const TDenseMatrix &M)
{
  ShareMemory(BaseAbsoluteValue(M.matrix,M.rows*M.cols,matrix),M.rows,M.cols,M.column_major);
  return *this;
}
inline TDenseMatrix Abs(const TDenseMatrix &M)
{
  return TDenseMatrix(BaseAbsoluteValue(M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
}
inline TDenseMatrix& TDenseMatrix::Transpose(void) 
{ 
  int r=rows; 
  rows=cols; 
  cols=r;
  column_major=1-column_major;
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Transpose(const TDenseMatrix &M) 
{
  ShareMemory(M.matrix,M.cols,M.rows,1-M.column_major); 
  return *this;
}
inline TDenseMatrix Transpose(const TDenseMatrix &M)
{ 
  return TDenseMatrix(M.matrix,M.cols,M.rows,1-M.column_major); 
}

//=== Inverse ===================================================================
inline TDenseMatrix& TDenseMatrix::Inverse(int method)
{
  ShareMemory(BaseInverse(matrix,rows,cols,column_major,matrix,method),rows,cols,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Inverse(const TDenseMatrix &M, int method)
{
  ShareMemory(BaseInverse(M.matrix,M.rows,M.cols,M.column_major,matrix,method),M.rows,M.cols,1);
  return *this;
}
inline TDenseMatrix Inverse(const TDenseMatrix &M, int method)
{
  return TDenseMatrix(BaseInverse(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,method),M.rows,M.cols,1);
}

//======= Revised by HWu ===================
inline TDenseMatrix& TDenseMatrix::GeneralizedInverse(void)
{
  ShareMemory(BaseGeneralizedInverse(matrix,rows,cols,column_major,Stride(),matrix),cols, rows,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::GeneralizedInverse(const TDenseMatrix &M)
{
  ShareMemory(BaseGeneralizedInverse(M.matrix,M.rows,M.cols,M.column_major,M.Stride(),matrix),M.cols, M.rows,1);
  return *this;
}
inline TDenseMatrix GeneralizedInverse(const TDenseMatrix &M)
{
  return TDenseMatrix(BaseGeneralizedInverse(M.matrix,M.rows,M.cols,M.column_major,M.Stride(),(double*)NULL),M.cols, M.rows,1);
}
//============================================

//== Permutation Unary Operators ================================================
inline TPermutationMatrix& TPermutationMatrix::Inverse(void)
{
  return Transpose();
}
inline TPermutationMatrix& TPermutationMatrix::Inverse(const TPermutationMatrix &P)
{
  return Transpose(P);
}

//=== Addition/Subtraction ======================================================
inline TDenseMatrix& TDenseMatrix::Add(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  ShareMemory(BaseAdd(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix),M1.rows,M1.cols,M1.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::Add(const TDenseVector &v1, const TDenseVector &v2)
{
  ShareMemory(BaseAdd(v1.vector,v1.dim,v2.vector,v2.dim,vector),v1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Subtract(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  ShareMemory(BaseSubtract(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix),M1.rows,M1.cols,M1.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::Subtract(const TDenseVector &v1, const TDenseVector &v2)
{
  ShareMemory(BaseSubtract(v1.vector,v1.dim,v2.vector,v2.dim,vector),v1.dim);
  return *this;
}
inline TDenseMatrix operator+(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{ 
  return TDenseMatrix(BaseAdd(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M1.cols,M1.column_major);
}
inline TDenseVector operator+(const TDenseVector &v1, const TDenseVector &v2) 
{ 
  return TDenseVector(BaseAdd(v1.vector,v1.dim,v2.vector,v2.dim,(double*)NULL),v1.dim);
}
inline TDenseMatrix operator-(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{
  return TDenseMatrix(BaseSubtract(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M1.cols,M1.column_major);
}
inline TDenseVector operator-(const TDenseVector &v1, const TDenseVector &v2) 
{
  return TDenseVector(BaseSubtract(v1.vector,v1.dim,v2.vector,v2.dim,(double*)NULL),v1.dim);
}
inline TDenseMatrix Add(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{ 
  return TDenseMatrix(BaseAdd(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M1.cols,M1.column_major);
}
inline TDenseVector Add(const TDenseVector &v1, const TDenseVector &v2) 
{ 
  return TDenseVector(BaseAdd(v1.vector,v1.dim,v2.vector,v2.dim,(double*)NULL),v1.dim);
}
inline TDenseMatrix Subtract(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{
  return TDenseMatrix(BaseSubtract(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M1.cols,M1.column_major);
}
inline TDenseVector Subtract(const TDenseVector &v1, const TDenseVector &v2) 
{
  return TDenseVector(BaseSubtract(v1.vector,v1.dim,v2.vector,v2.dim,(double*)NULL),v1.dim);
}

//== Multiplication =============================================================
inline TDenseMatrix& TDenseMatrix::Multiply(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  ShareMemory(BaseMultiply(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix),M1.rows,M2.cols,1);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(const TDenseMatrix &M, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,v.vector,v.dim,1,1,vector),M.rows);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(const TDenseVector &v, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(v.vector,1,v.dim,1,M.matrix,M.rows,M.cols,M.column_major,vector),M.cols);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Multiply(double s, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(s,M.matrix,M.rows*M.cols,matrix),M.rows,M.cols,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Multiply(const TDenseMatrix &M, double s)
{
  ShareMemory(BaseMultiply(s,M.matrix,M.rows*M.cols,matrix),M.rows,M.cols,M.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(double s, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(s,v.vector,v.dim,vector),v.dim);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(const TDenseVector &v, double s)
{
  ShareMemory(BaseMultiply(s,v.vector,v.dim,vector),v.dim);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::Multiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  ShareMemory(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,0,permutation),P1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Multiply(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,0,matrix),M.rows,P.dim,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Multiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,0,M.matrix,M.rows,M.cols,M.column_major,matrix),P.dim,M.cols,M.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,0,v.vector,v.dim,1,1,vector),P.dim);
  return *this;
}
inline TDenseVector& TDenseVector::Multiply(const TDenseVector &v, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,0,vector),P.dim);
  return *this;
}

inline TDenseMatrix operator*(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{ 
  return TDenseMatrix(BaseMultiply(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M2.cols,1);
}
inline TDenseVector operator*(const TDenseMatrix &M, const TDenseVector &v) 
{ 
  return TDenseVector(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,v.vector,v.dim,1,1,(double*)NULL),M.rows);
}
inline TDenseVector operator*(const TDenseVector &v, const TDenseMatrix &M) 
{ 
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),M.cols);
}
inline TDenseMatrix operator*(const TDenseMatrix &M, double s) 
{ 
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
};
inline TDenseMatrix operator*(double s, const TDenseMatrix &M) 
{ 
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
}
inline TDenseVector operator*(const TDenseVector &v, double s) 
{ 
  return TDenseVector(BaseMultiply(s,v.vector,v.dim,(double*)NULL),v.dim);
}
inline TDenseVector operator*(double s, const TDenseVector &v) 
{ 
  return TDenseVector(BaseMultiply(s,v.vector,v.dim,(double*)NULL),v.dim);
}
inline TPermutationMatrix operator*(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,0,(int*)NULL),P1.dim);
}
inline TDenseMatrix operator*(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  return TDenseMatrix(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,0,(double*)NULL),M.rows,P.dim,M.column_major);
}
inline TDenseMatrix operator*(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(P.permutation,P.dim,0,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),P.dim,M.cols,M.column_major);
}
inline TDenseVector operator*(const TPermutationMatrix &P, const TDenseVector &v)
{
  return TDenseVector(BaseMultiply(P.permutation,P.dim,0,v.vector,v.dim,1,1,(double*)NULL),P.dim);
}
inline TDenseVector operator*(const TDenseVector &v, const TPermutationMatrix &P)
{
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,0,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseMatrix Multiply(const TDenseMatrix &M1, const TDenseMatrix &M2) 
{ 
  return TDenseMatrix(BaseMultiply(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.rows,M2.cols,1);
}
inline TDenseVector Multiply(const TDenseMatrix &M, const TDenseVector &v) 
{ 
  return TDenseVector(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,v.vector,v.dim,1,1,(double*)NULL),M.rows);
}
inline TDenseVector Multiply(const TDenseVector &v, const TDenseMatrix &M) 
{ 
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),M.cols);
}
inline TDenseMatrix Multiply(const TDenseMatrix &M, double s) 
{ 
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
};
inline TDenseMatrix Multiply(double s, const TDenseMatrix &M) 
{ 
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.rows,M.cols,M.column_major);
}
inline TDenseVector Multiply(const TDenseVector &v, double s) 
{ 
  return TDenseVector(BaseMultiply(s,v.vector,v.dim,(double*)NULL),v.dim);
}
inline TDenseVector Multiply(double s, const TDenseVector &v) 
{ 
  return TDenseVector(BaseMultiply(s,v.vector,v.dim,(double*)NULL),v.dim);
}
inline TPermutationMatrix Multiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,0,(int*)NULL),P1.dim);
}
inline TDenseMatrix Multiply(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  return TDenseMatrix(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,0,(double*)NULL),M.rows,P.dim,M.column_major);
}
inline TDenseMatrix Multiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(P.permutation,P.dim,0,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),P.dim,M.cols,M.column_major);
}
inline TDenseVector Multiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  return TDenseVector(BaseMultiply(P.permutation,P.dim,0,v.vector,v.dim,1,1,(double*)NULL),P.dim);
}
inline TDenseVector Multiply(const TDenseVector &v, const TPermutationMatrix &P)
{
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,0,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseMatrix& TDenseMatrix::MultiplyTranspose(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  ShareMemory(BaseMultiply(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.cols,M2.rows,1-M2.column_major,matrix),M1.rows,M2.rows,1);
  return *this;
}
inline TDenseVector& TDenseVector::MultiplyTranspose(const TDenseVector &v, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(v.vector,1,v.dim,1,M.matrix,M.cols,M.rows,1-M.column_major,(double*)NULL),M.rows);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyTranspose(double s, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(s,M.matrix,M.rows*M.cols,matrix),M.cols,M.rows,1-M.column_major);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::MultiplyTranspose(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  ShareMemory(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,1,permutation),P1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyTranspose(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,1,matrix),M.rows,P.dim,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyTranspose(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,0,M.matrix,M.cols,M.rows,1-M.column_major,matrix),P.dim,M.rows,1-M.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::MultiplyTranspose(const TDenseVector &v, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,1,vector),P.dim);
  return *this;
}
//===============================================================================
inline TDenseMatrix MultiplyTranspose(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  return TDenseMatrix(BaseMultiply(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.cols,M2.rows,1-M2.column_major,(double*)NULL),M1.rows,M2.rows,1);
}
inline TDenseVector MultiplyTranspose(const TDenseVector &v, const TDenseMatrix &M)
{
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,M.matrix,M.cols,M.rows,1-M.column_major,(double*)NULL),M.rows);
}
inline TDenseMatrix MultiplyTranspose(double s, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.cols,M.rows,1-M.column_major);
}
inline TPermutationMatrix MultiplyTranspose(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,1,(int*)NULL),P1.dim);
}
inline TDenseMatrix MultiplyTranspose(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  return TDenseMatrix(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,1,(double*)NULL),M.rows,P.dim,M.column_major);
}
inline TDenseMatrix MultiplyTranspose(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(P.permutation,P.dim,0,M.matrix,M.cols,M.rows,1-M.column_major,(double*)NULL),P.dim,M.rows,1-M.column_major);
}
inline TDenseVector MultiplyTranspose(const TDenseVector &v, const TPermutationMatrix &P)
{
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,1,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseMatrix& TDenseMatrix::TransposeMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  ShareMemory(BaseMultiply(M1.matrix,M1.cols,M1.rows,1-M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix),M1.cols,M2.cols,1);
  return *this;
}
inline TDenseVector& TDenseVector::TransposeMultiply(const TDenseMatrix &M, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(M.matrix,M.cols,M.rows,1-M.column_major,v.vector,v.dim,1,1,vector),M.cols);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::TransposeMultiply(const TDenseMatrix &M, double s)
{
  ShareMemory(BaseMultiply(s,M.matrix,M.rows*M.cols,matrix),M.cols,M.rows,1-M.column_major);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::TransposeMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  ShareMemory(BaseMultiply(P1.permutation,P1.dim,1,P2.permutation,P2.dim,0,permutation),P1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::TransposeMultiply(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(M.matrix,M.cols,M.rows,1-M.column_major,P.permutation,P.dim,0,matrix),M.cols,P.dim,1-M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::TransposeMultiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,1,M.matrix,M.rows,M.cols,M.column_major,matrix),P.dim,M.cols,M.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::TransposeMultiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,1,v.vector,v.dim,1,1,vector),P.dim);
  return *this;
}
//===============================================================================
inline TDenseMatrix TransposeMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2)
{
  return TDenseMatrix(BaseMultiply(M1.matrix,M1.cols,M1.rows,1-M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL),M1.cols,M2.cols,1);
}
inline TDenseVector TransposeMultiply(const TDenseMatrix &M, const TDenseVector &v)
{
  return TDenseVector(BaseMultiply(M.matrix,M.cols,M.rows,1-M.column_major,v.vector,v.dim,1,1,(double*)NULL),M.cols);
}
inline TDenseMatrix TransposeMultiply(const TDenseMatrix &M, double s)
{
  return TDenseMatrix(BaseMultiply(s,M.matrix,M.rows*M.cols,(double*)NULL),M.cols,M.rows,1-M.column_major);
}
inline TPermutationMatrix TransposeMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,1,P2.permutation,P2.dim,0,(int*)NULL),P1.dim);
}
inline TDenseMatrix TransposeMultiply(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  return TDenseMatrix(BaseMultiply(M.matrix,M.cols,M.rows,1-M.column_major,P.permutation,P.dim,0,(double*)NULL),M.cols,P.dim,1-M.column_major);
}
inline TDenseMatrix TransposeMultiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(P.permutation,P.dim,1,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),P.dim,M.cols,M.column_major);
}
inline TDenseVector TransposeMultiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  return TDenseVector(BaseMultiply(P.permutation,P.dim,1,v.vector,v.dim,1,1,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseMatrix& TDenseMatrix::MultiplyInverse(const TDenseMatrix &M1, const TDenseMatrix &M2, int method)
{
  ShareMemory(BaseSolve(M2.matrix,M2.cols,M2.rows,1-M2.column_major,M1.matrix,M1.cols,M1.rows,1-M1.column_major,matrix,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),M1.rows,M2.rows,0);
  return *this;
}
inline TDenseVector& TDenseVector::MultiplyInverse(const TDenseVector &v, const TDenseMatrix &M, int method)
{
  ShareMemory(BaseSolve(M.matrix,M.cols,M.rows,1-M.column_major,v.vector,v.dim,1,1,vector,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),v.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyInverse(double s, const TDenseMatrix &M, int method)
{
  double *X=BaseInverse(M.matrix,M.rows,M.cols,M.column_major,matrix,method);
  ShareMemory(BaseMultiply(s,X,M.rows*M.cols,X),M.cols,M.rows,1);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::MultiplyInverse(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  ShareMemory(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,1,permutation),P1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyInverse(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,1,matrix),M.rows,P.dim,M.column_major);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::MultiplyInverse(const TPermutationMatrix &P, const TDenseMatrix &M, int method)
{
  TDenseMatrix X(P);
  ShareMemory(BaseSolve(M.matrix,M.cols,M.rows,1-M.column_major,X.matrix,X.cols,X.rows,1-X.column_major,matrix,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),P.dim,M.rows,0);
  return *this;
}
inline TDenseVector& TDenseVector::MultiplyInverse(const TDenseVector &v, const TPermutationMatrix &P)
{
  ShareMemory(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,1,vector),P.dim);
  return *this;
}
//===============================================================================
inline TDenseMatrix MultiplyInverse(const TDenseMatrix &M1, const TDenseMatrix &M2, int method)
{
  return TDenseMatrix(BaseSolve(M2.matrix,M2.cols,M2.rows,1-M2.column_major,M1.matrix,M1.cols,M1.rows,1-M1.column_major,(double*)NULL,
				(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),M1.rows,M2.rows,0);
}
inline TDenseVector MultiplyInverse(const TDenseVector &v, const TDenseMatrix &M, int method)
{
  return TDenseVector(BaseSolve(M.matrix,M.cols,M.rows,1-M.column_major,v.vector,v.dim,1,1,(double*)NULL,
					     (method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),v.dim);
}
inline TDenseMatrix MultiplyInverse(double s, const TDenseMatrix &M, int method)
{
  double *X=BaseInverse(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,method);
  return TDenseMatrix(BaseMultiply(s,X,M.rows*M.cols,X),M.cols,M.rows,1);
}
inline TPermutationMatrix MultiplyInverse(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,0,P2.permutation,P2.dim,1,(int*)NULL),P1.dim);
}
inline TDenseMatrix MultiplyInverse(const TDenseMatrix &M, const TPermutationMatrix &P)
{
  return TDenseMatrix(BaseMultiply(M.matrix,M.rows,M.cols,M.column_major,P.permutation,P.dim,1,(double*)NULL),M.rows,P.dim,M.column_major);
}
inline TDenseMatrix MultiplyInverse(const TPermutationMatrix &P, const TDenseMatrix &M, int method)
{
  TDenseMatrix X(P);
  return TDenseMatrix(BaseSolve(M.matrix,M.cols,M.rows,1-M.column_major,X.matrix,X.cols,X.rows,1-X.column_major,(double*)NULL,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),P.dim,M.rows,0);
}
inline TDenseVector MultiplyInverse(const TDenseVector &v, const TPermutationMatrix &P)
{
  return TDenseVector(BaseMultiply(v.vector,1,v.dim,1,P.permutation,P.dim,1,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseMatrix& TDenseMatrix::InverseMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2, int method)
{
  ShareMemory(BaseSolve(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix,method),M1.cols,M2.cols,1);
  return *this;
}
inline TDenseVector& TDenseVector::InverseMultiply(const TDenseMatrix &M, const TDenseVector &v, int method)
{
  ShareMemory(BaseSolve(M.matrix,M.rows,M.cols,M.column_major,v.vector,v.dim,1,1,vector,method),v.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InverseMultiply(const TDenseMatrix &M, double s, int method)
{
  double *X=BaseInverse(M.matrix,M.rows,M.cols,M.column_major,matrix,method);
  ShareMemory(BaseMultiply(s,X,M.rows*M.cols,X),M.cols,M.rows,1);
  return *this;
}
inline TPermutationMatrix& TPermutationMatrix::InverseMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  ShareMemory(BaseMultiply(P1.permutation,P1.dim,1,P2.permutation,P2.dim,0,permutation),P1.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InverseMultiply(const TDenseMatrix &M, const TPermutationMatrix &P, int method)
{
  TDenseMatrix X(P);
  ShareMemory(BaseSolve(M.matrix,M.rows,M.cols,M.column_major,X.matrix,X.rows,X.cols,X.column_major,matrix,method),M.cols,X.cols,1);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::InverseMultiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,1,M.matrix,M.rows,M.cols,M.column_major,matrix),P.dim,M.cols,M.column_major);
  return *this;
}
inline TDenseVector& TDenseVector::InverseMultiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  ShareMemory(BaseMultiply(P.permutation,P.dim,1,v.vector,v.dim,1,1,vector),P.dim);
  return *this;
}
//===============================================================================
inline TDenseMatrix InverseMultiply(const TDenseMatrix &M1, const TDenseMatrix &M2, int method)
{
  return TDenseMatrix(BaseSolve(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,(double*)NULL,method),M1.cols,M2.cols,1);
}
inline TDenseVector InverseMultiply(const TDenseMatrix &M, const TDenseVector &v, int method)
{
  return TDenseVector(BaseSolve(M.matrix,M.rows,M.cols,M.column_major,v.vector,v.dim,1,1,(double*)NULL,method),v.dim);
}
inline TDenseMatrix InverseMultiply(const TDenseMatrix &M, double s, int method)
{
  double *X=BaseInverse(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,method);
  return TDenseMatrix(BaseMultiply(s,X,M.rows*M.cols,X),M.cols,M.rows,1);
}
inline TPermutationMatrix InverseMultiply(const TPermutationMatrix &P1, const TPermutationMatrix &P2)
{
  return TPermutationMatrix(BaseMultiply(P1.permutation,P1.dim,1,P2.permutation,P2.dim,0,(int*)NULL),P1.dim);
}
inline TDenseMatrix InverseMultiply(const TDenseMatrix &M, const TPermutationMatrix &P, int method)
{
  TDenseMatrix X(P);
  return TDenseMatrix(BaseSolve(M.matrix,M.rows,M.cols,M.column_major,X.matrix,X.rows,X.cols,X.column_major,(double*)NULL,method),M.cols,X.cols,1);
}
inline TDenseMatrix InverseMultiply(const TPermutationMatrix &P, const TDenseMatrix &M)
{
  return TDenseMatrix(BaseMultiply(P.permutation,P.dim,1,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),P.dim,M.cols,M.column_major);
}
inline TDenseVector InverseMultiply(const TPermutationMatrix &P, const TDenseVector &v)
{
  return TDenseVector(BaseMultiply(P.permutation,P.dim,1,v.vector,v.dim,1,1,(double*)NULL),P.dim);
}
//===============================================================================
inline TDenseVector& TDenseVector::Kron(const TDenseVector &x, const TDenseVector &y)
{
  ShareMemory(BaseKron(x.vector,x.dim,1,true,y.vector,y.dim,1,true,vector),x.dim*y.dim);
  return *this;
}
inline TDenseMatrix& TDenseMatrix::Kron(const TDenseMatrix &X, const TDenseMatrix &Y)
{
  ShareMemory(BaseKron(X.matrix,X.rows,X.cols,X.column_major,Y.matrix,Y.rows,Y.cols,Y.column_major,matrix),X.rows*Y.rows,X.cols*Y.cols,Y.column_major);
  return *this;
}
inline TDenseVector Kron(const TDenseVector &x, const TDenseVector &y)
{
  return TDenseVector(BaseKron(x.vector,x.dim,1,true,y.vector,y.dim,1,true,(double*)NULL),x.dim*y.dim);
}
inline TDenseMatrix Kron(const TDenseMatrix &X, const TDenseMatrix &Y)
{
  return TDenseMatrix(BaseKron(X.matrix,X.rows,X.cols,X.column_major,Y.matrix,Y.rows,Y.cols,Y.column_major,(double*)NULL),X.rows*Y.rows,X.cols*Y.cols,Y.column_major);
}

//== Solving Linear Systems =====================================================
inline TDenseMatrix& TDenseMatrix::LeftSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  ShareMemory(BaseSolve(A.matrix,A.rows,A.cols,A.column_major,B.matrix,B.rows,B.cols,B.column_major,matrix,method),B.rows,B.cols,1);
  return *this; 
}

inline TDenseMatrix& TDenseMatrix::LeftSolve_Refine(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  ShareMemory(BaseSolve_Refine(A.matrix,A.rows,A.cols,A.column_major,B.matrix,B.rows,B.cols,B.column_major,matrix,method),B.rows,B.cols,1);
  return *this;
}



// ================= Added by HWu ===========================
inline TDenseMatrix &TDenseMatrix::LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseMatrix &B)
{
	TDenseMatrix A = LeftSolve_Cholesky(T, B); 
	ShareMemory(A.matrix, A.rows, A.cols, 1); 
	return *this; 
}
// ==========================================================

inline TDenseMatrix LeftSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  return TDenseMatrix(BaseSolve(A.matrix,A.rows,A.cols,A.column_major,B.matrix,B.rows,B.cols,B.column_major,(double*)NULL,method),B.rows,B.cols,1);
}

inline TDenseMatrix LeftSolve_Refine(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  return TDenseMatrix(BaseSolve_Refine(A.matrix,A.rows,A.cols,A.column_major,B.matrix,B.rows,B.cols,B.column_major,(double*)NULL,method),B.rows,B.cols,1);
}

inline TDenseVector& TDenseVector::LeftSolve(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  ShareMemory(BaseSolve(A.matrix,A.rows,A.cols,A.column_major,b.vector,b.dim,1,1,vector,method),b.dim);
  return *this;
}

inline TDenseVector& TDenseVector::LeftSolve_Refine(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  ShareMemory(BaseSolve_Refine(A.matrix,A.rows,A.cols,A.column_major,b.vector,b.dim,1,1,vector,method),b.dim);
  return *this;
}
// ================= Added by HWu ============================
inline TDenseVector &TDenseVector::LeftSolve_Cholesky(const TDenseMatrix &T, const TDenseVector &x)
{
	TDenseVector y = LeftSolve_Cholesky(T, x);
	ShareMemory(y.vector, y.dim);  
	return *this; 
}
// ===========================================================
inline TDenseVector LeftSolve(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  return TDenseVector(BaseSolve(A.matrix,A.rows,A.cols,A.column_major,b.vector,b.dim,1,1,(double*)NULL,method),b.dim);
}

inline TDenseVector LeftSolve_Refine(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  return TDenseVector(BaseSolve_Refine(A.matrix,A.rows,A.cols,A.column_major,b.vector,b.dim,1,1,(double*)NULL,method),b.dim);
}

inline TDenseMatrix& TDenseMatrix::RightSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  ShareMemory(BaseSolve(A.matrix,A.cols,A.rows,1-A.column_major,B.matrix,B.cols,B.rows,1-B.column_major,matrix,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),B.rows,B.cols,0);
  return *this;
}
inline TDenseMatrix RightSolve(const TDenseMatrix &A, const TDenseMatrix &B, int method)
{
  return TDenseMatrix(BaseSolve(A.matrix,A.cols,A.rows,1-A.column_major,B.matrix,B.cols,B.rows,1-B.column_major,(double*)NULL,
				(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),B.rows,B.cols,0);
}
inline TDenseVector& TDenseVector::RightSolve(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  ShareMemory(BaseSolve(A.matrix,A.cols,A.rows,1-A.column_major,b.vector,b.dim,1,1,vector,
			(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),b.dim);
  return *this;
}
inline TDenseVector RightSolve(const TDenseMatrix &A, const TDenseVector &b, int method)
{
  return TDenseVector(BaseSolve(A.matrix,A.cols,A.rows,1-A.column_major,b.vector,b.dim,1,1,(double*)NULL,
				(method == SOLVE_UPPER_TRIANGULAR) ? SOLVE_LOWER_TRIANGULAR : ((method == SOLVE_LOWER_TRIANGULAR) ? SOLVE_UPPER_TRIANGULAR : method)),b.dim);
}

//== Decompostions ==============================================================
inline void Cholesky(TDenseMatrix &T, const TDenseMatrix &M, int options)
{
  T.ShareMemory(BaseCholesky(M.matrix,M.rows,M.cols,M.column_major,T.matrix,options),M.rows,M.cols,1);
}
inline TDenseMatrix Cholesky(const TDenseMatrix &M, int options)
{
  return TDenseMatrix(BaseCholesky(M.matrix,M.rows,M.cols,M.column_major,(double*)NULL,options),M.rows,M.cols,1);
}

//== Miscellaneous Routines =====================================================
inline double TDenseVector::Norm(void) const
{
  return ::Norm(*this);
}
inline double TDenseMatrix::Norm(void) const
{
  return ::Norm(*this);
}
inline double TDenseMatrix::MatrixNorm(void) const
{
  return ::MatrixNorm(*this);
}

inline double TDenseMatrix::LogAbsDeterminant() const
{
	return ::LogAbsDeterminant(*this); 
}

inline double TDenseMatrix::LogAbsDeterminant_Cholesky(const TDenseMatrix &T) const
{
	return ::LogAbsDeterminant_Cholesky(T);
}

inline double TDenseMatrix::Determinant() const
{
	return ::Determinant(*this); 
}

inline double TDenseMatrix::Determinant_Cholesky(const TDenseMatrix &T) const
{
	return ::Determinant_Cholesky(T); 
}

inline int TDenseMatrix::Rank() const
{
	return ::Rank(*this); 
}

inline double TDenseMatrix::RCond() const
{
	return ::RCond(*this); 
}

inline bool TDenseMatrix:: IsZeroMatrix() const
{
	return ::IsZeroMatrix(*this); 
}

//== Random Matrices ============================================================
inline TDenseVector& TDenseVector::RandomNormal(int d) 
{ 
  UniqueMemory(d);
  return RandomNormal();
}

inline TDenseVector RandomNormalVector(int d)
{
  TDenseVector v(d);
  return v.RandomNormal();
}

inline TDenseVector& TDenseVector::RandomUniform(int d)
{ 
  UniqueMemory(d);
  return RandomUniform();
}

inline TDenseVector RandomUniformVector(int d)
{
  TDenseVector v(d);
  return v.RandomUniform();
}

inline TDenseMatrix& TDenseMatrix::RandomNormal(int r, int c)
{
  UniqueMemory(r,c,1);
  return RandomNormal();
};

inline TDenseMatrix RandomNormalMatrix(int r, int c)
{
  TDenseMatrix X(r,c);
  return X.RandomNormal();
}

inline TDenseMatrix& TDenseMatrix::RandomUniform(int r, int c) 
{
  UniqueMemory(r,c,1);
  return RandomUniform();
}

inline TDenseMatrix RandomUniformMatrix(int r, int c)
{
  TDenseMatrix X(r,c);
  return X.RandomUniform();
}

inline TPermutationMatrix& TPermutationMatrix::RandomUniform(int d)
{
  UniqueMemory(d);
  return RandomUniform();
}

inline TPermutationMatrix RandomUniformPermutationMatrix(int d)
{
  TPermutationMatrix P(d);
  return P.RandomUniform();
}

//== Sort =======================================================================


//== Permutations ===============================================================
inline TDenseMatrix TPermutationMatrix::DenseMatrix(void)
{
  return TDenseMatrix(*this);
}


//===============================================================================
//== Complex matrices and vectors
//===============================================================================
//-- TDenseMatrixComplex Constructors/Destructors -------------------------------
inline TDenseMatrixComplex::~TDenseMatrixComplex() 
{
  FreeSharedMemory_double(matrix);
}

/*
   The array buffer MUST be part of the shared memory system, the dimension of
   the shared memory is 2*r*c with both r and c non-negative.
*/
inline TDenseMatrixComplex::TDenseMatrixComplex(double *buffer, int r, int c, bool col_major)
{
  IncrementSharedMemory_double(buffer);
  matrix=buffer;
  rows=r;
  cols=c;
  column_major=col_major;
}

inline TDenseMatrixComplex::TDenseMatrixComplex(void)
{
  matrix=(double*)NULL;
  rows=cols=0;
  column_major=1;
}

inline TDenseMatrixComplex::TDenseMatrixComplex(int r, int c)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrixComplex::TDenseMatrixComplex(int,int): negative index");
  matrix=AllocateSharedMemory_double(2*r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=1;
}

inline TDenseMatrixComplex::TDenseMatrixComplex(int r, int c, bool col_major)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrixComplex::TDenseMatrixComplex::TDenseMatrix(int,int,int): negative index");
  matrix=AllocateSharedMemory_double(2*r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
}

inline TDenseMatrixComplex::TDenseMatrixComplex(int r, int c, const std::complex<double> &s)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrixComplex::TDenseMatrixComplex(int,int,double): negative index");
  matrix=AllocateSharedMemory_double(2*r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=1;
  for (int i=2*rows*cols-1; i >= 0; matrix[i--]=s.imag(), matrix[i--]=s.real());
}

inline TDenseMatrixComplex::TDenseMatrixComplex(int r, int c, bool col_major, const std::complex<double> &s)
{
  if ((r < 0) || (c < 0)) throw dw_exception("TDenseMatrixComplex::TDenseMatrixComplex(int,int,double): negative index");
  matrix=AllocateSharedMemory_double(2*r*c);
  IncrementSharedMemory_double(matrix);
  rows=r;
  cols=c;
  column_major=col_major;
  for (int i=2*rows*cols-1; i >= 0; matrix[i--]=s.imag(), matrix[i--]=s.real());
}

inline TDenseMatrixComplex::TDenseMatrixComplex(const TDenseMatrixComplex &M)
{
  IncrementSharedMemory_double(M.matrix);
  matrix=M.matrix;
  rows=M.rows;
  cols=M.cols;
  column_major=M.column_major;
};

inline TDenseMatrixComplex::TDenseMatrixComplex(const TDenseMatrix &Re)
{
  matrix=AllocateSharedMemory_double(2*Re.rows*Re.cols);
  IncrementSharedMemory_double(matrix);
  rows=Re.rows;
  cols=Re.cols;
  column_major=Re.column_major;
  for (int i=rows*cols-1, j=2*rows*cols-1; i >= 0; matrix[j--]=0.0, matrix[j--]=Re.matrix[i--]);
}


inline void TDenseMatrixComplex::ShareMemory(double *buffer, int r, int c, bool col_major) 
{
  if (buffer != matrix)
    {
      FreeSharedMemory_double(matrix);
      IncrementSharedMemory_double(buffer);
      matrix=buffer;
    }
  rows=r;
  cols=c;
  column_major=col_major;
}

inline void TDenseMatrixComplex::UniqueMemory(void)
{
  matrix=UniqueSharedMemory_double(matrix);
}

inline TDenseMatrixComplex& TDenseMatrixComplex::operator=(const TDenseMatrixComplex &M)
{
  ShareMemory(M.matrix,M.rows,M.cols,M.column_major);
  return *this;
}

inline int TDenseMatrixComplex::Index(int r, int c) const
{
  return (column_major) ? 2*(c*rows+r) : 2*(r*cols+c);
}

inline std::complex<double> TDenseMatrixComplex::operator()(int r, int c) const
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrixComplex::operator() - index out of range");
  int k=Index(r,c);
  return std::complex<double>(matrix[k],matrix[k+1]);
}

inline double TDenseMatrixComplex::Real(int r, int c) const
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrixComplex::operator() - index out of range");
  return matrix[Index(r,c)];
}

inline double& TDenseMatrixComplex::Real(int r, int c)
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrixComplex::operator() - index out of range");
  return matrix[Index(r,c)];
}

inline double TDenseMatrixComplex::Imag(int r, int c) const
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrixComplex::operator() - index out of range");
  return matrix[Index(r,c)+1];
}

inline double& TDenseMatrixComplex::Imag(int r, int c)
{
  if ((r < 0) || (r >= rows) || (c < 0) || (c >= cols)) throw dw_exception("TDenseMatrixComplex::operator() - index out of range");
  return matrix[Index(r,c)+1];
}

inline TDenseMatrixComplex TDenseMatrixComplex::operator*(const TDenseMatrixComplex &M) 
{ 
  return TDenseMatrixComplex(BaseMultiplyComplexComplex(matrix,rows,cols,column_major,M.matrix,M.rows,M.cols,M.column_major,(double*)NULL),rows,M.cols,1);
}

inline TDenseMatrixComplex& TDenseMatrixComplex::InverseMultiply(const TDenseMatrixComplex &M1, const TDenseMatrixComplex &M2, int method)
{
  ShareMemory(BaseSolveComplex(M1.matrix,M1.rows,M1.cols,M1.column_major,M2.matrix,M2.rows,M2.cols,M2.column_major,matrix,method),M1.cols,M2.cols,1);
  return *this;
}

inline TDenseMatrixComplex& TDenseMatrixComplex::RandomNormal(int r, int c)
{
  UniqueMemory(r,c,true);
  return RandomNormal();
};

inline TDenseMatrixComplex& TDenseMatrixComplex::RandomUniform(int r, int c)
{
  UniqueMemory(r,c,true);
  return RandomUniform();
};

inline TDenseMatrixComplex RandomNormalComplexMatrix(int r, int c)
{
  TDenseMatrixComplex X(r,c);
  return X.RandomNormal();
}

inline TDenseMatrixComplex RandomUniformComplexMatrix(int r, int c)
{
  TDenseMatrixComplex X(r,c);
  return X.RandomNormal();
}

//===============================================================================
//===============================================================================
//===============================================================================

#endif
