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

#ifndef __DW_STANDARD_DEFINES__
#define __DW_STANDARD_DEFINES__

#if defined(MATLAB_MEX_FILE) || defined(OCTAVE_MEX_FILE)

#include "modify_for_mex.h"

#else

#include <stdlib.h>

#define dw_malloc  malloc
#define dw_calloc  calloc
#define dw_realloc  realloc
#define dw_free  free

#define dw_exit  exit

#define blas_int  int
#define lapack_int  int

#endif

#endif
