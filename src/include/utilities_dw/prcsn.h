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

/*
   Defines the precision to be used.  Default is double precision.  To use single precision
   define DW_SINGLE in the compiler flags.
*/

#ifndef __PRECISION_H__
#define __PRECISION_H__

#include <float.h>

#ifdef DW_SINGLE

/********** single precision **********/
#define PRECISION             float
#define MACHINE_EPSILON       5.97E-08
#define SQRT_MACHINE_EPSILON  2.45E-04
#define PRECISION_SIZE        4
#define PRECISION_SHIFT       2
#define PRECISION_WORD        dword
#define MINUS_INFINITY       -FLT_MAX
#define PLUS_INFINITY         FLT_MAX

#else

/********** double precision **********/
#define PRECISION              double
#define MACHINE_EPSILON        1.11E-16
#define SQRT_MACHINE_EPSILON   1.06E-08
#define PRECISION_SIZE         8
#define PRECISION_SHIFT        3
#define PRECISION_WORD         qword

/* Redefined to make compatible with Tao's conventions * 
#define MINUS_INFINITY       -DBL_MAX  
#define PLUS_INFINITY         DBL_MAX
**/
#define MINUS_INFINITY        -1.0E300
#define PLUS_INFINITY          1.0E300


#endif

#endif
