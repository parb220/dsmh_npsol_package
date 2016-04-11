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

#ifndef __ERROR_HANDLING__
#define __ERROR_HANDLING__

#include <stdio.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define NO_ERR                0x00000000
#define ALL_ERRORS            0x000F0FFF

//=== General Errors ===                        Default message
#define MEM_ERR               0x00000001     // Out of memory.
#define FILE_ERR              0x00000002     // File operation error.
#define PARSE_ERR             0x00000004     // Error parsing data.
#define FLOAT_ERR             0x00000008     // Floating point error.
#define NULL_ERR              0x00000010     // Unexpected null pointer encountered.
#define ARG_ERR               0x00000020     // Argument error.
#define ITERATION_ERR         0x00000040     // Maximum iteration limit exceeded.
#define USER_ERR              0x00000080     // Undocumented error.
#define NOT_IMPLEMENTED_ERR   0x00000100     // Feature not yet implemented.
#define UNKNOWN_ERR           0x00000200     // Unknown error.
#define FILE_OPEN_ERR         0x00000400     // Unable to open/create file.
#define FILE_READ_WRITE_ERR   0x00000800     // Unable to read/write to file.

//=== Matrix Errors ===
#define SIZE_ERR              0x00010000     // Matrices/vectors not conformable.
#define SING_ERR              0x00020000     // Singular matrix.
#define POSDEF_ERR            0x00040000     // Matrix not positive definite.
#define BLAS_LAPACK_ERR       0x00080000     // Blas/Lapack error.

//=== Error Routines ===
int dw_GetError(void);
char* dw_GetErrorMessage(void);
int dw_ClearError(void);
int dw_SetError(int err, char *msg);
int dw_Error(int err);
int dw_UserError(char *msg);
int dw_FileError(int err, char *filename);
int dw_SetVerboseErrors(int errors);
int dw_GetVerboseErrors(void);
int dw_SetTerminalErrors(int errors);
int dw_GetTerminalErrors(void);
int dw_CreateErrorMessageFile(char *filename);
int dw_AppendErrorMessageFile(char *filename);
void dw_ConsoleErrorMessage(void);
FILE* dw_GetErrorFilePointer(void);
char* dw_GetErrorFilename(void);

#ifdef __cplusplus
}
#endif

#endif
