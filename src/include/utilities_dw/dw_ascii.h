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

#ifndef __DW_ASCII_ROUTINES__
#define __DW_ASCII_ROUTINES__

#include <stdio.h>
#include "prcsn.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Flag codes.  See ParseDelimitedString() for explanation.
#define REMOVE_EMPTY_FIELDS           0x00000001
#define ALLOW_QUOTED_TEXT             0x00000002
#define STRIP_LEADING_WHITESPACE      0x00000004
#define STRIP_TRAILING_WHITESPACE     0x00000008
#define STRIP_WHITESPACE              0x0000000c

FILE *dw_OpenTextFile(char *filename);
FILE *dw_CreateTextFile(char *filename);
FILE *dw_AppendTextFile(char *filename);

int dw_SetFilePosition(FILE *f, char *id);
int dw_SetFilePositionNoRewind(FILE *f, char *id);
//int dw_SetFilePositionBySection(FILE *f, int n, ...);

int dw_NumberLines(FILE *f, char* filename);
char* dw_ReadLine(FILE *f, char *buffer, int *n);
void dw_NextLine(FILE *f, int n);
char** dw_ParseDelimitedString(char *buffer, char delimiter, int flag);
char** dw_ReadDelimitedLine(FILE *f, char delimiter, int flag);
char*** dw_ReadDelimitedFile(FILE *f, char* filename, char delimiter, int flag);
int dw_PrintDelimitedArray(FILE *f, void* array, char delimiter);

PRECISION* dw_ReadDelimitedLine_floating(FILE *f, char delimiter, int flag, PRECISION invalid_identifier);
PRECISION** dw_ReadDelimitedFile_floating(FILE *f, char* filename, char delimiter, int flag, PRECISION invalid_identifier);

//int dw_ReadDelimitedField(FILE *f, char **buffer, int *n);
//int dw_ReadDelimitedField(FILE *f, int delimiter, int terminal, int flag, char **buffer, int *n);

char* dw_DuplicateString(char *buffer);

#ifdef __cplusplus
}
#endif

#endif
