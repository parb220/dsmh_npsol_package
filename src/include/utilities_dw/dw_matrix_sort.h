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

#ifndef __SORT_MATRICES__
#define __SORT_MATRICES__

#include "dw_matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

TVector SortVectorAscending(TVector x, TVector y);
TVector SortVectorDescending(TVector x, TVector y);
TMatrix SortMatrixRowsAscending(TMatrix X, TMatrix Y, int j);
TMatrix SortMatrixRowsDescending(TMatrix X, TMatrix Y, int j);
TMatrix SortMatrixColumnsAscending(TMatrix X, TMatrix Y, int i);
TMatrix SortMatrixColumnsDescending(TMatrix X, TMatrix Y, int i);

#ifdef __cplusplus
}
#endif

#endif
