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

#ifndef __RANDOM_MATRIX__
#define __RANDOM_MATRIX__

#include "dw_matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Random Matrices and Vectors */
TVector dw_UniformVector(TVector x);
TMatrix dw_UniformMatrix(TMatrix X);
TVector dw_NormalVector(TVector x);
TMatrix dw_NormalMatrix(TMatrix X);
TVector dw_TruncatedNormalVector(TVector x, TMatrix R, TVector al, TVector au);
TVector dw_LogNormalVector(TVector x, PRECISION mean, PRECISION standard_deviation);
TMatrix dw_GammaMatrix(TMatrix X, TMatrix A, TMatrix B);
TMatrix dw_Wishart(TMatrix X, TMatrix S, int nu);
TVector dw_StudentT(TVector x, TMatrix T, int nu);
TMatrix dw_UniformOrthogonal(TMatrix Q);
TVector dw_UniformUnitSphere(TVector x);
TVector dw_UniformUnitBall(TVector x);

#ifdef __cplusplus
}
#endif

#endif
