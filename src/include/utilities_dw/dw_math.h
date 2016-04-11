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

#ifndef __DW_MATH_ROUTINES__
#define __DW_MATH_ROUTINES__

#include "prcsn.h"

#ifdef __cplusplus
extern "C"
{
#endif

  PRECISION AddLogs(PRECISION a, PRECISION b);
  PRECISION AddScaledLogs(PRECISION x, PRECISION a, PRECISION y, PRECISION b);

  int dw_IsFloat(const char *buffer);
  int dw_IsInteger(const char *buffer);

  PRECISION dw_log_gamma(PRECISION x);
  PRECISION dw_beta(PRECISION a, PRECISION b);
  PRECISION dw_incomplete_beta(PRECISION x, PRECISION a, PRECISION b);

#ifdef __cplusplus
}
#endif

#endif
