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

#ifndef __DW_RANDOM__
#define __DW_RANDOM__

#ifdef __cplusplus
extern "C"
{
#endif

#include "prcsn.h"
#include "dw_math.h"
#include <stdio.h>

#include "gsl/gsl_rng.h"

  void dw_set_gsl_uniform_type(const gsl_rng_type* gsl_uniform_rng_type);
  const char* dw_get_gsl_uniform_name(void);

  void dw_initialize_generator(int init);

  void* dw_get_generator_state(void);
  int dw_get_generator_state_size(void);
  void dw_set_generator_state(void *state);
  void dw_print_generator_state(FILE *f);
  void dw_read_generator_state(FILE *f);

  // =========== added by HW for uniform integers ====================
  // 	Returns a random integer from 0 to n-1 inclusive. 
  unsigned long int dw_uniform_int(unsigned long int n);
  void dw_random_shuffle(void *base, size_t n, size_t size); 
  double dw_cauchy(double a); 
  double dw_log_cauchy(double x, double a); 
  // =================================================================

  PRECISION dw_uniform_rnd(void);
  PRECISION dw_gaussian_rnd(void);
  PRECISION dw_lognormal_rnd(PRECISION mean, PRECISION standard_deviation);
  PRECISION dw_gamma_rnd(PRECISION a);
  PRECISION dw_truncated_gaussian_rnd(PRECISION a, PRECISION b);
  double dw_univariate_wishard_rnd(double T, double c);
  double dw_tdistribution_rnd(double nu);
  double dw_tdistribution_pdf(double x, double nu);

  PRECISION dw_normal_cdf(PRECISION x);
  PRECISION dw_normal_pdf(PRECISION x);
  PRECISION dw_chi_square_cdf(PRECISION x, int df);
  PRECISION dw_chi_square_invcdf(PRECISION p, int df);
  PRECISION dw_binomial_cdf(int x, PRECISION p, int n);
  int dw_binomial_invcdf(PRECISION p, PRECISION q, int n);

#ifdef __cplusplus
}
#endif

#endif
