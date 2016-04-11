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

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <memory.h>
#include <limits.h>
#include <unistd.h>
#include "prcsn.h"
#include "dw_rand.h"
#include "dw_std.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_sf_gamma.h"

/*******************************************************************************/
/*************************** Uniform Random Numbers ****************************/
/*******************************************************************************/
/* 
   Flag controling which uniform random number to choose
*/
static const gsl_rng_type* GSL_UNIFORM_RNG_TYPE=(gsl_rng_type *)NULL; //gsl_rng_mt19937;
static gsl_rng* GSL_UNIFORM_RNG=(gsl_rng*)NULL;

/*
   Sets the random number generator type
*/
void dw_set_gsl_uniform_type(const gsl_rng_type* gsl_uniform_rng_type)
{
  GSL_UNIFORM_RNG_TYPE=(gsl_uniform_rng_type) ? gsl_uniform_rng_type : gsl_rng_mt19937;
  if (GSL_UNIFORM_RNG) gsl_rng_free(GSL_UNIFORM_RNG);
  GSL_UNIFORM_RNG=gsl_rng_alloc(GSL_UNIFORM_RNG_TYPE);
}

/*
   Gets the random number generator name
*/
const char* dw_get_gsl_uniform_name(void)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return gsl_rng_name(GSL_UNIFORM_RNG);
}

/*
   Initializes seed value for uniform random number generator.  The seed value 
   can be any integer.  A value of 0 will initialize the seed from the system
   clock and the process pid.
*/
void dw_initialize_generator(int init)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  if (init == 0)
    {
      gsl_rng_set(GSL_UNIFORM_RNG,(unsigned long int)time((time_t *)NULL)*(unsigned long int)clock() + (unsigned long int)getpid());
      init=(int)gsl_rng_get(GSL_UNIFORM_RNG);
    }
  gsl_rng_set(GSL_UNIFORM_RNG,(unsigned long int)init);
}

/*
   Allocates memory and saves the state of the random number generator.  The
   calling routine is responsible for freeing the returned memory.
*/
void* dw_get_generator_state(void)
{
  int size;
  void *state, *rtrn;
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  size=gsl_rng_size(GSL_UNIFORM_RNG);
  state=gsl_rng_state(GSL_UNIFORM_RNG);
  rtrn=(void*)dw_malloc(size);
  memcpy(rtrn,state,size);
  return rtrn;
}

/*
   Returns the size in bytes of the void pointer returned by 
   dw_get_generator_state().
*/
int dw_get_generator_state_size(void)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return gsl_rng_size(GSL_UNIFORM_RNG);
}

/*
   Sets the state of the random number generator.  The void pointer must have
   been obtained via a call to dw_get_generator_state() and the generator
   type must not have been changed between calls!
*/
void dw_set_generator_state(void *new_state)
{
  int size;
  void *state;
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  size=gsl_rng_size(GSL_UNIFORM_RNG);
  state=gsl_rng_state(GSL_UNIFORM_RNG);
  memcpy(state,new_state,size);
}

void dw_print_generator_state(FILE *f)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  gsl_rng_fwrite(f,GSL_UNIFORM_RNG);
}

void dw_read_generator_state(FILE *f)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  gsl_rng_fread(f,GSL_UNIFORM_RNG);
}

/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

// =========== added by HW for uniform integers ====================
// 	Returns a random integer from 0 to n-1 inclusive. 
unsigned long int dw_uniform_int(unsigned long int n)
{
	if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
	return gsl_rng_uniform_int(GSL_UNIFORM_RNG, n); 
}

// commented out by DW - warning has become error!
/* void dw_random_shuffle(void *base, size_t n, size_t size) */
/* { */
/* 	if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE); */
/* 	gsl_ran_shuffle(GSL_UNIFORM_RNG_TYPE, base, n, size);  */
/* } */

double dw_cauchy(double a)
{
	if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
	return gsl_ran_cauchy(GSL_UNIFORM_RNG, a); 
}

double dw_log_cauchy(double x, double a)
{
	return log(gsl_ran_cauchy_pdf(x,a)); 
}

// // =================================================================

/*
   Generates a uniform (0,1) deviate.
*/
PRECISION dw_uniform_rnd(void)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return gsl_rng_uniform_pos(GSL_UNIFORM_RNG);
}


/*
   Returns a standard gaussian deviate.  The density function for the
   standard gaussian is

                          1
                     ----------- exp(-0.5*x^2)
                      sqrt(2*Pi)

*/
PRECISION dw_gaussian_rnd(void)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return (PRECISION)gsl_ran_ugaussian(GSL_UNIFORM_RNG);
}

/*
   Returns a standard truncated gaussian deviate.  Usually, a <= b, but if this 
   is not the case, then the values of a and b will be swapped.

   Based on ideas in Robert's "Simulation of Truncated Normal Variables,"
   Statistics and Computing, June 1995.  (There are be eariler cites).

   If a >= 0 or b <= 0, use rejection method with an exponential proposal.  The 
   exponential family is

           g(x) = alpha*exp(-alpha*(x-a))/(1-exp(-alpha*(b-a)))

   If a >= 0, alpha is chosen to be (a+sqrt(a^2+4))/2 if this number is less than
   or equal to b and (a+b)/2 otherwise.  Emperical evidence indicates that the 
   worst case is a=0, b=infinity with the probability acceptance equal to 0.7602.
   The case b <= 0 is similar. 

   If a < 0 < b, and b - a < sqrt(2*pi), use rejection method with uniform 
   proposal.  Worst case: a=0, b=sqrt(2*pi) with probability of acceptance equal
   to 0.4939.  If a=0, then the exponential proposal would be used, but if a
   is slightly less than 0, then the acceptance probability will be slightly more
   than 0.4939.

   If a < 0 < b, and b - a >= sqrt(2*pi), draw from gaussian until the constraint 
   is satisfied.  Worst case: a=0, b=sqrt(2*pi) with probability of acceptance
   equal to 0.4939.  If a=0, then the exponential proposal would be used, but if 
   a is slightly less than 0, then the acceptance probability will be slightly 
   more than 0.4939.
*/
PRECISION dw_truncated_gaussian_rnd(PRECISION a, PRECISION b)
{
  PRECISION alpha, c, x;

  if (a == b) return a;

  if (a > b)
    {
      x=a;
      a=b;
      b=x;
    }

  if (a >= 0)
    {
      alpha=0.5*(a+sqrt(a*a+4.0));
      if (alpha >= (x=0.5*(a+b))) alpha=x;
      c=1-exp(-alpha*(b-a));
      x=a-log(1-c*dw_uniform_rnd())/alpha;
      while (dw_uniform_rnd() > exp(-0.5*(x-alpha)*(x-alpha)))
	x=a-log(1-c*dw_uniform_rnd())/alpha;
      return x;
    }

  if (b <= 0)
    {
      alpha=0.5*(-b+sqrt(b*b+4));
      if (alpha >= (x=-0.5*(a+b))) alpha=x;
      c=1-exp(-alpha*(b-a));
      x=-b-log(1-c*dw_uniform_rnd())/alpha;
      while (dw_uniform_rnd() > exp(-0.5*(x-alpha)*(x-alpha)))
	x=-b-log(1-c*dw_uniform_rnd())/alpha;
      return -x;
    }

  if (b-a < 2.506628274631000)   // sqrt(2*pi)
    {
      x=a+(b-a)*dw_uniform_rnd();
      while (dw_uniform_rnd() > exp(-x*x/2))
	x=a+(b-a)*dw_uniform_rnd();
      return x;
    }

  x=dw_gaussian_rnd();
  while ((x < a) || (x > b))
    x=dw_gaussian_rnd();
  return x;
}

/*
   Returns a t-distribution deviate with nu degress of freedom.
*/
double dw_tdistribution_rnd(double nu)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return gsl_ran_tdist(GSL_UNIFORM_RNG, nu); 
}

double dw_tdistribution_pdf(double x, double nu)
{
  return gsl_ran_tdist_pdf(x,nu);
}


/*
   Returns a standard gamma deviate.  The density function for a standard gamma
   distribution is

                                           x^(a-1)*exp(-x)
                   gamma_density(x;a) =   ----------------
                                              gamma(a)

   for a > 0.  The function gamma(a) is the integral with from 0 to infinity of 
   exp(-t)*t^(a-1).

   A general gamma variate can be obtained as follows.  Let z=b*x.  Then,
   z is drawn from a general gamma distribution whose density is

                                        z^(a-1)*exp(-z/b)
                gamma_density(z;a,b) = ------------------
                                          gamma(a)*b^a

   Notes:
    Does not check if a > 0.
*/
PRECISION dw_gamma_rnd(PRECISION a)
{
  if (!GSL_UNIFORM_RNG) dw_set_gsl_uniform_type(GSL_UNIFORM_RNG_TYPE);
  return (PRECISION)gsl_ran_gamma(GSL_UNIFORM_RNG,a,1.0);
}

/*
   Returns a deviate drawn from the distribution with density function
   proportional to

                   f(x) = (|x|^T)*exp(-0.5*T*(x-c)^2)

   Uses rejection method with

                          w(0)                  w(1)
          g(x) = -------------------- + --------------------
                  1 + ((x-b(0))/a)^2     1 + ((x-b(1))/a)^2

   where a=sqrt(2/T), b(0)=(c+sqrt(c^2+4))/2, b(1)=(c-sqrt(c^2+4))/2,
   and w(i)=f(b(i)).  Drawing from the distribution with density proportional 
   to g can be accomplished by drawing from the distribution with density
   given by

                                     1
                  h(x) = --------------------------
                          a*pi(1 + ((x-b(i))/a)^2)

  with probability w(i)/(w(0)+w(1)).  If u is a uniformly distributed random
  variable on (0,1), then b(i)+a*tan(PI*u) has density equal to h.

  Note:  If u = sqrt(T)x, then the density of u is proportional to

               f(x) = (|u|^T)*exp(-0.5*(u-sqrt(T)c)^2)

*/
#define PI 3.14159265358979323846
double dw_univariate_wishard_rnd(double T, double c)
{
 double a, b0, b1, log_w0, w1_w0, b0_b1, w, x, t, s;

 a=sqrt(2.0/T);

 b0_b1=sqrt(c*c+4);
 if (c < 0)
   {
    b0=0.5*(c-b0_b1);
    b1=0.5*(c+b0_b1);
    b0_b1/=-a;                                      // (b0-b1)/a
   }
  else
   {
    b0=0.5*(c+b0_b1);
    b1=0.5*(c-b0_b1);
    b0_b1/=a;                                       // (b0-b1)/a
   }

 log_w0=T*(log(fabs(b0))-0.5*b1*b1);                // log(w0)=log(f(b0))

 w1_w0=exp(T*(log(fabs(b1))-0.5*b0*b0)-log_w0);     // w1/w0=f(b1)/f(b0)

 w=1.0/(1.0+w1_w0);                                 // w0/(w0+w1)=f(b0)/(f(b0)+f(b1));

 do
  {
   s=tan(PI*dw_uniform_rnd());
   if (dw_uniform_rnd() < w)
     {
      x=b0+a*s;
      t=s+b0_b1;                                    // (x-b1)/a
      s=1.0/(1.0+s*s)+w1_w0/(1.0+t*t);
     }
    else
     {
      x=b1+a*s;
      t=s-b0_b1;                                    // (x-b0)/a
      s=1.0/(1.0+t*t)+w1_w0/(1.0+s*s);
     }
   t=x-c;
  }
 while (dw_uniform_rnd()*s > exp(T*(log(fabs(x))-0.5*t*t)-log_w0));

 return x;
}
#undef PI

/*
   Returns a lognormal deviate.  The mean and standard deviations of the 
   underlying normal distributions are passed.
*/
PRECISION dw_lognormal_rnd(PRECISION mean, PRECISION standard_deviation)
{
  return (PRECISION)exp(standard_deviation * dw_gaussian_rnd() + mean);
}


/*
   Returns the integral from -infinity to x of 1/sqrt(2*PI)*exp(-y^2/2).
*/
PRECISION dw_normal_cdf(PRECISION x)
{
  return (PRECISION)gsl_cdf_ugaussian_P(x);
}

PRECISION dw_normal_pdf(PRECISION x)
{
  return (PRECISION)gsl_ran_ugaussian_pdf(x);
}


PRECISION dw_chi_square_cdf(PRECISION x, int df)
{
  return (PRECISION)gsl_cdf_chisq_P(x,df);
}


PRECISION dw_chi_square_invcdf(PRECISION p, int df)
{
  return (PRECISION)gsl_cdf_chisq_Pinv(p,df);
}

PRECISION dw_binomial_cdf(int x, PRECISION q, int n)
{
  return (PRECISION)gsl_cdf_binomial_P(x,q,n);
}

/*
   Returns the integer x such that dw_binominal_cdf(x,p,n) is closest to p.
*/
int dw_binominal_invcdf(PRECISION p, PRECISION q, int n)
{
  int min=0, max=n, mid=n/2;
  PRECISION pmin=0.0, pmax=1.0, pmid=dw_binomial_cdf(mid,p,n);
  if (p >= 1.0) return n;
  if (p <= 0) return 0;
  while (1)
    {
      if (p > mid)
	{
	  min=mid;
	  pmin=pmid;
	}
      else
	{
	  max=mid;
	  pmax=pmid;
	}
      if (max - min <= 1)
	return (pmax + pmin > 2.0*p) ? min : max;
      mid=(max+min)/2;
      pmid=dw_binomial_cdf(mid,p,n);
    }
}
