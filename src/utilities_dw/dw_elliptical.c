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

#include "dw_elliptical.h"
#include "dw_rand.h"
#include "dw_matrix_rand.h"
#include "dw_error.h"
#include "dw_ascii.h"
#include "dw_std.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979324
static PRECISION logdensity_draw(PRECISION *draw, TElliptical *elliptical);

void FreeElliptical(TElliptical *elliptical)
{
  if (elliptical)
    {
      FreeVector(elliptical->center);
      FreeMatrix(elliptical->scale);
      FreeMatrix(elliptical->quadratic_form);
      FreeVector(elliptical->draw);

      if (elliptical->data && elliptical->free) elliptical->free(elliptical->data);

      dw_free(elliptical);
    }
}

static TElliptical* CreateElliptical(int dim, TVector center, TMatrix scale)
{
 TElliptical* elliptical;

  if ((dim <= 0) || (center && DimV(center) != dim) || (scale && ((RowM(scale) !=  dim) || (ColM(scale) != dim))))
    {
      dw_Error(SIZE_ERR);
      return (TElliptical*)NULL;
    }

  elliptical=(TElliptical*)dw_malloc(sizeof(TElliptical));

  elliptical->dim=dim;
  elliptical->center=center ? EquateVector((TVector)NULL,center) : InitializeVector(CreateVector(dim),0.0);
  elliptical->scale=scale ? EquateMatrix((TMatrix)NULL,scale) : IdentityMatrix((TMatrix)NULL,dim);
  elliptical->draw=CreateVector(dim);

  elliptical->quadratic_form=ProductTransposeMM((TMatrix)NULL,scale,scale);
  Inverse_LU(elliptical->quadratic_form,elliptical->quadratic_form);
  ForceSymmetric(elliptical->quadratic_form);

  elliptical->logabsdet=-LogAbsDeterminant_LU(elliptical->scale);

  elliptical->data=(void*)NULL;
  elliptical->type=(char*)NULL;
  elliptical->logdensity_radius=NULL;
  elliptical->logdensity_draw=NULL;
  elliptical->cummulative_radius=NULL;
  elliptical->draw_vector=NULL;
  elliptical->print_info=NULL;
  elliptical->free=NULL;

  return elliptical;
}

PRECISION RadiusElliptical(PRECISION *draw, TElliptical *elliptical)
{
  memcpy(pElementV(elliptical->draw),draw,elliptical->dim*sizeof(PRECISION));
  SubtractVV(elliptical->draw,elliptical->draw,elliptical->center);
  return sqrt(InnerProduct(elliptical->draw,elliptical->draw,elliptical->quadratic_form));
}

void PrintEllipticalInfo_full(FILE *f_out, TElliptical *elliptical)
{
  PrintEllipticalInfo(f_out,elliptical);
  fprintf(f_out,"Variance scale: %lg\n",elliptical->variance_scale);
  fprintf(f_out,"Center:\n");
  dw_PrintVector(f_out,elliptical->center,"%lg ");
  fprintf(f_out,"Scale:\n");
  dw_PrintMatrix(f_out,elliptical->scale,"%lg ");
}

static PRECISION logdensity_draw(PRECISION *draw, TElliptical *elliptical)
{
  return elliptical->logdensity_radius(RadiusElliptical(draw,elliptical),elliptical);
}

/*******************************************************************************/
/***************************** Truncated Gaussian ******************************/
/*******************************************************************************/
static void FreeElliptical_gaussian(TElliptical_gaussian *data)
{
  if (data)
    {
      FreeMatrix(data->variance);
      dw_free(data);
    }
}

PRECISION VarianceScale_TruncatedGaussian(int dim, PRECISION lower_bound, PRECISION upper_bound)
{ 
  PRECISION a, b, c, d;
  if (upper_bound >= PLUS_INFINITY)
    a=c=1.0;
  else
    {
      a=dw_chi_square_cdf(upper_bound*upper_bound,dim);
      c=dw_chi_square_cdf(upper_bound*upper_bound,dim+2);
    }
  if (lower_bound <= 0.0)
    b=d=0.0;
  else
    {
      b=dw_chi_square_cdf(lower_bound*lower_bound,dim);
      d=dw_chi_square_cdf(lower_bound*lower_bound,dim+2);
    }
  return (c-d)/(a-b);
}

static PRECISION logdensity_radius_truncated_gaussian(PRECISION radius, TElliptical *elliptical)
{
  return -0.5*radius*radius + ((TElliptical_gaussian*)(elliptical->data))->logconstant;
}

static PRECISION cummulative_radius_truncated_gaussian(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_gaussian *d=(TElliptical_gaussian*)elliptical->data;
  return (radius <= d->lower_bound) ? 0.0 : (radius < d->upper_bound) ? 
    (dw_chi_square_cdf(radius*radius,elliptical->dim) - d->cumulative_lower_bound)/d->probability : 1.0;
}

static PRECISION draw_truncated_gaussian(PRECISION *draw, TElliptical *elliptical)
{
  PRECISION r, s;
  TElliptical_gaussian *d=(TElliptical_gaussian*)elliptical->data;
  do
    dw_NormalVector(elliptical->draw);
  while ((s=Norm(elliptical->draw)) == 0.0);
  r=sqrt(dw_chi_square_invcdf(d->cumulative_lower_bound + dw_uniform_rnd()*d->probability,elliptical->dim));
  ProductVS(elliptical->draw,elliptical->draw,r/s);
  ProductMV(elliptical->draw,elliptical->scale,elliptical->draw);
  AddVV(elliptical->draw,elliptical->draw,elliptical->center);
  if (draw && (draw != pElementV(elliptical->draw)))
    memcpy(draw,pElementV(elliptical->draw),elliptical->dim*sizeof(PRECISION));
  return r;
}

static void print_info_truncated_gaussian(FILE *f_out, TElliptical *elliptical)
{
  TElliptical_gaussian *d=(TElliptical_gaussian*)elliptical->data;
  if (f_out) fprintf(f_out,"%s\n dim=%d\n lower bound=%lg\n upper bound=%lg\n",
		     elliptical->type,elliptical->dim,d->lower_bound,d->upper_bound);
}

/*
   Let z be a normal random vector with given center and variance.  Truncate z so 
   that it is on the elliptical annulus given by the set of all z such that

        r = sqrt((z - center)' * Inverse(variance) * (z - center))

   is between lower_bound and upper_bound.  Choose scale so that variance=scale*scale'.
   The density is given by

                           Gamma(dim/2)            f(r)
                   ---------------------------  -----------
                    2 pi^(dim/2) |det(scale)|    r^(dim-1)

   where the density of r is given by


                                 r^(dim-1)*exp(-(r)^2/2)                      
    f(r) = ------------------------------------------------------------------------
            2^(dim/2-1)*Gamma(dim/2)*(c(upper_bound^2,dim) - c(lower_bound^2,dim))

   where c(x,n) is the cummulative chi-square density function with n degrees of
   freedom.  The random variable r^2 is has a truncated chi-square distribution. 

   Drawing z
     The vector z can be obtained by drawing y from the standard dim-dimensional 
     Gaussian distribtuion, r from the distribution on [lower_bound,upper_bound] 
     with density f(r), and defining z = r*scale*(y/norm(y)) + center. 

     Draws of r can be obtained (perhaps not most efficiently) by drawing u from 
     the uniform on [0,1] and defining r by

 r = sqrt(c_inverse(c(lower_bound^2,dim) + u*(c(upper_bound^2,dim) - c(lower_bound^2,n)),dim))

   The variance_scale = sigma^2 * E[r^2] is

                 (c(upper_bound^2,dim+2) - c(lower_bound^2,dim+2))
                ---------------------------------------------------
                   c(upper_bound^2,dim) - c(lower_bound^2,dim)


   Notes:  c(r,n) = dw_chi_square_cdf(r,n) and c_inverse(u,n) = dw_chi_square_invcdf(u,n).
   The GSL functions gsl_cdf_chisq_P() and gls_cdf_chisq_Pinv() are used to compute
   these functions.      
*/
TElliptical* CreateElliptical_TruncatedGaussian(int dim, TVector mean, TMatrix variance, PRECISION lower_bound, PRECISION upper_bound)
{
  TElliptical* elliptical;
  TElliptical_gaussian *data;
  TMatrix scale;

  if (variance)
    {
      scale=CholeskyLT((TMatrix)NULL,variance);
      Transpose(scale,scale);
      elliptical=CreateElliptical(dim,mean,scale);
      FreeMatrix(scale);
    }
  else
    elliptical=CreateElliptical(dim,mean,(TMatrix)NULL);
  
  elliptical->variance_scale=VarianceScale_TruncatedGaussian(dim,lower_bound,upper_bound);
  elliptical->type="truncated gaussian";

  data=(TElliptical_gaussian*)dw_malloc(sizeof(TElliptical_gaussian));
  data->lower_bound=lower_bound;
  data->upper_bound=upper_bound;
  data->cumulative_lower_bound=(lower_bound > 0.0) ? dw_chi_square_cdf(lower_bound*lower_bound,dim) : 0.0;
  data->cumulative_upper_bound=(upper_bound < PLUS_INFINITY) ? dw_chi_square_cdf(upper_bound*upper_bound,dim) : 1.0;
  data->probability=data->cumulative_upper_bound - data->cumulative_lower_bound;
  if (data->probability <= 0)
    {
      dw_UserError("Truncated gaussian defined only on positive probability regions\n");
      dw_free(data);
      return (TElliptical*)NULL;
    }
  data->logconstant=-0.5*dim*log(2.0*PI) + elliptical->logabsdet - log(data->probability);
  data->variance=variance ? EquateMatrix((TMatrix)NULL,variance) : IdentityMatrix((TMatrix)NULL,dim);

  elliptical->data=(void*)data;
  elliptical->logdensity_radius=logdensity_radius_truncated_gaussian;
  elliptical->logdensity_draw=logdensity_draw;
  elliptical->cummulative_radius=cummulative_radius_truncated_gaussian;
  elliptical->draw_vector=draw_truncated_gaussian;
  elliptical->print_info=print_info_truncated_gaussian;
  elliptical->free=(void (*)(void*))FreeElliptical_gaussian;

  return elliptical;
}
/*******************************************************************************/
/********************************** Gaussian ***********************************/
/*******************************************************************************/
static PRECISION draw_gaussian(PRECISION *draw, TElliptical *elliptical)
{
  PRECISION r;
  dw_NormalVector(elliptical->draw);
  r=sqrt(DotProduct(elliptical->draw,elliptical->draw));
  ProductMV(elliptical->draw,elliptical->scale,elliptical->draw);
  AddVV(elliptical->draw,elliptical->draw,elliptical->center);
  if (draw && (draw != pElementV(elliptical->draw)))
    memcpy(draw,pElementV(elliptical->draw),elliptical->dim*sizeof(PRECISION));
  return r;
}

static void print_info_gaussian(FILE *f_out, TElliptical *elliptical)
{
  if (f_out) fprintf(f_out,"%s\n dim=%d\n",elliptical->type,elliptical->dim);
}

/*
   Creates a Gaussian elliptical distribution with given mean and and variance.
   If mean is null, then it is taken to be the identity and if variance is null
   then it is taken to be the identity.
*/
TElliptical* CreateElliptical_Gaussian(int dim, TVector mean, TMatrix variance)
{
  TElliptical* elliptical;
  if (elliptical=CreateElliptical_TruncatedGaussian(dim,mean,variance,0.0,PLUS_INFINITY))
    {
      elliptical->type="gaussian";
      elliptical->draw_vector=draw_gaussian;
      elliptical->print_info=print_info_gaussian;
    }
  return elliptical;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/******************************* Truncated Power *******************************/
/*******************************************************************************/
PRECISION VarianceScale_TruncatedPower(int dim, PRECISION lower_bound, PRECISION upper_bound, PRECISION power)
{
  return power*(pow(upper_bound,power+2.0) - pow(lower_bound,power+2.0))
    /(dim*(power+2.0)*(pow(upper_bound,power) - pow(lower_bound,power)));
}

static PRECISION logdensity_radius_truncated_power(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  return ((radius <= d->lower_bound) || (radius > d->upper_bound)) ? MINUS_INFINITY :
    (d->power - elliptical->dim)*log(radius) + d->logconstant;
}

static PRECISION cummulative_radius_truncated_power(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  return (radius <= d->lower_bound) ? 0.0 : (radius < d->upper_bound) ? (pow(radius,d->power) - d->shift)/d->multiplier : 1.0;
}

static PRECISION draw_truncated_power(PRECISION *draw, TElliptical *elliptical)
{
  PRECISION r, s;
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  do
    dw_NormalVector(elliptical->draw);
  while ((s=Norm(elliptical->draw)) == 0.0);
  ProductSV(elliptical->draw,(r=pow(d->multiplier*dw_uniform_rnd() + d->shift,1.0/d->power))/s,elliptical->draw); 
  ProductMV(elliptical->draw,elliptical->scale,elliptical->draw);
  AddVV(elliptical->draw,elliptical->draw,elliptical->center);
  if (draw && (draw != pElementV(elliptical->draw)))
    memcpy(draw,pElementV(elliptical->draw),elliptical->dim*sizeof(PRECISION));
  return r;
}

static void print_info_truncated_power(FILE *f_out, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  if (f_out) fprintf(f_out,"%s\n dim=%d\n lower bound=%lg\n upper bound=%lg\n power=%lg\n",
		     elliptical->type,elliptical->dim,d->lower_bound,d->upper_bound,d->power);
}

/*
   Creates a power distribution on the elliptical annulus which is given by the 
   set of all z such that

        r = sqrt((z - center)' * Inverse(scale * scale') * (z - center))

   is between lower_bound and upper_bound.  The density is given by

                           Gamma(dim/2)            f(r)
                   ---------------------------  -----------
                    2 pi^(dim/2) |det(scale)|    r^(dim-1)

   where f(r) = power * r^(power-1) / (upper_bound^power - lower_bound^power) for
   lower_bound <= r <= upper_bound and power > 0.

   Drawing z
     The vector z can be obtained by drawing y from the standard dim-dimensional 
     Gaussian distribtuion, r from the distribution on [lower_bound,upper_bound] 
     with density f(r), and defining z = r*scale*(y/norm(y)) + center. 

     Since the cumulative density of r is 

       (r^power - lower_bound^power) / (upper_bound^power - lower_bound^power) 

     a draw of r can be obtained by drawing u from the uniform on [0,1] and 
     defining by

                   r = (multiplier*u + shift)^(1/power).  

     where multiplier = (upper_bound^power - lower_bound^power) and 
     shift = lower_bound^power
  
     The variance_scale = sigma^2 * E[r^2] is
 
               power*(upper_bound^(power+2) - lower_bound^(power+2))
              -------------------------------------------------------
               dim*(power+2)*(upper_bound^power - lower_bound^power)              
*/
TElliptical* CreateElliptical_TruncatedPower(int dim, TVector center, TMatrix scale, PRECISION lower_bound, 
					     PRECISION upper_bound, PRECISION power)
{
  TElliptical* elliptical;
  TElliptical_power *data;

  if ((lower_bound < 0) || (upper_bound <= lower_bound) || (power <= 0))
    {
      dw_Error(ARG_ERR);
      return (TElliptical*)NULL;
    }

  if (elliptical=CreateElliptical(dim,center,scale))
    {
      elliptical->variance_scale=VarianceScale_TruncatedPower(dim,lower_bound,upper_bound,power);
      elliptical->type="truncated power";

      data=(TElliptical_power*)dw_malloc(sizeof(TElliptical_power));

      data->lower_bound=lower_bound;
      data->upper_bound=upper_bound;
      data->power=power;
      data->shift=pow(lower_bound,power);
      data->multiplier=pow(upper_bound,power) - data->shift;
      data->logconstant=log(0.5*power) + dw_log_gamma(0.5*dim) - 0.5*dim*log(PI) - log(data->multiplier) + elliptical->logabsdet;

      elliptical->data=(void*)data;
      elliptical->logdensity_radius=logdensity_radius_truncated_power;
      elliptical->logdensity_draw=logdensity_draw;
      elliptical->cummulative_radius=cummulative_radius_truncated_power;
      elliptical->draw_vector=draw_truncated_power;
      elliptical->print_info=print_info_truncated_power;
      elliptical->free=(void (*)(void*))dw_free;
    }

  return elliptical;
}

/*******************************************************************************/
/************************************ Power ************************************/
/*******************************************************************************/
static void print_info_power(FILE *f_out, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  if (f_out) fprintf(f_out,"%s\n dim=%d\n upper bound=%lg\n power=%lg\n",
		     elliptical->type,elliptical->dim,d->upper_bound,d->power);
}

/*
   The power elliptical is a special case of the truncated power distribution 
   with lower_bound = 0.
*/
TElliptical* CreateElliptical_Power(int dim, TVector center, TMatrix scale, PRECISION bound, PRECISION power)
{
  TElliptical *elliptical;
  if (elliptical=CreateElliptical_TruncatedPower(dim,center,scale,0.0,bound,power))
    {
      elliptical->print_info=print_info_power;
      elliptical->type="power";
    }
  return elliptical;
}

/*******************************************************************************/
/*********************************** Uniform ***********************************/
/*******************************************************************************/
static void print_info_uniform(FILE *f_out, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  if (f_out) fprintf(f_out,"%s\n dim=%d\n lower bound=%lg\n upper bound=%lg\n",
		     elliptical->type,elliptical->dim,d->lower_bound,d->upper_bound);
}

static PRECISION logdensity_radius_uniform(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_power *d=(TElliptical_power*)elliptical->data;
  return ((radius < d->lower_bound) || (radius > d->upper_bound)) ? MINUS_INFINITY : d->logconstant;
}

/*
   Creates a uniform distribution on the elliptical annulus which is given by the 
   set of all z such that

        r = sqrt((z - center)' * Inverse(scale * scale') * (z - center))

   is between lower_bound and upper_bound.  This is a special case of the
   truncated power distribution with power = dim.
*/
TElliptical* CreateElliptical_Uniform(int dim, TVector center, TMatrix scale, PRECISION lower_bound, PRECISION upper_bound)
{
  TElliptical* elliptical;
  if (elliptical=CreateElliptical_TruncatedPower(dim,center,scale,lower_bound,upper_bound,dim))
    {
      elliptical->type="uniform";
      elliptical->logdensity_radius=logdensity_radius_uniform;
      elliptical->print_info=print_info_uniform;
    }
  return elliptical;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/************************************ Step *************************************/
/*******************************************************************************/
PRECISION VarianceScale_Step(int dim, PRECISION *t, int m)
{
  int i;
  PRECISION var=0.0;
  for (i=0; i < m; i++)
    var+=(t[i+1]*t[i+1]*t[i+1] - t[i]*t[i]*t[i])/(t[i+1]-t[i]);
  return var/(3.0*(PRECISION)(m*dim));
}

static void free_step(TElliptical_step *data)
{
  if (data)
    {
      if (data->t) dw_free(data->t);
      dw_free(data);
    }
}

static PRECISION logdensity_radius_step(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_step *d=(TElliptical_step*)elliptical->data;
  int min=0, max=d->m, mid;
  if ((radius < d->t[min]) || (radius > d->t[max])) 
    return MINUS_INFINITY;
  else
    {
      while (max - min > 1)
	if (radius > d->t[mid=(min + max)/2])
	  min=mid;
	else
	  max=mid;
      return (1 - elliptical->dim)*log(radius) - log((PRECISION)d->m*(d->t[max] - d->t[min])) + d->logconstant;
    }
}

static PRECISION cummulative_radius_step(PRECISION radius, TElliptical *elliptical)
{
  TElliptical_step *d=(TElliptical_step*)elliptical->data;
  int i;
  if (radius <= d->t[0]) return 0.0;
  for (i=1; i <= d->m; i++)
    if (radius <= d->t[i]) return ((PRECISION)(i-1) + (radius - d->t[i-1])/(d->t[i] - d->t[i-1]))/(PRECISION)d->m; 
  return 1.0;
}

static PRECISION draw_step(PRECISION *draw, TElliptical *elliptical)
{
  PRECISION r, s;
  int i;
  TElliptical_step *d=( TElliptical_step*)elliptical->data;
  do
    dw_NormalVector(elliptical->draw);
  while ((s=Norm(elliptical->draw)) == 0.0);
  i=(int)floor(dw_uniform_rnd()*(PRECISION)(d->m));
  r=(i < d->m) ? d->t[i] + dw_uniform_rnd()*(d->t[i+1] - d->t[i]) : d->t[d->m];
  ProductSV(elliptical->draw,r/s,elliptical->draw); 
  ProductMV(elliptical->draw,elliptical->scale,elliptical->draw);
  AddVV(elliptical->draw,elliptical->draw,elliptical->center);
  if (draw && (draw != pElementV(elliptical->draw)))
    memcpy(draw,pElementV(elliptical->draw),elliptical->dim*sizeof(PRECISION));
  return r;
}

static void print_info_step(FILE *f_out, TElliptical *elliptical)
{
  TElliptical_step *d=(TElliptical_step*)(elliptical->data);
  int i;
  if (f_out)
    {
      fprintf(f_out,"%s\n dim=%d\n lower bound=%lg\n upper bound=%lg\n table size=%d\n table=",elliptical->type,elliptical->dim,d->t[0],d->t[d->m],d->m);
      for (i=0; i <= d->m; i++) fprintf(f_out,"%lg ",d->t[i]);
      fprintf(f_out,"\n");
    }
}

/*
   Creates a elliptical power distribution on the elliptical annulus which is 
   given by the set of all z such that

        r = sqrt((z - center)' * Inverse(scale * scale') * (z - center))

   is between table[0] and table[m].  The density is given by

                           Gamma(dim/2)            h(r)
                   ---------------------------  -----------
                    2 pi^(dim/2) |det(scale)|    r^(dim-1)

   where h(r) is a step function.  The value of h(r) is 

                     1/((table[i] - table[i-1])*m)

   if r is between table [i-1] and table[i].           

   Note:
     The vector x can be obtained by drawing y from the standard n-dimensional 
     Gaussian distribtuion, r from the distribution on [table[0],table[m]] with 
     density h(r), and defining x = r*scale*y/norm(y) + center. 

     Because the probability of being between table[i-1] and table[i] is 1/m, and
     the density is constant over each of these intervals, r can be simulated by
     randomly choosing, with equal probability, an index i between 1 and m, 
     drawing u from the uniform distribution on [0,1], and defining r by 

                      r = table[i-1] + u*(table[i] - table[i-1])  
  
     The elements in table must be non-decreasing.                      
*/
TElliptical* CreateElliptical_Step(int dim, TVector center, TMatrix scale, PRECISION *table, int m)
{
  TElliptical* elliptical;
  TElliptical_step *data;

  if (elliptical=CreateElliptical(dim,center,scale))
    {
      elliptical->variance_scale=VarianceScale_Step(dim,table,m);
      elliptical->type="step";

      data=(TElliptical_step*)dw_malloc(sizeof(TElliptical_step));
      data->t=(PRECISION*)dw_malloc((m+1)*sizeof(PRECISION));
      memcpy(data->t,table,(m+1)*sizeof(PRECISION));
      data->m=m;
      data->logconstant=log(0.5) + dw_log_gamma(0.5*dim) - 0.5*dim*log(PI) + elliptical->logabsdet;

      elliptical->data=(void*)data;
      elliptical->logdensity_radius=logdensity_radius_step;
      elliptical->cummulative_radius=cummulative_radius_step;
      elliptical->logdensity_draw=logdensity_draw;
      elliptical->draw_vector=draw_step;
      elliptical->print_info=print_info_step;
      elliptical->free=(void (*)(void*))free_step;
    }

  return elliptical;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/
#undef PI
