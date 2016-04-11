/*
 * Copyright (C) 1996-2011 Daniel Waggoner and Tao Zha
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

#include "mdd_function.h"
#include "dw_matrix.h"
#include "dw_array.h"
#include "dw_matrix_rand.h"
#include "dw_ascii.h"
#include "dw_rand.h"
#include "dw_math.h"
#include "dw_matrix_sort.h"
#include "dw_elliptical.h"
#include "dw_parse_cmd.h"
#include "dw_std.h"
#include "mdd_constant.hpp"

#include <math.h>
#include <string.h>
#include <time.h>

static PRECISION ComputeLinear(TMatrix proposal, PRECISION log_c, int *intercept);
static PRECISION ComputeInverseLinear(TMatrix posterior, PRECISION log_c, int *intercept);

/*******************************************************************************/
/*********** Waggoner/Zha method for computing marginal data density ***********/  
/*******************************************************************************/
/*
   Let h(x) and f(x) be properly scaled probability density functions and let c 
   be an unknown constant.  We use the following notation.

      x      - parameters
      y      - data
      f(x)   - posterior distribution = p(x|y)
      h(x)   - proposal distribution
      c      - marginal data distribution = p(y)
      c*f(x) - likelihood*prior = p(y|x)*p(x)

   Assumes:
     proposal  : N x 2 matrix with proposal[i][0] = ln(h(x(i))) and 
                 proposal[i][1] = ln(c*f(x(i))) where x(i) is sampled from h(x).
     posterior : M x 2 matrix with posterior[i][0] = ln(h(x(i))) and 
                 posterior[i][1] = ln(c*f(x(i))) where x(i) is sampled from f(x).
     L1        : cutoff value (c*f(x) > L1)
     L2        : cutoff value for (h(x) < L2)

   Returns:
     Estimate of c or MINUS_INFINITY if no proposal draws satisfied the
     restriction given by the cutoff values of L1 and L2.

   Notes:
     Let S be the set of all x such that c*f(x) > exp(L1) and h(x) < exp(L2).  
     Then c = p(L1,L2)/I(L1,L2) where p(L1,L2) is the probability that x, sampled 
     from h(x), is in S and I(L1,L2) is the integral with respect to x over the 
     set S of

                                h(x)/(c*f(x)) * f(x) 

     p(L1,L2) can be approximated from the proposal draws by

                    (number of draws in S) / (total number of draws)

     I(L1,L2) can be approximated from the posterior draws by summing 

                                    h(x)/(c*f(x)) 
   
     over all posterior draws with x in S and then dividing by the total number 
     of posterior draws.       
*/
PRECISION ComputeLogMarginalDensity_WaggonerZha(TMatrix proposal, TMatrix posterior, PRECISION L1, PRECISION L2, int *in_proposal, int *in_posterior, PRECISION *I)
{
  int i;

  if (L1 == MINUS_INFINITY)
    if (L2 == PLUS_INFINITY)
      {
    	*in_proposal=RowM(proposal);

	*in_posterior=RowM(posterior);
	(*I)=MINUS_INFINITY;
	for (i=RowM(posterior)-1; i >= 0; i--)
	  (*I)=AddLogs(*I,ElementM(posterior,i,0) - ElementM(posterior,i,1));
      }
    else
      {
	for (*in_proposal=0, i=RowM(proposal)-1; i >= 0; i--)
	  if (ElementM(proposal,i,0) <= L2)
	    (*in_proposal)++;

	(*I)=MINUS_INFINITY;
	for (*in_posterior=0, i=RowM(posterior)-1; i >= 0; i--)
	  if (ElementM(posterior,i,0) <= L2)
	    {
	      (*in_posterior)++;
	      (*I)=AddLogs(*I,ElementM(posterior,i,0) - ElementM(posterior,i,1));
	    }
      }
  else
    if (L2 == PLUS_INFINITY)
      {
	for (*in_proposal=0, i=RowM(proposal)-1; i >= 0; i--)
	  if (ElementM(proposal,i,1) >= L1)
	    (*in_proposal)++;

	(*I)=MINUS_INFINITY;
	for (*in_posterior=0, i=RowM(posterior)-1; i >= 0; i--)
	  if (ElementM(posterior,i,1) >= L1)
	    {
	      (*in_posterior)++;
	      (*I)=AddLogs(*I,ElementM(posterior,i,0) - ElementM(posterior,i,1));
	    }
      }
    else
      {
	for (*in_proposal=0, i=RowM(proposal)-1; i >= 0; i--)
	  if ((ElementM(proposal,i,1) >= L1) && (ElementM(proposal,i,0) <= L2))
	    (*in_proposal)++;

	(*I)=MINUS_INFINITY;
	for (*in_posterior=0, i=RowM(posterior)-1; i >= 0; i--)
	  if ((ElementM(posterior,i,1) >= L1) && (ElementM(posterior,i,0) <= L2))
	    {
	      (*in_posterior)++;
	      (*I)=AddLogs(*I,ElementM(posterior,i,0) - ElementM(posterior,i,1));
	    }
      }

  if ((*in_posterior) > 0) (*I)-=log((PRECISION)RowM(posterior));

  return ((*in_proposal) == 0) ? MINUS_INFINITY : log((PRECISION)(*in_proposal)/(PRECISION)RowM(proposal)) - (*I);
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/************************** Bridge Method Alternative **************************/
/*******************************************************************************/
static PRECISION BridgeDifference(TMatrix proposal, TMatrix posterior, PRECISION logr)
{
  int i;
  PRECISION x, r, sum1=MINUS_INFINITY, sum2=MINUS_INFINITY, n1=RowM(posterior), n2=RowM(proposal);

  for (r=n2/n1, i=RowM(proposal)-1; i >= 0; i--)
    if ((x=ElementM(proposal,i,0)+logr-ElementM(proposal,i,1)) < 0)
      sum2=AddLogs(sum2,-log(1+r*exp(x)));
    else
      sum2=AddLogs(sum2,-x-log(exp(-x)+r));

  for (r=n1/n2, i=RowM(posterior)-1; i >= 0; i--)
    if ((x=ElementM(posterior,i,1)-ElementM(posterior,i,0)-logr) < 0)
      sum1=AddLogs(sum1,-log(1+r*exp(x)));
    else
      sum1=AddLogs(sum1,-x-log(exp(-x)+r));
    
  return sum2 - sum1;
}
#define MAX_C 1.0E50
#define TOL   1.0E-7
PRECISION ComputeLogMarginalDensity_Bridge(TMatrix proposal, TMatrix posterior)
{
  PRECISION min_c, max_c, mid_c=0.0, diff;
  int i;

  // Bracket the zero
  if ((diff=BridgeDifference(proposal,posterior,mid_c)) < 0.0)
    {
      max_c=mid_c;
      for (min_c=-1.0; min_c > -MAX_C; max_c=min_c, min_c*=10)
	if ((diff=BridgeDifference(proposal,posterior,min_c)) > 0) break;
      if (min_c <= -MAX_C) return min_c;
    }
  else
    {
      min_c=mid_c;
      for (max_c=1.0; max_c < MAX_C; min_c=max_c, max_c*=10)
	if ((diff=BridgeDifference(proposal,posterior,max_c)) < 0) break;
      if (max_c >= MAX_C) return max_c;
    }

  // Divide and conququer
  diff=BridgeDifference(proposal,posterior,mid_c=(min_c + max_c)/2.0);
  for (i=0; i < 50; i++)
    {
      if (diff > 0)
	min_c=mid_c;
      else
	max_c=mid_c;
      if ((fabs(diff=BridgeDifference(proposal,posterior,mid_c=(min_c + max_c)/2.0)) < TOL)) break;
    }
  return mid_c;
}
#undef MAX_C
#undef TOL
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

/*******************************************************************************/
/************ Mueller's method for computing marginal data density *************/
/*******************************************************************************/
#define MAX_C 1E50
static PRECISION ComputeLinear(TMatrix proposal, PRECISION log_c, int *intercept)
{
  int i;
  PRECISION slope=0.0, tmp;
  for (*intercept=0, i=RowM(proposal)-1; i >= 0; i--)
    if (log_c <= (tmp=ElementM(proposal,i,0) - ElementM(proposal,i,1)))
      {
	(*intercept)++;
	slope+=exp(log_c-tmp);
      }
  return slope;
}
static PRECISION ComputeInverseLinear(TMatrix posterior, PRECISION log_c, int *intercept)
{
  int i;
  PRECISION slope=0.0, tmp;
  for (*intercept=0, i=RowM(posterior)-1; i >= 0; i--)
    if (log_c >= (tmp=ElementM(posterior,i,0)-ElementM(posterior,i,1)))
      {
	(*intercept)++;
	slope+=exp(tmp-log_c);
      }
  return slope;
}
PRECISION ComputeLogMarginalDensity_Mueller(TMatrix proposal, TMatrix posterior,int *in1, int *in2)
{
  PRECISION log_c=0.0, slope1, slope2, N1=1.0/(PRECISION)RowM(proposal), N2=1.0/(PRECISION)RowM(posterior),
    intercept1, intercept2, max_log_c, min_log_c, diff, tmp;
  int min_in1, max_in1, min_in2, max_in2;

  slope1=ComputeLinear(proposal,log_c,in1);
  slope2=ComputeInverseLinear(posterior,log_c,in2);
  diff=N1*((PRECISION)(*in1) - slope1) - N2*((PRECISION)(*in2) - slope2);

  // Bracket the intersection
  if (diff < 0.0)
    {
      do
	if (log_c < -MAX_C) 
	  return log_c;
	else
	  {
	    max_in1=*in1;
	    max_in2=*in2;
	    max_log_c=log_c;
	    log_c=10*(log_c-1);
	    slope1=ComputeLinear(proposal,log_c,in1);
	    slope2=ComputeInverseLinear(posterior,log_c,in2);
	    diff=N1*((PRECISION)(*in1) - slope1) - N2*((PRECISION)(*in2) - slope2);
	  }
      while (diff < 0.0);
      min_in1=*in1;
      min_in2=*in2;
      min_log_c=log_c;
    }
  else
    {
      do
	if (log_c > MAX_C) 
	  return log_c;
	else
	  {
	    min_in1=*in1;
	    min_in2=*in2;
	    min_log_c=log_c;
	    log_c=10*(log_c+1);
	    slope1=ComputeLinear(proposal,log_c,in1);
	    slope2=ComputeInverseLinear(posterior,log_c,in2);
	    diff=N1*((PRECISION)(*in1) - slope1) - N2*((PRECISION)(*in2) - slope2);
	  }
      while (diff >= 0.0);
      max_in1=*in1;
      max_in2=*in2;
      max_log_c=log_c;
    }

  // At this point diff(min_log_c) >= 0 and diff(max_log_c) < 0.
  while ((min_in1 != max_in1) || (min_in2 != max_in2))
    {
      log_c=(min_log_c + max_log_c)/2.0;
      slope1=ComputeLinear(proposal,log_c,in1);
      slope2=ComputeInverseLinear(posterior,log_c,in2);
      diff=N1*((PRECISION)(*in1) - slope1) - N2*((PRECISION)(*in2) - slope2);
      if (diff > 0)
	{
	  min_in1=*in1;
	  min_in2=*in2;
	  min_log_c=log_c;
	}
      else
	{
	  max_in1=*in1;
	  max_in2=*in2;
	  max_log_c=log_c;
	}
    }

  slope1=N1*ComputeLinear(proposal,min_log_c,in1);
  intercept1=N1*(PRECISION)(*in1);
  slope2=N2*ComputeInverseLinear(posterior,min_log_c,in2);
  intercept2=N2*(PRECISION)(*in2);

  tmp=intercept1-intercept2;
  if (slope1 > 0)
    {
      tmp+=sqrt(tmp*tmp + 4*slope1*slope2);
      if (tmp >= 2.0*slope1)
	tmp=min_log_c + log(tmp) - log(2.0*slope1);
      else
	return -min_log_c;
    }
  else
    {
      if (tmp > 0)
	if (slope2 > tmp)
	  tmp=min_log_c + log(slope2) - log(tmp);
	else
	  return -min_log_c;
      else
	return -max_log_c;
    }
  return (tmp > max_log_c) ? -max_log_c : -tmp;
}
#undef MAX_C
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/* Read/interpret posterior draws and setup elliptical/Dirichlet distributions */
/*******************************************************************************/
/*
   Assumes the file has 2 + ntheta + nq columns.  The data is arranged as:

                     posterior  likelihood  theta  q

   (1) Loops through posterior draws computing first and second moments of theta, 
       the mode of the posterior and the first and second moments of each 
       individual element of q

   (2) Sets the value of the center and scale dependent on the value of flag.
       The center will be either the mode or mean of theta and the scale will 
       satisfy scale*scale'=E[(theta-center)(theta-center)'].

   (3) Sets the values of alpha.

   The vector center, the matrix scale, and array alpha must be properly allocated of 
   the proper size.  

   If 0 < percentage_draws < 1 then the first percentage_draws of the posterior draws 
   are used to the center, the scale, and alpha.  If -1 < percentage_draws < 0, then 
   the last (1 - percentage_draws) are used to compute the center, the scale and alpha.  
   Otherwise, all the draws are used.
*/
void PosteriorDistributionInfo(FILE *f_in, int flag, TVector center, TMatrix scale, int ndraws)
{
  int i, j, k, n, p, ntheta, ntotal;
  TVector theta=(TVector)NULL, theta_sum=(TVector)NULL, theta_mode=(TVector)NULL;
  TMatrix theta_square=(TMatrix)NULL, theta_square_tmp=(TMatrix)NULL, quadratic_form=(TMatrix)NULL;
  PRECISION max=MINUS_INFINITY, inc, tmp, sum, *x;

  // set dimensions and allocate memory
  ntheta=center ? DimV(center) : 0;
  ntotal=2+ntheta;

  if (ntheta > 0)
    {
      theta=CreateVector(ntheta);
      InitializeVector(theta_sum=CreateVector(ntheta),0.0);
      InitializeMatrix(theta_square=CreateMatrix(ntheta,ntheta),0.0);
      theta_square_tmp=CreateMatrix(ntheta,ntheta);
      theta_mode=CreateVector(ntheta);
      quadratic_form=CreateMatrix(ntheta,ntheta);
    }


  // loop and accumulate 1st and 2nd non-central moments and posterior mode
  for (i=0; i < ndraws; i++)
    {
      x=dw_ReadDelimitedLine_floating(f_in,' ',REMOVE_EMPTY_FIELDS,MINUS_INFINITY);
      if (!x || (dw_DimA(x) != ntotal)) 
	{
	  printf("Error reading posterior file\n");
	  dw_exit(0);
	}
      if (ntheta > 0)
	{
	  memcpy(pElementV(theta),x+2,ntheta*sizeof(PRECISION));
	  if (x[0] > max)
	    {
	      max=x[0];
	      EquateVector(theta_mode,theta);
	    }
	  AddVV(theta_sum,theta_sum,theta);
	  OuterProduct(theta_square_tmp,theta,theta);
	  AddMM(theta_square,theta_square,theta_square_tmp);
	}

      dw_FreeArray(x);
    }

  if (ntheta > 0)
    {
      // scale 1st and 2nd non-central moments
      ProductVS(theta_sum,theta_sum,1.0/(PRECISION)ndraws);
      ProductMS(theta_square,theta_square,1.0/(PRECISION)ndraws);

      // set center and scale using flag
      if (flag & USE_MODE)
	{
	  EquateVector(center,theta_mode);
	  OuterProduct(theta_square_tmp,theta_mode,theta_mode);
	  AddMM(theta_square,theta_square,theta_square_tmp);
	  OuterProduct(theta_square_tmp,theta_sum,theta_mode);
	  SubtractMM(theta_square,theta_square,theta_square_tmp);
	  OuterProduct(theta_square_tmp,theta_mode,theta_sum);
	  SubtractMM(theta_square,theta_square,theta_square_tmp);
	  CholeskyLT(scale,theta_square);
	  Transpose(scale,scale);
	}
      else
	{
	  EquateVector(center,theta_sum);
	  OuterProduct(theta_square_tmp,theta_sum,theta_sum);
	  SubtractMM(theta_square,theta_square,theta_square_tmp);
	  CholeskyLT(scale,theta_square);
	  Transpose(scale,scale);
	}
    }

  // compute alpha
  if (ntheta > 0)
    {
      FreeVector(theta);
      FreeVector(theta_sum);
      FreeMatrix(theta_square);
      FreeMatrix(theta_square_tmp);
      FreeVector(theta_mode);
      FreeMatrix(quadratic_form);
    }
}

/*
   Assumes the file has 2 + ntheta + nq columns.  The data is arranged as:

                     posterior  likelihood  theta  q

   Creates and returns a vector containing the radius of each draw of theta,
   computed used the center and scale.
*/
TVector PosteriorRadius(FILE *f_in, TVector center, TMatrix scale, int ndraws)
{
  int i, ntheta, ntotal;
  TVector  theta, Radii=(TVector)NULL;
  TMatrix quadratic_form;
  PRECISION *x;

  ntheta=center ? DimV(center) : 0;
  ntotal=ntheta+2;

  if (ntheta > 0)
    {
      theta=CreateVector(ntheta);
      quadratic_form=CreateMatrix(ntheta,ntheta);

      Radii=CreateVector(ndraws);
      quadratic_form=ProductTransposeMM((TMatrix)NULL,scale,scale);
      Inverse_LU(quadratic_form,quadratic_form);
      ForceSymmetric(quadratic_form);
     
      for (i=0; i < ndraws; i++)
	{
	  x=dw_ReadDelimitedLine_floating(f_in,' ',REMOVE_EMPTY_FIELDS,MINUS_INFINITY);
	  if (!x || (dw_DimA(x) < ntotal)) 
	    {
	      printf("Error reading posterior file\n");
	      dw_exit(0);
	    }
	  memcpy(pElementV(theta),x+2,ntheta*sizeof(PRECISION));
	  SubtractVV(theta,theta,center);
	  ElementV(Radii,i)=sqrt(InnerProduct(theta,theta,quadratic_form));
	  dw_FreeArray(x);
	}

      FreeVector(theta);
      FreeMatrix(quadratic_form);
    }

  return Radii;
}

TElliptical* CreateEllipticalFromPosterior_TruncatedGaussian(TVector R, int dim, TVector center, TMatrix scale, PRECISION p_lo, PRECISION p_hi)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  PRECISION a, b;
  TMatrix V;
  int i;

  if (dim > 0)
    {
      SortVectorAscending(R,R);

      i=floor(p_lo*(PRECISION)(DimV(R)-1));
      a=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);
      i=floor(p_hi*(PRECISION)(DimV(R)-1));
      b=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);

      V=ProductTransposeMM((TMatrix)NULL,scale,scale);
      elliptical=CreateElliptical_TruncatedGaussian(dim,center,V,a,b);
      FreeMatrix(V);
    }

  return elliptical;
}

TElliptical* CreateEllipticalFromPosterior_Gaussian(TVector R, int dim, TVector center, TMatrix scale)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  TMatrix variance;
  if (dim > 0)
    {
      elliptical=CreateElliptical_Gaussian(dim,center,variance=ProductTransposeMM((TMatrix)NULL,scale,scale));
      FreeMatrix(variance);
    }
  return elliptical;
}

static PRECISION GetPower_truncatedpower(int dim, PRECISION lower_bound, PRECISION upper_bound)
{
  PRECISION lo=0.0001, hi=1.0, mid;
  int i;
  if (VarianceScale_TruncatedPower(dim,lower_bound,upper_bound,lo) > 1.0)
    return lo;
  for (i=0; i < 20; i++)
    if (VarianceScale_TruncatedPower(dim,lower_bound,upper_bound,hi) < 1.0)
      {
	lo=hi;
	hi*=2.0;
      }
    else
      {
	for (i=0; i < 30; i++)
	  {
	    mid=0.5*(lo+hi);
	    if (VarianceScale_TruncatedPower(dim,lower_bound,upper_bound,mid) < 1.0)
	      lo=mid;
	    else
	      hi=mid;
	  }
	return mid;
      }
  return hi;
}

TElliptical* CreateEllipticalFromPosterior_TruncatedPower(TVector R, int dim, TVector center, TMatrix scale, PRECISION p_lo, PRECISION p_hi)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  PRECISION a, b, power;
  int i;

  if (dim > 0)
    {
      SortVectorAscending(R,R);

      i=floor(p_lo*(PRECISION)(DimV(R)-1));
      a=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);
      i=floor(p_hi*(PRECISION)(DimV(R)-1));
      b=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);
      power=GetPower_truncatedpower(dim,a,b);
      
      // The code below uses the old method to choosing the power and b.  The new 
      // method chooses the power by trying to match the scale of the variance.
      /* i=floor(0.10*(PRECISION)DimV(R)); */
      /* a=(i > 0) ? 0.5*(sqrt(ElementV(R,i-1)) + sqrt(ElementV(R,i))) : sqrt(ElementV(R,i)); */
      /* i=floor(0.90*(PRECISION)DimV(R)); */
      /* b=(i > 0) ? 0.5*(sqrt(ElementV(R,i-1)) + sqrt(ElementV(R,i))) : sqrt(ElementV(R,i)); */
      /* power=log(0.10/0.90)/log(a/b); */
      /* b/=pow(0.90,1/power); */
      /* i=floor(0.10*(PRECISION)DimV(R)); */
      /* a=(i > 0) ? 0.5*(ElementV(R,i-1) + ElementV(R,i)) : ElementV(R,i); */
      /* i=floor(0.90*(PRECISION)DimV(R)); */
      /* b=(i > 0) ? 0.5*(ElementV(R,i-1) + ElementV(R,i)) : ElementV(R,i); */
      /* power=log(0.10/0.90)/log(a/b); */
      /* b/=pow(0.90,1/power); */

      elliptical=CreateElliptical_TruncatedPower(dim,center,scale,a,b,power);
    }

  return elliptical;
}

TElliptical* CreateEllipticalFromPosterior_Power(TVector R, int dim, TVector center, TMatrix scale, PRECISION p_hi)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  PRECISION b, power;
  int i;

  if (dim > 0)
    {
      SortVectorAscending(R,R);

      i=floor(p_hi*(PRECISION)(DimV(R)-1));
      b=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);
      power=GetPower_truncatedpower(dim,0.0,b);

      // The code below uses the old method to choosing the power and b.  The new 
      // method chooses the power by trying to match the scale of the variance.
      /* i=floor(0.20*(PRECISION)DimV(R)); */
      /* PRECISION a = (i > 0) ? 0.5*(sqrt(ElementV(R,i-1)) + sqrt(ElementV(R,i))) : sqrt(ElementV(R,i)); */
      /* i=floor(0.80*(PRECISION)DimV(R)); */
      /* b=(i > 0) ? 0.5*(sqrt(ElementV(R,i-1)) + sqrt(ElementV(R,i))) : sqrt(ElementV(R,i)); */
      /* power=log(0.20/0.80)/log(a/b); */
      /* b/=pow(0.80,1/power); */
      /* i=floor(0.20*(PRECISION)DimV(R)); */
      /* PRECISION a = (i > 0) ? 0.5*(ElementV(R,i-1) + ElementV(R,i)) : ElementV(R,i); */
      /* i=floor(0.80*(PRECISION)DimV(R)); */
      /* b=(i > 0) ? 0.5*(ElementV(R,i-1) + ElementV(R,i)) : ElementV(R,i); */
      /* power=log(0.20/0.80)/log(a/b); */
      /* b/=pow(0.80,1/power); */

      elliptical=CreateElliptical_Power(dim,center,scale,b,power);
    }

  return elliptical;
}

TElliptical* CreateEllipticalFromPosterior_Uniform(TVector R, int dim, TVector center, TMatrix scale, PRECISION p_lo, PRECISION p_hi)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  PRECISION a, b;
  int i;

  if (dim > 0)
    {
      SortVectorAscending(R,R);

      i=floor(p_lo*(PRECISION)(DimV(R)-1));
      a=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);
      i=floor(p_hi*(PRECISION)(DimV(R)-1));
      b=(i < DimV(R)-1) ? 0.5*(ElementV(R,i+1) + ElementV(R,i)) : ElementV(R,i);

      elliptical=CreateElliptical_Uniform(dim,center,scale,a,b);
    }

  return elliptical;
}

TElliptical* CreateEllipticalFromPosterior_Step(TVector R, int dim, TVector center, TMatrix scale, PRECISION p_lo, PRECISION p_hi)
{
  TElliptical *elliptical=(TElliptical*)NULL;
  PRECISION *table, inc, x, y;
  int i, j, k, m=30;

  if (dim > 0)
    {
      table=(PRECISION*)dw_malloc((m+1)*sizeof(PRECISION));

      SortVectorAscending(R,R);

      for (k=m; k >= 1; k--)
	{
	  inc=(p_hi-p_lo)/(PRECISION)k;
	  for (x=p_lo, j=0; j <= k; x+=inc, j++)
	    {
	      i=floor(y=x*(PRECISION)(DimV(R)-1));
	      if (i >= DimV(R)-1)
		table[j]=ElementV(R,DimV(R)-1);
	      else
		table[j]=(1.0-y+i)*ElementV(R,i) + (y-i)*ElementV(R,i+1);
	      //table[j]=(i > 0) ? 0.5*(ElementV(R,i-1) + ElementV(R,i)) : ElementV(R,i);
	    }
	  for (j=1; j <= k; j++)
	    if (table[j] <= table[j-1]) break;
	  if (j > k) break;
	}

      if (k <= 0)
	{
	  dw_free(table);
	  printf("Not enough variation in the posterior draws to form proposal\n");
	  dw_exit(0);
	}

      elliptical=CreateElliptical_Step(dim,center,scale,table,k);
      dw_free(table);
    }

  return elliptical;
}
/*******************************************************************************/
/*******************************************************************************/


/*******************************************************************************/
/*********** Create proposal and posterior matrices of log densities ***********/
/*******************************************************************************/
/*
   Assumes:
     ndraws         : Number of draws from proposal to create
     model          : Valid pointer to TStateModel structure
     elliptical     : Valid pointer to TElliptical structure
     alpha          : Array of parameters for the independent Dirichlet 
                      distributions.
     dirichlet_dims : integer array of dimensions of the independent Dirichlet
                      distributions.

   Results:
     Draws theta from the elliptical distribution and q from independent 
     Dirichlet distributions.  Computes the log density, properly scaled, of the 
     proposal distribution and stores this in the first column of proposal.  
     Computes the log posterior (likelihood times prior) with the states 
     integerated out, and stores this in the second column of proposal.

   Notes:
     The integer array dirichlet_dims should be created with the routine
     CreateDirichletDimensions().

TMatrix CreateProposal(int ndraws, TStateModel* model, TElliptical *elliptical, PRECISION* alpha, int *dirichlet_dims)
{
  TMatrix proposal;
  PRECISION logdensity, *q=(PRECISION*)NULL, *theta=(PRECISION*)NULL;
  int i, ntheta, nq, n_dirichlet, period, count;

  if (ndraws <= 0) return (TMatrix)NULL;

  // create proposal
  proposal=CreateMatrix(ndraws,2);

  // set dimensions and allocate memory
  ntheta=elliptical ? elliptical->dim : 0;
  n_dirichlet=dirichlet_dims ? dw_DimA(dirichlet_dims) : 0;
  for (i=nq=0; i < n_dirichlet; i++) nq+=dirichlet_dims[i]-1;
  if (ntheta) theta=(PRECISION*)dw_malloc(ntheta*sizeof(PRECISION));  
  if (nq) q=(PRECISION*)dw_malloc(nq*sizeof(PRECISION));

  period=ndraws/50;
  for (count=period, i=0;  i < ndraws; i++)
    {
      // draw theta from elliptical and compute log density
      logdensity=(theta) ? LogDensityElliptical_Radius(DrawElliptical(theta,elliptical),elliptical) : 0.0;

      // draw q from Dirichet and compute log density
      if (q)
	{
	  DrawIndependentDirichlet_free(q,alpha,dirichlet_dims,n_dirichlet);
	  logdensity+=LogIndependentDirichlet_pdf(q,alpha,dirichlet_dims,n_dirichlet);
	}

      // store log density of proposal
      ElementM(proposal,i,0)=logdensity;

      // force free parameters into model and store log density of posterior
      if (theta) ConvertFreeParametersToTheta(model,theta);
      if (q) ConvertFreeParametersToQ(model,q);
      ElementM(proposal,i,1)=LogPosterior_StatesIntegratedOut(model); //printf("%le\n",ElementM(proposal,i,1));

      // Output status
      if (i == count)
	{
	  printf("%d draws out of %d\n",count,ndraws);
	  count+=period;
	}
    }

  // clean up
  if (theta) dw_free(theta);
  if (q) dw_free(q);

  return proposal;
}*/

/*
    Assumes:
      Posterior file has at least ndraws rows and 2+ntheta+nq columns
        Column  0                      = log posterior density of (theta,q).
        Column  1                      = log likelihood of (theta,q).
        Columns 2 : 1+ntheta           = theta.
        Columns 2+ntheta : 1+ntheta+nq = q.

    Returns:
      ndraws x 2 matrix.  The first column is the log of the proposal density of the
      posterior draw and the second column is the log of the posterior density of
      the posterior draw.

    Notes:
      The parameters are theta and q.  The proposal density is independent across
      these with independent Dirichlet distributions on q and an elliptical 
      density on theta.
*/
TMatrix CreatePosterior(FILE *f_in, TElliptical *elliptical, int ndraws)
{
  int i, ntheta, ntotal;
  PRECISION logdensity, *x;
  TMatrix posterior;

  // set dimensions and allocate memory
  posterior=CreateMatrix(ndraws,2);
  ntheta=elliptical ? elliptical->dim : 0;
  ntotal=2+ntheta;

  for (i=0; i < ndraws; i++)
    {
      x=dw_ReadDelimitedLine_floating(f_in,' ',REMOVE_EMPTY_FIELDS,MINUS_INFINITY);
      if (!x || (dw_DimA(x) < ntotal)) 
	{
	  printf("Error reading posterior file\n");
	  dw_exit(0);
	}
      logdensity=(ntheta > 0) ? LogDensityElliptical_Draw(x+2,elliptical) : 0.0; 
      ElementM(posterior,i,0)=logdensity;
      ElementM(posterior,i,1)=x[0];
      dw_FreeArray(x);
    } 

  return posterior;
}

int ModifyUsingSWZ(TMatrix *modified_proposal, TMatrix *modified_posterior, TMatrix proposal, TMatrix posterior, PRECISION cut)
{
  int i, j, count;
  PRECISION log_p;

  // Number proposal satisfying cut
  for (count=0, i=RowM(proposal)-1; i >= 0; i--)
    if (ElementM(proposal,i,1) >= cut) count++;

  // Abort if none satisfy
  if (count == 0) return 0;

  // Compute log probability of proposal satisfying cut
  log_p=log((PRECISION)count/(PRECISION)RowM(proposal));

  // Create matrices if necessary
  if (!(*modified_proposal) || (RowM(*modified_proposal) != count))
    {
      if (*modified_proposal) FreeMatrix(*modified_proposal);
      *modified_proposal=CreateMatrix(count,2);
    }
  if (!(*modified_posterior) || (RowM(*modified_posterior) != RowM(posterior)))
    {
      if (*modified_posterior) FreeMatrix(*modified_posterior);
      *modified_posterior=CreateMatrix(RowM(posterior),2);
    }

  // Set modified posterior
  for (i=RowM(posterior)-1; i >= 0; i--)
    {
      if (ElementM(posterior,i,1) > cut)
	ElementM(*modified_posterior,i,0)=ElementM(posterior,i,0)-log_p;
      else
	ElementM(*modified_posterior,i,0)=MINUS_INFINITY;
      ElementM(*modified_posterior,i,1)=ElementM(posterior,i,1);
    }

  // Set modified proposal
  for (i=j=0; i < RowM(proposal); i++)
    if (ElementM(proposal,i,1) >= cut)
      {
	ElementM(*modified_proposal,j,0)=ElementM(proposal,i,0)-log_p;
	ElementM(*modified_proposal,j,1)=ElementM(proposal,i,1);
      }

  // Return number satisfing cut
  return count;
}
/*******************************************************************************/
/*******************************************************************************/
/*******************************************************************************/

