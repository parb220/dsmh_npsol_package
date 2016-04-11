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

#ifndef __ELLIPTICAL_DISTRIBUTIONS__
#define __ELLIPTICAL_DISTRIBUTIONS__

#ifdef __cplusplus
extern "C"
{
#endif

#include "dw_matrix.h"
#include <stdio.h>

typedef struct TElliptical_tag
{
  PRECISION logabsdet;           // = -ln(abs(det(scale))))
  int dim;                       // = dimension of elliptical random vector
  PRECISION variance_scale;      // variance = variance_scale * (scale * scale')
  TVector center;                // mean = center
  TMatrix scale;
  TMatrix quadratic_form;        // Inverse(scale * scale')
  TVector draw;
  char *type;

  void *data;

  PRECISION (*logdensity_radius)(PRECISION, struct TElliptical_tag *);
  PRECISION (*logdensity_draw)(PRECISION*, struct TElliptical_tag *);
  PRECISION (*cummulative_radius)(PRECISION, struct TElliptical_tag *);
  PRECISION (*draw_vector)(PRECISION*, struct TElliptical_tag *);
  void (*print_info)(FILE *, struct TElliptical_tag *);
  void (*free)(void*);

} TElliptical;

typedef struct 
{
  TMatrix variance;
  PRECISION logconstant;
  PRECISION lower_bound;
  PRECISION upper_bound;
  PRECISION cumulative_lower_bound;
  PRECISION cumulative_upper_bound;
  PRECISION probability;
} TElliptical_gaussian;

typedef struct 
{
  PRECISION lower_bound;
  PRECISION upper_bound;
  PRECISION power;
  PRECISION shift;
  PRECISION multiplier;
  PRECISION logconstant;
} TElliptical_power;

typedef struct 
{
  PRECISION *t;
  int m;
  PRECISION logconstant;
} TElliptical_step;

void FreeElliptical(TElliptical *elliptical);
PRECISION RadiusElliptical(PRECISION *draw, TElliptical *elliptical);
#define DrawElliptical(draw,elliptical) ((elliptical)->draw_vector(draw,elliptical))
#define LogDensityElliptical_Radius(radius,elliptical) ((elliptical)->logdensity_radius(radius,elliptical))
#define LogDensityElliptical_Draw(draw,elliptical) ((elliptical)->logdensity_draw(draw,elliptical))
#define CummulativeDensityElliptical_Radius(radius,elliptical) ((elliptical)->cummulative_radius(radius,elliptical))
#define PrintEllipticalInfo(f_out,elliptical) ((elliptical)->print_info(f_out,elliptical))
#define EllipticalType(elliptical) ((elliptical)->type)
void PrintEllipticalInfo_full(FILE *f_out, TElliptical *elliptical);

PRECISION VarianceScale_TruncatedPower(int dim, PRECISION lower_bound, PRECISION upper_bound, PRECISION power);
PRECISION VarianceScale_TruncatedGaussian(int dim, PRECISION lower_bound, PRECISION upper_bound);
PRECISION VarianceScale_Step(int dim, PRECISION *t, int m);

TElliptical* CreateElliptical_TruncatedGaussian(int dim, TVector mean, TMatrix variance, PRECISION lower_bound, PRECISION upper_bound);
TElliptical* CreateElliptical_Gaussian(int dim, TVector mean, TMatrix variance);
TElliptical* CreateElliptical_Uniform(int dim, TVector center, TMatrix scale, PRECISION lower_bound, PRECISION upper_bound);
TElliptical* CreateElliptical_Power(int dim, TVector center, TMatrix scale, PRECISION bound, PRECISION power);
TElliptical* CreateElliptical_TruncatedPower(int dim, TVector center, TMatrix scale, 
					     PRECISION lower_bound, PRECISION upper_bound, PRECISION power);
TElliptical* CreateElliptical_Step(int dim, TVector center, TMatrix scale, PRECISION *table, int m);

#ifdef __cplusplus
}
#endif

#endif

/********************************************************************************

The random vector x is spherical and centered at zero if and only if the density 
of x depends only on the norm of x.  

The random vector z is elliptical if and only if z = S*x + c for some spherical
random vector x centered at zero.  We call the square invertiable matrix S the 
scale and the vector c the center.  If h(x) is the density of the random vector 
x, then the density of the random vector z is 

                          h(Inverse(S)*(z - c))/abs(det(S)).

Elliptical distributions can be characterized by their scale, center, and a one
dimensional distribution with non-negative support.  If r is a one-dimensional
random variable with positive support, y is the random vector that has the
uniform distribution on the n-1 dimensional unit sphere, and r and y are 
independent, then z = r*S*y + c is elliptical with scale S and center c.  If the 
density of r is given by f(r), then the density of x = r*y is given by

                                  h(x) = f(r)/S(r,n-1)

where S(r,n-1) is the surface area of the n-1 dimensional sphere of radius r.  
S(r,n-1) is given by

                   S(r,n-1) = [2*pi^(n/2))/gamma(n/2)]*(r^(n-1))

where gamma() is the gamma function.  (Note that n*gamma(n/2)=2*gamma(n/2+1)).  
This gives that the log of the density of z is given by

    -ln(abs(det(S)))-ln(2)-0.5*n*ln(pi)+ln(gamma(n/2))-(n-1)*ln(r)+ln(f(r))

where r=sqrt((z-c)'*Inverse(S*S')*(z-c)).  

---------------------------------------------------------------------------------
Note that

    E[z] = c

and

    E[(z-E[z])*(z-E[z])'] = S*E[x*x']*S' = E[r^2]*S*E[y*y']*S' 
                          = sigma^2*E[r^2]*(S*S')

where E[y*y'] = (sigma^2)*I(n) and I(n) is the n-dimensional identity matrix.  By
considering the n-dimensional standard normal distribution, which is spherical, 
one sees that:

                                  sigma^2 = 1/n

---------------------------------------------------------------------------------
The mapping to the data structure is:

  variance_scale = sigma^2*E[r^2]
  center = c
  scale = S
  quadratic_form = Inverse(S*S')
  logabsdet = -ln(abs(det(S)))
  dim = n
 
********************************************************************************/
