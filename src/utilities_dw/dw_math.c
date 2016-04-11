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

#include "dw_math.h"
#include "dw_std.h"
#include <math.h>
#include <ctype.h>
#include "gsl/gsl_sf_gamma.h"


/*
   Returns ln(exp(a) + exp(b)) computed to avoid overflow.  If
   a = ln(c) and b = ln(d), as is usually the case, then the
   routine returns ln(c + d).

*/
double AddLogs(double a, double b)
{
  return (a > b) ? a + log(1.0 + exp(b-a)) : b + log(exp(a-b) + 1.0);
}

/*
   Returns ln(x*exp(a) + y*exp(b)) computed to avoid overflow.  If a = ln(c) and 
   b = ln(d), as is usually the case, then the routine returns ln(x*c + y*d).
   Both x and y must be non-negative.  It must be the case that x and y are non-
   negative. 
*/
double AddScaledLogs(double x, double a, double y, double b)
{
  if (x > 0)
    if (y > 0)
      return (a > b) ? a + log(x + y*exp(b-a)) : b + log(x*exp(a-b) + y);
    else
      return log(x) + a;
  else
    return (y > 0) ? log(y) + b : MINUS_INFINITY;
}

/*
  A floating point number is of the form

  [white space][+/-]digits[.[digits]][E/e[+/-]digits]white space/null character

  or

  [white space][+/-].digits[E/e[+/-]digits]white space/null character

  where characters in square brackets are optional.

  Returns one if valid floating point number and zero otherwise.
*/
int dw_IsFloat(const char *buffer)
{
 int i=0;

 if (!buffer) return 0;

 /* Strip leading white space */
 while (isspace(buffer[i])) i++;

 /* Mantissa OK? */
 if ((buffer[i] == '+') || (buffer[i] == '-')) i++;
 if (isdigit(buffer[i]))
   {
    while (isdigit(buffer[++i]));
    if ((buffer[i] == '.'))
     while (isdigit(buffer[++i]));
   }
  else
   if ((buffer[i] == '.'))
     if (isdigit(buffer[++i]))
       while (isdigit(buffer[++i]));
      else
       return 0;
    else
     return 0;

 /* Is exponent OK? */
 if ((buffer[i] == 'e') || (buffer[i] == 'E'))
  {
   if ((buffer[++i] == '+') || (buffer[i] == '-')) i++;
   if (isdigit(buffer[i]))
     while (isdigit(buffer[++i]));
    else
     return 0;
  }

 /* Is end of string or trailing white space */
 if (buffer[i] && !isspace(buffer[i])) return 0;

 return 1;
}

/*
  Integers are of the form

   [white space][+/-]digits[.]white space/null character

  where characters in square brackets are optional.

  Returns one if valid integer and zero otherwise.
*/
int dw_IsInteger(const char *buffer)
{
 int i=0;

 if (!buffer) return 0;

 /* Strip leading white space */
 while (isspace(buffer[i])) i++;

 /* Leading sign */
 if ((buffer[i] == '+') || (buffer[i] == '-')) i++;

 /* At least one digits possibly followed by decimal point */
 if (isdigit(buffer[i]))
   {
    while (isdigit(buffer[++i]));
    if ((buffer[i] == '.')) i++;
   }
  else
   return 0;

 /* Is end of string or trailing white space */
 if (buffer[i] && !isspace(buffer[i])) return 0;

 return 1;
}

/*
   Returns the natural logrithm of the gamma function applied to x.  The gamma 
   function of x is the integral from 0 to infinity of t^(x-1)*exp(-t)dt.
*/
double dw_log_gamma(double x)
{
  return gsl_sf_lngamma(x);
}

/*
   Returns the beta function applied to (a,b).  The beta function of (a,b) is 
   Gamma(a)*Gamma(b)/Gamma(a+b), which is equal to the integral from 0 to 1 of
   x^a * (1-x)^b dx.  Neither a nor b can be negative integers.
*/
double dw_beta(double a, double b)
{
  return gsl_sf_beta(a,b);
}

/*
   Returns the normalized incomplete beta function.  The normalized incomplete
   beta function is the integral from 0 to x of (x^a * (1-x)^b)/beta(a,b) dx.
   Neither a nor b can be negative integers and 0 <= x <= 1.
*/
double dw_incomplete_beta(double x, double a, double b)
{
  return gsl_sf_beta_inc(a,b,x);
}
