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

#include "dw_parse_cmd.h"
#include "dw_math.h"
#include "dw_std.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define ARGUMENT_ID_1 '-'
#define ARGUMENT_ID_2 '/'

/* 
   Searches args for a leading ARGUMENT_ID followed by the character opt.  Returns
   the index if found and -1 otherwise.
*/
int dw_FindArgument(int nargs, char **args, char opt)
{
  int i;
  for (i=nargs-1; i >= 0; i--)
    if (((args[i][0] == ARGUMENT_ID_1) || (args[i][0] == ARGUMENT_ID_2)) && (args[i][1] == opt) && (args[i][2] == '\0')) break;
  return i;
}

/*
   Searches for the last argument whose leading character is ARGUMENT_ID 
   followed by the character opt.  If such an argument is not found, then the 
   integer def is returned.  If such an argument is found and there is an i+1
   argument and its characters form a valid integer, then this integer is
   returned.  Otherwise def is returned.  
*/
int dw_ParseInteger(int nargs, char **args, char opt, int def)
{
  int i=dw_FindArgument(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs) && dw_IsInteger(args[i+1])) return atoi(args[i+1]);
  return def;
}

/*
   Searches for the last argument whose leading character is ARGUMENT_ID
   followed by the character opt.  If such an argument is not found, then the 
   double def is returned.  If such an argument is found and there is an i+1 
   argument and its characters form a valid floating point number, then this 
   value is returned.  Otherwise def is returned.  
*/
double dw_ParseFloating(int nargs, char **args, char opt, double def)
{
  int i=dw_FindArgument(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs) && dw_IsFloat(args[i+1])) return atof(args[i+1]);
  return def;
}


/*
   Searches for the last argument whose leading character is ARGUMENT_ID
   followed by the character opt.  If such an argument is not found, then the 
   pointer def is returned.  If such an argument is found and there is an i+1 
   argument then a pointer to this argument is returned.  Otherwise def is 
   returned.  
*/
char* dw_ParseString(int nargs, char **args, char opt, const char *def)
{
  int i=dw_FindArgument(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs)) return args[i+1];
  return (char*)def;
}

/* 
   Searches args for a leading ARGUMENT_ID followed by the string opt.  Returns
   the index if found and -1 otherwise.
*/
int dw_FindArgument_String(int nargs, char **args, const char *opt)
{
  int i;
  for (i=nargs-1; i >= 0; i--)
    if (((args[i][0] == ARGUMENT_ID_1) || (args[i][0] == ARGUMENT_ID_2)) && !strcmp(args[i]+1,opt)) break;
  return i;
}

/*
   Searches for the last argument whose leading character is a ARGUMENT_ID
   followed by the string opt.  If such an argument is not found, then the 
   integer def is returned.  If such an argument is found and there is an i+1 
   argument and its characters form a valid integer, then this integer is 
   returned.  Otherwise def is returned.  
*/
int dw_ParseInteger_String(int nargs, char **args, const char *opt, int def)
{
  int i=dw_FindArgument_String(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs) && dw_IsInteger(args[i+1])) return atoi(args[i+1]);
  return def;
}

/*
   Searches for the last argument whose leading character is ARGUMENT_ID
   followed by the string opt.  If such an argument is not found, then the 
   double def is returned.  If such an argument is found and there is an i+1 
   argument and its characters form a valid floating point number, then this 
   value is returned.  Otherwise def is returned.  
*/
double dw_ParseFloating_String(int nargs, char **args, const char *opt, double def)
{
  int i=dw_FindArgument_String(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs) && dw_IsFloat(args[i+1])) return atof(args[i+1]);
  return def;
}


/*
   Searches for the last argument whose leading character is ARGUMENT_ID
   followed by the string opt.  If such an argument is not found, then the 
   pointer def is returned.  If such an argument is found and there is an i+1 
   argument, then a pointer to this argument is returned.  Otherwise def is 
   returned.  
*/
char* dw_ParseString_String(int nargs, char **args, const char *opt, const char *def)
{
  int i=dw_FindArgument_String(nargs,args,opt);
  if ((i != -1) && (i+1 < nargs)) return args[i+1];
  return (char*)def;
}

