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

#ifndef __PARSE_COMMAND_LINE__
#define __PARSE_COMMAND_LINE__

#ifdef __cplusplus
extern "C"
{
#endif

int dw_FindArgument(int nargs, char **args, char opt);
int dw_ParseInteger(int nargs, char **args, char opt, int def);
double dw_ParseFloating(int nargs, char **args, char opt, double def);
char* dw_ParseString(int nargs, char **args, char opt, const char *def);

int dw_FindArgument_String(int nargs, char **args, const char *opt);
int dw_ParseInteger_String(int nargs, char **args, const char *opt, int def);
double dw_ParseFloating_String(int nargs, char **args, const char *opt, double def);
char* dw_ParseString_String(int nargs, char **args, const char *opt, const char *def);

#ifdef __cplusplus
}
#endif

#endif
