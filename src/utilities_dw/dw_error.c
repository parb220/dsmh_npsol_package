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

#include "dw_error.h"
#include "dw_std.h"
#include <stdlib.h>
#include <string.h>

#define ERROR_MESSAGE_BUFFER_LENGTH 256
static int ERROR_FLAG=NO_ERR;
static char ERROR_MESSAGE[ERROR_MESSAGE_BUFFER_LENGTH]="";
static int TerminalErrors=ALL_ERRORS;
static int VerboseErrors=ALL_ERRORS;
static FILE* f_err=(FILE*)NULL;
static char* filename_err=(char*)NULL;

/*
   Returns the value of the current error flag.
*/
int dw_GetError(void)
{
  return ERROR_FLAG;
}

/*
   Returns pointer to current error message.  This buffer should not be modified
   or freed.
*/
char* dw_GetErrorMessage(void)
{
  return ERROR_MESSAGE;
}

/*
   Clears the error flag and returns the value of the previous flag.  This is the
   most efficient way to clear the error flag and message. 
*/
int dw_ClearError(void)
{
  int rtrn=ERROR_FLAG;
  ERROR_FLAG=NO_ERR;
  ERROR_MESSAGE[0]='\0';
  return rtrn;
}

/*
   Sets the error flag to err, the error message to msg and returns the value of 
   the previous flag.  If the null terminated string msg is longer than 255 
   characters, only the first 255 characters are used.  If msg is null, then a
   predefined error message is used.
*/
int dw_SetError(int err, char *msg)
{
  int rtrn=ERROR_FLAG;
  if (msg)
    switch (ERROR_FLAG=err)
      {
      case MEM_ERR:
      case FILE_ERR:
      case FILE_OPEN_ERR:
      case FILE_READ_WRITE_ERR:
      case PARSE_ERR:
      case FLOAT_ERR:
      case NULL_ERR:
      case ARG_ERR:
      case ITERATION_ERR:
      case NOT_IMPLEMENTED_ERR:
      case SIZE_ERR:
      case SING_ERR:
      case POSDEF_ERR:
      case BLAS_LAPACK_ERR:
      case USER_ERR:
	strncpy(ERROR_MESSAGE,msg,ERROR_MESSAGE_BUFFER_LENGTH-1);
	ERROR_MESSAGE[ERROR_MESSAGE_BUFFER_LENGTH-1]='\0'; 
	break;
      case NO_ERR:
	ERROR_MESSAGE[0]='\0';
	break;
      default:
	ERROR_FLAG=UNKNOWN_ERR;
	strcpy(ERROR_MESSAGE,"Unknown error.");
	break;
      }
  else
    switch (ERROR_FLAG=err)
      {
      case MEM_ERR:
	strcpy(ERROR_MESSAGE,"Out of memory.");
	break;
      case FILE_ERR:
        strcpy(ERROR_MESSAGE,"File operation error.");
	break;
      case FILE_OPEN_ERR:
	strcpy(ERROR_MESSAGE,"Unable to open/create file.");
	break;
      case FILE_READ_WRITE_ERR:
	strcpy(ERROR_MESSAGE,"Unable to read/write to file.");
	break;
      case PARSE_ERR:
	strcpy(ERROR_MESSAGE,"Error parsing data.");
	break;
      case FLOAT_ERR:
	strcpy(ERROR_MESSAGE,"Floating point error.");
	break;
      case NULL_ERR:
	strcpy(ERROR_MESSAGE,"Unexpected null pointer encountered.");
	break;
      case ARG_ERR:
	strcpy(ERROR_MESSAGE,"Argument error.");
	break;
      case ITERATION_ERR:
	strcpy(ERROR_MESSAGE,"Maximum iteration limit exceeded.");
	break;
      case NOT_IMPLEMENTED_ERR:
	strcpy(ERROR_MESSAGE,"Feature not yet implemented.");
	break;
      case SIZE_ERR:
	strcpy(ERROR_MESSAGE,"Matrices/vectors not conformable.");
	break;
      case SING_ERR:
	strcpy(ERROR_MESSAGE,"Singular matrix.");
	break;
      case POSDEF_ERR:
	strcpy(ERROR_MESSAGE,"Matrix not positive definite.");
	break;
      case BLAS_LAPACK_ERR:
	strcpy(ERROR_MESSAGE,"Blas/Lapack error.");
	break;
      case USER_ERR:
        strcpy(ERROR_MESSAGE,"Undocumented error.");
	break;
      case NO_ERR:
	ERROR_MESSAGE[0]='\0';
	break;
      default:
	ERROR_FLAG=UNKNOWN_ERR;
	strcpy(ERROR_MESSAGE,"Unknown error.");
	break;
      }
  if (VerboseErrors & ERROR_FLAG)
    if (f_err)
      fprintf(f_err,"%s\n",ERROR_MESSAGE);
    else
      printf("%s\n",ERROR_MESSAGE);
  if (TerminalErrors & ERROR_FLAG) dw_exit(ERROR_FLAG);
  return rtrn;
}

/*
   Sets the error flag and to err, sets the error message to the predefined error
   message, and returns the value of the previous error flag.  
*/
int dw_Error(int err)
{
  return dw_SetError(err,(char*)NULL);
}

/*
   Sets the error flag and to USER_ERR, sets the error message to msg, and 
   returns the value of the previous error flag.  
*/
int dw_UserError(char *msg)
{
  return dw_SetError(USER_ERR,msg);
}

/*
   Sets the error flag error message using the passed filename.  Returns the 
   value of the previous error flag.  
*/
int dw_FileError(int err, char *filename)
{
  char *msg, *fmt;
  if (filename)
    {
      switch(err)
	{
	case FILE_ERR:
	  fmt="File operation error on %s"; break;
	case FILE_OPEN_ERR:
	  fmt="Unable to open/create %s"; break;
	case FILE_READ_WRITE_ERR:
	  fmt="Unable to read/write %s"; break;
	default:
	  return dw_SetError(UNKNOWN_ERR,(char*)NULL);
	}
      sprintf(msg=(char*)dw_malloc(strlen(fmt)+strlen(filename)-1),fmt,filename);
      err=dw_SetError(err,msg);
      dw_free(msg);
      return err;
    }
  else
    switch(err)
      {
      case FILE_ERR:             
      case FILE_OPEN_ERR:        
      case FILE_READ_WRITE_ERR:  
	return dw_SetError(err,(char*)NULL);
      default:                   
	return dw_SetError(UNKNOWN_ERR,(char*)NULL);
      }
}

/*
   Sets errors which terminate program.  The integer err should be a combination
   of the error flags defined in dw_error.h.
*/
int dw_SetTerminalErrors(int err)
{
  int rtrn=TerminalErrors;
  TerminalErrors=err & ALL_ERRORS;
  return rtrn;
}

/*
  Returns the current terminal errors.
*/
int dw_GetTerminalErrors(void)
{
  return TerminalErrors;
}

/*
   Sets errors which causes program to print a error message to f_err.  The 
   integer err should be a combination of the error flags defined in dw_error.h.
*/
int dw_SetVerboseErrors(int err)
{
  int rtrn=VerboseErrors;
  VerboseErrors=err & ALL_ERRORS;
  return rtrn;
}

/*
  Returns the current verbose errors.
*/
int dw_GetVerboseErrors(void)
{
  return VerboseErrors;
}

/*
   Creates file for writing error messages.  Returns one upon success and zero
   upon failure.  Upon failure, error messages will be sent to the console.  
*/
int dw_CreateErrorMessageFile(char *filename)
{
  if (!filename)
    dw_ConsoleErrorMessage();
  else
    {
      if (f_err) fclose(f_err);
      if (filename_err) dw_free(filename_err);
      if (f_err=fopen(filename,"wt"))
	if (filename_err=(char*)malloc(strlen(filename)+1))
	  {
	    strcpy(filename_err,filename);
	    return 1;
	  }
	else
	  {
	    dw_Error(MEM_ERR);
	    dw_ConsoleErrorMessage();
	  }
      else
	{
	  filename_err=(char*)NULL;
	  dw_FileError(FILE_OPEN_ERR,filename);
	}
    }
  return 0;
}

/*
   Opens file for appending error messages.  Returns one upon success and zero
   upon failure.  Upon failure, error messages will be sent to the console.  
*/
int dw_AppendErrorMessageFile(char *filename)
{
  if (!filename)
    dw_ConsoleErrorMessage();
  else
    {
      if (f_err) fclose(f_err);
      if (filename_err) dw_free(filename_err);
      if (f_err=fopen(filename,"at"))
	if (filename_err=(char*)malloc(strlen(filename)+1))
	  {
	    strcpy(filename_err,filename);
	    return 1;
	  }
	else
	  {
	    dw_Error(MEM_ERR);
	    dw_ConsoleErrorMessage();
	  }
      else
	{
	  filename_err=(char*)NULL;
	  dw_FileError(FILE_OPEN_ERR,filename);
	}
    }
  return 0;
}

/*
   Sends error messages to the console (stdout).
*/
void dw_ConsoleErrorMessage(void)
{
  if (f_err) fclose(f_err);
  if (filename_err) dw_free(filename_err);
  f_err=(FILE*)NULL;
  filename_err=(char*)NULL;
}

/*
   Get current error file pointer.  This allows the user to:
     1) Print an error message without changing the current error number.
     2) Not have to construct a complex error message string.
   The returned file pointer must not be closed.  Because this circumvents the 
   error handling framework, it should be used with care.
*/
FILE* dw_GetErrorFilePointer(void)
{
  return f_err ? f_err : stdout;
}

/*
   Get current error filename.  A null return indicates that error messages are
   sent to the console (stdout).  The returned string must not be modified in any
   way.  This function is useful if the user wants to 
     1) Determine where error messages are currently being sent.
     2) Change the current error file  via a dw_CreateErrorMessage(),
        dw_AppendErrorMessage(), or dw_ConsoleErrorMessage() call, and then 
        reset it via a dw_AppendErrorMessageFile() or dw_ConsoleErrorMessage()
        call.
*/
char* dw_GetErrorFilename(void)
{
  return filename_err;
}
