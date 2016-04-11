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

#include "dw_ascii.h"
#include "dw_parse_cmd.h"
#include "dw_array.h"
#include "dw_math.h"
#include "dw_error.h"
#include "dw_std.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>
#include <math.h>

/*
   Attempts to open filename for reading.  Returns pointer to file upon success 
   and prints error message and exits upon failure.  The file must exist.
*/
FILE *dw_OpenTextFile(char *filename)
{
  FILE *f=fopen(filename,"rt");
  if (!f) dw_FileError(FILE_OPEN_ERR,filename);
  return f;
}

/*
   Attempts to create filename for writing.  Returns pointer to file upon success 
   and prints error message and exits upon failure.  If the file exists, it is
   overwritten.
*/
FILE *dw_CreateTextFile(char *filename)
{
  FILE *f=fopen(filename,"wt");
  if (!f) dw_FileError(FILE_OPEN_ERR,filename);
  return f;
}

/*
   Attempts to create filename for writing.  Returns pointer to file upon success 
   and prints error message and exits upon failure.  The file is created if it 
   does not exist and is opened with the file pointer positioned at the end of
   file if it does exist.
*/
FILE *dw_AppendTextFile(char *filename)
{
  FILE *f=fopen(filename,"at");
  if (!f) dw_FileError(FILE_OPEN_ERR,filename);
  return f;
}

/*
   Returns the number of lines in the file.  The number of lines in a file is
   equal to the number of newlines ('\n') in the file if the last character is
   a newline and this number plus one if the last character is not a newline.
   An empty file will return 0 and if f is the null pointer -1 is returned.

   If f is not null, upon return the file pointer is set to the beginning of the 
   file.  One of f or filename must be non-null.
*/
int dw_NumberLines(FILE *f, char *filename)
{
  int ch, lines=0;
  FILE *f_in=f ? f : (filename ? fopen(filename,"rt") : (FILE*)NULL);
  if (!f_in) return -1;
  fseek(f_in,0,SEEK_SET);
  while (EOF != (ch=getc(f_in)))
    if (ch == '\n') lines++;
  fseek(f_in,-1,SEEK_END);
  ch=getc(f_in);
  if ((ch != '\n') && (ch != EOF)) lines++;
  fseek(f_in,0,SEEK_SET);
  if (!f) fclose(f_in);
  return lines;
}

/*
   Assumes:
     f      : valid file pointer
     buffer : pointer to character or null pointer
     n      : pointer to integer containing the length of buffer

   Returns:
     Pointer to null terminated string containing the characters from the file up
     to and including the terminating new line character.  A null pointer return
     indicates that there was a memory error or no characters to be read.  Call
     dw_GetError() to determine if a error occured.   

   Results:
     Reads line, beginning at current position from file f  Returns a pointer to 
     the buffer containing the file and resets *n if necessary.  The if the 
     passed buffer is null or is not large enough to contain the line, buffer is 
     freed and a new buffer is allocated.  Because of this, the passed buffer 
     must either null or allocated with dw_malloc(), dw_realloc(), or dw_calloc() and the 
     calling routine is responsible for eventually freeing the memory if the 
     return value is not null.  

   Notes:
     If buffer is null, then value pointed to by the pointer n is not used.
*/
#define SIZE_INCREMENT 1024
char* dw_ReadLine(FILE *f, char *buffer, int *n)
{
  char *ptr, *nbuffer;
  int i, k=0;
  if (!buffer && !(buffer=(char*)dw_malloc(*n=SIZE_INCREMENT)))
    {
      *n=0;
      return (char*)NULL;
    }
  ptr=buffer;
  while (fgets(ptr,*n-k,f))
    if (ptr[(i=(int)strlen(ptr))-1] == '\n')
      return buffer;
    else
      if (!(nbuffer=(char*)dw_realloc(buffer,*n+=SIZE_INCREMENT)))
	{
	  dw_free(buffer);
	  *n=0;
	  return (char*)NULL;
	}
      else
	ptr=(buffer=nbuffer) + (k+=i);
  if (ptr != buffer)
    return buffer;
  else
    {
      dw_free(buffer);
      *n=0;
      return (char*)NULL;
    }
}
#undef SIZE_INCREMENT

/*
   Assumes:
     n : non-negative integer
     f : valid file pointer

   Results:
     Reads and discards n lines.

*/
void dw_NextLine(FILE *f, int n)
{
  char *buffer, *ptr;
  if (!(ptr=buffer=(char*)dw_malloc(1024*sizeof(char))))
    dw_Error(MEM_ERR);
  else
    {
      while (ptr && (n-- > 0))
	while (ptr=fgets(buffer,1024 ,f))
	  if (ptr[strlen(ptr)-1] == '\n') break;
      dw_free(buffer);
    }
}

/*
   Assumes:
     buffer: a null terminated string
     delimiter : field delimiter character
     flag : one of the values defined in dw_ascii.h

   Returns:
     A string array containing the delimited fields in buffer.  The
     fields are returned according to flag. 
*/
char** dw_ParseDelimitedString(char *buffer, char delimiter, int flag)
{
  struct StringList
  {
    struct StringList *next;
    char *string;
    int length;
  } *head, *ptr;
  int k=0, n, m;
  char **v;
  if (!buffer) return (char**)NULL;
  for (head=ptr=(struct StringList*)NULL; *buffer; buffer+=buffer[n] ? n+1 : n)
    {
      if (flag & STRIP_LEADING_WHITESPACE)
	while (*buffer && (*buffer != delimiter) && isspace(*buffer)) buffer++;
      for (n=0; buffer[n] && (buffer[n] != delimiter); n++);
      if (flag & STRIP_TRAILING_WHITESPACE)
        for (m=n-1; (m >= 0) && isspace(buffer[m]); m--);
      else
        m=n-1;
      if ((m >= 0) || !(flag & REMOVE_EMPTY_FIELDS))
        {
          ptr=(struct StringList*)dw_malloc(sizeof(struct StringList));
          ptr->string=buffer;
          ptr->length=m+1;
          ptr->next=head;
          head=ptr;
          k++;
        }
    }
  v=dw_CreateArray_string(k);
  while (--k >= 0)
    {
      v[k]=(char*)dw_malloc(head->length+1);
      if (head->length > 0) memcpy(v[k],head->string,head->length);
      v[k][head->length]='\0';
      ptr=head;
      head=head->next;
      dw_free(ptr);
    }
  return v;
}

/*
   Assumes
     f:  valid file pointer
     delimiter:  field deliniter.
     flag:  one of the values defined in dw_ascii.h

   Returns
     One-dimensional string array of the delimited fields of the current line of 
     the file f or a null pointer.

   Notes
     The file is read starting from the current file position.  If the file 
     contains no fields or there is a memory error, then a null pointer is 
     returned.  The delimiter character defines the fields in each row and the 
     new line character defines the rows. 
*/
char** dw_ReadDelimitedLine(FILE *f, char delimiter, int flag)
{
  int n=0;
  char **v=(char**)NULL, *buffer=dw_ReadLine(f,(char*)NULL,&n);
  if (buffer)
    {
      v=dw_ParseDelimitedString(buffer,delimiter,flag);
      dw_free(buffer);
    }
  return v;
}

/*
   Assumes
     f:  valid file pointer or null pointer.
     filename:  pointer to null terminated string or null pointer.
     delimiter:  field deliniter.
     flag:  one of the values defined in dw_ascii.h

   Returns
     Two-dimensional string array of the deliminted fields of f or a null 
     pointer.

   Notes
     One of f and filename should be non-null.  If f is non-null, the file is 
     read starting from the current file position.  If f is null, an attempt is
     made to open the file.  If successful, the file is read from the beginning.
     If the file does not exist or contains no fields, then a null pointer is 
     returned.  The delimiter character defines the fields in each row and the 
     new line character defines the rows. 

*/
char*** dw_ReadDelimitedFile(FILE *f, char* filename, char delimiter, int flag)
{
  struct LineList
    {
      struct LineList *next;
      char **line;
    } *head=(struct LineList*)NULL, *ptr;
  int n=0;
  char **v, ***M=(char***)NULL;
  FILE *f_in=f ? f : fopen(filename,"rt");
  if (f_in)
    {
      do
	if (v=dw_ReadDelimitedLine(f_in,delimiter,flag))
          {
	    ptr=(struct LineList*)dw_malloc(sizeof(struct LineList));
	    ptr->line=v;
            ptr->next=head;
            head=ptr;
            n++;
          }
      while (feof(f_in) || ferror(f_in));
      if (!f) fclose(f_in);
      if (n > 0)
        {
          M=(char***)dw_CreateArray_array(n);
          while (--n >= 0)
            {
              M[n]=head->line;
              ptr=head;
              head=head->next;
              dw_free(ptr);
            }
        }
    }
  return M;
}

int dw_PrintDelimitedArray(FILE *f, void* array, char delimiter)
{
  char format[4];
  format[0]='%';
  format[1]='s';
  format[2]=delimiter;
  format[3]='\0';
  return dw_PrintArray(f,array,format);
}

/*
   Assumes
     f:  valid file pointer
     delimiter:  field deliniter.
     flag: one of the values defined in dw_ascii.h
     invalid_identifier: PRECISION

   Returns
     A PRECISION array containing the value of the fields with valid real numbers.  
     If the flag sets REMOVE_EMPTY_FIELDS, invalid fields are not included.  If
     REMOVE_EMPTH_FIELDS is not set, then invalid fields are included and set to
     the value invalid_identifier.

   Notes
     The file is read starting from the current file position.  If the file 
     contains no fields or there is a memory error, then a null pointer is 
     returned.  The delimiter character defines the fields in each row and the 
     new line character defines the rows. 

     The calling routine must use dw_FreeArray() to deallocate the returned 
     array.
*/
PRECISION* dw_ReadDelimitedLine_floating(FILE *f, char delimiter, int flag, PRECISION invalid_identifier)
{
  char **s;
  PRECISION *x, *y;
  int i, j, n;
  if (!(s=dw_ReadDelimitedLine(f,delimiter,STRIP_WHITESPACE | (flag & REMOVE_EMPTY_FIELDS))))
    return (PRECISION*)NULL;
  if (flag & REMOVE_EMPTY_FIELDS)
    {
      y=(PRECISION*)dw_malloc((n=dw_DimA(s))*sizeof(PRECISION));
      for (i=j=0; i < n; i++)
	if (dw_IsFloat(s[i])) y[j++]=atof(s[i]);
      if (j > 0) 
	{
	  x=dw_CreateArray_floating(j);
	  memcpy(x,y,j*sizeof(PRECISION));
	}
      else
	x=(PRECISION*)NULL;
      dw_free(y);
    }
  else
    {
      x=dw_CreateArray_floating(dw_DimA(s));
      for (i=dw_DimA(s)-1; i >= 0; i--)
	x[i]=dw_IsFloat(s[i]) ? atof(s[i]) : invalid_identifier;
    }
  dw_FreeArray(s);
  return x;
}

/*
   Assumes
     f:  valid file pointer or null pointer.
     filename:  pointer to null terminated string or null pointer.
     delimiter:  field deliniter.
     flag:  one of the values defined in dw_ascii.h
     invalid_identifier: PRECISION

   Returns
     Two-dimensional array of PRECISION or a null pointer.

   Notes
     One of f and filename should be non-null.  If f is non-null, the file is 
     read starting from the current file position.  If f is null, an attempt is
     made to open the file.  If successful, the file is read from the beginning.
     If the file does not exist or contains no fields, then a null pointer is 
     returned.  The delimiter character defines the fields in each row and the 
     new line character defines the rows. 

     The calling routine must use dw_FreeArray() to deallocate the returned 
     array.
*/
PRECISION** dw_ReadDelimitedFile_floating(FILE *f, char* filename, char delimiter, int flag, PRECISION invalid_identifier)
{
  struct LineList
    {
      struct LineList *next;
      PRECISION *line;
    } *head=(struct LineList*)NULL, *ptr;
  int n=0;
  PRECISION *x, **X=(PRECISION**)NULL;
  FILE *f_in=f ? f : fopen(filename,"rt");
  if (f_in)
    {
      do
	if (x=dw_ReadDelimitedLine_floating(f_in,delimiter,flag,invalid_identifier)) 
          {
	    ptr=(struct LineList*)dw_malloc(sizeof(struct LineList));
	    ptr->line=x;
            ptr->next=head;
            head=ptr;
            n++;
          }
      while (!feof(f_in) && !ferror(f_in));
      if (!f) fclose(f_in);
      if (n > 0)
        {
          X=(PRECISION**)dw_CreateArray_array(n);
          while (--n >= 0)
            {
              X[n]=head->line;
              ptr=head;
              head=head->next;
              dw_free(ptr);
            }
        }
    }
  return X;
}

/*
   Assumes:
     f         : valid file pointer
     delimiter : field terminator
     terminal  : line terminator
     flag      : determine how characters are processed
     buffer    : pointer to pointer to character or null pointer
     n         : pointer to integer containing the length of buffer

   Returns:
     0 : memory error occured
     1 : field read, terminated by delimiter
     2 : field read, terminated by terminal
     3 : field read, terminated by EOF

   Results:
     If necessary, memory ia reallocated.  The length of this reallocated memory 
     is stored in n.  It is the calling routines responsibility to free the
     memory pointed to by *buffer.  

   Notes:
     flag values
       ALLOW_QUOTED_TEXT
         If set the delimiter and terminal characters do not stop processing when
         encountered between quotes.  To produce a quote in quoted text, use two
         consectutive quotes.  Outside quoted text, a quote always begins quoted
         text.

       PRINTABLE_ONLY_IN_QUOTES

       PRINTABLE_ONLY

       STRIP_LEADING_WHITESPACE

       STRIP_TRAILING_WHITESPACE

       STRIP_WHITESPACE


*/
//#define INCREMENT 1024
//int dw_ReadDelimitedField(FILE *f, int delimiter, int terminal, int flag, char **buffer, int *n)
//{
/*   int ch;        // next character read */
/*   int k=0;       // position to store next char, always less than *n */
/*   int quoted=0; */
/*   int leading=(flag & STRIP_LEADING_WHITESPACE) ? 1 : 0; */
/*   char *ptr; */

/*   ch=fgetc(f); */

/*   while (ch != EOF) */
/*     { */
/*       //=== reallocate memory if necessary */
/*       if (k+1 > *n) */
/* 	if (!(ptr=(char*)dw_realloc(buffer,*n+=INCREMENT))) */
/* 	  { */
/* 	    *n-=INCREMENT; */
/* 	    return 0; */
/* 	  } */
/* 	else */
/* 	  buffer=ptr; */

/*       //=== process character */
/*       if (quoted) */
/* 	{ */
/* 	  if (ch == '"') */
/* 	    if ((ch=fgets(f)) != '"')  */
/* 	      { */
/* 		quoted=0; */
/* 		continue; */
/* 	      } */
/* 	  if (!(flag & PRINTABLE_ONLY_IN_QUOTES) || isprint(ch))  */
/* 	    buffer[k++]=ch; */
/* 	} */
/*       else */
/* 	if ((ch == delimiter) || (ch == terminal))  */
/* 	  break; */
/* 	else */
/* 	  if ((ch == '"') && (flag & ALLOW_QUOTED_TEXT)) */
/* 	    quoted=1; */
/* 	  else */
/* 	    if (!(flag & PRINTABLE_ONLY) || isprint(ch)) */
/* 	      { */
/* 		if ((ch == "\r") && (terminal == '\n')) */
/* 		  { */
/* 		    if ((ch=fgetc(f)) == '\n') break; */
/* 		    if (!leading) buffer[k++]='\r'; */
/* 		    continue; */
/* 		  } */
/* 		if (leading) */
/* 		  if (isspace(ch))  */
/* 		    { */
/* 		      ch=fgetc(f); */
/* 		      continue; */
/* 		    } */
/* 		  else */
/* 		    leading=0; */
/* 		buffer[k++]=ch; */
/* 	      } */

/*       ch=fgets(f); */
/*     } */

/*   buffer[k]='\0'; */

/*   return (ch == EOF) ? 3 : (ch == terminal) ? 2 : 1;  */
//}
//#undef INCREMENT

/*
   Returns 1 if the null terminated string id is found at the beginning of a line
   in the file and 0 otherwise.  The file pointer is set to the line immediately 
   after the line containing id.  The search starts at the current position of
   the file.  If id is not found, then the file is rewound and the search is 
   continued until the initial file position is passed.
*/
int dw_SetFilePosition(FILE *f, char *id)
{
  char *buffer=(char*)NULL;
  int m, n, pos;
  if ((n=(int)strlen(id)) > 0)
    {
      pos=ftell(f);
      while (buffer=dw_ReadLine(f,buffer,&m))
	if (!memcmp(buffer,id,n)) 
	  {
	    dw_free(buffer);
	    return 1;
	  }
      if (pos > 0)
	{
	  rewind(f);
	  while ((ftell(f) < pos) && (buffer=dw_ReadLine(f,buffer,&m)))
	    if (!memcmp(buffer,id,n))
	      {
		dw_free(buffer);
		return 1;
	      }
	  if (buffer) dw_free(buffer);
	}
    }
  return 0;
}

/*
   Returns 1 if the null terminated string id is found at the beginning of a line
   in the file and 0 otherwise.  The file pointer is set to the line immediately 
   after the line containing id.  The search starts at the current position of
   the file.  If id is not found, the file is not rewound.
*/
int dw_SetFilePositionNoRewind(FILE *f, char *id)
{
  char *buffer=(char*)NULL;
  int m, n;
  if ((n=(int)strlen(id)) > 0)
    {
      while (buffer=dw_ReadLine(f,buffer,&m))
	if (!memcmp(buffer,id,n)) 
	  {
	    dw_free(buffer);
	    return 1;
	  }
      if (buffer) dw_free(buffer);
    }
  return 0;
}


/* int dw_SetFilePositionBySection(FILE *f, int n, ...) */
/* { */
/*   char *arg; */
/*   int i; */
/*   va_list ap; */
/*   rewind(f); */
/*   va_start(ap,n); */
/*   for (i=0; i  < n; i++) */
/*     if (!(arg=va_arg(ap,char*)) || !dw_SetFilePositionNoRewind(f,arg)) */
/*       { */
/* 	va_end(ap); */
/* 	return 0; */
/*       } */
/*   va_end(ap); */
/*   return 1; */
/* } */

char* dw_DuplicateString(char *buffer)
{
  char *rtrn=(char*)NULL;
  if (buffer && (rtrn=(char*)dw_malloc(strlen(buffer)+1))) strcpy(rtrn,buffer);
  return rtrn;
}

#undef READLINE_NOERR  
#undef READLINE_MEMERR 
#undef READLINE_EOF    
