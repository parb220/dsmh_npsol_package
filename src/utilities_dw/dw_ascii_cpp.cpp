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

#include "dw_ascii.hpp"
#include "dw_exception.hpp"

double _stod(const std::string &str, size_t *idx)
{
        char* pEnd;
        double value=strtod(str.c_str(), &pEnd);
        if (idx != NULL)
                *idx = (size_t)(pEnd-str.c_str());
        return value;
}

int _stoi(const std::string &str, size_t *idx)
{
        char * pEnd;
        int value = strtol(str.c_str(), &pEnd, 0);
        if (idx != NULL)
                *idx = (size_t)(pEnd-str.c_str());
        return value;
}

std::string cluster_to_string(int i)
{
        std::stringstream convert;
        convert.str(std::string());
        convert << i;
        return convert.str();
}

//===============================================================================
//=== class StringVector
//===============================================================================
void StringVector::ParseLine(const std::string &line, char delimiter, int flag)
{
  int i=0, m=line.length(), k, j;
  while (i < m)
    {
      if (flag & SV_STRIP_LEADING_WHITESPACE)
	while ((i < m) && isspace(line[i])) i++;

      if (i < m)
	{
	  for (k=i; (k < m) && (line[k] != delimiter); k++);
	  j=k-1;
	  if (flag & SV_STRIP_TRAILING_WHITESPACE)
	    while ((j >= i) && isspace(line[j])) j--;
	  if (j >= i)
	    push_back(line.substr(i,j-i+1));
	  else if (!(flag & SV_REMOVE_EMPTY_FIELDS))
	    push_back("");
	  i=k+1;
	}
    }
}

void StringVector::ParseLine(std::istream &input, char delimiter, int flag)
{
  std::string line;
  getline(input,line);
  ParseLine(line,delimiter,flag);
}

std::ostream& operator<<(std::ostream &output, StringVector &string_vector)
{
  for (unsigned int i=0; i < string_vector.size(); i++) output << string_vector[i] << ' ';
  output << std::endl;
  return output;
}

// Reads and parses line starting from the current postion.  Default
// values are used for the delimiter and flags
std::istream& operator>>(std::istream &input, StringVector &string_vector)
{
  string_vector.clear();
  string_vector.ParseLine(input);
  return input;
}
//===============================================================================
//===============================================================================
//===============================================================================

//===============================================================================
//=== class StringMatrix
//===============================================================================
// Reads and parses entire file starting from the current position by making 
// repeated calls to AddLine().  Returns the number of lines appended.
int StringMatrix::ParseLines(std::istream &input, char delimiter, int flag)
{
  int k=0;
  while (input.good()) k+=AddLine(input,delimiter,flag);
  return k;
}

// Reads and parses entire file starting from the current position by making 
// repeated calls to AddLine().  Returns the number of lines appended.
int StringMatrix::ParseLines(std::istream &input, const std::string &comment, char delimiter, int flag)
{
  int k=0;
  while (input.good()) k+=AddLine(input,comment,delimiter,flag);
  return k;
}

// Reads and parses up to n lines from the input stream.  Less than n lines may 
// be appended if there were blank lines that were ignored.  Returns the number of
// lines appended.
int StringMatrix::ParseLines(int n, std::istream &input, char delimiter, int flag)
{
  int k=0;
  for (int i=0; i < n; i++) k+=AddLine(input,delimiter,flag);
  return k;
}

// Reads and parses up to n lines from the input stream.  Less than n lines may 
// be appended if there were blank lines or comments that were ignored.
int StringMatrix::ParseLines(int n, std::istream &input, const std::string &comment, char delimiter, int flag)
{
  int k=0;
  for (int i=0; i < n; i++) k+=AddLine(input,comment,delimiter,flag);
  return k;
}

// Parse the string line.  Appends the line if one of the following is true:
//  (1) the number of fields is positve
//  (2) the REMOVE_EMPTY_LINE bit is not set
// Returns the number of lines appended.
int StringMatrix::AddLine(const std::string &line, char delimiter, int flag) 
{ 
  StringVector string_vector(line,delimiter,flag);
  if ((string_vector.size() == 0) && (flag & SV_REMOVE_EMPTY_LINES)) 
    return 0;
  else
    {
      push_back(string_vector);
      return 1;
    }
}

// Parse the string line.  Appends the line if either of the following is true:
//  (1) the number of fields is positive and the initial segment of field zero is 
//      not equal to comment
//  (2) the number of fields is zero and the REMOVE_EMPTY_LINE bit is not set
// Returns the number of lines appended.
int StringMatrix::AddLine(const std::string &line, const std::string &comment, char delimiter, int flag)
{
  StringVector string_vector(line,delimiter,flag);
  if (string_vector.size() > 0)
    if (string_vector[0].compare(0,comment.length(),comment) == 0)
      return 0;
    else
      {
	push_back(string_vector);
	return 1;
      }
  else
    if (flag & SV_REMOVE_EMPTY_LINES)
      return 0;
    else
      {
	push_back(string_vector);
	return 1;
      }
}

// Reads and parses the next line from the imput stream.  Calls Addline().
int StringMatrix::AddLine(std::istream &input, char delimiter, int flag)
{ 
  std::string line;
  getline(input,line);
  return input.good() ? AddLine(line,delimiter,flag) : 0;
}

// Reads and parses the next line from the imput stream.  Calls Addline().
int StringMatrix::AddLine(std::istream &input, const std::string &comment, char delimiter, int flag)
{
  std::string line;
  getline(input,line);
  return input.good() ? AddLine(line,comment,delimiter,flag) : 0;
}

TDenseMatrix StringMatrix::GetNumericMatrix(int pos, int n_rows, int n_cols) const
{
  if ((n_rows <= 0) || (pos < 0) || (pos+n_rows > (int)size()))
    throw dw_exception("GetNumericMatrix(): invalid arguments");
  if (n_cols < 0) n_cols=this->operator[](pos).size();
  for (int i=n_rows-1; i >= 0; i--) 
    if ((int)(*this)[pos+i].size() != n_cols)
      throw dw_exception("GetNumericMatrix(): rows not of required length");
  TDenseMatrix M(n_rows,n_cols);
  try
    {

      for (int i=n_rows-1; i >= 0; i--)
	for (int j=n_cols-1; j >= 0; j--)
	  M(i,j)=_stod((*this)[pos+i][j]);
    }
  catch (std::exception)
    {
      throw dw_exception("GetNumericMatrix(): non-numeric entry");
    }
  return M;
}

TDenseVector StringMatrix::GetNumericVector(int pos, int n_cols) const
{
  if ((pos < 0) || (pos >= (int)size()))
    throw dw_exception("GetNumericVector(): invalid arguments");
  if (n_cols < 0) 
    n_cols=(*this)[pos].size();
  else
    if (n_cols != (int)(*this)[pos].size())
      throw dw_exception("GetNumericVector(): row not of required length");
  TDenseVector v(n_cols);
  try
    {
      for (int j=n_cols-1; j >= 0; j--) v(j)=_stod((*this)[pos][j]);
    }
  catch (std::exception)
    {
      throw dw_exception("GetNumericVector(): non-numeric entry");
    }
  return v;
}


std::ostream& operator<<(std::ostream &output, StringMatrix &string_matrix)
{
  for (unsigned int i=0; i < string_matrix.size(); i++) output << string_matrix[i];
  return output;
}

// Reads and parses the entire file starting from the current postion.  Default
// values are used for the delimiter and flags
std::istream& operator>>(std::istream &input, StringMatrix &string_matrix)
{
  string_matrix.clear();
  while (input.good()) string_matrix.AddLine(input);
  return input;
}
//===============================================================================
//===============================================================================
//===============================================================================

/*
   Returns 1 if the string id is found at the beginning of a line in the stream
   and 0 otherwise.  The file pointer is set to the line immediately after the 
   line containing id.  The search starts at the current position of the file.  
   If id is not found, then the file is rewound and the search is continued until 
   the end of the file reached again.  If an error is encountered while reading 
   the stream or the id is not found an exception is thrown.
*/
void SetFilePosition(std::istream &in, const std::string &id)
{
  std::string line;
  int len=id.length();
  while (1)
    {
      getline(in,line);
      if (in.good())
	{ if (line.compare(0,len,id) == 0) return; }
      else
	{
	  in.clear();
	  in.seekg(0);
	  while (1)
	    {
	      getline(in,line);
	      if (in.good())
		{ if (line.compare(0,len,id) == 0) return; }
	      else
		throw dw_exception("SetFilePostion(): " + id + " not found");
	    }  
	}
    }
}
