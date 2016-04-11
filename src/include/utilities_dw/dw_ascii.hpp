/*
 * Copyright (C) 1996-2013 Daniel Waggoner
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

#ifndef __DW_CPP_ASCII_ROUTINES__
#define __DW_CPP_ASCII_ROUTINES__

#include <string>
#include <vector>
#include <iostream>

#include "dw_dense_matrix.hpp"

// Flag codes
#define SV_REMOVE_EMPTY_FIELDS           0x00000001
#define SV_ALLOW_QUOTED_TEXT             0x00000002     // not implemented
#define SV_STRIP_LEADING_WHITESPACE      0x00000004
#define SV_STRIP_TRAILING_WHITESPACE     0x00000008
#define SV_STRIP_WHITESPACE              0x0000000c
#define SV_REMOVE_EMPTY_LINES            0x00000010
#define SV_DEFAULT                       SV_STRIP_WHITESPACE | SV_REMOVE_EMPTY_FIELDS | SV_REMOVE_EMPTY_LINES

class StringVector : public std::vector<std::string>
{
public:
  // constructors
  StringVector(void) { };
  StringVector(const StringVector &string_vector) : std::vector<std::string>(string_vector) { };
  StringVector(const std::string &line, char delimiter=' ', int flag=SV_DEFAULT) { ParseLine(line,delimiter,flag); };
  explicit StringVector(std::istream &input, char delimiter=' ', int flag=SV_DEFAULT) 
    { ParseLine(input,delimiter,flag); };

  // read/parse line
  void ParseLine(const std::string &line, char delimiter=' ', int flag=SV_DEFAULT);
  void ParseLine(std::istream &input, char delimiter=' ', int flag=SV_DEFAULT);
};
std::ostream& operator<<(std::ostream &output, StringVector &s_vector);
std::istream& operator>>(std::istream &input, StringVector &s_vector);

class StringMatrix : public std::vector<StringVector>
{
public:
  StringMatrix(void) { };
  StringMatrix(const StringMatrix &string_matrix) : std::vector<StringVector>(string_matrix) { };
  StringMatrix(const StringVector &string_vector) { push_back(string_vector); };
  explicit StringMatrix(std::istream &input, char delimiter=' ', int flag=SV_DEFAULT) { ParseLines(input,delimiter,flag); };
  StringMatrix(std::istream &input, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT) { ParseLines(input,comment,delimiter,flag); };
  StringMatrix(int n, std::istream &input, char delimiter=' ', int flag=SV_DEFAULT) { ParseLines(n,input,delimiter,flag); };
  StringMatrix(int n, std::istream &input, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT) { ParseLines(n,input,comment,delimiter,flag); };

  int ParseLines(std::istream &input, char delimiter=' ', int flag=SV_DEFAULT);
  int ParseLines(std::istream &input, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT);
  int ParseLines(int n, std::istream &input, char delimiter=' ', int flag=SV_DEFAULT);
  int ParseLines(int n, std::istream &input, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT);

  void AddLine(const StringVector &string_vector) { push_back(string_vector); };
  int AddLine(const std::string &line, char delimiter=' ', int flag=SV_DEFAULT);
  int AddLine(const std::string &line, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT);
  int AddLine(std::istream &input, char delimiter=' ', int flag=SV_DEFAULT);
  int AddLine(std::istream &input, const std::string &comment, char delimiter=' ', int flag=SV_DEFAULT);

  TDenseMatrix GetNumericMatrix(int pos, int n_rows, int n_cols=-1) const;
  TDenseVector GetNumericVector(int pos, int n_cols=-1) const;
};
std::ostream& operator<<(std::ostream &output, StringMatrix &string_matrix);
std::istream& operator>>(std::istream &input, StringMatrix &string_matrix);

void SetFilePosition(std::istream &in, const std::string &id);

double _stod(const std::string &str, size_t *idx=0); 
int _stoi(const std::string &str, size_t *idx=0); 
std::string cluster_to_string(int i);
#endif
