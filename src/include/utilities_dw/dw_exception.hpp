
#ifndef __DW_EXCEPTION_HPP__
#define __DW_EXCEPTION_HPP__

#include <exception>
#include <string>
#include <string.h>

class dw_exception : public std::exception
{
protected: 
  char *msg;

public:
  dw_exception(void) : exception() { msg=(char*)NULL; };
  dw_exception(const char *message) : exception() { if (!message) msg=(char*)NULL; else if ((msg=new (std::nothrow) char[strlen(message)+1])) strcpy(msg,message); };
  dw_exception(const std::string &message) : exception() { if ((msg=new (std::nothrow) char[message.length()+1])) strcpy(msg,message.c_str()); };
  dw_exception(const dw_exception &err) : exception() { if (!err.msg) msg=(char*)NULL; else if ((msg=new (std::nothrow) char[strlen(err.msg)+1])) strcpy(msg,err.msg); };
  ~dw_exception() throw() { delete[] msg; };
  const char* what() { return msg; };
};

#endif
