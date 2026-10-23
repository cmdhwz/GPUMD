#pragma once

#include "rpmd_ja_utf8_file_io.h"

#include <cstdio>

#ifdef _WIN32
#include <io.h>
#include <sys/locking.h>
#else
#include <sys/file.h>
#include <unistd.h>
#endif

namespace rpmd_ja_process_lock
{
inline bool try_lock(std::FILE* file)
{
  if(file==nullptr||std::fseek(file,0,SEEK_END)!=0)return false;
  if(std::ftell(file)==0){if(std::fputc('\0',file)==EOF||std::fflush(file)!=0)return false;}
  if(std::fseek(file,0,SEEK_SET)!=0)return false;
#ifdef _WIN32
  return _locking(_fileno(file),_LK_NBLCK,1)==0;
#else
  return flock(fileno(file),LOCK_EX|LOCK_NB)==0;
#endif
}

inline void unlock(std::FILE* file)
{
  if(file==nullptr)return;
  std::fseek(file,0,SEEK_SET);
#ifdef _WIN32
  _locking(_fileno(file),_LK_UNLCK,1);
#else
  flock(fileno(file),LOCK_UN);
#endif
}
}
