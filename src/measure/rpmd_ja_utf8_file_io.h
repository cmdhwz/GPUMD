#pragma once

#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <sys/types.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace rpmd_ja_file_io
{
// RPMD-JA paths use UTF-8; Windows CRT and stream entry points receive UTF-16.
#ifdef _WIN32
inline std::wstring utf8_to_wide_path(const std::string& path)
{
  if(path.empty())return {};
  if(path.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("UTF-8 file path exceeds Windows conversion limits");
  const int input_size=static_cast<int>(path.size());
  const int output_size=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path.data(),input_size,nullptr,0);
  if(output_size<=0)throw std::runtime_error("file path is not valid UTF-8");
  std::wstring wide(static_cast<std::size_t>(output_size),L'\0');
  if(MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,path.data(),input_size,&wide[0],output_size)!=output_size)
    throw std::runtime_error("failed converting UTF-8 file path to UTF-16");
  return wide;
}
#endif

class ifstream : public std::ifstream
{
public:
  ifstream()=default;
  explicit ifstream(const std::string& path,const std::ios::openmode mode=std::ios::in){open(path,mode);}
  void open(const std::string& path,const std::ios::openmode mode=std::ios::in)
  {
#ifdef _WIN32
    const std::wstring wide=utf8_to_wide_path(path);std::ifstream::open(wide.c_str(),mode);
#else
    std::ifstream::open(path.c_str(),mode);
#endif
  }
};

class ofstream : public std::ofstream
{
public:
  ofstream()=default;
  explicit ofstream(const std::string& path,const std::ios::openmode mode=std::ios::out){open(path,mode);}
  void open(const std::string& path,const std::ios::openmode mode=std::ios::out)
  {
#ifdef _WIN32
    const std::wstring wide=utf8_to_wide_path(path);std::ofstream::open(wide.c_str(),mode);
#else
    std::ofstream::open(path.c_str(),mode);
#endif
  }
};

inline std::FILE* open_file(const std::string& path,const std::string& mode)
{
#ifdef _WIN32
  const std::wstring wide_path=utf8_to_wide_path(path),wide_mode=utf8_to_wide_path(mode);
  return ::_wfopen(wide_path.c_str(),wide_mode.c_str());
#else
  return std::fopen(path.c_str(),mode.c_str());
#endif
}

inline int remove_file(const std::string& path)
{
#ifdef _WIN32
  const std::wstring wide=utf8_to_wide_path(path);return ::_wremove(wide.c_str());
#else
  return std::remove(path.c_str());
#endif
}

// Keep large-file truncation compatible with the C++14 Makefile and UTF-8 paths.
inline int resize_file(const std::string& path,const std::uint64_t size)
{
#ifdef _WIN32
  using FileOffset = __int64;
#else
  using FileOffset = off_t;
#endif
  if(size>static_cast<std::uint64_t>(std::numeric_limits<FileOffset>::max())){
    errno=EFBIG;return -1;
  }
  std::FILE* file=open_file(path,"r+b");
  if(!file)return -1;
#ifdef _WIN32
  const int result=static_cast<int>(::_chsize_s(::_fileno(file),static_cast<FileOffset>(size)));
  if(result!=0)errno=result;
#else
  const int result=::ftruncate(::fileno(file),static_cast<FileOffset>(size));
#endif
  const int resize_errno=errno;
  const int close_result=std::fclose(file);
  if(result!=0){errno=resize_errno;return -1;}
  return close_result;
}

inline int rename_file(const std::string& from,const std::string& to)
{
#ifdef _WIN32
  const std::wstring wide_from=utf8_to_wide_path(from),wide_to=utf8_to_wide_path(to);
  return ::_wrename(wide_from.c_str(),wide_to.c_str());
#else
  return std::rename(from.c_str(),to.c_str());
#endif
}

inline int atomic_replace_file(const std::string& from,const std::string& to)
{
#ifdef _WIN32
  const std::wstring wide_from=utf8_to_wide_path(from),wide_to=utf8_to_wide_path(to);
  return ::MoveFileExW(wide_from.c_str(),wide_to.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)?0:-1;
#else
  return std::rename(from.c_str(),to.c_str());
#endif
}

inline int atomic_commit_new_file(const std::string& from,const std::string& to)
{
#ifdef _WIN32
  const std::wstring wide_from=utf8_to_wide_path(from),wide_to=utf8_to_wide_path(to);
  return ::MoveFileExW(wide_from.c_str(),wide_to.c_str(),MOVEFILE_WRITE_THROUGH)?0:-1;
#else
  if(::link(from.c_str(),to.c_str())!=0)return -1;
  return ::unlink(from.c_str());
#endif
}

inline std::uint64_t available_disk_bytes_for_path(const std::string& file_path)
{
  const std::size_t separator=file_path.find_last_of("/\\");
  const std::string directory=separator==std::string::npos?".":separator==0?file_path.substr(0,1):
    file_path.substr(0,separator==2&&file_path.size()>2&&file_path[1]==':'?3:separator);
#ifdef _WIN32
  ULARGE_INTEGER available{};const std::wstring wide=utf8_to_wide_path(directory);
  if(!GetDiskFreeSpaceExW(wide.c_str(),&available,nullptr,nullptr))
    throw std::runtime_error("cannot query free disk space for output directory");
  return static_cast<std::uint64_t>(available.QuadPart);
#else
  struct statvfs information{};
  if(::statvfs(directory.c_str(),&information)!=0)
    throw std::runtime_error("cannot query free disk space for output directory");
  const std::uint64_t blocks=static_cast<std::uint64_t>(information.f_bavail);
  const std::uint64_t block_size=static_cast<std::uint64_t>(information.f_frsize?information.f_frsize:information.f_bsize);
  if(block_size&&blocks>std::numeric_limits<std::uint64_t>::max()/block_size)
    throw std::runtime_error("free disk-space query overflowed");
  return blocks*block_size;
#endif
}

inline std::uint64_t cgroup_memory_remaining_bytes(std::istream& maximum,std::istream& current,
  const bool cgroup_v1=false)
{
  std::string limit_text,current_text;
  if(!(maximum>>limit_text)||!(current>>current_text))
    throw std::runtime_error("cannot read cgroup memory limit or current usage");
  auto parse=[](const std::string& value){
    if(value.empty())throw std::runtime_error("invalid cgroup memory accounting value");
    std::uint64_t parsed=0;
    for(const char digit:value){
      if(digit<'0'||digit>'9')throw std::runtime_error("invalid cgroup memory accounting value");
      const std::uint64_t next=static_cast<std::uint64_t>(digit-'0');
      if(parsed>(std::numeric_limits<std::uint64_t>::max()-next)/10)
        throw std::runtime_error("invalid cgroup memory accounting value");
      parsed=parsed*10+next;
    }
    return parsed;
  };
  const std::uint64_t usage=parse(current_text);
  if(limit_text=="max")return std::numeric_limits<std::uint64_t>::max();
  const std::uint64_t limit=parse(limit_text);
  if(cgroup_v1&&limit>=(std::uint64_t(1)<<60))return std::numeric_limits<std::uint64_t>::max();
  return usage>=limit?0:limit-usage;
}

struct CgroupMemoryMount{std::string root,mountpoint;bool cgroup_v1=false;};
enum class CgroupFileReadStatus{ok,missing,error};

inline CgroupFileReadStatus read_cgroup_file(const std::string& path,std::string& value)
{
  errno=0;std::ifstream input(path);
  if(!input)return errno==ENOENT?CgroupFileReadStatus::missing:CgroupFileReadStatus::error;
  std::string contents;char buffer[4096];
  for(;;){
    input.read(buffer,static_cast<std::streamsize>(sizeof(buffer)));const std::streamsize count=input.gcount();
    if(count>0)contents.append(buffer,static_cast<std::size_t>(count));
    if(input.bad())return CgroupFileReadStatus::error;
    if(input.eof())break;
    if(input.fail())return CgroupFileReadStatus::error;
  }
  value=std::move(contents);return CgroupFileReadStatus::ok;
}

inline std::string decode_mountinfo_path(std::string path)
{
  const std::pair<const char*,const char*> escapes[]={{"\\040"," "},{"\\011","\t"},{"\\012","\n"},{"\\134","\\"}};
  for(const auto& escape:escapes){std::size_t position=0;while((position=path.find(escape.first,position))!=std::string::npos){
      path.replace(position,4,escape.second);position+=std::char_traits<char>::length(escape.second);}}
  return path;
}

inline bool cgroup_path_relative_to_mount_root(const std::string& path,const std::string& root,std::string& relative)
{
  if(root=="/"){relative=path.size()>1?path.substr(1):std::string();return !path.empty()&&path[0]=='/';}
  if(path==root){relative.clear();return true;}
  if(path.size()>root.size()&&path.compare(0,root.size(),root)==0&&path[root.size()]=='/'){
    relative=path.substr(root.size()+1);return true;}
  return false;
}

inline std::vector<std::string> cgroup_memory_ancestor_paths(const std::string& mountpoint,const std::string& relative)
{
  std::string root=mountpoint;
  while(root.size()>1&&root.back()=='/')root.pop_back();
  std::string leaf=root;
  if(!relative.empty()){if(leaf!="/")leaf+='/';leaf+=relative;}
  std::vector<std::string> paths;
  for(std::string current=leaf;;){
    paths.push_back(current);if(current==root)break;
    const std::size_t separator=current.find_last_of('/');
    if(separator==std::string::npos)throw std::runtime_error("invalid cgroup memory mount path");
    current=separator==0?"/":current.substr(0,separator);
    if(current.size()<root.size()||(current.compare(0,root.size(),root)!=0)||
       (current.size()>root.size()&&current[root.size()]!='/'))
      throw std::runtime_error("cgroup memory path escaped its mount root");
  }
  return paths;
}

inline bool cgroup_comma_list_contains(const std::string& list,const std::string& item)
{
  std::istringstream values(list);std::string value;
  while(std::getline(values,value,','))if(value==item)return true;
  return false;
}

inline bool cgroup_space_list_contains(const std::string& list,const std::string& item)
{
  std::istringstream values(list);std::string value;
  while(values>>value)if(value==item)return true;
  return false;
}

inline std::string cgroup_file_path(const std::string& directory,const std::string& filename)
{
  return directory+(directory=="/"?"":"/")+filename;
}

template<typename ReadFile>
inline bool cgroup_memory_remaining_bytes(std::istream& self_cgroup,std::istream& mountinfo,
  ReadFile&& read_file,std::uint64_t& remaining)
{
  struct Membership{std::string path;bool cgroup_v1=false;};
  std::vector<Membership> memberships;std::string line;
  while(std::getline(self_cgroup,line)){
    const std::size_t first=line.find(':'),second=first==std::string::npos?first:line.find(':',first+1);
    if(first==std::string::npos||second==std::string::npos)continue;
    const std::string hierarchy=line.substr(0,first),controllers=line.substr(first+1,second-first-1);
    if(controllers.empty()&&hierarchy=="0")memberships.push_back({line.substr(second+1),false});
    else if(cgroup_comma_list_contains(controllers,"memory"))memberships.push_back({line.substr(second+1),true});
  }
  if(memberships.empty()){remaining=std::numeric_limits<std::uint64_t>::max();return false;}

  std::vector<CgroupMemoryMount> mounts;std::vector<std::string> fields;
  while(std::getline(mountinfo,line)){
    std::istringstream row(line);fields.clear();std::string field;
    while(row>>field)fields.push_back(field);
    const auto separator=std::find(fields.begin(),fields.end(),"-");
    if(separator==fields.end())continue;
    const std::size_t dash=static_cast<std::size_t>(separator-fields.begin());
    if(dash<6||fields.size()<=dash+3)continue;
    const std::string filesystem=fields[dash+1];
    if(filesystem=="cgroup2")mounts.push_back({decode_mountinfo_path(fields[3]),decode_mountinfo_path(fields[4]),false});
    else if(filesystem=="cgroup"&&(cgroup_comma_list_contains(fields[5],"memory")||
             cgroup_comma_list_contains(fields[dash+3],"memory")||
             cgroup_comma_list_contains(fields[dash+2],"memory")))
      mounts.push_back({decode_mountinfo_path(fields[3]),decode_mountinfo_path(fields[4]),true});
  }

  bool matched=false;remaining=std::numeric_limits<std::uint64_t>::max();
  for(const Membership& membership:memberships){
    bool membership_mounted=false;
    for(const CgroupMemoryMount& mount:mounts){
      if(mount.cgroup_v1!=membership.cgroup_v1)continue;
      std::string relative;
      if(!cgroup_path_relative_to_mount_root(membership.path,mount.root,relative)){
        if(mount.cgroup_v1||membership.path.empty()||membership.path[0]!='/')continue;
        relative=membership.path.size()>1?membership.path.substr(1):std::string();
      }
      membership_mounted=true;matched=true;
      const std::vector<std::string> ancestors=cgroup_memory_ancestor_paths(mount.mountpoint,relative);
      for(std::size_t i=0;i<ancestors.size();++i){
        const std::string& ancestor=ancestors[i];
        const std::string limit_path=cgroup_file_path(ancestor,mount.cgroup_v1?"memory.limit_in_bytes":"memory.max");
        const std::string usage_path=cgroup_file_path(ancestor,mount.cgroup_v1?"memory.usage_in_bytes":"memory.current");
        std::string limit_text,usage_text;
        const CgroupFileReadStatus limit_status=read_file(limit_path,limit_text);
        const CgroupFileReadStatus usage_status=read_file(usage_path,usage_text);
        if(limit_status==CgroupFileReadStatus::error||usage_status==CgroupFileReadStatus::error)
          throw std::runtime_error("cannot read memory limit for the current cgroup or an ancestor");
        if(!mount.cgroup_v1&&limit_status==CgroupFileReadStatus::missing){
          if(mount.root=="/"&&i+1==ancestors.size()&&usage_status==CgroupFileReadStatus::missing)continue;
          if(usage_status==CgroupFileReadStatus::missing){
            if(i+1==ancestors.size())
              throw std::runtime_error("cannot determine whether the mounted cgroup root has a memory limit");
            const std::size_t separator=ancestor.find_last_of('/');
            const std::string parent=separator==0?"/":ancestor.substr(0,separator);
            std::string controllers;
            if(read_file(cgroup_file_path(parent,"cgroup.subtree_control"),controllers)!=CgroupFileReadStatus::ok)
              throw std::runtime_error("cannot read cgroup memory controller state");
            if(!cgroup_space_list_contains(controllers,"memory"))continue;
          }
          throw std::runtime_error("cannot read memory limit for the current cgroup or an ancestor");
        }
        if(limit_status!=CgroupFileReadStatus::ok||usage_status!=CgroupFileReadStatus::ok)
          throw std::runtime_error("cannot read memory limit for the current cgroup or an ancestor");
        std::istringstream limit_stream(limit_text),usage_stream(usage_text);
        remaining=std::min(remaining,cgroup_memory_remaining_bytes(limit_stream,usage_stream,mount.cgroup_v1));
      }
    }
    if(!membership_mounted)throw std::runtime_error("cannot locate the current memory cgroup mount");
  }
  return matched;
}

inline std::uint64_t available_physical_memory_bytes()
{
#ifdef _WIN32
  MEMORYSTATUSEX status{};status.dwLength=sizeof(status);
  if(!GlobalMemoryStatusEx(&status))throw std::runtime_error("cannot query available physical memory");
  return static_cast<std::uint64_t>(status.ullAvailPhys);
#else
  const long pages=::sysconf(_SC_AVPHYS_PAGES),page_size=::sysconf(_SC_PAGESIZE);
  if(pages<=0||page_size<=0||static_cast<std::uint64_t>(pages)>
     std::numeric_limits<std::uint64_t>::max()/static_cast<std::uint64_t>(page_size))
    throw std::runtime_error("cannot query available physical memory");
  std::uint64_t available=static_cast<std::uint64_t>(pages)*static_cast<std::uint64_t>(page_size);
  std::ifstream self_cgroup("/proc/self/cgroup"),mountinfo("/proc/self/mountinfo");
  if(!self_cgroup||!mountinfo)throw std::runtime_error("cannot locate the current process memory cgroup");
  std::uint64_t cgroup_remaining=std::numeric_limits<std::uint64_t>::max();
  const auto read_file=[](const std::string& path,std::string& value){return read_cgroup_file(path,value);};
  if(cgroup_memory_remaining_bytes(self_cgroup,mountinfo,read_file,cgroup_remaining))
    available=std::min(available,cgroup_remaining);
  return available;
#endif
}
} // namespace rpmd_ja_file_io
