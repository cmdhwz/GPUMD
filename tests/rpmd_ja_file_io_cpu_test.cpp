#include "measure/rpmd_ja_utf8_file_io.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
using Contents=std::map<std::string,std::string>;

#define REQUIRE(condition) do { if(!(condition)) throw std::runtime_error("require failed: " #condition); } while(false)

std::uint64_t query_cgroup_memory(const std::string& membership,const std::string& mountinfo,const Contents& contents)
{
  std::istringstream groups(membership),mounts(mountinfo);std::uint64_t remaining=0;
  const auto reader=[&](const std::string& path,std::string& value){const auto it=contents.find(path);
    if(it==contents.end())return rpmd_ja_file_io::CgroupFileReadStatus::missing;
    value=it->second;return rpmd_ja_file_io::CgroupFileReadStatus::ok;};
  if(!rpmd_ja_file_io::cgroup_memory_remaining_bytes(groups,mounts,reader,remaining))
    throw std::runtime_error("expected a mounted memory cgroup");
  return remaining;
}

void test_cgroup_v2_nested_limit_and_tighter_ancestor()
{
  const std::string membership="0::/user.slice/job.scope/step.scope\n";
  const std::string mount="29 23 0:26 / /sys/fs/cgroup rw,nosuid,nodev - cgroup2 cgroup rw\n";
  const Contents values={{"/sys/fs/cgroup/user.slice/job.scope/step.scope/memory.max","100"},
    {"/sys/fs/cgroup/user.slice/job.scope/step.scope/memory.current","20"},
    {"/sys/fs/cgroup/user.slice/job.scope/memory.max","90"},
    {"/sys/fs/cgroup/user.slice/job.scope/memory.current","20"},
    {"/sys/fs/cgroup/user.slice/memory.max","80"},{"/sys/fs/cgroup/user.slice/memory.current","40"},
    {"/sys/fs/cgroup/memory.max","max"},{"/sys/fs/cgroup/memory.current","50"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==40);
  Contents tighter=values;tighter["/sys/fs/cgroup/user.slice/job.scope/memory.max"]="40";
  tighter["/sys/fs/cgroup/user.slice/job.scope/memory.current"]="30";
  REQUIRE(query_cgroup_memory(membership,mount,tighter)==10);
}

void test_cgroup_v2_mount_root_and_escaped_mountpoint()
{
  const std::string membership="0::/tenant/user/job\n";
  const std::string mount="31 23 0:26 /tenant /sys/fs/cgroup/job\\040scope rw - cgroup2 cgroup rw\n";
  const Contents values={{"/sys/fs/cgroup/job scope/user/job/memory.max","500"},
    {"/sys/fs/cgroup/job scope/user/job/memory.current","100"},
    {"/sys/fs/cgroup/job scope/user/memory.max","400"},{"/sys/fs/cgroup/job scope/user/memory.current","250"},
    {"/sys/fs/cgroup/job scope/memory.max","max"},{"/sys/fs/cgroup/job scope/memory.current","300"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==150);
}

void test_cgroup_v2_namespace_root_keeps_delegated_mount_limit()
{
  const std::string membership="0::/user/job\n";
  const std::string mount="31 23 0:26 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n";
  const Contents values={{"/sys/fs/cgroup/user/job/memory.max","100"},
    {"/sys/fs/cgroup/user/job/memory.current","20"},
    {"/sys/fs/cgroup/user/memory.max","90"},{"/sys/fs/cgroup/user/memory.current","30"},
    {"/sys/fs/cgroup/memory.max","60"},{"/sys/fs/cgroup/memory.current","50"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==10);
}

void test_cgroup_v2_real_root_without_memory_files()
{
  const std::string membership="0::/job\n";
  const std::string mount="29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
  const Contents values={{"/cg/job/memory.max","100"},{"/cg/job/memory.current","20"},
    {"/cg/cgroup.subtree_control","memory"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==80);
}

void test_cgroup_v2_root_missing_limit_with_current_is_rejected()
{
  const std::string membership="0::/job\n";
  const std::string mount="29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
  const Contents values={{"/cg/job/memory.max","100"},{"/cg/job/memory.current","20"},
    {"/cg/memory.current","0"}};
  bool incomplete_root_rejected=false;
  try{(void)query_cgroup_memory(membership,mount,values);}
  catch(const std::runtime_error&){incomplete_root_rejected=true;}
  REQUIRE(incomplete_root_rejected);
}

void test_cgroup_v2_unavailable_memory_controller()
{
  const std::string membership="0::/job\n";
  const std::string mount="29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
  const Contents values={{"/cg/cgroup.subtree_control",""},{"/cg/cgroup.controllers","cpu io"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==std::numeric_limits<std::uint64_t>::max());
}

void test_cgroup_v2_missing_enabled_child_files_is_rejected()
{
  const std::string membership="0::/job/child\n";
  const std::string mount="29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
  const Contents values={{"/cg/job/memory.max","100"},{"/cg/job/memory.current","20"},
    {"/cg/job/cgroup.subtree_control","memory"},{"/cg/cgroup.subtree_control","memory"}};
  bool missing_child_rejected=false;
  try{(void)query_cgroup_memory(membership,mount,values);}
  catch(const std::runtime_error&){missing_child_rejected=true;}
  REQUIRE(missing_child_rejected);
}

void test_cgroup_v1_memory_mount_and_unlimited_limit()
{
  const std::string membership="5:cpu,memory:/batch/42\n";
  const std::string mount="32 23 0:27 / /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n";
  const Contents values={{"/sys/fs/cgroup/memory/batch/42/memory.limit_in_bytes","9223372036854771712"},
    {"/sys/fs/cgroup/memory/batch/42/memory.usage_in_bytes","500"},
    {"/sys/fs/cgroup/memory/batch/memory.limit_in_bytes","1000"},
    {"/sys/fs/cgroup/memory/batch/memory.usage_in_bytes","700"},
    {"/sys/fs/cgroup/memory/memory.limit_in_bytes","2000"},
    {"/sys/fs/cgroup/memory/memory.usage_in_bytes","1000"}};
  REQUIRE(query_cgroup_memory(membership,mount,values)==300);
}

void test_cgroup_memory_unlimited_missing_and_unreadable()
{
  const std::string membership="0::/job\n";
  const std::string mount="29 23 0:26 / /cg rw - cgroup2 cgroup rw\n";
  const Contents unlimited={{"/cg/job/memory.max","max"},{"/cg/job/memory.current","1"},
    {"/cg/memory.max","max"},{"/cg/memory.current","2"}};
  REQUIRE(query_cgroup_memory(membership,mount,unlimited)==std::numeric_limits<std::uint64_t>::max());
  bool unreadable=false;
  try{(void)query_cgroup_memory(membership,mount,{{"/cg/job/memory.max","100"}});}
  catch(const std::runtime_error&){unreadable=true;}
  REQUIRE(unreadable);
  std::istringstream no_controller("2:cpu:/job\n"),mounts(mount);std::uint64_t remaining=0;
  const auto missing_reader=[](const std::string&,std::string&){return rpmd_ja_file_io::CgroupFileReadStatus::missing;};
  REQUIRE(!rpmd_ja_file_io::cgroup_memory_remaining_bytes(no_controller,mounts,missing_reader,remaining));
}

void test_cgroup_v2_read_error_is_not_treated_as_missing_controller()
{
  std::istringstream groups("0::/job\n"),mounts("29 23 0:26 / /cg rw - cgroup2 cgroup rw\n");
  std::uint64_t remaining=0;
  const auto failing_reader=[](const std::string& path,std::string&){
    if(path=="/cg/job/memory.max")return rpmd_ja_file_io::CgroupFileReadStatus::error;
    return rpmd_ja_file_io::CgroupFileReadStatus::missing;
  };
  bool read_error_rejected=false;
  try{(void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(groups,mounts,failing_reader,remaining);}
  catch(const std::runtime_error&){read_error_rejected=true;}
  REQUIRE(read_error_rejected);
}

void test_cgroup_memory_values_are_nonnegative_decimal()
{
  bool negative_limit_rejected=false;
  try{std::istringstream maximum("-1"),current("0");
    (void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current);}
  catch(const std::runtime_error&){negative_limit_rejected=true;}
  REQUIRE(negative_limit_rejected);

  bool negative_usage_rejected=false;
  try{std::istringstream maximum("100"),current("-1");
    (void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current);}
  catch(const std::runtime_error&){negative_usage_rejected=true;}
  REQUIRE(negative_usage_rejected);

  bool negative_usage_rejected_when_unlimited=false;
  try{std::istringstream maximum("max"),current("-1");
    (void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current);}
  catch(const std::runtime_error&){negative_usage_rejected_when_unlimited=true;}
  REQUIRE(negative_usage_rejected_when_unlimited);

  bool overflow_rejected=false;
  try{std::istringstream maximum("18446744073709551616"),current("0");
    (void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current);}
  catch(const std::runtime_error&){overflow_rejected=true;}
  REQUIRE(overflow_rejected);

  bool usage_overflow_rejected=false;
  try{std::istringstream maximum("100"),current("18446744073709551616");
    (void)rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current);}
  catch(const std::runtime_error&){usage_overflow_rejected=true;}
  REQUIRE(usage_overflow_rejected);
}

void test_cgroup_file_reader_preserves_controller_text_and_empty_files()
{
  const std::string path="rpmd_ja_cgroup_reader_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".tmp";
  struct Cleanup{std::string path;~Cleanup(){std::remove(path.c_str());}} cleanup{path};
  {std::ofstream out(path,std::ios::trunc);REQUIRE(out.good());out<<"cpu io memory\n";REQUIRE(out.good());}
  std::string value;
  REQUIRE(rpmd_ja_file_io::read_cgroup_file(path,value)==rpmd_ja_file_io::CgroupFileReadStatus::ok);
  REQUIRE(value=="cpu io memory\n");
  {std::ofstream out(path,std::ios::trunc);REQUIRE(out.good());}
  value="stale";
  REQUIRE(rpmd_ja_file_io::read_cgroup_file(path,value)==rpmd_ja_file_io::CgroupFileReadStatus::ok);
  REQUIRE(value.empty());
#ifndef _WIN32
  REQUIRE(rpmd_ja_file_io::read_cgroup_file(".",value)==rpmd_ja_file_io::CgroupFileReadStatus::error);
#endif
}
}

void test_resize_file_preserves_prefix_and_rejects_invalid_size()
{
  const std::string path="rpmd_ja_resize_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+u8"_测试.tmp";
  struct Cleanup{std::string path;~Cleanup(){rpmd_ja_file_io::remove_file(path);}} cleanup{path};
  {rpmd_ja_file_io::ofstream out(path,std::ios::binary);out<<"abcdef";REQUIRE(out.good());}
  REQUIRE(rpmd_ja_file_io::resize_file(path,3)==0);
  {rpmd_ja_file_io::ifstream in(path,std::ios::binary);
    std::string value((std::istreambuf_iterator<char>(in)),std::istreambuf_iterator<char>());
    REQUIRE(value=="abc");}
  REQUIRE(rpmd_ja_file_io::resize_file(path,std::numeric_limits<std::uint64_t>::max())!=0);
  {rpmd_ja_file_io::ifstream in(path,std::ios::binary|std::ios::ate);REQUIRE(in.tellg()==std::streampos(3));}
  REQUIRE(rpmd_ja_file_io::resize_file(path,0)==0);
  {rpmd_ja_file_io::ifstream in(path,std::ios::binary|std::ios::ate);REQUIRE(in.tellg()==std::streampos(0));}
  REQUIRE(rpmd_ja_file_io::resize_file(path+".missing",1)!=0);
  {rpmd_ja_file_io::ifstream in(path+".missing");REQUIRE(!in.good());}
}

int main()
{
  test_resize_file_preserves_prefix_and_rejects_invalid_size();
  test_cgroup_v2_nested_limit_and_tighter_ancestor();
  test_cgroup_v2_mount_root_and_escaped_mountpoint();
  test_cgroup_v2_namespace_root_keeps_delegated_mount_limit();
  test_cgroup_v2_real_root_without_memory_files();
  test_cgroup_v2_root_missing_limit_with_current_is_rejected();
  test_cgroup_v2_unavailable_memory_controller();
  test_cgroup_v2_missing_enabled_child_files_is_rejected();
  test_cgroup_v1_memory_mount_and_unlimited_limit();
  test_cgroup_memory_unlimited_missing_and_unreadable();
  test_cgroup_v2_read_error_is_not_treated_as_missing_controller();
  test_cgroup_memory_values_are_nonnegative_decimal();
  test_cgroup_file_reader_preserves_controller_text_and_empty_files();
}
