#include "../src/measure/rpmd_ja_a0_reuse.h"

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

static std::string read_bytes(const char* path)
{
  std::ifstream input(path,std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
}

int main(int argc,char** argv)
{
  if(argc!=6)return 2;
  const std::string manifest_bytes=read_bytes(argv[1]),state_bytes=read_bytes(argv[2]);
  std::istringstream input(manifest_bytes);std::string line;
  if(!std::getline(input,line)||line!="RPJAFEAS9")return 3;
  std::map<std::string,std::string> manifest;
  while(std::getline(input,line)){
    if(line.empty())continue;std::istringstream row(line);std::string key,value;
    if(!(row>>key>>std::quoted(value))||!manifest.emplace(key,value).second)return 4;
    row>>std::ws;if(!row.eof())return 5;
  }
  const auto record=rpmd_ja_a0_reuse::make_export_record(manifest_bytes,manifest,state_bytes,argv[4],argv[5],12.5,18.0,true,true,false);
  try{(void)rpmd_ja_a0_reuse::make_export_record(manifest_bytes,manifest,state_bytes,argv[4],argv[5],12.5,18.0,false,true,false);return 7;}
  catch(const std::invalid_argument&){}
  const std::string record_bytes=rpmd_ja_a0_reuse::serialize(record);
  std::string crlf=record_bytes;for(std::size_t pos=0;(pos=crlf.find('\n',pos))!=std::string::npos;pos+=2)crlf.replace(pos,1,"\r\n");
  std::istringstream crlf_input(crlf);if(rpmd_ja_a0_reuse::parse(crlf_input).source_job_id!=argv[5])return 8;
  int boundary=0;
  for(const auto point:{rpmd_ja_a0_reuse::WriteInterruption::AfterManifest,
                        rpmd_ja_a0_reuse::WriteInterruption::AfterState,
                        rpmd_ja_a0_reuse::WriteInterruption::AfterRecord}){
    const std::string interrupted_path=std::string(argv[3])+".interrupt."+std::to_string(boundary++);
    try{rpmd_ja_a0_reuse::write_package_atomic(interrupted_path,manifest_bytes,state_bytes,record,point);return 9;}
    catch(const std::runtime_error&){}
    if(rpmd_ja_file_io::ifstream(interrupted_path,std::ios::binary).good())return 10;
    rpmd_ja_a0_reuse::write_package_atomic(interrupted_path,manifest_bytes,state_bytes,record);
    const auto recovered=rpmd_ja_a0_reuse::parse_package(read_bytes(interrupted_path));
    if(recovered.manifest_bytes!=manifest_bytes||recovered.state_bytes!=state_bytes||recovered.record.source_job_id!=argv[5])return 13;
  }
  rpmd_ja_a0_reuse::write_package_atomic(argv[3],manifest_bytes,state_bytes,record);
  rpmd_ja_a0_reuse::write_package_atomic(argv[3],manifest_bytes,state_bytes,record);
  const auto package=rpmd_ja_a0_reuse::parse_package(read_bytes(argv[3]));
  if(package.manifest_bytes!=manifest_bytes||package.state_bytes!=state_bytes||package.record.source_job_id!=argv[5])return 11;
  auto foreign=record;foreign.source_job_id+="-changed";
  try{rpmd_ja_a0_reuse::write_package_atomic(argv[3],manifest_bytes,state_bytes,foreign);return 12;}
  catch(const std::runtime_error&){}
  return 0;
}
