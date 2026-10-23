#pragma once

#include "rpmd_ja_sha256.h"
#include "rpmd_ja_utf8_file_io.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rpmd_ja_a0_reuse
{
constexpr const char* magic="RPJAA0REUSE2";
constexpr const char* package_magic="RPJAA0PACKAGE1";
constexpr const char* qualification="CONVERGED_A0_WITH_COMPLETE_RESPONSE_V2";

struct Record
{
  std::string source_manifest_sha256,contract_sha256,state_sha256;
  std::string source_machine_fingerprint,source_job_id,historical_origin_job_id;
  std::string qualification_status=qualification;
  double source_optimizer_seconds=0.0,source_fit_elapsed_seconds=0.0;
  double current_import_validation_seconds=0.0,current_response_seconds=0.0;
  bool development_response_complete=false,development_response_pass=false;
};

struct Package
{
  Record record;
  std::string manifest_bytes,state_bytes,record_bytes;
};

enum class WriteInterruption { None,AfterManifest,AfterState,AfterRecord };

inline void validate(const Record& record);

inline bool is_sha256(const std::string& value)
{if(value.size()!=64)return false;for(const char ch:value)if(!((ch>='0'&&ch<='9')||(ch>='a'&&ch<='f')))return false;return true;}

inline std::string sha256(const std::string& bytes)
{
  rpmd_ja_sha256::Digest digest;digest.update(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size());
  return digest.finish_hex();
}

inline std::string contract_sha256(const std::map<std::string,std::string>& fields)
{
  std::ostringstream canonical;
  for(const auto& field:fields){
    if(field.first=="job_id"||field.first.rfind("a0_reuse_",0)==0||
       (field.first.size()>=5&&field.first.compare(field.first.size()-5,5,"_path")==0))continue;
    canonical<<field.first<<'\n'<<field.second<<'\n';
  }
  return sha256(canonical.str());
}

inline Record make_export_record(const std::string& source_manifest_bytes,
  const std::map<std::string,std::string>& source_manifest,const std::string& state_bytes,
  const std::string& source_machine_fingerprint,const std::string& source_job_id,
  const double source_optimizer_seconds,const double source_fit_elapsed_seconds,
  const bool fit_converged,const bool development_response_complete,const bool development_response_pass,
  const double current_import_validation_seconds=0.0,const double current_response_seconds=0.0,
  const std::string& historical_origin_job_id={})
{
  if(!fit_converged)throw std::invalid_argument("A0 reuse export requires a converged fit");
  Record record;record.source_manifest_sha256=sha256(source_manifest_bytes);
  record.contract_sha256=contract_sha256(source_manifest);record.state_sha256=sha256(state_bytes);
  record.source_machine_fingerprint=source_machine_fingerprint;record.source_job_id=source_job_id;
  record.historical_origin_job_id=historical_origin_job_id.empty()?source_job_id:historical_origin_job_id;
  record.source_optimizer_seconds=source_optimizer_seconds;record.source_fit_elapsed_seconds=source_fit_elapsed_seconds;
  record.current_import_validation_seconds=current_import_validation_seconds;record.current_response_seconds=current_response_seconds;
  record.development_response_complete=development_response_complete;
  record.development_response_pass=development_response_pass;validate(record);return record;
}

inline void verify_import(const Record& record,const std::string& record_bytes,const std::string& expected_record_sha256,
  const std::string& source_manifest_bytes,const std::map<std::string,std::string>& source_manifest,
  const std::string& state_bytes,const std::map<std::string,std::string>& current_manifest,
  const std::string& current_manifest_sha256,const std::string& current_machine_fingerprint,
  const std::string& current_job_id)
{
  validate(record);
  if(!is_sha256(expected_record_sha256)||sha256(record_bytes)!=expected_record_sha256||
     sha256(source_manifest_bytes)!=record.source_manifest_sha256||
     contract_sha256(source_manifest)!=record.contract_sha256||
     contract_sha256(current_manifest)!=record.contract_sha256||
     sha256(state_bytes)!=record.state_sha256||current_manifest_sha256==record.source_manifest_sha256||
     record.source_machine_fingerprint!=current_machine_fingerprint||record.source_job_id==current_job_id)
    throw std::runtime_error("A0 reuse package identity or current contract does not match");
}

inline void validate(const Record& record)
{
  if(!is_sha256(record.source_manifest_sha256)||!is_sha256(record.contract_sha256)||!is_sha256(record.state_sha256)||
     !is_sha256(record.source_machine_fingerprint)||record.source_job_id.empty()||record.historical_origin_job_id.empty()||
     record.qualification_status!=qualification||!record.development_response_complete||
     !std::isfinite(record.source_optimizer_seconds)||record.source_optimizer_seconds<0.0||
     !std::isfinite(record.source_fit_elapsed_seconds)||record.source_fit_elapsed_seconds<0.0||
     !std::isfinite(record.current_import_validation_seconds)||record.current_import_validation_seconds<0.0||
     !std::isfinite(record.current_response_seconds)||record.current_response_seconds<0.0)
    throw std::invalid_argument("A0 reuse record is incomplete or invalid");
}

inline std::string serialize(const Record& record)
{
  validate(record);std::ostringstream out;out<<std::setprecision(17)<<magic<<'\n'
    <<"source_manifest_sha256 "<<std::quoted(record.source_manifest_sha256)<<'\n'
    <<"contract_sha256 "<<std::quoted(record.contract_sha256)<<'\n'
    <<"state_sha256 "<<std::quoted(record.state_sha256)<<'\n'
    <<"source_machine_fingerprint "<<std::quoted(record.source_machine_fingerprint)<<'\n'
    <<"source_job_id "<<std::quoted(record.source_job_id)<<'\n'
    <<"historical_origin_job_id "<<std::quoted(record.historical_origin_job_id)<<'\n'
    <<"qualification_status "<<std::quoted(record.qualification_status)<<'\n'
    <<"source_optimizer_seconds "<<record.source_optimizer_seconds<<'\n'
    <<"source_fit_elapsed_seconds "<<record.source_fit_elapsed_seconds<<'\n'
    <<"current_import_validation_seconds "<<record.current_import_validation_seconds<<'\n'
    <<"current_response_seconds "<<record.current_response_seconds<<'\n'
    <<"development_response_complete 1\n"
    <<"development_response_pass "<<(record.development_response_pass?1:0)<<'\n';
  return out.str();
}

inline Record parse(std::istream& in)
{
  std::string line;if(!std::getline(in,line))throw std::runtime_error("unsupported A0 reuse record version");
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line!=magic)throw std::runtime_error("unsupported A0 reuse record version");
  std::map<std::string,std::string> fields;
  while(std::getline(in,line)){
    if(!line.empty()&&line.back()=='\r')line.pop_back();if(line.empty())continue;
    std::istringstream row(line);std::string key,value;
    if(!(row>>key))throw std::runtime_error("malformed A0 reuse record field");
    if(key=="source_optimizer_seconds"||key=="source_fit_elapsed_seconds"||
       key=="current_import_validation_seconds"||key=="current_response_seconds"||
       key=="development_response_complete"||key=="development_response_pass"){
      if(!(row>>value))throw std::runtime_error("malformed A0 reuse record scalar");
    }else if(!(row>>std::quoted(value)))throw std::runtime_error("malformed A0 reuse record string");
    row>>std::ws;if(!row.eof()||!fields.emplace(key,value).second)
      throw std::runtime_error("duplicate or trailing A0 reuse record field");
  }
  if(!in.eof())throw std::runtime_error("A0 reuse record read failed");
  auto required=[&](const char* key)->const std::string&{const auto found=fields.find(key);
    if(found==fields.end())throw std::runtime_error(std::string("missing A0 reuse record field ")+key);return found->second;};
  auto scalar=[&](const char* key){const std::string& text=required(key);std::istringstream row(text);double value=0.0;std::string extra;
    if(!(row>>value)||(row>>extra)||!std::isfinite(value)||value<0.0)throw std::runtime_error("invalid A0 reuse record duration");return value;};
  auto flag=[&](const char* key){const std::string& text=required(key);if(text!="0"&&text!="1")throw std::runtime_error("invalid A0 reuse record flag");return text=="1";};
  Record record;record.source_manifest_sha256=required("source_manifest_sha256");record.contract_sha256=required("contract_sha256");
  record.state_sha256=required("state_sha256");record.source_machine_fingerprint=required("source_machine_fingerprint");
  record.source_job_id=required("source_job_id");record.historical_origin_job_id=required("historical_origin_job_id");
  record.qualification_status=required("qualification_status");
  record.source_optimizer_seconds=scalar("source_optimizer_seconds");record.source_fit_elapsed_seconds=scalar("source_fit_elapsed_seconds");
  record.current_import_validation_seconds=scalar("current_import_validation_seconds");
  record.current_response_seconds=scalar("current_response_seconds");
  record.development_response_complete=flag("development_response_complete");record.development_response_pass=flag("development_response_pass");
  if(fields.size()!=13)throw std::runtime_error("A0 reuse record has unknown fields");
  validate(record);return record;
}

inline std::string serialize_package(const std::string& manifest_bytes,const std::string& state_bytes,const Record& record)
{
  validate(record);
  if(sha256(manifest_bytes)!=record.source_manifest_sha256||sha256(state_bytes)!=record.state_sha256)
    throw std::invalid_argument("A0 reuse package snapshots do not match their record digests");
  const std::string record_bytes=serialize(record);std::ostringstream header;
  header<<package_magic<<'\n'<<"manifest_bytes "<<manifest_bytes.size()<<'\n'
    <<"state_bytes "<<state_bytes.size()<<'\n'<<"record_bytes "<<record_bytes.size()<<'\n';
  return header.str()+manifest_bytes+state_bytes+record_bytes;
}

inline Package parse_package(const std::string& bytes)
{
  std::size_t cursor=0;
  auto line=[&](){const std::size_t end=bytes.find('\n',cursor);if(end==std::string::npos)
      throw std::runtime_error("truncated A0 reuse package header");
    std::string value=bytes.substr(cursor,end-cursor);cursor=end+1;
    if(!value.empty()&&value.back()=='\r')value.pop_back();return value;};
  if(line()!=package_magic)throw std::runtime_error("unsupported A0 reuse package version");
  auto size=[&](const char* key){const std::string row=line(),prefix=std::string(key)+" ";
    if(row.compare(0,prefix.size(),prefix)!=0)throw std::runtime_error("malformed A0 reuse package length");
    std::size_t consumed=0;unsigned long long parsed=0;
    try{parsed=std::stoull(row.substr(prefix.size()),&consumed);}catch(...){throw std::runtime_error("malformed A0 reuse package length");}
    if(consumed!=row.size()-prefix.size()||parsed>bytes.size()-cursor)throw std::runtime_error("A0 reuse package length is out of range");
    return static_cast<std::size_t>(parsed);};
  const std::size_t manifest_size=size("manifest_bytes"),state_size=size("state_bytes"),record_size=size("record_bytes");
  if(manifest_size>bytes.size()-cursor||state_size>bytes.size()-cursor-manifest_size||
     record_size>bytes.size()-cursor-manifest_size-state_size||
     cursor+manifest_size+state_size+record_size!=bytes.size())
    throw std::runtime_error("A0 reuse package payload lengths do not match the file");
  Package package;package.manifest_bytes=bytes.substr(cursor,manifest_size);cursor+=manifest_size;
  package.state_bytes=bytes.substr(cursor,state_size);cursor+=state_size;
  package.record_bytes=bytes.substr(cursor,record_size);
  std::istringstream record_input(package.record_bytes);package.record=parse(record_input);
  if(sha256(package.manifest_bytes)!=package.record.source_manifest_sha256||
     sha256(package.state_bytes)!=package.record.state_sha256)
    throw std::runtime_error("A0 reuse package snapshot digest mismatch");
  return package;
}

inline void write_package_atomic(const std::string& path,const std::string& manifest_bytes,const std::string& state_bytes,
  const Record& record,const WriteInterruption interruption=WriteInterruption::None)
{
  const std::string record_bytes=serialize(record),contents=serialize_package(manifest_bytes,state_bytes,record);
  auto same_source=[&](const std::string& bytes){
    const Package existing=parse_package(bytes);
    return existing.manifest_bytes==manifest_bytes&&existing.state_bytes==state_bytes&&
      existing.record.source_job_id==record.source_job_id&&
      existing.record.source_machine_fingerprint==record.source_machine_fingerprint;
  };
  {
    rpmd_ja_file_io::ifstream existing(path,std::ios::binary);
    if(existing){const std::string bytes(std::istreambuf_iterator<char>(existing),std::istreambuf_iterator<char>());
      if(!same_source(bytes))throw std::runtime_error("A0 reuse package already exists with different evidence");return;}
  }
  static std::atomic<unsigned long long> sequence{0};
  const auto nonce=static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count());
  std::string temporary;std::FILE* output=nullptr;
  for(unsigned attempt=0;attempt<128&&!output;++attempt){
    temporary=path+".tmp."+std::to_string(nonce)+"."+std::to_string(sequence.fetch_add(1)+attempt);
    output=rpmd_ja_file_io::open_file(temporary,"wbx");}
  if(!output)throw std::runtime_error("cannot create unique A0 reuse package staging file");
  auto write=[&](const char* data,const std::size_t count){if(count&&std::fwrite(data,1,count,output)!=count){std::fclose(output);
      throw std::runtime_error("cannot write A0 reuse package staging file");}};
  const std::size_t header_size=contents.size()-manifest_bytes.size()-state_bytes.size()-record_bytes.size();
  const std::string header=contents.substr(0,header_size);
  write(header.data(),header.size());write(manifest_bytes.data(),manifest_bytes.size());
  if(interruption==WriteInterruption::AfterManifest){std::fclose(output);throw std::runtime_error("injected A0 reuse interruption after manifest snapshot");}
  write(state_bytes.data(),state_bytes.size());
  if(interruption==WriteInterruption::AfterState){std::fclose(output);throw std::runtime_error("injected A0 reuse interruption after state snapshot");}
  write(record_bytes.data(),record_bytes.size());
  if(interruption==WriteInterruption::AfterRecord){std::fclose(output);throw std::runtime_error("injected A0 reuse interruption after record");}
  const int flush_status=std::fflush(output),close_status=std::fclose(output);
  if(flush_status!=0||close_status!=0)throw std::runtime_error("cannot flush A0 reuse package staging file");
  if(rpmd_ja_file_io::atomic_commit_new_file(temporary,path)!=0){
    rpmd_ja_file_io::ifstream raced(path,std::ios::binary);
    if(raced){const std::string bytes(std::istreambuf_iterator<char>(raced),std::istreambuf_iterator<char>());
      if(bytes==contents||same_source(bytes))return;}
    throw std::runtime_error("cannot atomically commit A0 reuse package without replacing existing evidence");}
}
} // namespace rpmd_ja_a0_reuse
