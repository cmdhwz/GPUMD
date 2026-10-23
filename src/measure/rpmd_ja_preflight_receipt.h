#pragma once

#include <iomanip>
#include <istream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rpmd_ja_preflight_receipt
{
inline std::map<std::string,std::string> read(std::istream& input,const char* expected_magic="RPJAPREFLIGHT5")
{
  std::string line;
  if(!std::getline(input,line))throw std::runtime_error("FEASIBILITY_PREFLIGHT_FAIL: receipt is empty");
  if(!line.empty()&&line.back()=='\r')line.pop_back();
  if(line!=expected_magic)throw std::runtime_error("FEASIBILITY_PREFLIGHT_FAIL: receipt version is unsupported");
  std::map<std::string,std::string> fields;
  while(std::getline(input,line)){
    if(!line.empty()&&line.back()=='\r')line.pop_back();
    if(line.empty())continue;
    std::istringstream row(line);std::string key,value;
    if(!(row>>key>>std::quoted(value)))throw std::runtime_error("FEASIBILITY_PREFLIGHT_FAIL: malformed receipt key-value row");
    row>>std::ws;if(!row.eof()||!fields.emplace(key,value).second)
      throw std::runtime_error("FEASIBILITY_PREFLIGHT_FAIL: duplicate or trailing receipt field");
  }
  if(!input.eof())throw std::runtime_error("FEASIBILITY_PREFLIGHT_FAIL: receipt read failed");
  return fields;
}
} // namespace rpmd_ja_preflight_receipt
