#include "measure/rpmd_ja_preflight_receipt.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc,char** argv)
{
  try{
    std::ifstream file;
    std::istringstream fixture(R"(RPJAPREFLIGHT5
status "PASS"
path "D:\\runs\\run.in"
output "a \"quoted\" name"
)");
    std::istream* input=&fixture;
    const std::string expected_path="D:\\runs\\run.in";
    if(argc==2){
      file.open(argv[1],std::ios::binary);
      if(!file)throw std::runtime_error("native receipt parser cannot open the launcher receipt");
      input=&file;
    }else if(argc!=1)throw std::invalid_argument("usage: rpmd_ja_preflight_receipt_test [receipt]");
    const auto fields=rpmd_ja_preflight_receipt::read(*input);
    if(fields.at("status")!="PASS"||fields.at("path")!=(argc==1?expected_path:u8"D:\\测试数据\\run.in")||
       fields.at("output")!="a \"quoted\" name")
      throw std::runtime_error("native receipt parser changed launcher field values");
    return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
