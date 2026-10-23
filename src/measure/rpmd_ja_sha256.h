#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rpmd_ja_sha256
{
class Digest
{
 public:
  void update(const unsigned char* data, const std::size_t length)
  {
    if(length>(std::numeric_limits<std::uint64_t>::max()-bit_count_)/8)
      throw std::runtime_error("SHA-256 input length exceeds its 64-bit bit count");
    bit_count_+=static_cast<std::uint64_t>(length)*8;
    std::size_t offset=0;
    while(offset<length){
      const std::size_t count=std::min(length-offset,sizeof(block_)-block_used_);
      for(std::size_t i=0;i<count;++i)block_[block_used_+i]=data[offset+i];
      block_used_+=count;offset+=count;
      if(block_used_==sizeof(block_)){transform(block_);block_used_=0;}
    }
  }

  std::string finish_hex()
  {
    const std::uint64_t original_bit_count=bit_count_;
    const unsigned char padding[64]={0x80};
    const std::size_t padding_size=block_used_<56?56-block_used_:120-block_used_;
    update(padding,padding_size);
    unsigned char length[8];
    for(int i=0;i<8;++i)length[i]=static_cast<unsigned char>(original_bit_count>>(56-8*i));
    update(length,sizeof(length));
    std::ostringstream result;result<<std::hex<<std::setfill('0');
    for(const std::uint32_t word:state_)result<<std::setw(8)<<word;
    return result.str();
  }

 private:
  static std::uint32_t rotate_right(const std::uint32_t value,const unsigned int count)
  {return (value>>count)|(value<<(32-count));}

  static const std::uint32_t* constants()
  {
    static const std::uint32_t values[64]={
      0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
      0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
      0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
      0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
      0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
      0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
      0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
      0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    return values;
  }

  void transform(const unsigned char* block)
  {
    std::uint32_t words[64];
    for(int i=0;i<16;++i)words[i]=(static_cast<std::uint32_t>(block[4*i])<<24)|
      (static_cast<std::uint32_t>(block[4*i+1])<<16)|(static_cast<std::uint32_t>(block[4*i+2])<<8)|block[4*i+3];
    for(int i=16;i<64;++i){
      const std::uint32_t s0=rotate_right(words[i-15],7)^rotate_right(words[i-15],18)^(words[i-15]>>3);
      const std::uint32_t s1=rotate_right(words[i-2],17)^rotate_right(words[i-2],19)^(words[i-2]>>10);
      words[i]=words[i-16]+s0+words[i-7]+s1;
    }
    std::uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    const std::uint32_t* k=constants();
    for(int i=0;i<64;++i){
      const std::uint32_t s1=rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
      const std::uint32_t choice=(e&f)^(~e&g),t1=h+s1+choice+k[i]+words[i];
      const std::uint32_t s0=rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
      const std::uint32_t majority=(a&b)^(a&c)^(b&c),t2=s0+majority;
      h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
    state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }

  std::uint32_t state_[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  unsigned char block_[64]={};
  std::size_t block_used_=0;
  std::uint64_t bit_count_=0;
};
}
