#pragma once
#include "rpmd_ja_sha256.h"
#include <cstdint>
#include <streambuf>
#include <string>

class RpmdJAIdentityStreambuf : public std::streambuf
{
public:
  explicit RpmdJAIdentityStreambuf(std::streambuf* source) : source_(source) {}
  std::uint64_t fingerprint() const { return hash_; }
  std::string sha256() { return digest_.finish_hex(); }
protected:
  int_type underflow() override { return source_->sgetc(); }
  int_type uflow() override
  {
    const int_type value = source_->sbumpc();
    if (!traits_type::eq_int_type(value, traits_type::eof())) {
      const char byte = traits_type::to_char_type(value);
      consume(&byte, 1);
    }
    return value;
  }
  std::streamsize xsgetn(char* data, std::streamsize count) override
  {
    const std::streamsize read = source_->sgetn(data, count);
    consume(data, read);
    return read;
  }
  pos_type seekoff(off_type offset, std::ios_base::seekdir direction, std::ios_base::openmode mode) override
  {
    if (offset == 0 && direction == std::ios_base::cur && (mode & std::ios_base::in))
      return pos_type(position_);
    return pos_type(off_type(-1));
  }
private:
  void consume(const char* data, std::streamsize count)
  {
    digest_.update(reinterpret_cast<const unsigned char*>(data), static_cast<std::size_t>(count));
    for (std::streamsize i = 0; i < count; ++i)
      hash_ = (hash_ ^ static_cast<unsigned char>(data[i])) * 1099511628211ULL;
    position_ += count;
  }
  std::streambuf* source_;
  rpmd_ja_sha256::Digest digest_;
  std::uint64_t hash_ = 14695981039346656037ULL;
  off_type position_ = 0;
};
