#include "measure/rpmd_ja_identity_stream.cuh"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <chrono>
#include <sstream>
#include <string>
#include <vector>

int main()
{
  for (std::size_t size : {std::size_t(0), std::size_t(1), std::size_t(8191), std::size_t(8192), std::size_t(8193), std::size_t(16385)}) {
    std::string bytes(size, '\0');
    for (std::size_t i = 0; i < size; ++i) bytes[i] = static_cast<char>((i * 73) & 255);
    std::stringbuf source(bytes, std::ios::in);
    RpmdJAIdentityStreambuf identity(&source);
    std::istream in(&identity);
    assert(in.tellg() == 0);
    if (size) { assert(in.peek() == static_cast<unsigned char>(bytes[0])); assert(in.tellg() == 0); }
    in.seekg(1, std::ios::beg);
    assert(in.fail()); in.clear(); assert(in.tellg() == 0);
    std::vector<char> read(size + 1);
    std::size_t consumed = 0;
    while (consumed < size) {
      const auto first = in.peek();
      assert(first == in.peek());
      assert(in.tellg() == static_cast<std::streamoff>(consumed));
      assert(in.get() == first); read[consumed++] = static_cast<char>(first);
      const std::size_t chunk = std::min<std::size_t>(137, size - consumed);
      if (chunk) { in.read(read.data() + consumed, static_cast<std::streamsize>(chunk)); consumed += chunk; }
    }
    assert(std::string(read.data(), size) == bytes);
    assert(in.tellg() == static_cast<std::streamoff>(size));
    assert(in.peek() == std::char_traits<char>::eof());
    rpmd_ja_sha256::Digest expected_sha;
    expected_sha.update(reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
    const std::string observed_sha = identity.sha256();
    assert(observed_sha == expected_sha.finish_hex());
    if (size == 0) assert(observed_sha == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    std::uint64_t expected = 14695981039346656037ULL;
    for (unsigned char byte : bytes) expected = (expected ^ byte) * 1099511628211ULL;
    assert(identity.fingerprint() == expected);
    std::stringbuf short_source(bytes, std::ios::in);
    RpmdJAIdentityStreambuf short_identity(&short_source);
    std::istream truncated(&short_identity);
    truncated.read(read.data(), static_cast<std::streamsize>(size + 1));
    assert(truncated.fail() && truncated.gcount() == static_cast<std::streamsize>(size));
  }
  std::stringbuf abc_source("abc", std::ios::in);
  RpmdJAIdentityStreambuf abc_identity(&abc_source);
  std::istream abc(&abc_identity);
  assert(abc.get() == 'a'); char tail[2]; abc.read(tail, 2);
  assert(abc_identity.sha256() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

  const std::string path = "rpmd_ja_identity_test_" +
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tmp";
  { std::ofstream file(path, std::ios::binary); file << "real file"; assert(file.good()); }
  {
    std::ifstream file(path, std::ios::binary); assert(file.good());
    RpmdJAIdentityStreambuf identity(file.rdbuf()); std::istream in(&identity);
    std::string text(9, '\0'); in.read(&text[0], 9); assert(text == "real file");
    assert(in.peek() == std::char_traits<char>::eof());
    rpmd_ja_sha256::Digest expected; expected.update(reinterpret_cast<const unsigned char*>(text.data()), text.size());
    assert(identity.sha256() == expected.finish_hex());
  }
  assert(std::remove(path.c_str()) == 0);
}
