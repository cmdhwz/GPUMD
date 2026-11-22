#pragma once
#include "rpmd_ja_utf8_file_io.h"
#include "rpmd_ja_sha256.h"
#include <cmath>
#include <stdexcept>
#include <string>

struct R164JAReference {
  std::string id, potential_sha256, file_sha256;
  int atoms=0, row[3]={};
  double temperature=0, cell[3]={}, mass[3]={}, reference[3][3]={}, basis[3][3][2]={}, D[3][2][2]={};
};

inline R164JAReference read_r164_ja_reference(const std::string& path) {
  rpmd_ja_file_io::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file) throw std::runtime_error("cannot open JA reference file");
  std::ostringstream buffer;
  buffer << file.rdbuf();
  if (file.bad() || buffer.bad()) throw std::runtime_error("failed reading JA reference file");
  const std::string contents=buffer.str();
  rpmd_ja_sha256::Digest digest;
  digest.update(reinterpret_cast<const unsigned char*>(contents.data()),contents.size());
  std::istringstream in(contents);
  R164JAReference r;
  r.file_sha256=digest.finish_hex();
  auto key=[&](const char* expected) { std::string s; if (!(in>>s) || s!=expected) throw std::runtime_error(std::string("expected ")+expected); };
  int version=0;
  key("r164_ja_reference"); if (!(in>>version) || version!=1) throw std::runtime_error("unsupported JA reference version");
  key("id"); if (!(in>>r.id)) throw std::runtime_error("missing reference id");
  key("potential_sha256"); if (!(in>>r.potential_sha256) || r.potential_sha256.size()!=64) throw std::runtime_error("invalid potential SHA256");
  for (char c:r.potential_sha256) if (!((c>='0' && c<='9') || (c>='a' && c<='f'))) throw std::runtime_error("invalid potential SHA256");
  key("atoms"); if (!(in>>r.atoms) || r.atoms<=0) throw std::runtime_error("invalid atom count");
  key("temperature_K"); if (!(in>>r.temperature) || !std::isfinite(r.temperature) || r.temperature<=0) throw std::runtime_error("invalid temperature");
  auto number=[&](double& x) { if (!(in>>x) || !std::isfinite(x)) throw std::runtime_error("missing or nonfinite reference value"); };
  key("cell_A"); for (double& x:r.cell) { number(x); if (x<=0) throw std::runtime_error("invalid cell"); }
  key("rows"); for (int& i:r.row) if (!(in>>i) || i<0 || i>=r.atoms) throw std::runtime_error("invalid row index");
  if (r.row[0]==r.row[1] || r.row[0]==r.row[2] || r.row[1]==r.row[2]) throw std::runtime_error("duplicate row index");
  key("masses_eV_fs2_A2"); for (double& x:r.mass) { number(x); if (x<=0) throw std::runtime_error("invalid mass"); }
  key("reference_A"); for (auto& a:r.reference) for (double& x:a) number(x);
  key("basis"); for (auto& a:r.basis) for (auto& d:a) for (double& x:d) number(x);
  key("D"); for (auto& d:r.D) for (auto& p:d) for (double& x:p) number(x);
  std::string extra; if (in>>extra) throw std::runtime_error("trailing JA reference input");
  for (int k=0;k<2;++k) for (int l=0;l<2;++l) {
    double dot=0; for (int a=0;a<3;++a) for (int d=0;d<3;++d) dot+=r.mass[a]*r.basis[a][d][k]*r.basis[a][d][l];
    if (!std::isfinite(dot) || std::abs(dot-(k==l ? 1.0 : 0.0))>1e-6) throw std::runtime_error("basis is not mass orthonormal");
  }
  return r;
}
