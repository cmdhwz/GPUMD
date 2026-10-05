#pragma once

#include "rpmd_ja_reference.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

struct RpmdJASparseRows
{
  std::vector<std::size_t> offsets;
  std::vector<int> columns;
  std::vector<double> values;
};

struct RpmdJAAdditiveData
{
  int beads = 0;
  double epsilon = 0.0;
  RpmdJASparseRows k;
  RpmdJASparseRows h[3]; // Stored as H^T for the existing prepare tile orientation.
  std::vector<double> linear_gradient; // SoA [axis, atom].
};

struct RpmdJAAdditiveEntry { int row, column; double value; };

inline RpmdJASparseRows rpmd_ja_additive_compress(const int d, std::vector<RpmdJAAdditiveEntry>& entries)
{
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return a.row < b.row || (a.row == b.row && a.column < b.column);
  });
  RpmdJASparseRows out;
  out.offsets.assign(static_cast<std::size_t>(d) + 1, 0);
  for (std::size_t i = 0; i < entries.size();) {
    const int row = entries[i].row, column = entries[i].column;
    double value = 0.0;
    do { value += entries[i++].value; } while (i < entries.size() && entries[i].row == row && entries[i].column == column);
    if (value != 0.0) {
      if (!std::isfinite(value)) throw std::runtime_error("non-finite assembled GPJAADD1 sparse coefficient");
      ++out.offsets[static_cast<std::size_t>(row) + 1];
      out.columns.push_back(column); out.values.push_back(value);
    }
  }
  for (int row = 0; row < d; ++row) out.offsets[row + 1] += out.offsets[row];
  return out;
}

template <typename T> inline T rpmd_ja_additive_read(std::istream& in)
{
  T x{};
  in.read(reinterpret_cast<char*>(&x), sizeof(x));
  if (!in) throw std::runtime_error("truncated GPJAADD1 package");
  return x;
}

inline RpmdJAAdditiveData read_rpmd_ja_additive(
  const std::string& path, const std::string& raw_path, const RpmdJAReference& ref,
  const int beads, const std::vector<double>& raw_gradient)
{
  std::ifstream size_in(path, std::ios::binary | std::ios::ate);
  if (!size_in || size_in.tellg() < 88) throw std::runtime_error("GPJAADD1 package is shorter than its header");
  const std::streamoff package_bytes = size_in.tellg();
  if (static_cast<std::uintmax_t>(package_bytes) > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("GPJAADD1 package exceeds host addressable size");
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open GPJAADD1 package: " + path);
  char magic[8]{}; in.read(magic, sizeof(magic));
  const int version = rpmd_ja_additive_read<std::uint32_t>(in);
  const auto endian = rpmd_ja_additive_read<std::uint32_t>(in);
  const int n = rpmd_ja_additive_read<std::int32_t>(in);
  const int p = rpmd_ja_additive_read<std::int32_t>(in);
  const auto source = rpmd_ja_additive_read<std::uint64_t>(in);
  const double temperature = rpmd_ja_additive_read<double>(in);
  const double epsilon = rpmd_ja_additive_read<double>(in);
  const double response = rpmd_ja_additive_read<double>(in);
  const double budget = rpmd_ja_additive_read<double>(in);
  const auto training = rpmd_ja_additive_read<std::uint64_t>(in);
  const auto validation = rpmd_ja_additive_read<std::uint64_t>(in);
  if (std::string(magic, sizeof(magic)) != "GPJAADD1" || version != 1 || endian != 0x01020304 ||
      n < 2 || n != ref.number_of_atoms || p < 1 || (beads > 0 && p != beads) || !(temperature > 0.0) ||
      !std::isfinite(temperature) || temperature != ref.temperature ||
      !(epsilon > 0.0) || !std::isfinite(epsilon) || !(response >= 0.0) || !(budget >= response) ||
      !std::isfinite(response) || !std::isfinite(budget) || !training || !validation)
    throw std::runtime_error("GPJAADD1 dimensions, temperature, bead count, source, or metadata do not match");
  if (n > std::numeric_limits<int>::max() / 3) throw std::runtime_error("GPJAADD1 dimensions overflow");
  const int d = 3 * n;
  if (raw_gradient.size() != static_cast<std::size_t>(d))
    throw std::runtime_error("raw qNEP gradient length does not match additive dimensions");
  if (source != rpmd_ja_model_fingerprint(raw_path))
    throw std::runtime_error("GPJAADD1 source qNEP raw fingerprint does not match");
  RpmdJAAdditiveData out; out.beads = p; out.epsilon = epsilon;
  out.linear_gradient.assign(static_cast<std::size_t>(d), 0.0);
  std::vector<RpmdJAAdditiveEntry> k_entries;
  std::vector<RpmdJAAdditiveEntry> h_entries[3];
  struct Edge { int j; int image[3]; };
  std::vector<std::vector<Edge>> graph(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
  {
    const int z = rpmd_ja_additive_read<std::int32_t>(in);
    if (z < 0 || z > d || z > std::numeric_limits<int>::max()/3)
      throw std::runtime_error("invalid GPJAADD1 local coordination");
    const std::uint64_t q = static_cast<std::uint64_t>(z) * 3;
    const std::uint64_t max_bytes = std::min<std::uint64_t>(std::numeric_limits<std::size_t>::max(),
      static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()));
    if (q > max_bytes / sizeof(double) || (q && q > max_bytes / sizeof(double) / q))
      throw std::runtime_error("GPJAADD1 local block size overflows host address space");
    const std::uint64_t edge_bytes = static_cast<std::uint64_t>(z) * 16;
    const std::uint64_t ell_bytes = q * sizeof(double);
    const std::uint64_t b_bytes = q * q * sizeof(double);
    if (edge_bytes > max_bytes - ell_bytes || edge_bytes + ell_bytes > max_bytes - b_bytes)
      throw std::runtime_error("GPJAADD1 local block byte count overflows");
    const std::uint64_t need = edge_bytes + ell_bytes + b_bytes;
    const std::streamoff pos = in.tellg();
    if (pos < 0 || pos > package_bytes || need > static_cast<std::uint64_t>(package_bytes - pos))
      throw std::runtime_error("truncated GPJAADD1 site block");
    graph[i].resize(static_cast<std::size_t>(z));
    std::set<std::array<int,4>> keys;
    for (Edge& e : graph[i])
    {
      e.j = rpmd_ja_additive_read<std::int32_t>(in);
      for (int a = 0; a < 3; ++a) e.image[a] = rpmd_ja_additive_read<std::int32_t>(in);
      if (e.j < 0 || e.j >= n || e.j == i || !keys.insert({e.j,e.image[0],e.image[1],e.image[2]}).second)
        throw std::runtime_error("invalid or duplicate GPJAADD1 edge key");
      if (e.image[0] == std::numeric_limits<int>::min() || e.image[1] == std::numeric_limits<int>::min() ||
          e.image[2] == std::numeric_limits<int>::min())
        throw std::runtime_error("GPJAADD1 image index cannot be negated");
    }
    const int qi = 3 * z;
    std::vector<double> ell(static_cast<std::size_t>(qi));
    std::vector<double> b(static_cast<std::size_t>(qi) * qi);
    in.read(reinterpret_cast<char*>(ell.data()), static_cast<std::streamsize>(ell.size()*sizeof(double)));
    in.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()*sizeof(double)));
    if (!in) throw std::runtime_error("truncated GPJAADD1 coefficients");
    for (double x : ell) if (!std::isfinite(x)) throw std::runtime_error("non-finite GPJAADD1 linear coefficient");
    double bscale=0.0;
    for (int x=0; x<qi; ++x) for (int y=0; y<qi; ++y) {
      const double v=b[static_cast<std::size_t>(x)*qi+y];
      if (!std::isfinite(v)) throw std::runtime_error("non-finite GPJAADD1 quadratic coefficient");
      bscale=std::max(bscale,std::abs(v));
    }
    if (bscale>0.0) {
      double bnorm=0.0, asym=0.0;
      for(int x=0;x<qi;++x)for(int y=0;y<qi;++y){const double v=b[static_cast<std::size_t>(x)*qi+y]/bscale;const double dv=v-b[static_cast<std::size_t>(y)*qi+x]/bscale;bnorm+=v*v;asym+=dv*dv;}
      if(std::sqrt(asym)>1.0e-12*std::sqrt(bnorm))throw std::runtime_error("asymmetric GPJAADD1 local block");
    }
    std::vector<double> geometry(static_cast<std::size_t>(z)*3);
    for (int e=0;e<z;++e) {
      const Edge& key=graph[i][e];
      for (int a=0;a<3;++a) {
        double g=ref.positions[a*n+key.j]-ref.positions[a*n+i];
        for(int c=0;c<3;++c)g += ref.cell[3*a+c]*key.image[c];
        if(!std::isfinite(g))throw std::runtime_error("non-finite GPJAADD1 image geometry");
        geometry[3*e+a]=g;
      }
    }
    for(int x=0;x<qi;++x) {
      const int axis=x%3; const Edge& edge=graph[i][x/3];
      out.linear_gradient[axis*n+edge.j] += ell[x];
      out.linear_gradient[axis*n+i] -= ell[x];
    }
    for(int x=0;x<qi;++x) for(int y=0;y<qi;++y) {
      const int ax=x%3, ay=y%3; const Edge& ex=graph[i][x/3]; const Edge& ey=graph[i][y/3];
      const int ix[2]={ax*n+ex.j,ax*n+i}, iy[2]={ay*n+ey.j,ay*n+i};
      const double sign[2]={1.0,-1.0}; const double value=b[static_cast<std::size_t>(x)*qi+y];
      if(value==0.0)continue;
      for(int u=0;u<2;++u) for(int v=0;v<2;++v)
        k_entries.push_back({ix[u],iy[v],sign[u]*value*sign[v]});
    }
    for(int e=0;e<z;++e) for(int mu=0;mu<3;++mu) {
      const int velocity=mu*n+graph[i][e].j;
      for(int y=0;y<qi;++y) {
        const int axis=y%3; const Edge& edge=graph[i][y/3]; const int displacement[2]={axis*n+edge.j,axis*n+i};
        const double sign[2]={1.0,-1.0}; const double value=b[static_cast<std::size_t>(3*e+mu)*qi+y];
        if(value==0.0)continue;
        for(int alpha=0;alpha<3;++alpha) for(int v=0;v<2;++v)
          h_entries[alpha].push_back({velocity,displacement[v],-geometry[3*e+alpha]*value*sign[v]});
      }
    }
  }
  for(int i=0;i<n;++i) for(const Edge& e:graph[i]) {
    bool reverse=false;
    for(const Edge& b:graph[e.j]) if(b.j==i && b.image[0]==-e.image[0] && b.image[1]==-e.image[1] && b.image[2]==-e.image[2]) reverse=true;
    if(!reverse) throw std::runtime_error("GPJAADD1 graph lacks a reverse image edge");
  }
  std::vector<char> seen(static_cast<std::size_t>(n)); std::vector<int> todo(1,0); seen[0]=1;
  while(!todo.empty()){int i=todo.back();todo.pop_back();for(const Edge& e:graph[i])if(!seen[e.j]){seen[e.j]=1;todo.push_back(e.j);}}
  if(std::find(seen.begin(),seen.end(),0)!=seen.end())throw std::runtime_error("disconnected GPJAADD1 graph");
  char trailing;
  if(in.read(&trailing,1))throw std::runtime_error("trailing data in GPJAADD1 package");
  out.k = rpmd_ja_additive_compress(d, k_entries);
  std::vector<RpmdJAAdditiveEntry>().swap(k_entries);
  for(int alpha=0;alpha<3;++alpha){
    out.h[alpha]=rpmd_ja_additive_compress(d,h_entries[alpha]);
    std::vector<RpmdJAAdditiveEntry>().swap(h_entries[alpha]);
  }
  for(double x:out.linear_gradient)if(!std::isfinite(x))throw std::runtime_error("non-finite assembled GPJAADD1 gradient");
  double residual=0.0, norm=0.0;
  for(int i=0;i<d;++i){if(!std::isfinite(raw_gradient[i]))throw std::runtime_error("non-finite raw qNEP reference gradient");residual=std::hypot(residual,raw_gradient[i]+out.linear_gradient[i]);norm=std::hypot(norm,raw_gradient[i]);}
  if(!std::isfinite(residual)||!std::isfinite(norm)||residual>1.0e-8*std::max(1.0,norm))
    throw std::runtime_error("GPJAADD1 linear term does not cancel raw reference gradient");
  return out;
}
