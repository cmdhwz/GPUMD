#pragma once

#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_reference_math.cuh"
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
  bool internal_mass_com = false;
  double epsilon = 0.0;
  RpmdJASparseRows k;
  RpmdJASparseRows h[3]; // Stored as H^T for the existing prepare tile orientation.
  std::vector<double> linear_gradient; // SoA [axis, atom].
};

struct RpmdJAAdditiveEntry { int row, column; double value; };

constexpr std::uint64_t rpmd_ja_additive_max_expanded_elements = 4000000;

inline std::uint64_t rpmd_ja_additive_validate_site_dimension(const std::uint64_t z,const int site=-1)
{
  if(z>std::numeric_limits<std::uint64_t>::max()/3)
    throw std::runtime_error("GPJAADD1 local matrix dimension overflows");
  const std::uint64_t q=3*z;
  if(q&&q>rpmd_ja_additive_max_expanded_elements/q)
    throw std::runtime_error("GPJAADD1 local matrix exceeds the per-site element limit"+
      (site>=0?" at site "+std::to_string(site):std::string())+
      " ("+std::to_string(q)+"x"+std::to_string(q)+")");
  return q*q;
}

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
  rpmd_ja_file_io::ifstream size_in(path, std::ios::binary | std::ios::ate);
  if (!size_in || size_in.tellg() < 88) throw std::runtime_error("GPJAADD1 package is shorter than its header");
  const std::streamoff package_bytes = size_in.tellg();
  if (static_cast<std::uintmax_t>(package_bytes) > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("GPJAADD1 package exceeds host addressable size");
  rpmd_ja_file_io::ifstream in(path, std::ios::binary);
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
  if (std::string(magic, sizeof(magic)) != "GPJAADD1" || (version < 1 || version > 6) || endian != 0x01020304 ||
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
  if (static_cast<std::uint64_t>(n) > static_cast<std::uint64_t>(package_bytes - 80) / sizeof(std::int32_t))
    throw std::runtime_error("GPJAADD1 declared site count exceeds package length");
  RpmdJAAdditiveData out; out.beads = p; out.epsilon = epsilon; out.internal_mass_com = version == 2 || version == 4 || version == 6;
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
    const std::uint64_t dense_elements=rpmd_ja_additive_validate_site_dimension(z,i);
    const bool sparse_quadratic=version>=3;
    const bool mode_quadratic=version>=5;
    const std::uint64_t edge_bytes = static_cast<std::uint64_t>(z) * 16;
    const std::uint64_t ell_bytes = q * sizeof(double);
    const std::uint64_t b_bytes = dense_elements * sizeof(double);
    const std::uint64_t quadratic_header_bytes=mode_quadratic?sizeof(std::uint32_t):sparse_quadratic?sizeof(std::uint64_t):b_bytes;
    if (edge_bytes > max_bytes - ell_bytes || edge_bytes + ell_bytes > max_bytes - quadratic_header_bytes)
      throw std::runtime_error("GPJAADD1 local block byte count overflows");
    const std::uint64_t need = edge_bytes + ell_bytes + quadratic_header_bytes;
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
    in.read(reinterpret_cast<char*>(ell.data()), static_cast<std::streamsize>(ell.size()*sizeof(double)));
    if(!in)throw std::runtime_error("truncated GPJAADD1 linear coefficients");
    std::vector<double> b;
    auto read_sparse_scalars=[&](const char* description){
      const std::uint64_t count=rpmd_ja_additive_read<std::uint64_t>(in);
      const std::uint64_t maximum_entries=dense_elements;
      if(count>maximum_entries||count>max_bytes/(2*sizeof(std::int32_t)+sizeof(double)))
        throw std::runtime_error(std::string("invalid GPJAADD1 ")+description+" count at site "+std::to_string(i));
      const std::uint64_t entry_bytes=count*(2*sizeof(std::int32_t)+sizeof(double));
      const std::streamoff payload_pos=in.tellg();
      if(payload_pos<0||payload_pos>package_bytes||entry_bytes>static_cast<std::uint64_t>(package_bytes-payload_pos))
        throw std::runtime_error(std::string("truncated GPJAADD1 ")+description+" at site "+std::to_string(i)+
          " ("+std::to_string(q)+"x"+std::to_string(q)+")");
      std::uint64_t previous_index=0;bool first_entry=true;
      for(std::uint64_t entry=0;entry<count;++entry){
        const int row=rpmd_ja_additive_read<std::int32_t>(in),column=rpmd_ja_additive_read<std::int32_t>(in);
        const double value=rpmd_ja_additive_read<double>(in);
        if(row<0||row>=qi||column<0||column>=qi||!std::isfinite(value)||(value==0.0&&!std::signbit(value)))
          throw std::runtime_error(std::string("invalid GPJAADD1 ")+description+" entry at site "+std::to_string(i));
        const std::uint64_t index=static_cast<std::uint64_t>(row)*qi+column;
        if(!first_entry&&index<=previous_index)
          throw std::runtime_error(std::string("unordered or duplicate GPJAADD1 ")+description+" entry at site "+std::to_string(i));
        previous_index=index;first_entry=false;
      }
      in.clear();in.seekg(payload_pos);
      b.assign(static_cast<std::size_t>(dense_elements),0.0);
      for(std::uint64_t entry=0;entry<count;++entry){
        const int row=rpmd_ja_additive_read<std::int32_t>(in),column=rpmd_ja_additive_read<std::int32_t>(in);
        const double value=rpmd_ja_additive_read<double>(in);
        b[static_cast<std::size_t>(row)*qi+column]=value;
      }
    };
    if(!mode_quadratic&&!sparse_quadratic){
      const std::streamoff payload_pos=in.tellg();
      if(payload_pos<0||payload_pos>package_bytes||b_bytes>static_cast<std::uint64_t>(package_bytes-payload_pos))
        throw std::runtime_error("truncated GPJAADD1 dense quadratic at site "+std::to_string(i)+
          " ("+std::to_string(q)+"x"+std::to_string(q)+")");
      b.resize(static_cast<std::size_t>(dense_elements));
      in.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b_bytes));
    }else if(!mode_quadratic){
      read_sparse_scalars("sparse quadratic");
    }else{
      const std::uint32_t mode=rpmd_ja_additive_read<std::uint32_t>(in);
      if(mode==0){
        const std::streamoff payload_pos=in.tellg();
        if(payload_pos<0||payload_pos>package_bytes||b_bytes>static_cast<std::uint64_t>(package_bytes-payload_pos))
          throw std::runtime_error("truncated GPJAADD1 dense quadratic at site "+std::to_string(i)+
            " ("+std::to_string(q)+"x"+std::to_string(q)+")");
        b.resize(static_cast<std::size_t>(dense_elements));
        in.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b_bytes));
      }else if(mode==1){
        read_sparse_scalars("sparse quadratic");
      }else if(mode==2){
        const std::uint64_t count=rpmd_ja_additive_read<std::uint64_t>(in);
        const std::uint64_t maximum_blocks=static_cast<std::uint64_t>(z)*z;
        if(count>maximum_blocks||count>max_bytes/(2*sizeof(std::int32_t)+9*sizeof(double)))
          throw std::runtime_error("invalid GPJAADD1 sparse 3x3 block count at site "+std::to_string(i));
        const std::uint64_t block_bytes=count*(2*sizeof(std::int32_t)+9*sizeof(double));
        const std::streamoff payload_pos=in.tellg();
        if(payload_pos<0||payload_pos>package_bytes||block_bytes>static_cast<std::uint64_t>(package_bytes-payload_pos))
          throw std::runtime_error("truncated GPJAADD1 sparse 3x3 blocks at site "+std::to_string(i)+
            " ("+std::to_string(z)+"x"+std::to_string(z)+")");
        std::uint64_t previous_index=0;bool first_block=true;
        for(std::uint64_t block=0;block<count;++block){
          const int row=rpmd_ja_additive_read<std::int32_t>(in),column=rpmd_ja_additive_read<std::int32_t>(in);
          if(row<0||row>=z||column<0||column>=z)
            throw std::runtime_error("invalid GPJAADD1 sparse 3x3 block index at site "+std::to_string(i));
          const std::uint64_t index=static_cast<std::uint64_t>(row)*z+column;
          if(!first_block&&index<=previous_index)
            throw std::runtime_error("unordered or duplicate GPJAADD1 sparse 3x3 block at site "+std::to_string(i));
          for(int value=0;value<9;++value)if(!std::isfinite(rpmd_ja_additive_read<double>(in)))
            throw std::runtime_error("non-finite GPJAADD1 sparse 3x3 block value at site "+std::to_string(i));
          previous_index=index;first_block=false;
        }
        in.clear();in.seekg(payload_pos);
        b.assign(static_cast<std::size_t>(dense_elements),0.0);
        for(std::uint64_t block=0;block<count;++block){
          const int row=rpmd_ja_additive_read<std::int32_t>(in),column=rpmd_ja_additive_read<std::int32_t>(in);
          for(int a=0;a<3;++a)for(int c=0;c<3;++c)
            b[static_cast<std::size_t>(3*row+a)*qi+3*column+c]=rpmd_ja_additive_read<double>(in);
        }
      }else{
        throw std::runtime_error("invalid GPJAADD1 quadratic mode "+std::to_string(mode)+" at site "+
          std::to_string(i)+" ("+std::to_string(q)+"x"+std::to_string(q)+")");
      }
    }
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
  const std::vector<double> target_gradient = out.internal_mass_com ?
    rpmd_ja_reference_math::mass_com_covector_pullback(raw_gradient, ref.masses) : raw_gradient;
  double residual=0.0, norm=0.0;
  for(int i=0;i<d;++i){if(!std::isfinite(target_gradient[i]))throw std::runtime_error("non-finite raw qNEP reference gradient");residual=std::hypot(residual,target_gradient[i]+out.linear_gradient[i]);norm=std::hypot(norm,target_gradient[i]);}
  if(!std::isfinite(residual)||!std::isfinite(norm)||residual>1.0e-8*std::max(1.0,norm))
    throw std::runtime_error("GPJAADD1 linear term does not cancel raw reference gradient");
  return out;
}
