#pragma once

#include "netcdf.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace proton_tunneling_netcdf
{
inline void netcdf_check(const int status, const char* operation)
{
  if (status != NC_NOERR) {
    std::fprintf(stderr, "Proton observer NetCDF error in %s: %s\n", operation, nc_strerror(status));
    std::exit(2);
  }
}

inline int netcdf_variable(
  const int group,
  const char* name,
  const nc_type type,
  const std::vector<int>& dimensions,
  const std::vector<size_t>& lengths,
  const int compression_level,
  const bool field_chunks = false)
{
  int variable = -1;
  netcdf_check(
    nc_def_var(group, name, type, static_cast<int>(dimensions.size()), dimensions.data(), &variable),
    "nc_def_var");
  if (!dimensions.empty()) {
    std::vector<size_t> chunks(dimensions.size(), 1);
    for (size_t i = 0; i < dimensions.size(); ++i)
      chunks[i] = std::max<size_t>(1, std::min<size_t>(lengths[i], 16384));
    if (field_chunks)
      chunks.back() = 1;
    netcdf_check(nc_def_var_chunking(group, variable, NC_CHUNKED, chunks.data()), "nc_def_var_chunking");
    netcdf_check(
      nc_def_var_deflate(group, variable, 1, 1, compression_level), "nc_def_var_deflate");
  }
  return variable;
}

inline void netcdf_write_double(const int group, const int variable, const std::vector<double>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_double(group, variable, values.data()), "nc_put_var_double");
}

inline void netcdf_write_longlong(
  const int group,
  const int variable,
  const std::vector<long long>& values)
{
  if (!values.empty())
    netcdf_check(nc_put_var_longlong(group, variable, values.data()), "nc_put_var_longlong");
}
}
