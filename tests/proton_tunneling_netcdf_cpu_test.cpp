#include "measure/proton_tunneling_netcdf.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

using namespace proton_tunneling_netcdf;

namespace
{
struct Table
{
  int group;
  int value_var;
  int count_var;
  int rows;
  int value_fields;
  int count_fields;
  const char* row_name;
  const char* value_names;
  const char* count_names;
};

void require(const bool condition, const char* message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", message);
    std::exit(1);
  }
}

void put_field_names(const int group, const int variable, const char* names)
{
  netcdf_check(nc_put_att_text(group, variable, "field_names", std::strlen(names), names),
    "nc_put_att_text");
}

Table define_table(
  const int ncid,
  const char* group_name,
  const char* row_name,
  const int rows,
  const int value_fields,
  const int count_fields,
  const char* value_names,
  const char* count_names)
{
  Table table{};
  table.rows = rows;
  table.value_fields = value_fields;
  table.count_fields = count_fields;
  table.row_name = row_name;
  table.value_names = value_names;
  table.count_names = count_names;
  netcdf_check(nc_def_grp(ncid, group_name, &table.group), "nc_def_grp");
  int row_dim = -1, value_dim = -1, count_dim = -1;
  netcdf_check(nc_def_dim(table.group, row_name, rows, &row_dim), "nc_def_dim");
  netcdf_check(nc_def_dim(table.group, "value", value_fields, &value_dim), "nc_def_dim");
  netcdf_check(nc_def_dim(table.group, "count", count_fields, &count_dim), "nc_def_dim");
  table.value_var = netcdf_variable(table.group, "value", NC_FLOAT,
    {row_dim, value_dim}, {static_cast<size_t>(rows), static_cast<size_t>(value_fields)}, 1, true);
  table.count_var = netcdf_variable(table.group, "count", NC_INT64,
    {row_dim, count_dim}, {static_cast<size_t>(rows), static_cast<size_t>(count_fields)}, 1, true);
  put_field_names(table.group, table.value_var, value_names);
  put_field_names(table.group, table.count_var, count_names);
  return table;
}

std::vector<double> make_values(const int rows, const int fields)
{
  std::vector<double> values(static_cast<size_t>(rows) * fields);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = i % 101 == 0 ? std::numeric_limits<double>::quiet_NaN() : 0.1 + i * 0.125;
  return values;
}

std::vector<long long> make_counts(const int rows, const int fields)
{
  std::vector<long long> counts(static_cast<size_t>(rows) * fields);
  for (size_t i = 0; i < counts.size(); ++i)
    counts[i] = 9007199254740993LL + static_cast<long long>(i);
  return counts;
}

void write_table(const Table& table)
{
  netcdf_write_double(table.group, table.value_var, make_values(table.rows, table.value_fields));
  netcdf_write_longlong(table.group, table.count_var, make_counts(table.rows, table.count_fields));
}

void check_var_schema(
  const int group,
  const int variable,
  const nc_type expected_type,
  const char* first_dimension,
  const char* second_dimension,
  const size_t rows,
  const size_t fields,
  const char* expected_names)
{
  nc_type type = NC_NAT;
  netcdf_check(nc_inq_vartype(group, variable, &type), "nc_inq_vartype");
  require(type == expected_type, "external variable type");
  int rank = 0;
  netcdf_check(nc_inq_varndims(group, variable, &rank), "nc_inq_varndims");
  require(rank == 2, "table variable rank");
  int dimensions[2] = {-1, -1};
  netcdf_check(nc_inq_vardimid(group, variable, dimensions), "nc_inq_vardimid");
  char name[NC_MAX_NAME + 1] = {};
  netcdf_check(nc_inq_dimname(group, dimensions[0], name), "nc_inq_dimname");
  require(std::strcmp(name, first_dimension) == 0, "first dimension order");
  netcdf_check(nc_inq_dimname(group, dimensions[1], name), "nc_inq_dimname");
  require(std::strcmp(name, second_dimension) == 0, "second dimension order");
  size_t chunks[2] = {};
  int storage = NC_CONTIGUOUS;
  netcdf_check(nc_inq_var_chunking(group, variable, &storage, chunks), "nc_inq_var_chunking");
  require(storage == NC_CHUNKED, "variable chunked");
  require(chunks[0] == std::min<size_t>(rows, 16384) && chunks[1] == 1, "field chunk shape");
  int shuffle = 0, deflate = 0, level = 0;
  netcdf_check(nc_inq_var_deflate(group, variable, &shuffle, &deflate, &level), "nc_inq_var_deflate");
  require(shuffle == 1 && deflate == 1 && level == 1, "shuffle and deflate settings");
  size_t attribute_length = 0;
  netcdf_check(nc_inq_attlen(group, variable, "field_names", &attribute_length), "nc_inq_attlen");
  std::vector<char> names(attribute_length + 1, '\0');
  netcdf_check(nc_get_att_text(group, variable, "field_names", names.data()), "nc_get_att_text");
  require(std::strcmp(names.data(), expected_names) == 0, "field names order");
  size_t actual_rows = 0, actual_fields = 0;
  netcdf_check(nc_inq_dimlen(group, dimensions[0], &actual_rows), "nc_inq_dimlen");
  netcdf_check(nc_inq_dimlen(group, dimensions[1], &actual_fields), "nc_inq_dimlen");
  require(actual_rows == rows && actual_fields == fields, "dimension lengths");
}

void verify_table(const Table& table)
{
  const size_t value_size = static_cast<size_t>(table.rows) * table.value_fields;
  std::vector<float> values(value_size);
  netcdf_check(nc_get_var_float(table.group, table.value_var, values.data()), "nc_get_var_float");
  const std::vector<double> expected_values = make_values(table.rows, table.value_fields);
  for (size_t i = 0; i < value_size; ++i) {
    if (std::isnan(expected_values[i]))
      require(std::isnan(values[i]), "NaN readback");
    else
      require(values[i] == static_cast<float>(expected_values[i]), "float32 value readback");
  }

  const size_t count_size = static_cast<size_t>(table.rows) * table.count_fields;
  std::vector<long long> counts(count_size);
  netcdf_check(nc_get_var_longlong(table.group, table.count_var, counts.data()), "nc_get_var_longlong");
  require(counts == make_counts(table.rows, table.count_fields), "int64 count readback");
  check_var_schema(table.group, table.value_var, NC_FLOAT, table.row_name, "value",
    table.rows, table.value_fields, table.value_names);
  check_var_schema(table.group, table.count_var, NC_INT64, table.row_name, "count",
    table.rows, table.count_fields, table.count_names);
}

void test_rows(const int rows)
{
  const char* path = "proton_tunneling_netcdf_cpu_test.nc";
  int ncid = -1;
  netcdf_check(nc_create(path, NC_NETCDF4 | NC_CLOBBER, &ncid), "nc_create");
  const Table window = define_table(ncid, "window", "window", rows, 8, 3,
    "B_mean,f_02,f_04,mean_abs_delta_f,flip_rate_per_ps,positive_defects,negative_defects,valid_pairs_per_frame",
    "active_bonds,assignment_ambiguous_samples,pair_conflict_samples");
  const Table edge_window = define_table(ncid, "edge_window", "row", rows, 34, 9,
    "geometry_occupancy,asymmetry,abs_asymmetry,delta_f,success_probability,mean_delta,mean_abs_delta,mean_dOO,mean_rperp,mean_E_parallel,std_E_parallel,corr_delta_E_parallel,mean_E_success,mean_E_return,nearest_ion1_distance,nearest_ion2_distance,log_population_ratio,beta_DeltaF_high_minus_low,abs_beta_DeltaF,mean_delta_phi_ion,std_delta_phi_ion,corr_delta_delta_phi,mean_ion1_to_O_low,mean_ion1_to_O_high,mean_delta_d_ion1,mean_ion2_to_O_low,mean_ion2_to_O_high,mean_delta_d_ion2,t_core_minus_fs,t_core_plus_fs,t_core_center_fs,t_state_minus_fs,t_state_plus_fs,observation_gap_fs",
    "n_plus,n_minus,n_deadband,attempts,successes,returns,geometry_lost,run_end,observation_gaps");
  int endpoint_dim = -1;
  netcdf_check(nc_def_dim(window.group, "endpoint", 2, &endpoint_dim), "nc_def_dim");
  int window_dim = -1;
  netcdf_check(nc_inq_dimid(window.group, "window", &window_dim), "nc_inq_dimid");
  int time_var = netcdf_variable(window.group, "time_fs", NC_DOUBLE,
    {window_dim, endpoint_dim}, {static_cast<size_t>(rows), 2}, 1);
  netcdf_check(nc_enddef(ncid), "nc_enddef");
  write_table(window);
  write_table(edge_window);
  std::vector<double> times(static_cast<size_t>(rows) * 2);
  for (size_t i = 0; i < times.size(); ++i)
    times[i] = 100.0 + i * 0.125;
  netcdf_write_double(window.group, time_var, times);
  netcdf_check(nc_close(ncid), "nc_close");

  netcdf_check(nc_open(path, NC_NOWRITE, &ncid), "nc_open");
  int window_group = -1, edge_group = -1, window_value_var = -1, window_count_var = -1;
  int edge_value_var = -1, edge_count_var = -1, read_time_var = -1;
  netcdf_check(nc_inq_ncid(ncid, "window", &window_group), "nc_inq_ncid");
  netcdf_check(nc_inq_ncid(ncid, "edge_window", &edge_group), "nc_inq_ncid");
  netcdf_check(nc_inq_varid(window_group, "value", &window_value_var), "nc_inq_varid");
  netcdf_check(nc_inq_varid(window_group, "count", &window_count_var), "nc_inq_varid");
  netcdf_check(nc_inq_varid(window_group, "time_fs", &read_time_var), "nc_inq_varid");
  netcdf_check(nc_inq_varid(edge_group, "value", &edge_value_var), "nc_inq_varid");
  netcdf_check(nc_inq_varid(edge_group, "count", &edge_count_var), "nc_inq_varid");
  verify_table({window_group, window_value_var, window_count_var, rows, 8, 3, "window",
    window.value_names, window.count_names});
  verify_table({edge_group, edge_value_var, edge_count_var, rows, 34, 9, "row",
    edge_window.value_names, edge_window.count_names});

  nc_type time_type = NC_NAT;
  netcdf_check(nc_inq_vartype(window_group, read_time_var, &time_type), "nc_inq_vartype");
  require(time_type == NC_DOUBLE, "absolute window time remains float64");
  int time_ndims = 0;
  netcdf_check(nc_inq_varndims(window_group, read_time_var, &time_ndims), "nc_inq_varndims");
  require(time_ndims == 2, "absolute window time rank");
  int time_dims[2] = {-1, -1};
  netcdf_check(nc_inq_vardimid(window_group, read_time_var, time_dims), "nc_inq_vardimid");
  char time_dim_name[NC_MAX_NAME + 1] = {};
  netcdf_check(nc_inq_dimname(window_group, time_dims[0], time_dim_name), "nc_inq_dimname");
  require(std::strcmp(time_dim_name, "window") == 0, "absolute time first dimension");
  netcdf_check(nc_inq_dimname(window_group, time_dims[1], time_dim_name), "nc_inq_dimname");
  require(std::strcmp(time_dim_name, "endpoint") == 0, "absolute time endpoint dimension");
  std::vector<double> read_times(times.size());
  netcdf_check(nc_get_var_double(window_group, read_time_var, read_times.data()), "nc_get_var_double");
  require(read_times == times, "absolute window time readback");
  netcdf_check(nc_close(ncid), "nc_close");
  std::remove(path);
}
}

int main()
{
  test_rows(1);
  test_rows(16385);
  std::puts("PASS proton_tunneling_netcdf_cpu_test");
  return 0;
}
