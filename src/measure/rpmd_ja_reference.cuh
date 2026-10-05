#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

class Atom;
class Box;
class Force;

struct RpmdJASparseMatrix
{
  std::vector<std::uint64_t> row_offsets;
  std::vector<int> columns;
  std::vector<double> values;
};

struct RpmdJAMatrixTile
{
  int row = 0, column = 0, rows = 0, columns = 0, rank = 0;
  std::vector<double> left, right;
};

struct RpmdJABlockMatrix
{
  int tile_size = 128;
  std::vector<RpmdJAMatrixTile> tiles;
};

struct RpmdJAReference
{
  int backend = 0; // 0: dense v1, 1: sparse/operator v2, 2: qNEP block v3.
  int number_of_atoms = 0;
  double temperature = 0.0;
  double fd_step = 0.0;
  std::uint64_t model_fingerprint = 0;
  std::uint64_t reference_edge_count = 0;
  std::uint64_t reference_edge_fingerprint = 0;
  std::string edge_policy = "fixed_d0_raw_nep_image_keys_v1";
  std::vector<int> types;
  std::vector<double> masses;
  std::vector<double> positions; // Structure-of-arrays: d * N + i.
  double cell[9] = {};
  int pbc[3] = {};
  std::vector<double> delta_h[3]; // Row-major, row displacement and column velocity.
  RpmdJASparseMatrix dynamical; // D rows by input-displacement columns.
  RpmdJASparseMatrix site_transpose[3]; // B_alpha^T: output velocity rows, input displacement columns.
  double spectral_bound = 0.0;
  double kernel_u = 0.0;
  double kernel_error[2] = {};
  double kernel_s2[2] = {};
  int kernel_degree = 0;
  int p_rank = 0;
  int q_rank = 0;
  std::vector<double> p_values, q_values;
  std::vector<double> p_vectors, q_vectors; // Row-major [degree + 1, rank].
  std::string stability_certificate;
  double minimum_cholesky_pivot = 0.0;
  double additive_epsilon = 0.0;
  double additive_reconstruction_bound = 0.0;
  int additive_beads = 0;
  double relative_operator_bound = 0.0;
  double fd_relative_d = 0.0;
  double fd_relative_b[3] = {};
  bool stability_checked = false;
  RpmdJABlockMatrix block_dynamical;
  RpmdJABlockMatrix block_site_transpose[3];
  int q_charge_mode = -1;
  bool q_uses_pppm = false;
  double q_mesh_spacing = 0.0;
  std::string mechanical_policy;
  std::uint64_t mechanical_config_fingerprint = 0;
  double energy_gradient_relative_error = 0.0;
  double force_gradient_relative_error = 0.0;
  double hessian_symmetry_relative_error = 0.0;
  double energy_second_probe_relative_error = 0.0;
  double force_balance_residual = 0.0;
  double projection_relative_change = 0.0;
  double energy_gradient_absolute_rms = 0.0;
  double force_gradient_absolute_rms = 0.0;
  double site_derivative_absolute_rms[3] = {};
  double site_transport_difference_absolute_rms[3] = {};
  double site_transport_relative_error[3] = {};
  double block_relative_residual[4] = {};
};

RpmdJAReference read_rpmd_ja_reference(const std::string& path);
int rpmd_ja_reference_policy_beads(const std::string& policy);
void load_rpmd_ja_kernel_table(const std::string& path, RpmdJAReference& reference);
std::streampos write_rpmd_ja_qnep_v3_stream_prefix(std::ostream& out, const RpmdJAReference& reference);
void generate_rpmd_ja_reference(
  const std::string& path,
  double temperature,
  double fd_step,
  Atom& atom,
  Box& box,
  Force& force);
void generate_rpmd_ja_sparse_reference(
  const std::string& path,
  double temperature,
  double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force);
void generate_rpmd_ja_qnep_reference(
  const std::string& path,
  double temperature,
  double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force);
void generate_rpmd_ja_qnep_raw(
  const std::string& raw_path,
  double temperature,
  double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force,
  bool require_zero_net_gradient = false);
void diagnose_rpmd_ja_qnep_reference(double fd_step, Atom& atom, Box& box, Force& force);
std::uint64_t rpmd_ja_model_fingerprint(const std::string& path);
std::uint64_t rpmd_ja_qnep_config_fingerprint(Force& force);
