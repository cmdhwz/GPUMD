#pragma once

#include <cstdint>
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

struct RpmdJAReference
{
  int backend = 0; // 0: dense v1, 1: sparse/operator v2.
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
  double fd_relative_d = 0.0;
  double fd_relative_b[3] = {};
  bool stability_checked = false;
};

RpmdJAReference read_rpmd_ja_reference(const std::string& path);
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
std::uint64_t rpmd_ja_model_fingerprint(const std::string& path);
