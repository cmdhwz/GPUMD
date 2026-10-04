#include "measure/rpmd_ja.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace
{
using Dense = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

RpmdJASparseMatrix to_csr(const Dense& matrix)
{
  RpmdJASparseMatrix csr;
  csr.row_offsets.push_back(0);
  for (const auto& row : matrix) {
    for (std::size_t column = 0; column < row.size(); ++column) {
      if (row[column] != 0.0) {
        csr.columns.push_back(static_cast<int>(column));
        csr.values.push_back(row[column]);
      }
    }
    csr.row_offsets.push_back(csr.columns.size());
  }
  return csr;
}

Vector project(const Vector& input, const Vector& sqrt_mass)
{
  const int N = static_cast<int>(sqrt_mass.size());
  Vector result = input;
  double mass_sum = 0.0;
  for (double mass_root : sqrt_mass) mass_sum += mass_root * mass_root;
  for (int axis = 0; axis < 3; ++axis) {
    double coefficient = 0.0;
    for (int atom = 0; atom < N; ++atom)
      coefficient += sqrt_mass[atom] * input[axis * N + atom] / mass_sum;
    for (int atom = 0; atom < N; ++atom)
      result[axis * N + atom] -= coefficient * sqrt_mass[atom];
  }
  return result;
}

Vector apply(const Dense& matrix, const Vector& input)
{
  Vector output(matrix.size(), 0.0);
  for (std::size_t row = 0; row < matrix.size(); ++row)
    for (std::size_t column = 0; column < input.size(); ++column)
      output[row] += matrix[row][column] * input[column];
  return output;
}

double dot(const Vector& a, const Vector& b)
{
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
  return result;
}

double chebyshev_polynomial(const Vector& coefficients, const int degree, const double z)
{
  double previous = 1.0;
  double value = coefficients[0] * previous;
  if (degree == 0) return value;
  double current = z;
  value += coefficients[1] * current;
  for (int k = 2; k <= degree; ++k) {
    const double next = 2.0 * z * current - previous;
    value += coefficients[k] * next;
    previous = current;
    current = next;
  }
  return value;
}

double cached_modal_oracle(
  const Vector& weighted_y,
  const Vector& weighted_u,
  const std::array<Dense, 3>& bt,
  const int alpha,
  const RpmdJAReference& reference)
{
  constexpr int N = 2;
  const double inverse_sqrt5 = 1.0 / std::sqrt(5.0);
  const Vector mode{-2.0 * inverse_sqrt5, inverse_sqrt5};
  const Vector lambda{1.0e-10, 0.6, 0.6 + 1.0e-10};
  Vector modal_y(3), modal_u(3);
  for (int axis = 0; axis < 3; ++axis) {
    for (int atom = 0; atom < N; ++atom) {
      modal_y[axis] += mode[atom] * weighted_y[axis * N + atom];
      modal_u[axis] += mode[atom] * weighted_u[axis * N + atom];
    }
  }
  double correction = 0.0;
  const double tau = 1.0;
  const double Lambda = reference.kernel_u * reference.kernel_u / (tau * tau);
  for (int a = 0; a < 3; ++a) {
    const double za = 2.0 * lambda[a] / Lambda - 1.0;
    const double p_a = chebyshev_polynomial(reference.p_vectors, reference.kernel_degree, za);
    const double q_a = chebyshev_polynomial(reference.q_vectors, reference.kernel_degree, za);
    for (int b = 0; b < 3; ++b) {
      const double zb = 2.0 * lambda[b] / Lambda - 1.0;
      const double p_b = chebyshev_polynomial(reference.p_vectors, reference.kernel_degree, zb);
      const double q_b = chebyshev_polynomial(reference.q_vectors, reference.kernel_degree, zb);
      double at_ab = 0.0;
      double at_ba = 0.0;
      for (int atom_a = 0; atom_a < N; ++atom_a) {
        for (int atom_b = 0; atom_b < N; ++atom_b) {
          at_ab += mode[atom_a] * bt[alpha][a * N + atom_a][b * N + atom_b] * mode[atom_b];
          at_ba += mode[atom_b] * bt[alpha][b * N + atom_b][a * N + atom_a] * mode[atom_a];
        }
      }
      const double P = reference.p_values[0] * p_a * p_b;
      const double Q = reference.q_values[0] * q_a * q_b;
      const double K = lambda[a] * lambda[b] * P * at_ab + lambda[b] * Q * at_ba;
      correction += modal_u[a] * K * modal_y[b];
    }
  }
  return correction;
}

Dense translation_projected_dynamical()
{
  constexpr int N = 2;
  constexpr int D = 6;
  const double inverse_sqrt5 = 1.0 / std::sqrt(5.0);
  const Vector mode{-2.0 * inverse_sqrt5, inverse_sqrt5};
  const Vector lambda{1.0e-10, 0.6, 0.6 + 1.0e-10};
  Dense matrix(D, Vector(D, 0.0));
  for (int axis = 0; axis < 3; ++axis)
    for (int row = 0; row < N; ++row)
      for (int column = 0; column < N; ++column)
        matrix[axis * N + row][axis * N + column] =
          lambda[axis] * mode[row] * mode[column];
  return matrix;
}

Dense site_matrix(const int alpha)
{
  Dense matrix(6, Vector(6, 0.0));
  if (alpha == 1) return matrix;
  for (int row = 0; row < 6; ++row) {
    for (int column = 0; column < 6; ++column) {
      matrix[row][column] = row == column ? 0.13 * (alpha + 1)
                                          : 0.027 * (row + 1) - 0.019 * (column + alpha + 1);
    }
  }
  return matrix;
}

RpmdJABlockMatrix to_blocks(
  const Dense& matrix,
  const int tile_size,
  const bool include_low_rank = true,
  const bool use_matrix_values = false)
{
  RpmdJABlockMatrix blocks;
  blocks.tile_size = tile_size;
  const int D = static_cast<int>(matrix.size());
  const int number_of_tiles = (D - 1) / tile_size + 1;
  for (int row_tile = 0; row_tile < number_of_tiles; ++row_tile) {
    for (int column_tile = 0; column_tile < number_of_tiles; ++column_tile) {
      RpmdJAMatrixTile tile;
      tile.row = row_tile * tile_size;
      tile.column = column_tile * tile_size;
      tile.rows = std::min(tile_size, D - tile.row);
      tile.columns = std::min(tile_size, D - tile.column);
      if (include_low_rank && row_tile == 0 && column_tile == 0) {
        tile.rank = 1;
        tile.left.resize(tile.rows);
        tile.right.resize(tile.columns);
        for (int row = 0; row < tile.rows; ++row) tile.left[row] = 0.2 * (row + 1);
        for (int column = 0; column < tile.columns; ++column) tile.right[column] = 0.3 * (column + 2);
      } else {
        tile.rank = 0;
        tile.left.resize(static_cast<std::size_t>(tile.rows) * tile.columns);
        for (int row = 0; row < tile.rows; ++row) {
          for (int column = 0; column < tile.columns; ++column) {
            const int global_row = tile.row + row;
            const int global_column = tile.column + column;
            tile.left[static_cast<std::size_t>(row) * tile.columns + column] = use_matrix_values
              ? matrix[global_row][global_column]
              : row_tile == 0 && column_tile == 1
                ? 0.0
                : 0.07 * (global_row + 1) - 0.03 * (global_column + 2) +
                    0.011 * (global_row + 1) * (global_column + 1);
          }
        }
      }
      blocks.tiles.push_back(std::move(tile));
    }
  }
  return blocks;
}

Dense dense_from_blocks(const RpmdJABlockMatrix& blocks, const int dimension)
{
  Dense dense(dimension, Vector(dimension, 0.0));
  for (const auto& tile : blocks.tiles) {
    for (int row = 0; row < tile.rows; ++row) {
      for (int column = 0; column < tile.columns; ++column) {
        double value = 0.0;
        if (tile.rank == 0) {
          value = tile.left[static_cast<std::size_t>(row) * tile.columns + column];
        } else {
          for (int rank = 0; rank < tile.rank; ++rank)
            value += tile.left[static_cast<std::size_t>(row) * tile.rank + rank] *
              tile.right[static_cast<std::size_t>(rank) * tile.columns + column];
        }
        dense[tile.row + row][tile.column + column] = value;
      }
    }
  }
  return dense;
}

Vector correction_oracle(
  const Vector& y,
  const Vector& velocity,
  const Vector& sqrt_mass,
  const Dense& bt,
  const double lambda_p,
  const double lambda_q,
  const double p_factor,
  const double q_factor)
{
  const Vector py = project(y, sqrt_mass);
  const Vector pu = project(velocity, sqrt_mass);
  // D is identity, Lambda=2, hence Z=0 and T_0(Z)-T_2(Z)=I.
  const Vector l = py;
  const Vector r = pu;
  Vector result(1, 0.0);
  result[0] = lambda_p * p_factor * p_factor * dot(apply(bt, l), r) +
              lambda_q * q_factor * q_factor * dot(l, apply(bt, pu));
  return result;
}

void check_close(const double actual, const double expected)
{
  assert(std::isfinite(actual));
  assert(std::fabs(actual - expected) <= 2.0e-10 * std::max({1.0, std::fabs(actual), std::fabs(expected)}));
}
} // namespace

int main()
{
  constexpr int N = 2;
  constexpr int D = 3 * N;
  RpmdJAReference reference;
  reference.backend = 1;
  reference.number_of_atoms = N;
  reference.temperature = HBAR / K_B; // tau=1 in native time units.
  reference.stability_checked = true;
  reference.spectral_bound = 1.0;
  reference.kernel_u = std::sqrt(2.0); // Lambda=2; Z=2D/Lambda-I=0.
  reference.kernel_degree = 2;
  reference.p_rank = reference.q_rank = 1;
  reference.p_values = {-0.7};
  reference.q_values = {0.35};
  reference.p_vectors = {0.6, 0.2, -0.1};
  reference.q_vectors = {-0.4, 0.3, -0.25};
  Dense d(D, Vector(D, 0.0));
  for (int i = 0; i < D; ++i) d[i][i] = 1.0;
  reference.dynamical = to_csr(d);
  std::array<Dense, 3> bt;
  for (int alpha = 0; alpha < 3; ++alpha) {
    bt[alpha] = site_matrix(alpha);
    reference.site_transpose[alpha] = to_csr(bt[alpha]);
  }
  const Vector masses{1.0, 4.0};
  const Vector sqrt_mass{1.0, 2.0};
  RpmdJASparseWorkspace workspace;
  workspace.initialize(reference, masses);
  assert(workspace.dynamical_nnz() == D);
  assert(workspace.site_nnz(1) == 0);
  assert(workspace.allocated_bytes() > 0);

  RpmdJAReference zero_rank_reference = reference;
  zero_rank_reference.p_rank = zero_rank_reference.q_rank = 0;
  zero_rank_reference.p_values.clear();
  zero_rank_reference.q_values.clear();
  zero_rank_reference.p_vectors.clear();
  zero_rank_reference.q_vectors.clear();
  RpmdJASparseWorkspace zero_rank_workspace;
  zero_rank_workspace.initialize(zero_rank_reference, masses);

  Vector position(D), reference_position(D, 0.0), velocity(D);
  for (int i = 0; i < D; ++i) {
    position[i] = sqrt_mass[i % N] == 1.0 ? 0.21 * (i + 1) : -0.17 * (i + 1);
    velocity[i] = 0.11 * (i + 2) - 0.31;
  }
  Vector weighted_y(D);
  for (int i = 0; i < D; ++i) weighted_y[i] = sqrt_mass[i % N] * (position[i] - reference_position[i]);
  Vector weighted_u(D);
  for (int i = 0; i < D; ++i) weighted_u[i] = sqrt_mass[i % N] * velocity[i];
  // Match the production p/q modal coefficients: c0+c2*T2(0).
  constexpr double p_factor = 0.6 - (-0.1);
  constexpr double q_factor = -0.4 - (-0.25);

  GPU_Vector<double> positions(D), ref_positions(D), velocities(D), result(3);
  GPU_Vector<int> error(1, 0);
  positions.copy_from_host(position.data());
  ref_positions.copy_from_host(reference_position.data());
  velocities.copy_from_host(velocity.data());
  workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  int device_error = 0;
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  Vector actual(3);
  result.copy_to_host(actual.data());
  for (int alpha = 0; alpha < 3; ++alpha) {
    const double expected = correction_oracle(
      weighted_y, weighted_u, sqrt_mass, bt[alpha], -0.7, 0.35, p_factor, q_factor)[0];
    check_close(actual[alpha], expected);
  }
  check_close(actual[1], 0.0); // Zero B direction.

  RpmdJAReference block_reference = reference;
  block_reference.backend = 2;
  const Dense fixed_d = translation_projected_dynamical();
  block_reference.spectral_bound = 0.6 + 1.0e-10;
  block_reference.block_dynamical = to_blocks(fixed_d, 128, false, true);
  for (int alpha = 0; alpha < 3; ++alpha)
    block_reference.block_site_transpose[alpha] = to_blocks(bt[alpha], 128, false, true);
  RpmdJASparseWorkspace block_workspace;
  block_workspace.initialize(block_reference, masses);
  error.fill(0);
  block_workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  result.copy_to_host(actual.data());
  const Vector cached_actual = actual;
  for (int alpha = 0; alpha < 3; ++alpha) {
    const double expected = cached_modal_oracle(
      weighted_y, weighted_u, bt, alpha, block_reference);
    check_close(actual[alpha], expected);
  }
  check_close(actual[1], 0.0); // Zero B direction.

  RpmdJAReference zero_rank_block_reference = block_reference;
  zero_rank_block_reference.p_rank = zero_rank_block_reference.q_rank = 0;
  zero_rank_block_reference.p_values.clear();
  zero_rank_block_reference.q_values.clear();
  zero_rank_block_reference.p_vectors.clear();
  zero_rank_block_reference.q_vectors.clear();
  RpmdJASparseWorkspace zero_rank_block_workspace;
  zero_rank_block_workspace.initialize(zero_rank_block_reference, masses);
  error.fill(0);
  zero_rank_block_workspace.compute_correction(
    positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  result.copy_to_host(actual.data());
  for (double value : actual) check_close(value, 0.0);

  // Production block matvec covers dense, low-rank, and explicit zero tiles without expansion.
  constexpr int block_D = 5;
  constexpr int block_vectors = 2;
  const RpmdJABlockMatrix block_matrix = to_blocks(Dense(block_D, Vector(block_D, 0.0)), 2);
  const Dense block_oracle = dense_from_blocks(block_matrix, block_D);
  RpmdJABlockOperator block_operator;
  block_operator.initialize(block_matrix, block_D, block_vectors);
  assert(block_operator.tile_count() == 9);
  Vector block_input(block_D * block_vectors), block_expected(block_D * block_vectors);
  for (int vector = 0; vector < block_vectors; ++vector) {
    for (int column = 0; column < block_D; ++column)
      block_input[vector * block_D + column] = 0.13 * (column + 1) - 0.21 * vector;
    const Vector expected = apply(
      block_oracle,
      Vector(block_input.begin() + vector * block_D, block_input.begin() + (vector + 1) * block_D));
    std::copy(expected.begin(), expected.end(), block_expected.begin() + vector * block_D);
  }
  GPU_Vector<double> block_input_gpu(block_input.size()), block_output_gpu(block_expected.size());
  block_input_gpu.copy_from_host(block_input.data());
  error.fill(0);
  block_operator.apply(block_input_gpu.data(), block_vectors, block_output_gpu.data(), error.data());
  Vector block_actual(block_expected.size());
  block_output_gpu.copy_to_host(block_actual.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  for (int i = 0; i < static_cast<int>(block_actual.size()); ++i)
    check_close(block_actual[i], block_expected[i]);

  // Reinitialize the same operator with a single partial dense tile and no low-rank scratch use.
  constexpr int small_D = 3;
  const RpmdJABlockMatrix small_block_matrix = to_blocks(
    Dense(small_D, Vector(small_D, 0.0)), 128, false);
  const Dense small_block_oracle = dense_from_blocks(small_block_matrix, small_D);
  block_operator.initialize(small_block_matrix, small_D, 1);
  Vector small_input{0.2, -0.4, 0.7};
  const Vector small_expected = apply(small_block_oracle, small_input);
  GPU_Vector<double> small_input_gpu(small_D), small_output_gpu(small_D);
  small_input_gpu.copy_from_host(small_input.data());
  error.fill(0);
  block_operator.apply(small_input_gpu.data(), 1, small_output_gpu.data(), error.data());
  Vector small_actual(small_D);
  small_output_gpu.copy_to_host(small_actual.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  for (int i = 0; i < small_D; ++i) check_close(small_actual[i], small_expected[i]);

  RpmdJABlockMatrix overflow_blocks;
  overflow_blocks.tile_size = 128;
  RpmdJAMatrixTile overflow_tile;
  overflow_tile.row = overflow_tile.column = 0;
  overflow_tile.rows = overflow_tile.columns = 1;
  overflow_tile.left = {std::numeric_limits<double>::max()};
  overflow_blocks.tiles.push_back(overflow_tile);
  block_operator.initialize(overflow_blocks, 1, 1);
  GPU_Vector<double> overflow_input_gpu(1), overflow_output_gpu(1);
  const double overflow_input = 2.0;
  overflow_input_gpu.copy_from_host(&overflow_input);
  error.fill(RPMD_JA_ERROR_BRANCH);
  block_operator.apply(
    overflow_input_gpu.data(), 1, overflow_output_gpu.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert((device_error & RPMD_JA_ERROR_BRANCH) != 0);
  assert((device_error & RPMD_JA_ERROR_NUMERICAL) != 0);

  error.fill(0);
  zero_rank_workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  result.copy_to_host(actual.data());
  for (double value : actual) check_close(value, 0.0);

  // Adding a rigid translation and common velocity must be removed by P.
  for (int atom = 0; atom < N; ++atom) {
    for (int axis = 0; axis < 3; ++axis) {
      position[axis * N + atom] += 3.7;
      velocity[axis * N + atom] += 0.8;
    }
  }
  for (int i = 0; i < D; ++i) weighted_u[i] = sqrt_mass[i % N] * velocity[i];
  positions.copy_from_host(position.data());
  velocities.copy_from_host(velocity.data());
  error.fill(0);
  workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  result.copy_to_host(actual.data());
  for (int alpha = 0; alpha < 3; ++alpha) {
    const double expected = correction_oracle(
      weighted_y, weighted_u, sqrt_mass, bt[alpha], -0.7, 0.35, p_factor, q_factor)[0];
    check_close(actual[alpha], expected);
  }
  error.fill(0);
  block_workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  result.copy_to_host(actual.data());
  for (int alpha = 0; alpha < 3; ++alpha)
    check_close(actual[alpha], cached_actual[alpha]);

  // Full-fractional wrapping supports offsets spanning multiple cells.
  Box box;
  for (int i = 0; i < 18; ++i) box.cpu_h[i] = 0.0;
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10.0;
  box.cpu_h[1] = 2.0;
  box.cpu_h[5] = 1.0;
  box.cpu_h[9] = 0.1;
  box.cpu_h[10] = -0.02;
  box.cpu_h[11] = 0.002;
  box.cpu_h[13] = 0.1;
  box.cpu_h[14] = -0.01;
  box.cpu_h[17] = 0.1;
  const Vector wrap_input{19.28, -5.46, -36.17, 46.33, -50.7, 61.3};
  positions.copy_from_host(wrap_input.data());
  GPU_Vector<double> wrapped(D);
  error.fill(0);
  rpmd_ja_wrap_positions(N, positions, wrapped, box, error.data());
  Vector wrapped_host(D);
  wrapped.copy_to_host(wrapped_host.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error == 0);
  const Vector expected_wrap{7.28, 6.54, 9.83, 0.33, 9.3, 1.3};
  for (int i = 0; i < D; ++i) check_close(wrapped_host[i], expected_wrap[i]);

  // Non-finite production input must latch the device error instead of succeeding.
  position[0] = std::numeric_limits<double>::quiet_NaN();
  positions.copy_from_host(position.data());
  error.fill(0);
  workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert(device_error != 0);
  error.fill(RPMD_JA_ERROR_BRANCH);
  block_workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert((device_error & RPMD_JA_ERROR_BRANCH) != 0);
  assert((device_error & RPMD_JA_ERROR_NUMERICAL) != 0);

  // The SpMV's fused finite check must latch overflow while sanitizing its row.
  RpmdJAReference overflow_reference = reference;
  overflow_reference.dynamical.values[0] = std::numeric_limits<double>::max();
  RpmdJASparseWorkspace overflow_workspace;
  overflow_workspace.initialize(overflow_reference, masses);
  position.assign(D, 0.0);
  position[0] = 2.0;
  positions.copy_from_host(position.data());
  error.fill(RPMD_JA_ERROR_BRANCH);
  overflow_workspace.compute_correction(positions, ref_positions, velocities, result.data(), error.data());
  error.copy_to_host(&device_error, 1);
  assert((device_error & RPMD_JA_ERROR_BRANCH) != 0);
  assert((device_error & RPMD_JA_ERROR_NUMERICAL) != 0);
}
