#include "measure/rpmd_ja.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
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
