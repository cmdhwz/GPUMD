#include "measure/rpmd_ja_reference_math.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{
void check(const bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

long double direct_g(const long double x)
{
  if (std::abs(x) < 1.0e-4L) {
    const long double x2 = x * x;
    return 1.0L + x2 / 6.0L + x2 * x2 / 120.0L;
  }
  return std::sinh(x) / x;
}

long double direct_s_weight(const long double u, const long double v, const bool sum)
{
  const long double q = 0.5L * (sum ? u + v : u - v);
  const long double squared = direct_g(q) / (direct_g(0.5L * u) * direct_g(0.5L * v));
  return std::sqrt(squared);
}

std::vector<double> direct_fock_map(
  const int n,
  const std::vector<double>& site,
  const std::vector<double>& omega,
  const double beta_hbar)
{
  std::vector<double> mapped(site.size());
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      const long double u = static_cast<long double>(beta_hbar) * omega[a];
      const long double v = static_cast<long double>(beta_hbar) * omega[b];
      const long double sp = direct_s_weight(u, v, true);
      const long double sm = direct_s_weight(u, v, false);
      mapped[static_cast<size_t>(a) * n + b] = static_cast<double>(
        0.5L * (sp + sm) * site[static_cast<size_t>(a) * n + b] +
        0.5L * (sp - sm) * (omega[a] / omega[b]) * site[static_cast<size_t>(b) * n + a]);
    }
  return mapped;
}

std::vector<double> direct_fock_delta(
  const int n,
  const std::vector<double>& site,
  const std::vector<double>& omega,
  const double beta_hbar)
{
  std::vector<double> delta(site.size());
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      const long double u = static_cast<long double>(beta_hbar) * omega[a];
      const long double v = static_cast<long double>(beta_hbar) * omega[b];
      const long double sp = direct_s_weight(u, v, true);
      const long double sm = direct_s_weight(u, v, false);
      const long double diagonal = 0.5L * (sp + sm) - 1.0L;
      const long double transpose = 0.5L * (sp - sm) * omega[a] / omega[b];
      delta[static_cast<size_t>(a) * n + b] = static_cast<double>(
        diagonal * site[static_cast<size_t>(a) * n + b] +
        transpose * site[static_cast<size_t>(b) * n + a]);
    }
  return delta;
}

void check_close(const double actual, const double expected, const double tolerance, const char* message)
{
  const double scale = std::max(1.0, std::abs(expected));
  check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance * scale, message);
}

void check_fock_oracle()
{
  const std::vector<double> site = {1.0, 2.0, -3.0, 4.0};
  std::vector<double> production;
  const std::vector<std::vector<double>> cases = {
    {0.3, 0.9},                         // Unequal frequencies.
    {1.0, 1.0 + 1.0e-12},              // Near degeneracy.
    {1.0e-10, 2.0e-10},                // Small frequencies.
    {5.0, 5.0 + 1.0e-8}};              // Low temperature and near degeneracy.
  const std::vector<double> betas = {0.7, 0.7, 0.7, 20.0};
  for (size_t k = 0; k < cases.size(); ++k) {
    rpmd_ja_reference_math::map_site_matrix(2, site, cases[k], betas[k], production);
    const std::vector<double> oracle = direct_fock_map(2, site, cases[k], betas[k]);
    for (size_t i = 0; i < production.size(); ++i)
      check_close(production[i], oracle[i], 2.0e-12, "production modal map disagrees with direct Fock kernel");
  }

  std::vector<double> delta, delta_oracle = direct_fock_delta(2, site, cases[0], betas[0]);
  rpmd_ja_reference_math::map_site_delta_matrix(2, site, cases[0], betas[0], delta);
  for (size_t i = 0; i < delta.size(); ++i)
    check_close(delta[i], delta_oracle[i], 2.0e-12, "factorized correction disagrees with direct Fock kernel");

  const std::vector<std::vector<double>> acoustic_cases = {{10.0, 1.0e-14}, {1.0e-14, 10.0}};
  for (size_t k = 0; k < acoustic_cases.size(); ++k) {
    const std::vector<double>& omega = acoustic_cases[k];
    std::vector<double> actual;
    rpmd_ja_reference_math::map_site_delta_matrix(2, site, omega, 1.0, actual);
    const long double high = 10.0L / (2.0L * std::tanh(5.0L));
    const long double q_axis = (high - 1.0L) / 200.0L;
    const double expected = k == 0 ? -3.0 * 100.0 * static_cast<double>(q_axis) :
      2.0 * 1.0e-28 * static_cast<double>(q_axis);
    const double actual_transpose = k == 0 ? actual[1] : actual[2];
    const double abs_error = std::abs(actual_transpose - expected);
    const double rel_error = abs_error / std::max(std::abs(expected), 1.0e-30);
    check(std::isfinite(actual_transpose) && (abs_error < 1.0e-28 || rel_error < 2.0e-10),
      "factorized acoustic correction disagrees with the independent zero-axis Fock limit");
  }
}

void check_complete_large_b_kernel()
{
  const std::vector<double> site = {1.0, 2.0, -3.0, 4.0};
  std::vector<double> mapped;
  rpmd_ja_reference_math::map_site_matrix(2, site, {1000.0, 1000.0}, 2.0, mapped);
  for (double value : mapped) check(std::isfinite(value), "large-b modal map is non-finite");
  check_close(mapped[0], 0.5 * std::sqrt(1000.0), 1.0e-12, "large-b diagonal kernel limit is incorrect");
  check_close(mapped[1], -0.5 * std::sqrt(1000.0), 1.0e-12, "large-b off-diagonal kernel limit is incorrect");
}

void check_nonsymmetric_row_major_layout()
{
  const int n = 2;
  const std::vector<double> h = {1.0, 2.0, -3.0, 4.0};
  const double c = std::cos(0.37), s = std::sin(0.37);
  const double e[2][2] = {{c, -s}, {s, c}}; // E stored row-major.
  double modal[2][2] = {};
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          modal[a][b] += e[i][a] * h[static_cast<size_t>(i) * n + j] * e[j][b];
  check(std::abs(modal[0][1] - modal[1][0]) > 1.0e-2, "row-major oracle lost nonsymmetric site flow");
  const std::vector<double> modal_row_major = {modal[0][0], modal[0][1], modal[1][0], modal[1][1]};
  std::vector<double> production;
  rpmd_ja_reference_math::map_site_delta_matrix(n, modal_row_major, {0.7, 1.3}, 0.9, production);
  const std::vector<double> oracle = direct_fock_delta(n, modal_row_major, {0.7, 1.3}, 0.9);
  for (size_t i = 0; i < production.size(); ++i)
    check_close(production[i], oracle[i], 2.0e-12, "row-major modal map changed matrix orientation");
}

void check_translation_complement()
{
  for (const std::vector<double>& masses : {
         std::vector<double>{1.1, 1.7}, std::vector<double>{1.0, 2.0, 4.0}}) {
    const int n = static_cast<int>(masses.size()), d = 3 * n, r = d - 3;
    std::vector<double> q;
    rpmd_ja_reference_math::make_translation_complement(n, masses, q);
    double mass_sum = 0.0;
    for (double mass : masses) mass_sum += mass;
    for (int axis = 0; axis < 3; ++axis)
      for (int col = 0; col < r; ++col) {
        double overlap = 0.0;
        for (int i = 0; i < n; ++i)
          overlap += std::sqrt(masses[i] / mass_sum) * q[static_cast<size_t>(axis * n + i) * r + col];
        check(std::abs(overlap) < 1.0e-12, "COM complement overlaps a translation");
      }
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < r; ++j) {
        double dot = 0.0;
        for (int k = 0; k < d; ++k) dot += q[static_cast<size_t>(k) * r + i] * q[static_cast<size_t>(k) * r + j];
        check(std::abs(dot - (i == j ? 1.0 : 0.0)) < 1.0e-12, "COM complement is not orthonormal");
      }
  }
}

void check_finite_difference_site_hessian()
{
  // Fixed-reference edge flow uses d0 as its weight, not the current d.
  const double d0 = 0.7, step = 1.0e-4;
  const auto gradient = [](const double d) { return d * d * d + 4.0 * d; };
  const auto flow = [&](const double d) { return -d0 * gradient(d); };
  const double numerical = (flow(d0 + step) - flow(d0 - step)) / (2.0 * step);
  const double exact = -d0 * (3.0 * d0 * d0 + 4.0);
  check_close(numerical, exact, 1.0e-8, "fixed-reference site-flow derivative failed");

  const auto current_virial = [&](const double d) { return -d * gradient(d); };
  const double current_derivative = (current_virial(d0 + step) - current_virial(d0 - step)) / (2.0 * step);
  const double geometry = -gradient(d0);
  check_close(current_derivative - geometry, exact, 1.0e-8,
    "current-virial geometric term was not removed to recover fixed-edge site flow");
}

void check_cross_edges_and_duplicate_images()
{
  const double d0[2] = {0.7, -0.4};
  const double k[2][2] = {{3.0, 1.2}, {1.2, 5.0}}; // Includes the cross-neighbor Hessian block.
  const double step = 1.0e-5;
  const auto gradients = [&](const double x_center, const double x_neighbor) {
    const double d[2] = {d0[0] + x_neighbor - x_center, d0[1] + x_neighbor - x_center};
    return std::array<double, 2>{{k[0][0] * d[0] + k[0][1] * d[1],
      k[1][0] * d[0] + k[1][1] * d[1]}};
  };
  const auto fixed_flow = [&](const double x_center, const double x_neighbor) {
    const std::array<double, 2> g = gradients(x_center, x_neighbor);
    return -(d0[0] * g[0] + d0[1] * g[1]); // Both image edges share one velocity atom.
  };
  for (int input = 0; input < 2; ++input) {
    const double plus = input == 0 ? fixed_flow(step, 0.0) : fixed_flow(0.0, step);
    const double minus = input == 0 ? fixed_flow(-step, 0.0) : fixed_flow(0.0, -step);
    const double numerical = (plus - minus) / (2.0 * step);
    const double sign = input == 0 ? -1.0 : 1.0;
    const double exact = sign * (d0[0] * (k[0][0] + k[0][1]) + d0[1] * (k[1][0] + k[1][1]));
    check_close(numerical, exact, 1.0e-10,
      "cross-edge or repeated-image contribution/sign failed the analytic site-flow oracle");
  }
}
} // namespace

int main()
{
  check_fock_oracle();
  check_complete_large_b_kernel();
  check_nonsymmetric_row_major_layout();
  check_translation_complement();
  check_finite_difference_site_hessian();
  check_cross_edges_and_duplicate_images();
  std::puts("rpmd_ja_reference_math: all CPU checks passed");
}
