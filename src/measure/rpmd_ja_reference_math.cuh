#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rpmd_ja_reference_math
{
inline long double h(const long double x)
{
  if (std::abs(x) < 1.0e-4L) {
    const long double x2 = x * x;
    return 1.0L + x2 / 12.0L - x2 * x2 / 720.0L + x2 * x2 * x2 / 30240.0L;
  }
  return x / (2.0L * std::tanh(x / 2.0L));
}

inline long double log_g(const long double x)
{
  const long double ax = std::abs(x);
  if (ax < 1.0e-4L) {
    const long double x2 = x * x;
    return x2 / 6.0L - x2 * x2 / 180.0L + x2 * x2 * x2 / 2835.0L;
  }
  return ax - std::log(2.0L * ax) + std::log1p(-std::exp(-2.0L * ax));
}

inline long double tail_moment(const int s, const int n)
{
  static const long double bernoulli_over_factorial[] = {
    1.0L / 12.0L, -1.0L / 720.0L, 1.0L / 30240.0L,
    -1.0L / 1209600.0L, 1.0L / 47900160.0L, -691.0L / 1307674368000.0L};
  const long double cscale = 4.0L * std::acos(-1.0L) * std::acos(-1.0L);
  const int p = 2 * s;
  long double sum = 0.0L;
  const int end = n + 64;
  for (int k = n + 1; k <= end; ++k) {
    const long double c = cscale * k * k;
    sum += std::pow(c, -s);
  }
  const long double a = static_cast<long double>(end) + 1.0L;
  long double zeta_tail = std::pow(a, 1 - p) / (p - 1) + 0.5L * std::pow(a, -p);
  long double rising = p;
  for (int k = 0; k < 6; ++k) {
    if (k > 0) {
      rising *= static_cast<long double>(p + 2 * k - 1) * (p + 2 * k);
    }
    zeta_tail += bernoulli_over_factorial[k] * rising * std::pow(a, -p - 2 * k - 1);
  }
  return sum + std::pow(cscale, -s) * zeta_tail;
}

inline void kernel_rt(
  const long double x,
  const long double y,
  long double& r,
  long double& t)
{
  const long double pi = std::acos(-1.0L);
  const long double cscale = 4.0L * pi * pi;
  const long double n_real = std::ceil(std::sqrt(8.0L * (x + y) / cscale));
  if (!std::isfinite(n_real) || n_real > std::numeric_limits<int>::max() - 65)
    throw std::invalid_argument("RPMD-JA dimensionless frequency is too large");
  const int n = std::max(1, static_cast<int>(n_real));
  const long double c_next = cscale * static_cast<long double>(n + 1) * (n + 1);
  const long double rho = (x + y) / c_next;
  const long double m1 = tail_moment(1, n), m2 = tail_moment(2, n);
  const long double tolerance = 2.0e-19L;
  int order = 0;
  long double rho_power = rho;
  while (2.0L * rho_power * m1 / (1.0L - rho) > tolerance || 2.0L * rho_power * m2 / (1.0L - rho) > tolerance) {
    if (++order > 80) throw std::runtime_error("RPMD-JA kernel tail failed to converge");
    rho_power *= rho;
  }
  std::array<long double, 83> moments = {};
  for (int s = 1; s <= order + 2; ++s) moments[s] = tail_moment(s, n);

  r = 0.0L;
  t = 0.0L;
  for (int k = 1; k <= n; ++k) {
    const long double c = cscale * static_cast<long double>(k) * k;
    r += 2.0L * c / ((x + c) * (y + c));
    t += 2.0L / ((x + c) * (y + c));
  }
  for (int k = 0; k <= order; ++k) {
    long double hk = 0.0L;
    for (int m = 0; m <= k; ++m) hk += std::pow(x, m) * std::pow(y, k - m);
    const long double sign = k % 2 == 0 ? 1.0L : -1.0L;
    r += 2.0L * sign * hk * moments[k + 1];
    t += 2.0L * sign * hk * moments[k + 2];
  }
}

inline void make_translation_complement(
  const int n, const std::vector<double>& masses, std::vector<double>& q)
{
  if (n <= 1 || n > std::numeric_limits<int>::max() / 3 || masses.size() != static_cast<size_t>(n))
    throw std::invalid_argument("invalid COM complement dimensions");
  const int d = 3 * n, r = d - 3;
  q.assign(static_cast<size_t>(d) * r, 0.0);
  std::vector<double> translations(static_cast<size_t>(d) * 3, 0.0);
  for (int axis = 0; axis < 3; ++axis) {
    double mass_sum = 0.0;
    for (double mass : masses) {
      if (!(mass > 0.0) || !std::isfinite(mass)) throw std::invalid_argument("invalid COM complement mass");
      mass_sum += mass;
    }
    const double inv_norm = 1.0 / std::sqrt(mass_sum);
    for (int i = 0; i < n; ++i) translations[(axis * n + i) * 3 + axis] = std::sqrt(masses[i]) * inv_norm;
  }
  const std::vector<double> original_translations = translations;
  for (int row = 3; row < d; ++row) q[static_cast<size_t>(row) * r + row - 3] = 1.0;
  std::vector<std::vector<double>> householder(3, std::vector<double>(d, 0.0));
  for (int k = 0; k < 3; ++k) {
    std::vector<double>& v = householder[k];
    double norm2 = 0.0;
    for (int i = k; i < d; ++i) {
      v[i] = translations[static_cast<size_t>(i) * 3 + k];
      norm2 += v[i] * v[i];
    }
    const double norm = std::sqrt(norm2);
    if (!(norm > 0.0)) throw std::runtime_error("could not remove exact translation mode");
    v[k] += std::copysign(norm, v[k] == 0.0 ? 1.0 : v[k]);
    double vnorm2 = 0.0;
    for (int i = k; i < d; ++i) vnorm2 += v[i] * v[i];
    const double inv_vnorm = 1.0 / std::sqrt(vnorm2);
    for (int i = k; i < d; ++i) v[i] *= inv_vnorm;
    for (int col = 0; col < 3; ++col) {
      double dot = 0.0;
      for (int i = k; i < d; ++i) dot += v[i] * translations[static_cast<size_t>(i) * 3 + col];
      for (int i = k; i < d; ++i) translations[static_cast<size_t>(i) * 3 + col] -= 2.0 * v[i] * dot;
    }
  }
  for (int k = 2; k >= 0; --k) {
    const std::vector<double>& v = householder[k];
    for (int col = 0; col < r; ++col) {
      double dot = 0.0;
      for (int i = k; i < d; ++i) dot += v[i] * q[static_cast<size_t>(i) * r + col];
      for (int i = k; i < d; ++i) q[static_cast<size_t>(i) * r + col] -= 2.0 * v[i] * dot;
    }
  }
  for (int axis = 0; axis < 3; ++axis)
    for (int col = 0; col < r; ++col) {
      double overlap = 0.0;
      for (int i = 0; i < d; ++i)
        overlap += original_translations[static_cast<size_t>(i) * 3 + axis] * q[static_cast<size_t>(i) * r + col];
      if (std::abs(overlap) > 1.0e-12) throw std::runtime_error("COM complement is not orthogonal to translation modes");
    }
}

inline void map_site_delta_matrix(
  const int n,
  const std::vector<double>& site,
  const std::vector<double>& omega,
  const double beta_hbar,
  std::vector<double>& delta)
{
  if (n <= 0 || site.size() != static_cast<size_t>(n) * n || omega.size() != static_cast<size_t>(n) ||
      !(beta_hbar > 0.0) || !std::isfinite(beta_hbar))
    throw std::invalid_argument("invalid RPMD-JA modal matrix dimensions");
  delta.resize(site.size());
  for (double w : omega)
    if (!(w > 0.0) || !std::isfinite(w)) throw std::invalid_argument("invalid RPMD-JA frequency");
  for (int a = 0; a < n; ++a) {
    const long double u = static_cast<long double>(beta_hbar) * omega[a];
    for (int b = 0; b < n; ++b) {
      const long double v = static_cast<long double>(beta_hbar) * omega[b];
      const long double x = u * u, y = v * v;
      long double r, t;
      kernel_rt(x, y, r, t);
      long double a_weight = 1.0L;
      if (u > 0.0L && v > 0.0L) {
        const long double splus2 = (u * h(v) + v * h(u)) / (u + v);
        const long double log_sminus = 0.5L * (log_g(0.5L * std::abs(u - v)) - log_g(0.5L * u) - log_g(0.5L * v));
        const long double splus = std::sqrt(splus2);
        const long double sminus = std::exp(log_sminus);
        a_weight = 0.5L * (splus + sminus);
      }
      const long double q = r / (2.0L * a_weight);
      const long double p = (t - q * q) / (a_weight + 1.0L);
      const long double ab = site[static_cast<size_t>(a) * n + b];
      const long double ba = site[static_cast<size_t>(b) * n + a];
      delta[static_cast<size_t>(a) * n + b] = static_cast<double>(x * y * p * ab + x * q * ba);
    }
  }
}

inline void map_site_matrix(
  const int n,
  const std::vector<double>& site,
  const std::vector<double>& omega,
  const double beta_hbar,
  std::vector<double>& mapped)
{
  map_site_delta_matrix(n, site, omega, beta_hbar, mapped);
  for (size_t i = 0; i < mapped.size(); ++i) mapped[i] += site[i];
}
} // namespace rpmd_ja_reference_math
