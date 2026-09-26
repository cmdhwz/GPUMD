#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace quantum_heat_moments
{
using Complex = std::complex<double>;

inline bool is_canonical_primitive_pimd(
  const bool is_pimd,
  const bool constant_temperature,
  const bool has_pressure_control,
  const bool use_scr_barostat,
  const bool use_eco_pimd)
{
  return is_pimd && constant_temperature && !has_pressure_control &&
    !use_scr_barostat && !use_eco_pimd;
}

inline double temperature_at_step(
  const double start,
  const double end,
  const int step,
  const int number_of_steps)
{
  return start + (end - start) * (static_cast<double>(step) / number_of_steps);
}

inline bool parse_integer(const char* text, int& value)
{
  if (text == nullptr) return false;
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0' ||
      parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) return false;
  value = static_cast<int>(parsed);
  return true;
}

inline bool parse_seed(const char* text, std::uint64_t& value)
{
  if (text == nullptr) return false;
  const char* first = text;
  while (*first == ' ' || *first == '\t' || *first == '\n' || *first == '\r' ||
         *first == '\f' || *first == '\v') ++first;
  if (*first == '-' || *first == '\0') return false;
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0') return false;
  const std::uint64_t converted = static_cast<std::uint64_t>(parsed);
  if (static_cast<unsigned long long>(converted) != parsed) return false;
  value = converted;
  return true;
}

struct ComplexStats
{
  int count = 0;
  Complex mean = 0.0;
  double m2_real = 0.0;
  double m2_imag = 0.0;
  double co_real_imag = 0.0;
  double mean_abs_value = 0.0;
  double max_abs = 0.0;

  bool is_finite_state() const
  {
    return std::isfinite(mean.real()) && std::isfinite(mean.imag()) &&
      std::isfinite(m2_real) && std::isfinite(m2_imag) &&
      std::isfinite(co_real_imag) && std::isfinite(mean_abs_value) && std::isfinite(max_abs);
  }

  bool is_finite() const
  {
    if (!is_finite_state()) return false;
    if (count <= 1) return true;
    const double total_m2 = m2_real + m2_imag;
    return std::isfinite(total_m2) && total_m2 >= 0.0 &&
      std::isfinite(variance()) && std::isfinite(covariance()) && std::isfinite(standard_error());
  }

  void add(const Complex value)
  {
    ++count;
    const double dr = value.real() - mean.real();
    const double di = value.imag() - mean.imag();
    mean += (value - mean) / static_cast<double>(count);
    m2_real += dr * (value.real() - mean.real());
    m2_imag += di * (value.imag() - mean.imag());
    co_real_imag += dr * (value.imag() - mean.imag());
    const double value_abs = std::abs(value);
    mean_abs_value += (value_abs - mean_abs_value) / count;
    max_abs = std::max(max_abs, value_abs);
  }

  bool add_if_finite(const Complex value)
  {
    if (!is_finite()) return false;
    ComplexStats next = *this;
    next.add(value);
    if (!next.is_finite()) return false;
    *this = next;
    return true;
  }

  void merge(const ComplexStats& other)
  {
    if (other.count == 0) return;
    if (count == 0) {
      *this = other;
      return;
    }
    const double total = static_cast<double>(count + other.count);
    const Complex delta = other.mean - mean;
    const double other_weight = other.count / total;
    m2_real += other.m2_real + delta.real() * delta.real() * count * other.count / total;
    m2_imag += other.m2_imag + delta.imag() * delta.imag() * count * other.count / total;
    co_real_imag += other.co_real_imag + delta.real() * delta.imag() * count * other.count / total;
    mean_abs_value += (other.mean_abs_value - mean_abs_value) * other_weight;
    max_abs = std::max(max_abs, other.max_abs);
    mean += delta * other_weight;
    count += other.count;
  }

  double variance_real() const
  {
    return count > 1 ? m2_real / (count - 1) : std::numeric_limits<double>::quiet_NaN();
  }
  double variance_imag() const
  {
    return count > 1 ? m2_imag / (count - 1) : std::numeric_limits<double>::quiet_NaN();
  }
  double covariance() const
  {
    return count > 1 ? co_real_imag / (count - 1) : std::numeric_limits<double>::quiet_NaN();
  }
  double standard_error() const
  {
    return count > 1
      ? std::sqrt((m2_real + m2_imag) / ((count - 1.0) * count))
      : std::numeric_limits<double>::quiet_NaN();
  }
  double standard_error_real() const
  {
    return count > 1 ? std::sqrt(variance_real() / count) : std::numeric_limits<double>::quiet_NaN();
  }
  double standard_error_imag() const
  {
    return count > 1 ? std::sqrt(variance_imag() / count) : std::numeric_limits<double>::quiet_NaN();
  }
  double variance() const { return variance_real() + variance_imag(); }
  double mean_abs() const
  {
    return count > 0 ? mean_abs_value : std::numeric_limits<double>::quiet_NaN();
  }
};

inline bool has_complete_candidate_probe_set(const int finite_probe_count, const int expected_probe_count)
{
  return expected_probe_count > 0 && finite_probe_count == expected_probe_count;
}

inline bool has_complete_candidate_probe_set(const ComplexStats& stats, const int expected_probe_count)
{
  return has_complete_candidate_probe_set(stats.count, expected_probe_count) && stats.is_finite();
}

struct RealStats
{
  int count = 0;
  double mean = 0.0;
  double m2 = 0.0;
  double mean_abs_value = 0.0;
  double max_abs = 0.0;

  bool add(const double value)
  {
    if (!std::isfinite(value) || !std::isfinite(mean) || !std::isfinite(m2) ||
        !std::isfinite(mean_abs_value) || !std::isfinite(max_abs)) return false;
    const double delta = value - mean;
    if (!std::isfinite(delta)) return false;
    const int next_count = count + 1;
    const double next_mean = mean + delta / next_count;
    const double next_m2 = m2 + delta * (value - next_mean);
    const double value_abs = std::fabs(value);
    const double next_mean_abs_value = mean_abs_value + (value_abs - mean_abs_value) / next_count;
    const double next_max_abs = std::max(max_abs, value_abs);
    if (!std::isfinite(next_mean) || !std::isfinite(next_m2) || next_m2 < 0.0 ||
        !std::isfinite(next_mean_abs_value) || !std::isfinite(next_max_abs)) return false;
    count = next_count;
    mean = next_mean;
    m2 = next_m2;
    mean_abs_value = next_mean_abs_value;
    max_abs = next_max_abs;
    return true;
  }

  double variance() const
  {
    return count > 1 ? m2 / (count - 1) : std::numeric_limits<double>::quiet_NaN();
  }
  double standard_error() const
  {
    return count > 1 ? std::sqrt(variance() / count) : std::numeric_limits<double>::quiet_NaN();
  }
  double mean_abs() const
  {
    return count > 0 ? mean_abs_value : std::numeric_limits<double>::quiet_NaN();
  }
};

struct MoyalProbe
{
  std::array<std::vector<double>, 3> direction;
};

struct ArResult
{
  Complex value = 0.0;
  ComplexStats moyal;
};

struct RecursionSettings
{
  double fd_step_r = 0.0;
  double fd_step_p = 0.0;
  double hbar = 0.0;
  int alpha = 0;
  int frame = 0;
  std::uint64_t seed = 0;
};

inline std::uint64_t mix64(std::uint64_t value)
{
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline std::uint64_t static_probe_seed(
  const std::uint64_t base_seed,
  const int frame,
  const int alpha,
  const int bead,
  const std::uint64_t estimator_tag)
{
  const std::uint64_t frame_alpha =
    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(frame)) << 32) |
    static_cast<std::uint32_t>(alpha);
  return mix64(base_seed ^ mix64(frame_alpha) ^
    mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(bead))) ^
    mix64(estimator_tag));
}

inline double uniform_open01(const std::uint64_t key)
{
  return (static_cast<double>(mix64(key) >> 11) + 0.5) / 9007199254740992.0;
}

inline std::vector<MoyalProbe> make_moyal_probes(
  const int count,
  const int dimension,
  const RecursionSettings& settings)
{
  std::vector<MoyalProbe> result(static_cast<size_t>(count));
  for (int k = 0; k < count; ++k) {
    for (int direction = 0; direction < 3; ++direction) {
      auto& values = result[k].direction[direction];
      values.resize(dimension);
      for (int d = 0; d < dimension; ++d) {
        std::uint64_t key = settings.seed ^ (static_cast<std::uint64_t>(settings.frame) << 32);
        key ^= static_cast<std::uint64_t>(settings.alpha + 1) * 0xd6e8feb86659fd93ULL;
        key ^= static_cast<std::uint64_t>(k + 1) * 0xa0761d6478bd642fULL;
        key ^= static_cast<std::uint64_t>(direction + 1) * 0xe7037ed1a0b428dbULL;
        key ^= static_cast<std::uint64_t>(d + 1) * 0x8ebc6af09c88c6e3ULL;
        values[d] = (mix64(key) & 1ULL) ? 1.0 : -1.0;
      }
    }
  }
  return result;
}

inline double normal_from_key(const std::uint64_t key)
{
  const double u1 = uniform_open01(key);
  const double u2 = uniform_open01(key ^ 0x8ebc6af09c88c6e3ULL);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.2831853071795864769 * u2);
}

template <class Scalar>
inline Scalar normalized_symmetrize(const std::vector<Scalar>& permutations)
{
  if (permutations.empty()) throw std::invalid_argument("symmetrization needs at least one permutation");
  Scalar result = 0.0;
  for (const Scalar& value : permutations) result += value;
  return result / static_cast<double>(permutations.size());
}

template <class Scalar>
inline Scalar open_endpoint_pair_average(const std::vector<Scalar>& values)
{
  if (values.empty()) throw std::invalid_argument("open-endpoint estimator needs at least one bead");
  Scalar sum = 0.0;
  Scalar sum_squares = 0.0;
  for (const Scalar& value : values) {
    sum += value;
    sum_squares += value * value;
  }
  const double P = static_cast<double>(values.size());
  return (sum * sum - sum_squares) / (P * P);
}

template <class CoefficientDerivative, class KernelDerivative>
inline Complex weyl_monomial_operator(
  const std::vector<int>& momentum_powers,
  const double hbar,
  CoefficientDerivative&& coefficient_derivative,
  KernelDerivative&& kernel_derivative)
{
  int order = 0;
  for (const int power : momentum_powers) {
    if (power < 0) throw std::invalid_argument("Weyl momentum powers must be nonnegative");
    order += power;
  }
  std::vector<int> coefficient_orders(momentum_powers.size(), 0);
  std::vector<int> kernel_orders(momentum_powers.size(), 0);
  Complex sum = 0.0;
  auto visit = [&](auto&& self, const size_t dimension, double weight) -> void {
    if (dimension == momentum_powers.size()) {
      sum += weight * coefficient_derivative(coefficient_orders) * kernel_derivative(kernel_orders);
      return;
    }
    int binomial = 1;
    for (int nu = 0; nu <= momentum_powers[dimension]; ++nu) {
      coefficient_orders[dimension] = nu;
      kernel_orders[dimension] = momentum_powers[dimension] - nu;
      self(self, dimension + 1, weight * binomial * std::pow(0.5, nu));
      if (nu < momentum_powers[dimension]) {
        binomial = binomial * (momentum_powers[dimension] - nu) / (nu + 1);
      }
    }
  };
  visit(visit, 0, 1.0);
  Complex phase = 1.0;
  for (int k = 0; k < order; ++k) phase *= Complex(0.0, -hbar);
  return phase * sum;
}

struct ClosestImage
{
  std::array<int, 3> image{};
  std::array<double, 3> displacement{};
  double length = 0.0;
};

inline ClosestImage closest_cartesian_lattice_image(
  const std::array<double, 3>& displacement,
  const std::array<double, 9>& h)
{
  const double determinant =
    h[0] * (h[4] * h[8] - h[5] * h[7]) -
    h[1] * (h[3] * h[8] - h[5] * h[6]) +
    h[2] * (h[3] * h[7] - h[4] * h[6]);
  if (!std::isfinite(determinant) || determinant == 0.0)
    throw std::invalid_argument("periodic cell matrix is singular");
  const std::array<double, 9> inverse{
    (h[4] * h[8] - h[5] * h[7]) / determinant,
    (h[2] * h[7] - h[1] * h[8]) / determinant,
    (h[1] * h[5] - h[2] * h[4]) / determinant,
    (h[5] * h[6] - h[3] * h[8]) / determinant,
    (h[0] * h[8] - h[2] * h[6]) / determinant,
    (h[2] * h[3] - h[0] * h[5]) / determinant,
    (h[3] * h[7] - h[4] * h[6]) / determinant,
    (h[1] * h[6] - h[0] * h[7]) / determinant,
    (h[0] * h[4] - h[1] * h[3]) / determinant};
  const std::array<double, 3> fractional{
    inverse[0] * displacement[0] + inverse[1] * displacement[1] + inverse[2] * displacement[2],
    inverse[3] * displacement[0] + inverse[4] * displacement[1] + inverse[5] * displacement[2],
    inverse[6] * displacement[0] + inverse[7] * displacement[1] + inverse[8] * displacement[2]};
  double inverse_frobenius_squared = 0.0;
  for (const double value : inverse) inverse_frobenius_squared += value * value;
  ClosestImage result;
  for (int d = 0; d < 3; ++d) result.image[d] = -static_cast<int>(std::nearbyint(fractional[d]));
  auto cartesian = [&](const std::array<int, 3>& image) {
    return std::array<double, 3>{
      displacement[0] + h[0] * image[0] + h[1] * image[1] + h[2] * image[2],
      displacement[1] + h[3] * image[0] + h[4] * image[1] + h[5] * image[2],
      displacement[2] + h[6] * image[0] + h[7] * image[1] + h[8] * image[2]};
  };
  result.displacement = cartesian(result.image);
  auto squared_length = [](const std::array<double, 3>& x) {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  };
  double best_squared = squared_length(result.displacement);
  const double bound_value = std::ceil(std::sqrt(inverse_frobenius_squared * best_squared)) + 1.0;
  if (!std::isfinite(bound_value) || bound_value > std::numeric_limits<int>::max() / 2)
    throw std::invalid_argument("periodic cell is too ill-conditioned for exact image enumeration");
  const int bound = static_cast<int>(bound_value);
  const int lower[3] = {
    static_cast<int>(std::ceil(-fractional[0] - bound)),
    static_cast<int>(std::ceil(-fractional[1] - bound)),
    static_cast<int>(std::ceil(-fractional[2] - bound))};
  const int upper[3] = {
    static_cast<int>(std::floor(-fractional[0] + bound)),
    static_cast<int>(std::floor(-fractional[1] + bound)),
    static_cast<int>(std::floor(-fractional[2] + bound))};
  for (int nx = lower[0]; nx <= upper[0]; ++nx) {
    for (int ny = lower[1]; ny <= upper[1]; ++ny) {
      for (int nz = lower[2]; nz <= upper[2]; ++nz) {
        const std::array<int, 3> image{nx, ny, nz};
        const std::array<double, 3> candidate = cartesian(image);
        const double candidate_squared = squared_length(candidate);
        if (candidate_squared < best_squared) {
          result.image = image;
          result.displacement = candidate;
          best_squared = candidate_squared;
        }
      }
    }
  }
  result.length = std::sqrt(best_squared);
  return result;
}

inline double shortest_lattice_vector(const std::array<double, 9>& h)
{
  const double determinant =
    h[0] * (h[4] * h[8] - h[5] * h[7]) -
    h[1] * (h[3] * h[8] - h[5] * h[6]) +
    h[2] * (h[3] * h[7] - h[4] * h[6]);
  if (!std::isfinite(determinant) || determinant == 0.0) {
    throw std::invalid_argument("periodic cell matrix is singular");
  }
  const std::array<double, 9> inverse{
    (h[4] * h[8] - h[5] * h[7]) / determinant,
    (h[2] * h[7] - h[1] * h[8]) / determinant,
    (h[1] * h[5] - h[2] * h[4]) / determinant,
    (h[5] * h[6] - h[3] * h[8]) / determinant,
    (h[0] * h[8] - h[2] * h[6]) / determinant,
    (h[2] * h[3] - h[0] * h[5]) / determinant,
    (h[3] * h[7] - h[4] * h[6]) / determinant,
    (h[1] * h[6] - h[0] * h[7]) / determinant,
    (h[0] * h[4] - h[1] * h[3]) / determinant};
  double inverse_frobenius_squared = 0.0;
  for (const double value : inverse) inverse_frobenius_squared += value * value;
  const double inverse_frobenius = std::sqrt(inverse_frobenius_squared);
  double best_squared = std::numeric_limits<double>::infinity();
  for (int column = 0; column < 3; ++column) {
    const double x = h[column];
    const double y = h[3 + column];
    const double z = h[6 + column];
    best_squared = std::min(best_squared, x * x + y * y + z * z);
  }
  const double bound_value = std::ceil(std::sqrt(best_squared) * inverse_frobenius) + 1.0;
  if (!std::isfinite(bound_value) || bound_value > std::numeric_limits<int>::max() / 2) {
    throw std::invalid_argument("periodic cell is too ill-conditioned for exact lattice enumeration");
  }
  const int bound = static_cast<int>(bound_value);
  for (int nx = -bound; nx <= bound; ++nx) {
    for (int ny = -bound; ny <= bound; ++ny) {
      for (int nz = -bound; nz <= bound; ++nz) {
        if (nx == 0 && ny == 0 && nz == 0) continue;
        const double x = h[0] * nx + h[1] * ny + h[2] * nz;
        const double y = h[3] * nx + h[4] * ny + h[5] * nz;
        const double z = h[6] * nx + h[7] * ny + h[8] * nz;
        best_squared = std::min(best_squared, x * x + y * y + z * z);
      }
    }
  }
  return std::sqrt(best_squared);
}

inline int nearest_image_lift(const double fractional_difference)
{
  return -static_cast<int>(std::nearbyint(fractional_difference));
}

inline std::vector<double> make_auxiliary_momenta(
  const int probe,
  const int bead,
  const std::vector<double>& mass_by_dof,
  const double beta,
  const int number_of_beads,
  const RecursionSettings& settings)
{
  std::vector<double> result(mass_by_dof.size());
  for (size_t d = 0; d < mass_by_dof.size(); ++d) {
    std::uint64_t key = settings.seed ^ (static_cast<std::uint64_t>(settings.frame) << 32);
    key ^= static_cast<std::uint64_t>(probe + 1) * 0xa0761d6478bd642fULL;
    key ^= static_cast<std::uint64_t>(bead + 1) * 0xe7037ed1a0b428dbULL;
    key ^= static_cast<std::uint64_t>(d + 1) * 0x8ebc6af09c88c6e3ULL;
    result[d] = normal_from_key(key) *
      std::sqrt(mass_by_dof[d] * number_of_beads / beta);
  }
  return result;
}

template <class Scalar, class Function>
auto mixed_directional_derivative(
  const std::vector<Scalar>& x,
  const std::vector<std::vector<double>>& directions,
  const std::vector<double>& steps,
  Function&& evaluate) -> decltype(evaluate(x))
{
  const size_t order = directions.size();
  if (order > 5 || steps.size() != order) {
    throw std::invalid_argument("mixed directional derivative supports orders 0 through 5");
  }
  if (order == 0) return evaluate(x);
  for (const auto& direction : directions) {
    if (direction.size() != x.size()) {
      throw std::invalid_argument("direction dimension does not match state dimension");
    }
  }
  decltype(evaluate(x)) sum = 0.0;
  double denominator = 1.0;
  const std::uint32_t combinations = 1U << order;
  for (size_t k = 0; k < order; ++k) denominator *= 2.0 * steps[k];
  for (std::uint32_t mask = 0; mask < combinations; ++mask) {
    std::vector<Scalar> displaced = x;
    int weight = 1;
    for (size_t k = 0; k < order; ++k) {
      const double sign = (mask & (1U << k)) ? 1.0 : -1.0;
      weight *= static_cast<int>(sign);
      for (size_t d = 0; d < x.size(); ++d) {
        displaced[d] += Scalar(sign * steps[k] * directions[k][d]);
      }
    }
    sum += static_cast<double>(weight) * evaluate(displaced);
  }
  return sum / denominator;
}

template <class Scalar, class Function>
auto directional_derivative(
  const std::vector<Scalar>& x,
  const std::vector<double>& direction,
  const double step,
  Function&& evaluate) -> decltype(evaluate(x))
{
  return mixed_directional_derivative<Scalar>(x, {direction}, {step}, evaluate);
}

template <class Model>
ArResult evaluate_Ar(
  const int order,
  const std::vector<double>& R,
  const std::vector<Complex>& p,
  const std::vector<double>& mass_by_dof,
  const RecursionSettings& settings,
  const std::vector<MoyalProbe>& probes,
  Model& model)
{
  if (order < 0 || order > 3 || R.size() != p.size() || p.size() != mass_by_dof.size()) {
    throw std::invalid_argument("invalid A_r order or phase-space dimensions");
  }
  if (order == 0) {
    ArResult result;
    result.value = model.evaluate_A0(R, p, settings.alpha, settings.fd_step_r);
    return result;
  }

  auto previous = [&](const std::vector<double>& r, const std::vector<Complex>& momentum) {
    return evaluate_Ar(order - 1, r, momentum, mass_by_dof, settings, probes, model).value;
  };
  const std::vector<double> force = model.force(R);
  std::vector<double> velocity_real(p.size()), velocity_imag(p.size());
  for (size_t d = 0; d < p.size(); ++d) {
    velocity_real[d] = p[d].real() / mass_by_dof[d];
    velocity_imag[d] = p[d].imag() / mass_by_dof[d];
  }

  auto previous_at_R = [&](const std::vector<double>& displaced) { return previous(displaced, p); };
  auto previous_at_p = [&](const std::vector<Complex>& displaced) { return previous(R, displaced); };
  Complex poisson = directional_derivative<double>(R, velocity_real, settings.fd_step_r, previous_at_R);
  poisson += Complex(0.0, 1.0) *
    directional_derivative<double>(R, velocity_imag, settings.fd_step_r, previous_at_R);
  poisson += directional_derivative<Complex>(p, force, settings.fd_step_p, previous_at_p);

  ArResult result;
  result.value = poisson;
  if (order == 1 || order == 3) {
    if (probes.empty()) throw std::invalid_argument("Moyal recursion requires at least one probe");
    for (const auto& probe : probes) {
      const std::vector<std::vector<double>> directions = {
        probe.direction[0], probe.direction[1], probe.direction[2]};
      const std::vector<double> r_steps(3, settings.fd_step_r);
      const std::vector<double> p_steps(3, settings.fd_step_p);
      const double u3 = mixed_directional_derivative<double>(
        R, directions, r_steps, [&](const std::vector<double>& displaced) {
          return model.energy(displaced);
        });
      const Complex fp3 = mixed_directional_derivative<Complex>(
        p, directions, p_steps, [&](const std::vector<Complex>& displaced) {
          return previous(R, displaced);
        });
      result.moyal.add(u3 * fp3);
    }
    result.value += (settings.hbar * settings.hbar / 24.0) * result.moyal.mean;
  }
  return result;
}
} // namespace quantum_heat_moments
