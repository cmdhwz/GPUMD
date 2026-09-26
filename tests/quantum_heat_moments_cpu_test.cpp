#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#define __host__
#define __device__
#include "model/box.cuh"
#undef __host__
#undef __device__
#include "measure/quantum_heat_moments_math.cuh"

using namespace quantum_heat_moments;

struct Oscillator
{
  std::vector<double> masses;
  double omega = 0.0;
  double lambda = 0.0;

  double energy(const std::vector<double>& r)
  {
    return 0.5 * omega * omega * r[0] * r[0] + lambda * std::pow(r[0], 4);
  }
  std::vector<double> force(const std::vector<double>& r)
  {
    return {-omega * omega * r[0] - 4.0 * lambda * std::pow(r[0], 3)};
  }
  Complex evaluate_A0(
    const std::vector<double>& r,
    const std::vector<Complex>& p,
    int,
    double)
  {
    const double mass = masses[0];
    return p[0] * p[0] * p[0] / (2.0 * mass * mass) + energy(r) * p[0] / mass;
  }
};

static int failures = 0;
static double max_absolute_error = 0.0;
static double max_relative_error = 0.0;

void check(const char* name, const double actual, const double expected, const double tolerance)
{
  const double absolute = std::fabs(actual - expected);
  const double relative = absolute / std::max(std::fabs(expected), 1.0e-30);
  max_absolute_error = std::max(max_absolute_error, absolute);
  max_relative_error = std::max(max_relative_error, relative);
  const bool pass = std::isfinite(actual) && absolute <= tolerance;
  std::printf("%s %s abs=%.6e rel=%.6e\n", pass ? "PASS" : "FAIL", name, absolute, relative);
  if (!pass) ++failures;
}

std::vector<MoyalProbe> probes(const RecursionSettings& settings, const int count, const int dimension)
{
  return make_moyal_probes(count, dimension, settings);
}

Complex run_ar(Oscillator& model, const int order, const double x, const double p,
  RecursionSettings settings, const std::vector<MoyalProbe>& probe_set)
{
  return evaluate_Ar(order, std::vector<double>{x}, std::vector<Complex>{p},
    model.masses, settings, probe_set, model).value;
}

void test_free_particle()
{
  Oscillator model{{1.7}, 0.0, 0.0};
  RecursionSettings settings{1.0e-3, 1.0e-3, 0.8, 0, 1, 42};
  const auto probe_set = probes(settings, 8, 1);
  const double p = 0.43;
  const double a0 = p * p * p / (2.0 * model.masses[0] * model.masses[0]);
  check("free A0 cubic kinetic", run_ar(model, 0, 0.8, p, settings, probe_set).real(), a0, 1.0e-14);
  for (int order = 1; order <= 3; ++order) {
    check((std::string("free A") + std::to_string(order)).c_str(),
      std::abs(run_ar(model, order, 0.8, p, settings, probe_set)), 0.0, 1.0e-12);
  }
}

void test_harmonic_recursion()
{
  Oscillator model{{1.0}, 1.0, 0.0};
  const double x = 0.7, p = 0.3, w2 = model.omega * model.omega;
  RecursionSettings h{2.0e-3, 2.0e-3, 1.0, 0, 7, 123};
  RecursionSettings h2 = h;
  h2.fd_step_r *= 0.5;
  h2.fd_step_p *= 0.5;
  const auto probe_set = probes(h, 16, 1);
  check("oscillator model A0 independent of FD step",
    run_ar(model, 0, x, p, h2, probe_set).real(),
    run_ar(model, 0, x, p, h, probe_set).real(), 0.0);
  const double a1 = -0.5 * w2 * x * p * p - 0.5 * w2 * w2 * x * x * x;
  const double a2 = -0.5 * w2 * p * p * p - 0.5 * w2 * w2 * x * x * p;
  const double a3 = 0.5 * w2 * w2 * x * p * p + 0.5 * w2 * w2 * w2 * x * x * x;
  check("harmonic A1", run_ar(model, 1, x, p, h, probe_set).real(), a1, 2.0e-8);
  check("harmonic A2", run_ar(model, 2, x, p, h, probe_set).real(), a2, 2.0e-7);
  check("harmonic A3", run_ar(model, 3, x, p, h, probe_set).real(), a3, 2.0e-6);
  check("harmonic A2=-A0", run_ar(model, 2, x, p, h, probe_set).real(),
    -run_ar(model, 0, x, p, h, probe_set).real(), 2.0e-7);
  check("harmonic A3=-A1", run_ar(model, 3, x, p, h, probe_set).real(),
    -run_ar(model, 1, x, p, h, probe_set).real(), 2.0e-6);
  for (int order = 0; order <= 3; ++order) {
    const double positive = run_ar(model, order, x, p, h, probe_set).real();
    const double negative = run_ar(model, order, x, -p, h, probe_set).real();
    const double parity = (order % 2 == 0) ? -1.0 : 1.0;
    check((std::string("A") + std::to_string(order) + " momentum parity").c_str(),
      negative, parity * positive, 2.0e-6);
  }
  const ArResult harmonic_a1 = evaluate_Ar(1, {x}, {p}, model.masses, h, probe_set, model);
  check("harmonic U''' Moyal term", harmonic_a1.moyal.mean.real(), 0.0, 1.0e-9);
  const double h_value = run_ar(model, 3, x, p, h, probe_set).real();
  const double h2_value = run_ar(model, 3, x, p, h2, probe_set).real();
  check("harmonic h to h/2 convergence", h2_value, h_value, 2.0e-6);
}

void test_quartic_moyal()
{
  Oscillator model{{1.0}, 1.2, 0.08};
  const double x = 0.61, p = 0.37, hbar = 0.72;
  RecursionSettings settings{2.0e-3, 2.0e-3, hbar, 0, 3, 456};
  const auto probe_set = probes(settings, 32, 1);
  const ArResult a1 = evaluate_Ar(1, {x}, {p}, model.masses, settings, probe_set, model);
  const double expected = 3.0 * hbar * hbar * model.lambda * x /
    (model.masses[0] * model.masses[0]);
  check("quartic hbar2 correction", hbar * hbar * a1.moyal.mean.real() / 24.0, expected, 2.0e-8);
}

double periodic_pair_current(const Box& box, double x0, double x1, double v0, double v1)
{
  double dx = x0 - x1, dy = 0.0, dz = 0.0;
  apply_mic(box, dx, dy, dz);
  const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double dr = r - 1.0;
  const double potential = 0.5 * dr * dr;
  const double derivative = dr * (dx / r);
  const double ui = 0.5 * potential;
  const double convective = ui * (v0 + v1);
  const double transport = -0.5 * dx * derivative * (v0 + v1);
  return convective + transport;
}

void test_periodic_translation()
{
  Box box;
  box.is_orthogonal = true;
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10.0;
  box.cpu_h[9] = box.cpu_h[13] = box.cpu_h[17] = 0.1;
  box.cpu_h[10] = box.cpu_h[11] = box.cpu_h[12] = box.cpu_h[14] = box.cpu_h[15] = box.cpu_h[16] = 0.0;
  const double a = periodic_pair_current(box, 0.2, 9.7, 0.1, -0.2);
  const double shifted = periodic_pair_current(box, 10.2, 9.7, 0.1, -0.2);
  check("periodic lattice translation current", shifted, a, 1.0e-14);
}

void test_open_endpoint_no_conjugate()
{
  const std::vector<Complex> gamma{Complex(0.0, 1.0), Complex(0.0, 2.0)};
  check("open endpoint uses Gamma_s Gamma_t", open_endpoint_pair_average(gamma).real(), -1.0, 1.0e-14);
  const Complex conjugated =
    (std::conj(gamma[0]) * gamma[1] + std::conj(gamma[1]) * gamma[0]) / 4.0;
  check("no-conjugate regression", std::abs(open_endpoint_pair_average(gamma) - conjugated), 2.0, 1.0e-14);
  check("normalized symmetrization", normalized_symmetrize(std::vector<double>{1, 2, 3, 4, 5, 6}),
    3.5, 1.0e-14);
  check("real open endpoint excludes equal links",
    open_endpoint_pair_average(std::vector<double>{1.0, 2.0, 3.0}), 22.0 / 9.0, 1.0e-14);
  const std::vector<Complex> gamma0{Complex(0.0, 1.0), Complex(0.0, 2.0)};
  const std::vector<Complex> gamma2{Complex(0.0, 3.0), Complex(0.0, 4.0)};
  check("Gamma0 open endpoint does not conjugate", open_endpoint_pair_average(gamma0).real(), -1.0, 1.0e-14);
  check("Gamma2 open endpoint does not conjugate", open_endpoint_pair_average(gamma2).real(), -3.0, 1.0e-14);
  ComplexStats stats;
  stats.add_if_finite(Complex(1.0, 2.0));
  stats.add_if_finite(Complex(3.0, 4.0));
  check("complex statistics real stderr", stats.standard_error_real(), 1.0, 1.0e-14);
  check("complex statistics imaginary stderr", stats.standard_error_imag(), 1.0, 1.0e-14);
}

void test_bead_specific_static_probe_seeds()
{
  const std::uint64_t base = 1234567;
  const std::uint64_t first = static_probe_seed(base, 9, 1, 0, 0x6d75325f74726163ULL);
  check("static probe seed is deterministic",
    first == static_probe_seed(base, 9, 1, 0, 0x6d75325f74726163ULL) ? 1.0 : 0.0, 1.0, 0.0);
  check("static probe seed differs by bead",
    first != static_probe_seed(base, 9, 1, 1, 0x6d75325f74726163ULL) ? 1.0 : 0.0, 1.0, 0.0);
  check("static probe seed differs by estimator",
    first != static_probe_seed(base, 9, 1, 0, 0x6d75345f74726163ULL) ? 1.0 : 0.0, 1.0, 0.0);
  check("static probe seed differs by frame and direction",
    (first != static_probe_seed(base, 10, 1, 0, 0x6d75325f74726163ULL) &&
      first != static_probe_seed(base, 9, 2, 0, 0x6d75325f74726163ULL)) ? 1.0 : 0.0, 1.0, 0.0);
  RecursionSettings same_bead_h{1.0e-4, 1.0e-4, 1.0, 1, 9, first};
  RecursionSettings same_bead_h2 = same_bead_h;
  same_bead_h2.fd_step_r *= 0.5;
  same_bead_h2.fd_step_p *= 0.5;
  RecursionSettings other_bead = same_bead_h;
  other_bead.seed = static_probe_seed(base, 9, 1, 1, 0x6d75325f74726163ULL);
  const auto probes_h = make_moyal_probes(8, 64, same_bead_h);
  const auto probes_h2 = make_moyal_probes(8, 64, same_bead_h2);
  const auto probes_other_bead = make_moyal_probes(8, 64, other_bead);
  const auto same_probe_set = [](const std::vector<MoyalProbe>& left, const std::vector<MoyalProbe>& right) {
    if (left.size() != right.size()) return false;
    for (size_t k = 0; k < left.size(); ++k)
      for (int direction = 0; direction < 3; ++direction)
        if (left[k].direction[direction] != right[k].direction[direction]) return false;
    return true;
  };
  check("same-bead h and h2 retain paired probes",
    same_probe_set(probes_h, probes_h2) ? 1.0 : 0.0, 1.0, 0.0);
  check("beads use independent probe realizations",
    !same_probe_set(probes_h, probes_other_bead) ? 1.0 : 0.0, 1.0, 0.0);

  const int P = 4;
  const double expected_independent = static_cast<double>(P * (P - 1)) / (P * P);
  for (const int probes : {16, 64, 256}) {
    double shared_mean = 0.0, independent_mean = 0.0;
    for (int probe = 0; probe < probes; ++probe) {
      const double shared_error = (probe & 1) == 0 ? 1.0 : -1.0;
      std::vector<double> shared(P), independent(P);
      for (int bead = 0; bead < P; ++bead) {
        shared[bead] = 1.0 + shared_error;
        const double independent_error = ((probe >> bead) & 1) == 0 ? 1.0 : -1.0;
        independent[bead] = 1.0 + independent_error;
      }
      shared_mean += open_endpoint_pair_average(shared) / probes;
      independent_mean += open_endpoint_pair_average(independent) / probes;
    }
    check((std::string("independent-bead probe contraction P=") +
      std::to_string(probes).c_str()).c_str(), independent_mean, expected_independent, 1.0e-14);
    check((std::string("shared-probe cross-term remains biased P=") +
      std::to_string(probes).c_str()).c_str(), shared_mean, 2.0 * expected_independent, 1.0e-14);
  }
}

void test_weyl_reference_coefficients()
{
  const double hbar = 2.0;
  auto coefficient = [](const int derivative_order) {
    return [derivative_order](const std::vector<int>& orders) {
      return orders[0] == derivative_order ? 1.0 : 0.0;
    };
  };
  auto kernel = [](const std::vector<int>& orders) {
    return Complex(std::pow(2.0, orders[0]), 0.0);
  };
  const Complex phase_p3(0.0, hbar * hbar * hbar);
  check("Weyl p^3 coefficient 1", std::abs(
    weyl_monomial_operator({3}, hbar, coefficient(0), kernel) - phase_p3 * 8.0), 0.0, 1.0e-14);
  check("Weyl p^3 coefficient 3/2", std::abs(
    weyl_monomial_operator({3}, hbar, coefficient(1), kernel) - phase_p3 * 6.0), 0.0, 1.0e-14);
  check("Weyl p^3 coefficient 3/4", std::abs(
    weyl_monomial_operator({3}, hbar, coefficient(2), kernel) - phase_p3 * 1.5), 0.0, 1.0e-14);
  check("Weyl p^3 coefficient 1/8", std::abs(
    weyl_monomial_operator({3}, hbar, coefficient(3), kernel) - phase_p3 * 0.125), 0.0, 1.0e-14);
  const Complex phase_p1(0.0, -hbar);
  check("Weyl p coefficient 1", std::abs(
    weyl_monomial_operator({1}, hbar, coefficient(0), kernel) - phase_p1 * 2.0), 0.0, 1.0e-14);
  check("Weyl p coefficient 1/2", std::abs(
    weyl_monomial_operator({1}, hbar, coefficient(1), kernel) - phase_p1), 0.0, 1.0e-14);
  const std::array<std::array<int, 2>, 6> mixed_orders{{
    {{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}, {{2, 0}}, {{2, 1}}}};
  const std::array<double, 6> mixed_weights{{1.0, 0.5, 1.0, 0.5, 0.25, 0.125}};
  const auto unit_kernel = [](const std::vector<int>&) { return Complex(1.0, 0.0); };
  for (size_t i = 0; i < mixed_orders.size(); ++i) {
    const auto selected_coefficient = [target = mixed_orders[i]](const std::vector<int>& orders) {
      return orders[0] == target[0] && orders[1] == target[1] ? 1.0 : 0.0;
    };
    check((std::string("mixed-index Weyl weight ") + std::to_string(i).c_str()).c_str(),
      std::abs(weyl_monomial_operator({2, 1}, 1.0, selected_coefficient, unit_kernel) -
        Complex(0.0, mixed_weights[i])), 0.0, 1.0e-14);
  }
}

void test_closest_cartesian_lattice_image()
{
  const std::array<double, 9> h{1.0, 0.9, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 2.0};
  const ClosestImage closest = closest_cartesian_lattice_image({0.931, 0.098, 0.0}, h);
  const std::array<int, 3> fractional_image{
    -static_cast<int>(std::nearbyint(0.931 - 0.9 * (0.098 / 0.2))),
    -static_cast<int>(std::nearbyint(0.098 / 0.2)), 0};
  const double fractional_rounding_length = std::sqrt(0.931 * 0.931 + 0.098 * 0.098);
  check("triclinic Cartesian closest image is shorter than fractional rounding",
    closest.length < fractional_rounding_length ? 1.0 : 0.0, 1.0, 0.0);
  check("triclinic Cartesian closest image selects skew lattice vector",
    closest.image[1], -1.0, 0.0);
  check("triclinic link diagnostic detects fractional/Cartesian image mismatch",
    closest.image == fractional_image ? 0.0 : 1.0, 1.0, 0.0);
}

void test_shortest_triclinic_lattice_vector()
{
  const std::array<double, 9> h{1.0, 0.9, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 2.0};
  check("triclinic shortest lattice vector", shortest_lattice_vector(h), std::sqrt(0.05), 1.0e-12);
}

using HarmonicPolynomialTerm = std::pair<double, std::array<int, 2>>;

std::vector<double> harmonic_ring_covariance(const int P)
{
  std::vector<double> covariance(P);
  for (int lag = 0; lag < P; ++lag) {
    for (int k = 0; k < P; ++k) {
      const double lambda = 4.0 * P * std::pow(std::sin(3.14159265358979323846 * k / P), 2) + 1.0 / P;
      covariance[lag] += std::cos(6.2831853071795864769 * k * lag / P) / (P * lambda);
    }
  }
  return covariance;
}

double gaussian_moment(const std::vector<int>& variables, const std::array<std::array<double, 4>, 4>& covariance)
{
  if (variables.empty()) return 1.0;
  if (variables.size() % 2 != 0) return 0.0;
  const int first = variables.front();
  double result = 0.0;
  for (size_t j = 1; j < variables.size(); ++j) {
    std::vector<int> remaining;
    remaining.reserve(variables.size() - 2);
    for (size_t k = 1; k < variables.size(); ++k) if (k != j) remaining.push_back(variables[k]);
    result += covariance[first][variables[j]] * gaussian_moment(remaining, covariance);
  }
  return result;
}

double harmonic_pair_expectation(
  const std::vector<HarmonicPolynomialTerm>& polynomial,
  const int P,
  const int lag,
  const std::vector<double>& C)
{
  const auto periodic = [P](int value) { return (value % P + P) % P; };
  const auto cov_q = [&](const int left, const int right) { return C[periodic(right - left)]; };
  const std::array<std::vector<std::pair<int, double>>, 4> forms{{
    {{0, 1.0}}, {{1, 1.0}, {0, -1.0}}, {{lag, 1.0}}, {{lag + 1, 1.0}, {lag, -1.0}}}};
  std::array<std::array<double, 4>, 4> covariance{};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (const auto& a : forms[i]) {
        for (const auto& b : forms[j]) {
          covariance[i][j] += a.second * b.second * cov_q(a.first, b.first);
        }
      }
    }
  }
  double result = 0.0;
  for (const auto& left : polynomial) {
    for (const auto& right : polynomial) {
      std::vector<int> variables;
      for (int n = 0; n < left.second[0]; ++n) variables.push_back(0);
      for (int n = 0; n < left.second[1]; ++n) variables.push_back(1);
      for (int n = 0; n < right.second[0]; ++n) variables.push_back(2);
      for (int n = 0; n < right.second[1]; ++n) variables.push_back(3);
      result += left.first * right.first * gaussian_moment(variables, covariance);
    }
  }
  return result;
}

std::vector<HarmonicPolynomialTerm> harmonic_gamma1_polynomial(const double epsilon)
{
  return {{0.5 / (epsilon * epsilon), {1, 2}}, {-0.5, {2, 1}},
    {epsilon * epsilon / 8.0 - 0.5, {3, 0}},
    {0.5 / epsilon, {0, 1}}, {-0.5 / epsilon - 0.5 * epsilon, {1, 0}}};
}

std::vector<HarmonicPolynomialTerm> harmonic_gamma0_imaginary_polynomial(const double epsilon)
{
  const double H = -1.0 / epsilon - 0.5 * epsilon;
  return {{0.5 / (epsilon * epsilon * epsilon), {0, 3}}, {-0.75 / epsilon, {1, 2}},
    {0.375 * epsilon - 0.5 / epsilon, {2, 1}},
    {0.25 * epsilon - epsilon * epsilon * epsilon / 16.0, {3, 0}},
    {-0.5 - 0.75 * epsilon * H, {1, 0}}, {1.5 * H / epsilon, {0, 1}}};
}

std::vector<HarmonicPolynomialTerm> harmonic_gamma2_imaginary_polynomial(const double epsilon)
{
  // Direct harmonic Gamma2 expansion; do not reuse Gamma0 samples in the mu4 tests.
  const double H = -1.0 / epsilon - 0.5 * epsilon;
  return {{-0.5 / (epsilon * epsilon * epsilon), {0, 3}}, {0.75 / epsilon, {1, 2}},
    {0.5 / epsilon - 0.375 * epsilon, {2, 1}},
    {epsilon * epsilon * epsilon / 16.0 - 0.25 * epsilon, {3, 0}},
    {0.5 + 0.75 * epsilon * H, {1, 0}}, {-1.5 * H / epsilon, {0, 1}}};
}

double harmonic_finite_p_open_moment(const int P, const int order)
{
  const double epsilon = 1.0 / P;
  const auto C = harmonic_ring_covariance(P);
  const auto polynomial = order == 0 ? harmonic_gamma0_imaginary_polynomial(epsilon) :
    order == 2 ? harmonic_gamma1_polynomial(epsilon) : harmonic_gamma2_imaginary_polynomial(epsilon);
  double pair_sum = 0.0;
  for (int lag = 1; lag < P; ++lag)
    pair_sum += harmonic_pair_expectation(polynomial, P, lag, C);
  return (order == 2 ? 1.0 : -1.0) * pair_sum / P;
}

void test_harmonic_finite_p_moments()
{
  const std::array<int, 3> P{16, 64, 256};
  const std::array<double, 3> mu2_reference{6.081863240, 6.413673096, 6.496463414};
  const double mu_exact = 6.524041565;
  const double f_zero = 7.058861008;
  for (size_t i = 0; i < P.size(); ++i) {
    const double mu2 = harmonic_finite_p_open_moment(P[i], 2);
    check((std::string("harmonic finite-P mu2 P=") + std::to_string(P[i]).c_str()).c_str(),
      mu2, mu2_reference[i], 2.0e-8);
    check((std::string("harmonic P times mu2 bias P=") + std::to_string(P[i]).c_str()).c_str(),
      P[i] * (mu2 - mu_exact), -f_zero, i == 0 ? 0.03 : 0.01);
  }
  const double mu0_256 = harmonic_finite_p_open_moment(256, 0);
  const double mu4_256 = harmonic_finite_p_open_moment(256, 4);
  check("harmonic mu0 finite-P convergence", mu0_256, mu_exact, 0.1);
  check("harmonic mu4 finite-P convergence", mu4_256, mu_exact, 0.1);
}

void test_harmonic_gamma_structure()
{
  const double epsilon = 0.125, x = 0.37, delta = -0.19;
  const double H = -1.0 / epsilon - 0.5 * epsilon;
  const auto gamma0_imag = [&](const double q, const double dq) {
    const double g = dq / epsilon - 0.5 * epsilon * q;
    return -0.5 * q * q * g - 0.5 * q + 0.5 * g * g * g + 1.5 * H * g;
  };
  const auto gamma1_real = [&](const double q, const double dq) {
    const double g = dq / epsilon - 0.5 * epsilon * q;
    return -0.5 * q * q * q + 0.5 * q * g * g - q / (2.0 * epsilon) -
      0.25 * epsilon * q + 0.5 * g;
  };
  const Complex gamma0(0.0, gamma0_imag(x, delta));
  const double gamma1 = gamma1_real(x, delta);
  const double g = delta / epsilon - 0.5 * epsilon * x;
  const double gamma2_imag = 0.5 * x * x * g + 0.5 * x - 0.5 * g * g * g - 1.5 * H * g;
  const Complex gamma2(0.0, gamma2_imag);
  check("harmonic Gamma0 is imaginary", gamma0.real(), 0.0, 0.0);
  check("harmonic Gamma1 is real", std::imag(Complex(gamma1, 0.0)), 0.0, 0.0);
  check("harmonic Gamma2 is imaginary", gamma2.real(), 0.0, 0.0);
  check("harmonic Gamma0 coordinate-link parity", gamma0_imag(-x, -delta), -gamma0.imag(), 1.0e-14);
  check("harmonic Gamma1 coordinate-link parity", gamma1_real(-x, -delta), -gamma1, 1.0e-14);
  check("harmonic Gamma2=-Gamma0", gamma2.imag(), -gamma0.imag(), 0.0);
}

std::vector<double> sample_harmonic_ring_path(const int P, const int frame)
{
  const std::vector<double> C = harmonic_ring_covariance(P);
  std::vector<double> lower(static_cast<size_t>(P) * P, 0.0);
  for (int i = 0; i < P; ++i) {
    for (int j = 0; j <= i; ++j) {
      double value = C[(j - i + P) % P];
      for (int k = 0; k < j; ++k) value -= lower[static_cast<size_t>(i) * P + k] * lower[static_cast<size_t>(j) * P + k];
      lower[static_cast<size_t>(i) * P + j] = i == j ? std::sqrt(value) : value / lower[static_cast<size_t>(j) * P + j];
    }
  }
  std::vector<double> normal(P), position(P);
  for (int j = 0; j < P; ++j) {
    const std::uint64_t key = mix64(0x243f6a8885a308d3ULL ^
      (static_cast<std::uint64_t>(P) << 32) ^ (static_cast<std::uint64_t>(frame + 1) << 16) ^ j);
    normal[j] = normal_from_key(key);
  }
  for (int i = 0; i < P; ++i)
    for (int j = 0; j <= i; ++j) position[i] += lower[static_cast<size_t>(i) * P + j] * normal[j];
  return position;
}

void test_harmonic_estimator_variance_vs_p()
{
  const std::array<int, 5> bead_counts{8, 16, 32, 64, 128};
  const int frames = 512;
  for (const int P : bead_counts) {
    RealStats mu0_samples, mu2_samples, mu4_samples;
    const double epsilon = 1.0 / P;
    for (int frame = 0; frame < frames; ++frame) {
      const std::vector<double> q = sample_harmonic_ring_path(P, frame);
      std::vector<double> gamma0_imag(P), gamma1(P);
      const double H = -1.0 / epsilon - 0.5 * epsilon;
      for (int bead = 0; bead < P; ++bead) {
        const double x = q[bead];
        const double delta = q[(bead + 1) % P] - x;
        const double g = delta / epsilon - 0.5 * epsilon * x;
        gamma0_imag[bead] = -0.5 * x * x * g - 0.5 * x + 0.5 * g * g * g + 1.5 * H * g;
        gamma1[bead] = -0.5 * x * x * x + 0.5 * x * g * g -
          x / (2.0 * epsilon) - 0.25 * epsilon * x + 0.5 * g;
      }
      const double mu0 = -open_endpoint_pair_average(gamma0_imag);
      const double mu2 = open_endpoint_pair_average(gamma1);
      std::vector<double> gamma2_imag(P);
      for (int bead = 0; bead < P; ++bead) {
        const double x = q[bead];
        const double delta = q[(bead + 1) % P] - x;
        const double g = delta / epsilon - 0.5 * epsilon * x;
        gamma2_imag[bead] = 0.5 * x * x * g + 0.5 * x - 0.5 * g * g * g - 1.5 * H * g;
      }
      const double mu4 = -open_endpoint_pair_average(gamma2_imag);
      const bool mu0_ok = mu0_samples.add(mu0);
      const bool mu2_ok = mu2_samples.add(mu2);
      const bool mu4_ok = mu4_samples.add(mu4);
      if (!mu0_ok || !mu2_ok || !mu4_ok) ++failures;
    }
    const double values[] = {mu0_samples.mean, mu0_samples.variance(), mu0_samples.standard_error(),
      mu0_samples.variance() / P, mu0_samples.variance() / (P * P),
      mu2_samples.mean, mu2_samples.variance(), mu2_samples.standard_error(),
      mu2_samples.variance() / P, mu2_samples.variance() / (P * P),
      mu4_samples.mean, mu4_samples.variance(), mu4_samples.standard_error(),
      mu4_samples.variance() / P, mu4_samples.variance() / (P * P)};
    bool finite_summary = mu0_samples.count == frames && mu2_samples.count == frames && mu4_samples.count == frames;
    for (const double value : values) finite_summary = finite_summary && std::isfinite(value);
    check((std::string("harmonic variance-vs-P finite summary P=") +
      std::to_string(P).c_str()).c_str(), finite_summary ? 1.0 : 0.0, 1.0, 0.0);
    std::printf("variance_vs_P P=%d n=%d mu0(mean,var,stderr,var/P,var/P2)=(%.8e %.8e %.8e %.8e %.8e) "
      "mu2=(%.8e %.8e %.8e %.8e %.8e) mu4=(%.8e %.8e %.8e %.8e %.8e)\n", P, frames,
      values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
      values[8], values[9], values[10], values[11], values[12], values[13], values[14]);
  }
}

void test_static_pimd_eligibility()
{
  check("canonical NVT primitive PIMD accepted",
    is_canonical_primitive_pimd(true, true, false, false, false), 1.0, 0.0);
  check("Eco-PIMD excluded from exact statics",
    is_canonical_primitive_pimd(true, true, false, false, true), 0.0, 0.0);
  check("NPT-PIMD excluded from exact statics",
    is_canonical_primitive_pimd(true, true, true, false, false), 0.0, 0.0);
  check("SCR-PIMD excluded from exact statics",
    is_canonical_primitive_pimd(true, true, true, true, false), 0.0, 0.0);
  check("variable-temperature PIMD excluded from exact statics",
    is_canonical_primitive_pimd(true, false, false, false, false), 0.0, 0.0);
  check("RPMD excluded from exact statics",
    is_canonical_primitive_pimd(false, true, false, false, false), 0.0, 0.0);
}

void test_temperature_interpolation()
{
  check("PIMD first-step temperature", temperature_at_step(100.0, 200.0, 0, 10), 100.0, 1.0e-14);
  check("PIMD last-step interpolated temperature", temperature_at_step(100.0, 200.0, 9, 10), 190.0, 1.0e-14);
}

void test_numeric_option_parsing()
{
  int parsed_integer = 0;
  check("integer option accepted", parse_integer("37", parsed_integer) ? 1.0 : 0.0, 1.0, 0.0);
  check("integer option value", parsed_integer, 37.0, 0.0);
  const std::string integer_overflow =
    std::to_string(static_cast<long long>(std::numeric_limits<int>::max()) + 1);
  check("integer overflow rejected", parse_integer(integer_overflow.c_str(), parsed_integer) ? 1.0 : 0.0,
    0.0, 0.0);

  std::uint64_t parsed_seed = 0;
  check("negative seed rejected", parse_seed("-1", parsed_seed) ? 1.0 : 0.0, 0.0, 0.0);
  check("whitespace-prefixed negative seed rejected",
    parse_seed(" -1", parsed_seed) ? 1.0 : 0.0, 0.0, 0.0);
  check("seed overflow rejected", parse_seed("18446744073709551616", parsed_seed) ? 1.0 : 0.0,
    0.0, 0.0);
  const std::string maximum_seed = std::to_string(std::numeric_limits<std::uint64_t>::max());
  check("uint64 maximum seed accepted",
    parse_seed(maximum_seed.c_str(), parsed_seed) &&
      parsed_seed == std::numeric_limits<std::uint64_t>::max() ? 1.0 : 0.0,
    1.0, 0.0);
}

void test_candidate_probe_frame_validity()
{
  check("complete finite candidate probe frame",
    has_complete_candidate_probe_set(8, 8), 1.0, 0.0);
  check("partial candidate probe frame rejected",
    has_complete_candidate_probe_set(7, 8), 0.0, 0.0);
  check("empty candidate probe frame rejected",
    has_complete_candidate_probe_set(0, 0), 0.0, 0.0);
  ComplexStats finite_stats;
  finite_stats.add(Complex(1.0, 0.0));
  finite_stats.add(Complex(2.0, 0.0));
  check("complete candidate frame requires finite statistics",
    has_complete_candidate_probe_set(finite_stats, 2) ? 1.0 : 0.0, 1.0, 0.0);
}

void test_candidate_stats_reject_overflow()
{
  ComplexStats same_frame_stats;
  check("first large candidate probe accepted",
    same_frame_stats.add_if_finite(Complex(1.0e308, 0.0)) ? 1.0 : 0.0, 1.0, 0.0);
  check("second equal large candidate probe accepted",
    same_frame_stats.add_if_finite(Complex(1.0e308, 0.0)) ? 1.0 : 0.0, 1.0, 0.0);
  check("equal large candidate probes keep finite mean absolute value",
    same_frame_stats.mean_abs(), 1.0e308, 0.0);
  check("equal large candidate probes count as a complete frame",
    has_complete_candidate_probe_set(same_frame_stats, 2) ? 1.0 : 0.0, 1.0, 0.0);

  ComplexStats frame_stats;
  check("first extreme candidate probe accepted",
    frame_stats.add_if_finite(Complex(1.0e308, 0.0)) ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing candidate probe rejected",
    frame_stats.add_if_finite(Complex(-1.0e308, 0.0)) ? 1.0 : 0.0, 0.0, 0.0);
  check("overflowing candidate frame mean remains finite",
    std::isfinite(frame_stats.mean.real()) ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing candidate frame moments remain finite",
    frame_stats.is_finite() ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing candidate frame probe is not counted",
    frame_stats.count, 1.0, 0.0);
  check("overflowing candidate frame is incomplete",
    has_complete_candidate_probe_set(frame_stats, 2) ? 1.0 : 0.0, 0.0, 0.0);

  ComplexStats invalid_frame_stats;
  invalid_frame_stats.count = 2;
  invalid_frame_stats.m2_real = std::numeric_limits<double>::infinity();
  check("candidate frame with non-finite intermediate statistic rejected",
    has_complete_candidate_probe_set(invalid_frame_stats, 2) ? 1.0 : 0.0, 0.0, 0.0);
  invalid_frame_stats.m2_real = 0.0;
  invalid_frame_stats.mean = Complex(std::numeric_limits<double>::infinity(), 0.0);
  check("candidate frame with non-finite mean rejected",
    has_complete_candidate_probe_set(invalid_frame_stats, 2) ? 1.0 : 0.0, 0.0, 0.0);

  ComplexStats across_frames;
  check("first extreme cross-frame mean accepted",
    across_frames.add_if_finite(Complex(1.0e308, 0.0)) ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing cross-frame mean rejected",
    across_frames.add_if_finite(Complex(-1.0e308, 0.0)) ? 1.0 : 0.0, 0.0, 0.0);
  check("overflowing cross-frame mean remains finite",
    std::isfinite(across_frames.mean.real()) ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing cross-frame moments remain finite",
    across_frames.is_finite() ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing cross-frame sample is not counted",
    across_frames.count, 1.0, 0.0);
  check("overflowing cross-frame moments remain finite",
    across_frames.is_finite() ? 1.0 : 0.0, 1.0, 0.0);

  ComplexStats merge_left, merge_right;
  merge_left.add_if_finite(Complex(1.0e308, 0.0));
  merge_right.add_if_finite(Complex(1.0e308, 0.0));
  merge_left.merge(merge_right);
  check("merging equal large candidate statistics keeps mean absolute finite",
    merge_left.mean_abs(), 1.0e308, 0.0);
  check("merging equal large candidate statistics stays finite",
    merge_left.is_finite() ? 1.0 : 0.0, 1.0, 0.0);

  ComplexStats weighted_left, weighted_right;
  weighted_left.add_if_finite(Complex(1.0, 0.0));
  weighted_right.add_if_finite(Complex(0.0, 0.0));
  weighted_right.add_if_finite(Complex(0.0, 0.0));
  weighted_left.merge(weighted_right);
  check("merged candidate absolute mean uses sample weights",
    weighted_left.mean_abs(), 1.0 / 3.0, 1.0e-15);
}

void test_real_stats_overflow_rejected()
{
  RealStats same_values;
  check("first large real sample accepted", same_values.add(1.0e308) ? 1.0 : 0.0, 1.0, 0.0);
  check("second equal large real sample accepted", same_values.add(1.0e308) ? 1.0 : 0.0, 1.0, 0.0);
  check("equal large real samples are counted", same_values.count, 2.0, 0.0);
  check("equal large real samples keep finite mean absolute value",
    same_values.mean_abs(), 1.0e308, 0.0);

  RealStats opposite_values;
  check("first extreme real sample accepted", opposite_values.add(1.0e308) ? 1.0 : 0.0, 1.0, 0.0);
  check("overflowing real sample rejected", opposite_values.add(-1.0e308) ? 1.0 : 0.0, 0.0, 0.0);
  check("overflowing real sample is not counted", opposite_values.count, 1.0, 0.0);
  check("overflowing real mean retains prior finite value",
    opposite_values.mean, 1.0e308, 0.0);
}

void test_explicit_link_lifts_and_winding()
{
  const std::array<double, 9> cubic_box{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::vector<double> aligned_beads{0.1, 0.35, -0.4, -0.15};
  double spring_squared = 0.0;
  double qhm_squared = 0.0;
  for (int bead = 0; bead < static_cast<int>(aligned_beads.size()); ++bead) {
    const double spring_link = aligned_beads[bead] - aligned_beads[(bead + 1) % aligned_beads.size()];
    const double qhm_link = aligned_beads[(bead + 1) % aligned_beads.size()] - aligned_beads[bead];
    spring_squared += spring_link * spring_link;
    qhm_squared += qhm_link * qhm_link;
  }
  check("PIMD native spring and QHM links match in squared displacement", qhm_squared, spring_squared, 1.0e-14);
  const double native_long_link = aligned_beads[2] - aligned_beads[1];
  const ClosestImage remic_long_link = closest_cartesian_lattice_image({native_long_link, 0.0, 0.0}, cubic_box);
  check("QHM and PIMD retain the same bead-aligned long link", native_long_link, -0.75, 1.0e-14);
  check("per-link MIC would shorten the winding spring link", remic_long_link.length, 0.25, 1.0e-14);
  check("per-link MIC would change the native spring contribution",
    remic_long_link.length * remic_long_link.length, 0.0625, 1.0e-14);

  const std::vector<double> q{0.95, 0.02, 0.05, 0.98};
  int winding = 0;
  double net_displacement = 0.0;
  for (int bead = 0; bead < static_cast<int>(q.size()); ++bead) {
    const double raw = q[(bead + 1) % q.size()] - q[bead];
    const int lift = nearest_image_lift(raw);
    winding += lift;
    net_displacement += raw + lift;
  }
  check("zero-winding link lifts", static_cast<double>(winding), 0.0, 0.0);
  check("zero-winding unwrapped chain closes", net_displacement, 0.0, 1.0e-14);

  std::vector<double> wrapped(aligned_beads.size());
  std::vector<int> image(aligned_beads.size());
  for (size_t bead = 0; bead < aligned_beads.size(); ++bead) {
    image[bead] = static_cast<int>(std::floor(aligned_beads[bead]));
    wrapped[bead] = aligned_beads[bead] - image[bead];
  }
  int action_lift_sum = 0, closest_lift_sum = 0;
  for (int bead = 0; bead < static_cast<int>(aligned_beads.size()); ++bead) {
    const int next = (bead + 1) % static_cast<int>(aligned_beads.size());
    action_lift_sum += image[next] - image[bead];
    closest_lift_sum += nearest_image_lift(wrapped[next] - wrapped[bead]);
  }
  check("native action lift telescopes without filtering", action_lift_sum, 0.0, 0.0);
  check("Cartesian closest lift remains diagnostic", closest_lift_sum, 1.0, 0.0);
}

void test_cpu_moyal_reference()
{
  const std::vector<double> coefficients_u{0.7, 1.1};
  const std::vector<double> coefficients_f{1.2, -0.4};
  const std::vector<double> r{0.2, -0.3};
  const std::vector<Complex> p{0.4, -0.6};
  const double step = 1.0e-3;
  double cartesian = 0.0;
  for (int i = 0; i < 2; ++i) {
    std::vector<std::vector<double>> dirs(3, std::vector<double>(2, 0.0));
    for (auto& direction : dirs) direction[i] = 1.0;
    const double u3 = mixed_directional_derivative<double>(r, dirs, {step, step, step},
      [&](const std::vector<double>& x) {
        return coefficients_u[0] * std::pow(x[0], 3) + coefficients_u[1] * std::pow(x[1], 3);
      });
    const Complex f3 = mixed_directional_derivative<Complex>(p, dirs, {step, step, step},
      [&](const std::vector<Complex>& x) {
        return coefficients_f[0] * x[0] * x[0] * x[0] + coefficients_f[1] * x[1] * x[1] * x[1];
      });
    cartesian += (u3 * f3).real();
  }

  RecursionSettings settings{step, step, 1.0, 2, 10, 98765};
  const auto probe_set = probes(settings, 30000, 2);
  ComplexStats stochastic;
  for (const auto& probe : probe_set) {
    const std::vector<std::vector<double>> dirs = {
      probe.direction[0], probe.direction[1], probe.direction[2]};
    const double u3 = mixed_directional_derivative<double>(r, dirs, {step, step, step},
      [&](const std::vector<double>& x) {
        return coefficients_u[0] * std::pow(x[0], 3) + coefficients_u[1] * std::pow(x[1], 3);
      });
    const Complex f3 = mixed_directional_derivative<Complex>(p, dirs, {step, step, step},
      [&](const std::vector<Complex>& x) {
        return coefficients_f[0] * x[0] * x[0] * x[0] + coefficients_f[1] * x[1] * x[1] * x[1];
      });
    stochastic.add(u3 * f3);
  }
  check("CPU Cartesian vs stochastic Moyal", stochastic.mean.real(), cartesian,
    6.0 * stochastic.standard_error() + 1.0e-8);
}

int main()
{
  test_free_particle();
  test_harmonic_recursion();
  test_quartic_moyal();
  test_periodic_translation();
  test_open_endpoint_no_conjugate();
  test_bead_specific_static_probe_seeds();
  test_weyl_reference_coefficients();
  test_closest_cartesian_lattice_image();
  test_shortest_triclinic_lattice_vector();
  test_temperature_interpolation();
  test_numeric_option_parsing();
  test_static_pimd_eligibility();
  test_candidate_probe_frame_validity();
  test_candidate_stats_reject_overflow();
  test_real_stats_overflow_rejected();
  test_harmonic_finite_p_moments();
  test_harmonic_gamma_structure();
  test_harmonic_estimator_variance_vs_p();
  test_explicit_link_lifts_and_winding();
  test_cpu_moyal_reference();
  std::printf("%s quantum heat CPU tests max_abs=%.6e max_rel=%.6e\n",
    failures == 0 ? "PASS" : "FAIL", max_absolute_error, max_relative_error);
  return failures == 0 ? 0 : 1;
}
