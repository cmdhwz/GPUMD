#include "force/pppm.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace
{
constexpr int N = 3;

double total(const std::vector<double>& values)
{
  double result = 0.0;
  for (double value : values) result += value;
  return result;
}

void wrap(const Box& box, std::vector<double>& position)
{
  // This fixture uses an orthogonal cell, so primary-cell wrapping is direct.
  for (int i = 0; i < N; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      const int index = i + axis * N;
      const double length = box.cpu_h[axis * 3 + axis];
      position[index] -= std::floor(position[index] / length) * length;
    }
  }
}

void evaluate(
  PPPM& pppm,
  const Box& box,
  GPU_Vector<float>& charge,
  GPU_Vector<double>& position,
  GPU_Vector<float>& d_real,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  GPU_Vector<double>& potential,
  const std::vector<float>& host_charge,
  const std::vector<double>& host_position,
  const unsigned long long id,
  std::vector<double>& site_energy,
  const bool request_peratom_virial = true)
{
  charge.copy_from_host(host_charge.data());
  position.copy_from_host(host_position.data());
  d_real.fill(0.0f);
  force.fill(0.0);
  virial.fill(0.0);
  potential.fill(0.0);
  pppm.find_force(N, 0, N, box, charge, position, d_real, force, virial, potential,
                  request_peratom_virial, id);
  site_energy.resize(N);
  potential.copy_to_host(site_energy.data());
}

void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}
} // namespace

int main()
{
  try {
    Box box;
    for (int i = 0; i < 18; ++i) box.cpu_h[i] = 0.0;
    box.cpu_h[0] = 8.0;
    box.cpu_h[4] = 9.0;
    box.cpu_h[8] = 10.0;
    box.get_inverse();
    box.set_is_orthogonal();

    PPPM pppm;
    pppm.initialize(0.35f, false, false, 0.6);
    GPU_Vector<float> charge(N), d_real(N);
    GPU_Vector<double> position(3 * N), force(3 * N), virial(9 * N), potential(N);
    GPU_Vector<double> direction(3 * N), charge_direction(N), dsite, explicit_gradient, ik_force;
    GPU_Vector<double> gradient_only, ik_only;
    std::vector<float> q{0.71f, -0.43f, -0.28f};
    std::vector<double> r{0.02, 3.11, 7.52, 1.2, 4.03, 8.75, 2.8, 5.2, 9.31};
    const std::vector<double> u{-1.5, 0.07, 0.02, 0.03, -0.05, 0.01, -0.02, 0.04, -0.06};
    const std::vector<double> dq{0.11, -0.035, -0.075};
    direction.copy_from_host(u.data());
    charge_direction.copy_from_host(dq.data());

    std::vector<double> baseline_site;
    evaluate(pppm, box, charge, position, d_real, force, virial, potential, q, r, 1, baseline_site);
    require(pppm.compute_reference_energy_tangent(
      N, box, charge, position, &direction, &charge_direction, 1, &dsite,
      explicit_gradient, ik_force), "reference PPPM tangent rejected its current mesh");
    std::vector<double> tangent_site(N), tangent_gradient(3 * N), tangent_ik(3 * N);
    dsite.copy_to_host(tangent_site.data());
    explicit_gradient.copy_to_host(tangent_gradient.data());
    ik_force.copy_to_host(tangent_ik.data());
    const double tangent = total(tangent_site);

    PPPMReferenceTranslationReport translation_report;
    require(pppm.diagnose_reference_translation_energy(
      N, box, charge, position, 1, translation_report, 6.0e-7),
      "reference translation oracle rejected its current mesh");
    require(translation_report.even_G && translation_report.mesh_zero_mode,
      "reference translation oracle found an invalid reciprocal operator");
    require(translation_report.mesh_invariant_pass,
      "reference translation oracle failed integer-grid translation closure");
    require(translation_report.assignment_closure_pass,
      "reference translation oracle failed charge-assignment closure");
    require(translation_report.fd_platform_pass,
      "reference translation oracle did not establish a finite-difference platform");
    require(std::isfinite(translation_report.native_vs_fp64_energy_error),
      "reference translation oracle did not report the native/FP64 energy difference");
    require(std::abs(total(baseline_site) - translation_report.native_reciprocal_energy) < 2.0e-5,
      "reference translation oracle used an inconsistent reciprocal FFT energy factor");
    for (int axis = 0; axis < 3; ++axis) {
      require(std::isfinite(translation_report.axis[axis].native_energy_derivative),
        "reference translation oracle produced a non-finite native derivative");
      require(std::isfinite(translation_report.axis[axis].native_vs_fp64_error[0]),
        "reference translation oracle did not compare native and FP64 derivatives");
      for (int phase = 0; phase < 2; ++phase) {
        require(std::isfinite(translation_report.axis[axis].richardson_derivative[phase][1]) &&
                translation_report.axis[axis].fd_plateau_error[phase] <= 6.0e-7,
          "reference translation oracle failed its three-step Richardson convergence check");
        for (int step = 0; step < 3; ++step)
          require(std::isfinite(translation_report.axis[axis].fd_derivative[phase][step]),
            "reference translation oracle produced a non-finite finite difference");
      }
    }

    GPU_Vector<double> repeated_site;
    require(pppm.compute_reference_energy_tangent(
      N, box, charge, position, &direction, &charge_direction, 1, &repeated_site,
      gradient_only, ik_only), "repeated PPPM tangent invalidated the native field");
    std::vector<double> repeated(N);
    repeated_site.copy_to_host(repeated.data());
    for (int i = 0; i < N; ++i)
      require(std::abs(repeated[i] - tangent_site[i]) < 1.0e-12,
              "repeated PPPM tangent changed site derivatives");

    std::vector<float> baseline_d_real(N);
    d_real.copy_to_host(baseline_d_real.data());
    double adjoint_reference = 0.0;
    for (int i = 0; i < N; ++i) adjoint_reference += dq[i] * baseline_d_real[i];
    for (int i = 0; i < 3 * N; ++i) adjoint_reference += u[i] * tangent_gradient[i];
    require(std::abs(tangent - adjoint_reference) < 2.0e-5,
            "PPPM site tangent failed the charge/shape adjoint identity");

    require(!pppm.compute_reference_energy_tangent(
      N, box, charge, position, &direction, &charge_direction, 999, &dsite,
      explicit_gradient, ik_force), "PPPM tangent accepted a stale force-frame ID");

    require(pppm.compute_reference_energy_tangent(
      N, box, charge, position, nullptr, nullptr, 1, nullptr, gradient_only, ik_only),
      "gradient-only PPPM path rejected its current mesh");
    std::vector<double> gradient_only_host(3 * N), ik_only_host(3 * N);
    gradient_only.copy_to_host(gradient_only_host.data());
    ik_only.copy_to_host(ik_only_host.data());
    for (int i = 0; i < 3 * N; ++i) {
      require(std::isfinite(tangent_gradient[i]) && std::isfinite(tangent_ik[i]),
              "PPPM gradient output was non-finite");
      require(gradient_only_host[i] == tangent_gradient[i], "gradient-only path changed explicit gradient");
      require(ik_only_host[i] == tangent_ik[i], "gradient-only path changed native ik force");
    }

    const std::vector<double> zero_u(3 * N, 0.0), zero_dq(N, 0.0);
    direction.copy_from_host(zero_u.data());
    charge_direction.copy_from_host(zero_dq.data());
    GPU_Vector<double> zero_dsite;
    require(pppm.compute_reference_energy_tangent(
      N, box, charge, position, &direction, &charge_direction, 1, &zero_dsite,
      gradient_only, ik_only), "zero-direction PPPM tangent rejected its current mesh");
    std::vector<double> zero_site(N);
    zero_dsite.copy_to_host(zero_site.data());
    for (double value : zero_site) require(value == 0.0, "zero PPPM tangent was nonzero");

    auto finite_difference = [&](const double step, const unsigned long long plus_id,
                                 const unsigned long long minus_id, std::vector<double>& derivative) {
      std::vector<double> plus_r = r, minus_r = r, plus_site, minus_site;
      std::vector<float> plus_q = q, minus_q = q;
      for (int i = 0; i < 3 * N; ++i) {
        plus_r[i] += step * u[i];
        minus_r[i] -= step * u[i];
      }
      wrap(box, plus_r);
      wrap(box, minus_r);
      for (int i = 0; i < N; ++i) {
        plus_q[i] = static_cast<float>(q[i] + step * dq[i]);
        minus_q[i] = static_cast<float>(q[i] - step * dq[i]);
      }
      evaluate(pppm, box, charge, position, d_real, force, virial, potential, plus_q, plus_r, plus_id, plus_site);
      evaluate(pppm, box, charge, position, d_real, force, virial, potential, minus_q, minus_r, minus_id, minus_site);
      derivative.resize(N);
      for (int i = 0; i < N; ++i) derivative[i] = (plus_site[i] - minus_site[i]) / (2.0 * step);
    };
    std::vector<double> fd_h, fd_half;
    finite_difference(0.05, 2, 3, fd_h);
    finite_difference(0.025, 4, 5, fd_half);
    double fd_error_h = 0.0, fd_error_half = 0.0, fd_step_difference = 0.0, tangent_scale = 0.0;
    for (int i = 0; i < N; ++i) {
      fd_error_h = std::max(fd_error_h, std::abs(tangent_site[i] - fd_h[i]));
      fd_error_half = std::max(fd_error_half, std::abs(tangent_site[i] - fd_half[i]));
      fd_step_difference = std::max(fd_step_difference, std::abs(fd_h[i] - fd_half[i]));
      tangent_scale = std::max(tangent_scale, std::max(
        std::abs(tangent_site[i]), std::max(std::abs(fd_h[i]), std::abs(fd_half[i]))));
    }
    const double fd_tolerance = 1.0e-4 + 2.0e-3 * tangent_scale;
    require(fd_error_h < fd_tolerance && fd_error_half < fd_tolerance,
            "PPPM site tangent disagrees with independent site-energy differences");
    require(fd_step_difference < fd_tolerance, "PPPM site-energy differences are step-unstable");

    double closure = 0.0;
    for (int i = 0; i < 3 * N; ++i) closure = std::max(closure, std::abs(tangent_gradient[i] + tangent_ik[i]));
    std::printf("PPPM per-site tangent FD errors %.9g / %.9g (h / h2), scale %.9g, tolerance %.9g; explicit-gradient plus native-ik residual %.9g\n",
                fd_error_h, fd_error_half, tangent_scale, fd_tolerance, closure);
    evaluate(pppm, box, charge, position, d_real, force, virial, potential, q, r, 6,
             baseline_site, false);
    require(!pppm.compute_reference_energy_tangent(
      N, box, charge, position, &direction, &charge_direction, 6, &dsite,
      explicit_gradient, ik_force), "PPPM tangent accepted a frame without per-atom energy allocation");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "PPPM reference tangent CUDA check failed: %s\n", error.what());
    return 1;
  }
}
