#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../../src/force/nep_plane_flux_view.cuh"
#include <cassert>
#include <cmath>
#include <vector>

static double three_body_energy(const double xi, const double xj, const double xk)
{
  return 0.5 * (xj - xi) * (xj - xi) * (xk - xi);
}

int main()
{
  assert(NEP_PLANE_FLUX_ERROR_DISPLACEMENT != NEP_PLANE_FLUX_ERROR_MISSING_SLOT);
  assert(NEP_PLANE_FLUX_ERROR_DISPLACEMENT != NEP_PLANE_FLUX_ERROR_NONFINITE);
  assert(NEP_PLANE_FLUX_ERROR_MISSING_SLOT != NEP_PLANE_FLUX_ERROR_NONFINITE);
  constexpr double length = 24.0;
  assert(nep_plane_crossing_direction(23.5, 0.5, 0.0, length, true) == 1);
  assert(nep_plane_crossing_direction(0.5, 23.5, 0.0, length, true) == -1);
  assert(nep_plane_crossing_direction(0.5, 23.5, 24.0, length, true) == -1);
  assert(nep_plane_crossing_direction(23.5, 0.5, 24.0, length, true) == 1);
  assert(nep_plane_crossing_direction(23.5, 0.5, 12.0, length, true) == 0);
  assert(nep_plane_crossing_direction(2.0, 4.0, 3.0, length, false) == 1);
  assert(nep_plane_crossing_direction(4.0, 2.0, 3.0, length, false) == -1);
  assert(nep_plane_crossing_direction(23.5, 0.5, 24.0, length, false) == 0);
  assert(nep_plane_flux_bounds_valid(7.0, 0.5, length, true));
  assert(!nep_plane_flux_bounds_valid(7.0, 1.0, 18.0, true));
  assert(nep_plane_flux_bounds_valid(100.0, 1.0, 0.0, false));

  const double reference_z[] = {23.5, 0.5, 12.0};
  int compact_slot[3] = {-1, -1, -1};
  int next_slot = 0;
  for (int atom = 0; atom < 3; ++atom) {
    if (nep_plane_atom_selected(reference_z[atom], 0.0, length, true, 1.0)) {
      compact_slot[atom] = next_slot++;
    }
  }
  assert(next_slot == 2 && compact_slot[0] == 0 && compact_slot[1] == 1 && compact_slot[2] == -1);

  const double dij[3] = {1.25, -0.75, 2.0};
  const double dji[3] = {-dij[0], -dij[1], -dij[2]};
  const double vi[3] = {0.2, 0.4, -0.3};
  const double vj[3] = {-0.1, 0.8, 0.5};
  double q_view = 0.0;
  double q_partition = 0.0;
  const int ij_direction = nep_plane_crossing_direction(23.5, 0.5, 0.0, length, true);
  const int ji_direction = nep_plane_crossing_direction(0.5, 23.5, 0.0, length, true);
  for (int d = 0; d < 3; ++d) {
    const double a_j = -ij_direction * dij[d];
    const double a_i = -ji_direction * dji[d];
    q_view += a_i * vi[d] + a_j * vj[d];
    q_partition += dji[d] * vi[d] - dij[d] * vj[d];
  }
  assert(std::abs(q_view - q_partition) < 1.0e-12);

  const double xi = 0.3;
  const double xj = 1.1;
  const double xk = -0.2;
  const double dij_local = (xj - xi) * (xk - xi);
  const double dik_local = 0.5 * (xj - xi) * (xj - xi);
  const double dii_local = -dij_local - dik_local;
  const double epsilon = 1.0e-6;
  const double fd_j = (three_body_energy(xi, xj + epsilon, xk) -
                       three_body_energy(xi, xj - epsilon, xk)) / (2.0 * epsilon);
  const double fd_i = (three_body_energy(xi + epsilon, xj, xk) -
                       three_body_energy(xi - epsilon, xj, xk)) / (2.0 * epsilon);
  assert(std::abs(fd_j - dij_local) < 1.0e-9);
  assert(std::abs(fd_i - dii_local) < 1.0e-9);
  const double vi_local = 0.3;
  const double vj_local = -0.2;
  const double vk_local = 0.4;
  const double derivative = dii_local * vi_local + dij_local * vj_local + dik_local * vk_local;
  const double dt = 1.0e-5;
  const double fd_time =
    (three_body_energy(xi + vi_local * dt, xj + vj_local * dt, xk + vk_local * dt) -
     three_body_energy(xi - vi_local * dt, xj - vj_local * dt, xk - vk_local * dt)) / (2.0 * dt);
  assert(std::abs(fd_time - derivative) < 1.0e-8);

  const double dt_ps = 0.25;
  const double correlation_plus[] = {4.0, -8.0};
  const double correlation_minus[] = {0.0, 3.0};
  const double nyquist = 3.14159265358979323846 / dt_ps;
  const double negative_zero_frequency = interface_shc_spectrum_value(
    dt_ps, 0.0, correlation_plus, correlation_minus, 2);
  assert(negative_zero_frequency < 0.0);
  const double integral = interface_shc_frequency_integral(
    dt_ps, nyquist, correlation_plus, correlation_minus, 2);
  assert(std::abs(integral - correlation_plus[0]) < 1.0e-12);

  const double dt_resolved = 0.25;
  const double max_omega = 3.14159265358979323846 / dt_resolved;
  const double resolved_plus[] = {4.0, -2.0, 0.5};
  const double resolved_minus[] = {0.0, 1.5, -0.2};
  assert(interface_shc_frequency_grid_resolved(dt_resolved, max_omega, 3, 3));
  assert(!interface_shc_frequency_grid_resolved(dt_resolved, max_omega, 2, 3));
  assert(interface_shc_frequency_grid_resolved(dt_resolved, max_omega, 33, 3));
  const double analytic = interface_shc_frequency_integral(
    dt_resolved, max_omega, resolved_plus, resolved_minus, 3);
  std::vector<double> exported(3);
  for (int omega_id = 0; omega_id < 3; ++omega_id) {
    const double omega = max_omega * omega_id / 2.0;
    exported[omega_id] = interface_shc_spectrum_value(
      dt_resolved, omega, resolved_plus, resolved_minus, 3);
  }
  const double grid_trapezoid = interface_shc_trapezoid_integral(exported.data(), 3, max_omega);
  assert(std::abs(analytic - resolved_plus[0]) < 1.0e-12);
  assert(std::abs(grid_trapezoid - resolved_plus[0]) < 1.0e-12);
  return 0;
}
