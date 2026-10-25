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

  constexpr int sparse_atom_count = 10;
  constexpr int sparse_plane_count = 6;
  const double sparse_coordinates[sparse_atom_count] = {
    23.5, 0.5, 5.0, 6.0, 11.5, 12.0, 12.5, 18.0, 20.0, 10.0};
  const double sparse_planes[sparse_plane_count] = {12.5, 0.0, 6.0, 18.0, 12.0, 15.0};
  std::vector<int> selected_atoms;
  std::vector<int> plane_offsets(1, 0);
  std::vector<int> dense_slots(static_cast<size_t>(sparse_atom_count) * sparse_plane_count, -1);
  for (int p = 0; p < sparse_plane_count; ++p) {
    for (int n = 0; n < sparse_atom_count; ++n) {
      if (nep_plane_atom_selected(sparse_coordinates[n], sparse_planes[p], length, true, 1.0)) {
        dense_slots[static_cast<size_t>(p) * sparse_atom_count + n] =
          static_cast<int>(selected_atoms.size());
        selected_atoms.push_back(n);
      }
    }
    plane_offsets.push_back(static_cast<int>(selected_atoms.size()));
  }
  std::vector<int> candidate_offsets;
  std::vector<int> candidate_planes;
  assert(nep_plane_build_candidate_index(
    sparse_atom_count, sparse_plane_count, selected_atoms.data(),
    static_cast<int>(selected_atoms.size()), plane_offsets.data(), candidate_offsets, candidate_planes));
  assert(candidate_planes.size() == selected_atoms.size());
  assert(plane_offsets[5] == plane_offsets[6]);
  assert(candidate_offsets[5] < candidate_offsets[6]);
  assert(candidate_planes[candidate_offsets[5]] == 0);
  assert(candidate_planes[candidate_offsets[5] + 1] == 4);
  assert(candidate_offsets[9] == candidate_offsets[10]);
  for (int p = 0; p < sparse_plane_count; ++p) {
    for (int n = 0; n < sparse_atom_count; ++n) {
      assert(nep_plane_find_selected_slot(
        selected_atoms.data(), plane_offsets.data(), sparse_plane_count, p, n) ==
        dense_slots[static_cast<size_t>(p) * sparse_atom_count + n]);
    }
  }
  for (int n = 0; n < sparse_atom_count; ++n) {
    int previous_plane = -1;
    for (int c = candidate_offsets[n]; c < candidate_offsets[n + 1]; ++c) {
      const int p = candidate_planes[c];
      assert(p > previous_plane && p < sparse_plane_count);
      assert(nep_plane_atom_selected(sparse_coordinates[n], sparse_planes[p], length, true, 1.0));
      previous_plane = p;
    }
  }
  std::vector<double> dense_coefficients(selected_atoms.size() * 3, 0.0);
  std::vector<double> sparse_coefficients(selected_atoms.size() * 3, 0.0);
  for (int n1 = 0; n1 < sparse_atom_count; ++n1) {
    for (int n2 = 0; n2 < sparse_atom_count; ++n2) {
      if (n1 == n2) continue;
      double displacement = sparse_coordinates[n2] - sparse_coordinates[n1];
      displacement -= std::nearbyint(displacement / length) * length;
      if (std::abs(displacement) > 1.0) continue;
      const float derivative[3] = {
        static_cast<float>(0.25 + 0.01 * n1),
        static_cast<float>(-0.4 + 0.005 * n2),
        static_cast<float>(0.1 + 0.002 * (n1 + n2))};
      for (int p = 0; p < sparse_plane_count; ++p) {
        const int direction = nep_plane_crossing_direction(
          sparse_coordinates[n1], sparse_coordinates[n2], sparse_planes[p], length, true);
        if (direction == 0) continue;
        bool sender_candidate_found = false;
        for (int c = candidate_offsets[n1]; c < candidate_offsets[n1 + 1]; ++c) {
          sender_candidate_found = sender_candidate_found || candidate_planes[c] == p;
        }
        assert(sender_candidate_found);
        const int slot = dense_slots[static_cast<size_t>(p) * sparse_atom_count + n2];
        assert(slot >= 0);
        for (int d = 0; d < 3; ++d) {
          dense_coefficients[static_cast<size_t>(slot) * 3 + d] -= direction * derivative[d];
        }
      }
      for (int c = candidate_offsets[n1]; c < candidate_offsets[n1 + 1]; ++c) {
        const int p = candidate_planes[c];
        const int direction = nep_plane_crossing_direction(
          sparse_coordinates[n1], sparse_coordinates[n2], sparse_planes[p], length, true);
        if (direction == 0) continue;
        const int slot = nep_plane_find_selected_slot(
          selected_atoms.data(), plane_offsets.data(), sparse_plane_count, p, n2);
        assert(slot >= 0);
        const double sign = nep_plane_flux_coefficient_sign(direction);
        for (int d = 0; d < 3; ++d) {
          sparse_coefficients[static_cast<size_t>(slot) * 3 + d] += sign * derivative[d];
        }
      }
    }
  }
  for (size_t i = 0; i < dense_coefficients.size(); ++i) {
    assert(std::abs(dense_coefficients[i] - sparse_coefficients[i]) < 1.0e-12);
  }

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
