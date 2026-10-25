#ifdef NDEBUG
#undef NDEBUG
#endif

#include "force/nep.cuh"
#include "model/box.cuh"
#include "measure/interface_geometry.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

struct Evaluation
{
  std::vector<double> potential;
  std::vector<double> force;
  std::vector<double> virial;
  std::vector<double> coefficients;
};

static void evaluate(
  NEP& nep,
  Box& box,
  const GPU_Vector<int>& type,
  GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial,
  const std::vector<double>& host_position,
  const int atom_count,
  const int selected_count,
  const bool capture,
  const NEPPlaneFluxView& plane_flux_view,
  GPU_Vector<int>& error_flag,
  GPU_Vector<double>& coefficients,
  Evaluation& result)
{
  position.copy_from_host(host_position.data(), host_position.size());
  potential.fill(0.0);
  force.fill(0.0);
  virial.fill(0.0);
  if (capture) {
    coefficients.fill(0.0);
    error_flag.fill(0);
  }
  nep.set_plane_flux_view(capture ? plane_flux_view : NEPPlaneFluxView{});
  nep.compute(box, type, position, potential, force, virial);
  result.potential.resize(atom_count);
  result.force.resize(static_cast<size_t>(atom_count) * 3);
  result.virial.resize(static_cast<size_t>(atom_count) * 9);
  potential.copy_to_host(result.potential.data(), result.potential.size());
  force.copy_to_host(result.force.data(), result.force.size());
  virial.copy_to_host(result.virial.data(), result.virial.size());
  if (capture) {
    int error = 0;
    error_flag.copy_to_host(&error, 1);
    assert(error == 0);
    result.coefficients.resize(static_cast<size_t>(selected_count) * 3);
    coefficients.copy_to_host(result.coefficients.data(), result.coefficients.size());
    for (const double coefficient : result.coefficients) assert(std::isfinite(coefficient));
  }
  nep.clear_plane_flux_view();
}

static double partition_energy(const std::vector<double>& site_energy, const std::vector<double>& z, const double plane, const bool lower)
{
  double sum = 0.0;
  for (size_t n = 0; n < z.size(); ++n) {
    if ((z[n] < plane) == lower) sum += site_energy[n];
  }
  return sum;
}

static bool in_reference_slab(const double x, const double lower, const double upper)
{
  return x >= lower && x < upper;
}

static double partition_energy_region(
  const std::vector<double>& site_energy,
  const std::vector<double>& reference_x,
  const double lower,
  const double upper)
{
  double sum = 0.0;
  for (size_t n = 0; n < reference_x.size(); ++n) {
    if (in_reference_slab(reference_x[n], lower, upper)) sum += site_energy[n];
  }
  return sum;
}

static double region_v_dot_f(
  const std::vector<double>& force,
  const std::vector<double>& velocity,
  const std::vector<double>& reference_x,
  const double lower,
  const double upper,
  const int atom_count)
{
  double sum = 0.0;
  for (int n = 0; n < atom_count; ++n) {
    if (!in_reference_slab(reference_x[n], lower, upper)) continue;
    for (int d = 0; d < 3; ++d) {
      const size_t index = static_cast<size_t>(d) * atom_count + n;
      sum += force[index] * velocity[index];
    }
  }
  return sum;
}

static std::vector<double> plane_fluxes(
  const Evaluation& evaluation,
  const std::vector<int>& selected_atoms,
  const std::vector<int>& plane_offsets,
  const std::vector<double>& velocity,
  const int atom_count)
{
  const int planes = static_cast<int>(plane_offsets.size()) - 1;
  std::vector<double> result(planes, 0.0);
  for (int p = 0; p < planes; ++p) {
    for (int slot = plane_offsets[p]; slot < plane_offsets[p + 1]; ++slot) {
      const int atom = selected_atoms[slot];
      for (int d = 0; d < 3; ++d) {
        result[p] += evaluation.coefficients[static_cast<size_t>(slot) * 3 + d] *
          velocity[static_cast<size_t>(d) * atom_count + atom];
      }
    }
    assert(std::isfinite(result[p]));
  }
  return result;
}

static double dot_force_velocity_in_lower(
  const std::vector<double>& force,
  const std::vector<double>& velocity,
  const std::vector<double>& z,
  const double plane,
  const int atom_count)
{
  double sum = 0.0;
  for (int n = 0; n < atom_count; ++n) {
    if (z[n] < plane) {
      for (int d = 0; d < 3; ++d) {
        sum += force[static_cast<size_t>(d) * atom_count + n] * velocity[static_cast<size_t>(d) * atom_count + n];
      }
    }
  }
  return sum;
}

static bool close(const double a, const double b, const double absolute, const double relative)
{
  return std::abs(a - b) <= absolute + relative * std::max(std::abs(a), std::abs(b));
}

static void assert_force_results_equal(const Evaluation& a, const Evaluation& b)
{
  assert(a.potential.size() == b.potential.size());
  assert(a.force.size() == b.force.size());
  assert(a.virial.size() == b.virial.size());
  for (size_t i = 0; i < a.potential.size(); ++i) {
    assert(close(a.potential[i], b.potential[i], 1.0e-12, 1.0e-12));
  }
  for (size_t i = 0; i < a.force.size(); ++i) {
    assert(close(a.force[i], b.force[i], 1.0e-12, 1.0e-12));
  }
  for (size_t i = 0; i < a.virial.size(); ++i) {
    assert(close(a.virial[i], b.virial[i], 1.0e-12, 1.0e-12));
  }
}

int main(int argc, char** argv)
{
  assert(argc == 2);
  constexpr double lattice = 3.567;
  constexpr int repeats = 6;
  constexpr int unit_atoms = 8;
  constexpr int atom_count = repeats * repeats * repeats * unit_atoms;
  constexpr double box_length = lattice * repeats;
  constexpr double plane = 10.9;
  const double basis[unit_atoms][3] = {
    {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0},
    {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  std::vector<double> reference(static_cast<size_t>(atom_count) * 3, 0.0);
  std::vector<double> z(atom_count);
  int atom = 0;
  for (int ix = 0; ix < repeats; ++ix) {
    for (int iy = 0; iy < repeats; ++iy) {
      for (int iz = 0; iz < repeats; ++iz) {
        for (int b = 0; b < unit_atoms; ++b, ++atom) {
          for (int d = 0; d < 3; ++d) {
            const int cell = d == 0 ? ix : (d == 1 ? iy : iz);
            reference[static_cast<size_t>(d) * atom_count + atom] =
              (cell + basis[b][d]) * lattice;
          }
          z[atom] = reference[static_cast<size_t>(2) * atom_count + atom];
        }
      }
    }
  }
  const int distorted_atom =
    ((repeats / 2 * repeats + repeats / 2) * repeats + repeats / 2) * unit_atoms + 4;
  reference[distorted_atom] += 0.02;
  reference[static_cast<size_t>(2) * atom_count + distorted_atom] += 0.013;
  z[distorted_atom] += 0.013;

  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = box_length;
  box.pbc_x = 1;
  box.pbc_y = 1;
  box.pbc_z = 0;
  box.get_inverse();
  box.set_is_orthogonal();

  std::vector<std::string> no_input_lines;
  RunInput run_input(no_input_lines);
  NEP nep(argv[1], atom_count, run_input);
  nep.N1 = 0;
  nep.N2 = atom_count;
  assert(nep.supports_interface_plane_flux());
  assert(nep.uses_large_box_for_interface_flux(box));

  std::vector<int> host_type(atom_count, 0);
  GPU_Vector<int> type(atom_count);
  type.copy_from_host(host_type.data(), host_type.size());
  GPU_Vector<double> position(static_cast<size_t>(atom_count) * 3);
  GPU_Vector<double> potential(atom_count);
  GPU_Vector<double> force(static_cast<size_t>(atom_count) * 3);
  GPU_Vector<double> virial(static_cast<size_t>(atom_count) * 9);
  GPU_Vector<int> error_flag(1, 0);
  GPU_Vector<double> reference_device(reference.size());
  GPU_Vector<double> plane_device(1);
  GPU_Vector<int> offsets_device(2);
  std::vector<int> selected_atoms;
  std::vector<int> plane_offsets(1, 0);
  for (int n = 0; n < atom_count; ++n) {
    if (nep_plane_atom_selected(z[n], plane, box_length, false,
                                nep.get_interface_plane_flux_cutoff() + 2.0 * 0.1)) {
      selected_atoms.push_back(n);
    }
  }
  plane_offsets.push_back(static_cast<int>(selected_atoms.size()));
  const int selected_count = static_cast<int>(selected_atoms.size());
  std::vector<int> candidate_offsets;
  std::vector<int> candidate_planes;
  assert(nep_plane_build_candidate_index(
    atom_count, 1, selected_atoms.data(), selected_count, plane_offsets.data(),
    candidate_offsets, candidate_planes));
  assert(selected_count > 0 && candidate_planes.size() == selected_atoms.size());
  GPU_Vector<int> candidate_offsets_device(candidate_offsets.size());
  GPU_Vector<int> candidate_planes_device(candidate_planes.size());
  GPU_Vector<int> selected_atoms_device(selected_atoms.size());
  GPU_Vector<double> coefficients(static_cast<size_t>(selected_count) * 3, 0.0);
  reference_device.copy_from_host(reference.data(), reference.size());
  plane_device.copy_from_host(&plane, 1);
  offsets_device.copy_from_host(plane_offsets.data(), plane_offsets.size());
  candidate_offsets_device.copy_from_host(candidate_offsets.data(), candidate_offsets.size());
  candidate_planes_device.copy_from_host(candidate_planes.data(), candidate_planes.size());
  selected_atoms_device.copy_from_host(selected_atoms.data(), selected_atoms.size());
  NEPPlaneFluxView view;
  view.enabled = 1;
  view.atom_count = atom_count;
  view.plane_count = 1;
  view.axis = 2;
  view.pbc[0] = 1;
  view.pbc[1] = 1;
  view.pbc[2] = 0;
  view.box_lengths[0] = view.box_lengths[1] = view.box_lengths[2] = box_length;
  view.max_reference_displacement = 0.1;
  view.reference_positions = reference_device.data();
  view.plane_positions = plane_device.data();
  view.plane_offsets = offsets_device.data();
  view.candidate_offsets = candidate_offsets_device.data();
  view.candidate_planes = candidate_planes_device.data();
  view.selected_atoms = selected_atoms_device.data();
  view.coefficients = coefficients.data();
  view.error_flag = error_flag.data();
  nep.set_plane_flux_view(view);

  Evaluation base;
  evaluate(nep, box, type, position, potential, force, virial, reference, atom_count, selected_count, false, view, error_flag, coefficients, base);
  Evaluation captured;
  evaluate(nep, box, type, position, potential, force, virial, reference, atom_count, selected_count, true, view, error_flag, coefficients, captured);
  assert_force_results_equal(base, captured);

  int atom_a = -1;
  int atom_b = -1;
  for (int n = 0; n < atom_count; ++n) {
    if (z[n] < plane && (atom_a < 0 || z[n] > z[atom_a])) atom_a = n;
    if (z[n] >= plane && (atom_b < 0 || z[n] < z[atom_b])) atom_b = n;
  }
  assert(atom_a >= 0 && atom_b >= 0);
  double coefficient_norm = 0.0;
  for (double value : captured.coefficients) coefficient_norm += value * value;
  coefficient_norm = std::sqrt(coefficient_norm);
  assert(coefficient_norm > 0.0);
  std::printf("captured coefficient norm %.12g; finite-difference tolerance abs=2e-4 rel=2e-2\n", coefficient_norm);

  for (const int selected : {atom_a, atom_b}) {
    const bool lower = selected == atom_a;
    const int selected_slot = nep_plane_find_selected_slot(
      selected_atoms.data(), plane_offsets.data(), 1, 0, selected);
    assert(selected_slot >= 0);
    for (int d = 0; d < 3; ++d) {
      for (const double h : {0.004, 0.008}) {
        std::vector<double> plus = reference;
        std::vector<double> minus = reference;
        plus[static_cast<size_t>(d) * atom_count + selected] += h;
        minus[static_cast<size_t>(d) * atom_count + selected] -= h;
        Evaluation ep, em;
        evaluate(nep, box, type, position, potential, force, virial, plus, atom_count, selected_count, false, view, error_flag, coefficients, ep);
        evaluate(nep, box, type, position, potential, force, virial, minus, atom_count, selected_count, false, view, error_flag, coefficients, em);
        const bool differentiated_lower_partition = !lower;
        const double eplus = partition_energy(ep.potential, z, plane, differentiated_lower_partition);
        const double eminus = partition_energy(em.potential, z, plane, differentiated_lower_partition);
        const double d_partition = (eplus - eminus) / (2.0 * h);
        const double expected = (lower ? 1.0 : -1.0) * d_partition;
        const double actual = captured.coefficients[static_cast<size_t>(selected_slot) * 3 + d];
        const double error = std::abs(actual - expected);
        std::printf("atom %d axis %d h %.4g coefficient %.9g finite_difference %.9g abs_error %.4g\n",
          selected, d, h, actual, expected, error);
        assert(close(actual, expected, 2.0e-4, 2.0e-2));
      }
    }
  }

  std::vector<double> velocity(static_cast<size_t>(atom_count) * 3);
  for (int d = 0; d < 3; ++d) {
    for (int n = 0; n < atom_count; ++n) {
      velocity[static_cast<size_t>(d) * atom_count + n] =
        0.05 * std::sin(0.37 * (n + 1) * (d + 1));
    }
  }
  double q = 0.0;
  for (int slot = 0; slot < selected_count; ++slot) {
    const int n = selected_atoms[slot];
    for (int d = 0; d < 3; ++d) {
      q += captured.coefficients[static_cast<size_t>(slot) * 3 + d] *
        velocity[static_cast<size_t>(d) * atom_count + n];
    }
  }
  assert(std::abs(q) > 1.0e-8);
  const double h_time = 0.1;
  std::vector<double> plus = reference;
  std::vector<double> minus = reference;
  for (size_t i = 0; i < reference.size(); ++i) {
    plus[i] += h_time * velocity[i];
    minus[i] -= h_time * velocity[i];
  }
  Evaluation ep, em;
  evaluate(nep, box, type, position, potential, force, virial, plus, atom_count, selected_count, false, view, error_flag, coefficients, ep);
  evaluate(nep, box, type, position, potential, force, virial, minus, atom_count, selected_count, false, view, error_flag, coefficients, em);
  const double d_ua_dt =
    (partition_energy(ep.potential, z, plane, true) - partition_energy(em.potential, z, plane, true)) / (2.0 * h_time);
  const double v_dot_f_a = dot_force_velocity_in_lower(base.force, velocity, z, plane, atom_count);
  const double identity_residual = d_ua_dt + v_dot_f_a + q;
  std::printf("Q %.12g dUA_dt %.12g sumA_vF %.12g residual %.4g (abs=5e-4 rel=3e-2)\n",
    q, d_ua_dt, v_dot_f_a, identity_residual);
  assert(close(d_ua_dt + v_dot_f_a, -q, 5.0e-4, 3.0e-2));

  // Independently check a slab bounded by the periodic seam and an interior plane.
  std::vector<double> reference_x = reference;
  std::vector<double> x(atom_count);
  for (int n = 0; n < atom_count; ++n) {
    reference_x[n] += 0.125 * lattice;
    assert(interface_wrap_coordinate(reference_x[n], box_length));
    x[n] = reference_x[n];
  }
  const double slab_plane = 10.9;
  const double x_planes[2] = {0.0, slab_plane};
  const int x_plane_count = 2;
  std::vector<int> x_selected_atoms;
  std::vector<int> x_plane_offsets{0};
  const double x_selection_cutoff = nep.get_interface_plane_flux_cutoff() + 1.0;
  assert(nep_plane_flux_bounds_valid(
    nep.get_interface_plane_flux_cutoff(), 0.5, box_length, true));
  for (int p = 0; p < x_plane_count; ++p) {
    for (int n = 0; n < atom_count; ++n) {
      if (nep_plane_atom_selected(x[n], x_planes[p], box_length, true, x_selection_cutoff)) {
        x_selected_atoms.push_back(n);
      }
    }
    x_plane_offsets.push_back(static_cast<int>(x_selected_atoms.size()));
  }
  const int x_selected_count = static_cast<int>(x_selected_atoms.size());
  assert(x_selected_count > 0 && x_plane_offsets[1] > 0 && x_plane_offsets[2] > x_plane_offsets[1]);
  std::vector<int> x_candidate_offsets;
  std::vector<int> x_candidate_planes;
  assert(nep_plane_build_candidate_index(
    atom_count, x_plane_count, x_selected_atoms.data(), x_selected_count,
    x_plane_offsets.data(), x_candidate_offsets, x_candidate_planes));

  GPU_Vector<double> x_reference_device(reference_x.size());
  GPU_Vector<double> x_plane_device(x_plane_count);
  GPU_Vector<int> x_plane_offsets_device(x_plane_offsets.size());
  GPU_Vector<int> x_candidate_offsets_device(x_candidate_offsets.size());
  GPU_Vector<int> x_candidate_planes_device(x_candidate_planes.size());
  GPU_Vector<int> x_selected_atoms_device(x_selected_atoms.size());
  GPU_Vector<double> x_coefficients(static_cast<size_t>(x_selected_count) * 3, 0.0);
  GPU_Vector<int> x_error_flag(1, 0);
  x_reference_device.copy_from_host(reference_x.data(), reference_x.size());
  x_plane_device.copy_from_host(x_planes, x_plane_count);
  x_plane_offsets_device.copy_from_host(x_plane_offsets.data(), x_plane_offsets.size());
  x_candidate_offsets_device.copy_from_host(x_candidate_offsets.data(), x_candidate_offsets.size());
  x_candidate_planes_device.copy_from_host(x_candidate_planes.data(), x_candidate_planes.size());
  x_selected_atoms_device.copy_from_host(x_selected_atoms.data(), x_selected_atoms.size());
  NEPPlaneFluxView x_view;
  x_view.enabled = 1;
  x_view.atom_count = atom_count;
  x_view.plane_count = x_plane_count;
  x_view.axis = 0;
  x_view.pbc[0] = x_view.pbc[1] = 1;
  x_view.box_lengths[0] = x_view.box_lengths[1] = x_view.box_lengths[2] = box_length;
  x_view.max_reference_displacement = 0.5;
  x_view.reference_positions = x_reference_device.data();
  x_view.plane_positions = x_plane_device.data();
  x_view.plane_offsets = x_plane_offsets_device.data();
  x_view.candidate_offsets = x_candidate_offsets_device.data();
  x_view.candidate_planes = x_candidate_planes_device.data();
  x_view.selected_atoms = x_selected_atoms_device.data();
  x_view.coefficients = x_coefficients.data();
  x_view.error_flag = x_error_flag.data();

  Evaluation x_state;
  evaluate(nep, box, type, position, potential, force, virial, reference_x,
    atom_count, x_selected_count, true, x_view, x_error_flag, x_coefficients, x_state);
  Evaluation x_without_view;
  evaluate(nep, box, type, position, potential, force, virial, reference_x,
    atom_count, x_selected_count, false, x_view, x_error_flag, x_coefficients, x_without_view);
  assert_force_results_equal(x_state, x_without_view);
  const std::vector<double> x_flux = plane_fluxes(
    x_state, x_selected_atoms, x_plane_offsets, velocity, atom_count);
  assert(std::abs(x_flux[0]) > 1.0e-8 && std::abs(x_flux[1]) > 1.0e-8);

  int seam_negative_side = -1;
  int seam_positive_side = -1;
  int interior_negative_side = -1;
  int interior_positive_side = -1;
  for (int n = 0; n < atom_count; ++n) {
    if (x[n] > box_length * 0.5 &&
        (seam_negative_side < 0 || x[n] > x[seam_negative_side])) seam_negative_side = n;
    if (x[n] < box_length * 0.5 &&
        (seam_positive_side < 0 || x[n] < x[seam_positive_side])) seam_positive_side = n;
    if (x[n] < slab_plane &&
        (interior_negative_side < 0 || x[n] > x[interior_negative_side])) interior_negative_side = n;
    if (x[n] >= slab_plane &&
        (interior_positive_side < 0 || x[n] < x[interior_positive_side])) interior_positive_side = n;
  }
  assert(seam_negative_side >= 0 && seam_positive_side >= 0 &&
         interior_negative_side >= 0 && interior_positive_side >= 0);
  for (int p = 0; p < x_plane_count; ++p) {
    const int side_atoms[2] = {
      p == 0 ? seam_negative_side : interior_negative_side,
      p == 0 ? seam_positive_side : interior_positive_side};
    for (int side = 0; side < 2; ++side) {
      const int selected_atom = side_atoms[side];
      const int selected_slot = nep_plane_find_selected_slot(
        x_selected_atoms.data(), x_plane_offsets.data(), x_plane_count, p, selected_atom);
      assert(selected_slot >= 0);
      const bool atom_in_slab = (p == 0) ? (side == 1) : (side == 0);
      const double region_lower = atom_in_slab ? slab_plane : 0.0;
      const double region_upper = atom_in_slab ? box_length : slab_plane;
      for (const double h : {0.004, 0.008}) {
        std::vector<double> plus = reference_x;
        std::vector<double> minus = reference_x;
        plus[selected_atom] += h;
        minus[selected_atom] -= h;
        Evaluation ep, em;
        evaluate(nep, box, type, position, potential, force, virial, plus,
          atom_count, x_selected_count, false, x_view, x_error_flag, x_coefficients, ep);
        evaluate(nep, box, type, position, potential, force, virial, minus,
          atom_count, x_selected_count, false, x_view, x_error_flag, x_coefficients, em);
        const double d_partition = (
          partition_energy_region(ep.potential, x, region_lower, region_upper) -
          partition_energy_region(em.potential, x, region_lower, region_upper)) / (2.0 * h);
        // The seam is the slab's left boundary; the interior plane is its right boundary.
        const double boundary_orientation = p == 0 ? -1.0 : 1.0;
        const double side_orientation = atom_in_slab ? 1.0 : -1.0;
        const double expected = boundary_orientation * side_orientation * d_partition;
        const double actual = x_state.coefficients[static_cast<size_t>(selected_slot) * 3];
        std::printf(
          "plane %d atom %d h %.4g coefficient %.9g finite_difference %.9g\n",
          p, selected_atom, h, actual, expected);
        assert(close(actual, expected, 2.0e-4, 2.0e-2));
      }
    }
  }

  std::vector<double> x_plus = reference_x;
  std::vector<double> x_minus = reference_x;
  const double time_difference = 0.1;
  for (size_t i = 0; i < reference_x.size(); ++i) {
    x_plus[i] += time_difference * velocity[i];
    x_minus[i] -= time_difference * velocity[i];
  }
  Evaluation x_ep, x_em;
  evaluate(nep, box, type, position, potential, force, virial, x_plus,
    atom_count, x_selected_count, false, x_view, x_error_flag, x_coefficients, x_ep);
  evaluate(nep, box, type, position, potential, force, virial, x_minus,
    atom_count, x_selected_count, false, x_view, x_error_flag, x_coefficients, x_em);
  const double slab_du_dt = (
    partition_energy_region(x_ep.potential, x, 0.0, slab_plane) -
    partition_energy_region(x_em.potential, x, 0.0, slab_plane)) / (2.0 * time_difference);
  const double slab_v_dot_f = region_v_dot_f(
    x_state.force, velocity, x, 0.0, slab_plane, atom_count);
  const double slab_rhs = x_flux[0] - x_flux[1];
  std::printf("periodic slab dU_dt+sum(v.F) %.12g; Q_seam-Q_inside %.12g\n",
    slab_du_dt + slab_v_dot_f, slab_rhs);
  assert(close(slab_du_dt + slab_v_dot_f, slab_rhs, 5.0e-4, 3.0e-2));

  const auto run_periodic_slab_nve = [&](const double dt_fs) {
    const double dt = dt_fs / 10.18051;
    const int steps = static_cast<int>(20.0 / dt_fs);
    std::vector<double> trajectory_position = reference_x;
    std::vector<double> trajectory_velocity = velocity;
    Evaluation current;
    evaluate(nep, box, type, position, potential, force, virial, trajectory_position,
      atom_count, x_selected_count, true, x_view, x_error_flag, x_coefficients, current);
    const auto energy = [&](const Evaluation& state, const std::vector<double>& speed) {
      double value = partition_energy_region(state.potential, x, 0.0, slab_plane);
      for (int n = 0; n < atom_count; ++n) {
        if (!in_reference_slab(x[n], 0.0, slab_plane)) continue;
        for (int d = 0; d < 3; ++d) {
          const double component = speed[static_cast<size_t>(d) * atom_count + n];
          value += 0.5 * 12.011 * component * component;
        }
      }
      return value;
    };
    const double initial_energy = energy(current, trajectory_velocity);
    const std::vector<double> q_initial = plane_fluxes(
      current, x_selected_atoms, x_plane_offsets, trajectory_velocity, atom_count);
    double previous_net_flux = q_initial[0] - q_initial[1];
    double integrated_flux = 0.0;
    double absolute_exchange = 0.0;
    double flux_activity = std::abs(q_initial[0]) + std::abs(q_initial[1]);
    double final_energy = initial_energy;
    double maximum_balance_error = 0.0;
    for (int step = 0; step < steps; ++step) {
      for (int d = 0; d < 3; ++d) {
        for (int n = 0; n < atom_count; ++n) {
          const size_t index = static_cast<size_t>(d) * atom_count + n;
          trajectory_velocity[index] += 0.5 * dt * current.force[index] / 12.011;
          trajectory_position[index] += dt * trajectory_velocity[index];
          if (d < 2) interface_wrap_coordinate(trajectory_position[index], box_length);
        }
      }
      Evaluation next;
      evaluate(nep, box, type, position, potential, force, virial, trajectory_position,
        atom_count, x_selected_count, true, x_view, x_error_flag, x_coefficients, next);
      for (int d = 0; d < 3; ++d) {
        for (int n = 0; n < atom_count; ++n) {
          const size_t index = static_cast<size_t>(d) * atom_count + n;
          trajectory_velocity[index] += 0.5 * dt * next.force[index] / 12.011;
        }
      }
      final_energy = energy(next, trajectory_velocity);
      const std::vector<double> q = plane_fluxes(
        next, x_selected_atoms, x_plane_offsets, trajectory_velocity, atom_count);
      const double net_flux = q[0] - q[1];
      integrated_flux += 0.5 * (previous_net_flux + net_flux) * dt;
      absolute_exchange += 0.5 * (std::abs(previous_net_flux) + std::abs(net_flux)) * dt;
      previous_net_flux = net_flux;
      flux_activity += std::abs(q[0]) + std::abs(q[1]);
      current = std::move(next);
      const double balance_error = energy(current, trajectory_velocity) - initial_energy - integrated_flux;
      maximum_balance_error = std::max(maximum_balance_error, std::abs(balance_error));
    }
    const double delta_energy = final_energy - initial_energy;
    const double endpoint_error = std::abs(delta_energy - integrated_flux);
    constexpr double absolute_error_floor_eV = 1.0e-3;
    constexpr double relative_exchange_tolerance = 0.10;
    const double error_tolerance = absolute_error_floor_eV +
      relative_exchange_tolerance * std::abs(integrated_flux);
    std::printf(
      "slab NVE dt %.4g fs steps %d deltaE %.12g eV integral(Q_left-Q_right) %.12g eV "
      "absolute_exchange %.12g eV endpoint_error %.6g eV max_ledger_error %.6g eV "
      "tolerance %.6g eV (1e-3 eV + 10%% signed exchanged energy) activity %.6g\n",
      dt_fs, steps, delta_energy, integrated_flux, absolute_exchange, endpoint_error,
      maximum_balance_error, error_tolerance, flux_activity);
    assert(flux_activity > 1.0e-6);
    assert(absolute_exchange > 1.0e-5 && std::abs(integrated_flux) > 1.0e-2);
    assert(endpoint_error <= error_tolerance);
    assert(maximum_balance_error <= error_tolerance);
    return maximum_balance_error;
  };
  const double coarse_nve_error = run_periodic_slab_nve(0.2);
  const double fine_nve_error = run_periodic_slab_nve(0.1);
  constexpr double floating_error_floor_eV = 1.0e-3;
  if (coarse_nve_error > floating_error_floor_eV) {
    assert(fine_nve_error < coarse_nve_error);
  } else {
    assert(fine_nve_error <= floating_error_floor_eV);
  }
  return 0;
}
