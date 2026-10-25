#ifdef NDEBUG
#undef NDEBUG
#endif

#include "force/nep.cuh"
#include "model/box.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
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
    result.coefficients.resize(static_cast<size_t>(atom_count) * 3);
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
  GPU_Vector<int> slots_device(atom_count);
  GPU_Vector<double> coefficients(static_cast<size_t>(atom_count) * 3, 0.0);
  const int offsets[] = {0, atom_count};
  std::vector<int> slots(atom_count);
  for (int n = 0; n < atom_count; ++n) slots[n] = n;
  reference_device.copy_from_host(reference.data(), reference.size());
  plane_device.copy_from_host(&plane, 1);
  offsets_device.copy_from_host(offsets, 2);
  slots_device.copy_from_host(slots.data(), slots.size());
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
  view.atom_slots = slots_device.data();
  view.coefficients = coefficients.data();
  view.error_flag = error_flag.data();
  nep.set_plane_flux_view(view);

  Evaluation base;
  evaluate(nep, box, type, position, potential, force, virial, reference, atom_count, false, view, error_flag, coefficients, base);
  Evaluation captured;
  evaluate(nep, box, type, position, potential, force, virial, reference, atom_count, true, view, error_flag, coefficients, captured);
  for (size_t i = 0; i < base.potential.size(); ++i) {
    assert(close(base.potential[i], captured.potential[i], 1.0e-12, 1.0e-12));
  }
  for (size_t i = 0; i < base.force.size(); ++i) {
    assert(close(base.force[i], captured.force[i], 1.0e-12, 1.0e-12));
  }
  for (size_t i = 0; i < base.virial.size(); ++i) {
    assert(close(base.virial[i], captured.virial[i], 1.0e-12, 1.0e-12));
  }

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
    for (int d = 0; d < 3; ++d) {
      for (const double h : {0.004, 0.008}) {
        std::vector<double> plus = reference;
        std::vector<double> minus = reference;
        plus[static_cast<size_t>(d) * atom_count + selected] += h;
        minus[static_cast<size_t>(d) * atom_count + selected] -= h;
        Evaluation ep, em;
        evaluate(nep, box, type, position, potential, force, virial, plus, atom_count, false, view, error_flag, coefficients, ep);
        evaluate(nep, box, type, position, potential, force, virial, minus, atom_count, false, view, error_flag, coefficients, em);
        const bool differentiated_lower_partition = !lower;
        const double eplus = partition_energy(ep.potential, z, plane, differentiated_lower_partition);
        const double eminus = partition_energy(em.potential, z, plane, differentiated_lower_partition);
        const double d_partition = (eplus - eminus) / (2.0 * h);
        const double expected = (lower ? 1.0 : -1.0) * d_partition;
        const double actual = captured.coefficients[static_cast<size_t>(selected) * 3 + d];
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
  for (int n = 0; n < atom_count; ++n) {
    for (int d = 0; d < 3; ++d) {
      q += captured.coefficients[static_cast<size_t>(n) * 3 + d] *
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
  evaluate(nep, box, type, position, potential, force, virial, plus, atom_count, false, view, error_flag, coefficients, ep);
  evaluate(nep, box, type, position, potential, force, virial, minus, atom_count, false, view, error_flag, coefficients, em);
  const double d_ua_dt =
    (partition_energy(ep.potential, z, plane, true) - partition_energy(em.potential, z, plane, true)) / (2.0 * h_time);
  const double v_dot_f_a = dot_force_velocity_in_lower(base.force, velocity, z, plane, atom_count);
  const double identity_residual = d_ua_dt + v_dot_f_a + q;
  std::printf("Q %.12g dUA_dt %.12g sumA_vF %.12g residual %.4g (abs=5e-4 rel=3e-2)\n",
    q, d_ua_dt, v_dot_f_a, identity_residual);
  assert(close(d_ua_dt + v_dot_f_a, -q, 5.0e-4, 3.0e-2));
  return 0;
}
