#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace
{
constexpr double LATTICE = 3.567;
constexpr int REPEATS = 6;
constexpr int ATOMS_PER_CELL = 8;
constexpr double NYQUIST_RAD_PS = 3.14159265358979323846 / 0.0005;

bool write_fixture(
  const std::string& mode, const std::string& directory, const std::string& carbon_model)
{
  const std::string geometry_name = directory + "/geometry.in";
  const std::string model_name = directory + "/model.xyz";
  const std::string run_name = directory + "/run.in";
  std::ofstream geometry(geometry_name.c_str());
  std::ofstream model(model_name.c_str());
  std::ofstream run(run_name.c_str());
  if (!geometry || !model || !run) return false;
  model << std::setprecision(17);

  geometry << "axis z\n" << "bin_width_A " << LATTICE << "\n";
  if (mode == "geometry_split") {
    geometry << "interfaces 1 5\n" << "sampling_planes 1 16\n";
  } else if (mode == "geometry_legacy") {
    geometry << "interfaces 2 7 3\n";
  } else {
    geometry << "interfaces 0\n"
             << (mode == "empty_shc" ? "sampling_planes 0\n"
                                      : "sampling_planes 2 7.134 14.268\n");
  }
  geometry << "material_group 1\n" << "bath_groups 0 0 1\n" << "bin_group 2\n";

  model << REPEATS * REPEATS * REPEATS * ATOMS_PER_CELL << '\n'
        << "pbc=\"T T F\" Lattice=\"21.402 0 0 0 21.402 0 0 0 21.402\" "
        << "Properties=species:S:1:pos:R:3:vel:R:3:group:I:3\n";
  const double basis[ATOMS_PER_CELL][3] = {
    {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0},
    {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75}, {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  const double two_pi = 6.28318530717958647692;
  for (int ix = 0; ix < REPEATS; ++ix) {
    for (int iy = 0; iy < REPEATS; ++iy) {
      for (int iz = 0; iz < REPEATS; ++iz) {
        for (int b = 0; b < ATOMS_PER_CELL; ++b) {
          double x = (ix + basis[b][0]) * LATTICE;
          double y = (iy + basis[b][1]) * LATTICE;
          double z = (iz + basis[b][2]) * LATTICE;
          const int atom_id = ((ix * REPEATS + iy) * REPEATS + iz) * ATOMS_PER_CELL + b;
          const double distortion = 0.004 * std::sin(0.731 * (atom_id + 1));
          x += distortion;
          y += 0.5 * distortion;
          if (z > 0.02) z += 0.25 * distortion;
          const double vx = 0.004 * std::sin(two_pi * (ix + 0.25 * b) / REPEATS);
          const double vy = 0.004 * std::cos(two_pi * (iy + 0.25 * b) / REPEATS);
          const double vz = 0.004 * std::sin(two_pi * (iz + 0.25 * b) / REPEATS);
          const int bath_group = z < LATTICE ? 0 : (z >= 5.0 * LATTICE ? 1 : 2);
          const int bin_group = static_cast<int>(z / LATTICE);
          model << "C " << x << ' ' << y << ' ' << z << ' ' << vx << ' ' << vy << ' ' << vz
                << ' ' << bath_group << " 0 " << bin_group << '\n';
        }
      }
    }
  }

  run << std::setprecision(17) << "potential " << carbon_model << '\n'
      << "time_step 0.5\n";
  if (mode == "heat_lan") run << "ensemble heat_lan 300 100 50 0 1\n";
  else run << "ensemble nve\n";
  run << "interface_geometry geometry.in native_shared\n";
  if (mode != "empty_shc") {
    run << "compute_interface_profile 1 1 no native_profile.out\n";
    if (mode == "geometry_split" || mode == "geometry_legacy") {
      run << "compute_interface_shc 1 2 4 65 " << NYQUIST_RAD_PS << " 1.5 native_shc\n";
    } else {
      run << "compute_interface_shc 1 4 8 65 " << NYQUIST_RAD_PS << " 1.5 native_shc\n";
    }
  } else {
    run << "compute_interface_shc 1 4 8 65 " << NYQUIST_RAD_PS << " 1.5 empty_shc\n";
  }
  const int steps = mode == "heat_lan" ? 20000 : (mode == "empty_shc" ? 1 :
    ((mode == "geometry_split" || mode == "geometry_legacy") ? 8 : 320));
  run << "run " << steps << '\n';
  return static_cast<bool>(geometry) && static_cast<bool>(model) && static_cast<bool>(run);
}
} // namespace

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::cerr << "usage: interface_fixture_prepare <nve|heat_lan|empty_shc|geometry_split|geometry_legacy> <build-directory> <carbon-model>\n";
    return 2;
  }
  const std::string mode = argv[1];
  if (mode != "nve" && mode != "heat_lan" && mode != "empty_shc" &&
      mode != "geometry_split" && mode != "geometry_legacy") {
    std::cerr << "unknown interface fixture mode\n";
    return 2;
  }
  if (!write_fixture(mode, argv[2], argv[3])) {
    std::cerr << "cannot write interface runtime fixture\n";
    return 1;
  }
  return 0;
}
