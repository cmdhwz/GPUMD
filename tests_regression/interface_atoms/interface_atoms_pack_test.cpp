#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../../src/measure/interface_atoms_pack.cuh"
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

int main()
{
  constexpr int atoms = 3;
  const double position[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  const int position_image[9] = {1, 0, -1, -2, 0, 2, 3, 0, -3};
  const double velocity[9] = {21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0};
  const double force[9] = {31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0};
  const double potential[3] = {41.0, 42.0, 43.0};
  const double box_x = 10.0;
  const double box_y = 20.0;
  const double box_z = 30.0;
  const double velocity_conversion = 1000.0 / 10.18051;
  double row0[INTERFACE_ATOMS_COMPONENTS] = {};
  double row2[INTERFACE_ATOMS_COMPONENTS] = {};

  assert(interface_atoms_pack_record(
    0, atoms, position, position_image, velocity, force, potential,
    box_x, box_y, box_z, velocity_conversion, row0));
  assert(interface_atoms_pack_record(
    2, atoms, position, position_image, velocity, force, potential,
    box_x, box_y, box_z, velocity_conversion, row2));
  assert(row0[0] == 1.0 && row0[3] == 11.0 && row0[4] == -36.0 && row0[5] == 97.0);
  assert(row0[9] == 31.0 && row0[12] == 41.0);
  assert(row2[0] == 3.0 && row2[3] == -7.0 && row2[4] == 46.0 && row2[5] == -81.0);
  assert(row2[9] == 33.0 && row2[12] == 43.0);
  assert(std::abs(row2[6] - 23.0 * velocity_conversion) < 1.0e-12);
  assert(row2[6] != 23.0);

  const double position4[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const int position_image4[12] = {0, 0, 0, 2, 0, 0, 0, -1, 0, 0, 0, 1};
  const double velocity4[12] = {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
  const double force4[12] = {31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
  const double potential4[4] = {41, 42, 43, 44};
  double row3[INTERFACE_ATOMS_COMPONENTS] = {};
  assert(interface_atoms_pack_record(
    3, 4, position4, position_image4, velocity4, force4, potential4,
    box_x, box_y, box_z, velocity_conversion, row3));
  assert(row3[0] == 4.0 && row3[3] == 24.0 && row3[4] == -12.0 && row3[5] == 42.0);
  assert(row3[9] == 34.0 && row3[12] == 44.0);

  // A four-sample GPU block stores 104 values for two selected atoms. A three-sample
  // tail copies only 78 values, beginning at the same zero-based frame offset.
  size_t block_values = 0;
  size_t tail_values = 0;
  size_t third_frame_offset = 0;
  assert(interface_atoms_frame_value_count(4, 2, block_values) && block_values == 104);
  assert(interface_atoms_frame_value_count(3, 2, tail_values) && tail_values == 78);
  assert(interface_atoms_frame_offset(2, 2, third_frame_offset) && third_frame_offset == 52);
  assert(interface_atoms_frame_value_count(0, 2, tail_values) && tail_values == 0);
  assert(interface_atoms_frame_value_count(4, 0, tail_values) && tail_values == 0);

  size_t overflow = 0;
  assert(!interface_atoms_frame_value_count(
    std::numeric_limits<size_t>::max(), 2, overflow));

  double invalid_force[9];
  for (int i = 0; i < 9; ++i) invalid_force[i] = force[i];
  invalid_force[2] = std::numeric_limits<double>::infinity();
  assert(!interface_atoms_pack_record(
    2, atoms, position, position_image, velocity, invalid_force, potential,
    box_x, box_y, box_z, velocity_conversion, row2));
  return 0;
}
