#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct XYZAtom
{
  double position[3];
  double velocity[3];
  double force[3];
  double energy;
  double unwrapped[3];
};

struct XYZFrame
{
  double time_fs;
  std::vector<XYZAtom> atoms;
};

static std::vector<XYZFrame> read_xyz(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  std::vector<XYZFrame> frames;
  int atom_count = 0;
  while (input >> atom_count) {
    assert(atom_count >= 0);
    std::string line;
    std::getline(input, line);
    assert(static_cast<bool>(std::getline(input, line)));
    const size_t time_start = line.find("Time=");
    assert(time_start != std::string::npos);
    XYZFrame frame;
    frame.time_fs = std::stod(line.substr(time_start + 5));
    frame.atoms.resize(static_cast<size_t>(atom_count));
    for (XYZAtom& atom : frame.atoms) {
      assert(static_cast<bool>(std::getline(input, line)));
      std::istringstream row(line);
      std::string symbol;
      row >> symbol;
      assert(symbol == "Ar");
      for (double& value : atom.position) row >> value;
      for (double& value : atom.velocity) row >> value;
      for (double& value : atom.force) row >> value;
      row >> atom.energy;
      for (double& value : atom.unwrapped) row >> value;
      assert(static_cast<bool>(row));
    }
    frames.push_back(frame);
  }
  return frames;
}

static std::vector<std::vector<double>> read_numeric_rows(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<double> row;
    double value = 0.0;
    while (fields >> value) row.push_back(value);
    assert(fields.eof());
    rows.push_back(row);
  }
  return rows;
}

static bool contains_line(const char* filename, const std::string& text)
{
  std::ifstream input(filename);
  assert(input.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(text) != std::string::npos) return true;
  }
  return false;
}

static bool near(const double a, const double b)
{
  return std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= 1.0e-11 * (1.0 + std::abs(b));
}

static void verify_late_unwrapped(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  int atom_count = 0;
  input >> atom_count;
  assert(atom_count == 8);
  std::string line;
  std::getline(input, line);
  assert(static_cast<bool>(std::getline(input, line))); // frame header
  assert(static_cast<bool>(std::getline(input, line))); // atom 0
  std::istringstream row(line);
  std::string symbol;
  double x, y, z, xu, yu, zu;
  row >> symbol >> x >> y >> z >> xu >> yu >> zu;
  assert(symbol == "Ar" && static_cast<bool>(row));
  assert(near(xu, 13.9));
}

int main(int argc, char** argv)
{
  assert(argc == 4);
  const char* baseline_file = argv[1];
  const char* local_file = argv[2];
  const char* late_file = argv[3];
  assert(contains_line(local_file, "# dump_interface_atoms version 1"));
  assert(contains_line(local_file, "# geometry_source local_geometry_geometry.json"));
  assert(contains_line(
    local_file,
    "# columns: frame step time_ps atom_id x_A y_A z_A xu_A yu_A zu_A vx_A_ps vy_A_ps vz_A_ps"));
  const std::vector<XYZFrame> baseline = read_xyz(baseline_file);
  const auto rows = read_numeric_rows(local_file);
  assert(baseline.size() == 3 && rows.size() == 6);
  for (size_t frame = 0; frame < baseline.size(); ++frame) {
    assert(baseline[frame].atoms.size() == 8);
    for (size_t selected = 0; selected < 2; ++selected) {
      const auto& row = rows[2 * frame + selected];
      assert(row.size() == 17);
      const int atom_id = static_cast<int>(selected);
      const XYZAtom& ref = baseline[frame].atoms[atom_id];
      assert(row[0] == static_cast<double>(frame));
      assert(row[1] == static_cast<double>(frame + 1));
      assert(row[3] == static_cast<double>(atom_id));
      assert(std::abs(row[2] * 1000.0 - baseline[frame].time_fs) <= 5.1e-9);
      for (int d = 0; d < 3; ++d) {
        assert(near(row[4 + d], ref.position[d]));
        assert(near(row[7 + d], ref.unwrapped[d]));
        assert(near(row[10 + d], ref.velocity[d] * 1000.0));
        assert(near(row[13 + d], ref.force[d]));
      }
      assert(near(row[16], ref.energy));
    }
  }
  verify_late_unwrapped(late_file);
  return 0;
}
