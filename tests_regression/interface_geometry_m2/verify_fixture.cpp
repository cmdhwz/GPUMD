#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::vector<std::string>> read_rows(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  std::vector<std::vector<std::string>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<std::string> row;
    std::string value;
    while (fields >> value) row.emplace_back(value);
    rows.emplace_back(row);
  }
  return rows;
}

static bool has_line(const char* filename, const char* prefix)
{
  std::ifstream input(filename);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(prefix) == 0) return true;
  }
  return false;
}

int main()
{
  assert(has_line("shared_geometry.json", "{"));
  assert(has_line("shared_geometry.json", "  \"geometry_validity\": \"orthogonal_static_reference\""));
  const auto bins = read_rows("shared_bins.dat");
  const auto groups = read_rows("shared_groups.dat");
  assert(bins.size() == 4 && groups.size() == 8);
  assert(bins[0].size() == 8 && bins[1].size() == 8 && bins[2].size() == 8);
  assert(bins[0][5] == "-1" && bins[0][6] == "1" && bins[0][7] == "1");
  assert(bins[1][5] == "-1" && bins[1][6] == "0" && bins[1][7] == "1");
  assert(bins[2][5] == "1" && bins[2][6] == "2" && bins[2][7] == "0");
  assert(bins[3][4] == "0" && bins[3][5] == "-1");
  assert(groups[3][4] == "1" && groups[5][5] == "1");

  assert(has_line("profile.out", "# compute_interface_profile version 3"));
  const auto profile = read_rows("profile.out");
  assert(profile.size() == 8);
  for (const auto& row : profile) assert(row.size() == 19);
  assert(profile[0][0] == "4" && profile[0][15] == "4");
  assert(profile[4][0] == "6" && profile[4][15] == "2");

  const auto vacf_all = read_rows("dos_all_vacf.out");
  const auto vdos_all = read_rows("dos_all_vdos.out");
  const auto vacf_type = read_rows("dos_type0_vacf.out");
  const auto vdos_type = read_rows("dos_type0_vdos.out");
  assert(vacf_all.size() == 4 * 3 * 2 && vdos_all.size() == 4 * 3 * 5);
  assert(vacf_type.size() == 1 * 3 * 2 && vdos_type.size() == 1 * 3 * 5);
  assert(has_line("dos_all_vdos.out", "# frequency_grid_underresolved 0"));
  assert(has_line("dos_all_vdos.out", "band_limited 1"));
  for (const auto& row : vacf_all) assert(row.size() == 12);
  for (const auto& row : vdos_all) assert(row.size() == 13);
  assert(vacf_all[0][4] == "4" && vacf_all[1][4] == "3");
  for (const auto& row : vacf_type) assert(row[5] == "1" && row[6] == "0");
  bool empty_bin_nan = false;
  for (const auto& row : vacf_all) {
    if (row[5] == "3") empty_bin_nan = row[10] == "nan" && row[11] == "nan";
  }
  assert(empty_bin_nan);
  for (size_t first = 0; first < vdos_type.size(); first += 5) {
    const auto& first_row = vdos_type[first];
    assert(std::isfinite(std::stod(first_row[9])));
    assert(std::isfinite(std::stod(first_row[10])));
    assert(std::isfinite(std::stod(first_row[11])));
    double trapezoid = 0.0;
    for (int w = 0; w < 5; ++w) {
      const auto& row = vdos_type[first + w];
      assert(row.size() == 13);
      assert(std::isfinite(std::stod(row[12])));
      const double weight = w == 0 || w == 4 ? 0.5 : 1.0;
      trapezoid += weight * std::stod(row[12]);
    }
    const double max_omega = std::stod(vdos_type[first + 4][8]);
    trapezoid *= max_omega / 4.0 / 3.14159265358979323846;
    assert(std::abs(trapezoid - std::stod(first_row[11])) <=
      1.0e-8 + 1.0e-10 * std::max(std::abs(trapezoid), std::abs(std::stod(first_row[11]))));
  }
  return 0;
}
