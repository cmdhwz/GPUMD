#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
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

int main()
{
  const auto vacf = read_rows("stationary_vacf.out");
  const auto vdos = read_rows("stationary_vdos.out");
  assert(vacf.size() == 6 && vdos.size() == 15);
  bool found_nonzero_x = false;
  for (const auto& row : vacf) {
    assert(row.size() == 12 && row[5] == "1");
    const double raw = std::stod(row[10]);
    if (row[7] == "x") {
      assert(std::isfinite(raw));
      found_nonzero_x = found_nonzero_x || raw > 0.0;
      if (row[8] == "0") assert(raw > 0.0 && std::stod(row[11]) == 1.0);
    } else {
      assert((row[7] == "y" || row[7] == "z") && raw == 0.0);
      assert(row[11] == "nan");
    }
  }
  assert(found_nonzero_x);

  for (const auto& row : vdos) {
    assert(row.size() == 13 && row[5] == "1");
    if (row[7] == "y" || row[7] == "z") {
      assert(std::stod(row[9]) == 0.0);
      assert(std::stod(row[10]) == 0.0);
      assert(std::stod(row[11]) == 0.0);
      assert(std::stod(row[12]) == 0.0);
    }
  }
  return 0;
}
