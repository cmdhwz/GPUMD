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

static std::vector<std::vector<double>> read_profile(const char* path)
{
  std::ifstream input(path);
  assert(input.good());
  std::string line;
  bool version_2 = false;
  std::vector<std::vector<double>> rows;
  while (std::getline(input, line)) {
    if (line.find("# compute_chunk interface_profile version 2") == 0) version_2 = true;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream values(line);
    std::vector<double> row;
    std::string value;
    while (values >> value) row.push_back(std::stod(value));
    assert(row.size() == 17);
    rows.push_back(row);
  }
  assert(version_2);
  assert(!rows.empty());
  return rows;
}

static const std::vector<double>& frame(const std::vector<std::vector<double>>& rows,
                                       int step, int bin)
{
  for (const auto& row : rows) {
    if (static_cast<int>(row[0]) == step && static_cast<int>(row[3]) == bin) return row;
  }
  assert(false);
  return rows.front();
}

static std::vector<std::vector<double>> read_compute()
{
  std::ifstream input("compute.out");
  assert(input.good());
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream values(line);
    std::vector<double> row;
    std::string value;
    while (values >> value) row.push_back(std::stod(value));
    assert(row.size() >= 2);
    rows.push_back(row);
  }
  assert(rows.size() == 45);
  return rows;
}

static bool close(double left, double right)
{
  return std::abs(left - right) <= 1e-6 * std::max({1.0, std::abs(left), std::abs(right)});
}

int main()
{
  const auto com_removed = read_profile("com_removed.out");
  const auto raw_temperature = read_profile("raw_temperature.out");
  const auto compute = read_compute();
  int valid_raw_bins = 0;
  int excluded_raw_bins = 0;
  for (int bin = 0; bin < 5; ++bin) {
    const auto& com = frame(com_removed, 40, bin);
    const auto& raw = frame(raw_temperature, 40, bin);
    assert(static_cast<int>(com[13]) == 1 && static_cast<int>(raw[13]) == 1);
    if (com[8] > 1.0) {
      assert(static_cast<int>(com[14]) == 1);
      assert(close(com[12], 3.0) && std::isfinite(com[11]));
    } else {
      assert(static_cast<int>(com[14]) == 0);
      assert(com[12] == 0.0);
      assert(std::isnan(com[11]));
    }
    if (raw[8] > 0.0) {
      ++valid_raw_bins;
      assert(static_cast<int>(raw[14]) == 1);
      assert(close(raw[12], 3.0 * raw[8]) && std::isfinite(raw[11]));
    } else {
      ++excluded_raw_bins;
      assert(static_cast<int>(raw[14]) == 0);
      assert(raw[12] == 0.0);
      assert(std::isnan(raw[11]));
    }
  }
  assert(valid_raw_bins == 3 && excluded_raw_bins == 2);
  assert(static_cast<int>(frame(raw_temperature, 30, 0)[13]) == 3);
  assert(static_cast<int>(frame(raw_temperature, 40, 0)[13]) == 1);

  const auto& residual = frame(raw_temperature, 40, 0);
  const auto& at_step_40 = compute[39];
  const auto& at_run_end = compute[44];
  assert(close(residual[15], at_step_40[at_step_40.size() - 2]));
  assert(close(residual[16], at_step_40.back()));
  assert(!close(residual[15], at_run_end[at_run_end.size() - 2]) ||
         !close(residual[16], at_run_end.back()));
  return 0;
}
