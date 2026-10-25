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

static std::vector<std::vector<std::string>> rows(const char* filename)
{
  std::ifstream input(filename);
  assert(input.good());
  std::vector<std::vector<std::string>> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<std::string> row;
    std::string token;
    while (fields >> token) row.push_back(token);
    result.push_back(row);
  }
  return result;
}

static bool header_contains(const char* filename, const std::string& needle)
{
  std::ifstream input(filename);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(needle) != std::string::npos) return true;
  }
  return false;
}

static bool close(const double a, const double b)
{
  return std::abs(a - b) <= 1.0e-8 + 5.0e-3 * std::max(std::abs(a), std::abs(b));
}

int main()
{
  const auto flux = rows("m3_plane_flux.out");
  const auto correlation = rows("m3_shc_correlation.out");
  const auto spectrum = rows("m3_shc.out");
  assert(flux.size() == 8 && correlation.size() == 6 && spectrum.size() == 66);
  bool nonzero_flux = false;
  for (const auto& row : flux) {
    assert(row.size() == 7);
    assert(std::isfinite(std::stod(row[3])) && std::isfinite(std::stod(row[4])));
    assert(row[5] == "nan" && row[6] == "nan");
    nonzero_flux = nonzero_flux || std::abs(std::stod(row[3])) > 1.0e-12;
  }
  assert(nonzero_flux);
  bool nonzero_correlation = false;
  for (const auto& row : correlation) {
    assert(row.size() == 6);
    assert(std::isfinite(std::stod(row[4])));
    assert(row[5] == "4" || row[5] == "3");
    nonzero_correlation = nonzero_correlation || std::abs(std::stod(row[4])) > 1.0e-12;
  }
  assert(nonzero_correlation);
  assert(correlation[0][2] == "0" && correlation[1][2] == "1" && correlation[2][2] == "-1");
  assert(correlation[3][0] == "1" && correlation[3][2] == "0");
  assert(header_contains("m3_shc.out", "# compute_interface_shc_spectrum version 3"));
  assert(header_contains("m3_shc.out", "accumulation_start_ps accumulation_end_ps accumulation_samples"));
  for (const auto& row : spectrum) {
    assert(row.size() == 18);
    assert(row[7] == "1" || row[7] == "2");
    assert(row[3] == "4");
    assert(std::isfinite(std::stod(row[16])));
    assert(close(std::stod(row[12]), std::stod(row[14])));
    if (row[7] == "1") {
      assert(row[6] == "4");
      assert(close(std::stod(row[4]), std::stod(row[1])));
      assert(close(std::stod(row[5]), std::stod(row[2])));
      assert(std::isnan(std::stod(row[11])) && std::isnan(std::stod(row[13])));
      assert(std::isnan(std::stod(row[15])) && std::isnan(std::stod(row[17])));
    } else {
      assert(row[6] == "8");
      assert(close(std::stod(row[4]), std::stod(spectrum.front()[1])));
      assert(close(std::stod(row[5]), std::stod(row[2])));
      assert(std::isfinite(std::stod(row[11])) && std::isfinite(std::stod(row[13])));
      assert(std::isfinite(std::stod(row[15])) && std::isfinite(std::stod(row[17])));
    }
  }
  assert(spectrum.front()[9] == "0" && spectrum[32][9] == "300000");
  assert(header_contains("m3_shc.out", "band_limited 1"));
}
