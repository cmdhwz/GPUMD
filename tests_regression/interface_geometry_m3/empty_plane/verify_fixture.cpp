#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
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

int main()
{
  const auto flux = rows("empty_plane_flux.out");
  const auto correlation = rows("empty_shc_correlation.out");
  const auto spectrum = rows("empty_shc.out");
  const auto block_spectrum = rows("empty_shc_blocks.out");
  assert(flux.size() == 2 && correlation.size() == 2 && spectrum.size() == 4);
  assert(block_spectrum.size() == spectrum.size());
  for (const auto& row : flux) {
    assert(row.size() == 7);
    assert(std::stod(row[3]) == 0.0 && std::stod(row[4]) == 0.0);
  }
  for (const auto& row : correlation) {
    assert(row.size() == 6 && std::stod(row[4]) == 0.0 && row[5] == "2");
  }
  assert(header_contains("empty_shc.out", "# compute_interface_shc_spectrum version 3"));
  assert(header_contains("empty_shc.out", "accumulation_start_ps accumulation_end_ps accumulation_samples"));
  for (const char* filename : {
         "empty_plane_flux.out", "empty_shc_correlation.out", "empty_shc.out", "empty_shc_blocks.out"}) {
    assert(header_contains(filename, "geometry_source empty_shared_geometry.json"));
    assert(header_contains(filename, "plane_id_source sampling_planes_A"));
    assert(header_contains(filename, "# sampling_planes_A 15"));
  }
  for (size_t row_index = 0; row_index < spectrum.size(); ++row_index) {
    const auto& row = spectrum[row_index];
    const auto& block_row = block_spectrum[row_index];
    assert(row.size() == 18);
    assert(block_row.size() == 11);
    assert(block_row[0] == row[0] && block_row[4] == row[8] && block_row[5] == row[9]);
    assert(block_row[6] == "0" && block_row[7] == "0");
    assert(block_row[8] == "0" && block_row[9] == "0" && block_row[10] == "0");
    assert(row[3] == "2" && row[6] == "2" && row[7] == "1" && row[8] == "0");
    assert(std::stod(row[10]) == 0.0);
    assert(std::stod(row[12]) == 0.0);
    assert(std::stod(row[14]) == 0.0);
    assert(std::stod(row[16]) == 0.0);
    assert(std::isnan(std::stod(row[11])));
    assert(std::isnan(std::stod(row[13])));
    assert(std::isnan(std::stod(row[15])));
    assert(std::isnan(std::stod(row[17])));
    assert(std::stod(row[1]) == std::stod(row[4]));
    assert(std::stod(row[2]) == std::stod(row[5]));
  }
  return 0;
}
