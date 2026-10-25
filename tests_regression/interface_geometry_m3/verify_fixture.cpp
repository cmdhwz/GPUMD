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

struct FixtureSampling
{
  double time_step = 0.0;
  int sample_interval = 0;
};

static FixtureSampling fixture_sampling()
{
  std::ifstream input("run.in");
  assert(input.good());
  FixtureSampling result;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string keyword;
    fields >> keyword;
    if (keyword == "time_step") {
      fields >> result.time_step;
    } else if (keyword == "compute_interface_shc") {
      fields >> result.sample_interval;
    }
  }
  assert(result.time_step > 0.0 && result.sample_interval > 0);
  return result;
}

static bool read_sampling_metadata(
  const char* filename, int& sample_interval, double& dt_ps)
{
  std::ifstream input(filename);
  assert(input.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("sample_interval") == std::string::npos ||
        line.find("dt_ps") == std::string::npos) continue;
    std::replace(line.begin(), line.end(), ';', ' ');
    std::istringstream fields(line);
    std::string token;
    while (fields >> token) {
      if (token == "sample_interval") fields >> sample_interval;
      if (token == "dt_ps") fields >> dt_ps;
    }
    return true;
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
  const auto block_spectrum = rows("m3_shc_blocks.out");
  assert(flux.size() == 8 && correlation.size() == 6 && spectrum.size() == 66);
  assert(block_spectrum.size() == spectrum.size());
  assert(header_contains("m3_shc_blocks.out", "statistics are per complete block only"));
  assert(header_contains("m3_shc_correlation.out", "sample_interval 1; dt_ps"));
  for (const char* filename : {
         "m3_plane_flux.out", "m3_shc_correlation.out", "m3_shc.out", "m3_shc_blocks.out"}) {
    assert(header_contains(filename, "geometry_source m3_shared_geometry.json"));
    assert(header_contains(filename, "plane_id_source sampling_planes_A"));
    assert(header_contains(filename, "# sampling_planes_A 10.9"));
  }
  const FixtureSampling sampling = fixture_sampling();
  const double expected_dt_ps = sampling.time_step * 10.18051 * sampling.sample_interval / 1000.0;
  for (const char* filename : {"m3_shc_correlation.out", "m3_shc_blocks.out"}) {
    int output_sample_interval = 0;
    double output_dt_ps = 0.0;
    assert(read_sampling_metadata(filename, output_sample_interval, output_dt_ps));
    assert(output_sample_interval == sampling.sample_interval);
    assert(output_dt_ps > 0.0);
    assert(std::abs(output_dt_ps - expected_dt_ps) <= 1.0e-15 + 1.0e-12 * expected_dt_ps);
  }
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
  assert(spectrum.size() % 2 == 0);
  const size_t rows_per_block = spectrum.size() / 2;
  for (size_t row_index = 0; row_index < spectrum.size(); ++row_index) {
    const auto& row = spectrum[row_index];
    const auto& block_row = block_spectrum[row_index];
    assert(row.size() == 18);
    assert(block_row.size() == 11);
    assert(block_row[0] == row[0] && block_row[4] == row[8] && block_row[5] == row[9]);
    assert(block_row[1] == row[1] && block_row[2] == row[2] && block_row[3] == row[3]);
    assert(std::isfinite(std::stod(block_row[6])) && std::isfinite(std::stod(block_row[7])));
    assert(std::isfinite(std::stod(block_row[8])) && std::isfinite(std::stod(block_row[9])));
    assert(row[7] == "1" || row[7] == "2");
    assert(row[3] == "4");
    assert(std::isfinite(std::stod(row[16])));
    assert(close(std::stod(row[12]), std::stod(row[14])));
    if (row[7] == "1") {
      assert(row_index < rows_per_block);
      assert(row[6] == "4");
      assert(close(std::stod(row[4]), std::stod(row[1])));
      assert(close(std::stod(row[5]), std::stod(row[2])));
      assert(close(std::stod(block_row[10]), std::stod(row[16])));
      assert(std::isnan(std::stod(row[11])) && std::isnan(std::stod(row[13])));
      assert(std::isnan(std::stod(row[15])) && std::isnan(std::stod(row[17])));
    } else {
      assert(row_index >= rows_per_block);
      assert(row[6] == "8");
      assert(close(std::stod(row[4]), std::stod(spectrum.front()[1])));
      assert(close(std::stod(row[5]), std::stod(row[2])));
      const auto& first_block_row = block_spectrum[row_index - rows_per_block];
      assert(first_block_row[4] == block_row[4] && first_block_row[5] == block_row[5]);
      assert(close(
        std::stod(row[16]),
        0.5 * (std::stod(first_block_row[10]) + std::stod(block_row[10]))));
      assert(std::isfinite(std::stod(row[11])) && std::isfinite(std::stod(row[13])));
      assert(std::isfinite(std::stod(row[15])) && std::isfinite(std::stod(row[17])));
    }
  }
  assert(spectrum.front()[9] == "0" && spectrum[32][9] == "300000");
  assert(header_contains("m3_shc.out", "band_limited 1"));
}
