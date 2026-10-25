#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{
using Row = std::vector<double>;

std::vector<Row> read_rows(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  assert(input.good());
  std::vector<Row> result;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::vector<std::string> tokens;
    std::string token;
    while (fields >> token) tokens.push_back(token);
    Row row;
    for (const std::string& value : tokens) row.push_back(std::stod(value));
    result.push_back(row);
  }
  return result;
}

bool header_contains(const std::string& filename, const std::string& text)
{
  std::ifstream input(filename.c_str());
  assert(input.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.find(text) != std::string::npos) return true;
  }
  return false;
}

std::string read_text(const std::string& filename)
{
  std::ifstream input(filename.c_str());
  assert(input.good());
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

double sample_se(const std::vector<double>& values)
{
  assert(values.size() >= 2);
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double sum = 0.0;
  for (double value : values) sum += (value - mean) * (value - mean);
  return std::sqrt(sum / (values.size() * (values.size() - 1)));
}

double slope(const std::vector<std::pair<double, double>>& points)
{
  assert(points.size() >= 2);
  double mean_t = 0.0;
  double mean_y = 0.0;
  for (const auto& point : points) {
    mean_t += point.first;
    mean_y += point.second;
  }
  mean_t /= points.size();
  mean_y /= points.size();
  double numerator = 0.0;
  double denominator = 0.0;
  for (const auto& point : points) {
    numerator += (point.first - mean_t) * (point.second - mean_y);
    denominator += (point.first - mean_t) * (point.first - mean_t);
  }
  assert(denominator > 0.0);
  return numerator / denominator;
}

struct FixtureConfig
{
  double time_step_fs = 0.0;
  int sample_interval = 0;
  int correlation_samples = 0;
  int block_samples = 0;
  int num_omega = 0;
  double max_omega_rad_ps = 0.0;
  int run_steps = 0;
};

FixtureConfig read_fixture_config()
{
  std::ifstream input("run.in");
  assert(input.good());
  FixtureConfig config;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string keyword;
    fields >> keyword;
    if (keyword == "time_step") fields >> config.time_step_fs;
    else if (keyword == "compute_interface_shc") {
      fields >> config.sample_interval >> config.correlation_samples >> config.block_samples
             >> config.num_omega >> config.max_omega_rad_ps;
    } else if (keyword == "run") {
      fields >> config.run_steps;
    }
  }
  assert(config.time_step_fs > 0.0 && config.sample_interval > 0);
  assert(config.correlation_samples > 0 && config.block_samples >= 2 * config.correlation_samples);
  assert(config.num_omega >= 2 && config.max_omega_rad_ps > 0.0 && config.run_steps > 0);
  return config;
}

double header_value(const std::string& filename, const std::string& key)
{
  std::ifstream input(filename.c_str());
  assert(input.good());
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] != '#') continue;
    std::istringstream fields(line);
    std::string token;
    while (fields >> token) {
      if (token != key) continue;
      std::string value;
      fields >> value;
      if (!value.empty() && value.back() == ';') value.pop_back();
      return std::stod(value);
    }
  }
  assert(false && "missing numeric output metadata");
  return 0.0;
}

double comparison_tolerance(const double a, const double b)
{
  const double scale = std::max({std::abs(a), std::abs(b), 1.0});
  return 5.0e-7 + 2.0e-8 * scale;
}

void verify_geometry_and_headers()
{
  const std::string json = read_text("native_shared_geometry.json");
  assert(json.find("\"algorithm_version\": 2") != std::string::npos);
  assert(json.find("\"interfaces_A\": []") != std::string::npos);
  assert(json.find("\"sampling_planes_A\": [7.134") != std::string::npos);
  assert(json.find(", 14.268") != std::string::npos);
  assert(json.find("\"reference_positions_role\": \"geometry_mapping_only\"") != std::string::npos);
  assert(json.find("\"equilibrium_reference\": \"external_required\"") != std::string::npos);
  assert(header_contains("native_shared_groups.dat", "reference_positions_role geometry_mapping_only"));
  assert(header_contains("native_shc_plane_flux.out", "# sampling_planes_A 7.134"));
  assert(header_contains("native_shc_plane_flux.out", "14.268"));
  for (const char* filename : {
         "native_shc_plane_flux.out", "native_shc_shc_correlation.out",
         "native_shc_shc.out", "native_shc_shc_blocks.out"}) {
    assert(header_contains(filename, "geometry_source native_shared_geometry.json"));
    assert(header_contains(filename, "plane_id_source sampling_planes_A"));
    assert(header_contains(filename, "# sampling_planes_A 7.134"));
    assert(header_contains(filename, "14.268"));
  }
}

void verify_spectrum_rows(
  const std::vector<Row>& flux,
  const FixtureConfig& config,
  const int plane_count)
{
  const auto correlation = read_rows("native_shc_shc_correlation.out");
  const auto blocks = read_rows("native_shc_shc_blocks.out");
  assert(!flux.empty() && !correlation.empty() && !blocks.empty());
  const int sample_count = config.run_steps / config.sample_interval;
  assert(config.run_steps % config.sample_interval == 0);
  assert(sample_count % config.block_samples == 0);
  const int block_count = sample_count / config.block_samples;
  assert(flux.size() == static_cast<size_t>(sample_count) * plane_count);
  assert(blocks.size() == static_cast<size_t>(block_count) * plane_count * config.num_omega);
  assert(correlation.size() == static_cast<size_t>(block_count) * plane_count *
                               (2 * config.correlation_samples - 1));

  const double area = 21.402 * 21.402;
  const double flux_conversion = 1.602176634e13 / area;
  const double expected_dt_ps = config.time_step_fs * config.sample_interval / 1000.0;
  const double expected_nyquist = 3.14159265358979323846 / expected_dt_ps;
  assert(std::abs(header_value("native_shc_shc_blocks.out", "dt_ps") - expected_dt_ps) <=
    1.0e-15 + 1.0e-10 * expected_dt_ps);
  assert(static_cast<int>(header_value("native_shc_shc_blocks.out", "numomega")) == config.num_omega);
  assert(std::abs(header_value("native_shc_shc_blocks.out", "maxomega_rad_ps") - expected_nyquist) <=
    1.0e-6 + 1.0e-10 * expected_nyquist);
  assert(std::abs(config.max_omega_rad_ps - expected_nyquist) <= 1.0e-10 * expected_nyquist);

  std::map<std::pair<int, int>, double> c0;
  for (size_t row_id = 0; row_id < correlation.size(); ++row_id) {
    const Row& row = correlation[row_id];
    assert(row.size() == 6);
    const int rows_per_plane = 2 * config.correlation_samples - 1;
    const int expected_block = static_cast<int>(row_id / (plane_count * rows_per_plane));
    const int plane_row = static_cast<int>(row_id % (plane_count * rows_per_plane));
    const int expected_plane = plane_row / rows_per_plane;
    const int lag_row = plane_row % rows_per_plane;
    const int expected_lag = lag_row == 0 ? 0 :
      ((lag_row % 2) ? (lag_row + 1) / 2 : -lag_row / 2);
    assert(static_cast<int>(row[0]) == expected_block && static_cast<int>(row[1]) == expected_plane);
    assert(static_cast<int>(row[2]) == expected_lag);
    assert(static_cast<int>(row[5]) == config.block_samples - std::abs(expected_lag));
    assert(std::abs(row[3] - expected_lag * expected_dt_ps) <= 1.0e-12 + 1.0e-9 * expected_dt_ps);
    if (row[2] == 0.0) {
      c0[std::make_pair(static_cast<int>(row[0]), static_cast<int>(row[1]))] = row[4];
    }
  }

  for (int sample = 0; sample < sample_count; ++sample) {
    for (int plane = 0; plane < plane_count; ++plane) {
      const Row& row = flux[static_cast<size_t>(sample) * plane_count + plane];
      assert(row.size() == 7);
      const int expected_step = (sample + 1) * config.sample_interval;
      const double expected_time = expected_step * config.time_step_fs / 1000.0;
      assert(static_cast<int>(row[0]) == expected_step && static_cast<int>(row[2]) == plane);
      assert(std::abs(row[1] - expected_time) <= 1.0e-12 + 1.0e-9 * expected_time);
    }
  }

  for (int block = 0; block < block_count; ++block) {
    for (int plane = 0; plane < plane_count; ++plane) {
      const size_t base = (static_cast<size_t>(block) * plane_count + plane) * config.num_omega;
      const Row& first = blocks[base];
      assert(first.size() == 11 && static_cast<int>(first[0]) == block);
      assert(static_cast<int>(first[3]) == config.block_samples && static_cast<int>(first[4]) == plane);
      const int first_sample = block * config.block_samples;
      const int last_sample = first_sample + config.block_samples - 1;
      const double expected_start =
        (first_sample + 1) * config.sample_interval * config.time_step_fs / 1000.0;
      const double expected_end =
        (last_sample + 1) * config.sample_interval * config.time_step_fs / 1000.0;
      assert(std::abs(first[1] - expected_start) <= 1.0e-12 + 1.0e-9 * expected_start);
      assert(std::abs(first[2] - expected_end) <= 1.0e-12 + 1.0e-9 * expected_end);
      assert(first[5] == 0.0);
      assert(std::abs(blocks[base + config.num_omega - 1][5] - expected_nyquist) <=
        1.0e-6 + 1.0e-10 * expected_nyquist);
      double flux_sum = 0.0;
      for (int sample = 0; sample < config.block_samples; ++sample) {
        const Row& row = flux[static_cast<size_t>(first_sample + sample) * plane_count + plane];
        assert(row.size() == 7 && static_cast<int>(row[2]) == plane);
        flux_sum += row[3];
      }
      const double mean_flux = flux_sum / config.block_samples * flux_conversion;
      const auto c0_it = c0.find(std::make_pair(static_cast<int>(block), plane));
      assert(c0_it != c0.end());
      assert(std::abs(first[6] - c0_it->second) <= comparison_tolerance(first[6], c0_it->second));
      assert(std::abs(first[7] - mean_flux) <= comparison_tolerance(first[7], mean_flux));
      assert(std::abs(first[6] - first[7]) <= comparison_tolerance(first[6], first[7]));

      double grid_integral = 0.0;
      double absolute_trapezoid = 0.0;
      for (int omega = 0; omega < config.num_omega; ++omega) {
        const Row& row = blocks[base + omega];
        assert(row.size() == 11 && static_cast<int>(row[4]) == plane);
        assert(static_cast<int>(row[0]) == block);
        const double expected_omega = expected_nyquist * omega / (config.num_omega - 1);
        assert(std::abs(row[5] - expected_omega) <= 1.0e-7 + 1.0e-10 * expected_nyquist);
        assert(std::isfinite(row[5]) && std::isfinite(row[10]));
        if (omega > 0) {
          const Row& previous = blocks[base + omega - 1];
          const double delta_omega = row[5] - previous[5];
          assert(std::isfinite(delta_omega) && delta_omega > 0.0);
          grid_integral += 0.5 * (previous[10] + row[10]) * delta_omega;
          absolute_trapezoid += 0.5 * (std::abs(previous[10]) + std::abs(row[10])) * delta_omega;
        }
      }
      assert(std::isfinite(grid_integral) && std::isfinite(absolute_trapezoid));
      // Output rows use 12 significant digits; scale the integration tolerance by the
      // absolute trapezoid so cancellation in the signed integral does not hide quantization.
      const double spectrum_quantization_tolerance = 5.0e-11 * absolute_trapezoid;
      assert(std::abs(first[9] - grid_integral) <= spectrum_quantization_tolerance);
      assert(std::abs(grid_integral - first[6]) <= spectrum_quantization_tolerance);
      assert(std::abs(grid_integral - first[7]) <= spectrum_quantization_tolerance);
      assert(std::abs(first[8] - first[6]) <= comparison_tolerance(first[8], first[6]));
    }
  }
}

void verify_nve_balance()
{
  const auto profile = read_rows("native_profile.out");
  const auto flux = read_rows("native_shc_plane_flux.out");
  assert(profile.size() == 320 * 6 && flux.size() == 320 * 2);
  std::map<int, double> region_energy;
  std::map<int, double> region_time;
  for (const Row& row : profile) {
    assert(row.size() == 19);
    const int step = static_cast<int>(row[0]);
    const int bin = static_cast<int>(row[3]);
    assert(row[9] == 0.0); // homogeneous diamond has no material interface
    if (bin == 2 || bin == 3) {
      region_energy[step] += row[11] + row[12];
      region_time[step] = row[1];
    }
  }
  assert(region_energy.size() == 320 && region_time.size() == 320);
  assert(region_energy.begin()->first == 1 && region_energy.rbegin()->first == 320);
  std::map<int, std::array<double, 2>> plane_flux;
  std::map<int, double> flux_time;
  for (const Row& row : flux) {
    assert(row.size() == 7);
    const int step = static_cast<int>(row[0]);
    const int plane = static_cast<int>(row[2]);
    assert(plane == 0 || plane == 1);
    plane_flux[step][plane] = row[3];
    flux_time[step] = row[1];
  }
  assert(plane_flux.size() == 320 && flux_time.size() == 320);
  assert(plane_flux.begin()->first == 1 && plane_flux.rbegin()->first == 320);
  const int first_step = region_energy.begin()->first;
  const int last_step = region_energy.rbegin()->first;
  double integrated = 0.0;
  double activity = 0.0;
  for (int step = first_step; step <= last_step; ++step) {
    const auto& q = plane_flux.at(step);
    assert(region_time.at(step) == flux_time.at(step));
    activity += std::abs(q[0]) + std::abs(q[1]);
    if (step == last_step) break;
    const double left_now = q[0] - q[1];
    const auto& q_next = plane_flux.at(step + 1);
    const double left_next = q_next[0] - q_next[1];
    const double dt_ps = flux_time.at(step + 1) - flux_time.at(step);
    assert(dt_ps > 0.0);
    integrated += 0.5 * (left_now + left_next) * dt_ps;
  }
  const double delta_energy = region_energy.at(last_step) - region_energy.at(first_step);
  const double scale = std::max(std::abs(delta_energy), std::abs(integrated));
  const double residual = delta_energy - integrated;
  std::printf(
    "NVE slab: deltaE=%.9g eV, integral(Q_left-Q_right)=%.9g eV, residual=%.9g eV, flux activity=%.9g eV/ps\n",
    delta_energy, integrated, residual, activity);
  assert(activity > 1.0e-6 && scale > 1.0e-5);
  assert(std::abs(residual) <= 0.05 * scale + 1.0e-5);
}

void verify_geometry_split()
{
  const std::string json = read_text("native_shared_geometry.json");
  assert(json.find("\"interfaces_A\": [5]") != std::string::npos);
  assert(json.find("\"sampling_planes_A\": [16]") != std::string::npos);
  const auto profile = read_rows("native_profile.out");
  assert(profile.size() == 8 * 6);
  for (int step = 0; step < 8; ++step) {
    for (int bin = 0; bin < 6; ++bin) {
      const Row& row = profile[static_cast<size_t>(step) * 6 + bin];
      assert(row.size() == 19 && static_cast<int>(row[0]) == step + 1);
      assert(static_cast<int>(row[3]) == bin);
      assert(static_cast<int>(row[9]) == (bin == 1 ? 1 : 0));
    }
  }
  for (const char* filename : {
         "native_shc_plane_flux.out", "native_shc_shc_correlation.out",
         "native_shc_shc.out", "native_shc_shc_blocks.out"}) {
    assert(header_contains(filename, "geometry_source native_shared_geometry.json"));
    assert(header_contains(filename, "plane_id_source sampling_planes_A"));
    assert(header_contains(filename, "# sampling_planes_A 16"));
  }
  assert(!header_contains("native_shc_plane_flux.out", "# sampling_planes_A 5"));
  verify_spectrum_rows(read_rows("native_shc_plane_flux.out"), read_fixture_config(), 1);
}

void verify_geometry_legacy()
{
  const std::string json = read_text("native_shared_geometry.json");
  assert(json.find("\"interfaces_A\": [3, 7]") != std::string::npos);
  assert(json.find("\"sampling_planes_A\": [3, 7]") != std::string::npos);
  const auto profile = read_rows("native_profile.out");
  assert(profile.size() == 8 * 6);
  for (int step = 0; step < 8; ++step) {
    for (int bin = 0; bin < 6; ++bin) {
      const Row& row = profile[static_cast<size_t>(step) * 6 + bin];
      assert(row.size() == 19 && static_cast<int>(row[0]) == step + 1);
      assert(static_cast<int>(row[3]) == bin);
      assert(static_cast<int>(row[9]) == (bin == 0 || bin == 1 ? 1 : 0));
    }
  }
  for (const char* filename : {
         "native_shc_plane_flux.out", "native_shc_shc_correlation.out",
         "native_shc_shc.out", "native_shc_shc_blocks.out"}) {
    assert(header_contains(filename, "geometry_source native_shared_geometry.json"));
    assert(header_contains(filename, "plane_id_source sampling_planes_A"));
    assert(header_contains(filename, "# sampling_planes_A 3 7"));
  }
  verify_spectrum_rows(read_rows("native_shc_plane_flux.out"), read_fixture_config(), 2);
}

void verify_heat_lan_balance()
{
  const auto profile = read_rows("native_profile.out");
  const auto flux = read_rows("native_shc_plane_flux.out");
  assert(profile.size() == 20000 * 6 && flux.size() == 20000 * 2);
  std::map<int, std::array<double, 2>> bath;
  std::map<int, double> time;
  std::map<int, double> system_energy;
  for (const Row& row : profile) {
    assert(row.size() == 19);
    assert(row[9] == 0.0);
    const int step = static_cast<int>(row[0]);
    assert(std::isfinite(row[17]) && std::isfinite(row[18]));
    if (!bath.count(step)) {
      bath[step] = {row[17], row[18]};
      time[step] = row[1];
    }
    assert(row[17] == bath[step][0] && row[18] == bath[step][1]);
    system_energy[step] += row[11] + row[12];
  }
  assert(bath.size() == 20000 && system_energy.size() == 20000);
  assert(bath.begin()->first == 1 && bath.rbegin()->first == 20000);
  std::map<int, std::array<double, 2>> plane_flux;
  for (const Row& row : flux) {
    assert(row.size() == 7);
    const int step = static_cast<int>(row[0]);
    const int plane = static_cast<int>(row[2]);
    assert(plane == 0 || plane == 1);
    plane_flux[step][plane] = row[3];
  }
  assert(plane_flux.size() == 20000);
  assert(plane_flux.begin()->first == 1 && plane_flux.rbegin()->first == 20000);

  struct HeatBlock
  {
    double source_power;
    double sink_power;
    double bath_power;
    double plane_power[2];
    double system_energy_slope;
  };
  constexpr int steady_start_step = 10001; // first production sample after 10,000 warm-up steps
  constexpr int steady_end_step = 20000;
  constexpr int block_size = 500;
  std::vector<HeatBlock> heat_blocks;
  for (int start = steady_start_step; start <= steady_end_step; start += block_size) {
    std::vector<std::pair<double, double>> source_points;
    std::vector<std::pair<double, double>> sink_points;
    std::vector<std::pair<double, double>> energy_points;
    double plane_sums[2] = {0.0, 0.0};
    for (int step = start; step < start + block_size; ++step) {
      source_points.emplace_back(time.at(step), bath.at(step)[0]);
      sink_points.emplace_back(time.at(step), bath.at(step)[1]);
      energy_points.emplace_back(time.at(step), system_energy.at(step));
      plane_sums[0] += plane_flux.at(step)[0];
      plane_sums[1] += plane_flux.at(step)[1];
    }
    HeatBlock block;
    block.source_power = -slope(source_points);
    block.sink_power = -slope(sink_points);
    block.bath_power = 0.5 * (block.source_power - block.sink_power);
    block.plane_power[0] = plane_sums[0] / block_size;
    block.plane_power[1] = plane_sums[1] / block_size;
    block.system_energy_slope = slope(energy_points);
    heat_blocks.push_back(block);
  }
  assert(heat_blocks.size() == 20);

  const auto summarize = [](const std::vector<double>& values) {
    const double average = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    return std::make_pair(average, sample_se(values));
  };
  const auto append_values = [](const std::vector<HeatBlock>& blocks, const int begin,
                                 const int end, const int quantity) {
    std::vector<double> values;
    for (int i = begin; i < end; ++i) {
      const HeatBlock& block = blocks[static_cast<size_t>(i)];
      if (quantity == 0) values.push_back(block.source_power);
      else if (quantity == 1) values.push_back(block.sink_power);
      else if (quantity == 2) values.push_back(block.bath_power);
      else if (quantity == 3) values.push_back(block.plane_power[0]);
      else if (quantity == 4) values.push_back(block.plane_power[1]);
      else values.push_back(block.system_energy_slope);
    }
    return values;
  };
  bool passed = true;
  const auto check_signal = [&](const char* name, const std::vector<double>& values, const int sign) {
    const auto stats = summarize(values);
    const bool enough_statistics = stats.second <= 0.25 * std::abs(stats.first);
    const bool correct_sign = sign * stats.first > 0.0;
    const bool resolved = sign * stats.first > 0.0 &&
      std::abs(stats.first) >= 2.0 * stats.second && stats.second <= 0.25 * std::abs(stats.first);
    const char* status = !enough_statistics ? "statistics_insufficient" :
      (!correct_sign ? "wrong_sign_or_no_signal" : (resolved ? "resolved" : "unresolved_signal"));
    std::printf("%s mean %.9g eV/ps, block-SE %.9g, status=%s\n",
      name, stats.first, stats.second, status);
    passed = passed && resolved;
    return stats;
  };
  const auto check_paired_residual = [&](const char* name, const std::vector<double>& residuals,
                                          const double scale) {
    const auto stats = summarize(residuals);
    const bool enough_statistics = stats.second <= 0.10 * scale;
    const bool matches = std::abs(stats.first) <= 0.20 * scale + stats.second;
    const char* status = !enough_statistics ? "statistics_insufficient" :
      (matches ? "within_residual_budget" : "residual_mismatch");
    std::printf("%s paired residual %.9g eV/ps, paired block-SE %.9g, SE budget %.9g, status=%s\n",
      name, stats.first, stats.second, 0.10 * scale, status);
    passed = passed && enough_statistics && matches;
    return stats;
  };
  const auto check_half_stability = [&](const char* name, const std::vector<double>& first_half,
                                         const std::vector<double>& second_half) {
    const auto first = summarize(first_half);
    const auto second = summarize(second_half);
    const double scale = std::max({std::abs(first.first), std::abs(second.first), 1.0e-12});
    const double combined_se = std::sqrt(first.second * first.second + second.second * second.second);
    const bool enough_statistics = combined_se <= 0.10 * scale;
    const bool stable = std::abs(first.first - second.first) <= 0.20 * scale + combined_se;
    std::printf("%s half means %.9g / %.9g eV/ps, half-SE %.9g / %.9g, stable=%s\n",
      name, first.first, second.first, first.second, second.second,
      enough_statistics && stable ? "yes" : "no");
    passed = passed && enough_statistics && stable;
  };

  for (int half = 0; half < 2; ++half) {
    const int begin = half * 10;
    const int end = begin + 10;
    const std::vector<double> source = append_values(heat_blocks, begin, end, 0);
    const std::vector<double> sink = append_values(heat_blocks, begin, end, 1);
    const std::vector<double> bath_power = append_values(heat_blocks, begin, end, 2);
    const std::vector<double> plane0 = append_values(heat_blocks, begin, end, 3);
    const std::vector<double> plane1 = append_values(heat_blocks, begin, end, 4);
    const std::vector<double> energy_slopes = append_values(heat_blocks, begin, end, 5);
    const auto source_stats = check_signal("P_in_source", source, +1);
    const auto sink_stats = check_signal("P_in_sink", sink, -1);
    const auto bath_stats = check_signal("Q_bath", bath_power, +1);
    const auto plane0_stats = check_signal("Q_plane0", plane0, +1);
    const auto plane1_stats = check_signal("Q_plane1", plane1, +1);
    const double source_sink_scale = std::max(std::abs(source_stats.first), std::abs(sink_stats.first));
    std::vector<double> source_sink_residuals;
    std::vector<double> plane_residuals[3];
    for (int i = begin; i < end; ++i) {
      const HeatBlock& block = heat_blocks[static_cast<size_t>(i)];
      source_sink_residuals.push_back(block.source_power + block.sink_power);
      plane_residuals[0].push_back(block.plane_power[0] - block.plane_power[1]);
      plane_residuals[1].push_back(block.plane_power[0] - block.bath_power);
      plane_residuals[2].push_back(block.plane_power[1] - block.bath_power);
    }
    check_paired_residual("P_in_source + P_in_sink", source_sink_residuals, source_sink_scale);
    check_paired_residual("Q_plane0 - Q_plane1", plane_residuals[0],
      std::max(std::abs(plane0_stats.first), std::abs(plane1_stats.first)));
    check_paired_residual("Q_plane0 - Q_bath", plane_residuals[1],
      std::max(std::abs(plane0_stats.first), std::abs(bath_stats.first)));
    check_paired_residual("Q_plane1 - Q_bath", plane_residuals[2],
      std::max(std::abs(plane1_stats.first), std::abs(bath_stats.first)));
    const auto energy_stats = summarize(energy_slopes);
    const double energy_budget = 0.10 * std::abs(bath_stats.first);
    const bool energy_stable = energy_stats.second <= energy_budget &&
      std::abs(energy_stats.first) <= 0.20 * std::abs(bath_stats.first) + energy_stats.second;
    const char* energy_status = energy_stats.second > energy_budget
      ? "statistics_insufficient" : (energy_stable ? "stable" : "energy_drift");
    std::printf("half %d system-energy slope %.9g eV/ps, block-SE %.9g, budget %.9g, status=%s\n",
      half + 1, energy_stats.first, energy_stats.second, energy_budget, energy_status);
    passed = passed && energy_stable;
  }

  for (int quantity = 2; quantity <= 4; ++quantity) {
    check_half_stability(
      quantity == 2 ? "Q_bath" : (quantity == 3 ? "Q_plane0" : "Q_plane1"),
      append_values(heat_blocks, 0, 10, quantity), append_values(heat_blocks, 10, 20, quantity));
  }
  const auto energy_first = summarize(append_values(heat_blocks, 0, 10, 5));
  const auto energy_second = summarize(append_values(heat_blocks, 10, 20, 5));
  const auto bath_first = summarize(append_values(heat_blocks, 0, 10, 2));
  const auto bath_second = summarize(append_values(heat_blocks, 10, 20, 2));
  const double energy_scale = std::max(std::abs(bath_first.first), std::abs(bath_second.first));
  const double energy_se = std::sqrt(energy_first.second * energy_first.second +
                                     energy_second.second * energy_second.second);
  const bool energy_half_stable = energy_se <= 0.10 * energy_scale &&
    std::abs(energy_first.first - energy_second.first) <= 0.20 * energy_scale + energy_se;
  std::printf(
    "system-energy slope half means %.9g / %.9g eV/ps, combined block-SE %.9g, budget %.9g, status=%s\n",
    energy_first.first, energy_second.first, energy_se, 0.10 * energy_scale,
    energy_se > 0.10 * energy_scale ? "statistics_insufficient" :
      (energy_half_stable ? "stable" : "drift_changed_between_halves"));
  passed = passed && energy_half_stable;
  std::printf(
    "heat_lan candidate production window is steps 10001-20000 after 10000 warm-up steps; "
    "adjacent blocks are variability checks, not independent confidence intervals.\n");
  assert(passed);
}
} // namespace

int main(int argc, char** argv)
{
  assert(argc == 2);
  const std::string mode = argv[1];
  const auto flux = read_rows("native_shc_plane_flux.out");
  const FixtureConfig config = read_fixture_config();
  if (mode == "nve") {
    verify_geometry_and_headers();
    verify_spectrum_rows(flux, config, 2);
    verify_nve_balance();
  } else if (mode == "heat_lan") {
    verify_geometry_and_headers();
    verify_spectrum_rows(flux, config, 2);
    verify_heat_lan_balance();
  } else if (mode == "geometry_split") {
    verify_geometry_split();
  } else if (mode == "geometry_legacy") {
    verify_geometry_legacy();
  }
  else assert(false);
  return 0;
}
