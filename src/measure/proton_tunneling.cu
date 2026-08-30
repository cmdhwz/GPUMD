/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Track centroid O-H-O geometry, local proton-state bias, and persistent state changes.

This is deliberately an observer. It does not alter the NEP/qNEP energy, force, charge, or
heat-current paths. The sparse transfer stream can be reconstructed into defect loops offline.
--------------------------------------------------------------------------------------------------*/

#include "proton_tunneling.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{
unsigned long long make_bond_key(const int oxygen_low, const int oxygen_high)
{
  return (static_cast<unsigned long long>(static_cast<unsigned int>(oxygen_low)) << 32) |
    static_cast<unsigned int>(oxygen_high);
}

void decode_bond_key(
  const unsigned long long key,
  int& oxygen_low,
  int& oxygen_high)
{
  oxygen_low = static_cast<int>(key >> 32);
  oxygen_high = static_cast<int>(key & 0xffffffffULL);
}
}

Proton_Tunneling::Proton_Tunneling(
  const char** param,
  const int num_param,
  const Atom& atom)
{
  parse(param, num_param, atom);
  property_name = "compute_proton_tunneling";
}

void Proton_Tunneling::parse(
  const char** param,
  const int num_param,
  const Atom& atom)
{
  printf("Compute centroid proton tunneling observer.\n");

  if (num_param != 8 && num_param != 10) {
    PRINT_INPUT_ERROR(
      "compute_proton_tunneling should have 7 parameters, with optional O and H symbols.");
  }
  if (!is_valid_int(param[1], &sample_interval_) || sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling sample interval should be a positive integer.");
  }
  if (!is_valid_int(param[2], &window_samples_) || window_samples_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling window size should be a positive integer.");
  }
  if (!is_valid_real(param[3], &delta_cutoff_) || delta_cutoff_ <= 0.0) {
    PRINT_INPUT_ERROR("proton tunneling delta cutoff should be a positive number.");
  }
  if (!is_valid_int(param[4], &hold_samples_) || hold_samples_ <= 0) {
    PRINT_INPUT_ERROR("proton tunneling hold samples should be a positive integer.");
  }
  if (!is_valid_real(param[5], &dOO_min_) || !is_valid_real(param[6], &dOO_max_) ||
      dOO_min_ <= 0.0 || dOO_max_ <= dOO_min_) {
    PRINT_INPUT_ERROR("proton tunneling O-O distance range is invalid.");
  }
  if (!is_valid_real(param[7], &rperp_max_) || rperp_max_ <= 0.0) {
    PRINT_INPUT_ERROR("proton tunneling perpendicular-distance cutoff should be positive.");
  }
  if (num_param == 10) {
    oxygen_symbol_ = param[8];
    hydrogen_symbol_ = param[9];
  }
  if (oxygen_symbol_.empty() || hydrogen_symbol_.empty() || oxygen_symbol_ == hydrogen_symbol_) {
    PRINT_INPUT_ERROR("proton tunneling O and H symbols should be different and non-empty.");
  }

  for (int i = 0; i < atom.number_of_atoms; ++i) {
    if (atom.cpu_atom_symbol[i] == oxygen_symbol_)
      oxygen_indices_.push_back(i);
    if (atom.cpu_atom_symbol[i] == hydrogen_symbol_)
      hydrogen_indices_.push_back(i);
  }
  if (oxygen_indices_.empty() || hydrogen_indices_.empty()) {
    PRINT_INPUT_ERROR("compute_proton_tunneling could not find the requested O and H species.");
  }

  printf("    sample interval is %d steps.\n", sample_interval_);
  printf("    window size is %d sampled frames.\n", window_samples_);
  printf("    delta cutoff is %.6f Angstrom.\n", delta_cutoff_);
  printf("    state hold time is %d sampled frames.\n", hold_samples_);
  printf("    O-O range is %.6f to %.6f Angstrom.\n", dOO_min_, dOO_max_);
  printf("    perpendicular-distance cutoff is %.6f Angstrom.\n", rperp_max_);
  printf("    using oxygen symbol %s and hydrogen symbol %s.\n",
    oxygen_symbol_.c_str(), hydrogen_symbol_.c_str());
}

void Proton_Tunneling::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  (void)number_of_steps;
  (void)integrate;
  (void)group;
  (void)box;
  (void)force;

  number_of_atoms_ = atom.number_of_atoms;
  time_step_ = time_step;
  cpu_position_.resize(number_of_atoms_ * 3);
  hydrogen_count_.resize(number_of_atoms_, 0);
  hydrogen_states_.resize(hydrogen_indices_.size());

  bias_file_ = my_fopen("proton_bias.out", "a");
  transfer_file_ = my_fopen("proton_transfer.out", "a");
  attempt_file_ = my_fopen("proton_attempt.out", "a");
  edge_window_file_ = my_fopen("proton_edge_window.out", "a");
  bond_file_ = my_fopen("proton_bond.out", "a");

  fprintf(bias_file_,
    "# compute_proton_tunneling %d %d %.10e %d %.10e %.10e %.10e %s %s\n",
    sample_interval_, window_samples_, delta_cutoff_, hold_samples_, dOO_min_, dOO_max_,
    rperp_max_, oxygen_symbol_.c_str(), hydrogen_symbol_.c_str());
  fprintf(bias_file_,
    "# columns time_fs B_mean F_A_gt_0.2 F_A_gt_0.4 mean_abs_DeltaF_over_kBT "
    "flip_rate_per_ps active_bonds positive_defects negative_defects valid_pairs_per_frame\n");

  fprintf(transfer_file_,
    "# columns event_id time_start_fs time_confirm_fs H_id O_from O_to O_pair_low O_pair_high "
    "dx dy dz delta_start delta_confirm nH_from nH_to\n");
  fprintf(attempt_file_,
    "# columns attempt_id time_start_fs time_end_fs H_id O_low O_high O_from O_target outcome "
    "delta_start min_abs_delta delta_end\n");
  fprintf(edge_window_file_,
    "# columns window_id time_start_fs time_end_fs O_low O_high geometry_occupancy "
    "n_plus n_minus n_deadband A abs_A DeltaF_over_kBT attempts successes returns geometry_lost "
    "success_probability mean_delta mean_abs_delta mean_dOO mean_rperp\n");
  fprintf(bond_file_,
    "# columns O_pair_low O_pair_high geometry_samples n_plus n_minus transitions "
    "A abs_A mean_abs_delta\n");

  fflush(bias_file_);
  fflush(transfer_file_);
  fflush(attempt_file_);
  fflush(edge_window_file_);
  fflush(bond_file_);
  initialized_ = true;
}

bool Proton_Tunneling::find_geometry(
  const int hydrogen,
  const Box& box,
  GeometryResult& geometry) const
{
  geometry = GeometryResult();

  int first_oxygen = -1;
  int second_oxygen = -1;
  double first_distance_square = std::numeric_limits<double>::max();
  double second_distance_square = std::numeric_limits<double>::max();

  const double hx = cpu_position_[hydrogen];
  const double hy = cpu_position_[hydrogen + number_of_atoms_];
  const double hz = cpu_position_[hydrogen + 2 * number_of_atoms_];

  for (const int oxygen : oxygen_indices_) {
    double dx = cpu_position_[oxygen] - hx;
    double dy = cpu_position_[oxygen + number_of_atoms_] - hy;
    double dz = cpu_position_[oxygen + 2 * number_of_atoms_] - hz;
    apply_mic(box, dx, dy, dz);
    const double distance_square = dx * dx + dy * dy + dz * dz;
    if (distance_square < first_distance_square) {
      second_oxygen = first_oxygen;
      second_distance_square = first_distance_square;
      first_oxygen = oxygen;
      first_distance_square = distance_square;
    } else if (distance_square < second_distance_square) {
      second_oxygen = oxygen;
      second_distance_square = distance_square;
    }
  }

  if (first_oxygen < 0 || second_oxygen < 0)
    return false;

  geometry.nearest_oxygen = first_oxygen;
  geometry.oxygen_low = std::min(first_oxygen, second_oxygen);
  geometry.oxygen_high = std::max(first_oxygen, second_oxygen);
  const double low_oxygen_distance = (geometry.oxygen_low == first_oxygen)
    ? std::sqrt(first_distance_square)
    : std::sqrt(second_distance_square);
  const double high_oxygen_distance = (geometry.oxygen_high == first_oxygen)
    ? std::sqrt(first_distance_square)
    : std::sqrt(second_distance_square);
  geometry.delta = low_oxygen_distance - high_oxygen_distance;

  double ox = cpu_position_[geometry.oxygen_high] - cpu_position_[geometry.oxygen_low];
  double oy = cpu_position_[geometry.oxygen_high + number_of_atoms_] -
    cpu_position_[geometry.oxygen_low + number_of_atoms_];
  double oz = cpu_position_[geometry.oxygen_high + 2 * number_of_atoms_] -
    cpu_position_[geometry.oxygen_low + 2 * number_of_atoms_];
  apply_mic(box, ox, oy, oz);
  const double dOO_square = ox * ox + oy * oy + oz * oz;
  if (dOO_square < dOO_min_ * dOO_min_ || dOO_square > dOO_max_ * dOO_max_)
    return false;
  geometry.dOO = std::sqrt(dOO_square);
  geometry.low_to_high_dx = ox;
  geometry.low_to_high_dy = oy;
  geometry.low_to_high_dz = oz;

  double hx_from_low = hx - cpu_position_[geometry.oxygen_low];
  double hy_from_low = hy - cpu_position_[geometry.oxygen_low + number_of_atoms_];
  double hz_from_low = hz - cpu_position_[geometry.oxygen_low + 2 * number_of_atoms_];
  apply_mic(box, hx_from_low, hy_from_low, hz_from_low);
  const double projection = (hx_from_low * ox + hy_from_low * oy + hz_from_low * oz) /
    dOO_square;
  const double px = hx_from_low - projection * ox;
  const double py = hy_from_low - projection * oy;
  const double pz = hz_from_low - projection * oz;
  geometry.rperp = std::sqrt(px * px + py * py + pz * pz);
  if (geometry.rperp > rperp_max_)
    return false;

  // The two nearest O atoms must form a hydrogen-bond-like O-H-O geometry. This rejects
  // most accidental pairs from the other interpenetrating ice-VII network.
  double low_x = cpu_position_[geometry.oxygen_low] - hx;
  double low_y = cpu_position_[geometry.oxygen_low + number_of_atoms_] - hy;
  double low_z = cpu_position_[geometry.oxygen_low + 2 * number_of_atoms_] - hz;
  double high_x = cpu_position_[geometry.oxygen_high] - hx;
  double high_y = cpu_position_[geometry.oxygen_high + number_of_atoms_] - hy;
  double high_z = cpu_position_[geometry.oxygen_high + 2 * number_of_atoms_] - hz;
  apply_mic(box, low_x, low_y, low_z);
  apply_mic(box, high_x, high_y, high_z);
  const double low_distance = low_oxygen_distance;
  const double high_distance = high_oxygen_distance;
  if (low_distance <= 0.0 || high_distance <= 0.0 ||
      low_distance > 1.60 || high_distance > 1.60)
    return false;
  const double angle_cosine = (low_x * high_x + low_y * high_y + low_z * high_z) /
    (low_distance * high_distance);
  if (angle_cosine > -0.5)
    return false;

  geometry.valid = true;
  return true;
}

int Proton_Tunneling::classify_delta(const double delta) const
{
  if (delta > delta_cutoff_)
    return 1;
  if (delta < -delta_cutoff_)
    return -1;
  return 0;
}

void Proton_Tunneling::record_bond(
  std::unordered_map<unsigned long long, BondStats>& bond_stats,
  const GeometryResult& geometry,
  const int state)
{
  BondStats& stats = bond_stats[make_bond_key(geometry.oxygen_low, geometry.oxygen_high)];
  ++stats.geometry_samples;
  stats.sum_abs_delta += std::abs(geometry.delta);
  stats.sum_delta += geometry.delta;
  stats.sum_dOO += geometry.dOO;
  stats.sum_rperp += geometry.rperp;
  if (state > 0)
    ++stats.n_plus;
  else if (state < 0)
    ++stats.n_minus;
  else
    ++stats.n_deadband;
}

void Proton_Tunneling::start_attempt(
  HydrogenState& hydrogen_state,
  const int stable_state,
  const double time_fs,
  const double delta)
{
  hydrogen_state.attempt_active = true;
  hydrogen_state.attempt_from_state = stable_state;
  hydrogen_state.attempt_id = next_attempt_id_++;
  hydrogen_state.attempt_start_time_fs = time_fs;
  hydrogen_state.attempt_delta_start = delta;
  hydrogen_state.attempt_min_abs_delta = std::abs(delta);
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
}

const char* Proton_Tunneling::outcome_name(const AttemptOutcome outcome) const
{
  switch (outcome) {
  case AttemptOutcome::success:
    return "success";
  case AttemptOutcome::return_to_state:
    return "return";
  case AttemptOutcome::geometry_lost:
    return "geometry_lost";
  case AttemptOutcome::run_end:
    return "run_end";
  }
  return "unknown";
}

void Proton_Tunneling::finish_attempt(
  const int hydrogen,
  HydrogenState& hydrogen_state,
  const AttemptOutcome outcome,
  const double time_fs,
  const double delta_end,
  const GeometryResult* geometry)
{
  if (!hydrogen_state.attempt_active)
    return;

  const int oxygen_low = hydrogen_state.oxygen_low;
  const int oxygen_high = hydrogen_state.oxygen_high;
  const int oxygen_from = (hydrogen_state.attempt_from_state < 0)
    ? oxygen_low
    : oxygen_high;
  const int oxygen_target = (hydrogen_state.attempt_from_state < 0)
    ? oxygen_high
    : oxygen_low;
  const unsigned long long key = make_bond_key(oxygen_low, oxygen_high);
  BondStats& total_stats = total_bonds_[key];
  BondStats& window_stats = window_bonds_[key];
  ++total_stats.attempts;
  ++window_stats.attempts;
  if (outcome == AttemptOutcome::success) {
    ++total_stats.successes;
    ++window_stats.successes;
    ++total_stats.transitions;
    ++window_stats.transitions;
    ++window_flip_count_;
  } else if (outcome == AttemptOutcome::return_to_state) {
    ++total_stats.returns;
    ++window_stats.returns;
  } else if (outcome == AttemptOutcome::geometry_lost) {
    ++total_stats.geometry_lost;
    ++window_stats.geometry_lost;
  }

  fprintf(
    attempt_file_,
    "%lld %.10e %.10e %d %d %d %d %d %s %.10e %.10e %.10e\n",
    hydrogen_state.attempt_id,
    hydrogen_state.attempt_start_time_fs,
    time_fs,
    hydrogen,
    oxygen_low,
    oxygen_high,
    oxygen_from,
    oxygen_target,
    outcome_name(outcome),
    hydrogen_state.attempt_delta_start,
    hydrogen_state.attempt_min_abs_delta,
    delta_end);

  if (outcome == AttemptOutcome::success && geometry != nullptr) {
    double dx = geometry->low_to_high_dx;
    double dy = geometry->low_to_high_dy;
    double dz = geometry->low_to_high_dz;
    if (oxygen_from != geometry->oxygen_low) {
      dx = -dx;
      dy = -dy;
      dz = -dz;
    }
    fprintf(
      transfer_file_,
      "%lld %.10e %.10e %d %d %d %d %d %.10e %.10e %.10e %.10e %.10e %d %d\n",
      hydrogen_state.attempt_id,
      hydrogen_state.attempt_start_time_fs,
      time_fs,
      hydrogen,
      oxygen_from,
      oxygen_target,
      oxygen_low,
      oxygen_high,
      dx,
      dy,
      dz,
      hydrogen_state.attempt_delta_start,
      delta_end,
      hydrogen_count_[oxygen_from],
      hydrogen_count_[oxygen_target]);
  }

  if (outcome == AttemptOutcome::success)
    hydrogen_state.stable_state = -hydrogen_state.attempt_from_state;
  else if (outcome == AttemptOutcome::geometry_lost || outcome == AttemptOutcome::run_end)
    hydrogen_state.stable_state = 0;
  hydrogen_state.attempt_active = false;
  hydrogen_state.attempt_from_state = 0;
  hydrogen_state.attempt_id = 0;
  hydrogen_state.attempt_start_time_fs = 0.0;
  hydrogen_state.attempt_delta_start = 0.0;
  hydrogen_state.attempt_min_abs_delta = 0.0;
  hydrogen_state.pending_state = 0;
  hydrogen_state.pending_count = 0;
  hydrogen_state.pending_start_time_fs = 0.0;
}

void Proton_Tunneling::observe_frame(
  const double time_fs,
  const Box& box,
  Atom& atom)
{
  atom.position_per_atom.copy_to_host(cpu_position_.data());
  if (window_sample_count_ == 0)
    window_start_time_fs_ = time_fs;

  std::fill(hydrogen_count_.begin(), hydrogen_count_.end(), 0);
  for (size_t h_index = 0; h_index < hydrogen_indices_.size(); ++h_index) {
    const int hydrogen = hydrogen_indices_[h_index];
    GeometryResult geometry;
    const bool valid_geometry = find_geometry(hydrogen, box, geometry);
    if (geometry.nearest_oxygen >= 0)
      ++hydrogen_count_[geometry.nearest_oxygen];

    HydrogenState& hydrogen_state = hydrogen_states_[h_index];
    if (!valid_geometry) {
      if (hydrogen_state.attempt_active) {
        finish_attempt(
          hydrogen,
          hydrogen_state,
          AttemptOutcome::geometry_lost,
          time_fs,
          hydrogen_state.last_delta,
          nullptr);
      }
      hydrogen_state = HydrogenState();
      continue;
    }

    ++window_valid_pair_count_;
    const int state = classify_delta(geometry.delta);

    if (hydrogen_state.oxygen_low != geometry.oxygen_low ||
        hydrogen_state.oxygen_high != geometry.oxygen_high) {
      if (hydrogen_state.attempt_active) {
        finish_attempt(
          hydrogen,
          hydrogen_state,
          AttemptOutcome::geometry_lost,
          time_fs,
          hydrogen_state.last_delta,
          nullptr);
      }
      hydrogen_state = HydrogenState();
      hydrogen_state.oxygen_low = geometry.oxygen_low;
      hydrogen_state.oxygen_high = geometry.oxygen_high;
    }

    record_bond(window_bonds_, geometry, state);
    record_bond(total_bonds_, geometry, state);

    if (hydrogen_state.stable_state == 0) {
      if (state == 0) {
        hydrogen_state.pending_state = 0;
        hydrogen_state.pending_count = 0;
      } else {
        if (hydrogen_state.pending_state == state) {
          ++hydrogen_state.pending_count;
        } else {
          hydrogen_state.pending_state = state;
          hydrogen_state.pending_count = 1;
        }
        if (hydrogen_state.pending_count >= hold_samples_) {
          hydrogen_state.stable_state = state;
          hydrogen_state.pending_state = 0;
          hydrogen_state.pending_count = 0;
        }
      }
    } else {
      if (!hydrogen_state.attempt_active && state != hydrogen_state.stable_state) {
        // If sampling jumps across the dead band, retain the event rather than silently
        // losing it. With a sufficiently small sample interval, normal attempts start in state 0.
        start_attempt(hydrogen_state, hydrogen_state.stable_state, time_fs, geometry.delta);
      }

      if (hydrogen_state.attempt_active) {
        hydrogen_state.attempt_min_abs_delta = std::min(
          hydrogen_state.attempt_min_abs_delta,
          std::abs(geometry.delta));
        if (state == hydrogen_state.stable_state) {
          finish_attempt(
            hydrogen,
            hydrogen_state,
            AttemptOutcome::return_to_state,
            time_fs,
            geometry.delta,
            nullptr);
        } else if (state == -hydrogen_state.attempt_from_state) {
          if (hydrogen_state.pending_state == state) {
            ++hydrogen_state.pending_count;
          } else {
            hydrogen_state.pending_state = state;
            hydrogen_state.pending_count = 1;
            hydrogen_state.pending_start_time_fs = time_fs;
          }
          if (hydrogen_state.pending_count >= hold_samples_) {
            finish_attempt(
              hydrogen,
              hydrogen_state,
              AttemptOutcome::success,
              time_fs,
              geometry.delta,
              &geometry);
          }
        } else {
          hydrogen_state.pending_state = 0;
          hydrogen_state.pending_count = 0;
          hydrogen_state.pending_start_time_fs = 0.0;
        }
      } else if (state == 0) {
        start_attempt(hydrogen_state, hydrogen_state.stable_state, time_fs, geometry.delta);
      }
    }
    hydrogen_state.last_delta = geometry.delta;
  }

  for (const int oxygen : oxygen_indices_) {
    if (hydrogen_count_[oxygen] > 2)
      ++window_positive_defect_sum_;
    else if (hydrogen_count_[oxygen] < 2)
      ++window_negative_defect_sum_;
  }

  ++window_sample_count_;
  last_time_fs_ = time_fs;
  if (window_sample_count_ == window_samples_)
    write_window(time_fs);
}

void Proton_Tunneling::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  (void)number_of_steps;
  (void)fixed_group;
  (void)move_group;
  (void)temperature;
  (void)integrate;
  (void)group;
  (void)thermo;
  (void)force;
  if (!initialized_ || (step + 1) % sample_interval_ != 0)
    return;

  observe_frame(global_time * TIME_UNIT_CONVERSION, box, atom);
}

void Proton_Tunneling::write_window(const double time_fs)
{
  if (window_sample_count_ == 0)
    return;

  double b_mean = 0.0;
  double f_02 = 0.0;
  double f_04 = 0.0;
  double delta_f_sum = 0.0;
  int active_bonds = 0;
  int delta_f_bonds = 0;
  for (const auto& item : window_bonds_) {
    const BondStats& stats = item.second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    if (biased_samples == 0)
      continue;
    ++active_bonds;
    const double asymmetry = static_cast<double>(stats.n_plus - stats.n_minus) /
      static_cast<double>(biased_samples);
    const double abs_asymmetry = std::abs(asymmetry);
    b_mean += abs_asymmetry;
    if (abs_asymmetry > 0.2)
      f_02 += 1.0;
    if (abs_asymmetry > 0.4)
      f_04 += 1.0;
    if (stats.n_plus > 0 && stats.n_minus > 0) {
      delta_f_sum += std::abs(std::log(
        static_cast<double>(stats.n_plus) / static_cast<double>(stats.n_minus)));
      ++delta_f_bonds;
    }
  }

  if (active_bonds > 0) {
    b_mean /= active_bonds;
    f_02 /= active_bonds;
    f_04 /= active_bonds;
  }
  const double mean_abs_delta_f = (delta_f_bonds > 0)
    ? delta_f_sum / delta_f_bonds
    : std::numeric_limits<double>::quiet_NaN();
  const double sample_dt_ps = time_step_ * sample_interval_ * TIME_UNIT_CONVERSION / 1000.0;
  const double window_time_ps = sample_dt_ps * window_sample_count_;
  const double flip_rate = (window_time_ps > 0.0)
    ? static_cast<double>(window_flip_count_) / window_time_ps
    : 0.0;
  const double valid_pairs_per_frame = static_cast<double>(window_valid_pair_count_) /
    window_sample_count_;
  const double positive_defects = static_cast<double>(window_positive_defect_sum_) /
    window_sample_count_;
  const double negative_defects = static_cast<double>(window_negative_defect_sum_) /
    window_sample_count_;

  fprintf(
    bias_file_,
    "%.10e %.10e %.10e %.10e %.10e %.10e %d %.10e %.10e %.10e\n",
    time_fs,
    b_mean,
    f_02,
    f_04,
    mean_abs_delta_f,
    flip_rate,
    active_bonds,
    positive_defects,
    negative_defects,
    valid_pairs_per_frame);
  fflush(bias_file_);

  write_edge_window(window_start_time_fs_, time_fs);

  window_bonds_.clear();
  window_sample_count_ = 0;
  window_start_time_fs_ = 0.0;
  window_flip_count_ = 0;
  window_valid_pair_count_ = 0;
  window_positive_defect_sum_ = 0;
  window_negative_defect_sum_ = 0;
}

void Proton_Tunneling::write_edge_window(
  const double time_start_fs,
  const double time_end_fs)
{
  BondStats empty_stats;
  for (const auto& item : total_bonds_) {
    int oxygen_low;
    int oxygen_high;
    decode_bond_key(item.first, oxygen_low, oxygen_high);
    const auto window_item = window_bonds_.find(item.first);
    const BondStats& stats = (window_item == window_bonds_.end())
      ? empty_stats
      : window_item->second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    const long long completed_attempts = stats.successes + stats.returns;
    const double geometry_occupancy = static_cast<double>(stats.geometry_samples) /
      window_sample_count_;
    const double asymmetry = (biased_samples > 0)
      ? static_cast<double>(stats.n_plus - stats.n_minus) / static_cast<double>(biased_samples)
      : std::numeric_limits<double>::quiet_NaN();
    const double delta_f = (stats.n_plus > 0 && stats.n_minus > 0)
      ? std::abs(std::log(static_cast<double>(stats.n_plus) /
                          static_cast<double>(stats.n_minus)))
      : std::numeric_limits<double>::quiet_NaN();
    const double success_probability = (completed_attempts > 0)
      ? static_cast<double>(stats.successes) / static_cast<double>(completed_attempts)
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_delta = (stats.geometry_samples > 0)
      ? stats.sum_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_abs_delta = (stats.geometry_samples > 0)
      ? stats.sum_abs_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_dOO = (stats.geometry_samples > 0)
      ? stats.sum_dOO / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_rperp = (stats.geometry_samples > 0)
      ? stats.sum_rperp / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    fprintf(
      edge_window_file_,
      "%lld %.10e %.10e %d %d %.10e %lld %lld %lld %.10e %.10e %.10e "
      "%lld %lld %lld %lld %.10e %.10e %.10e %.10e %.10e\n",
      window_id_,
      time_start_fs,
      time_end_fs,
      oxygen_low,
      oxygen_high,
      geometry_occupancy,
      stats.n_plus,
      stats.n_minus,
      stats.n_deadband,
      asymmetry,
      std::abs(asymmetry),
      delta_f,
      stats.attempts,
      stats.successes,
      stats.returns,
      stats.geometry_lost,
      success_probability,
      mean_delta,
      mean_abs_delta,
      mean_dOO,
      mean_rperp);
  }
  fflush(edge_window_file_);
  ++window_id_;
}

void Proton_Tunneling::write_final_bonds()
{
  for (const auto& item : total_bonds_) {
    int oxygen_low;
    int oxygen_high;
    decode_bond_key(item.first, oxygen_low, oxygen_high);
    const BondStats& stats = item.second;
    const long long biased_samples = stats.n_plus + stats.n_minus;
    const double asymmetry = (biased_samples > 0)
      ? static_cast<double>(stats.n_plus - stats.n_minus) / static_cast<double>(biased_samples)
      : std::numeric_limits<double>::quiet_NaN();
    const double mean_abs_delta = (stats.geometry_samples > 0)
      ? stats.sum_abs_delta / stats.geometry_samples
      : std::numeric_limits<double>::quiet_NaN();
    fprintf(
      bond_file_,
      "%d %d %lld %lld %lld %lld %.10e %.10e %.10e\n",
      oxygen_low,
      oxygen_high,
      stats.geometry_samples,
      stats.n_plus,
      stats.n_minus,
      stats.transitions,
      asymmetry,
      std::abs(asymmetry),
      mean_abs_delta);
  }
  fflush(bond_file_);
}

void Proton_Tunneling::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  (void)atom;
  (void)box;
  (void)integrate;
  (void)number_of_steps;
  (void)time_step;
  (void)temperature;
  if (!initialized_)
    return;

  for (size_t h_index = 0; h_index < hydrogen_states_.size(); ++h_index) {
    HydrogenState& hydrogen_state = hydrogen_states_[h_index];
    if (hydrogen_state.attempt_active) {
      finish_attempt(
        hydrogen_indices_[h_index],
        hydrogen_state,
        AttemptOutcome::run_end,
        last_time_fs_,
        hydrogen_state.last_delta,
        nullptr);
    }
  }
  if (window_sample_count_ > 0)
    write_window(last_time_fs_);
  write_final_bonds();
  fclose(bias_file_);
  fclose(transfer_file_);
  fclose(attempt_file_);
  fclose(edge_window_file_);
  fclose(bond_file_);
  bias_file_ = nullptr;
  transfer_file_ = nullptr;
  attempt_file_ = nullptr;
  edge_window_file_ = nullptr;
  bond_file_ = nullptr;
  initialized_ = false;
}
