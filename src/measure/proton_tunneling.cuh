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

#pragma once
#include "property.cuh"
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

class Proton_Tunneling : public Property
{
public:
  Proton_Tunneling(const char** param, const int num_param, const Atom& atom);

  void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;

  void process(
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
    Force& force) override;

  void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  struct BondStats
  {
    long long geometry_samples = 0;
    long long n_plus = 0;
    long long n_minus = 0;
    long long transitions = 0;
    double sum_abs_delta = 0.0;
  };

  struct HydrogenState
  {
    int oxygen_low = -1;
    int oxygen_high = -1;
    int stable_state = 0;
    int pending_state = 0;
    int pending_count = 0;
    double last_delta = 0.0;
  };

  void parse(const char** param, const int num_param, const Atom& atom);
  bool find_geometry(
    const int hydrogen,
    const Box& box,
    int& nearest_oxygen,
    int& oxygen_low,
    int& oxygen_high,
    double& delta) const;
  int classify_delta(const double delta) const;
  void record_bond(
    std::unordered_map<unsigned long long, BondStats>& bond_stats,
    const int oxygen_low,
    const int oxygen_high,
    const double delta,
    const int state);
  void observe_frame(const double time_fs, const Box& box, Atom& atom);
  void write_window(const double time_fs);
  void write_final_bonds();

  int sample_interval_ = 1;
  int window_samples_ = 1000;
  int hold_samples_ = 2;
  int window_sample_count_ = 0;
  int number_of_atoms_ = 0;
  long long window_flip_count_ = 0;
  long long window_valid_pair_count_ = 0;
  long long window_positive_defect_sum_ = 0;
  long long window_negative_defect_sum_ = 0;
  double delta_cutoff_ = 0.10;
  double dOO_min_ = 2.20;
  double dOO_max_ = 2.60;
  double rperp_max_ = 0.80;
  double time_step_ = 0.0;
  double last_time_fs_ = 0.0;
  bool initialized_ = false;

  std::string oxygen_symbol_ = "O";
  std::string hydrogen_symbol_ = "H";
  std::vector<int> oxygen_indices_;
  std::vector<int> hydrogen_indices_;
  std::vector<double> cpu_position_;
  std::vector<HydrogenState> hydrogen_states_;

  std::unordered_map<unsigned long long, BondStats> window_bonds_;
  std::unordered_map<unsigned long long, BondStats> total_bonds_;
  FILE* bias_file_ = nullptr;
  FILE* transfer_file_ = nullptr;
  FILE* bond_file_ = nullptr;
};
