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

#ifdef USE_NETCDF

#pragma once
#include "property.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class HAC;

class Centroid_DeltaF_O : public Property
{
public:
  Centroid_DeltaF_O(const char** param, int num_param);

  void set_hac(HAC* hac) { hac_ = hac; }
  int sample_interval() const { return sample_interval_; }

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
  int sample_interval_ = 1;
  int number_of_atoms_ = 0;
  int number_of_oxygen_ = 0;
  HAC* hac_ = nullptr;
  GPU_Vector<int> oxygen_indices_;
  GPU_Vector<float> gpu_delta_force_;
  std::vector<int> oxygen_atom_ids_;
  std::vector<long long> sampled_steps_;
  std::vector<double> sample_times_fs_;
  std::vector<float> delta_force_history_;
  std::vector<double> reference_positions_;
  double box_matrix_[9] = {};
  bool reference_position_set_ = false;

  void write_netcdf_();
};

#endif
