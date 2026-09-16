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
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <vector>

class HAC;

class Centroid_Force_Diagnostic : public Property
{
public:
  Centroid_Force_Diagnostic(const char** param, int num_param);

  void set_hac(HAC* hac) { hac_ = hac; }

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
  int number_of_types_ = 0;
  HAC* hac_ = nullptr;
  GPU_Vector<int> species_by_type_;
  GPU_Vector<double> gpu_statistics_;
  std::vector<double> cpu_statistics_;
  FILE* fid_ = nullptr;

  void write_header_();
  void write_row_(const int step, const double global_time);
};
