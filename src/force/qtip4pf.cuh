/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#pragma once

#include "ewald.cuh"
#include "neighbor.cuh"
#include "potential.cuh"
#include "pppm.cuh"
#include <memory>
#include <stdio.h>

constexpr int QTIP4PF_NUM_REAL_TYPES = 4;
constexpr int QTIP4PF_NUM_CHARGE_TYPES = 5;

struct QTIP4PF_Para
{
  float weight[3]; // O, H1, H2 weights used to construct M
  float charge[QTIP4PF_NUM_CHARGE_TYPES]; // O, H, Na, Cl, M
  float lj_s6e4[QTIP4PF_NUM_REAL_TYPES][QTIP4PF_NUM_REAL_TYPES];
  float lj_s12e4[QTIP4PF_NUM_REAL_TYPES][QTIP4PF_NUM_REAL_TYPES];
  float lj_cutoff_square;
  float coulomb_cutoff_square;
  float alpha;
  float two_alpha_over_sqrt_pi;
  float bond_r0;
  float bond_D;
  float bond_a;
  float angle_theta0;
  float angle_k;
};

class QTIP4PF : public Potential
{
public:
  using Potential::compute;
  QTIP4PF(FILE* fid, int num_types, int num_atoms);
  virtual ~QTIP4PF(void);

  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

private:
  int number_of_atoms_;
  int number_of_waters_ = 0;
  int number_of_sites_ = 0;
  bool topology_initialized_ = false;
  bool use_pppm_ = true;
  double lj_cutoff_ = 0.0;
  double coulomb_cutoff_ = 0.0;
  QTIP4PF_Para para_{};

  Neighbor lj_neighbor_;
  Neighbor charge_neighbor_;
  std::unique_ptr<Ewald> ewald_;
  std::unique_ptr<PPPM> pppm_;

  GPU_Vector<int> water_O_;
  GPU_Vector<int> water_H1_;
  GPU_Vector<int> water_H2_;
  GPU_Vector<int> site_type_;
  GPU_Vector<float> site_charge_;
  GPU_Vector<float> site_D_real_;
  GPU_Vector<double> site_position_;
  GPU_Vector<double> site_potential_;
  GPU_Vector<double> site_force_;
  GPU_Vector<double> site_virial_;

  void initialize_topology(const GPU_Vector<int>& type);
};
