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
#include "ensemble.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif
#include <memory>
#include <random>
#include <vector>

class Ensemble_PIMD : public Ensemble
{
public:
  struct DistributedReplica
  {
    int device_id = 0;
    int bead_begin = 0;
    int bead_end = 0;
    Atom atom;
    GPU_Vector<double> thermo;
    std::unique_ptr<Ensemble_PIMD> ensemble;
  };

  Ensemble_PIMD(
    int number_of_atoms_input,
    int number_of_beads_input,
    bool thermostat_internal,
    Atom& atom,
    bool use_exact_propagator,
    double pile_scale,
    bool fix_com,
    bool reseed_from_centroid = false);

  Ensemble_PIMD(
    int number_of_atoms_input,
    int number_of_beads_input,
    double temperature_coupling,
    Atom& atom,
    bool use_exact_propagator,
    double pile_scale,
    bool fix_com,
    bool reseed_from_centroid = false);

  Ensemble_PIMD(
    int number_of_atoms_input,
    int number_of_beads_input,
    double temperature_coupling,
    int num_target_pressure_components,
    double target_pressure[6],
    double pressure_coupling[6],
    Atom& atom,
    bool use_exact_propagator,
    double pile_scale,
    bool fix_com,
    bool use_scr_barostat,
    bool reseed_from_centroid = false);

  virtual ~Ensemble_PIMD(void);

  virtual void compute1(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);

  virtual void compute2(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);
  void enable_distributed(int num_devices, Atom& atom, GPU_Vector<double>& thermo);
  bool distributed_enabled() const { return distributed_enabled_; }
  void compute_force_distributed(Force& force, Box& box, std::vector<Group>& group, Atom& atom);

protected:
  int number_of_atoms = 0;
  int number_of_beads = 0;
  bool thermostat_internal = false;
  bool thermostat_centroid = false;
  bool use_exact_propagator_ = true;
  double pile_scale_ = 2.0;
  bool fix_com_ = true;
  bool use_scr_barostat_ = false;
  bool reseed_from_centroid_ = false;
  double omega_n;
  GPU_Vector<gpurandState> curand_states;
  GPU_Vector<double*> position_beads;
  GPU_Vector<double*> velocity_beads;
  GPU_Vector<double*> potential_beads;
  GPU_Vector<double*> force_beads;
  GPU_Vector<double*> virial_beads;
  GPU_Vector<double> transformation_matrix;
  GPU_Vector<double> free_ring_polymer_frequency;
  GPU_Vector<double> free_ring_polymer_cosine;
  GPU_Vector<double> free_ring_polymer_sine;
  bool free_ring_polymer_propagator_initialized_ = false;
  double free_ring_polymer_cached_omega_n_ = 0.0;
  double free_ring_polymer_cached_time_step_ = 0.0;
  GPU_Vector<double> kinetic_energy_virial_part;

  GPU_Vector<double> sum_1024; // for intermidiate summation

  void initialize(Atom& atom);
  void update_free_ring_polymer_propagator_(const double time_step);
  void langevin(const double time_step, Atom& atom);
  void compute1_local_(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);
  void compute2_local_pre_pressure_(
    const double time_step,
    const std::vector<Group>& group,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& thermo);
  void apply_pressure_local_orthogonal_(Atom& atom, double scale_factor[3]);
  void apply_pressure_local_isotropic_(Atom& atom, double scale_factor);
  void apply_pressure_local_triclinic_(Atom& atom, double mu[9]);
  void clone_atom_to_current_device_(const Atom& source, Atom& destination, int source_device, int destination_device);
  std::mt19937 rng;
  void initialize_rng();
  bool distributed_enabled_ = false;
  std::vector<std::unique_ptr<DistributedReplica>> distributed_replicas_;
};
