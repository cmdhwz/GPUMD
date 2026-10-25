#pragma once

#include "action.cuh"
#include "force/nep_plane_flux_view.cuh"
#include <cstdint>
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

class InterfaceGeometry;
class NEP;

class ComputeInterfaceSHC : public Action
{
public:
  ComputeInterfaceSHC(
    const std::vector<std::string>& tokens,
    const std::shared_ptr<InterfaceGeometry>& interface_geometry);

  void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;
  void pre_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;
  void post_force(
    const int step,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force) override;
  void end_of_step(
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
  void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature) override;

private:
  std::shared_ptr<InterfaceGeometry> interface_geometry_;
  NEP* nep_ = nullptr;
  int sample_interval_ = 1;
  int correlation_samples_ = 1;
  int block_samples_ = 2;
  int num_omega_ = 2;
  double max_omega_rad_ps_ = 0.0;
  double max_displacement_A_ = 0.0;
  double nominal_step_size_ = 0.0;
  double last_step_time_ps_ = 0.0;
  double sample_dt_ps_ = 0.0;
  double block_start_ps_ = 0.0;
  double block_end_ps_ = 0.0;
  double accumulation_start_ps_ = 0.0;
  double accumulation_end_ps_ = 0.0;
  std::uint64_t accumulation_samples_ = 0;
  int samples_in_block_ = 0;
  int block_index_ = 0;
  bool have_step_time_ = false;
  bool have_accumulation_range_ = false;
  bool armed_ = false;
  bool active_ = false;
  std::string prefix_;
  std::vector<int> selected_atoms_;
  std::vector<int> slot_planes_;
  std::vector<int> plane_offsets_;
  std::vector<int> candidate_offsets_;
  std::vector<int> candidate_planes_;
  std::vector<int> cached_steps_;
  std::vector<double> cached_times_ps_;
  std::vector<double> cached_bath_source_eV_;
  std::vector<double> cached_bath_sink_eV_;
  std::vector<double> cached_plane_flux_;
  std::vector<double> cached_correlations_;
  std::vector<double> cached_block_start_ps_;
  std::vector<double> cached_block_end_ps_;
  size_t sample_capacity_ = 0;
  size_t complete_block_capacity_ = 0;
  size_t cached_sample_count_ = 0;
  std::vector<double> spectrum_mean_;
  std::vector<double> spectrum_m2_;
  std::vector<double> c0_mean_;
  std::vector<double> c0_m2_;
  std::vector<double> integral_mean_;
  std::vector<double> integral_m2_;
  std::vector<double> grid_integral_mean_;
  std::vector<double> grid_integral_m2_;
  int spectrum_blocks_ = 0;

  GPU_Vector<double> gpu_reference_positions_;
  GPU_Vector<double> gpu_plane_positions_;
  GPU_Vector<int> gpu_plane_offsets_;
  GPU_Vector<int> gpu_candidate_offsets_;
  GPU_Vector<int> gpu_candidate_planes_;
  GPU_Vector<int> gpu_selected_atoms_;
  GPU_Vector<int> gpu_slot_planes_;
  GPU_Vector<double> gpu_coefficients_;
  GPU_Vector<int> gpu_error_info_;
  GPU_Vector<double> gpu_coefficient_history_;
  GPU_Vector<double> gpu_velocity_history_;
  GPU_Vector<double> gpu_correlation_plus_;
  GPU_Vector<double> gpu_correlation_minus_;
  GPU_Vector<double> gpu_plane_flux_;
  NEPPlaneFluxView view_;
  FILE* fid_plane_flux_ = nullptr;
  FILE* fid_correlation_ = nullptr;
  FILE* fid_shc_ = nullptr;
  FILE* fid_shc_blocks_ = nullptr;
  size_t allocated_bytes_ = 0;

  void capture_sample(
    const int step,
    const double time_ps,
    Integrate& integrate,
    Atom& atom);
  void flush_sample_block(const bool complete_block);
  void output_cached_samples();
  void output_complete_block(const size_t block_index);
  void check_block_error();
};
