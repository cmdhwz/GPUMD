#pragma once

#include "rpmd_ja_reference.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <vector>

class RpmdJAQNEPCached
{
public:
  RpmdJAQNEPCached() = default;
  ~RpmdJAQNEPCached();
  RpmdJAQNEPCached(const RpmdJAQNEPCached&) = delete;
  RpmdJAQNEPCached& operator=(const RpmdJAQNEPCached&) = delete;

  void initialize(const RpmdJAReference& reference, const std::vector<double>& masses);
  void compute_correction(
    const GPU_Vector<double>& continuous,
    const GPU_Vector<double>& reference_positions,
    const GPU_Vector<double>& velocity,
    double* device_result,
    int* device_error);

  std::size_t allocated_bytes() const { return allocated_bytes_; }
  std::size_t dynamical_tile_count() const { return dynamical_tile_count_; }
  std::size_t site_tile_count(const int alpha) const { return site_tile_count_[alpha]; }
  double preparation_seconds() const { return preparation_seconds_; }
  std::size_t estimated_peak_bytes() const { return estimated_peak_bytes_; }

private:
  GPU_Vector<double> eigenvectors_;
  GPU_Vector<double> kernel_[3];
  GPU_Vector<double> sqrt_mass_;
  GPU_Vector<double> weighted_yu_;
  GPU_Vector<double> modal_yu_;
  GPU_Vector<double> scratch_;
  GPU_Vector<double> alpha_scalar_;
  GPU_Vector<double> beta_scalar_;
  void* blas_handle_ = nullptr;
  int number_of_atoms_ = 0;
  int dimension_ = 0;
  int modal_dimension_ = 0;
  int degree_ = 0;
  int p_rank_ = 0;
  int q_rank_ = 0;
  double tau_ = 0.0;
  double lambda_scale_ = 0.0;
  std::size_t allocated_bytes_ = 0;
  std::size_t estimated_peak_bytes_ = 0;
  std::size_t dynamical_tile_count_ = 0;
  std::size_t site_tile_count_[3] = {};
  double preparation_seconds_ = 0.0;
};
