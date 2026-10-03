#pragma once

#include "rpmd_ja_reference.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstddef>
#include <cstdint>
#include <vector>

class Box;

constexpr int RPMD_JA_ERROR_BRANCH = 1;
constexpr int RPMD_JA_ERROR_NUMERICAL = 2;

void rpmd_ja_wrap_positions(
  const int number_of_atoms,
  const GPU_Vector<double>& continuous,
  GPU_Vector<double>& wrapped,
  const Box& box,
  int* device_error);

class RpmdJASparseWorkspace
{
public:
  void initialize(const RpmdJAReference& reference, const std::vector<double>& masses);
  void compute_correction(
    const GPU_Vector<double>& continuous,
    const GPU_Vector<double>& reference_positions,
    const GPU_Vector<double>& velocity,
    double* device_result,
    int* device_error);
  std::size_t allocated_bytes() const { return allocated_bytes_; }
  std::uint64_t dynamical_nnz() const { return dynamical_columns_.size(); }
  std::uint64_t site_nnz(const int alpha) const { return site_columns_[alpha].size(); }

private:
  GPU_Vector<std::uint64_t> dynamical_rows_;
  GPU_Vector<int> dynamical_columns_;
  GPU_Vector<double> dynamical_values_;
  GPU_Vector<std::uint64_t> site_rows_[3];
  GPU_Vector<int> site_columns_[3];
  GPU_Vector<double> site_values_[3];
  GPU_Vector<double> sqrt_mass_;
  GPU_Vector<double> p_values_;
  GPU_Vector<double> q_values_;
  GPU_Vector<double> p_vectors_;
  GPU_Vector<double> q_vectors_;
  GPU_Vector<double> weighted_yu_;
  GPU_Vector<double> dydu_;
  GPU_Vector<double> seed_;
  GPU_Vector<double> cheb_a_;
  GPU_Vector<double> cheb_b_;
  GPU_Vector<double> d_action_;
  GPU_Vector<double> projection_coefficients_;
  GPU_Vector<double> p_accumulators_;
  GPU_Vector<double> q_accumulators_;
  GPU_Vector<double> b_results_;
  int number_of_atoms_ = 0;
  int dimension_ = 0;
  int degree_ = 0;
  int p_rank_ = 0;
  int q_rank_ = 0;
  double tau_ = 0.0;
  double lambda_scale_ = 0.0;
  double inverse_mass_sum_ = 0.0;
  std::size_t allocated_bytes_ = 0;
};
