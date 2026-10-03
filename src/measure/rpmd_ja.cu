#include "force/nep.cuh"
#include "hac.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
constexpr double branch_limit = 0.45;

__device__ inline void apply_ja_mic(const Box& box, double& dx, double& dy, double& dz)
{
  double sx = box.cpu_h[9] * dx + box.cpu_h[10] * dy + box.cpu_h[11] * dz;
  double sy = box.cpu_h[12] * dx + box.cpu_h[13] * dy + box.cpu_h[14] * dz;
  double sz = box.cpu_h[15] * dx + box.cpu_h[16] * dy + box.cpu_h[17] * dz;
  if (box.pbc_x) sx -= nearbyint(sx);
  if (box.pbc_y) sy -= nearbyint(sy);
  if (box.pbc_z) sz -= nearbyint(sz);
  dx = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
  dy = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
  dz = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
}

static __global__ void initialize_centroid(
  const int N,
  const Box box,
  const double* raw,
  const double* reference,
  double* last_raw,
  double* continuous,
  int* branch_error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  if (!isfinite(raw[i]) || !isfinite(raw[i + N]) || !isfinite(raw[i + 2 * N])) {
    atomicOr(branch_error, RPMD_JA_ERROR_BRANCH);
    return;
  }
  double dx = raw[i] - reference[i];
  double dy = raw[i + N] - reference[i + N];
  double dz = raw[i + 2 * N] - reference[i + 2 * N];
  apply_ja_mic(box, dx, dy, dz);
  last_raw[i] = raw[i];
  last_raw[i + N] = raw[i + N];
  last_raw[i + 2 * N] = raw[i + 2 * N];
  continuous[i] = reference[i] + dx;
  continuous[i + N] = reference[i + N] + dy;
  continuous[i + 2 * N] = reference[i + 2 * N] + dz;
  const double sx = box.cpu_h[9] * dx + box.cpu_h[10] * dy + box.cpu_h[11] * dz;
  const double sy = box.cpu_h[12] * dx + box.cpu_h[13] * dy + box.cpu_h[14] * dz;
  const double sz = box.cpu_h[15] * dx + box.cpu_h[16] * dy + box.cpu_h[17] * dz;
  if ((box.pbc_x && fabs(sx) >= branch_limit) ||
      (box.pbc_y && fabs(sy) >= branch_limit) ||
      (box.pbc_z && fabs(sz) >= branch_limit)) {
    atomicOr(branch_error, RPMD_JA_ERROR_BRANCH);
  }
}

static __global__ void update_centroid(
  const int N,
  const Box box,
  const double* raw,
  const double* reference,
  double* last_raw,
  double* continuous,
  int* branch_error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  double dx = raw[i] - last_raw[i];
  double dy = raw[i + N] - last_raw[i + N];
  double dz = raw[i + 2 * N] - last_raw[i + 2 * N];
  if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
    atomicOr(branch_error, RPMD_JA_ERROR_BRANCH);
    return;
  }
  apply_ja_mic(box, dx, dy, dz);
  const double step_sx = box.cpu_h[9] * dx + box.cpu_h[10] * dy + box.cpu_h[11] * dz;
  const double step_sy = box.cpu_h[12] * dx + box.cpu_h[13] * dy + box.cpu_h[14] * dz;
  const double step_sz = box.cpu_h[15] * dx + box.cpu_h[16] * dy + box.cpu_h[17] * dz;
  if ((box.pbc_x && fabs(step_sx) >= branch_limit) ||
      (box.pbc_y && fabs(step_sy) >= branch_limit) ||
      (box.pbc_z && fabs(step_sz) >= branch_limit)) atomicOr(branch_error, RPMD_JA_ERROR_BRANCH);
  continuous[i] += dx;
  continuous[i + N] += dy;
  continuous[i + 2 * N] += dz;
  last_raw[i] = raw[i];
  last_raw[i + N] = raw[i + N];
  last_raw[i + 2 * N] = raw[i + 2 * N];
  const double x = continuous[i] - reference[i];
  const double y = continuous[i + N] - reference[i + N];
  const double z = continuous[i + 2 * N] - reference[i + 2 * N];
  const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  if ((box.pbc_x && fabs(sx) >= branch_limit) ||
      (box.pbc_y && fabs(sy) >= branch_limit) ||
      (box.pbc_z && fabs(sz) >= branch_limit)) {
    atomicOr(branch_error, RPMD_JA_ERROR_BRANCH);
  }
}

static __global__ void delta_current(
  const int D,
  const double* position,
  const double* reference,
  const double* velocity,
  const double* hx,
  const double* hy,
  const double* hz,
  double* result)
{
  const int alpha = blockIdx.x;
  const int tid = threadIdx.x;
  const double* h = alpha == 0 ? hx : (alpha == 1 ? hy : hz);
  __shared__ double sums[128];
  double sum = 0.0;
  for (int i = tid; i < D; i += blockDim.x) {
    double row = 0.0;
    for (int j = 0; j < D; ++j) row += h[static_cast<size_t>(i) * D + j] * velocity[j];
    sum += (position[i] - reference[i]) * row;
  }
  sums[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  if (tid == 0) result[alpha] = sums[0];
}

static __global__ void store_current(
  const int Nd,
  const int frame,
  const double* centroid_heat,
  const double* delta,
  double* jcent_history,
  double* delta_history,
  double* ja_history,
  int* error)
{
  const int d = threadIdx.x;
  if (d >= 3) return;
  const double jcent = d == 0 ? centroid_heat[frame + Nd * 0] + centroid_heat[frame + Nd * 1]
                    : d == 1 ? centroid_heat[frame + Nd * 2] + centroid_heat[frame + Nd * 3]
                             : centroid_heat[frame + Nd * 4];
  if (!isfinite(jcent) || !isfinite(delta[d]) || !isfinite(jcent + delta[d]))
    atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  const size_t index = static_cast<size_t>(frame) + static_cast<size_t>(Nd) * d;
  jcent_history[index] = jcent;
  delta_history[index] = delta[d];
  ja_history[index] = jcent + delta[d];
}

} // namespace

void HAC::pre_run_rpmd_ja_(
  const int number_of_frames, Integrate& integrate, Atom& atom, Box& box, Force& force)
{
  if (!use_centroid_heat_flux_) {
    PRINT_INPUT_ERROR("rpmd_ja on requires compute_hac centroid flag 1.");
  }
  if (qnep_full_a_) PRINT_INPUT_ERROR("rpmd_ja does not support hac_current qnep_full_a.");
  if (split_qnep_heat_by_type_ != 0 || deferred_centroid_qnep_ != 0) {
    PRINT_INPUT_ERROR("rpmd_ja does not support split or deferred centroid HAC.");
  }
  if (output_interval > Nc) PRINT_INPUT_ERROR("rpmd_ja requires output_interval not to exceed Nc.");
  if (!qnep_existing_file_has_schema(
        "heat_current_rpmd_ja.out",
        {"# segment_metadata_version 1",
         "# columns time_ps Jcent_x Jcent_y Jcent_z DeltaJ_x DeltaJ_y DeltaJ_z JA_x JA_y JA_z"}) ||
      !qnep_existing_file_has_schema(
        "hac_rpmd_ja.out",
        {"# segment_metadata_version 1",
         "# columns lag_index_first lag_time_ps HAC_x HAC_y HAC_z RTC_x RTC_y RTC_z"})) {
    PRINT_INPUT_ERROR(
      "existing rpmd_ja output has an incompatible schema; remove or rename it before starting a new run.");
  }
  if (integrate.get_type() != EnsembleType::RPMD) {
    PRINT_INPUT_ERROR("rpmd_ja requires the fixed-cell RPMD ensemble.");
  }
  if (
    integrate.get_deform_x() != 0 || integrate.get_deform_y() != 0 || integrate.get_deform_z() != 0 ||
    integrate.get_use_scr_barostat() || integrate.get_use_eco_pimd()) {
    PRINT_INPUT_ERROR("rpmd_ja requires fixed-cell primitive RPMD without SCR or Eco-PIMD.");
  }
  const double target_temperature = integrate.get_temperature2();
  if (
    !std::isfinite(target_temperature) || !(target_temperature > 0.0) ||
    std::fabs(integrate.get_temperature1() - target_temperature) >
      1.0e-12 * std::max(1.0, std::fabs(target_temperature))) {
    PRINT_INPUT_ERROR("rpmd_ja requires a constant positive RPMD target temperature.");
  }
  if (force.potentials.size() != 1 || force.primary_nep_model_path().empty()) {
    PRINT_INPUT_ERROR("rpmd_ja requires exactly one short-range NEP potential.");
  }
  auto* active_nep = dynamic_cast<NEP*>(force.potentials[0].get());
  if (active_nep == nullptr || !active_nep->supports_local_edge_derivatives()) {
    PRINT_INPUT_ERROR("rpmd_ja requires one short-range NEP without unsupported corrections.");
  }

  rpmd_ja_reference_ = read_rpmd_ja_reference(rpmd_ja_reference_path_);
  rpmd_ja_reference_file_fingerprint_ = rpmd_ja_model_fingerprint(rpmd_ja_reference_path_);
  const int N = atom.number_of_atoms;
  const size_t D = static_cast<size_t>(N) * 3;
  if (
    rpmd_ja_reference_.number_of_atoms != N || rpmd_ja_reference_.types.size() != N ||
    rpmd_ja_reference_.masses.size() != N || rpmd_ja_reference_.positions.size() != D ||
    rpmd_ja_reference_.model_fingerprint !=
      rpmd_ja_model_fingerprint(force.primary_nep_model_path())) {
    PRINT_INPUT_ERROR("rpmd_ja reference atom metadata or NEP model does not match this run.");
  }
  if (
    !std::isfinite(rpmd_ja_reference_.temperature) ||
    std::fabs(rpmd_ja_reference_.temperature - target_temperature) >
      1.0e-10 * std::max(1.0, std::fabs(rpmd_ja_reference_.temperature))) {
    PRINT_INPUT_ERROR("rpmd_ja reference temperature must match the constant RPMD target.");
  }
  if (!std::isfinite(rpmd_ja_reference_.fd_step) || !(rpmd_ja_reference_.fd_step > 0.0)) {
    PRINT_INPUT_ERROR("rpmd_ja reference finite-difference step must be positive and finite.");
  }
  if (!std::all_of(rpmd_ja_reference_.positions.begin(), rpmd_ja_reference_.positions.end(),
                   [](const double value) { return std::isfinite(value); })) {
    PRINT_INPUT_ERROR("rpmd_ja reference positions must be finite.");
  }
  for (int i = 0; i < N; ++i) {
    if (
      rpmd_ja_reference_.types[i] != atom.cpu_type[i] ||
      !std::isfinite(rpmd_ja_reference_.masses[i]) ||
      std::fabs(rpmd_ja_reference_.masses[i] - atom.cpu_mass[i]) >
        1.0e-12 * std::max(1.0, std::fabs(atom.cpu_mass[i]))) {
      PRINT_INPUT_ERROR("rpmd_ja reference atom order, type, or mass does not match.");
    }
  }
  for (int d = 0; d < 3; ++d) {
    const int pbc = d == 0 ? box.pbc_x : d == 1 ? box.pbc_y : box.pbc_z;
    if (rpmd_ja_reference_.pbc[d] != pbc) {
      PRINT_INPUT_ERROR("rpmd_ja reference periodic-boundary metadata does not match.");
    }
  }
  for (int i = 0; i < 9; ++i) {
    if (
      !std::isfinite(rpmd_ja_reference_.cell[i]) ||
      std::fabs(rpmd_ja_reference_.cell[i] - box.cpu_h[i]) >
        1.0e-10 * std::max(1.0, std::fabs(box.cpu_h[i]))) {
      PRINT_INPUT_ERROR("rpmd_ja requires the fixed reference cell.");
    }
  }
  box.get_inverse();
  box.set_is_orthogonal();
  if (rpmd_ja_reference_.backend == 0) {
    for (int d = 0; d < 3; ++d) {
      if (rpmd_ja_reference_.delta_h[d].size() != D * D ||
          !std::all_of(rpmd_ja_reference_.delta_h[d].begin(), rpmd_ja_reference_.delta_h[d].end(),
                       [](const double value) { return std::isfinite(value); })) {
        PRINT_INPUT_ERROR("rpmd_ja dense reference matrix has invalid shape or non-finite values.");
      }
      rpmd_ja_delta_h_[d].resize(D * D);
      rpmd_ja_delta_h_[d].copy_from_host(rpmd_ja_reference_.delta_h[d].data());
    }
  } else if (rpmd_ja_reference_.backend == 1) {
    if (!rpmd_ja_reference_.stability_checked)
      PRINT_INPUT_ERROR("rpmd_ja sparse reference has no verified stability certificate.");
  } else {
    PRINT_INPUT_ERROR("rpmd_ja reference uses an unsupported backend.");
  }
  rpmd_ja_reference_positions_.resize(D);
  rpmd_ja_reference_positions_.copy_from_host(rpmd_ja_reference_.positions.data());
  centroid_position_work_.resize(D);
  rpmd_ja_last_wrapped_centroid_.resize(D);
  rpmd_ja_continuous_centroid_.resize(D);
  rpmd_ja_delta_current_.resize(3);
  for (auto& history : rpmd_ja_current_) history.resize(static_cast<size_t>(3) * number_of_frames);
  rpmd_ja_number_of_frames_ = number_of_frames;
  rpmd_ja_branch_error_.resize(1, 0);
  initialize_centroid<<<(N - 1) / 128 + 1, 128>>>(
    N, box, atom.position_per_atom.data(), rpmd_ja_reference_positions_.data(),
    rpmd_ja_last_wrapped_centroid_.data(), rpmd_ja_continuous_centroid_.data(),
    rpmd_ja_branch_error_.data());
  GPU_CHECK_KERNEL
  int branch_error = 0;
  rpmd_ja_branch_error_.copy_to_host(&branch_error, 1);
  if (branch_error != 0) {
    PRINT_INPUT_ERROR("rpmd_ja initial structure is non-finite or outside the reference crystal branch.");
  }

  rpmd_ja_nep_.reset(new NEP(force.primary_nep_model_path().c_str(), N, force.get_run_input()));
  rpmd_ja_nep_->N1 = 0;
  rpmd_ja_nep_->N2 = N;
  rpmd_ja_nep_->set_neighbor_rebuild(false);
  rpmd_ja_nep_->set_neighbor_log_enabled(false);
  if (rpmd_ja_reference_.backend == 1) {
    rpmd_ja_sparse_workspace_.initialize(rpmd_ja_reference_, atom.cpu_mass);
    printf(
      "rpmd_ja sparse workspace: D nnz=%llu, B^T nnz=(%llu,%llu,%llu), workspace bytes=%llu (workspace only; full GPUMD peak not measured).\n",
      static_cast<unsigned long long>(rpmd_ja_sparse_workspace_.dynamical_nnz()),
      static_cast<unsigned long long>(rpmd_ja_sparse_workspace_.site_nnz(0)),
      static_cast<unsigned long long>(rpmd_ja_sparse_workspace_.site_nnz(1)),
      static_cast<unsigned long long>(rpmd_ja_sparse_workspace_.site_nnz(2)),
      static_cast<unsigned long long>(rpmd_ja_sparse_workspace_.allocated_bytes()));
  }
  rpmd_ja_tracker_initialized_ = true;
}

void HAC::update_rpmd_ja_centroid_(const int step, const bool check_error, Atom& atom, Box& box)
{
  if (!rpmd_ja_tracker_initialized_) PRINT_INPUT_ERROR("rpmd_ja centroid tracker was not initialized.");
  update_centroid<<<(atom.number_of_atoms - 1) / 128 + 1, 128>>>(
    atom.number_of_atoms, box, atom.position_per_atom.data(), rpmd_ja_reference_positions_.data(),
    rpmd_ja_last_wrapped_centroid_.data(), rpmd_ja_continuous_centroid_.data(),
    rpmd_ja_branch_error_.data());
  GPU_CHECK_KERNEL
  if (check_error) {
    int branch_error = 0;
    rpmd_ja_branch_error_.copy_to_host(&branch_error, 1);
    if (branch_error != 0) {
      PRINT_INPUT_ERROR("rpmd_ja detected a non-finite centroid, ambiguous per-step image shift, or displacement outside the fixed reference crystal branch (fractional limit 0.45).");
    }
  }
}

void HAC::compute_rpmd_ja_current_(const int frame, Atom& atom, Box&)
{
  if (rpmd_ja_reference_.backend == 0) {
    const int D = atom.number_of_atoms * 3;
    delta_current<<<3, 128>>>(
      D, rpmd_ja_continuous_centroid_.data(), rpmd_ja_reference_positions_.data(),
      atom.velocity_per_atom.data(), rpmd_ja_delta_h_[0].data(), rpmd_ja_delta_h_[1].data(),
      rpmd_ja_delta_h_[2].data(), rpmd_ja_delta_current_.data());
  } else {
    rpmd_ja_sparse_workspace_.compute_correction(
      rpmd_ja_continuous_centroid_, rpmd_ja_reference_positions_, atom.velocity_per_atom,
      rpmd_ja_delta_current_.data(), rpmd_ja_branch_error_.data());
  }
  GPU_CHECK_KERNEL
  store_current<<<1, 128>>>(
    rpmd_ja_number_of_frames_, frame, heat_all.data(), rpmd_ja_delta_current_.data(),
    rpmd_ja_current_[0].data(), rpmd_ja_current_[1].data(), rpmd_ja_current_[2].data(),
    rpmd_ja_branch_error_.data());
  GPU_CHECK_KERNEL
  int sample_error = 0;
  rpmd_ja_branch_error_.copy_to_host(&sample_error, 1);
  if ((sample_error & RPMD_JA_ERROR_BRANCH) != 0) {
    if ((sample_error & RPMD_JA_ERROR_NUMERICAL) != 0)
      PRINT_INPUT_ERROR("rpmd_ja detected an invalid reference branch and a non-finite sparse operator or sampled centroid current.");
    PRINT_INPUT_ERROR("rpmd_ja detected a non-finite centroid, ambiguous image shift, or displacement outside the fixed reference crystal branch (fractional limit 0.45).");
  }
  if ((sample_error & RPMD_JA_ERROR_NUMERICAL) != 0)
    PRINT_INPUT_ERROR("rpmd_ja sparse operator or sampled centroid current produced a non-finite value.");
}
