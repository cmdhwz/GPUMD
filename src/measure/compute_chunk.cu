/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Compute chunk-averaged properties using coordinate-based spatial binning.

Contributed by Wenhao Luo (Sun Yat-sen University).

Syntax:
  compute_chunk <sample_interval> <output_interval>
    bin/1d <dim> <origin> <delta>
    temperature [density/number] [density/mass] [vx] [vy] [vz] [fx] [fy] [fz]

Output: compute_chunk.out
------------------------------------------------------------------------------*/

#include "compute_chunk.cuh"
#include "interface_geometry.cuh"
#include "interface_profile_math.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <cmath>
#include <limits>

#define DIM 3
#define K_B_CHUNK 8.617333262e-5

static size_t interface_profile_checked_product(const size_t a, const size_t b)
{
  if (a && b > std::numeric_limits<size_t>::max() / a) {
    PRINT_INPUT_ERROR("Interface profile allocation overflows size_t.\n");
  }
  return a * b;
}

static size_t interface_profile_checked_sum(const size_t a, const size_t b)
{
  if (b > std::numeric_limits<size_t>::max() - a) {
    PRINT_INPUT_ERROR("Interface profile allocation overflows size_t.\n");
  }
  return a + b;
}

static void interface_profile_check_gpu_fill_size(const size_t elements)
{
  if (elements > static_cast<size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("Interface profile GPU fill buffer exceeds its supported index range.\n");
  }
}

// GPU kernel: assign atoms to 1D bins
static __global__ void gpu_assign_bins_1d(
  int N, const double* x, const double* y, const double* z,
  int axis, double origin, double invdelta, int nlayers, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double pos = (axis == 0) ? x[n] : ((axis == 1) ? y[n] : z[n]);
    int bin_id = static_cast<int>((pos - origin) * invdelta);
    if (bin_id < 0) bin_id = 0;
    if (bin_id >= nlayers) bin_id = nlayers - 1;
    ichunk[n] = bin_id;
  }
}

// GPU kernel: assign atoms to 2D bins
static __global__ void gpu_assign_bins_2d(
  int N, const double* x, const double* y, const double* z,
  int axis0, int axis1,
  double origin0, double origin1,
  double invdelta0, double invdelta1,
  int nlayers0, int nlayers1, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double pos0 = (axis0 == 0) ? x[n] : ((axis0 == 1) ? y[n] : z[n]);
    double pos1 = (axis1 == 0) ? x[n] : ((axis1 == 1) ? y[n] : z[n]);
    int b0 = static_cast<int>((pos0 - origin0) * invdelta0);
    int b1 = static_cast<int>((pos1 - origin1) * invdelta1);
    if (b0 < 0) b0 = 0; if (b0 >= nlayers0) b0 = nlayers0 - 1;
    if (b1 < 0) b1 = 0; if (b1 >= nlayers1) b1 = nlayers1 - 1;
    ichunk[n] = b0 + nlayers0 * b1;
  }
}

// GPU kernel: assign atoms to 3D bins
static __global__ void gpu_assign_bins_3d(
  int N, const double* x, const double* y, const double* z,
  int axis0, int axis1, int axis2,
  double o0, double o1, double o2,
  double id0, double id1, double id2,
  int n0, int n1, int n2, int* ichunk)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double pos[3] = {x[n], y[n], z[n]};
    int b0 = static_cast<int>((pos[axis0] - o0) * id0);
    int b1 = static_cast<int>((pos[axis1] - o1) * id1);
    int b2 = static_cast<int>((pos[axis2] - o2) * id2);
    if (b0 < 0) b0 = 0; if (b0 >= n0) b0 = n0 - 1;
    if (b1 < 0) b1 = 0; if (b1 >= n1) b1 = n1 - 1;
    if (b2 < 0) b2 = 0; if (b2 >= n2) b2 = n2 - 1;
    ichunk[n] = b0 + n0 * (b1 + n1 * b2);
  }
}

// GPU kernel: count atoms per chunk
static __global__ void gpu_count_per_chunk(
  int N, const int* ichunk, int nchunk, int* count)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk) atomicAdd(&count[c], 1);
  }
}

static __global__ void gpu_mask_fixed_and_moving_atoms(
  int N, int* ichunk, const int* fixed_labels, int fixed_group,
  const int* move_labels, int move_group)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N &&
      ((fixed_group >= 0 && fixed_labels[n] == fixed_group) ||
       (move_group >= 0 && move_labels[n] == move_group))) {
    ichunk[n] = -1;
  }
}

// GPU kernel: sum scalar property per chunk
static __global__ void gpu_sum_scalar_per_chunk(
  int N, const int* ichunk, const double* prop,
  int nchunk, int offset, int nvalues, double* out)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk)
      atomicAdd(&out[c * nvalues + offset], prop[n]);
  }
}

// GPU kernel: sum kinetic energy per chunk (for temperature)
static __global__ void gpu_sum_ke_per_chunk(
  int N, const int* ichunk, const double* mass,
  const double* vx, const double* vy, const double* vz,
  int nchunk, int offset, int nvalues, double* out)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk) {
      double v2 = vx[n] * vx[n] + vy[n] * vy[n] + vz[n] * vz[n];
      atomicAdd(&out[c * nvalues + offset], 0.5 * mass[n] * v2);
    }
  }
}

static __global__ void gpu_sum_mass_momentum_per_chunk(
  int N, const int* ichunk, const double* mass,
  const double* vx, const double* vy, const double* vz,
  int nchunk, double* out)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int c = ichunk[n];
    if (c >= 0 && c < nchunk) {
      double m = mass[n];
      atomicAdd(&out[c * 4], m);
      atomicAdd(&out[c * 4 + 1], m * vx[n]);
      atomicAdd(&out[c * 4 + 2], m * vy[n]);
      atomicAdd(&out[c * 4 + 3], m * vz[n]);
    }
  }
}

static __global__ void gpu_accumulate_interface_profile_sample(
  const int nchunk,
  const int number_of_scalars,
  const int* count,
  const double* values,
  const bool remove_drift,
  const double* mass_momentum,
  InterfaceProfileBlockAccum* block_accum,
  int* error_info,
  double* error_values,
  const int step)
{
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin >= nchunk) return;
  const double mass_sum = remove_drift ? mass_momentum[bin * 4] : 0.0;
  const double px = remove_drift ? mass_momentum[bin * 4 + 1] : 0.0;
  const double py = remove_drift ? mass_momentum[bin * 4 + 2] : 0.0;
  const double pz = remove_drift ? mass_momentum[bin * 4 + 3] : 0.0;
  const double raw_kinetic = values[bin * number_of_scalars];
  const double potential = values[bin * number_of_scalars + 1];
  const int error = interface_profile_accumulate_sample(
    block_accum[bin], count[bin], remove_drift, raw_kinetic, potential,
    mass_sum, px, py, pz, K_B);
  if (error != INTERFACE_PROFILE_SAMPLE_OK) {
    interface_profile_record_first_error(
      error_info, error_values, error, step, bin, raw_kinetic,
      mass_sum, px, py, pz, potential);
  }
}

static __global__ void gpu_reset_interface_profile_block(
  const int nchunk,
  InterfaceProfileBlockAccum* block_accum,
  int* error_info,
  double* error_values)
{
  const int bin = blockIdx.x * blockDim.x + threadIdx.x;
  if (bin < nchunk) interface_profile_reset_block_accum(block_accum[bin]);
  if (bin == 0) interface_profile_reset_error(error_info, error_values);
}

ComputeChunk::ComputeChunk(const std::vector<std::string>& tokens, Box& box)
{
  action_name = "compute_chunk";
  for (int i = 0; i < 3; i++) {
    axis_[i] = 0; origin_[i] = 0.0;
    delta_[i] = 1.0; invdelta_[i] = 1.0; nlayers_[i] = 1;
  }
  parse(tokens, box);
}

ComputeChunk::ComputeChunk(
  const std::vector<std::string>& tokens,
  const std::shared_ptr<InterfaceGeometry>& interface_geometry)
  : interface_geometry_(interface_geometry)
{
  action_name = "compute_interface_profile";
  dim_ = 1;
  ncoord_ = 1;
  for (int i = 0; i < 3; i++) {
    axis_[i] = 0; origin_[i] = 0.0;
    delta_[i] = 1.0; invdelta_[i] = 1.0; nlayers_[i] = 1;
  }
  if (tokens.size() != 5 || !is_valid_int(tokens[1], &sample_interval_) || sample_interval_ <= 0 ||
      !is_valid_int(tokens[2], &output_interval_) || output_interval_ <= 0) {
    PRINT_INPUT_ERROR(
      "compute_interface_profile syntax: <sample_steps> <samples_per_block> <yes|no> <output_file>.\n");
  }
  if (tokens[3] == "yes") remove_drift_ = true;
  else if (tokens[3] != "no") PRINT_INPUT_ERROR("Drift option must be yes or no.\n");
  if (tokens[4].empty()) PRINT_INPUT_ERROR("compute_interface_profile output file cannot be empty.\n");
  interface_output_file_ = tokens[4];
  interface_mode_ = true;
  geometry_mode_ = true;
  compute_temperature_ = 1;
  compute_potential_ = 1;
  number_of_scalars_ = 2;
}

void ComputeChunk::parse(const std::vector<std::string>& tokens, Box& box)
{
  printf("Compute chunk-averaged properties:\n");
  const int num_param = tokens.size();

  if (num_param < 6) {
    PRINT_INPUT_ERROR("compute_chunk requires at least 5 parameters.\n");
  }

  // param[1]: sample_interval
  if (!is_valid_int(tokens[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample_interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample_interval should > 0.\n");
  }

  // param[2]: output_interval
  if (!is_valid_int(tokens[2], &output_interval_)) {
    PRINT_INPUT_ERROR("output_interval should be an integer.\n");
  }
  if (output_interval_ <= 0) {
    PRINT_INPUT_ERROR("output_interval should > 0.\n");
  }

  // param[3]: bin style -> parse bin params, returns index of next param
  bool interface_requested = false;
  for (const std::string& token : tokens) {
    if (token == "interface") interface_requested = true;
  }
  int next = parse_bin_params(tokens, num_param, 3, box, interface_requested);

  // remaining params are properties, or one opt-in interface profile option
  if (next >= num_param) {
    PRINT_INPUT_ERROR("compute_chunk requires at least one property.\n");
  }

  for (int k = next; k < num_param; ++k) {
    if (tokens[k] == "interface") {
      if (interface_mode_ || number_of_scalars_ != 0 || dim_ != 1 || k + 6 != num_param) {
        PRINT_INPUT_ERROR(
          "interface requires 1D binning and must be the only property option.\n");
      }
      if (!is_valid_real(tokens[k + 1], &interface_coordinate_) ||
          !is_valid_int(tokens[k + 2], &material_lower_id_) ||
          !is_valid_int(tokens[k + 3], &material_upper_id_)) {
        PRINT_INPUT_ERROR("Invalid interface coordinate or material ID.\n");
      }
      if (!std::isfinite(interface_coordinate_) || !std::isfinite(delta_[0])) {
        PRINT_INPUT_ERROR("Interface coordinate and bin width must be finite.\n");
      }
      if (material_lower_id_ < 0 || material_upper_id_ < 0 ||
          material_lower_id_ == material_upper_id_) {
        PRINT_INPUT_ERROR("Interface material IDs should be distinct nonnegative integers.\n");
      }
      if (tokens[k + 4] == "yes") {
        remove_drift_ = true;
      } else if (tokens[k + 4] != "no") {
        PRINT_INPUT_ERROR("interface drift option should be yes or no.\n");
      }
      interface_output_file_ = tokens[k + 5];
      interface_mode_ = true;
      compute_temperature_ = 1;
      compute_potential_ = 1;
      number_of_scalars_ = 2;
      printf(
        "    interface profile at %g A, material IDs %d/%d, remove drift %s.\n",
        interface_coordinate_, material_lower_id_, material_upper_id_,
        remove_drift_ ? "yes" : "no");
      break;
    } else if (tokens[k] == "temperature") {
      compute_temperature_ = 1;
      number_of_scalars_++;
      printf("    temperature\n");
    } else if (tokens[k] == "density/number") {
      compute_density_number_ = 1;
      number_of_scalars_++;
      printf("    density/number\n");
    } else if (tokens[k] == "density/mass") {
      compute_density_mass_ = 1;
      number_of_scalars_++;
      printf("    density/mass\n");
    } else if (tokens[k] == "vx") {
      compute_vx_ = 1; number_of_scalars_++; printf("    vx\n");
    } else if (tokens[k] == "vy") {
      compute_vy_ = 1; number_of_scalars_++; printf("    vy\n");
    } else if (tokens[k] == "vz") {
      compute_vz_ = 1; number_of_scalars_++; printf("    vz\n");
    } else if (tokens[k] == "fx") {
      compute_fx_ = 1; number_of_scalars_++; printf("    fx\n");
    } else if (tokens[k] == "fy") {
      compute_fy_ = 1; number_of_scalars_++; printf("    fy\n");
    } else if (tokens[k] == "fz") {
      compute_fz_ = 1; number_of_scalars_++; printf("    fz\n");
    } else {
      PRINT_INPUT_ERROR("Invalid property for compute_chunk.\n");
    }
  }

  printf("    with %dD binning, %d chunks.\n", dim_, nchunk_);
  printf("    with sampling interval %d.\n", sample_interval_);
  printf("    and output interval %d.\n", output_interval_);
}

static int parse_one_axis(
  const std::vector<std::string>& tokens, int idx, int* axis, double* origin, double* delta,
  double* invdelta, int* nlayers, double* box_len_out, const Box& box,
  bool require_finite_width)
{
  // Use geometric thickness (volume/area) which is correct for triclinic boxes
  double vol = box.get_volume();
  double box_lengths[3] = {
    vol / box.get_area(0),  // thickness along x
    vol / box.get_area(1),  // thickness along y
    vol / box.get_area(2)   // thickness along z
  };

  if (tokens[idx] == "x") *axis = 0;
  else if (tokens[idx] == "y") *axis = 1;
  else if (tokens[idx] == "z") *axis = 2;
  else { PRINT_INPUT_ERROR("dim must be x, y, or z.\n"); }

  double box_len = box_lengths[*axis];

  // GPUMD wraps positions to [0, L], bins always start from 0.
  if (tokens[idx + 1] == "lower") {
    *origin = 0.0;
  } else {
    PRINT_INPUT_ERROR("origin must be lower.\n");
  }

  if (!is_valid_real(tokens[idx + 2], delta)) {
    PRINT_INPUT_ERROR("delta must be a positive number.\n");
  }
  if (require_finite_width && !std::isfinite(*delta)) {
    PRINT_INPUT_ERROR("Interface bin width must be finite.\n");
  }
  if (*delta <= 0.0) {
    PRINT_INPUT_ERROR("delta must be positive.\n");
  }
  *invdelta = 1.0 / (*delta);
  if (require_finite_width && !std::isfinite(*invdelta)) {
    PRINT_INPUT_ERROR("Interface inverse bin width must be finite.\n");
  }
  // Interface mode validates before converting the possibly unbounded quotient to int.
  if (require_finite_width && (!std::isfinite(box_len) || box_len <= 0.0)) {
    PRINT_INPUT_ERROR("Interface box length must be finite and positive.\n");
  }
  const double layer_count = ceil(box_len / (*delta));
  if (require_finite_width &&
      (!std::isfinite(layer_count) || layer_count <= 0.0 ||
       layer_count > std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("Interface bin count is outside the supported integer range.\n");
  }
  // Use ceil so the last bin is not wider than delta
  *nlayers = static_cast<int>(layer_count);
  if (*nlayers <= 0) *nlayers = 1;
  *box_len_out = box_len;

  return idx + 3;
}

int ComputeChunk::parse_bin_params(
  const std::vector<std::string>& tokens, int num_param, int start, Box& box,
  bool require_finite_width)
{
  if (tokens[start] == "bin/1d") {
    dim_ = 1; ncoord_ = 1;
    if (start + 4 > num_param) {
      PRINT_INPUT_ERROR("bin/1d requires: dim origin delta.\n");
    }
    int next = parse_one_axis(
      tokens, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box, require_finite_width);
    nchunk_ = nlayers_[0];
    return next;
  } else if (tokens[start] == "bin/2d") {
    dim_ = 2; ncoord_ = 2;
    if (start + 7 > num_param) {
      PRINT_INPUT_ERROR("bin/2d requires: dim origin delta dim origin delta.\n");
    }
    int next = parse_one_axis(
      tokens, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box, require_finite_width);
    next = parse_one_axis(
      tokens, next, &axis_[1], &origin_[1], &delta_[1],
      &invdelta_[1], &nlayers_[1], &box_length_[1], box, require_finite_width);
    if (axis_[0] == axis_[1]) {
      PRINT_INPUT_ERROR("bin/2d requires two different axes.\n");
    }
    nchunk_ = nlayers_[0] * nlayers_[1];
    return next;
  } else if (tokens[start] == "bin/3d") {
    dim_ = 3; ncoord_ = 3;
    if (start + 10 > num_param) {
      PRINT_INPUT_ERROR("bin/3d requires: dim origin delta (x3).\n");
    }
    int next = parse_one_axis(
      tokens, start + 1, &axis_[0], &origin_[0], &delta_[0],
      &invdelta_[0], &nlayers_[0], &box_length_[0], box, require_finite_width);
    next = parse_one_axis(
      tokens, next, &axis_[1], &origin_[1], &delta_[1],
      &invdelta_[1], &nlayers_[1], &box_length_[1], box, require_finite_width);
    next = parse_one_axis(
      tokens, next, &axis_[2], &origin_[2], &delta_[2],
      &invdelta_[2], &nlayers_[2], &box_length_[2], box, require_finite_width);
    if (axis_[0] == axis_[1] || axis_[0] == axis_[2] || axis_[1] == axis_[2]) {
      PRINT_INPUT_ERROR("bin/3d requires three different axes.\n");
    }
    nchunk_ = nlayers_[0] * nlayers_[1] * nlayers_[2];
    return next;
  } else {
    PRINT_INPUT_ERROR("bin style must be bin/1d, bin/2d, or bin/3d.\n");
    return start;
  }
}

void ComputeChunk::pre_run(
  const int number_of_steps, const double time_step,
  Integrate& integrate, std::vector<Group>& group,
  Atom& atom, Box& box, Force& force)
{
  if (number_of_scalars_ == 0) return;

  if (interface_mode_ && !geometry_mode_) {
    if (!box.is_orthogonal) {
      PRINT_INPUT_ERROR("Interface profiles require an orthogonal box.\n");
    }
    const double area = box.get_area(axis_[0]);
    const double box_length = box.get_volume() / area;
    const double layer_count = ceil(box_length / delta_[0]);
    if (!std::isfinite(area) || area <= 0.0 || !std::isfinite(box_length) ||
        box_length <= 0.0 || !std::isfinite(layer_count) || layer_count <= 0.0 ||
        layer_count > std::numeric_limits<int>::max()) {
      PRINT_INPUT_ERROR("Interface profile box geometry is invalid.\n");
    }
    if (interface_coordinate_ <= origin_[0] ||
        interface_coordinate_ >= origin_[0] + box_length) {
      PRINT_INPUT_ERROR("Interface coordinate must lie inside the binning range.\n");
    }
    box_length_[0] = box_length;
    invdelta_[0] = 1.0 / delta_[0];
    nlayers_[0] = static_cast<int>(layer_count);
    nchunk_ = nlayers_[0];
  }
  if (geometry_mode_) {
    if (is_pimd(integrate.get_type())) {
      PRINT_INPUT_ERROR("compute_interface_profile with shared geometry supports classical MD only.\n");
    }
    if (!interface_geometry_ || !interface_geometry_->initialized()) {
      PRINT_INPUT_ERROR("Shared interface geometry must be initialized before the profile.\n");
    }
    if (interface_geometry_->atom_bin().size() != static_cast<size_t>(atom.number_of_atoms)) {
      PRINT_INPUT_ERROR("Shared interface geometry atom count differs from the current model.\n");
    }
    axis_[0] = interface_geometry_->axis();
    origin_[0] = 0.0;
    delta_[0] = interface_geometry_->bin_width_A();
    invdelta_[0] = 1.0 / delta_[0];
    box_length_[0] = interface_geometry_->box_length_A();
    nlayers_[0] = interface_geometry_->bin_count();
    nchunk_ = nlayers_[0];
  }

  if (interface_mode_) {
    if (number_of_steps < 0 || nchunk_ <= 0) {
      PRINT_INPUT_ERROR("Interface profile requires a nonnegative run length and at least one bin.\n");
    }
    const size_t count_values = static_cast<size_t>(nchunk_);
    const size_t scalar_values = interface_profile_checked_product(
      count_values, static_cast<size_t>(number_of_scalars_));
    const size_t auxiliary_values = remove_drift_
      ? interface_profile_checked_product(count_values, 4) : 0;
    interface_profile_check_gpu_fill_size(count_values);
    interface_profile_check_gpu_fill_size(scalar_values);
    if (remove_drift_) interface_profile_check_gpu_fill_size(auxiliary_values);

    interface_expected_sample_count_ =
      static_cast<size_t>(number_of_steps) / static_cast<size_t>(sample_interval_);
    interface_block_cache_capacity_ = interface_profile_block_cache_capacity(
      interface_expected_sample_count_, static_cast<size_t>(output_interval_));
    interface_cached_block_count_ = 0;
    const size_t cache_values = interface_profile_checked_product(
      interface_block_cache_capacity_, count_values);
    const size_t cpu_cache_bytes = interface_profile_checked_sum(
      interface_profile_checked_product(cache_values, sizeof(InterfaceProfileBlockAccum)),
      interface_profile_checked_product(
        interface_block_cache_capacity_, sizeof(InterfaceProfileBlockMetadata)));
    size_t gpu_cache_bytes = interface_profile_checked_product(
      count_values, sizeof(InterfaceProfileBlockAccum));
    gpu_cache_bytes = interface_profile_checked_sum(
      gpu_cache_bytes, interface_profile_checked_product(count_values, sizeof(int)));
    gpu_cache_bytes = interface_profile_checked_sum(
      gpu_cache_bytes, interface_profile_checked_product(scalar_values, sizeof(double)));
    gpu_cache_bytes = interface_profile_checked_sum(
      gpu_cache_bytes, interface_profile_checked_product(auxiliary_values, sizeof(double)));
    gpu_cache_bytes = interface_profile_checked_sum(gpu_cache_bytes, 3 * sizeof(int));
    gpu_cache_bytes = interface_profile_checked_sum(
      gpu_cache_bytes, INTERFACE_PROFILE_ERROR_VALUE_COUNT * sizeof(double));
    if (cache_values > std::vector<InterfaceProfileBlockAccum>().max_size() ||
        interface_block_cache_capacity_ >
          std::vector<InterfaceProfileBlockMetadata>().max_size()) {
      PRINT_INPUT_ERROR("Interface profile CPU cache exceeds vector capacity.\n");
    }
    printf(
      "    interface profile buffers: CPU block cache %zu bytes, GPU reduction/accumulation %zu bytes (%zu blocks).\n",
      cpu_cache_bytes, gpu_cache_bytes, interface_block_cache_capacity_);

    cpu_interface_block_cache_.resize(cache_values);
    cpu_interface_block_metadata_.resize(interface_block_cache_capacity_);
    gpu_interface_block_accum_.resize(count_values);
    gpu_interface_error_info_.resize(3);
    gpu_interface_error_values_.resize(INTERFACE_PROFILE_ERROR_VALUE_COUNT);
    if (remove_drift_) gpu_auxiliary_.resize(auxiliary_values);

    for (int i = 0; i < 9; ++i) interface_initial_box_h_[i] = box.cpu_h[i];
    interface_initial_pbc_[0] = box.pbc_x;
    interface_initial_pbc_[1] = box.pbc_y;
    interface_initial_pbc_[2] = box.pbc_z;
    last_bath_source_eV_ = std::numeric_limits<double>::quiet_NaN();
    last_bath_sink_eV_ = std::numeric_limits<double>::quiet_NaN();
    FILE* existing_file = fopen(interface_output_file_.c_str(), "r");
    if (existing_file) {
      fclose(existing_file);
      PRINT_INPUT_ERROR("Interface output file already exists; choose a new filename.\n");
    }
  } else {
    cpu_count_sum_.resize(nchunk_, 0);
    cpu_values_sum_.resize(nchunk_ * number_of_scalars_, 0.0);
  }
  ichunk_.resize(atom.number_of_atoms);
  gpu_count_.resize(nchunk_);
  gpu_values_.resize(nchunk_ * number_of_scalars_);
  num_samples_ = 0;
  if (interface_mode_) reset_interface_block();

  calculate_chunk_volumes(box);
  calculate_chunk_coords(box);

  if (geometry_mode_) {
    std::vector<int> fixed_bin = interface_geometry_->atom_bin();
    for (size_t n = 0; n < fixed_bin.size(); ++n) {
      fixed_bin[n] = interface_profile_atom_bin(
        fixed_bin[n], interface_geometry_->atom_fixed()[n] != 0,
        interface_geometry_->atom_moving()[n] != 0);
    }
    ichunk_.copy_from_host(fixed_bin.data(), fixed_bin.size());
  }

  if (interface_mode_) {
    fid_ = my_fopen(interface_output_file_.c_str(), "w");
    fprintf(fid_, geometry_mode_
      ? "# compute_interface_profile version 3\n"
      : "# compute_chunk interface_profile version 2\n");
    fprintf(fid_, "# axis %c\n", "xyz"[axis_[0]]);
    fprintf(fid_, "# timestep_fs %.10e\n", time_step * TIME_UNIT_CONVERSION);
    fprintf(fid_, "# sample_interval %d\n", sample_interval_);
    fprintf(fid_, "# output_interval_samples %d\n", output_interval_);
    fprintf(fid_, "# bin_width_A %.10e\n", delta_[0]);
    fprintf(fid_, "# box_thickness_A %.10e\n", box_length_[0]);
    fprintf(fid_, "# cross_section_A2 %.10e\n", box.get_area(axis_[0]));
    fprintf(fid_, "# pbc %d %d %d\n", box.pbc_x, box.pbc_y, box.pbc_z);
    fprintf(fid_, "# geometry_valid orthogonal_static_runtime_checked\n");
    fprintf(fid_, "# bin_count %d\n", nchunk_);
    if (geometry_mode_) {
      fprintf(fid_, "# geometry_source %s_geometry.json\n", interface_geometry_->output_prefix().c_str());
      fprintf(fid_, "# fixed_reference_mapping yes\n");
      fprintf(fid_, "# interfaces_A");
      for (double plane : interface_geometry_->interfaces_A()) fprintf(fid_, " %.10e", plane);
      fprintf(fid_, "\n# material_id and bath_mask come from prefix_bins.dat\n");
    } else {
      fprintf(fid_, "# interface_coordinate_A %.10e\n", interface_coordinate_);
      fprintf(fid_, "# material_lower_id %d\n", material_lower_id_);
      fprintf(fid_, "# material_upper_id %d\n", material_upper_id_);
    }
    fprintf(fid_, "# remove_bin_drift %d\n", remove_drift_ ? 1 : 0);
    fprintf(fid_, "# temperature_dof 3*N-3*remove_bin_drift\n");
    fprintf(fid_, "# bath_energy_sign system_to_bath\n");
    fprintf(fid_, geometry_mode_
      ? "# columns step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A coordinate_center_A material_id bath_mask interface_crossing atom_count kinetic_energy_eV potential_energy_eV temperature_K dof samples temperature_samples bath_source_eV bath_sink_eV\n"
      : "# columns step time_ps block_start_ps bin_id coordinate_min_A coordinate_max_A coordinate_center_A material_id atom_count kinetic_energy_eV potential_energy_eV temperature_K dof samples temperature_samples bath_source_eV bath_sink_eV\n");
    fflush(fid_);
  } else {
    fid_ = my_fopen("compute_chunk.out", "a");
  }
}

void ComputeChunk::end_of_step(
  const int number_of_steps, int step,
  const int fixed_group, const int move_group,
  const double global_time, const double temperature,
  Integrate& integrate, Box& box, std::vector<Group>& group,
  GPU_Vector<double>& thermo, Atom& atom, Force& force)
{
  if (number_of_scalars_ == 0) return;
  if (interface_mode_ && !geometry_mode_) {
    if (!box.is_orthogonal || box.pbc_x != interface_initial_pbc_[0] ||
        box.pbc_y != interface_initial_pbc_[1] || box.pbc_z != interface_initial_pbc_[2]) {
      PRINT_INPUT_ERROR("Interface profiles do not support changing box geometry or PBC.\n");
    }
    for (int i = 0; i < 9; ++i) {
      if (box.cpu_h[i] != interface_initial_box_h_[i]) {
        PRINT_INPUT_ERROR("Interface profiles do not support changing box geometry or PBC.\n");
      }
    }
  }
  if ((step + 1) % sample_interval_ != 0) return;

  if (interface_mode_ && num_samples_ == 0) {
    block_start_time_ps_ = global_time * TIME_UNIT_CONVERSION / 1000.0;
  }
  if (!geometry_mode_) assign_chunks(atom, box);
  if (interface_mode_ && !geometry_mode_) mask_fixed_and_moving_atoms(group, integrate, fixed_group, move_group);
  sample(atom, step + 1);
  num_samples_++;
  if (interface_mode_) {
    last_step_ = step + 1;
    last_time_ps_ = global_time * TIME_UNIT_CONVERSION / 1000.0;
    const EnsembleType type = integrate.get_type();
    if (type == EnsembleType::HEAT_LAN || type == EnsembleType::HEAT_NHC ||
        type == EnsembleType::HEAT_BDP) {
      const double* bath_energy = integrate.get_energy_transferred();
      last_bath_source_eV_ = bath_energy[0];
      last_bath_sink_eV_ = bath_energy[1];
    } else {
      last_bath_source_eV_ = std::numeric_limits<double>::quiet_NaN();
      last_bath_sink_eV_ = std::numeric_limits<double>::quiet_NaN();
    }
  }

  // output_interval_ means "output every N samples" (same as compute command)
  if (num_samples_ % output_interval_ == 0) {
    if (interface_mode_) {
      cache_interface_block();
      reset_interface_block();
    } else {
      output_results(step + 1);
      std::fill(cpu_count_sum_.begin(), cpu_count_sum_.end(), 0);
      std::fill(cpu_values_sum_.begin(), cpu_values_sum_.end(), 0.0);
      num_samples_ = 0;
    }
  }
}

void ComputeChunk::post_run(
  Atom& atom, Box& box, Integrate& integrate,
  const int number_of_steps, const double time_step, const double temperature)
{
  if (number_of_scalars_ == 0) return;
  if (interface_mode_) {
    if (num_samples_ > 0) {
      cache_interface_block();
      reset_interface_block();
    }
    for (std::size_t block = 0; block < interface_cached_block_count_; ++block) {
      output_interface_results(block);
    }
    if (fid_) fflush(fid_);
  }
  if (fid_) fclose(fid_);
  fid_ = nullptr;

  compute_temperature_ = 0;
  compute_potential_ = 0;
  compute_density_number_ = 0;
  compute_density_mass_ = 0;
  compute_vx_ = compute_vy_ = compute_vz_ = 0;
  compute_fx_ = compute_fy_ = compute_fz_ = 0;
  interface_mode_ = false;
  geometry_mode_ = false;
  interface_geometry_.reset();
  remove_drift_ = false;
  interface_output_file_.clear();
  number_of_scalars_ = 0;
}

void ComputeChunk::assign_chunks(const Atom& atom, const Box& box)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  const double* x = atom.position_per_atom.data();
  const double* y = x + N;
  const double* z = y + N;

  if (dim_ == 1) {
    gpu_assign_bins_1d<<<grid_size, block_size>>>(
      N, x, y, z, axis_[0], origin_[0], invdelta_[0], nlayers_[0],
      ichunk_.data());
  } else if (dim_ == 2) {
    gpu_assign_bins_2d<<<grid_size, block_size>>>(
      N, x, y, z, axis_[0], axis_[1],
      origin_[0], origin_[1], invdelta_[0], invdelta_[1],
      nlayers_[0], nlayers_[1], ichunk_.data());
  } else {
    gpu_assign_bins_3d<<<grid_size, block_size>>>(
      N, x, y, z,
      axis_[0], axis_[1], axis_[2],
      origin_[0], origin_[1], origin_[2],
      invdelta_[0], invdelta_[1], invdelta_[2],
      nlayers_[0], nlayers_[1], nlayers_[2], ichunk_.data());
  }
  GPU_CHECK_KERNEL
}

void ComputeChunk::mask_fixed_and_moving_atoms(
  const std::vector<Group>& group, const Integrate& integrate,
  const int fixed_group, const int move_group)
{
  if (fixed_group < 0 && move_group < 0) return;

  const int N = ichunk_.size();
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;
  const int fixed_method = integrate.get_fixed_grouping_method();
  const int move_method = integrate.get_move_grouping_method();
  const int* fixed_labels =
    (fixed_group >= 0) ? group[fixed_method].label.data() : nullptr;
  const int* move_labels =
    (move_group >= 0) ? group[move_method].label.data() : nullptr;

  gpu_mask_fixed_and_moving_atoms<<<grid_size, block_size>>>(
    N, ichunk_.data(), fixed_labels, fixed_group, move_labels, move_group);
  GPU_CHECK_KERNEL
}

void ComputeChunk::sample(const Atom& atom, const int step)
{
  const int N = atom.number_of_atoms;
  const int block_size = 128;
  const int grid_size = (N - 1) / block_size + 1;

  gpu_count_.fill(0);
  gpu_values_.fill(0.0);

  gpu_count_per_chunk<<<grid_size, block_size>>>(
    N, ichunk_.data(), nchunk_, gpu_count_.data());
  GPU_CHECK_KERNEL

  const double* mass = atom.mass.data();
  const double* vx = atom.velocity_per_atom.data();
  const double* vy = vx + N;
  const double* vz = vy + N;
  const double* fx = atom.force_per_atom.data();
  const double* fy = fx + N;
  const double* fz = fy + N;

  int idx = 0;
  if (compute_temperature_) {
    gpu_sum_ke_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), mass, vx, vy, vz,
      nchunk_, idx, number_of_scalars_, gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_potential_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), atom.potential_per_atom.data(), nchunk_, idx,
      number_of_scalars_, gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_density_number_) { idx++; } // count-based, no kernel needed
  if (compute_density_mass_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), mass, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vx_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vx, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vy_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vy, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_vz_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), vz, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fx_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fx, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fy_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fy, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }
  if (compute_fz_) {
    gpu_sum_scalar_per_chunk<<<grid_size, block_size>>>(
      N, ichunk_.data(), fz, nchunk_, idx, number_of_scalars_,
      gpu_values_.data());
    GPU_CHECK_KERNEL
    idx++;
  }

  if (interface_mode_) {
    if (remove_drift_) {
      gpu_auxiliary_.fill(0.0);
      gpu_sum_mass_momentum_per_chunk<<<grid_size, block_size>>>(
        N, ichunk_.data(), mass, vx, vy, vz, nchunk_, gpu_auxiliary_.data());
      GPU_CHECK_KERNEL
    }
    gpu_accumulate_interface_profile_sample<<<
      (nchunk_ - 1) / block_size + 1, block_size>>>(
        nchunk_, number_of_scalars_, gpu_count_.data(), gpu_values_.data(),
        remove_drift_, remove_drift_ ? gpu_auxiliary_.data() : nullptr,
        gpu_interface_block_accum_.data(), gpu_interface_error_info_.data(),
        gpu_interface_error_values_.data(), step);
    GPU_CHECK_KERNEL
    return;
  }

  // The non-interface compute_chunk path retains its existing per-sample host reduction.
  std::vector<int> cpu_count(nchunk_);
  std::vector<double> cpu_values(nchunk_ * number_of_scalars_);
  gpu_count_.copy_to_host(cpu_count.data(), nchunk_);
  gpu_values_.copy_to_host(cpu_values.data(), nchunk_ * number_of_scalars_);
  for (int i = 0; i < nchunk_; i++) cpu_count_sum_[i] += cpu_count[i];
  for (int i = 0; i < nchunk_ * number_of_scalars_; i++)
    cpu_values_sum_[i] += cpu_values[i];
}

void ComputeChunk::output_results(int step)
{
  if (num_samples_ == 0) return;

  for (int chunk = 0; chunk < nchunk_; chunk++) {
    double count = static_cast<double>(cpu_count_sum_[chunk]) / num_samples_;

    // write: chunk_id coord(s)
    fprintf(fid_, "%d ", chunk);
    for (int c = 0; c < ncoord_; c++)
      fprintf(fid_, "%.6f ", chunk_coords_cpu_[chunk * ncoord_ + c]);
    fprintf(fid_, "%.1f ", count);

    int idx = 0;
    if (compute_temperature_) {
      double ke = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      double temp = (count > 0) ? (2.0 * ke / (K_B_CHUNK * DIM * count)) : 0.0;
      fprintf(fid_, "%.10e ", temp);
      idx++;
    }
    if (compute_density_number_) {
      double dn = count / chunk_volume_cpu_[chunk];
      fprintf(fid_, "%.10e ", dn);
      idx++;
    }
    if (compute_density_mass_) {
      double dm = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      dm /= chunk_volume_cpu_[chunk];
      fprintf(fid_, "%.10e ", dm);
      idx++;
    }
    if (compute_vx_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_vy_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_vz_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fx_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fy_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    if (compute_fz_) {
      double val = cpu_values_sum_[chunk * number_of_scalars_ + idx] / num_samples_;
      if (count > 0) val /= count;
      fprintf(fid_, "%.10e ", val);
      idx++;
    }
    fprintf(fid_, "\n");
  }
  fflush(fid_);
}

void ComputeChunk::cache_interface_block()
{
  if (num_samples_ == 0) return;

  const std::size_t block_index = interface_cached_block_count_;
  const std::size_t expected_samples = interface_profile_expected_block_samples(
    interface_expected_sample_count_, static_cast<std::size_t>(output_interval_), block_index);
  if (expected_samples != static_cast<std::size_t>(num_samples_) ||
      block_index >= interface_block_cache_capacity_) {
    PRINT_INPUT_ERROR("Interface profile block cache does not match the sampled run length.\n");
  }
  std::size_t cache_offset = 0;
  if (!interface_profile_block_cache_offset(
        block_index, static_cast<std::size_t>(nchunk_),
        interface_block_cache_capacity_, cache_offset) ||
      cache_offset > cpu_interface_block_cache_.size() ||
      static_cast<std::size_t>(nchunk_) > cpu_interface_block_cache_.size() - cache_offset) {
    PRINT_INPUT_ERROR("Interface profile block cache index is outside its allocation.\n");
  }

  gpu_interface_error_info_.copy_to_host(interface_error_info_host_, 3);
  if (interface_error_info_host_[0] != INTERFACE_PROFILE_SAMPLE_OK) {
    gpu_interface_error_values_.copy_to_host(
      interface_error_values_host_, INTERFACE_PROFILE_ERROR_VALUE_COUNT);
    const char* reason = interface_error_info_host_[0] == INTERFACE_PROFILE_SAMPLE_INVALID
      ? "invalid or non-finite per-sample data"
      : interface_error_info_host_[0] == INTERFACE_PROFILE_SAMPLE_COUNTER_OVERFLOW
        ? "sample counter overflow" : "non-finite accumulated block sum";
    fprintf(
      stderr,
      "Interface profile rejected %s at step %d, bin %d (raw kinetic %.17g eV, "
      "mass %.17g, momentum %.17g %.17g %.17g, potential %.17g eV).\n",
      reason, interface_error_info_host_[1], interface_error_info_host_[2],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_RAW_KINETIC],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_MASS_SUM],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_MOMENTUM_X],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_MOMENTUM_Y],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_MOMENTUM_Z],
      interface_error_values_host_[INTERFACE_PROFILE_ERROR_POTENTIAL]);
    PRINT_INPUT_ERROR("Interface profile rejected the sampled block after its GPU error flag was checked.\n");
  }

  gpu_interface_block_accum_.copy_to_host(
    cpu_interface_block_cache_.data() + cache_offset, static_cast<size_t>(nchunk_));
  for (int bin = 0; bin < nchunk_; ++bin) {
    const InterfaceProfileBlockAccum& accum = cpu_interface_block_cache_[cache_offset + bin];
    if (!std::isfinite(accum.kinetic_sum) || !std::isfinite(accum.potential_sum) ||
        !std::isfinite(accum.temperature_sum) || !std::isfinite(accum.dof_sum) ||
        accum.temperature_samples > static_cast<unsigned long long>(num_samples_)) {
      PRINT_INPUT_ERROR("Interface profile rejected a non-finite or inconsistent GPU block sum.\n");
    }
  }
  InterfaceProfileBlockMetadata& metadata = cpu_interface_block_metadata_[block_index];
  metadata.step = last_step_;
  metadata.time_ps = last_time_ps_;
  metadata.start_time_ps = block_start_time_ps_;
  metadata.samples = num_samples_;
  metadata.bath_source_eV = last_bath_source_eV_;
  metadata.bath_sink_eV = last_bath_sink_eV_;
  ++interface_cached_block_count_;
}

void ComputeChunk::output_interface_results(const std::size_t block_index)
{
  if (block_index >= interface_cached_block_count_) return;
  const InterfaceProfileBlockMetadata& metadata = cpu_interface_block_metadata_[block_index];
  std::size_t cache_offset = 0;
  if (!interface_profile_block_cache_offset(
        block_index, static_cast<std::size_t>(nchunk_),
        interface_block_cache_capacity_, cache_offset)) {
    PRINT_INPUT_ERROR("Interface profile block cache index is invalid during output.\n");
  }

  for (int chunk = 0; chunk < nchunk_; ++chunk) {
    const InterfaceProfileBlockAccum& accum = cpu_interface_block_cache_[cache_offset + chunk];
    const double coordinate_min = origin_[0] + chunk * delta_[0];
    const double remainder = box_length_[0] - (nlayers_[0] - 1) * delta_[0];
    const double width = (chunk < nlayers_[0] - 1) ? delta_[0] : remainder;
    const double coordinate_max = coordinate_min + width;
    int material_id = -1;
    int bath_mask = 0;
    int interface_crossing = 0;
    if (geometry_mode_) {
      const InterfaceBinGeometry& bin = interface_geometry_->bins()[chunk];
      material_id = bin.material_id;
      bath_mask = bin.bath_mask;
      interface_crossing = bin.interface_crossing;
    } else if (coordinate_max <= interface_coordinate_) {
      material_id = material_lower_id_;
    } else if (coordinate_min >= interface_coordinate_) {
      material_id = material_upper_id_;
    }

    const double inv_samples = 1.0 / metadata.samples;
    const double temperature = accum.temperature_samples > 0
      ? accum.temperature_sum / static_cast<double>(accum.temperature_samples)
      : std::numeric_limits<double>::quiet_NaN();
    fprintf(
      fid_, "%d %.10e %.10e %d %.10e %.10e %.10e ",
      metadata.step,
      metadata.time_ps,
      metadata.start_time_ps,
      chunk,
      coordinate_min,
      coordinate_max,
      chunk_coords_cpu_[chunk]);
    if (geometry_mode_) {
      fprintf(fid_, "%d %d %d ", material_id, bath_mask, interface_crossing);
    } else {
      fprintf(fid_, "%d ", material_id);
    }
    fprintf(
      fid_, "%.10e %.10e %.10e %.10e %.10e %d %llu %.10e %.10e\n",
      static_cast<double>(accum.atom_count_sum) * inv_samples,
      accum.kinetic_sum * inv_samples,
      accum.potential_sum * inv_samples,
      temperature,
      accum.dof_sum * inv_samples,
      metadata.samples,
      accum.temperature_samples,
      metadata.bath_source_eV,
      metadata.bath_sink_eV);
  }
}

void ComputeChunk::reset_interface_block()
{
  const int block_size = 128;
  const int grid_size = (nchunk_ - 1) / block_size + 1;
  gpu_reset_interface_profile_block<<<grid_size, block_size>>>(
    nchunk_, gpu_interface_block_accum_.data(), gpu_interface_error_info_.data(),
    gpu_interface_error_values_.data());
  GPU_CHECK_KERNEL
  num_samples_ = 0;
}

void ComputeChunk::calculate_chunk_volumes(const Box& box)
{
  chunk_volume_cpu_.resize(nchunk_);
  double box_vol = box.get_volume();

  // Helper: actual width of bin i along dimension d
  // Last bin may be narrower than delta when L is not a multiple of delta
  auto bin_width = [&](int d, int i) -> double {
    double remainder = box_length_[d] - (nlayers_[d] - 1) * delta_[d];
    return (i < nlayers_[d] - 1) ? delta_[d] : remainder;
  };

  if (dim_ == 1) {
    double cross_area = box_vol / box_length_[0];
    for (int i = 0; i < nchunk_; i++)
      chunk_volume_cpu_[i] = cross_area * bin_width(0, i);
  } else if (dim_ == 2) {
    int third = 3 - axis_[0] - axis_[1];
    double thickness = box_vol / box.get_area(third);
    int idx = 0;
    for (int j = 0; j < nlayers_[1]; j++)
      for (int i = 0; i < nlayers_[0]; i++) {
        chunk_volume_cpu_[idx] = bin_width(0, i) * bin_width(1, j) * thickness;
        idx++;
      }
  } else {
    int idx = 0;
    for (int k = 0; k < nlayers_[2]; k++)
      for (int j = 0; j < nlayers_[1]; j++)
        for (int i = 0; i < nlayers_[0]; i++) {
          chunk_volume_cpu_[idx] = bin_width(0, i) * bin_width(1, j) * bin_width(2, k);
          idx++;
        }
  }
}

void ComputeChunk::calculate_chunk_coords(const Box& box)
{
  chunk_coords_cpu_.resize(nchunk_ * ncoord_);

  // Bin center: for normal bins it's origin + (i+0.5)*delta,
  // for the last bin it's origin + (nlayers-1)*delta + remainder/2.
  auto bin_center = [&](int d, int i) -> double {
    if (i < nlayers_[d] - 1) {
      return origin_[d] + (i + 0.5) * delta_[d];
    } else {
      double remainder = box_length_[d] - (nlayers_[d] - 1) * delta_[d];
      return origin_[d] + (nlayers_[d] - 1) * delta_[d] + remainder * 0.5;
    }
  };

  if (dim_ == 1) {
    for (int i = 0; i < nchunk_; i++)
      chunk_coords_cpu_[i] = bin_center(0, i);
  } else if (dim_ == 2) {
    int idx = 0;
    for (int j = 0; j < nlayers_[1]; j++)
      for (int i = 0; i < nlayers_[0]; i++) {
        chunk_coords_cpu_[idx * 2 + 0] = bin_center(0, i);
        chunk_coords_cpu_[idx * 2 + 1] = bin_center(1, j);
        idx++;
      }
  } else {
    int idx = 0;
    for (int k = 0; k < nlayers_[2]; k++)
      for (int j = 0; j < nlayers_[1]; j++)
        for (int i = 0; i < nlayers_[0]; i++) {
          chunk_coords_cpu_[idx * 3 + 0] = bin_center(0, i);
          chunk_coords_cpu_[idx * 3 + 1] = bin_center(1, j);
          chunk_coords_cpu_[idx * 3 + 2] = bin_center(2, k);
          idx++;
        }
  }
}
