#include "rpmd_ja.cuh"
#include "force/nep.cuh"
#include "hac.cuh"
#include "integrate/integrate.cuh"
#include "model/box.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
constexpr int threads = 128;
constexpr int warp_lanes = 32;
constexpr int warps_per_block = threads / warp_lanes;

void add_bytes(std::size_t& total, const std::size_t count, const std::size_t element_size)
{
  if (count > (std::numeric_limits<std::size_t>::max() - total) / element_size)
    PRINT_INPUT_ERROR("rpmd_ja sparse workspace byte count overflowed.");
  total += count * element_size;
}

std::size_t block_operator_required_bytes(
  const RpmdJABlockMatrix& matrix, const int dimension, const int max_vectors)
{
  if (dimension <= 0 || max_vectors <= 0 || matrix.tile_size <= 0 || matrix.tile_size > 4096)
    PRINT_INPUT_ERROR("rpmd_ja block operator has invalid dimensions.");
  const int number_of_tiles = (dimension - 1) / matrix.tile_size + 1;
  const std::size_t expected_tiles = static_cast<std::size_t>(number_of_tiles) * number_of_tiles;
  if (matrix.tiles.size() != expected_tiles)
    PRINT_INPUT_ERROR("rpmd_ja block operator does not cover its full matrix grid.");
  std::size_t bytes = 0;
  add_bytes(bytes, matrix.tiles.size(), sizeof(RpmdJABlockTileDevice));
  std::size_t factor_bytes = 0;
  std::size_t right_bytes = 0;
  std::size_t projection_bytes = 0;
  for (const auto& tile : matrix.tiles) {
    if (tile.rank < 0 || tile.rank > std::min(tile.rows, tile.columns))
      PRINT_INPUT_ERROR("rpmd_ja block operator has an invalid tile rank.");
    add_bytes(factor_bytes, tile.left.size(), sizeof(double));
    add_bytes(right_bytes, tile.right.size(), sizeof(double));
    if (tile.rank > 0)
      add_bytes(projection_bytes, static_cast<std::size_t>(max_vectors) * tile.rank, sizeof(double));
  }
  add_bytes(bytes, factor_bytes, 1);
  add_bytes(bytes, std::max<std::size_t>(sizeof(double), right_bytes), 1);
  add_bytes(bytes, std::max<std::size_t>(sizeof(double), projection_bytes), 1);
  return bytes;
}

void validate_matrix(const RpmdJASparseMatrix& matrix, const int dimension, const char* name)
{
  if (matrix.row_offsets.size() != static_cast<std::size_t>(dimension) + 1 ||
      matrix.row_offsets.empty() || matrix.row_offsets.front() != 0 ||
      matrix.row_offsets.back() != matrix.columns.size() ||
      matrix.columns.size() != matrix.values.size()) {
    PRINT_INPUT_ERROR(name);
  }
  for (int row = 0; row < dimension; ++row) {
    if (matrix.row_offsets[row] > matrix.row_offsets[row + 1]) PRINT_INPUT_ERROR(name);
  }
  for (std::size_t i = 0; i < matrix.columns.size(); ++i) {
    if (matrix.columns[i] < 0 || matrix.columns[i] >= dimension || !std::isfinite(matrix.values[i]))
      PRINT_INPUT_ERROR(name);
  }
}

static __global__ void make_wrapped_position(
  const int N, const Box box, const double* continuous, double* wrapped, int* error)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= N) return;
  const double x = continuous[atom];
  const double y = continuous[atom + N];
  const double z = continuous[atom + 2 * N];
  if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
    atomicOr(error, RPMD_JA_ERROR_BRANCH);
    wrapped[atom] = wrapped[atom + N] = wrapped[atom + 2 * N] = 0.0;
    return;
  }
  double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  if (box.pbc_x) sx -= floor(sx);
  if (box.pbc_y) sy -= floor(sy);
  if (box.pbc_z) sz -= floor(sz);
  wrapped[atom] = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
  wrapped[atom + N] = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
  wrapped[atom + 2 * N] = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
}

static __global__ void build_weighted_inputs(
  const int N,
  const double* continuous,
  const double* reference,
  const double* velocity,
  const double* sqrt_mass,
  double* weighted_yu,
  int* error)
{
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int D = 3 * N;
  if (d >= D) return;
  const int atom = d % N;
  const double y = sqrt_mass[atom] * (continuous[d] - reference[d]);
  const double u = sqrt_mass[atom] * velocity[d];
  if (!isfinite(y) || !isfinite(u)) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  weighted_yu[d] = isfinite(y) ? y : 0.0;
  weighted_yu[D + d] = isfinite(u) ? u : 0.0;
}

static __global__ void reduce_translation(
  const int N,
  const int D,
  const int number_of_vectors,
  const double inverse_mass_sum,
  const double* sqrt_mass,
  const double* vectors,
  double* coefficients)
{
  const int vector = blockIdx.x / 3;
  const int axis = blockIdx.x % 3;
  const int tid = threadIdx.x;
  __shared__ double sums[threads];
  double sum = 0.0;
  if (vector < number_of_vectors) {
    for (int atom = tid; atom < N; atom += blockDim.x)
      sum += vectors[static_cast<std::size_t>(vector) * D + axis * N + atom] * sqrt_mass[atom];
  }
  sums[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  if (tid == 0 && vector < number_of_vectors)
    coefficients[3 * vector + axis] = sums[0] * inverse_mass_sum;
}

static __global__ void remove_translation(
  const int N,
  const int D,
  const int number_of_vectors,
  const double* sqrt_mass,
  const double* coefficients,
  double* vectors)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  const int vector = blockIdx.y;
  const int axis = blockIdx.z;
  if (atom < N && vector < number_of_vectors) {
    const std::size_t index = static_cast<std::size_t>(vector) * D + axis * N + atom;
    vectors[index] -= coefficients[3 * vector + axis] * sqrt_mass[atom];
  }
}

void project_vectors(
  const int N,
  const int D,
  const int number_of_vectors,
  const double inverse_mass_sum,
  const GPU_Vector<double>& sqrt_mass,
  GPU_Vector<double>& coefficients,
  double* vectors)
{
  reduce_translation<<<dim3(3 * number_of_vectors), threads>>>(
    N, D, number_of_vectors, inverse_mass_sum, sqrt_mass.data(), vectors, coefficients.data());
  GPU_CHECK_KERNEL
  remove_translation<<<dim3((N - 1) / threads + 1, number_of_vectors, 3), threads>>>(
    N, D, number_of_vectors, sqrt_mass.data(), coefficients.data(), vectors);
  GPU_CHECK_KERNEL
}

static __global__ void sparse_apply(
  const int D,
  const int number_of_vectors,
  const std::uint64_t* row_offsets,
  const int* columns,
  const double* values,
  const double* input,
  double* output,
  int* error)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int vector = blockIdx.y;
  if (row >= D || vector >= number_of_vectors) return;
  double sum = 0.0;
  for (std::uint64_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k)
    sum += values[k] * input[static_cast<std::size_t>(vector) * D + columns[k]];
  if (!isfinite(sum)) {
    atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
    sum = 0.0;
  }
  output[static_cast<std::size_t>(vector) * D + row] = sum;
}

static __global__ void pack_seeds(const int D, const double* dydu, const double* weighted_yu, double* seeds)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < D) {
    seeds[i] = dydu[i];
    seeds[D + i] = dydu[D + i];
    seeds[2 * D + i] = weighted_yu[D + i];
  }
}

static __global__ void chebyshev_step(
  const int D,
  const double lambda_scale,
  const int first_step,
  const double* previous,
  const double* current,
  const double* d_current,
  double* next,
  int* error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * D) return;
  double value = 0.0;
  if (first_step) {
    value = 2.0 * d_current[i] / lambda_scale - current[i];
  } else {
    value = 4.0 * d_current[i] / lambda_scale - 2.0 * current[i] - previous[i];
  }
  if (!isfinite(value)) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  next[i] = isfinite(value) ? value : 0.0;
}

static __global__ void accumulate_modal_vectors(
  const int D,
  const int p_rank,
  const int q_rank,
  const int degree,
  const double* p_vectors,
  const double* q_vectors,
  const double* chebyshev,
  double* p_accumulators,
  double* q_accumulators,
  int* error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int rank = blockIdx.y;
  if (i >= D) return;
  if (rank < p_rank) {
    const double coefficient = p_vectors[degree * p_rank + rank];
    p_accumulators[static_cast<std::size_t>(rank) * D + i] += coefficient * chebyshev[i];
    p_accumulators[static_cast<std::size_t>(p_rank + rank) * D + i] += coefficient * chebyshev[D + i];
    if (!isfinite(p_accumulators[static_cast<std::size_t>(rank) * D + i]) ||
        !isfinite(p_accumulators[static_cast<std::size_t>(p_rank + rank) * D + i])) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  }
  if (rank < q_rank) {
    const double coefficient = q_vectors[degree * q_rank + rank];
    q_accumulators[static_cast<std::size_t>(rank) * D + i] += coefficient * chebyshev[i];
    q_accumulators[static_cast<std::size_t>(q_rank + rank) * D + i] += coefficient * chebyshev[2 * D + i];
    if (!isfinite(q_accumulators[static_cast<std::size_t>(rank) * D + i]) ||
        !isfinite(q_accumulators[static_cast<std::size_t>(q_rank + rank) * D + i])) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  }
}

static __global__ void sparse_apply_b_batch(
  const int D,
  const int p_rank,
  const int q_rank,
  const std::uint64_t* row_offsets,
  const int* columns,
  const double* values,
  const double* p_left,
  const double* q_right,
  double* output,
  int* error)
{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int vector = blockIdx.y;
  const int total = p_rank + q_rank;
  if (row >= D || vector >= total) return;
  const double* input = vector < p_rank
    ? p_left + static_cast<std::size_t>(vector) * D
    : q_right + static_cast<std::size_t>(vector - p_rank) * D;
  double sum = 0.0;
  for (std::uint64_t k = row_offsets[row]; k < row_offsets[row + 1]; ++k)
    sum += values[k] * input[columns[k]];
  if (!isfinite(sum)) {
    atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
    sum = 0.0;
  }
  output[static_cast<std::size_t>(vector) * D + row] = sum;
}

__device__ double warp_sum_double(double value)
{
  for (int offset = warp_lanes / 2; offset > 0; offset >>= 1) {
#ifdef USE_HIP
    value += __shfl_down(value, offset, warp_lanes);
#else
    value += __shfl_down_sync(0xffffffff, value, offset, warp_lanes);
#endif
  }
  return value;
}

static __global__ void block_apply_right(
  const int D,
  const int tile_size,
  const int number_of_tiles,
  const int number_of_tile_columns,
  const RpmdJABlockTileDevice* tiles,
  const double* right,
  const double* input,
  double* projection,
  int* error)
{
  const int tile = blockIdx.x * warps_per_block + threadIdx.x / warp_lanes;
  const int lane = threadIdx.x % warp_lanes;
  const int vector = blockIdx.y;
  if (tile >= number_of_tiles) return;
  const RpmdJABlockTileDevice descriptor = tiles[tile];
  if (descriptor.rank == 0) return;
  const int column_tile = tile % number_of_tile_columns;
  const int column_start = column_tile * tile_size;
  for (int rank = 0; rank < descriptor.rank; ++rank) {
    double sum = 0.0;
    for (int column = lane; column < descriptor.columns; column += warp_lanes)
      sum += right[descriptor.right_offset + static_cast<std::size_t>(rank) * descriptor.columns + column] *
        input[static_cast<std::size_t>(vector) * D + column_start + column];
    sum = warp_sum_double(sum);
    if (lane == 0) {
      if (!isfinite(sum)) {
        atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
        sum = 0.0;
      }
      projection[descriptor.projection_offset + static_cast<std::size_t>(vector) * descriptor.rank + rank] = sum;
    }
  }
}

static __global__ void block_apply_rows(
  const int D,
  const int tile_size,
  const int number_of_tiles,
  const int number_of_tile_columns,
  const int number_of_vectors,
  const RpmdJABlockTileDevice* tiles,
  const double* left,
  const double* input,
  const double* projection,
  double* output,
  int* error)
{
  const int row = blockIdx.x * warps_per_block + threadIdx.x / warp_lanes;
  const int lane = threadIdx.x % warp_lanes;
  const int vector = blockIdx.y;
  if (row >= D || vector >= number_of_vectors) return;
  const int row_tile = row / tile_size;
  const int local_row = row % tile_size;
  double sum = 0.0;
  for (int column_tile = 0; column_tile < number_of_tile_columns; ++column_tile) {
    const int tile = row_tile * number_of_tile_columns + column_tile;
    if (tile >= number_of_tiles) continue;
    const RpmdJABlockTileDevice descriptor = tiles[tile];
    if (local_row >= descriptor.rows) continue;
    if (descriptor.rank == 0) {
      const int column_start = column_tile * tile_size;
      for (int column = lane; column < descriptor.columns; column += warp_lanes)
        sum += left[descriptor.left_offset + static_cast<std::size_t>(local_row) * descriptor.columns + column] *
          input[static_cast<std::size_t>(vector) * D + column_start + column];
    } else {
      for (int rank = lane; rank < descriptor.rank; rank += warp_lanes)
        sum += left[descriptor.left_offset + static_cast<std::size_t>(local_row) * descriptor.rank + rank] *
          projection[descriptor.projection_offset + static_cast<std::size_t>(vector) * descriptor.rank + rank];
    }
  }
  sum = warp_sum_double(sum);
  if (lane == 0) {
    if (!isfinite(sum)) {
      atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
      sum = 0.0;
    }
    output[static_cast<std::size_t>(vector) * D + row] = sum;
  }
}

static __global__ void reduce_modal_current(
  const int D,
  const int p_rank,
  const double tau,
  const double* p_values,
  const double* p_accumulators,
  const double* b_results,
  double* result,
  int* error)
{
  const int alpha = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ double sums[threads];
  double sum = 0.0;
  const std::size_t total = static_cast<std::size_t>(p_rank) * D;
  for (std::size_t index = tid; index < total; index += blockDim.x) {
    const int rank = static_cast<int>(index / D);
    const int i = static_cast<int>(index - static_cast<std::size_t>(rank) * D);
    sum += p_values[rank] * b_results[index] *
      p_accumulators[static_cast<std::size_t>(p_rank + rank) * D + i];
  }
  sums[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  if (tid == 0) {
    const double tau2 = tau * tau;
    const double value = tau2 * tau2 * sums[0];
    if (!isfinite(value)) {
      atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
      result[alpha] = 0.0;
    } else {
      result[alpha] = value;
    }
  }
}

static __global__ void add_modal_q_term(
  const int D,
  const int p_rank,
  const int q_rank,
  const double tau,
  const double* q_values,
  const double* q_accumulators,
  const double* b_results,
  double* result,
  int* error)
{
  const int alpha = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ double sums[threads];
  double sum = 0.0;
  const std::size_t total = static_cast<std::size_t>(q_rank) * D;
  for (std::size_t index = tid; index < total; index += blockDim.x) {
    const int rank = static_cast<int>(index / D);
    const int i = static_cast<int>(index - static_cast<std::size_t>(rank) * D);
    sum += q_values[rank] * q_accumulators[static_cast<std::size_t>(rank) * D + i] *
      b_results[static_cast<std::size_t>(p_rank + rank) * D + i];
  }
  sums[tid] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) sums[tid] += sums[tid + offset];
    __syncthreads();
  }
  if (tid == 0) {
    const double value = result[alpha] + tau * tau * sums[0];
    if (!isfinite(value)) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
    result[alpha] = isfinite(value) ? value : 0.0;
  }
}
} // namespace

void RpmdJABlockOperator::initialize(
  const RpmdJABlockMatrix& matrix, const int dimension, const int max_vectors)
{
  if (dimension <= 0 || max_vectors <= 0 || matrix.tile_size <= 0 || matrix.tile_size > 4096) {
    PRINT_INPUT_ERROR("rpmd_ja block operator has invalid dimensions.");
  }
  dimension_ = dimension;
  tile_size_ = matrix.tile_size;
  max_vectors_ = max_vectors;
  has_low_rank_ = false;
  number_of_tiles_ = (dimension_ - 1) / tile_size_ + 1;
  const std::size_t grid_size = static_cast<std::size_t>(number_of_tiles_) * number_of_tiles_;
  if (matrix.tiles.size() != grid_size || grid_size > std::numeric_limits<int>::max())
    PRINT_INPUT_ERROR("rpmd_ja block operator does not cover its full matrix grid.");
  block_count_ = static_cast<int>(grid_size);

  std::vector<RpmdJABlockTileDevice> descriptors(grid_size);
  std::vector<unsigned char> seen(grid_size, 0);
  std::size_t left_size = 0;
  std::size_t right_size = 0;
  std::size_t projection_size = 0;
  for (const auto& tile : matrix.tiles) {
    if (tile.row < 0 || tile.column < 0 || tile.row % tile_size_ != 0 ||
        tile.column % tile_size_ != 0 || tile.row >= dimension_ || tile.column >= dimension_) {
      PRINT_INPUT_ERROR("rpmd_ja block operator has a tile outside its aligned grid.");
    }
    const int row_tile = tile.row / tile_size_;
    const int column_tile = tile.column / tile_size_;
    const std::size_t index = static_cast<std::size_t>(row_tile) * number_of_tiles_ + column_tile;
    const int rows = std::min(tile_size_, dimension_ - tile.row);
    const int columns = std::min(tile_size_, dimension_ - tile.column);
    if (seen[index] || tile.rows != rows || tile.columns != columns || tile.rank < 0 ||
        tile.rank > std::min(rows, columns)) {
      PRINT_INPUT_ERROR("rpmd_ja block operator has duplicate, malformed, or oversized tiles.");
    }
    const std::size_t left_count = tile.rank == 0
      ? static_cast<std::size_t>(rows) * columns
      : static_cast<std::size_t>(rows) * tile.rank;
    const std::size_t right_count = tile.rank == 0
      ? 0
      : static_cast<std::size_t>(tile.rank) * columns;
    if (tile.left.size() != left_count || tile.right.size() != right_count ||
        !std::all_of(tile.left.begin(), tile.left.end(), [](double value) { return std::isfinite(value); }) ||
        !std::all_of(tile.right.begin(), tile.right.end(), [](double value) { return std::isfinite(value); })) {
      PRINT_INPUT_ERROR("rpmd_ja block operator has malformed or non-finite tile factors.");
    }
    auto& descriptor = descriptors[index];
    descriptor.rows = rows;
    descriptor.columns = columns;
    descriptor.rank = tile.rank;
    has_low_rank_ = has_low_rank_ || tile.rank > 0;
    descriptor.left_offset = left_size;
    descriptor.right_offset = right_size;
    descriptor.projection_offset = projection_size;
    if (left_count > std::numeric_limits<std::size_t>::max() - left_size ||
        right_count > std::numeric_limits<std::size_t>::max() - right_size)
      PRINT_INPUT_ERROR("rpmd_ja block operator factor storage size overflowed.");
    left_size += left_count;
    right_size += right_count;
    const std::size_t tile_projection_size = static_cast<std::size_t>(max_vectors_) * tile.rank;
    if (tile_projection_size > std::numeric_limits<std::size_t>::max() - projection_size)
      PRINT_INPUT_ERROR("rpmd_ja block operator projection storage size overflowed.");
    projection_size += tile_projection_size;
    seen[index] = 1;
  }
  if (std::any_of(seen.begin(), seen.end(), [](unsigned char present) { return present == 0; }))
    PRINT_INPUT_ERROR("rpmd_ja block operator omits one or more matrix tiles.");

  allocated_bytes_ = 0;
  add_bytes(allocated_bytes_, descriptors.size(), sizeof(RpmdJABlockTileDevice));
  add_bytes(allocated_bytes_, left_size, sizeof(double));
  add_bytes(allocated_bytes_, std::max<std::size_t>(1, right_size), sizeof(double));
  add_bytes(allocated_bytes_, std::max<std::size_t>(1, projection_size), sizeof(double));
  if (allocated_bytes_ != block_operator_required_bytes(matrix, dimension_, max_vectors_))
    PRINT_INPUT_ERROR("rpmd_ja block operator storage preflight mismatch.");
  std::size_t free_bytes = 0, total_bytes = 0;
#ifdef USE_HIP
  CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
#else
  CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
#endif
  if (allocated_bytes_ > free_bytes)
    PRINT_INPUT_ERROR("rpmd_ja block operator does not fit available GPU memory.");
  tiles_.resize(descriptors.size());
  tiles_.copy_from_host(descriptors.data());
  left_.resize(left_size);
  {
    right_.resize(std::max<std::size_t>(1, right_size));
  }
  projection_.resize(std::max<std::size_t>(1, projection_size));
  for (const auto& tile : matrix.tiles) {
    const int row_tile = tile.row / tile_size_;
    const int column_tile = tile.column / tile_size_;
    const std::size_t index = static_cast<std::size_t>(row_tile) * number_of_tiles_ + column_tile;
    const auto& descriptor = descriptors[index];
    if (!tile.left.empty())
      CHECK(gpuMemcpy(left_.data() + descriptor.left_offset, tile.left.data(),
        tile.left.size() * sizeof(double), gpuMemcpyHostToDevice));
    if (!tile.right.empty())
      CHECK(gpuMemcpy(right_.data() + descriptor.right_offset, tile.right.data(),
        tile.right.size() * sizeof(double), gpuMemcpyHostToDevice));
  }
}

void RpmdJABlockOperator::apply(
  const double* input, const int number_of_vectors, double* output, int* device_error)
{
  if (dimension_ <= 0 || input == nullptr || output == nullptr || device_error == nullptr ||
      number_of_vectors <= 0 || number_of_vectors > max_vectors_) {
    PRINT_INPUT_ERROR("rpmd_ja block matvec received invalid buffers or vector count.");
  }
  if (has_low_rank_) {
    const unsigned int grid_x = static_cast<unsigned int>((block_count_ - 1) / warps_per_block + 1);
    block_apply_right<<<dim3(grid_x, number_of_vectors), threads>>>(
      dimension_, tile_size_, block_count_, number_of_tiles_,
      tiles_.data(), right_.data(), input,
      projection_.data(), device_error);
    GPU_CHECK_KERNEL
  }
  block_apply_rows<<<dim3((dimension_ - 1) / warps_per_block + 1, number_of_vectors), threads>>>(
    dimension_, tile_size_, block_count_, number_of_tiles_, number_of_vectors,
    tiles_.data(), left_.data(), input, projection_.data(), output, device_error);
  GPU_CHECK_KERNEL
}

void RpmdJASparseWorkspace::initialize(
  const RpmdJAReference& reference,
  const std::vector<double>& masses)
{
  number_of_atoms_ = reference.number_of_atoms;
  degree_ = reference.kernel_degree;
  p_rank_ = reference.p_rank;
  q_rank_ = reference.q_rank;
  backend_ = reference.backend;
  if (!reference.stability_checked || number_of_atoms_ <= 0 ||
      number_of_atoms_ > (std::numeric_limits<int>::max() - (threads - 1)) / 9 ||
      degree_ < 0 || degree_ > 512 ||
      p_rank_ < 0 || q_rank_ < 0 || p_rank_ > degree_ + 1 || q_rank_ > degree_ + 1 ||
      2 * std::max(p_rank_, q_rank_) > 65535 ||
      p_rank_ + q_rank_ > 65535 ||
      masses.size() != static_cast<std::size_t>(number_of_atoms_)) {
    PRINT_INPUT_ERROR("rpmd_ja sparse reference is unstable or has invalid dimensions.");
  }
  dimension_ = 3 * number_of_atoms_;
  if (backend_ == 1) {
    validate_matrix(reference.dynamical, dimension_, "rpmd_ja sparse D matrix is malformed.");
    for (int alpha = 0; alpha < 3; ++alpha)
      validate_matrix(reference.site_transpose[alpha], dimension_, "rpmd_ja sparse B matrix is malformed.");
  } else if (backend_ == 2) {
    // The fixed-reference cache validates and uploads these source tiles.
  } else {
    PRINT_INPUT_ERROR("rpmd_ja sparse workspace received an unsupported backend.");
  }
  if (reference.p_values.size() != static_cast<std::size_t>(p_rank_) ||
      reference.q_values.size() != static_cast<std::size_t>(q_rank_) ||
      reference.p_vectors.size() != static_cast<std::size_t>(degree_ + 1) * p_rank_ ||
      reference.q_vectors.size() != static_cast<std::size_t>(degree_ + 1) * q_rank_ ||
      !std::isfinite(reference.spectral_bound) || !(reference.spectral_bound > 0.0) ||
      !std::isfinite(reference.kernel_u) || !(reference.kernel_u > 0.0) ||
      !std::isfinite(reference.kernel_error[0]) || reference.kernel_error[0] < 0.0 ||
      !std::isfinite(reference.kernel_error[1]) || reference.kernel_error[1] < 0.0 ||
      !std::isfinite(reference.kernel_s2[0]) || reference.kernel_s2[0] < 0.0 ||
      !std::isfinite(reference.kernel_s2[1]) || reference.kernel_s2[1] < 0.0 ||
      !std::all_of(reference.p_values.begin(), reference.p_values.end(), [](double x) { return std::isfinite(x); }) ||
      !std::all_of(reference.q_values.begin(), reference.q_values.end(), [](double x) { return std::isfinite(x); }) ||
      !std::all_of(reference.p_vectors.begin(), reference.p_vectors.end(), [](double x) { return std::isfinite(x); }) ||
      !std::all_of(reference.q_vectors.begin(), reference.q_vectors.end(), [](double x) { return std::isfinite(x); })) {
    PRINT_INPUT_ERROR("rpmd_ja sparse kernel table has invalid dimensions or non-finite values.");
  }
  tau_ = HBAR / (K_B * reference.temperature);
  lambda_scale_ = reference.kernel_u * reference.kernel_u / (tau_ * tau_);
  if (!std::isfinite(tau_) || !(tau_ > 0.0) || !std::isfinite(lambda_scale_) || !(lambda_scale_ > 0.0))
    PRINT_INPUT_ERROR("rpmd_ja sparse kernel scaling is non-finite.");
  if (reference.spectral_bound > lambda_scale_ * (1.0 + 1.0e-10))
    PRINT_INPUT_ERROR("rpmd_ja sparse kernel interval does not cover the reference spectral bound.");

  std::vector<double> sqrt_mass(number_of_atoms_);
  double mass_sum = 0.0;
  for (int i = 0; i < number_of_atoms_; ++i) {
    if (!std::isfinite(masses[i]) || !(masses[i] > 0.0))
      PRINT_INPUT_ERROR("rpmd_ja sparse reference has invalid masses.");
    sqrt_mass[i] = std::sqrt(masses[i]);
    mass_sum += masses[i];
  }
  if (!std::isfinite(mass_sum) || !(mass_sum > 0.0))
    PRINT_INPUT_ERROR("rpmd_ja sparse reference has an invalid total mass.");
  inverse_mass_sum_ = 1.0 / mass_sum;
  if (backend_ == 2) {
    qnep_cached_.initialize(reference, masses);
    allocated_bytes_ = qnep_cached_.allocated_bytes();
    return;
  }

  const int maximum_projected_vectors = std::max({3, 2 * p_rank_, 2 * q_rank_});
  allocated_bytes_ = 0;
  auto add_matrix_bytes = [&](const RpmdJASparseMatrix& matrix) {
    add_bytes(allocated_bytes_, matrix.row_offsets.size(), sizeof(std::uint64_t));
    add_bytes(allocated_bytes_, matrix.columns.size(), sizeof(int));
    add_bytes(allocated_bytes_, matrix.values.size(), sizeof(double));
  };
  add_matrix_bytes(reference.dynamical);
  for (int alpha = 0; alpha < 3; ++alpha) add_matrix_bytes(reference.site_transpose[alpha]);
  add_bytes(allocated_bytes_, sqrt_mass.size(), sizeof(double));
  add_bytes(allocated_bytes_, reference.p_values.size() + reference.q_values.size(), sizeof(double));
  add_bytes(allocated_bytes_, reference.p_vectors.size() + reference.q_vectors.size(), sizeof(double));
  add_bytes(allocated_bytes_, static_cast<std::size_t>(16) * dimension_, sizeof(double));
  add_bytes(allocated_bytes_, static_cast<std::size_t>(3) * maximum_projected_vectors, sizeof(double));
  add_bytes(allocated_bytes_, static_cast<std::size_t>(2) * (p_rank_ + q_rank_) * dimension_, sizeof(double));
  add_bytes(allocated_bytes_, static_cast<std::size_t>(p_rank_ + q_rank_) * dimension_, sizeof(double));
  std::size_t free_bytes = 0, total_bytes = 0;
#ifdef USE_HIP
  CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
#else
  CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
#endif
  if (allocated_bytes_ > free_bytes) {
    PRINT_INPUT_ERROR("rpmd_ja sparse workspace does not fit available GPU memory; dense fallback is disabled.");
  }

  if (backend_ == 1) {
    dynamical_rows_.resize(reference.dynamical.row_offsets.size());
    dynamical_rows_.copy_from_host(reference.dynamical.row_offsets.data());
    if (!reference.dynamical.columns.empty()) {
      dynamical_columns_.resize(reference.dynamical.columns.size());
      dynamical_columns_.copy_from_host(reference.dynamical.columns.data());
      dynamical_values_.resize(reference.dynamical.values.size());
      dynamical_values_.copy_from_host(reference.dynamical.values.data());
    }
    for (int alpha = 0; alpha < 3; ++alpha) {
      site_rows_[alpha].resize(reference.site_transpose[alpha].row_offsets.size());
      site_rows_[alpha].copy_from_host(reference.site_transpose[alpha].row_offsets.data());
      if (!reference.site_transpose[alpha].columns.empty()) {
        site_columns_[alpha].resize(reference.site_transpose[alpha].columns.size());
        site_columns_[alpha].copy_from_host(reference.site_transpose[alpha].columns.data());
        site_values_[alpha].resize(reference.site_transpose[alpha].values.size());
        site_values_[alpha].copy_from_host(reference.site_transpose[alpha].values.data());
      }
    }
  }
  sqrt_mass_.resize(sqrt_mass.size());
  sqrt_mass_.copy_from_host(sqrt_mass.data());
  if (!reference.p_values.empty()) {
    p_values_.resize(reference.p_values.size());
    p_values_.copy_from_host(reference.p_values.data());
  }
  if (!reference.q_values.empty()) {
    q_values_.resize(reference.q_values.size());
    q_values_.copy_from_host(reference.q_values.data());
  }
  if (!reference.p_vectors.empty()) {
    p_vectors_.resize(reference.p_vectors.size());
    p_vectors_.copy_from_host(reference.p_vectors.data());
  }
  if (!reference.q_vectors.empty()) {
    q_vectors_.resize(reference.q_vectors.size());
    q_vectors_.copy_from_host(reference.q_vectors.data());
  }
  weighted_yu_.resize(static_cast<std::size_t>(2) * dimension_);
  dydu_.resize(static_cast<std::size_t>(2) * dimension_);
  seed_.resize(static_cast<std::size_t>(3) * dimension_);
  cheb_a_.resize(static_cast<std::size_t>(3) * dimension_);
  cheb_b_.resize(static_cast<std::size_t>(3) * dimension_);
  d_action_.resize(static_cast<std::size_t>(3) * dimension_);
  projection_coefficients_.resize(static_cast<std::size_t>(3) * maximum_projected_vectors);
  if (p_rank_ > 0) p_accumulators_.resize(static_cast<std::size_t>(2) * p_rank_ * dimension_, 0.0);
  if (q_rank_ > 0) q_accumulators_.resize(static_cast<std::size_t>(2) * q_rank_ * dimension_, 0.0);
  if (p_rank_ + q_rank_ > 0) b_results_.resize(static_cast<std::size_t>(p_rank_ + q_rank_) * dimension_);
}

void rpmd_ja_wrap_positions(
  const int number_of_atoms,
  const GPU_Vector<double>& continuous,
  GPU_Vector<double>& wrapped,
  const Box& box,
  int* device_error)
{
  const std::size_t expected_size = static_cast<std::size_t>(number_of_atoms) * 3;
  if (number_of_atoms <= 0 || continuous.size() != expected_size || wrapped.size() != expected_size || device_error == nullptr)
    PRINT_INPUT_ERROR("rpmd_ja private centroid wrapping received an invalid buffer shape.");
  make_wrapped_position<<<(number_of_atoms - 1) / threads + 1, threads>>>(
    number_of_atoms, box, continuous.data(), wrapped.data(), device_error);
  GPU_CHECK_KERNEL
}

void RpmdJASparseWorkspace::compute_correction(
  const GPU_Vector<double>& continuous,
  const GPU_Vector<double>& reference_positions,
  const GPU_Vector<double>& velocity,
  double* device_result,
  int* device_error)
{
  if (backend_ == 2) {
    qnep_cached_.compute_correction(
      continuous, reference_positions, velocity, device_result, device_error);
    return;
  }
  const int D = dimension_;
  const int blocks = (D - 1) / threads + 1;
  build_weighted_inputs<<<blocks, threads>>>(
    number_of_atoms_, continuous.data(), reference_positions.data(), velocity.data(),
    sqrt_mass_.data(), weighted_yu_.data(), device_error);
  GPU_CHECK_KERNEL
  project_vectors(number_of_atoms_, D, 2, inverse_mass_sum_, sqrt_mass_, projection_coefficients_, weighted_yu_.data());
  sparse_apply<<<dim3(blocks, 2), threads>>>(
    D, 2, dynamical_rows_.data(), dynamical_columns_.data(), dynamical_values_.data(),
    weighted_yu_.data(), dydu_.data(), device_error);
  GPU_CHECK_KERNEL
  project_vectors(number_of_atoms_, D, 2, inverse_mass_sum_, sqrt_mass_, projection_coefficients_, dydu_.data());
  pack_seeds<<<blocks, threads>>>(D, dydu_.data(), weighted_yu_.data(), seed_.data());
  GPU_CHECK_KERNEL
  if (p_rank_ > 0) p_accumulators_.fill(0.0);
  if (q_rank_ > 0) q_accumulators_.fill(0.0);

  double* previous = nullptr;
  double* current = seed_.data();
  double* next = cheb_a_.data();
  for (int degree = 0; degree <= degree_; ++degree) {
    if (p_rank_ + q_rank_ > 0) {
      accumulate_modal_vectors<<<dim3(blocks, std::max(p_rank_, q_rank_)), threads>>>(
        D, p_rank_, q_rank_, degree, p_vectors_.data(), q_vectors_.data(), current,
        p_accumulators_.data(), q_accumulators_.data(), device_error);
      GPU_CHECK_KERNEL
    }
    if (degree == degree_) break;
    project_vectors(number_of_atoms_, D, 3, inverse_mass_sum_, sqrt_mass_, projection_coefficients_, current);
    sparse_apply<<<dim3(blocks, 3), threads>>>(
      D, 3, dynamical_rows_.data(), dynamical_columns_.data(), dynamical_values_.data(), current,
      d_action_.data(), device_error);
    GPU_CHECK_KERNEL
    project_vectors(number_of_atoms_, D, 3, inverse_mass_sum_, sqrt_mass_, projection_coefficients_, d_action_.data());
    chebyshev_step<<<blocks * 3, threads>>>(
      D, lambda_scale_, degree == 0 ? 1 : 0, previous, current, d_action_.data(), next, device_error);
    GPU_CHECK_KERNEL
    double* old_previous = previous;
    previous = current;
    current = next;
    next = old_previous == nullptr ? cheb_b_.data() : old_previous;
  }

  if (p_rank_ > 0) project_vectors(
    number_of_atoms_, D, 2 * p_rank_, inverse_mass_sum_, sqrt_mass_,
    projection_coefficients_, p_accumulators_.data());
  if (q_rank_ > 0) project_vectors(
    number_of_atoms_, D, 2 * q_rank_, inverse_mass_sum_, sqrt_mass_,
    projection_coefficients_, q_accumulators_.data());

  for (int alpha = 0; alpha < 3; ++alpha) {
    if (p_rank_ + q_rank_ == 0) {
      CHECK(gpuMemset(device_result + alpha, 0, sizeof(double)));
      continue;
    }
    sparse_apply_b_batch<<<dim3(blocks, p_rank_ + q_rank_), threads>>>(
      D, p_rank_, q_rank_, site_rows_[alpha].data(), site_columns_[alpha].data(), site_values_[alpha].data(),
      p_accumulators_.data(), q_rank_ ? q_accumulators_.data() + static_cast<std::size_t>(q_rank_) * D : nullptr,
      b_results_.data(), device_error);
    GPU_CHECK_KERNEL
    reduce_modal_current<<<1, threads>>>(
      D, p_rank_, tau_, p_values_.data(), p_accumulators_.data(), b_results_.data(),
      device_result + alpha, device_error);
    GPU_CHECK_KERNEL
    add_modal_q_term<<<1, threads>>>(
      D, p_rank_, q_rank_, tau_, q_values_.data(), q_accumulators_.data(), b_results_.data(),
      device_result + alpha, device_error);
    GPU_CHECK_KERNEL
  }
}
