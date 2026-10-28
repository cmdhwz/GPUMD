#include "rpmd_ja_qnep_cached.cuh"
#include "rpmd_ja.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef USE_HIP
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

namespace
{
constexpr int threads = 256;
constexpr int tile_size = 128;

void cuda_check(const cudaError_t status, const char* operation)
{
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("rpmd_ja qNEP cache CUDA failure at ") + operation + ": " + cudaGetErrorString(status));
}

void blas_check(const cublasStatus_t status, const char* operation)
{
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("rpmd_ja qNEP cache cuBLAS failure at ") + operation);
}

void solver_check(const cusolverStatus_t status, const char* operation)
{
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string("rpmd_ja qNEP cache cuSOLVER failure at ") + operation);
}

void launch_check(const char* operation)
{
  cuda_check(cudaGetLastError(), operation);
  cuda_check(cudaDeviceSynchronize(), operation);
}

void add_count(std::size_t& total, const std::size_t count, const std::size_t element_size)
{
  if (count > (std::numeric_limits<std::size_t>::max() - total) / element_size)
    throw std::runtime_error("rpmd_ja qNEP cache memory estimate overflowed");
  total += count * element_size;
}

void add_bytes(std::size_t& total, const std::size_t bytes)
{
  add_count(total, bytes, 1);
}

void require_gpu_bytes(const std::size_t bytes, const char* stage)
{
  std::size_t free_bytes = 0, total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
  if (bytes > free_bytes) {
    std::fprintf(stderr,
      "rpmd_ja qNEP cache %s requires %llu bytes but only %llu bytes are free; no dense fallback is available.\n",
      stage, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(free_bytes));
    std::exit(1);
  }
}

std::size_t validate_tiles(const RpmdJABlockMatrix& matrix, const int D)
{
  if (matrix.tile_size != tile_size)
    throw std::runtime_error("rpmd_ja qNEP cache requires 128-by-128 source tiles");
  const int tile_count = (D - 1) / tile_size + 1;
  const std::size_t expected = static_cast<std::size_t>(tile_count) * tile_count;
  if (matrix.tiles.size() != expected)
    throw std::runtime_error("rpmd_ja qNEP cache source tiles do not cover the full matrix");
  std::vector<unsigned char> seen(expected, 0);
  for (const auto& tile : matrix.tiles) {
    if (tile.row < 0 || tile.column < 0 || tile.row % tile_size || tile.column % tile_size ||
        tile.row >= D || tile.column >= D)
      throw std::runtime_error("rpmd_ja qNEP cache source tile has an invalid origin");
    const int rows = std::min(tile_size, D - tile.row);
    const int columns = std::min(tile_size, D - tile.column);
    const int rank_limit = std::min(rows, columns);
    const std::size_t index = static_cast<std::size_t>(tile.row / tile_size) * tile_count + tile.column / tile_size;
    const std::size_t left_count = tile.rank == 0
      ? static_cast<std::size_t>(rows) * columns
      : static_cast<std::size_t>(rows) * tile.rank;
    const std::size_t right_count = tile.rank == 0
      ? 0
      : static_cast<std::size_t>(tile.rank) * columns;
    if (seen[index] || tile.rows != rows || tile.columns != columns || tile.rank < 0 ||
        tile.rank > rank_limit || tile.left.size() != left_count || tile.right.size() != right_count ||
        !std::all_of(tile.left.begin(), tile.left.end(), [](double x) { return std::isfinite(x); }) ||
        !std::all_of(tile.right.begin(), tile.right.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("rpmd_ja qNEP cache source tile is malformed or non-finite");
    seen[index] = 1;
  }
  if (std::any_of(seen.begin(), seen.end(), [](unsigned char x) { return x == 0; }))
    throw std::runtime_error("rpmd_ja qNEP cache source tile grid has a hole");
  return matrix.tiles.size();
}

static __global__ void expand_tile(
  const int D, const int row0, const int column0, const int rows, const int columns, const int rank,
  const double* left, const double* right, double* matrix)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * columns;
  if (index >= count) return;
  const int row = index / columns;
  const int column = index - row * columns;
  double value = 0.0;
  if (rank == 0) {
    value = left[index];
  } else {
    for (int k = 0; k < rank; ++k)
      value += left[static_cast<std::size_t>(row) * rank + k] *
        right[static_cast<std::size_t>(k) * columns + column];
  }
  matrix[row0 + row + static_cast<std::size_t>(column0 + column) * D] = value;
}

void upload_tiles(
  const RpmdJABlockMatrix& source, const int D, GPU_Vector<double>& left,
  GPU_Vector<double>& right, double* matrix)
{
  for (const auto& tile : source.tiles) {
    left.copy_from_host(tile.left.data(), tile.left.size());
    if (!tile.right.empty()) right.copy_from_host(tile.right.data(), tile.right.size());
    expand_tile<<<(tile.rows * tile.columns + threads - 1) / threads, threads>>>(
      D, tile.row, tile.column, tile.rows, tile.columns, tile.rank,
      left.data(), tile.rank ? right.data() : nullptr, matrix);
    cuda_check(cudaGetLastError(), "launch source tile expansion");
  }
  launch_check("expand source matrix");
}

static __global__ void update_householder_matrix(
  const int D, const double* h, const double* w, const double scalar, double* matrix)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(D) * D;
  if (index >= count) return;
  const int row = static_cast<int>(index % D);
  const int column = static_cast<int>(index / D);
  matrix[index] += -2.0 * h[row] * w[column] - 2.0 * w[row] * h[column] +
    4.0 * scalar * h[row] * h[column];
}

static __global__ void compact_matrix(
  const int r, const int D, const int* map, const double* matrix, double* compact)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(r) * r;
  if (index >= count) return;
  const int row = static_cast<int>(index % r);
  const int column = static_cast<int>(index / r);
  compact[index] = matrix[map[row] + static_cast<std::size_t>(map[column]) * D];
}

static __global__ void embed_eigenvectors(
  const int r, const int D, const int* map, const double* compact, double* eigenvectors)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(r) * r;
  if (index >= count) return;
  const int row = static_cast<int>(index % r);
  const int column = static_cast<int>(index / r);
  eigenvectors[map[row] + static_cast<std::size_t>(column) * D] = compact[index];
}

static __global__ void apply_householder_eigenvectors(
  const int D, const int r, const double* h, const double* coefficients, double* eigenvectors)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(D) * r;
  if (index >= count) return;
  const int row = static_cast<int>(index % D);
  const int column = static_cast<int>(index / D);
  eigenvectors[index] -= 2.0 * h[row] * coefficients[column];
}

static __global__ void evaluate_cheb_modes(
  const int r, const int degree, const int p_rank, const int q_rank, const double lambda_scale,
  const double* eigenvalues, const double* p_vectors, const double* q_vectors, double* modes)
{
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t count = static_cast<std::size_t>(r) * (p_rank + q_rank);
  if (index >= count) return;
  const int mode_rank = static_cast<int>(index / r);
  const int mode = static_cast<int>(index - static_cast<std::size_t>(mode_rank) * r);
  const bool is_p = mode_rank < p_rank;
  const int rank = is_p ? mode_rank : mode_rank - p_rank;
  const double* vectors = is_p ? p_vectors : q_vectors;
  const int rank_count = is_p ? p_rank : q_rank;
  const double z = 2.0 * eigenvalues[mode] / lambda_scale - 1.0;
  double f = vectors[rank];
  if (degree >= 1) f += vectors[rank_count + rank] * z;
  double t0 = 1.0, t1 = z;
  for (int k = 2; k <= degree; ++k) {
    const double t2 = 2.0 * z * t1 - t0;
    f += vectors[static_cast<std::size_t>(k) * rank_count + rank] * t2;
    t0 = t1;
    t1 = t2;
  }
  modes[index] = f;
}

static __global__ void build_kernel_matrix(
  const int r, const int p_rank, const int q_rank, const double tau,
  const double* eigenvalues, const double* p_values, const double* q_values,
  const double* modes, double* at_kernel, int* error)
{
  const std::size_t pair = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int a = static_cast<int>(pair % r);
  const int b = static_cast<int>(pair / r);
  if (b >= r || a > b) return;
  double p = 0.0, q = 0.0;
  for (int rank = 0; rank < p_rank; ++rank)
    p += p_values[rank] * modes[static_cast<std::size_t>(rank) * r + a] *
      modes[static_cast<std::size_t>(rank) * r + b];
  for (int rank = 0; rank < q_rank; ++rank) {
    const std::size_t offset = static_cast<std::size_t>(p_rank + rank) * r;
    q += q_values[rank] * modes[offset + a] * modes[offset + b];
  }
  const double la = eigenvalues[a], lb = eigenvalues[b];
  const std::size_t ab = static_cast<std::size_t>(a) + static_cast<std::size_t>(b) * r;
  const std::size_t ba = static_cast<std::size_t>(b) + static_cast<std::size_t>(a) * r;
  const double at_ab = at_kernel[ab], at_ba = at_kernel[ba];
  const double tau2 = tau * tau;
  const double ta = tau2 * la, tb = tau2 * lb;
  const double value_ab = ta * tb * p * at_ab + tb * q * at_ba;
  const double value_ba = tb * ta * p * at_ba + ta * q * at_ab;
  if (!isfinite(value_ab) || !isfinite(value_ba)) atomicOr(error, 1);
  at_kernel[ab] = isfinite(value_ab) ? value_ab : 0.0;
  at_kernel[ba] = isfinite(value_ba) ? value_ba : 0.0;
}

static __global__ void build_weighted_inputs(
  const int N, const int D, const double* continuous, const double* reference,
  const double* velocity, const double* sqrt_mass, double* weighted, int* error)
{
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  const int atom = d % N;
  const double y = sqrt_mass[atom] * (continuous[d] - reference[d]);
  const double u = sqrt_mass[atom] * velocity[d];
  if (!isfinite(y) || !isfinite(u)) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  weighted[d] = isfinite(y) ? y : 0.0;
  weighted[D + d] = isfinite(u) ? u : 0.0;
}

static __global__ void check_sample_values(const int r, const double* modal, const double* action, const double* result, int* error)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < r && (!isfinite(modal[i]) || !isfinite(modal[r + i]) || !isfinite(action[i])))
    atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
  if (i == 0 && !isfinite(*result)) atomicOr(error, RPMD_JA_ERROR_NUMERICAL);
}

std::vector<int> make_index_map(const int D, const int N)
{
  std::vector<int> map;
  map.reserve(D - 3);
  for (int i = 0; i < D; ++i)
    if (i != 0 && i != N && i != 2 * N) map.push_back(i);
  return map;
}

std::vector<double> householder_vectors(
  const int N, const int D, const std::vector<double>& sqrt_mass, const double mass_sum)
{
  std::vector<double> result(static_cast<std::size_t>(3) * D, 0.0);
  for (int axis = 0; axis < 3; ++axis) {
    double norm2 = 0.0;
    for (int atom = 0; atom < N; ++atom) {
      const double t = sqrt_mass[atom] / std::sqrt(mass_sum);
      const double value = t - (atom == 0 ? 1.0 : 0.0);
      result[static_cast<std::size_t>(axis) * D + axis * N + atom] = value;
      norm2 += value * value;
    }
    if (!(norm2 > 0.0) || !std::isfinite(norm2))
      throw std::runtime_error("rpmd_ja qNEP cache cannot form a translation Householder vector");
    const double inverse_norm = 1.0 / std::sqrt(norm2);
    for (int atom = 0; atom < N; ++atom)
      result[static_cast<std::size_t>(axis) * D + axis * N + atom] *= inverse_norm;
  }
  return result;
}

struct SolverHandle
{
  cusolverDnHandle_t value = nullptr;
  SolverHandle() { solver_check(cusolverDnCreate(&value), "cusolverDnCreate"); }
  ~SolverHandle()
  {
    if (value && cusolverDnDestroy(value) != CUSOLVER_STATUS_SUCCESS) {
      std::fprintf(stderr, "rpmd_ja qNEP cache failed to destroy its cuSOLVER handle.\n");
      std::terminate();
    }
  }
};

void describe_spectrum(const std::vector<double>& eigenvalues, const double lambda_scale)
{
  for (int i = 0; i < static_cast<int>(eigenvalues.size()); ++i) {
    const double value = eigenvalues[i];
    if (!std::isfinite(value) || !(value > 0.0)) {
      std::fprintf(stderr, "rpmd_ja qNEP cached reference has a non-positive physical eigenvalue at mode %d: %.17g\n", i, value);
      std::exit(1);
    }
    if (value > lambda_scale * (1.0 + 1.0e-10)) {
      std::fprintf(stderr, "rpmd_ja qNEP cached eigenvalue %.17g exceeds kernel Lambda %.17g\n", value, lambda_scale);
      std::exit(1);
    }
  }
}
} // namespace
#endif

RpmdJAQNEPCached::~RpmdJAQNEPCached()
{
#ifndef USE_HIP
  if (blas_handle_) {
    const cublasStatus_t status = cublasDestroy(static_cast<cublasHandle_t>(blas_handle_));
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "rpmd_ja qNEP cache failed to destroy its cuBLAS handle.\n");
      std::terminate();
    }
  }
#endif
}

void RpmdJAQNEPCached::initialize(
  const RpmdJAReference& reference, const std::vector<double>& masses)
{
#ifdef USE_HIP
  (void)reference;
  (void)masses;
  PRINT_INPUT_ERROR("rpmd_ja qNEP v3 cached evaluation is CUDA-only; HIP remains supported for backend v2.");
#else
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  number_of_atoms_ = reference.number_of_atoms;
  dimension_ = 3 * number_of_atoms_;
  modal_dimension_ = dimension_ - 3;
  degree_ = reference.kernel_degree;
  p_rank_ = reference.p_rank;
  q_rank_ = reference.q_rank;
  if (reference.backend != 2 || !reference.stability_checked || number_of_atoms_ <= 1 ||
      masses.size() != static_cast<std::size_t>(number_of_atoms_) || degree_ < 0 ||
      p_rank_ < 0 || q_rank_ < 0 ||
      reference.p_values.size() != static_cast<std::size_t>(p_rank_) ||
      reference.q_values.size() != static_cast<std::size_t>(q_rank_) ||
      reference.p_vectors.size() != static_cast<std::size_t>(degree_ + 1) * p_rank_ ||
      reference.q_vectors.size() != static_cast<std::size_t>(degree_ + 1) * q_rank_)
    throw std::runtime_error("rpmd_ja qNEP cached reference has invalid dimensions or kernel tables");
  dynamical_tile_count_ = validate_tiles(reference.block_dynamical, dimension_);
  for (int alpha = 0; alpha < 3; ++alpha)
    site_tile_count_[alpha] = validate_tiles(reference.block_site_transpose[alpha], dimension_);
  double mass_sum = 0.0;
  std::vector<double> sqrt_mass(number_of_atoms_);
  for (int atom = 0; atom < number_of_atoms_; ++atom) {
    if (!(masses[atom] > 0.0) || !std::isfinite(masses[atom]))
      throw std::runtime_error("rpmd_ja qNEP cached reference has invalid masses");
    sqrt_mass[atom] = std::sqrt(masses[atom]);
    mass_sum += masses[atom];
  }
  if (!(mass_sum > 0.0) || !std::isfinite(mass_sum))
    throw std::runtime_error("rpmd_ja qNEP cached reference has an invalid total mass");
  tau_ = HBAR / (K_B * reference.temperature);
  lambda_scale_ = reference.kernel_u * reference.kernel_u / (tau_ * tau_);
  if (!std::isfinite(tau_) || !(tau_ > 0.0) || !std::isfinite(lambda_scale_) || !(lambda_scale_ > 0.0) ||
      reference.spectral_bound > lambda_scale_ * (1.0 + 1.0e-10))
    throw std::runtime_error("rpmd_ja qNEP cached kernel interval does not cover its reference");

  cublasHandle_t blas = nullptr;
  blas_check(cublasCreate(&blas), "cublasCreate");
  blas_handle_ = blas;
  blas_check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "set host pointer mode");
  SolverHandle solver;
  const std::size_t D2 = static_cast<std::size_t>(dimension_) * dimension_;
  const std::size_t Dr = static_cast<std::size_t>(dimension_) * modal_dimension_;
  const std::size_t r2 = static_cast<std::size_t>(modal_dimension_) * modal_dimension_;
  const std::size_t max_tile = static_cast<std::size_t>(tile_size) * tile_size;
  std::size_t query_bytes = 0;
  add_count(query_bytes, r2, sizeof(double));
  add_count(query_bytes, static_cast<std::size_t>(modal_dimension_), sizeof(double));
  add_count(query_bytes, static_cast<std::size_t>(number_of_atoms_), sizeof(double));
  require_gpu_bytes(query_bytes, "Dsyevd workspace query buffers");
  std::size_t peak_bytes = 0;
  GPU_Vector<double> eigenvalues(modal_dimension_);
  {
  GPU_Vector<double> compact(r2);
  sqrt_mass_.resize(number_of_atoms_);
  sqrt_mass_.copy_from_host(sqrt_mass.data());
  int lwork = 0;
  solver_check(cusolverDnDsyevd_bufferSize(
    solver.value, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    modal_dimension_, compact.data(), modal_dimension_, eigenvalues.data(), &lwork), "query Dsyevd workspace");
  if (lwork <= 0) throw std::runtime_error("rpmd_ja qNEP cached Dsyevd returned an invalid workspace size");

  std::size_t build_bytes = 0;
  add_count(build_bytes, D2, sizeof(double));
  add_count(build_bytes, static_cast<std::size_t>(2) * dimension_, sizeof(double));
  add_count(build_bytes, static_cast<std::size_t>(2) * max_tile, sizeof(double));
  add_count(build_bytes, 1, sizeof(int));
  add_bytes(build_bytes, query_bytes);
  std::size_t compact_total = build_bytes;
  add_count(compact_total, static_cast<std::size_t>(modal_dimension_), sizeof(int));
  std::size_t eigensolver_bytes = 0;
  add_count(eigensolver_bytes, r2, sizeof(double));
  add_count(eigensolver_bytes, static_cast<std::size_t>(lwork), sizeof(double));
  add_count(eigensolver_bytes, 1, sizeof(int));
  add_count(eigensolver_bytes, static_cast<std::size_t>(modal_dimension_), sizeof(int));
  add_count(eigensolver_bytes, static_cast<std::size_t>(2) * modal_dimension_ + number_of_atoms_, sizeof(double));
  std::size_t embed_bytes = 0;
  add_count(embed_bytes, r2 + Dr, sizeof(double));
  add_count(embed_bytes, static_cast<std::size_t>(modal_dimension_), sizeof(int));
  add_count(embed_bytes, static_cast<std::size_t>(dimension_ + 2 * modal_dimension_ + number_of_atoms_), sizeof(double));
  std::size_t cache_build_bytes = 0;
  add_count(cache_build_bytes, Dr, sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(3) * r2, sizeof(double));
  add_count(cache_build_bytes, D2, sizeof(double));
  add_count(cache_build_bytes, Dr, sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(p_rank_ + q_rank_) * modal_dimension_, sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(degree_ + 1) * (p_rank_ + q_rank_), sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(p_rank_ + q_rank_), sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(modal_dimension_), sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(number_of_atoms_), sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(4) * modal_dimension_ + 2 * dimension_, sizeof(double));
  add_count(cache_build_bytes, static_cast<std::size_t>(2) * max_tile, sizeof(double));
  add_count(cache_build_bytes, 1, sizeof(int));
  std::size_t steady_bytes = 0;
  add_count(steady_bytes, Dr, sizeof(double));
  add_count(steady_bytes, static_cast<std::size_t>(3) * r2, sizeof(double));
  add_count(steady_bytes, static_cast<std::size_t>(number_of_atoms_), sizeof(double));
  add_count(steady_bytes, static_cast<std::size_t>(2) * dimension_, sizeof(double));
  add_count(steady_bytes, static_cast<std::size_t>(3) * modal_dimension_, sizeof(double));
  add_count(steady_bytes, 2, sizeof(double));
  peak_bytes = std::max({build_bytes, compact_total, eigensolver_bytes,
    embed_bytes, cache_build_bytes, steady_bytes});
  estimated_peak_bytes_ = peak_bytes;
  require_gpu_bytes(peak_bytes - query_bytes, "initialization peak after workspace-query buffers");

  std::size_t physical_extra = 0;
  add_count(physical_extra, D2, sizeof(double));
  add_count(physical_extra, static_cast<std::size_t>(2) * dimension_ + 2 * max_tile, sizeof(double));
  add_count(physical_extra, 1, sizeof(int));
  require_gpu_bytes(physical_extra, "D assembly and translation projection");
  const auto householder = householder_vectors(number_of_atoms_, dimension_, sqrt_mass, mass_sum);
  const std::vector<int> host_map = make_index_map(dimension_, number_of_atoms_);
  {
    require_gpu_bytes(static_cast<std::size_t>(modal_dimension_) * sizeof(int), "translation-complement index map");
    GPU_Vector<int> device_map(modal_dimension_);
    device_map.copy_from_host(host_map.data());
    {
      GPU_Vector<double> physical(D2), h(dimension_), w(dimension_);
      GPU_Vector<double> tile_left(max_tile), tile_right(max_tile);
      physical.fill(0.0);
      upload_tiles(reference.block_dynamical, dimension_, tile_left, tile_right, physical.data());
      const double one = 1.0, zero = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        h.copy_from_host(householder.data() + static_cast<std::size_t>(axis) * dimension_);
        blas_check(cublasDgemv(blas, CUBLAS_OP_N, dimension_, dimension_,
          &one, physical.data(), dimension_, h.data(), 1, &zero, w.data(), 1), "D times translation reflector");
        double scalar = 0.0;
        blas_check(cublasDdot(blas, dimension_, h.data(), 1, w.data(), 1, &scalar), "translation reflector norm");
        update_householder_matrix<<<(D2 + threads - 1) / threads, threads>>>(
          dimension_, h.data(), w.data(), scalar, physical.data());
        cuda_check(cudaGetLastError(), "launch translation projection");
      }
      launch_check("project known translation modes");
      compact_matrix<<<(r2 + threads - 1) / threads, threads>>>(
        modal_dimension_, dimension_, device_map.data(), physical.data(), compact.data());
      launch_check("compact translation complement");
    }
    std::size_t eigensolver_extra = 0;
    add_count(eigensolver_extra, static_cast<std::size_t>(lwork), sizeof(double));
    add_count(eigensolver_extra, 1, sizeof(int));
    require_gpu_bytes(eigensolver_extra, "Dsyevd workspace");
    std::vector<double> host_eigenvalues(modal_dimension_);
    {
      GPU_Vector<double> work(lwork);
      GPU_Vector<int> info(1);
      solver_check(cusolverDnDsyevd(solver.value, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        modal_dimension_, compact.data(), modal_dimension_, eigenvalues.data(), work.data(), lwork, info.data()),
        "solve projected dynamical spectrum");
      launch_check("Dsyevd");
      int solver_info = 0;
      info.copy_to_host(&solver_info, 1);
      if (solver_info != 0) {
        std::fprintf(stderr, "rpmd_ja qNEP cached Dsyevd failed with info=%d.\n", solver_info);
        std::exit(1);
      }
      eigenvalues.copy_to_host(host_eigenvalues.data());
      describe_spectrum(host_eigenvalues, lambda_scale_);
    }
    std::size_t embedding_extra = 0;
    add_count(embedding_extra, Dr, sizeof(double));
    add_count(embedding_extra, static_cast<std::size_t>(dimension_ + modal_dimension_), sizeof(double));
    require_gpu_bytes(embedding_extra, "full eigenvector embedding");
    eigenvectors_.resize(Dr);
    eigenvectors_.fill(0.0);
    embed_eigenvectors<<<(r2 + threads - 1) / threads, threads>>>(
      modal_dimension_, dimension_, device_map.data(), compact.data(), eigenvectors_.data());
    launch_check("embed projected eigenvectors");
    {
      GPU_Vector<double> h(dimension_), coefficients(modal_dimension_);
      const double one = 1.0, zero = 0.0;
      for (int axis = 2; axis >= 0; --axis) {
        h.copy_from_host(householder.data() + static_cast<std::size_t>(axis) * dimension_);
        blas_check(cublasDgemv(blas, CUBLAS_OP_T, dimension_, modal_dimension_,
          &one, eigenvectors_.data(), dimension_, h.data(), 1, &zero, coefficients.data(), 1),
          "map eigenvectors back from translation reflector");
        apply_householder_eigenvectors<<<(Dr + threads - 1) / threads, threads>>>(
          dimension_, modal_dimension_, h.data(), coefficients.data(), eigenvectors_.data());
        cuda_check(cudaGetLastError(), "launch inverse Householder transform");
      }
      launch_check("restore physical eigenvectors");
    }
  }
  }
  std::size_t cache_extra = 0;
  add_count(cache_extra, static_cast<std::size_t>(3) * r2, sizeof(double));
  add_count(cache_extra, D2 + Dr, sizeof(double));
  add_count(cache_extra, static_cast<std::size_t>(p_rank_ + q_rank_) * modal_dimension_, sizeof(double));
  add_count(cache_extra, static_cast<std::size_t>(degree_ + 1) * (p_rank_ + q_rank_), sizeof(double));
  add_count(cache_extra, static_cast<std::size_t>(p_rank_ + q_rank_) + 5, sizeof(double));
  add_count(cache_extra, static_cast<std::size_t>(2) * max_tile, sizeof(double));
  add_count(cache_extra, 1, sizeof(int));
  require_gpu_bytes(cache_extra, "cached B projections");
  for (int alpha = 0; alpha < 3; ++alpha) kernel_[alpha].resize(r2);
  {
    GPU_Vector<double> bt(D2), transform(Dr);
    GPU_Vector<double> tile_left(max_tile), tile_right(max_tile);
    GPU_Vector<double> p_values(std::max(1, p_rank_)), q_values(std::max(1, q_rank_));
    GPU_Vector<double> p_vectors(std::max<std::size_t>(1, static_cast<std::size_t>(degree_ + 1) * p_rank_));
    GPU_Vector<double> q_vectors(std::max<std::size_t>(1, static_cast<std::size_t>(degree_ + 1) * q_rank_));
    GPU_Vector<double> modes(std::max<std::size_t>(1, static_cast<std::size_t>(p_rank_ + q_rank_) * modal_dimension_));
    GPU_Vector<int> cache_error(1, 0);
    if (p_rank_ > 0) {
      p_values.copy_from_host(reference.p_values.data());
      p_vectors.copy_from_host(reference.p_vectors.data());
    }
    if (q_rank_ > 0) {
      q_values.copy_from_host(reference.q_values.data());
      q_vectors.copy_from_host(reference.q_vectors.data());
    }
    if (p_rank_ + q_rank_ > 0) {
      const std::size_t mode_count = static_cast<std::size_t>(p_rank_ + q_rank_) * modal_dimension_;
      evaluate_cheb_modes<<<(mode_count + threads - 1) / threads, threads>>>(
        modal_dimension_, degree_, p_rank_, q_rank_, lambda_scale_, eigenvalues.data(),
        p_vectors.data(), q_vectors.data(), modes.data());
      launch_check("evaluate kernel polynomials on the reference spectrum");
    }
    const double one = 1.0, zero = 0.0;
    for (int alpha = 0; alpha < 3; ++alpha) {
      upload_tiles(reference.block_site_transpose[alpha], dimension_, tile_left, tile_right, bt.data());
      blas_check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, dimension_, modal_dimension_, dimension_,
        &one, bt.data(), dimension_, eigenvectors_.data(), dimension_, &zero, transform.data(), dimension_),
        "calculate B-transpose E");
      blas_check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, modal_dimension_, modal_dimension_, dimension_,
        &one, eigenvectors_.data(), dimension_, transform.data(), dimension_, &zero,
        kernel_[alpha].data(), modal_dimension_), "calculate E-transpose B-transpose E");
      cache_error.fill(0);
      const std::size_t pair_count = r2;
      build_kernel_matrix<<<(pair_count + threads - 1) / threads, threads>>>(
        modal_dimension_, p_rank_, q_rank_, tau_, eigenvalues.data(), p_values.data(), q_values.data(),
        modes.data(), kernel_[alpha].data(), cache_error.data());
      launch_check("build cached current matrix");
      int kernel_error = 0;
      cache_error.copy_to_host(&kernel_error, 1);
      if (kernel_error != 0)
        throw std::runtime_error("rpmd_ja qNEP cache contains a non-finite matrix element");
    }
  }

  std::size_t steady_actual = 0;
  add_count(steady_actual, eigenvectors_.size(), sizeof(double));
  for (int alpha = 0; alpha < 3; ++alpha) add_count(steady_actual, kernel_[alpha].size(), sizeof(double));
  add_count(steady_actual, sqrt_mass_.size(), sizeof(double));
  std::size_t sample_extra = 0;
  add_count(sample_extra, static_cast<std::size_t>(2) * dimension_ + 3 * modal_dimension_ + 2, sizeof(double));
  require_gpu_bytes(sample_extra, "steady sample buffers");
  weighted_yu_.resize(static_cast<std::size_t>(2) * dimension_);
  modal_yu_.resize(static_cast<std::size_t>(2) * modal_dimension_);
  scratch_.resize(modal_dimension_);
  const double one = 1.0, zero = 0.0;
  alpha_scalar_.resize(1);
  beta_scalar_.resize(1);
  alpha_scalar_.copy_from_host(&one);
  beta_scalar_.copy_from_host(&zero);
  add_count(steady_actual, 2, sizeof(double));
  add_count(steady_actual, weighted_yu_.size() + modal_yu_.size() + scratch_.size(), sizeof(double));
  allocated_bytes_ = steady_actual;
  preparation_seconds_ = std::chrono::duration<double>(Clock::now() - start).count();
  blas_check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE), "set device pointer mode");
#endif
}

void RpmdJAQNEPCached::compute_correction(
  const GPU_Vector<double>& continuous,
  const GPU_Vector<double>& reference_positions,
  const GPU_Vector<double>& velocity,
  double* device_result,
  int* device_error)
{
#ifdef USE_HIP
  (void)continuous; (void)reference_positions; (void)velocity; (void)device_result; (void)device_error;
  PRINT_INPUT_ERROR("rpmd_ja qNEP v3 cached evaluation is CUDA-only.");
#else
  if (!blas_handle_ || number_of_atoms_ <= 0 || modal_dimension_ <= 0 ||
      device_result == nullptr || device_error == nullptr)
    PRINT_INPUT_ERROR("rpmd_ja qNEP cached correction received invalid buffers.");
  cublasHandle_t blas = static_cast<cublasHandle_t>(blas_handle_);
  const int D = dimension_, r = modal_dimension_;
  cuda_check(cudaGetLastError(), "before qNEP cached sample");
  build_weighted_inputs<<<(D + threads - 1) / threads, threads>>>(
    number_of_atoms_, D, continuous.data(), reference_positions.data(), velocity.data(),
    sqrt_mass_.data(), weighted_yu_.data(), device_error);
  cuda_check(cudaGetLastError(), "build weighted sample inputs");
  blas_check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, r, 2, D,
    alpha_scalar_.data(), eigenvectors_.data(), D, weighted_yu_.data(), D,
    beta_scalar_.data(), modal_yu_.data(), r),
    "project sampled positions and velocities");
  for (int alpha = 0; alpha < 3; ++alpha) {
    blas_check(cublasDgemv(blas, CUBLAS_OP_N, r, r, alpha_scalar_.data(), kernel_[alpha].data(), r,
      modal_yu_.data(), 1, beta_scalar_.data(), scratch_.data(), 1), "apply cached current matrix");
    blas_check(cublasDdot(blas, r, modal_yu_.data() + r, 1, scratch_.data(), 1,
      device_result + alpha), "contract cached current");
    check_sample_values<<<(r + threads - 1) / threads, threads>>>(
      r, modal_yu_.data(), scratch_.data(), device_result + alpha, device_error);
    cuda_check(cudaGetLastError(), "check cached sample values");
  }
#endif
}
