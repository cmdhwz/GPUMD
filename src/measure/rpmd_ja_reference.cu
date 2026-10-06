#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_reference_math.cuh"
#include "rpmd_ja_qnep_prepare.cuh"
#include "force/force.cuh"
#include "force/nep.cuh"
#include "force/nep_charge.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>

#ifdef USE_HIP
#include <hipblas/hipblas.h>
#define JA_DGEMM hipblasDgemm
#define JA_BLAS_OP_N HIPBLAS_OP_N
#define JA_BLAS_OP_T HIPBLAS_OP_T
#define JA_BLAS_STATUS hipblasStatus_t
#define JA_BLAS_SUCCESS HIPBLAS_STATUS_SUCCESS
#else
#include <cublas_v2.h>
#define JA_DGEMM cublasDgemm
#define JA_BLAS_OP_N CUBLAS_OP_N
#define JA_BLAS_OP_T CUBLAS_OP_T
#define JA_BLAS_STATUS cublasStatus_t
#define JA_BLAS_SUCCESS CUBLAS_STATUS_SUCCESS
#endif

namespace
{
using EdgeKey = std::tuple<int, int, int, int, int>;
constexpr char kMagic[8] = {'G', 'P', 'U', 'M', 'D', 'J', 'A', '\0'};
constexpr char kUnits[] = "position:A;energy:eV;mass:amu;temperature:K;time:native";
constexpr char kLayoutDense[] = "xyz_soa;delta_h_rowmajor_displacement_velocity";
constexpr char kLayoutSparse[] = "xyz_soa;csr_output_row_input_col;runtime_translation_projection;cheb_rowmajor_degree_rank;fixed_d0_edges_v1";
constexpr char kLayoutBlock[] = "xyz_soa;native_reference_transport;block_tiles_128;D_then_Bt;rowmajor_output_input";
constexpr std::uint32_t kVersionDense = 1;
constexpr std::uint32_t kVersionSparse = 2;
constexpr std::uint32_t kVersionBlock = 3;
constexpr std::uint32_t kEndian = 0x01020304;
constexpr double kHessianSymmetryTolerance = 5.0e-2;
constexpr double kDifferenceTolerance = 5.0e-2;
constexpr double kForceTolerance = 1.0e-4;
constexpr double kSupportLeakTolerance = 1.0e-8;
constexpr char kNativeReferencePolicy[] = "native_reference_transport";
constexpr char kAnalyticSitePolicy[] = "native_reference_transport;analytic_site_gradient_v1";
constexpr char kAnalyticSiteFd4Policy[] = "native_reference_transport;analytic_site_gradient_fd4_v1";

bool parse_additive_policy(const std::string& policy, int& beads, int& derivative)
{
  const std::string p = policy.rfind("native_reference_transport;internal_mass_com_pullback_v1;", 0) == 0 ?
    "native_reference_transport;internal_mass_com_pullback_v1;finite_temperature_additive_v1;beads=" :
    "native_reference_transport;finite_temperature_additive_v1;beads=";
  if (policy.compare(0, p.size(), p) != 0) return false;
  const std::size_t split = policy.find(";derivative=", p.size());
  if (split == std::string::npos) return false;
  auto parse = [](const std::string& s, int& v) {
    if (s.empty() || (s.size() > 1 && s[0] == '0')) return false;
    int value = 0;
    for (char c : s) {
      if (c < '0' || c > '9' || value > (std::numeric_limits<int>::max() - (c-'0')) / 10) return false;
      value = value * 10 + (c-'0');
    }
    v = value;
    return true;
  };
  const std::string b = policy.substr(p.size(), split-p.size());
  const std::string v = policy.substr(split+12);
  if (!parse(b, beads) || !parse(v, derivative) || beads <= 0 || derivative < 1 || derivative > 3)
    return false;
  return std::to_string(beads) == b && std::to_string(derivative) == v;
}

bool is_supported_qnep_policy(const std::string& policy)
{
  int beads=0, derivative=0;
  return policy == kNativeReferencePolicy || policy == kAnalyticSitePolicy ||
    policy == kAnalyticSiteFd4Policy || parse_additive_policy(policy, beads, derivative);
}

std::vector<std::array<int, 3>> periodic_wrap_offsets(
  const Box& box,
  const std::vector<double>& xyz,
  const int n)
{
  std::vector<std::array<int, 3>> wraps(static_cast<std::size_t>(n));
  for (int atom = 0; atom < n; ++atom) {
    const double x = xyz[atom], y = xyz[n + atom], z = xyz[2 * n + atom];
    const double fractional[3] = {
      box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z,
      box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z,
      box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z};
    for (int axis = 0; axis < 3; ++axis) {
      const int periodic = axis == 0 ? box.pbc_x : (axis == 1 ? box.pbc_y : box.pbc_z);
      const double image = periodic ? std::floor(fractional[axis]) : 0.0;
      if (!std::isfinite(image) || image < std::numeric_limits<int>::min() || image > std::numeric_limits<int>::max())
        throw std::runtime_error("RPMD-JA coordinate lies too many cells from its reference branch");
      wraps[atom][axis] = static_cast<int>(image);
    }
  }
  return wraps;
}

void wrap_positions_once(const Box& box, std::vector<double>& xyz, const int n)
{
  const std::vector<std::array<int, 3>> wraps = periodic_wrap_offsets(box, xyz, n);
  for (int atom = 0; atom < n; ++atom) {
    const double shift[3] = {
      box.cpu_h[0] * wraps[atom][0] + box.cpu_h[1] * wraps[atom][1] + box.cpu_h[2] * wraps[atom][2],
      box.cpu_h[3] * wraps[atom][0] + box.cpu_h[4] * wraps[atom][1] + box.cpu_h[5] * wraps[atom][2],
      box.cpu_h[6] * wraps[atom][0] + box.cpu_h[7] * wraps[atom][1] + box.cpu_h[8] * wraps[atom][2]};
    for (int axis = 0; axis < 3; ++axis) xyz[axis * n + atom] -= shift[axis];
  }
}

void check_cutoff_margin(
  const Box& box,
  const NEP& nep,
  const std::vector<int>& types,
  const std::vector<double>& xyz,
  const int n,
  const double fd_step)
{
  double max_cutoff = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      max_cutoff = std::max(max_cutoff, static_cast<double>(std::max(
        nep.get_pair_radial_cutoff(types[i], types[j]), nep.get_pair_angular_cutoff(types[i], types[j]))));
  const double margin = 2.0 * fd_step + 64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, max_cutoff);
  const double radius = max_cutoff + margin;
  const double inv_rows[3] = {
    std::sqrt(box.cpu_h[9] * box.cpu_h[9] + box.cpu_h[10] * box.cpu_h[10] + box.cpu_h[11] * box.cpu_h[11]),
    std::sqrt(box.cpu_h[12] * box.cpu_h[12] + box.cpu_h[13] * box.cpu_h[13] + box.cpu_h[14] * box.cpu_h[14]),
    std::sqrt(box.cpu_h[15] * box.cpu_h[15] + box.cpu_h[16] * box.cpu_h[16] + box.cpu_h[17] * box.cpu_h[17])};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double raw[3] = {xyz[j] - xyz[i], xyz[n + j] - xyz[n + i], xyz[2 * n + j] - xyz[2 * n + i]};
      const double fractional[3] = {
        box.cpu_h[9] * raw[0] + box.cpu_h[10] * raw[1] + box.cpu_h[11] * raw[2],
        box.cpu_h[12] * raw[0] + box.cpu_h[13] * raw[1] + box.cpu_h[14] * raw[2],
        box.cpu_h[15] * raw[0] + box.cpu_h[16] * raw[1] + box.cpu_h[17] * raw[2]};
      int lo[3], hi[3];
      for (int axis = 0; axis < 3; ++axis) {
        const int periodic = axis == 0 ? box.pbc_x : (axis == 1 ? box.pbc_y : box.pbc_z);
        if (periodic) {
          const double extent = radius * inv_rows[axis];
          lo[axis] = static_cast<int>(std::ceil(fractional[axis] - extent));
          hi[axis] = static_cast<int>(std::floor(fractional[axis] + extent));
        } else {
          lo[axis] = hi[axis] = 0;
        }
      }
      for (int a = lo[0]; a <= hi[0]; ++a)
        for (int b = lo[1]; b <= hi[1]; ++b)
          for (int c = lo[2]; c <= hi[2]; ++c) {
            if (i == j && a == 0 && b == 0 && c == 0) continue;
            const double displacement[3] = {
              raw[0] - box.cpu_h[0] * a - box.cpu_h[1] * b - box.cpu_h[2] * c,
              raw[1] - box.cpu_h[3] * a - box.cpu_h[4] * b - box.cpu_h[5] * c,
              raw[2] - box.cpu_h[6] * a - box.cpu_h[7] * b - box.cpu_h[8] * c};
            const double distance = std::sqrt(displacement[0] * displacement[0] + displacement[1] * displacement[1] +
              displacement[2] * displacement[2]);
            const double radial = nep.get_pair_radial_cutoff(types[i], types[j]);
            const double angular = nep.get_pair_angular_cutoff(types[i], types[j]);
            if (std::abs(distance - radial) <= margin || std::abs(distance - angular) <= margin)
              throw std::runtime_error("RPMD-JA reference has a pair/image within the finite-difference cutoff margin");
          }
    }
}

template <typename T> void write_value(std::ostream& out, const T& value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
  if (!out) throw std::runtime_error("failed writing RPMD-JA reference");
}

template <typename T> void read_value(std::istream& in, T& value)
{
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("truncated RPMD-JA reference");
}

template <typename T> void write_vector(std::ostream& out, const std::vector<T>& values)
{
  if (!values.empty()) out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  if (!out) throw std::runtime_error("failed writing RPMD-JA reference array");
}

template <typename T> void read_vector(std::istream& in, std::vector<T>& values)
{
  if (!values.empty()) in.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
  if (!in) throw std::runtime_error("truncated RPMD-JA reference array");
}

std::size_t checked_bytes(const std::uint64_t count, const std::size_t item_size)
{
  if (count > std::numeric_limits<std::size_t>::max() / item_size)
    throw std::runtime_error("RPMD-JA reference array size overflows this platform");
  return static_cast<std::size_t>(count) * item_size;
}

void require_finite(const std::vector<double>& values, const char* what);

void require_remaining(std::istream& in, const std::streamoff file_size, const std::size_t bytes)
{
  const std::streampos position = in.tellg();
  if (position < 0 || file_size < static_cast<std::streamoff>(position) ||
      bytes > static_cast<std::uint64_t>(file_size - static_cast<std::streamoff>(position)))
    throw std::runtime_error("truncated or oversized RPMD-JA reference array");
}

template <typename T> void read_value_checked(
  std::istream& in,
  T& value,
  const std::streamoff file_size)
{
  require_remaining(in, file_size, sizeof(T));
  read_value(in, value);
}

template <typename T> void read_vector_checked(
  std::istream& in,
  std::vector<T>& values,
  const std::uint64_t count,
  const std::streamoff file_size)
{
  const std::size_t bytes = checked_bytes(count, sizeof(T));
  require_remaining(in, file_size, bytes);
  values.resize(static_cast<std::size_t>(count));
  read_vector(in, values);
}

void validate_sparse_matrix(const RpmdJASparseMatrix& matrix, const int dimension)
{
  if (matrix.row_offsets.size() != static_cast<std::size_t>(dimension) + 1 ||
      matrix.row_offsets.empty() || matrix.row_offsets[0] != 0 ||
      matrix.row_offsets.back() != matrix.values.size() || matrix.columns.size() != matrix.values.size())
    throw std::runtime_error("invalid RPMD-JA CSR dimensions");
  for (int row = 0; row < dimension; ++row) {
    const std::uint64_t begin = matrix.row_offsets[row], end = matrix.row_offsets[row + 1];
    if (begin > end || end > matrix.values.size()) throw std::runtime_error("invalid RPMD-JA CSR row offsets");
    int previous = -1;
    for (std::uint64_t k = begin; k < end; ++k) {
      if (matrix.columns[k] <= previous || matrix.columns[k] < 0 || matrix.columns[k] >= dimension ||
          !std::isfinite(matrix.values[k]))
        throw std::runtime_error("invalid RPMD-JA CSR column or value");
      previous = matrix.columns[k];
    }
  }
}

void write_sparse_matrix(std::ostream& out, const RpmdJASparseMatrix& matrix, const int dimension)
{
  validate_sparse_matrix(matrix, dimension);
  const std::uint64_t nonzeros = matrix.values.size();
  write_value(out, nonzeros);
  write_vector(out, matrix.row_offsets);
  write_vector(out, matrix.columns);
  write_vector(out, matrix.values);
}

void read_sparse_matrix(
  std::istream& in,
  RpmdJASparseMatrix& matrix,
  const int dimension,
  const std::streamoff file_size)
{
  std::uint64_t nonzeros = 0;
  read_value(in, nonzeros);
  const std::uint64_t row_count = static_cast<std::uint64_t>(dimension) + 1;
  const std::size_t rows_bytes = checked_bytes(row_count, sizeof(std::uint64_t));
  const std::size_t columns_bytes = checked_bytes(nonzeros, sizeof(int));
  const std::size_t values_bytes = checked_bytes(nonzeros, sizeof(double));
  if (columns_bytes > std::numeric_limits<std::size_t>::max() - values_bytes ||
      rows_bytes > std::numeric_limits<std::size_t>::max() - columns_bytes - values_bytes)
    throw std::runtime_error("RPMD-JA CSR payload size overflows this platform");
  require_remaining(in, file_size, rows_bytes + columns_bytes + values_bytes);
  read_vector_checked(in, matrix.row_offsets, row_count, file_size);
  read_vector_checked(in, matrix.columns, nonzeros, file_size);
  read_vector_checked(in, matrix.values, nonzeros, file_size);
  validate_sparse_matrix(matrix, dimension);
}

void validate_block_matrix(const RpmdJABlockMatrix& matrix, const int dimension)
{
  if (matrix.tile_size < 64 || matrix.tile_size > 512 || dimension <= 0)
    throw std::runtime_error("invalid RPMD-JA block dimensions");
  const int grid = 1 + (dimension - 1) / matrix.tile_size;
  if (matrix.tiles.size() != static_cast<std::size_t>(grid) * grid)
    throw std::runtime_error("RPMD-JA block matrix does not cover its complete tile grid");
  std::vector<unsigned char> seen(static_cast<std::size_t>(grid) * grid, 0);
  for (const RpmdJAMatrixTile& tile : matrix.tiles) {
    if (tile.row < 0 || tile.column < 0 || tile.row % matrix.tile_size || tile.column % matrix.tile_size ||
        tile.row >= dimension || tile.column >= dimension)
      throw std::runtime_error("invalid RPMD-JA block tile origin");
    const int expected_rows = std::min(matrix.tile_size, dimension - tile.row);
    const int expected_columns = std::min(matrix.tile_size, dimension - tile.column);
    if (tile.rows != expected_rows || tile.columns != expected_columns)
      throw std::runtime_error("invalid RPMD-JA block tile extent");
    const std::size_t index = static_cast<std::size_t>(tile.row / matrix.tile_size) * grid + tile.column / matrix.tile_size;
    if (seen[index]++) throw std::runtime_error("duplicate RPMD-JA block tile");
    if (tile.rank < 0 || tile.rank > std::min(tile.rows, tile.columns))
      throw std::runtime_error("invalid RPMD-JA block tile rank");
    const std::size_t left_count = tile.rank == 0 ? static_cast<std::size_t>(tile.rows) * tile.columns :
      static_cast<std::size_t>(tile.rows) * tile.rank;
    const std::size_t right_count = tile.rank == 0 ? 0 : static_cast<std::size_t>(tile.rank) * tile.columns;
    if (tile.left.size() != left_count || tile.right.size() != right_count)
      throw std::runtime_error("invalid RPMD-JA block tile factor sizes");
    require_finite(tile.left, "block tile left factors");
    require_finite(tile.right, "block tile right factors");
  }
}

void write_block_matrix(std::ostream& out, const RpmdJABlockMatrix& matrix, const int dimension)
{
  validate_block_matrix(matrix, dimension);
  write_value(out, matrix.tile_size);
  const std::uint64_t count = matrix.tiles.size();
  write_value(out, count);
  for (const RpmdJAMatrixTile& tile : matrix.tiles) {
    write_value(out, tile.row); write_value(out, tile.column); write_value(out, tile.rows);
    write_value(out, tile.columns); write_value(out, tile.rank);
    const std::uint64_t left_count = tile.left.size(), right_count = tile.right.size();
    write_value(out, left_count); write_value(out, right_count);
    write_vector(out, tile.left); write_vector(out, tile.right);
  }
}

void read_block_matrix(
  std::istream& in,
  RpmdJABlockMatrix& matrix,
  const int dimension,
  const std::streamoff file_size)
{
  std::uint64_t count = 0;
  read_value_checked(in, matrix.tile_size, file_size);
  read_value_checked(in, count, file_size);
  if (matrix.tile_size < 64 || matrix.tile_size > 512)
    throw std::runtime_error("invalid RPMD-JA block tile size");
  const std::uint64_t grid = 1 + static_cast<std::uint64_t>(dimension - 1) / matrix.tile_size;
  const std::uint64_t expected = grid * grid;
  if (count != expected)
    throw std::runtime_error("invalid RPMD-JA block matrix tile count");
  constexpr std::uint64_t minimum_tile_header = 5 * sizeof(int) + 2 * sizeof(std::uint64_t);
  if (count > std::numeric_limits<std::uint64_t>::max() / minimum_tile_header ||
      count * minimum_tile_header > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))
    throw std::runtime_error("RPMD-JA block matrix header size overflows platform offsets");
  require_remaining(in, file_size, static_cast<std::size_t>(count * minimum_tile_header));
  matrix.tiles.resize(static_cast<std::size_t>(count));
  for (RpmdJAMatrixTile& tile : matrix.tiles) {
    std::uint64_t left_count = 0, right_count = 0;
    read_value_checked(in, tile.row, file_size); read_value_checked(in, tile.column, file_size);
    read_value_checked(in, tile.rows, file_size); read_value_checked(in, tile.columns, file_size);
    read_value_checked(in, tile.rank, file_size); read_value_checked(in, left_count, file_size);
    read_value_checked(in, right_count, file_size);
    if (tile.row < 0 || tile.column < 0 || tile.row % matrix.tile_size || tile.column % matrix.tile_size ||
        tile.row >= dimension || tile.column >= dimension || tile.rows != std::min(matrix.tile_size, dimension - tile.row) ||
        tile.columns != std::min(matrix.tile_size, dimension - tile.column) || tile.rank < 0 ||
        tile.rank > std::min(tile.rows, tile.columns))
      throw std::runtime_error("invalid RPMD-JA tile shape or rank");
    const std::uint64_t expected_left = tile.rank == 0 ?
      static_cast<std::uint64_t>(tile.rows) * tile.columns : static_cast<std::uint64_t>(tile.rows) * tile.rank;
    const std::uint64_t expected_right = tile.rank == 0 ? 0 : static_cast<std::uint64_t>(tile.rank) * tile.columns;
    if (left_count != expected_left || right_count != expected_right)
      throw std::runtime_error("invalid RPMD-JA tile factor counts");
    if (left_count > std::numeric_limits<std::size_t>::max() / sizeof(double) ||
        right_count > std::numeric_limits<std::size_t>::max() / sizeof(double))
      throw std::runtime_error("RPMD-JA block factor size overflows platform size");
    read_vector_checked(in, tile.left, left_count, file_size);
    read_vector_checked(in, tile.right, right_count, file_size);
  }
  validate_block_matrix(matrix, dimension);
}

double block_max_abs_row_sum(const RpmdJABlockMatrix& matrix, const int dimension)
{
  std::vector<double> sums(static_cast<std::size_t>(dimension), 0.0);
  for (const RpmdJAMatrixTile& tile : matrix.tiles) {
    std::vector<double> right_abs_sums;
    if (tile.rank > 0) {
      right_abs_sums.assign(static_cast<std::size_t>(tile.rank), 0.0);
      for (int k = 0; k < tile.rank; ++k)
        for (int j = 0; j < tile.columns; ++j)
          right_abs_sums[k] += std::abs(tile.right[static_cast<std::size_t>(k) * tile.columns + j]);
    }
    for (int i = 0; i < tile.rows; ++i)
      if (tile.rank == 0) {
        for (int j = 0; j < tile.columns; ++j)
          sums[static_cast<std::size_t>(tile.row + i)] +=
            std::abs(tile.left[static_cast<std::size_t>(i) * tile.columns + j]);
      } else {
        double bound = 0.0;
        for (int k = 0; k < tile.rank; ++k) {
          bound += std::abs(tile.left[static_cast<std::size_t>(i) * tile.rank + k]) * right_abs_sums[k];
        }
        sums[static_cast<std::size_t>(tile.row + i)] += bound;
      }
  }
  return *std::max_element(sums.begin(), sums.end());
}

int reference_dimension(const int n)
{
  if (n <= 1 || n > std::numeric_limits<int>::max() / 3)
    throw std::runtime_error("invalid RPMD-JA atom count");
  return 3 * n;
}

std::size_t matrix_size(const int n)
{
  const std::size_t d = static_cast<std::size_t>(reference_dimension(n));
  if (d > std::numeric_limits<std::size_t>::max() / d / (3 * sizeof(double)))
    throw std::runtime_error("RPMD-JA matrix size overflows this platform");
  return d * d;
}

void require_finite(const std::vector<double>& values, const char* what)
{
  if (!std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error(std::string("non-finite values in RPMD-JA ") + what);
}

void check_blas(const JA_BLAS_STATUS status, const char* call)
{
  if (status != JA_BLAS_SUCCESS) throw std::runtime_error(std::string("RPMD-JA BLAS failure: ") + call);
}

void dgemm(
  gpublasHandle_t handle,
  const decltype(JA_BLAS_OP_N) trans_a,
  const decltype(JA_BLAS_OP_N) trans_b,
  const int m,
  const int n,
  const int k,
  const double* a,
  const int lda,
  const double* b,
  const int ldb,
  double* c,
  const int ldc)
{
  const double one = 1.0, zero = 0.0;
  check_blas(JA_DGEMM(handle, trans_a, trans_b, m, n, k, &one, a, lda, b, ldb, &zero, c, ldc), "dgemm");
}

struct Evaluation
{
  std::vector<double> force;
  std::map<EdgeKey, NEP_Local_Edge> edges;
};

struct Evaluator
{
  NEP nep;
  Box& box;
  GPU_Vector<int> type;
  GPU_Vector<double> position, potential, force, virial;
  int n;

  Evaluator(
    const std::string& model,
    const int atom_count,
    const RunInput& run_input,
    Box& input_box,
    const std::vector<int>& types)
    : nep(model.c_str(), atom_count, run_input), box(input_box), type(atom_count), position(3 * atom_count),
      potential(atom_count), force(3 * atom_count), virial(9 * atom_count), n(atom_count)
  {
    nep.N1 = 0;
    nep.N2 = n;
    if (!nep.supports_local_edge_derivatives())
      throw std::runtime_error("rpmd_ja reference generation requires a standard short-range NEP model");
    nep.enable_local_edge_derivatives();
    nep.set_neighbor_rebuild(true);
    nep.set_neighbor_log_enabled(false);
    type.copy_from_host(types.data());
  }

  Evaluation evaluate(const std::vector<double>& xyz)
  {
    position.copy_from_host(xyz.data());
    potential.fill(0.0);
    force.fill(0.0);
    virial.fill(0.0);
    nep.compute(box, type, position, potential, force, virial);
    Evaluation result;
    result.force.resize(static_cast<std::size_t>(3) * n);
    force.copy_to_host(result.force.data());
    require_finite(result.force, "NEP forces");
    std::vector<NEP_Local_Edge> edges;
    nep.copy_local_energy_edges(box, xyz, edges);
    for (const auto& edge : edges) {
      if (edge.center < 0 || edge.center >= n || edge.neighbor < 0 || edge.neighbor >= n)
        throw std::runtime_error("NEP local-edge extraction returned an invalid atom index");
      for (int d = 0; d < 3; ++d)
        if (!std::isfinite(edge.displacement[d]) || !std::isfinite(edge.derivative[d]))
          throw std::runtime_error("NEP local-edge extraction returned non-finite values");
      const EdgeKey key{edge.center, edge.neighbor, edge.image[0], edge.image[1], edge.image[2]};
      result.edges.emplace(key, edge);
    }
    return result;
  }
};

struct QEvaluation
{
  std::vector<double> energy, force, virial;
};

struct QEvaluator
{
  NEP_Charge qnep;
  Box& box;
  GPU_Vector<int> type;
  GPU_Vector<double> position, potential, force, virial, reference_direction, reference_site_derivative, reference_gradient;
  const int n;

  QEvaluator(
    const std::string& model,
    const int atom_count,
    const RunInput& run_input,
    Box& input_box,
    const std::vector<int>& types,
    const double pppm_spacing)
    : qnep(model.c_str(), atom_count, run_input), box(input_box), type(atom_count), position(3 * atom_count),
      potential(atom_count), force(3 * atom_count), virial(9 * atom_count),
      reference_direction(3 * atom_count), reference_site_derivative(atom_count),
      reference_gradient(3 * atom_count), n(atom_count)
  {
    qnep.N1 = 0;
    qnep.N2 = n;
    qnep.configure_mechanical_observer();
    qnep.set_neighbor_rebuild(false);
    qnep.set_neighbor_diagnostics(false);
    qnep.set_pimd_batch_profile(false);
    if (qnep.uses_pppm()) qnep.set_pppm_mesh_spacing(pppm_spacing);
    type.copy_from_host(types.data());
  }

  QEvaluation evaluate(const std::vector<double>& xyz, const bool include_electro = true)
  {
    std::vector<double> wrapped = xyz;
    wrap_positions_once(box, wrapped, n);
    position.copy_from_host(wrapped.data());
    potential.fill(0.0);
    force.fill(0.0);
    virial.fill(0.0);
    if (include_electro) {
      qnep.request_peratom_virial_for_next_force();
      qnep.compute(box, type, position, potential, force, virial);
    } else {
      qnep.compute_non_electro(box, type, position, potential, force, virial);
    }
    QEvaluation result;
    result.energy.resize(static_cast<std::size_t>(n));
    result.force.resize(static_cast<std::size_t>(3) * n);
    result.virial.resize(static_cast<std::size_t>(9) * n);
    potential.copy_to_host(result.energy.data());
    force.copy_to_host(result.force.data());
    virial.copy_to_host(result.virial.data());
    require_finite(result.energy, "qNEP site energies");
    require_finite(result.force, "qNEP forces");
    require_finite(result.virial, "qNEP per-atom virials");
    return result;
  }

  std::vector<double> analytic_gradient()
  {
    if (!qnep.compute_reference_site_energy_derivative(
          box, type, position, force, nullptr, nullptr, reference_gradient))
      throw std::runtime_error("qNEP rpmd_ja analytic gradient rejected the current force frame");
    std::vector<double> result(static_cast<std::size_t>(3) * n);
    reference_gradient.copy_to_host(result.data());
    require_finite(result, "qNEP analytic energy gradient");
    return result;
  }

  std::vector<double> analytic_site_jvp(const std::vector<double>& direction_host)
  {
    if (direction_host.size() != static_cast<std::size_t>(3) * n)
      throw std::runtime_error("qNEP analytic site JVP direction has an invalid size");
    reference_direction.copy_from_host(direction_host.data());
    if (!qnep.compute_reference_site_energy_derivative(
          box, type, position, force, &reference_direction, &reference_site_derivative, reference_gradient))
      throw std::runtime_error("qNEP rpmd_ja analytic site JVP rejected the current force frame");
    std::vector<double> result(n);
    reference_site_derivative.copy_to_host(result.data());
    require_finite(result, "qNEP analytic site JVP");
    return result;
  }

};

std::vector<double> site_flow_vector(
  const Evaluation& eval,
  const std::map<EdgeKey, NEP_Local_Edge>& reference_edges,
  const int alpha,
  const int n)
{
  std::vector<double> flow(static_cast<std::size_t>(3) * n, 0.0);
  for (const auto& entry : reference_edges) {
    const NEP_Local_Edge& ref = entry.second;
    const auto it = eval.edges.find(entry.first);
    if (it == eval.edges.end()) throw std::runtime_error("reference NEP periodic edge changed under finite difference");
    for (int mu = 0; mu < 3; ++mu)
      flow[static_cast<std::size_t>(ref.neighbor + mu * n)] -=
        ref.displacement[alpha] * it->second.derivative[mu];
  }
  if (eval.edges.size() != reference_edges.size())
    throw std::runtime_error("NEP periodic edge set changed under finite difference");
  return flow;
}

void check_same_edges(
  const std::map<EdgeKey, NEP_Local_Edge>& evaluated,
  const std::map<EdgeKey, NEP_Local_Edge>& reference)
{
  if (evaluated.size() != reference.size()) throw std::runtime_error("NEP periodic edge key set changed during RPMD-JA finite differences");
  auto a = evaluated.begin(), b = reference.begin();
  for (; a != evaluated.end(); ++a, ++b)
    if (a->first != b->first) throw std::runtime_error("NEP periodic edge key set changed during RPMD-JA finite differences");
}

std::vector<double> site_flow_vector_subset(
  const Evaluation& eval,
  const std::map<EdgeKey, NEP_Local_Edge>& reference_edges,
  const std::vector<std::vector<EdgeKey>>& keys_by_center,
  const std::vector<int>& centers,
  const int alpha,
  const int n)
{
  std::vector<double> flow(static_cast<std::size_t>(3) * n, 0.0);
  for (int center : centers)
    for (const EdgeKey& key : keys_by_center[center]) {
      const NEP_Local_Edge& ref = reference_edges.find(key)->second;
      const auto it = eval.edges.find(key);
      if (it == eval.edges.end()) throw std::runtime_error("fixed RPMD-JA reference image edge disappeared under finite difference");
      for (int mu = 0; mu < 3; ++mu)
        flow[static_cast<std::size_t>(ref.neighbor + mu * n)] -= ref.displacement[alpha] * it->second.derivative[mu];
    }
  return flow;
}

struct SparseProbe
{
  std::vector<double> force;
  std::vector<double> site_flow[3];
  double unaffected_edge_gradient_residual = 0.0;
};

SparseProbe evaluate_sparse_probe(
  Evaluator& evaluator,
  const std::vector<double>& xyz,
  const std::map<EdgeKey, NEP_Local_Edge>& reference_edges,
  const std::vector<std::vector<EdgeKey>>& keys_by_center,
  const std::vector<int>& centers,
  const int n)
{
  Evaluation eval = evaluator.evaluate(xyz);
  check_same_edges(eval.edges, reference_edges);
  SparseProbe result;
  std::vector<unsigned char> affected(evaluator.n, 0);
  for (int center : centers) affected[center] = 1;
  double gradient_scale = 0.0;
  for (const auto& entry : reference_edges) {
    if (affected[entry.second.center]) continue;
    const NEP_Local_Edge& ref = entry.second;
    const NEP_Local_Edge& current = eval.edges.find(entry.first)->second;
    for (int mu = 0; mu < 3; ++mu) {
      result.unaffected_edge_gradient_residual = std::max(result.unaffected_edge_gradient_residual,
        std::abs(current.derivative[mu] - ref.derivative[mu]));
      gradient_scale = std::max(gradient_scale, std::max(std::abs(current.derivative[mu]), std::abs(ref.derivative[mu])));
    }
  }
  const double gradient_tolerance = kSupportLeakTolerance * std::max(1.0, gradient_scale);
  if (result.unaffected_edge_gradient_residual > gradient_tolerance)
    throw std::runtime_error("NEP local-edge derivative changed outside the reference graph support");
  result.force = std::move(eval.force);
  for (int alpha = 0; alpha < 3; ++alpha)
    result.site_flow[alpha] = site_flow_vector_subset(eval, reference_edges, keys_by_center, centers, alpha, n);
  return result;
}

double relative_rms_difference(const std::vector<double>& x, const std::vector<double>& y)
{
  double diff2 = 0.0, reference2 = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double d = x[i] - y[i];
    diff2 += d * d;
    reference2 += y[i] * y[i];
  }
  return std::sqrt(diff2 / std::max(reference2, 1.0e-30));
}

double relative_antisymmetry(const std::vector<double>& a, const int d)
{
  double diff2 = 0.0, scale2 = 0.0;
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      const double x = a[static_cast<std::size_t>(i) * d + j];
      const double y = a[static_cast<std::size_t>(j) * d + i];
      diff2 += (x - y) * (x - y);
      scale2 += x * x;
    }
  return std::sqrt(diff2 / std::max(scale2, 1.0e-30));
}

double relative_translation_residual(const std::vector<double>& dmat, const std::vector<double>& masses, const int n)
{
  const int d = 3 * n;
  double matrix2 = 0.0, residual2 = 0.0;
  for (double value : dmat) matrix2 += value * value;
  for (int axis = 0; axis < 3; ++axis) {
    double mass_sum = 0.0;
    for (double mass : masses) mass_sum += mass;
    const double inv_norm = 1.0 / std::sqrt(mass_sum);
    for (int row = 0; row < d; ++row) {
      double value = 0.0;
      for (int atom = 0; atom < n; ++atom)
        value += dmat[static_cast<std::size_t>(row) * d + axis * n + atom] *
          std::sqrt(masses[atom]) * inv_norm;
      residual2 += value * value;
    }
  }
  return std::sqrt(residual2 / std::max(3.0 * matrix2, 1.0e-30));
}

void load_kernel_table(const std::string& path, RpmdJAReference& reference)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open RPMD-JA kernel table: " + path);
  std::string token;
  int version = 0;
  in >> token >> version;
  if (token != "GPUMDJA_KERNEL" || version != 1) throw std::runtime_error("unsupported RPMD-JA kernel table format");
  int p_rank = 0, q_rank = 0;
  in >> token >> reference.kernel_u;
  if (token != "U") throw std::runtime_error("missing RPMD-JA kernel U");
  in >> token >> reference.kernel_degree;
  if (token != "degree") throw std::runtime_error("missing RPMD-JA kernel degree");
  in >> token >> p_rank;
  if (token != "P_rank") throw std::runtime_error("missing RPMD-JA P rank");
  in >> token >> q_rank;
  if (token != "Q_rank") throw std::runtime_error("missing RPMD-JA Q rank");
  in >> token >> reference.kernel_error[0];
  if (token != "P_error") throw std::runtime_error("missing RPMD-JA P error budget");
  in >> token >> reference.kernel_error[1];
  if (token != "Q_error") throw std::runtime_error("missing RPMD-JA Q error budget");
  in >> token >> reference.kernel_s2[0];
  if (token != "P_S2") throw std::runtime_error("missing RPMD-JA P S2 budget");
  in >> token >> reference.kernel_s2[1];
  if (token != "Q_S2") throw std::runtime_error("missing RPMD-JA Q S2 budget");
  if (!in || reference.kernel_degree < 0 || reference.kernel_degree > 512 || p_rank <= 0 || q_rank <= 0 ||
      p_rank > reference.kernel_degree + 1 || q_rank > reference.kernel_degree + 1)
    throw std::runtime_error("invalid RPMD-JA kernel table dimensions");
  reference.p_rank = p_rank;
  reference.q_rank = q_rank;
  const auto read_named = [&](const char* expected, std::vector<double>& values, const std::size_t count) {
    in >> token;
    if (token != expected) throw std::runtime_error(std::string("missing RPMD-JA kernel table section: ") + expected);
    values.resize(count);
    for (double& value : values) in >> value;
    if (!in || !std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error(std::string("invalid RPMD-JA kernel table section: ") + expected);
  };
  read_named("P_values", reference.p_values, p_rank);
  read_named("P_vectors", reference.p_vectors, static_cast<std::size_t>(reference.kernel_degree + 1) * p_rank);
  read_named("Q_values", reference.q_values, q_rank);
  read_named("Q_vectors", reference.q_vectors, static_cast<std::size_t>(reference.kernel_degree + 1) * q_rank);
  in >> token;
  std::string extra;
  if (!in || token != "END" || in >> extra || !(reference.kernel_u > 0.0) ||
      !std::isfinite(reference.kernel_u))
    throw std::runtime_error("malformed RPMD-JA kernel table ending or U bound");
  for (int i = 0; i < 2; ++i)
    if (!(reference.kernel_error[i] >= 0.0) || !std::isfinite(reference.kernel_error[i]) ||
        !(reference.kernel_s2[i] >= 0.0) || !std::isfinite(reference.kernel_s2[i]))
      throw std::runtime_error("invalid RPMD-JA kernel table error budget");
}

void columns_to_csr(
  const int dimension,
  const std::vector<std::vector<std::pair<int, double>>>& columns,
  RpmdJASparseMatrix& matrix)
{
  if (columns.size() != static_cast<std::size_t>(dimension)) throw std::runtime_error("invalid sparse column count");
  matrix.row_offsets.assign(static_cast<std::size_t>(dimension) + 1, 0);
  std::uint64_t nonzeros = 0;
  for (const auto& column : columns) {
    nonzeros += column.size();
    for (const auto& entry : column) {
      if (entry.first < 0 || entry.first >= dimension) throw std::runtime_error("sparse derivative support outside matrix");
      ++matrix.row_offsets[static_cast<std::size_t>(entry.first) + 1];
    }
  }
  for (int row = 0; row < dimension; ++row) matrix.row_offsets[row + 1] += matrix.row_offsets[row];
  matrix.columns.resize(static_cast<std::size_t>(nonzeros));
  matrix.values.resize(static_cast<std::size_t>(nonzeros));
  std::vector<std::uint64_t> cursor = matrix.row_offsets;
  for (int column = 0; column < dimension; ++column)
    for (const auto& entry : columns[column]) {
      const std::uint64_t index = cursor[entry.first]++;
      matrix.columns[index] = column;
      matrix.values[index] = entry.second;
    }
  for (int row = 0; row < dimension; ++row) {
    const std::uint64_t begin = matrix.row_offsets[row], end = matrix.row_offsets[row + 1];
    std::vector<std::pair<int, double>> entries;
    entries.reserve(static_cast<std::size_t>(end - begin));
    for (std::uint64_t k = begin; k < end; ++k) entries.emplace_back(matrix.columns[k], matrix.values[k]);
    std::sort(entries.begin(), entries.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    for (std::size_t k = 0; k < entries.size(); ++k) {
      if (k > 0 && entries[k - 1].first == entries[k].first)
        throw std::runtime_error("duplicate structural entry in RPMD-JA sparse matrix");
      matrix.columns[begin + k] = entries[k].first;
      matrix.values[begin + k] = entries[k].second;
    }
  }
  validate_sparse_matrix(matrix, dimension);
}

double symmetrize_csr(RpmdJASparseMatrix& matrix, const int dimension)
{
  const std::vector<double> raw = matrix.values;
  double diff2 = 0.0, scale2 = 0.0;
  for (int row = 0; row < dimension; ++row)
    for (std::uint64_t k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k) {
      const int column = matrix.columns[k];
      const auto first = matrix.columns.begin() + matrix.row_offsets[column];
      const auto last = matrix.columns.begin() + matrix.row_offsets[column + 1];
      const auto found = std::lower_bound(first, last, row);
      if (found == last || *found != row) throw std::runtime_error("RPMD-JA dynamical support is not symmetric");
      const std::size_t transpose_index = static_cast<std::size_t>(found - matrix.columns.begin());
      const double value = raw[k], transpose = raw[transpose_index];
      diff2 += (value - transpose) * (value - transpose);
      scale2 += value * value;
      matrix.values[k] = 0.5 * (value + transpose);
    }
  return std::sqrt(diff2 / std::max(scale2, 1.0e-30));
}

double csr_translation_residual(const RpmdJASparseMatrix& matrix, const std::vector<double>& masses, const int n)
{
  const int dimension = 3 * n;
  double matrix2 = 0.0, residual2 = 0.0;
  double mass_sum = 0.0;
  for (double mass : masses) mass_sum += mass;
  for (double value : matrix.values) matrix2 += value * value;
  for (int axis = 0; axis < 3; ++axis)
    for (int row = 0; row < dimension; ++row) {
      double value = 0.0;
      for (std::uint64_t k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k) {
        const int col = matrix.columns[k];
        if (col / n == axis) value += matrix.values[k] * std::sqrt(masses[col % n] / mass_sum);
      }
      residual2 += value * value;
    }
  return std::sqrt(residual2 / std::max(3.0 * matrix2, 1.0e-30));
}

double csr_max_abs_row_sum(const RpmdJASparseMatrix& matrix)
{
  double bound = 0.0;
  for (std::size_t row = 0; row + 1 < matrix.row_offsets.size(); ++row) {
    double sum = 0.0;
    for (std::uint64_t k = matrix.row_offsets[row]; k < matrix.row_offsets[row + 1]; ++k)
      sum += std::abs(matrix.values[k]);
    bound = std::max(bound, sum);
  }
  return bound;
}

void check_eigensystem(
  const int r,
  const std::vector<double>& dmat,
  const std::vector<double>& values,
  const std::vector<double>& vectors)
{
  double scale2 = 0.0, residual2 = 0.0, orthogonal_max = 0.0;
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < r; ++j) {
      double dot = 0.0;
      for (int k = 0; k < r; ++k)
        dot += vectors[static_cast<std::size_t>(k) + static_cast<std::size_t>(i) * r] *
          vectors[static_cast<std::size_t>(k) + static_cast<std::size_t>(j) * r];
      const double e = dot - (i == j ? 1.0 : 0.0);
      orthogonal_max = std::max(orthogonal_max, std::abs(e));
    }
  for (int mode = 0; mode < r; ++mode) {
    for (int row = 0; row < r; ++row) {
      double dv = 0.0;
      for (int col = 0; col < r; ++col)
        dv += dmat[static_cast<std::size_t>(row) * r + col] *
          vectors[static_cast<std::size_t>(col) + static_cast<std::size_t>(mode) * r];
      const double e = dv - values[mode] * vectors[static_cast<std::size_t>(row) + static_cast<std::size_t>(mode) * r];
      residual2 += e * e;
      scale2 += dv * dv;
    }
  }
  if (orthogonal_max > 1.0e-7 || std::sqrt(residual2 / std::max(scale2, 1.0e-30)) > 1.0e-7)
    throw std::runtime_error("cuSOLVER eigenvectors failed orthogonality or residual checks");
}

void transform_site_matrix(
  const int d,
  const int r,
  const std::vector<double>& q,
  const std::vector<double>& masses,
  const std::vector<double>& dmat,
  const std::vector<double>& site,
  const std::vector<double>& omega,
  const std::vector<double>& eigenvalues,
  const double beta_hbar,
  gpublasHandle_t blas,
  const std::vector<double>& eigenvectors,
  const bool check_modes,
  std::vector<double>& delta_h)
{
  std::vector<double> weighted_site(site.size());
  for (int i = 0; i < d; ++i) {
    const double mi = std::sqrt(masses[i % (d / 3)]);
    for (int j = 0; j < d; ++j) {
      const double mj = std::sqrt(masses[j % (d / 3)]);
      weighted_site[static_cast<std::size_t>(i) * d + j] = site[static_cast<std::size_t>(i) * d + j] / (mi * mj);
    }
  }
  GPU_Vector<double> e_gpu(static_cast<std::size_t>(d) * r);
  GPU_Vector<double> h_gpu(weighted_site.size()), tmp(static_cast<std::size_t>(r) * d);
  GPU_Vector<double> a_gpu(static_cast<std::size_t>(r) * r);
  GPU_Vector<double> dm_gpu(static_cast<std::size_t>(r) * r), btmp(static_cast<std::size_t>(r) * d);
  GPU_Vector<double> out(static_cast<std::size_t>(d) * d);
  std::vector<double> e(static_cast<std::size_t>(d) * r);
  for (int i = 0; i < d; ++i)
    for (int a = 0; a < r; ++a) {
      double x = 0.0;
      for (int k = 0; k < r; ++k)
        x += q[static_cast<std::size_t>(i) * r + k] *
          eigenvectors[static_cast<std::size_t>(k) + static_cast<std::size_t>(a) * r];
      e[static_cast<std::size_t>(i) * r + a] = x;
    }
  e_gpu.copy_from_host(e.data());
  h_gpu.copy_from_host(weighted_site.data());
  if (check_modes) {
    GPU_Vector<double> gram(static_cast<std::size_t>(r) * r);
    std::vector<double> gram_colmajor(gram.size());
    dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_T, r, r, d, e_gpu.data(), r, e_gpu.data(), r, gram.data(), r);
    gram.copy_to_host(gram_colmajor.data());
    double max_error = 0.0;
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < r; ++j)
        max_error = std::max(max_error, std::abs(gram_colmajor[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * r] - (i == j ? 1.0 : 0.0)));
    if (max_error > 1.0e-7) throw std::runtime_error("RPMD-JA mass-weighted mode basis failed E^T E check");

    GPU_Vector<double> d_gpu(dmat.size()), de_gpu(static_cast<std::size_t>(r) * d);
    std::vector<double> de_colmajor(de_gpu.size());
    d_gpu.copy_from_host(dmat.data());
    dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_N, r, d, d, e_gpu.data(), r, d_gpu.data(), d, de_gpu.data(), r);
    de_gpu.copy_to_host(de_colmajor.data());
    double residual2 = 0.0, scale2 = 0.0;
    for (int mode = 0; mode < r; ++mode)
      for (int row = 0; row < d; ++row) {
        const double actual = de_colmajor[static_cast<std::size_t>(mode) + static_cast<std::size_t>(row) * r];
        const double expected = eigenvalues[mode] * e[static_cast<std::size_t>(row) * r + mode];
        const double error = actual - expected;
        residual2 += error * error;
        scale2 += actual * actual;
      }
    const double residual = std::sqrt(residual2 / std::max(scale2, 1.0e-30));
    if (residual > 1.0e-7) throw std::runtime_error("RPMD-JA modes failed full D E-E lambda residual check");
    std::printf("    rpmd_ja full mass-weighted eigen residual: %.3e\n", residual);
  }
  dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_T, r, d, d, e_gpu.data(), r, h_gpu.data(), d, tmp.data(), r);
  dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_T, r, r, d, tmp.data(), r, e_gpu.data(), r, a_gpu.data(), r);
  std::vector<double> a_transpose(static_cast<std::size_t>(r) * r), a(static_cast<std::size_t>(r) * r), mapped;
  a_gpu.copy_to_host(a_transpose.data());
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < r; ++j)
      a[static_cast<std::size_t>(i) * r + j] = a_transpose[static_cast<std::size_t>(j) * r + i];
  rpmd_ja_reference_math::map_site_delta_matrix(r, a, omega, beta_hbar, mapped);
  dm_gpu.copy_from_host(mapped.data());
  dgemm(blas, JA_BLAS_OP_T, JA_BLAS_OP_N, r, d, r, dm_gpu.data(), r, e_gpu.data(), r, btmp.data(), r);
  dgemm(blas, JA_BLAS_OP_T, JA_BLAS_OP_N, d, d, r, e_gpu.data(), r, btmp.data(), r, out.data(), d);
  std::vector<double> delta_t(out.size());
  out.copy_to_host(delta_t.data());
  delta_h.resize(out.size());
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j) {
      const double scale = std::sqrt(masses[i % (d / 3)] * masses[j % (d / 3)]);
      delta_h[static_cast<std::size_t>(i) * d + j] = delta_t[static_cast<std::size_t>(j) * d + i] * scale;
    }
  require_finite(delta_h, "DeltaH");
}

void write_reference(const std::string& path, const RpmdJAReference& reference)
{
  const bool sparse = reference.backend == 1;
  const bool block = reference.backend == 2;
  if (!sparse && !block && reference.backend != 0) throw std::runtime_error("unsupported RPMD-JA reference backend");
  const std::string temporary = path + ".tmp";
  bool created_temporary = false;
  try {
    std::ifstream existing(path, std::ios::binary);
    std::ifstream existing_temporary(temporary, std::ios::binary);
    if (existing.good() || existing_temporary.good())
      throw std::runtime_error("RPMD-JA reference destination or temporary file already exists");
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create RPMD-JA reference file: " + temporary);
    created_temporary = true;
    out.write(kMagic, sizeof(kMagic));
    const std::uint32_t version = block ? kVersionBlock : (sparse ? kVersionSparse : kVersionDense), endian = kEndian;
    write_value(out, version);
    write_value(out, endian);
    write_value(out, reference.number_of_atoms);
    write_value(out, reference.temperature);
    write_value(out, reference.fd_step);
    write_value(out, reference.model_fingerprint);
    out.write(kUnits, sizeof(kUnits));
    const char* layout = block ? kLayoutBlock : (sparse ? kLayoutSparse : kLayoutDense);
    out.write(layout, std::strlen(layout) + 1);
    out.write(reinterpret_cast<const char*>(reference.cell), sizeof(reference.cell));
    out.write(reinterpret_cast<const char*>(reference.pbc), sizeof(reference.pbc));
    write_vector(out, reference.types);
    write_vector(out, reference.masses);
    write_vector(out, reference.positions);
    if (!sparse && !block) {
      for (int a = 0; a < 3; ++a) write_vector(out, reference.delta_h[a]);
    } else if (sparse) {
      const int d = 3 * reference.number_of_atoms;
      if (!reference.delta_h[0].empty() || !reference.delta_h[1].empty() || !reference.delta_h[2].empty())
        throw std::runtime_error("sparse RPMD-JA reference must not contain dense DeltaH");
      const int edge_policy_version = 1;
      write_value(out, edge_policy_version);
      write_value(out, reference.reference_edge_count);
      write_value(out, reference.reference_edge_fingerprint);
      write_sparse_matrix(out, reference.dynamical, d);
      for (int a = 0; a < 3; ++a) write_sparse_matrix(out, reference.site_transpose[a], d);
      write_value(out, reference.spectral_bound);
      write_value(out, reference.kernel_u);
      write_vector(out, std::vector<double>{reference.kernel_error[0], reference.kernel_error[1]});
      write_vector(out, std::vector<double>{reference.kernel_s2[0], reference.kernel_s2[1]});
      write_value(out, reference.kernel_degree);
      write_value(out, reference.p_rank);
      write_value(out, reference.q_rank);
      write_value(out, reference.fd_relative_d);
      write_vector(out, std::vector<double>{reference.fd_relative_b[0], reference.fd_relative_b[1], reference.fd_relative_b[2]});
      write_vector(out, reference.p_values);
      write_vector(out, reference.q_values);
      write_vector(out, reference.p_vectors);
      write_vector(out, reference.q_vectors);
    } else {
      const int policy_size = static_cast<int>(reference.mechanical_policy.size());
      if (!is_supported_qnep_policy(reference.mechanical_policy))
        throw std::runtime_error("unsupported qNEP RPMD-JA mechanical policy");
      write_value(out, reference.mechanical_config_fingerprint);
      write_value(out, reference.q_charge_mode);
      const int uses_pppm = reference.q_uses_pppm ? 1 : 0;
      write_value(out, uses_pppm);
      write_value(out, reference.q_mesh_spacing);
      write_value(out, policy_size);
      out.write(reference.mechanical_policy.data(), policy_size);
      const double diagnostics[] = {
        reference.energy_gradient_relative_error, reference.force_gradient_relative_error,
        reference.hessian_symmetry_relative_error, reference.energy_second_probe_relative_error,
        reference.force_balance_residual, reference.projection_relative_change,
        reference.energy_gradient_absolute_rms, reference.force_gradient_absolute_rms,
        reference.site_derivative_absolute_rms[0], reference.site_derivative_absolute_rms[1], reference.site_derivative_absolute_rms[2],
        reference.site_transport_difference_absolute_rms[0], reference.site_transport_difference_absolute_rms[1], reference.site_transport_difference_absolute_rms[2],
        reference.site_transport_relative_error[0], reference.site_transport_relative_error[1], reference.site_transport_relative_error[2],
        reference.block_relative_residual[0], reference.block_relative_residual[1], reference.block_relative_residual[2], reference.block_relative_residual[3]};
      write_vector(out, std::vector<double>(diagnostics, diagnostics + sizeof(diagnostics) / sizeof(double)));
      write_value(out, reference.spectral_bound);
      write_value(out, reference.kernel_u);
      write_vector(out, std::vector<double>{reference.kernel_error[0], reference.kernel_error[1]});
      write_vector(out, std::vector<double>{reference.kernel_s2[0], reference.kernel_s2[1]});
      write_value(out, reference.kernel_degree);
      write_value(out, reference.p_rank);
      write_value(out, reference.q_rank);
      write_vector(out, reference.p_values);
      write_vector(out, reference.q_values);
      write_vector(out, reference.p_vectors);
      write_vector(out, reference.q_vectors);
      write_block_matrix(out, reference.block_dynamical, 3 * reference.number_of_atoms);
      for (int a = 0; a < 3; ++a) write_block_matrix(out, reference.block_site_transpose[a], 3 * reference.number_of_atoms);
      const int stability_checked = reference.stability_checked ? 1 : 0;
      write_value(out, stability_checked);
    }
    out.flush();
    if (!out) throw std::runtime_error("failed flushing RPMD-JA reference file");
    out.close();
    if (std::rename(temporary.c_str(), path.c_str()) != 0)
      throw std::runtime_error("cannot replace RPMD-JA reference destination");
  } catch (...) {
    if (created_temporary) std::remove(temporary.c_str());
    throw;
  }
}
} // namespace

void load_rpmd_ja_kernel_table(const std::string& path, RpmdJAReference& reference)
{
  load_kernel_table(path, reference);
}

int rpmd_ja_reference_policy_beads(const std::string& policy)
{
  int beads=0, derivative=0;
  return parse_additive_policy(policy, beads, derivative) ? beads : 0;
}

bool rpmd_ja_reference_policy_requires_pimd_fix_com(const std::string& policy)
{
  return policy.rfind("native_reference_transport;internal_mass_com_pullback_v1;", 0) == 0 &&
    rpmd_ja_reference_policy_beads(policy) > 0;
}

std::streampos write_rpmd_ja_qnep_v3_stream_prefix(std::ostream& out, const RpmdJAReference& reference)
{
  if (!is_supported_qnep_policy(reference.mechanical_policy))
    throw std::runtime_error("unsupported qNEP RPMD-JA derivative-source policy");
  out.write(kMagic, sizeof(kMagic));
  const std::uint32_t version = kVersionBlock, endian = kEndian;
  write_value(out, version); write_value(out, endian);
  write_value(out, reference.number_of_atoms); write_value(out, reference.temperature);
  write_value(out, reference.fd_step); write_value(out, reference.model_fingerprint);
  out.write(kUnits, sizeof(kUnits)); out.write(kLayoutBlock, sizeof(kLayoutBlock));
  out.write(reinterpret_cast<const char*>(reference.cell), sizeof(reference.cell));
  out.write(reinterpret_cast<const char*>(reference.pbc), sizeof(reference.pbc));
  write_vector(out, reference.types); write_vector(out, reference.masses); write_vector(out, reference.positions);
  write_value(out, reference.mechanical_config_fingerprint);
  write_value(out, reference.q_charge_mode);
  const int uses_pppm = reference.q_uses_pppm ? 1 : 0;
  write_value(out, uses_pppm); write_value(out, reference.q_mesh_spacing);
  const int policy_size = static_cast<int>(reference.mechanical_policy.size());
  write_value(out, policy_size); out.write(reference.mechanical_policy.data(), policy_size);
  const double diagnostics[] = {
    reference.energy_gradient_relative_error, reference.force_gradient_relative_error,
    reference.hessian_symmetry_relative_error, reference.energy_second_probe_relative_error,
    reference.force_balance_residual, reference.projection_relative_change,
    reference.energy_gradient_absolute_rms, reference.force_gradient_absolute_rms,
    reference.site_derivative_absolute_rms[0], reference.site_derivative_absolute_rms[1], reference.site_derivative_absolute_rms[2],
    reference.site_transport_difference_absolute_rms[0], reference.site_transport_difference_absolute_rms[1], reference.site_transport_difference_absolute_rms[2],
    reference.site_transport_relative_error[0], reference.site_transport_relative_error[1], reference.site_transport_relative_error[2],
    reference.block_relative_residual[0], reference.block_relative_residual[1], reference.block_relative_residual[2], reference.block_relative_residual[3]};
  const std::streampos diagnostics_position = out.tellp();
  out.write(reinterpret_cast<const char*>(diagnostics), sizeof(diagnostics));
  write_value(out, reference.spectral_bound); write_value(out, reference.kernel_u);
  write_vector(out, std::vector<double>{reference.kernel_error[0], reference.kernel_error[1]});
  write_vector(out, std::vector<double>{reference.kernel_s2[0], reference.kernel_s2[1]});
  write_value(out, reference.kernel_degree); write_value(out, reference.p_rank); write_value(out, reference.q_rank);
  write_vector(out, reference.p_values); write_vector(out, reference.q_values);
  write_vector(out, reference.p_vectors); write_vector(out, reference.q_vectors);
  if (!out) throw std::runtime_error("failed writing qNEP RPMD-JA v3 stream prefix");
  return diagnostics_position;
}

std::uint64_t rpmd_ja_model_fingerprint(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open NEP model for RPMD-JA fingerprint: " + path);
  std::uint64_t hash = 14695981039346656037ULL;
  char bytes[8192];
  while (in.read(bytes, sizeof(bytes)) || in.gcount() != 0)
    for (std::streamsize i = 0; i < in.gcount(); ++i)
      hash = (hash ^ static_cast<unsigned char>(bytes[i])) * 1099511628211ULL;
  if (!in.eof()) throw std::runtime_error("failed reading NEP model for RPMD-JA fingerprint");
  return hash;
}

namespace
{
void print_qnep_precheck_table(const double h, const char* label,
  const double gradient, const bool gradient_pass, const double jvp, const bool jvp_pass,
  const double k, const bool k_pass, const double c[3], const bool c_pass[3],
  const bool v_finite, const bool overall_failed, const bool requested_h)
{
  const auto status = [](const bool pass) { return pass ? "PASS" : "FAIL"; };
  std::printf("    %s precheck h=%.6g A%s\n", label, h, requested_h ? " (REQUESTED fd_step)" : "");
  std::printf("    %-25s %12s %-12s %12s %-6s\n", "Check", "Value", "Units", "Max/Required", "Result");
  std::printf("    %-25s %12.4g %-12s %12.4g %-6s\n", "Gradient/native force RMS", gradient, "eV/A", kForceTolerance, status(gradient_pass));
  std::printf("    %-25s %12.4g %-12s %12.4g %-6s\n", "Site JVP/gradient RMS", jvp, "eV/A", kForceTolerance, status(jvp_pass));
  std::printf("    %-25s %11.4g%% %-12s %11.4g%% %-6s\n", "K stencil", 100.0 * k, "relative", 100.0 * kDifferenceTolerance, status(k_pass));
  for (int alpha = 0; alpha < 3; ++alpha)
    std::printf("    %-25s %11.4g%% %-12s %11.4g%% %-6s\n", alpha == 0 ? "Cx stencil" : alpha == 1 ? "Cy stencil" : "Cz stencil",
      100.0 * c[alpha], "relative", 100.0 * kDifferenceTolerance,
      status(c_pass[alpha]));
  std::printf("    %-25s %12s %-12s %12s %-6s\n", "V finite", v_finite ? "yes" : "no", "-", "finite", status(v_finite));
  std::printf("    Overall: %s\n", overall_failed ? "FAIL" : "PASS");
}

std::uint64_t qnep_config_fingerprint(Force& force, NEP_Charge& qnep, const double mesh_spacing)
{
  if (force.get_run_input().contains("dftd3"))
    throw std::runtime_error("rpmd_ja qNEP reference does not support dftd3 corrections");
  if (!qnep.uses_pppm()) throw std::runtime_error("rpmd_ja qNEP reference currently supports kspace_method pppm only");
  std::uint64_t hash = rpmd_ja_model_fingerprint(force.primary_nep_model_path());
  const auto add_bytes = [&hash](const void* data, const std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ULL;
  };
  const int charge_mode = qnep.get_charge_mode();
  const int uses_pppm = qnep.uses_pppm() ? 1 : 0;
  const float ewald_alpha = qnep.get_ewald_alpha();
  const float realspace_cutoff = qnep.get_realspace_cutoff();
  add_bytes(&charge_mode, sizeof(charge_mode)); add_bytes(&uses_pppm, sizeof(uses_pppm));
  add_bytes(&mesh_spacing, sizeof(mesh_spacing)); add_bytes(&ewald_alpha, sizeof(ewald_alpha));
  add_bytes(&realspace_cutoff, sizeof(realspace_cutoff));
  constexpr char policy[] = "native_reference_transport";
  add_bytes(policy, sizeof(policy));
  return hash;
}
}

std::uint64_t rpmd_ja_qnep_config_fingerprint(Force& force)
{
  if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
    throw std::runtime_error("rpmd_ja qNEP fingerprint requires exactly one configured qNEP model");
  auto* qnep = dynamic_cast<NEP_Charge*>(&force.get_potential(0));
  if (qnep == nullptr)
    throw std::runtime_error("rpmd_ja qNEP fingerprint does not support non-qNEP or mixed potentials");
  return qnep_config_fingerprint(force, *qnep, qnep->get_pppm_mesh_spacing());
}

RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom& atom, Box& active_box, Force& force)
{
  return [&atom, &active_box, &force](const RpmdJAReference& raw, const std::vector<RpmdJADiagnosticMode>& modes) {
    const auto started = std::chrono::steady_clock::now();
    const int n = raw.number_of_atoms, d = 3 * n;
    if (atom.number_of_atoms != n || atom.cpu_type != raw.types || atom.cpu_mass.size() != raw.masses.size())
      throw std::runtime_error("INPUT_MISMATCH: current atom count, types, or mass count differs from raw reference");
    for (std::size_t i = 0; i < raw.masses.size(); ++i)
      if (atom.cpu_mass[i] != raw.masses[i]) throw std::runtime_error("INPUT_MISMATCH: current atom mass differs from raw reference");
    if (active_box.pbc_x != raw.pbc[0] || active_box.pbc_y != raw.pbc[1] || active_box.pbc_z != raw.pbc[2])
      throw std::runtime_error("INPUT_MISMATCH: current periodic boundary flags differ from raw reference");
    Box observer_box = active_box;
    for (int i = 0; i < 9; ++i) {
      if (!std::isfinite(active_box.cpu_h[i]) || std::abs(active_box.cpu_h[i] - raw.cell[i]) > 1.0e-12 * std::max(1.0, std::abs(raw.cell[i])))
        throw std::runtime_error("INPUT_MISMATCH: current cell differs from raw reference");
      observer_box.cpu_h[i] = raw.cell[i];
    }
    observer_box.get_inverse();
    observer_box.set_is_orthogonal();
    if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
      throw std::runtime_error("INPUT_MISMATCH: prepare requires the same single qNEP model used for raw generation");
    auto* active_qnep = dynamic_cast<NEP_Charge*>(&force.get_potential(0));
    if (!active_qnep || (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) || !active_qnep->uses_pppm())
      throw std::runtime_error("INPUT_MISMATCH: prepare requires qNEP charge mode 1 or 2 with PPPM");
    if (rpmd_ja_model_fingerprint(force.primary_nep_model_path()) != raw.model_fingerprint ||
        qnep_config_fingerprint(force, *active_qnep, raw.q_mesh_spacing) != raw.mechanical_config_fingerprint)
      throw std::runtime_error("INPUT_MISMATCH: model or qNEP configuration fingerprint differs from raw reference");
    QEvaluator evaluator(force.primary_nep_model_path(), n, force.get_run_input(), observer_box, raw.types, raw.q_mesh_spacing);
    const auto energy = [](const QEvaluation& e) {
      double sum = 0.0, correction = 0.0;
      for (double value : e.energy) { const double y = value - correction, t = sum + y; correction = (t - sum) - y; sum = t; }
      return sum;
    };
    const auto rms = [](const std::vector<double>& a, const std::vector<double>& b) {
      double sum = 0.0;
      for (std::size_t i = 0; i < a.size(); ++i) { const double x = a[i] - b[i]; sum += x*x; }
      return std::sqrt(sum / a.size());
    };
    const auto gradient_force_rms = [](const std::vector<double>& gradient, const std::vector<double>& force) {
      double sum = 0.0;
      for (std::size_t i = 0; i < gradient.size(); ++i) { const double x = gradient[i] + force[i]; sum += x*x; }
      return std::sqrt(sum / gradient.size());
    };
    std::uint64_t force_evaluations = 0, gradient_calls = 0, jvp_calls = 0;
    std::array<QEvaluation, 3> full0, short0;
    std::array<std::vector<double>, 3> gradient0;
    for (int i = 0; i < 3; ++i) {
      full0[i] = evaluator.evaluate(raw.positions, true); ++force_evaluations;
      gradient0[i] = evaluator.analytic_gradient(); ++gradient_calls;
      short0[i] = evaluator.evaluate(raw.positions, false); ++force_evaluations;
    }
    double full_min = energy(full0[0]), full_max = full_min, short_min = energy(short0[0]), short_max = short_min;
    for (int i = 1; i < 3; ++i) { full_min=std::min(full_min,energy(full0[i])); full_max=std::max(full_max,energy(full0[i])); short_min=std::min(short_min,energy(short0[i])); short_max=std::max(short_max,energy(short0[i])); }
    const double full_force_repeat = std::sqrt((rms(full0[0].force,full0[2].force)*rms(full0[0].force,full0[2].force)+rms(full0[1].force,full0[2].force)*rms(full0[1].force,full0[2].force))/2.0);
    const double short_force_repeat = std::sqrt((rms(short0[0].force,short0[2].force)*rms(short0[0].force,short0[2].force)+rms(short0[1].force,short0[2].force)*rms(short0[1].force,short0[2].force))/2.0);
    const double gradient_repeat = std::sqrt((rms(gradient0[0],gradient0[2])*rms(gradient0[0],gradient0[2])+rms(gradient0[1],gradient0[2])*rms(gradient0[1],gradient0[2]))/2.0);
    double ng2 = 0.0, nf2 = 0.0;
    for (int i=0;i<d;++i) { const double e=gradient0[2][i]+full0[2].force[i]; ng2+=e*e; nf2+=full0[2].force[i]*full0[2].force[i]; }
    std::ostringstream out;
    out << std::scientific << std::setprecision(8)
      << "\nmode_validation_status: complete\nR0_full_total_energy_span_eV: " << full_max-full_min
      << "\nR0_short_total_energy_span_eV: " << short_max-short_min
      << "\nR0_full_force_repeat_RMS_eV_per_A: " << full_force_repeat
      << "\nR0_short_force_repeat_RMS_eV_per_A: " << short_force_repeat
      << "\nR0_analytic_gradient_repeat_RMS_eV_per_A: " << gradient_repeat
      << "\nR0_native_force_vs_analytic_gradient_RMS_eV_per_A: " << std::sqrt(ng2/d)
      << "\nR0_native_force_vs_analytic_gradient_relative_RMS: " << std::sqrt(ng2/std::max(nf2,1.0e-300)) << '\n';
    std::vector<double> base_positions = raw.positions;
    bool any_direct_supported=false, any_energy_force_inconsistent=false, any_matrix_mismatch=false, any_grid_sensitive=false, any_jvp_mismatch=false;
    for (std::size_t mi=0; mi<modes.size(); ++mi) {
      const auto& mode=modes[mi];
      if (!(mode.eigenvalue < 0.0) || mode.mass_weighted_direction.size()!=static_cast<std::size_t>(d)) continue;
      std::vector<double> p(d); double maxp=0.0;
      for(int i=0;i<d;++i) { p[i]=mode.mass_weighted_direction[i]/std::sqrt(raw.masses[i%n]); maxp=std::max(maxp,std::abs(p[i])); }
      if (!(maxp>0.0)) continue;
      double mnorm=0.0; for(int i=0;i<d;++i) { p[i]/=maxp; mnorm+=raw.masses[i%n]*p[i]*p[i]; }
      std::vector<double> steps{raw.fd_step*0.5,raw.fd_step,raw.fd_step*2.0,0.005,0.01,0.02,0.05};
      std::sort(steps.begin(),steps.end()); steps.erase(std::unique(steps.begin(),steps.end()),steps.end());
      std::vector<double> lg,lf,le,ls,le_short,hvp_relative;
      out << "mode_validation_"<<mi+1<<"_lambda_eV_per_A2_amu: "<<mode.eigenvalue
          <<"\nmode_validation_"<<mi+1<<"_residual: "<<mode.residual<<"\nmode_validation_"<<mi+1<<"_Mnorm: "<<mnorm<<'\n';
      const QEvaluation e0=evaluator.evaluate(raw.positions,true); ++force_evaluations;
      const double E0=energy(e0);
      const std::vector<double> g0=evaluator.analytic_gradient(); ++gradient_calls;
      double force_projection=0.0; for(int i=0;i<d;++i) force_projection+=g0[i]*p[i];
      out<<"mode_validation_"<<mi+1<<"_reference_gradient_projection_eV_per_A: "<<force_projection<<'\n';
      double jvp_normalized_error=std::numeric_limits<double>::quiet_NaN();
      {
        const auto site=evaluator.analytic_site_jvp(p); ++jvp_calls;
        double site_sum=std::accumulate(site.begin(),site.end(),0.0), projection=0.0;
        for(int i=0;i<d;++i)projection+=g0[i]*p[i];
        jvp_normalized_error=std::abs(site_sum-projection)/std::sqrt(mnorm);
        out<<"mode_validation_"<<mi+1<<"_R0_sum_site_JVP_eV_per_A: "<<site_sum
           <<"\nmode_validation_"<<mi+1<<"_R0_gradient_dot_p_eV_per_A: "<<projection
           <<"\nmode_validation_"<<mi+1<<"_R0_JVP_identity_error: "<<site_sum-projection
           <<"\nmode_validation_"<<mi+1<<"_R0_JVP_normalized_error: "<<jvp_normalized_error<<'\n';
      }
      for(double h:steps) {
        std::vector<double> plus=base_positions,minus=base_positions;
        for(int i=0;i<d;++i){plus[i]+=h*p[i];minus[i]-=h*p[i];}
        const auto fp=evaluator.evaluate(plus,true); ++force_evaluations; const auto gp=evaluator.analytic_gradient(); ++gradient_calls;
        const auto fm=evaluator.evaluate(minus,true); ++force_evaluations; const auto gm=evaluator.analytic_gradient(); ++gradient_calls;
        const auto sp=evaluator.evaluate(plus,false); ++force_evaluations;
        const auto sm=evaluator.evaluate(minus,false); ++force_evaluations;
        double gradient_curv=0.0,native_curv=0.0,energy_curv=(energy(fp)+energy(fm)-2.0*E0)/(h*h*mnorm),short_force_curv=0.0,short_energy_curv=(energy(sp)+energy(sm)-2.0*energy(short0[2]))/(h*h*mnorm);
        double hvp2=0.0;
        for(int i=0;i<d;++i){gradient_curv+=p[i]*(gp[i]-gm[i])/(2.0*h)/(mnorm);native_curv-=p[i]*(fp.force[i]-fm.force[i])/(2.0*h)/(mnorm);short_force_curv-=p[i]*(sp.force[i]-sm.force[i])/(2.0*h)/(mnorm);const double y=(gp[i]-gm[i])/(2.0*h),err=(y-mode.eigenvalue*raw.masses[i%n]*p[i])/std::sqrt(raw.masses[i%n]);hvp2+=err*err;}
        double mismatch=std::sqrt(hvp2)/std::sqrt(mnorm);
        lg.push_back(gradient_curv);lf.push_back(native_curv);le.push_back(energy_curv);ls.push_back(short_force_curv);
        le_short.push_back(short_energy_curv);hvp_relative.push_back(mismatch/std::max(std::abs(mode.eigenvalue),1.0e-300));
        out<<"mode_"<<mi+1<<"_h_A: "<<h<<" gradient_lambda: "<<gradient_curv<<" native_force_lambda: "<<native_curv
           <<" energy_lambda: "<<energy_curv<<" short_force_lambda: "<<short_force_curv<<" short_energy_lambda: "<<short_energy_curv
           <<" electro_related_gradient_minus_short: "<<gradient_curv-short_force_curv
           <<" HVP_massweighted_error: "<<mismatch<<" HVP_relative_to_abs_lambda: "<<mismatch/std::max(std::abs(mode.eigenvalue),1.0e-300)<<'\n';
      }
      const auto spread=[](const std::vector<double>& values){const auto mm=std::minmax_element(values.begin(),values.end());return (std::abs(*mm.second-*mm.first))/std::max(std::abs(std::accumulate(values.begin(),values.end(),0.0)/values.size()),1.0e-300);};
      out<<"mode_"<<mi+1<<"_gradient_step_spread: "<<spread(lg)<<"\nmode_"<<mi+1<<"_native_force_step_spread: "<<spread(lf)<<"\nmode_"<<mi+1<<"_energy_step_spread: "<<spread(le)<<'\n';
      const auto fd_h=std::find(steps.begin(),steps.end(),raw.fd_step);
      if(fd_h!=steps.end()&&fd_h+1!=steps.end()&&*(fd_h+1)==2.0*raw.fd_step){
        const std::size_t index=static_cast<std::size_t>(fd_h-steps.begin());
        out<<"mode_"<<mi+1<<"_gradient_curvature_richardson_h_2h_diagnostic_eV_per_A2_amu: "
          <<(4.0*lg[index]-lg[index+1])/3.0<<'\n';
      }
      const auto evidence=rpmd_ja_reference_math::classify_curvature(mode.eigenvalue,mode.residual,
        jvp_normalized_error,full_max-full_min,short_max-short_min,mnorm,steps,lg,lf,le,hvp_relative,ls,le_short);
      const bool direct_supported=evidence.direct_negative_supported;
      const bool energy_force_inconsistent=evidence.energy_force_inconsistent;
      const bool matrix_mismatch=evidence.matrix_derivative_mismatch;
      bool grid_sensitive=false;
      if(evidence.selected_pair>=0)out<<"mode_"<<mi+1<<"_selected_reliable_h_interval_A: "<<steps[evidence.selected_pair]<<' '<<steps[evidence.selected_pair+1]<<'\n';
      out<<"mode_"<<mi+1<<"_DIRECT_NEGATIVE_CURVATURE_SUPPORTED: "<<(direct_supported?"yes":"no")
        <<"\nmode_"<<mi+1<<"_POTENTIAL_ENERGY_FORCE_INCONSISTENCY: "<<(energy_force_inconsistent?"yes":"no")
        <<"\nmode_"<<mi+1<<"_MATRIX_DERIVATIVE_MISMATCH: "<<(matrix_mismatch?"yes":"no")
        <<"\nmode_"<<mi+1<<"_SITE_JVP_GRADIENT_MISMATCH: "<<(evidence.site_jvp_gradient_mismatch?"yes":"no")
        <<"\nmode_"<<mi+1<<"_SHORT_RANGE_FORCE_ENERGY_CONSISTENT: "<<(evidence.short_range_consistent?"yes":"no")
        <<"\nmode_"<<mi+1<<"_NUMERICAL_OR_NONLINEAR_UNRESOLVED: "<<(evidence.unresolved?"yes":"no")<<'\n';
      any_direct_supported=any_direct_supported||direct_supported;
      any_energy_force_inconsistent=any_energy_force_inconsistent||energy_force_inconsistent;
      any_matrix_mismatch=any_matrix_mismatch||matrix_mismatch;
      any_jvp_mismatch=any_jvp_mismatch||evidence.site_jvp_gradient_mismatch;
      if(mi==0) {
        const double coarse_spacing=raw.q_mesh_spacing;
        Box fine_box=observer_box;
        QEvaluator fine_evaluator(force.primary_nep_model_path(),n,force.get_run_input(),fine_box,raw.types,coarse_spacing*0.5);
        const auto fine0a=fine_evaluator.evaluate(raw.positions,true); ++force_evaluations;
        const auto fineg0a=fine_evaluator.analytic_gradient(); ++gradient_calls;
        const auto fine0b=fine_evaluator.evaluate(raw.positions,true); ++force_evaluations;
        const auto fineg0b=fine_evaluator.analytic_gradient(); ++gradient_calls;
        const auto fine0c=fine_evaluator.evaluate(raw.positions,true); ++force_evaluations;
        const auto fineg0c=fine_evaluator.analytic_gradient(); ++gradient_calls;
        double fspan=std::max({energy(fine0a),energy(fine0b),energy(fine0c)})-std::min({energy(fine0a),energy(fine0b),energy(fine0c)});
        const double fine_r0_grad_delta=rms(fineg0c,g0)/std::max(std::sqrt(std::inner_product(g0.begin(),g0.end(),g0.begin(),0.0)/d),1.0e-300);
        const double fine_r0_force_delta=rms(fine0c.force,e0.force)/std::max(std::sqrt(std::inner_product(e0.force.begin(),e0.force.end(),e0.force.begin(),0.0)/d),1.0e-300);
        if(fine_r0_grad_delta>0.05||fine_r0_force_delta>0.05)grid_sensitive=true;
        out<<"fine_mesh_spacing_A: "<<coarse_spacing*0.5<<"\nfine_mesh_R0_repeat_energy_span_eV: "<<fspan
           <<"\nfine_mesh_R0_gradient_repeat_RMS_eV_per_A: "<<std::sqrt((rms(fineg0a,fineg0c)*rms(fineg0a,fineg0c)+rms(fineg0b,fineg0c)*rms(fineg0b,fineg0c))/2.0)
           <<"\nfine_mesh_R0_native_force_gradient_RMS_eV_per_A: "<<gradient_force_rms(fineg0c,fine0c.force)
           <<"\nfine_mesh_R0_gradient_relative_delta: "<<fine_r0_grad_delta
           <<"\nfine_mesh_R0_native_force_relative_delta: "<<fine_r0_force_delta<<'\n';
        for(double h:{0.005,0.01,0.02}) {
          std::vector<double> plus=base_positions,minus=base_positions;for(int i=0;i<d;++i){plus[i]+=h*p[i];minus[i]-=h*p[i];}
          const auto fp=fine_evaluator.evaluate(plus,true);++force_evaluations;const auto gp=fine_evaluator.analytic_gradient();++gradient_calls;
          const auto fm=fine_evaluator.evaluate(minus,true);++force_evaluations;const auto gm=fine_evaluator.analytic_gradient();++gradient_calls;
          double curv=0.0,native_curv=0.0;for(int i=0;i<d;++i){curv+=p[i]*(gp[i]-gm[i])/(2*h*mnorm);native_curv-=p[i]*(fp.force[i]-fm.force[i])/(2*h*mnorm);}
          const double ecurv=(energy(fp)+energy(fm)-2*energy(fine0c))/(h*h*mnorm);
          const auto coarse=std::find(steps.begin(),steps.end(),h);
          const std::size_t cj=static_cast<std::size_t>(coarse-steps.begin());
          const double gradient_delta=(curv-lg[cj])/std::max({std::abs(curv),std::abs(lg[cj]),1.0e-300});
          const double force_delta=(native_curv-lf[cj])/std::max({std::abs(native_curv),std::abs(lf[cj]),1.0e-300});
          if(std::abs(gradient_delta)>0.05||std::abs(force_delta)>0.05)grid_sensitive=true;
          out<<"fine_mesh_h_A: "<<h<<" gradient_lambda: "<<curv<<" native_force_lambda: "<<native_curv<<" energy_lambda: "<<ecurv
             <<" coarse_grid_gradient_relative_delta: "<<gradient_delta<<" coarse_grid_native_force_relative_delta: "<<force_delta<<'\n';
        }
        out<<"mode_"<<mi+1<<"_GRID_SENSITIVE: "<<(grid_sensitive?"yes":"no")<<'\n';
        any_grid_sensitive=any_grid_sensitive||grid_sensitive;
      }
    }
    out << "mode_validation_force_evaluations: " << force_evaluations << "\nmode_validation_gradient_calls: " << gradient_calls
        << "\nmode_validation_site_jvp_calls: " << jvp_calls
        << "\nmode_validation_elapsed_seconds: " << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
        << "\nmode_validation_peak_private_GPU_workspace: unknown (NEP/PPPM private allocation excluded)\n";
    out << "mode_validation_flags: ";
    if(any_direct_supported) out << "DIRECT_NEGATIVE_CURVATURE_SUPPORTED ";
    if(any_matrix_mismatch) out << "MATRIX_DERIVATIVE_MISMATCH ";
    if(any_energy_force_inconsistent) out << "POTENTIAL_ENERGY_FORCE_INCONSISTENCY ";
    if(any_grid_sensitive) out << "GRID_SENSITIVE ";
    if(any_jvp_mismatch) out << "SITE_JVP_GRADIENT_MISMATCH ";
    if(!any_direct_supported&&!any_matrix_mismatch&&!any_energy_force_inconsistent) out << "NUMERICAL_OR_NONLINEAR_UNRESOLVED ";
    out << "\nConclusion: ";
    if(any_direct_supported) out << "the evaluated reference potential has supported negative local curvature along at least one reported mode";
    else if(any_matrix_mismatch) out << "direct curvature evidence disagrees with the saved matrix mode";
    else if(any_energy_force_inconsistent) out << "energy and force curvature evidence remains inconsistent";
    else out << "the available finite-difference evidence does not resolve the curvature cause";
    out << ". This does not classify the sampled structure as wrong or establish material-wide instability.\n";
    return out.str();
  };
}

RpmdJAReference read_rpmd_ja_reference(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open RPMD-JA reference file: " + path);
  char magic[sizeof(kMagic)], units[sizeof(kUnits)];
  in.read(magic, sizeof(magic));
  std::uint32_t version, endian;
  RpmdJAReference result;
  read_value(in, version);
  read_value(in, endian);
  read_value(in, result.number_of_atoms);
  read_value(in, result.temperature);
  read_value(in, result.fd_step);
  read_value(in, result.model_fingerprint);
  in.read(units, sizeof(units));
  const char* expected_layout = version == kVersionBlock ? kLayoutBlock :
    (version == kVersionSparse ? kLayoutSparse : kLayoutDense);
  std::vector<char> layout(std::strlen(expected_layout) + 1);
  in.read(layout.data(), static_cast<std::streamsize>(layout.size()));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
      (version != kVersionDense && version != kVersionSparse && version != kVersionBlock) || endian != kEndian ||
      std::memcmp(units, kUnits, sizeof(kUnits)) != 0 || std::memcmp(layout.data(), expected_layout, layout.size()) != 0)
    throw std::runtime_error("unsupported RPMD-JA reference magic, version, endian, units, or layout");
  result.backend = version == kVersionBlock ? 2 : (version == kVersionSparse ? 1 : 0);
  if (!(result.temperature > 0.0) || !std::isfinite(result.temperature) || !(result.fd_step > 0.0) ||
      !std::isfinite(result.fd_step))
    throw std::runtime_error("invalid RPMD-JA reference temperature or finite-difference step");
  const int dimension = reference_dimension(result.number_of_atoms);
  const std::streampos data_position = in.tellg();
  in.seekg(0, std::ios::end);
  const std::streamoff file_size = in.tellg();
  in.seekg(data_position);
  if (file_size < 0 || data_position < 0) throw std::runtime_error("invalid RPMD-JA reference file size");
  const std::size_t n = static_cast<std::size_t>(result.number_of_atoms);
  if (result.backend == 0) {
    const std::size_t matrix = matrix_size(result.number_of_atoms);
    const std::uint64_t remaining = 9 * sizeof(double) + 3 * sizeof(int) + n * sizeof(int) +
      4 * n * sizeof(double) + 3 * matrix * sizeof(double);
    if (static_cast<std::uint64_t>(file_size - static_cast<std::streamoff>(data_position)) != remaining)
      throw std::runtime_error("RPMD-JA dense v1 file has an invalid length");
  }
  require_remaining(in, file_size, 9 * sizeof(double) + 3 * sizeof(int));
  in.read(reinterpret_cast<char*>(result.cell), sizeof(result.cell));
  in.read(reinterpret_cast<char*>(result.pbc), sizeof(result.pbc));
  if (!in) throw std::runtime_error("truncated RPMD-JA reference cell or PBC flags");
  read_vector_checked(in, result.types, n, file_size);
  read_vector_checked(in, result.masses, n, file_size);
  read_vector_checked(in, result.positions, static_cast<std::uint64_t>(dimension), file_size);
  if (result.backend == 0) {
    const std::size_t matrix = matrix_size(result.number_of_atoms);
    for (int a = 0; a < 3; ++a)
      read_vector_checked(in, result.delta_h[a], matrix, file_size);
  } else if (result.backend == 1) {
    int edge_policy_version = 0;
    read_value_checked(in, edge_policy_version, file_size);
    read_value_checked(in, result.reference_edge_count, file_size);
    read_value_checked(in, result.reference_edge_fingerprint, file_size);
    if (edge_policy_version != 1 || result.edge_policy != "fixed_d0_raw_nep_image_keys_v1")
      throw std::runtime_error("unsupported RPMD-JA fixed-edge policy");
    read_sparse_matrix(in, result.dynamical, dimension, file_size);
    for (int a = 0; a < 3; ++a) read_sparse_matrix(in, result.site_transpose[a], dimension, file_size);
    read_value_checked(in, result.spectral_bound, file_size);
    read_value_checked(in, result.kernel_u, file_size);
    std::vector<double> values;
    read_vector_checked(in, values, 2, file_size);
    std::copy(values.begin(), values.end(), result.kernel_error);
    values.clear();
    read_vector_checked(in, values, 2, file_size);
    std::copy(values.begin(), values.end(), result.kernel_s2);
    read_value_checked(in, result.kernel_degree, file_size);
    read_value_checked(in, result.p_rank, file_size);
    read_value_checked(in, result.q_rank, file_size);
    read_value_checked(in, result.fd_relative_d, file_size);
    values.clear();
    read_vector_checked(in, values, 3, file_size);
    std::copy(values.begin(), values.end(), result.fd_relative_b);
    if (result.kernel_degree < 0 || result.kernel_degree > 512 || result.p_rank <= 0 || result.q_rank <= 0 ||
        result.p_rank > result.kernel_degree + 1 || result.q_rank > result.kernel_degree + 1)
      throw std::runtime_error("invalid RPMD-JA sparse kernel ranks or degree");
    const std::uint64_t p_count = static_cast<std::uint64_t>(result.kernel_degree + 1) * result.p_rank;
    const std::uint64_t q_count = static_cast<std::uint64_t>(result.kernel_degree + 1) * result.q_rank;
    read_vector_checked(in, result.p_values, static_cast<std::uint64_t>(result.p_rank), file_size);
    read_vector_checked(in, result.q_values, static_cast<std::uint64_t>(result.q_rank), file_size);
    read_vector_checked(in, result.p_vectors, p_count, file_size);
    read_vector_checked(in, result.q_vectors, q_count, file_size);
    if (!(result.spectral_bound > 0.0) || !std::isfinite(result.spectral_bound) ||
        !(result.kernel_u > 0.0) || !std::isfinite(result.kernel_u))
      throw std::runtime_error("invalid RPMD-JA sparse spectral or kernel bound");
    for (int i = 0; i < 2; ++i)
      if (!(result.kernel_error[i] >= 0.0) || !std::isfinite(result.kernel_error[i]) ||
          !(result.kernel_s2[i] >= 0.0) || !std::isfinite(result.kernel_s2[i]))
        throw std::runtime_error("invalid RPMD-JA sparse kernel budget");
    if (!std::isfinite(result.fd_relative_d) ||
        !std::all_of(result.fd_relative_b, result.fd_relative_b + 3, [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("invalid RPMD-JA sparse finite-difference diagnostic");
  } else {
    read_value_checked(in, result.mechanical_config_fingerprint, file_size);
    read_value_checked(in, result.q_charge_mode, file_size);
    int uses_pppm = 0, policy_size = 0;
    read_value_checked(in, uses_pppm, file_size);
    result.q_uses_pppm = uses_pppm == 1;
    read_value_checked(in, result.q_mesh_spacing, file_size);
    read_value_checked(in, policy_size, file_size);
    if (policy_size <= 0 || policy_size > 128) throw std::runtime_error("invalid qNEP RPMD-JA mechanical policy length");
    require_remaining(in, file_size, static_cast<std::size_t>(policy_size));
    result.mechanical_policy.resize(static_cast<std::size_t>(policy_size));
    in.read(&result.mechanical_policy[0], policy_size);
    std::vector<double> diagnostics;
    read_vector_checked(in, diagnostics, 21, file_size);
    result.energy_gradient_relative_error = diagnostics[0];
    result.force_gradient_relative_error = diagnostics[1];
    result.hessian_symmetry_relative_error = diagnostics[2];
    result.energy_second_probe_relative_error = diagnostics[3];
    result.force_balance_residual = diagnostics[4];
    result.projection_relative_change = diagnostics[5];
    result.energy_gradient_absolute_rms = diagnostics[6];
    result.force_gradient_absolute_rms = diagnostics[7];
    std::copy(diagnostics.begin() + 8, diagnostics.begin() + 11, result.site_derivative_absolute_rms);
    std::copy(diagnostics.begin() + 11, diagnostics.begin() + 14, result.site_transport_difference_absolute_rms);
    std::copy(diagnostics.begin() + 14, diagnostics.begin() + 17, result.site_transport_relative_error);
    std::copy(diagnostics.begin() + 17, diagnostics.end(), result.block_relative_residual);
    read_value_checked(in, result.spectral_bound, file_size);
    read_value_checked(in, result.kernel_u, file_size);
    std::vector<double> values;
    read_vector_checked(in, values, 2, file_size);
    std::copy(values.begin(), values.end(), result.kernel_error);
    values.clear();
    read_vector_checked(in, values, 2, file_size);
    std::copy(values.begin(), values.end(), result.kernel_s2);
    read_value_checked(in, result.kernel_degree, file_size);
    read_value_checked(in, result.p_rank, file_size);
    read_value_checked(in, result.q_rank, file_size);
    if (result.kernel_degree < 0 || result.kernel_degree > 512 || result.p_rank <= 0 || result.q_rank <= 0 ||
        result.p_rank > result.kernel_degree + 1 || result.q_rank > result.kernel_degree + 1)
      throw std::runtime_error("invalid qNEP RPMD-JA kernel ranks or degree");
    const std::uint64_t p_count = static_cast<std::uint64_t>(result.kernel_degree + 1) * result.p_rank;
    const std::uint64_t q_count = static_cast<std::uint64_t>(result.kernel_degree + 1) * result.q_rank;
    read_vector_checked(in, result.p_values, result.p_rank, file_size);
    read_vector_checked(in, result.q_values, result.q_rank, file_size);
    read_vector_checked(in, result.p_vectors, p_count, file_size);
    read_vector_checked(in, result.q_vectors, q_count, file_size);
    read_block_matrix(in, result.block_dynamical, dimension, file_size);
    for (int alpha = 0; alpha < 3; ++alpha)
      read_block_matrix(in, result.block_site_transpose[alpha], dimension, file_size);
    int stability_checked = 0;
    read_value_checked(in, stability_checked, file_size);
    if (stability_checked != 1 || result.q_charge_mode < 1 || result.q_charge_mode > 2 || !result.q_uses_pppm ||
        !(result.q_mesh_spacing > 0.0) || !std::isfinite(result.q_mesh_spacing) ||
        !is_supported_qnep_policy(result.mechanical_policy) || result.mechanical_config_fingerprint == 0 ||
        !(result.spectral_bound > 0.0) || !std::isfinite(result.spectral_bound) ||
        !(result.kernel_u > 0.0) || !std::isfinite(result.kernel_u))
      throw std::runtime_error("invalid or unchecked qNEP RPMD-JA v3 metadata");
    const auto valid_nonnegative = [](const double value) { return value >= 0.0 && std::isfinite(value); };
    for (double value : diagnostics) if (!valid_nonnegative(value))
      throw std::runtime_error("non-finite qNEP RPMD-JA v3 diagnostics");
    for (double value : result.kernel_error) if (!valid_nonnegative(value))
      throw std::runtime_error("invalid qNEP RPMD-JA kernel error budget");
    for (double value : result.kernel_s2) if (!valid_nonnegative(value))
      throw std::runtime_error("invalid qNEP RPMD-JA kernel S2 budget");
    const double tau = HBAR / (K_B * result.temperature);
    if (tau * std::sqrt(result.spectral_bound) > result.kernel_u *
        (1.0 + 32.0 * std::numeric_limits<double>::epsilon()))
      throw std::runtime_error("qNEP RPMD-JA kernel table does not cover the stored spectrum");
  }
  if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing data in RPMD-JA reference file");
  require_finite(result.masses, "masses");
  require_finite(result.positions, "reference positions");
  for (double mass : result.masses) if (!(mass > 0.0)) throw std::runtime_error("nonpositive RPMD-JA mass");
  for (double value : result.cell) if (!std::isfinite(value)) throw std::runtime_error("non-finite RPMD-JA cell");
  for (int pbc : result.pbc) if (pbc != 0 && pbc != 1) throw std::runtime_error("invalid RPMD-JA PBC flag");
  if (result.backend == 0) {
    for (int a = 0; a < 3; ++a) require_finite(result.delta_h[a], "DeltaH");
  } else if (result.backend == 1) {
    validate_sparse_matrix(result.dynamical, dimension);
    for (int a = 0; a < 3; ++a) validate_sparse_matrix(result.site_transpose[a], dimension);
    const double actual_bound = csr_max_abs_row_sum(result.dynamical);
    if (actual_bound > result.spectral_bound * (1.0 + 32.0 * std::numeric_limits<double>::epsilon()))
      throw std::runtime_error("RPMD-JA sparse stored spectral bound is smaller than its CSR row-sum bound");
    const double tau = HBAR / (K_B * result.temperature);
    if (tau * std::sqrt(actual_bound) > result.kernel_u * (1.0 + 32.0 * std::numeric_limits<double>::epsilon()))
      throw std::runtime_error("RPMD-JA sparse kernel table does not cover the stored spectrum");
    require_finite(result.p_values, "P eigenvalues");
    require_finite(result.q_values, "Q eigenvalues");
    require_finite(result.p_vectors, "P eigenvectors");
    require_finite(result.q_vectors, "Q eigenvectors");
    const std::string stability_path = path + ".stability";
    std::ifstream stability(stability_path);
    std::string tag;
    int stability_version = 0, atom_count = 0;
    std::uint64_t fingerprint = 0;
    double minimum_pivot = 0.0, reconstruction_residual = 0.0;
    stability >> tag >> stability_version;
    if (tag != "GPUMDJA_STABILITY" || stability_version != 1)
      throw std::runtime_error("missing or unsupported RPMD-JA stability sidecar: " + stability_path);
    stability >> tag >> std::hex >> fingerprint >> std::dec;
    if (tag != "fingerprint") throw std::runtime_error("invalid RPMD-JA stability fingerprint field");
    stability >> tag >> atom_count;
    if (tag != "atoms") throw std::runtime_error("invalid RPMD-JA stability atom-count field");
    stability >> tag >> minimum_pivot;
    if (tag != "minimum_pivot") throw std::runtime_error("invalid RPMD-JA stability pivot field");
    stability >> tag >> reconstruction_residual;
    if (tag != "reconstruction_residual") throw std::runtime_error("invalid RPMD-JA stability residual field");
    std::string trailing;
    if (!stability || stability >> trailing || atom_count != result.number_of_atoms || !(minimum_pivot > 0.0) ||
        !std::isfinite(minimum_pivot) || !(reconstruction_residual >= 0.0) ||
        !std::isfinite(reconstruction_residual) || reconstruction_residual > 1.0e-8 ||
        fingerprint != rpmd_ja_model_fingerprint(path))
      throw std::runtime_error("RPMD-JA stability sidecar does not validate this reference file");
    result.stability_checked = true;
  } else {
    validate_block_matrix(result.block_dynamical, dimension);
    for (int a = 0; a < 3; ++a) validate_block_matrix(result.block_site_transpose[a], dimension);
    require_finite(result.p_values, "qNEP P eigenvalues");
    require_finite(result.q_values, "qNEP Q eigenvalues");
    require_finite(result.p_vectors, "qNEP P eigenvectors");
    require_finite(result.q_vectors, "qNEP Q eigenvectors");
    if (result.energy_gradient_absolute_rms > kForceTolerance ||
        result.projection_relative_change > 5.0e-2 ||
        result.force_gradient_relative_error > kDifferenceTolerance ||
        result.hessian_symmetry_relative_error > kHessianSymmetryTolerance ||
        result.energy_second_probe_relative_error > kDifferenceTolerance ||
        std::any_of(result.block_relative_residual, result.block_relative_residual + 4,
          [](double x) { return !(x >= 0.0 && x <= 1.0e-8 && std::isfinite(x)); }) ||
        std::any_of(result.pbc, result.pbc + 3, [](int p) { return p != 1; }) ||
        std::any_of(result.site_transport_relative_error, result.site_transport_relative_error + 3,
          [](double x) { return !(x >= 0.0 && x <= kDifferenceTolerance && std::isfinite(x)); }))
      throw std::runtime_error("qNEP RPMD-JA reference diagnostics exceed the accepted finite-difference limits");
    const double actual_bound = block_max_abs_row_sum(result.block_dynamical, dimension);
    if (actual_bound > result.spectral_bound * (1.0 + 32.0 * std::numeric_limits<double>::epsilon()))
      throw std::runtime_error("qNEP RPMD-JA stored spectral bound is below its block row-sum bound");
    const std::string stability_path = path + ".stability";
    std::ifstream stability(stability_path);
    std::string tag;
    int stability_version = 0, atom_count = 0;
    std::uint64_t file_fingerprint = 0, config_fingerprint = 0;
    double minimum_pivot = 0.0, operator_bound = 0.0, translation_residual = 0.0;
    double reconstruction_residual = 0.0, softmode_error = 0.0;
    stability >> tag >> stability_version;
    if (tag != "GPUMDJA_QNEP_STABILITY" || (stability_version < 1 || stability_version > 4))
      throw std::runtime_error("missing or unsupported qNEP RPMD-JA stability sidecar: " + stability_path);
    stability >> tag >> std::hex >> file_fingerprint >> std::dec;
    if (tag != "fingerprint") throw std::runtime_error("invalid qNEP RPMD-JA sidecar file fingerprint");
    stability >> tag >> std::hex >> config_fingerprint >> std::dec;
    if (tag != "config_fingerprint") throw std::runtime_error("invalid qNEP RPMD-JA sidecar config fingerprint");
    stability >> tag >> atom_count;
    if (tag != "atoms") throw std::runtime_error("invalid qNEP RPMD-JA sidecar atom count");
    if (stability_version == 3) {
      std::string derivative_policy;
      stability >> tag >> derivative_policy;
      if (tag != "derivative_policy" || derivative_policy != result.mechanical_policy)
        throw std::runtime_error("qNEP RPMD-JA sidecar derivative policy does not match the reference");
    }
    int policy_beads=0, policy_derivative=0;
    const bool additive_policy=parse_additive_policy(result.mechanical_policy,policy_beads,policy_derivative);
    if (additive_policy != (stability_version == 4))
      throw std::runtime_error("additive qNEP RPMD-JA policy requires its own shifted certificate");
    if (stability_version == 1) {
      stability >> tag >> minimum_pivot;
      if (tag != "minimum_positive_eigenvalue")
        throw std::runtime_error("invalid qNEP RPMD-JA sidecar minimum eigenvalue");
    } else if (stability_version == 4) {
      std::string derivative_policy;
      stability >> tag >> derivative_policy;
      if (tag != "derivative_policy" || derivative_policy != result.mechanical_policy)
        throw std::runtime_error("qNEP RPMD-JA sidecar derivative policy does not match the reference");
      std::string certificate;
      stability >> tag >> certificate;
      if (tag != "certificate" || certificate != "additive_shifted_frobenius_v1")
        throw std::runtime_error("unsupported additive qNEP RPMD-JA stability certificate");
      stability >> tag >> result.additive_epsilon;
      if (tag != "epsilon_num") throw std::runtime_error("invalid additive qNEP epsilon field");
      stability >> tag >> result.additive_reconstruction_bound;
      if (tag != "reconstruction_bound") throw std::runtime_error("invalid additive qNEP reconstruction bound");
      stability >> tag >> minimum_pivot;
      if (tag != "minimum_cholesky_pivot") throw std::runtime_error("invalid additive qNEP shifted Cholesky pivot");
      stability >> tag >> operator_bound;
      if (tag != "relative_operator_bound") throw std::runtime_error("invalid additive qNEP operator bound");
      result.additive_beads=policy_beads;
    } else {
      std::string certificate;
      stability >> tag >> certificate;
      if (tag != "certificate" || certificate != "cholesky_relative_bound")
        throw std::runtime_error("unsupported qNEP RPMD-JA stability certificate");
      stability >> tag >> minimum_pivot;
      if (tag != "minimum_cholesky_pivot")
        throw std::runtime_error("invalid qNEP RPMD-JA sidecar Cholesky pivot");
      stability >> tag >> operator_bound;
      if (tag != "relative_operator_bound")
        throw std::runtime_error("invalid qNEP RPMD-JA sidecar relative operator bound");
    }
    stability >> tag >> translation_residual;
    if (tag != "translation_residual") throw std::runtime_error("invalid qNEP RPMD-JA sidecar translation residual");
    stability >> tag >> reconstruction_residual;
    if (tag != "reconstruction_residual") throw std::runtime_error("invalid qNEP RPMD-JA sidecar reconstruction residual");
    stability >> tag >> softmode_error;
    if (tag != "softmode_relative_error") throw std::runtime_error("invalid qNEP RPMD-JA sidecar softmode error");
    std::string trailing;
    if (!stability || stability >> trailing || atom_count != result.number_of_atoms || !(minimum_pivot > 0.0) ||
        !std::isfinite(minimum_pivot) || !(translation_residual >= 0.0) || translation_residual > 1.0e-8 ||
        !std::isfinite(translation_residual) || !(reconstruction_residual >= 0.0) ||
        !std::isfinite(reconstruction_residual) || reconstruction_residual > 1.0e-8 ||
        !(softmode_error >= 0.0) || softmode_error > 1.0e-2 || !std::isfinite(softmode_error) ||
        config_fingerprint != result.mechanical_config_fingerprint ||
        file_fingerprint != rpmd_ja_model_fingerprint(path))
      throw std::runtime_error("qNEP RPMD-JA stability sidecar does not validate this reference file");
    if (stability_version >= 2 && stability_version < 4 &&
        (!(operator_bound >= 0.0) || operator_bound > 1.0e-2 || !std::isfinite(operator_bound)))
      throw std::runtime_error("qNEP RPMD-JA relative operator certificate exceeds its accepted bound");
    if (stability_version == 4 &&
        (!(result.additive_epsilon > 0.0) || !std::isfinite(result.additive_epsilon) ||
         !(result.additive_reconstruction_bound >= 0.0) ||
         !(result.additive_reconstruction_bound < 0.5 * result.additive_epsilon) ||
         !std::isfinite(result.additive_reconstruction_bound) || !(minimum_pivot > 0.0) ||
         !std::isfinite(minimum_pivot) || operator_bound != 0.0))
      throw std::runtime_error("additive qNEP RPMD-JA shifted Frobenius certificate is invalid");
    result.stability_certificate = stability_version == 4 ? "additive_shifted_frobenius_v1" :
      (stability_version >= 2 ? "cholesky_relative_bound" : "legacy_spectrum_v1");
    result.minimum_cholesky_pivot = minimum_pivot;
    result.relative_operator_bound = operator_bound;
    result.stability_checked = true;
  }
  return result;
}

void generate_rpmd_ja_reference(
  const std::string& path,
  const double temperature,
  const double fd_step,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!(temperature > 0.0) || !std::isfinite(temperature) || !(fd_step > 0.0) || !std::isfinite(fd_step))
    throw std::invalid_argument("rpmd_ja reference requires positive finite temperature and fd_step");
  if (path.empty()) throw std::invalid_argument("rpmd_ja reference requires an output path");
  std::ifstream existing(path, std::ios::binary), existing_temporary(path + ".tmp", std::ios::binary);
  if (existing.good() || existing_temporary.good())
    throw std::runtime_error("RPMD-JA reference destination or temporary file already exists");
  if (atom.number_of_atoms <= 1 || atom.cpu_type.size() != static_cast<size_t>(atom.number_of_atoms) ||
      atom.cpu_mass.size() != static_cast<size_t>(atom.number_of_atoms))
    throw std::runtime_error("rpmd_ja reference requires initialized atom types and masses");
  if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
    throw std::runtime_error("rpmd_ja reference supports exactly one ordinary short-range NEP potential");
  const std::string& model = force.primary_nep_model_path();
  const std::uint64_t fingerprint = rpmd_ja_model_fingerprint(model);
  const int n = atom.number_of_atoms;
  matrix_size(n);
  const int d = 3 * n, r = d - 3;
  const double delta_bytes = 3.0 * static_cast<double>(matrix_size(n)) * sizeof(double);
  std::printf("    rpmd_ja dense reference: N=%d, d=%d, output DeltaH=%.3f GiB; generation calls=%d NEP evaluations\n",
    n, d, delta_bytes / (1024.0 * 1024.0 * 1024.0), 4 * d + 1);

  RpmdJAReference result;
  result.number_of_atoms = n;
  result.temperature = temperature;
  result.fd_step = fd_step;
  result.model_fingerprint = fingerprint;
  result.types = atom.cpu_type;
  result.masses = atom.cpu_mass;
  for (int type : result.types)
    if (type < 0 || type >= NUM_ELEMENTS) throw std::runtime_error("rpmd_ja reference found an invalid atom type");
  result.positions.resize(static_cast<std::size_t>(d));
  atom.position_per_atom.copy_to_host(result.positions.data());
  require_finite(result.positions, "reference positions");
  for (int i = 0; i < n; ++i)
    if (!(result.masses[i] > 0.0) || !std::isfinite(result.masses[i]))
      throw std::runtime_error("rpmd_ja reference requires positive finite atomic masses");
  std::copy(box.cpu_h, box.cpu_h + 9, result.cell);
  result.pbc[0] = box.pbc_x;
  result.pbc[1] = box.pbc_y;
  result.pbc[2] = box.pbc_z;
  for (int pbc : result.pbc)
    if (pbc != 0 && pbc != 1) throw std::runtime_error("rpmd_ja reference requires valid fixed-cell PBC flags");
  for (double value : result.cell)
    if (!std::isfinite(value)) throw std::runtime_error("rpmd_ja reference requires a finite cell");

  Evaluator evaluator(model, n, force.get_run_input(), box, result.types);
  const Evaluation reference = evaluator.evaluate(result.positions);
  double max_force = 0.0;
  for (double f : reference.force) max_force = std::max(max_force, std::abs(f));
  if (reference.edges.empty()) throw std::runtime_error("RPMD-JA reference has no local NEP edges");

  const auto& edges_reference = reference.edges;
  std::vector<double> hessian(static_cast<std::size_t>(d) * d, 0.0);
  std::vector<double> site[3];
  for (int a = 0; a < 3; ++a) site[a].assign(static_cast<std::size_t>(d) * d, 0.0);
  std::vector<double> coarse_k(hessian.size()), coarse_site[3];
  for (int a = 0; a < 3; ++a) coarse_site[a].assign(hessian.size(), 0.0);
  std::vector<double> plus = result.positions, minus = result.positions;
  std::vector<double> plus_half = result.positions, minus_half = result.positions;
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    for (int i = 0; i < d; ++i) plus[i] = minus[i] = plus_half[i] = minus_half[i] = result.positions[i];
    plus[coordinate] += fd_step;
    minus[coordinate] -= fd_step;
    plus_half[coordinate] += 0.5 * fd_step;
    minus_half[coordinate] -= 0.5 * fd_step;
    const Evaluation ep = evaluator.evaluate(plus), em = evaluator.evaluate(minus);
    const Evaluation ehp = evaluator.evaluate(plus_half), ehm = evaluator.evaluate(minus_half);
    for (int column = 0; column < d; ++column) {
      const std::size_t index = static_cast<std::size_t>(coordinate) * d + column;
      const double hp = -(ep.force[column] - em.force[column]) / (2.0 * fd_step);
      const double hh = -(ehp.force[column] - ehm.force[column]) / fd_step;
      hessian[index] = hh;
      coarse_k[index] = hp;
    }
    for (int alpha = 0; alpha < 3; ++alpha) {
      const std::vector<double> bp = site_flow_vector(ep, edges_reference, alpha, n);
      const std::vector<double> bm = site_flow_vector(em, edges_reference, alpha, n);
      const std::vector<double> bhp = site_flow_vector(ehp, edges_reference, alpha, n);
      const std::vector<double> bhm = site_flow_vector(ehm, edges_reference, alpha, n);
      for (int column = 0; column < d; ++column) {
        const std::size_t index = static_cast<std::size_t>(coordinate) * d + column;
        site[alpha][index] = (bhp[column] - bhm[column]) / fd_step;
        coarse_site[alpha][index] = (bp[column] - bm[column]) / (2.0 * fd_step);
      }
    }
  }
  const double k_convergence = relative_rms_difference(hessian, coarse_k);
  if (k_convergence > kDifferenceTolerance)
    throw std::runtime_error("RPMD-JA Hessian finite-difference h versus h/2 check failed");
  for (int alpha = 0; alpha < 3; ++alpha) {
    const double error = relative_rms_difference(site[alpha], coarse_site[alpha]);
    std::printf("    rpmd_ja reference site-flow h/h2 axis %c: %.3e\n", "xyz"[alpha], error);
    if (error > kDifferenceTolerance)
      throw std::runtime_error("RPMD-JA site-flow finite-difference h versus h/2 check failed");
  }
  const double asymmetry = relative_antisymmetry(hessian, d);
  if (asymmetry > kHessianSymmetryTolerance)
    throw std::runtime_error("RPMD-JA total Hessian antisymmetry exceeds tolerance");
  std::printf("    rpmd_ja reference finite-difference checks: Hessian h/h2 %.3e, max force %.3e eV/A, antisymmetry %.3e\n",
    k_convergence, max_force, asymmetry);
  for (int i = 0; i < d; ++i)
    for (int j = i + 1; j < d; ++j) {
      const double value = 0.5 * (hessian[static_cast<std::size_t>(i) * d + j] + hessian[static_cast<std::size_t>(j) * d + i]);
      hessian[static_cast<std::size_t>(i) * d + j] = hessian[static_cast<std::size_t>(j) * d + i] = value;
    }

  std::vector<double> dmat(hessian.size());
  for (int i = 0; i < d; ++i)
    for (int j = 0; j < d; ++j)
      dmat[static_cast<std::size_t>(i) * d + j] = hessian[static_cast<std::size_t>(i) * d + j] /
        std::sqrt(result.masses[i % n] * result.masses[j % n]);
  const double translation_residual = relative_translation_residual(dmat, result.masses, n);
  if (translation_residual > kHessianSymmetryTolerance)
    throw std::runtime_error("RPMD-JA mass-weighted Hessian fails acoustic translation residual check");
  std::printf("    rpmd_ja mass-weighted translation residual: %.3e\n", translation_residual);
  std::vector<double> q;
  rpmd_ja_reference_math::make_translation_complement(n, result.masses, q);
  GPU_Vector<double> q_gpu(q.size()), d_gpu(dmat.size()), tmp(static_cast<std::size_t>(r) * d), reduced_gpu(static_cast<std::size_t>(r) * r);
  q_gpu.copy_from_host(q.data());
  d_gpu.copy_from_host(dmat.data());
  gpublasHandle_t blas;
  check_blas(gpublasCreate(&blas), "create");
  try {
    dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_N, r, d, d, q_gpu.data(), r, d_gpu.data(), d, tmp.data(), r);
    dgemm(blas, JA_BLAS_OP_N, JA_BLAS_OP_T, r, r, d, tmp.data(), r, q_gpu.data(), r, reduced_gpu.data(), r);
    std::vector<double> reduced_transpose(static_cast<std::size_t>(r) * r), reduced(reduced_transpose.size());
    reduced_gpu.copy_to_host(reduced_transpose.data());
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < r; ++j)
        reduced[static_cast<std::size_t>(i) * r + j] = reduced_transpose[static_cast<std::size_t>(j) * r + i];
    std::vector<double> eigenvalues(r), eigenvectors(static_cast<std::size_t>(r) * r);
    std::vector<double> reduced_for_solver(reduced.size());
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < r; ++j)
        reduced_for_solver[static_cast<std::size_t>(j) * r + i] = reduced[static_cast<std::size_t>(i) * r + j];
    eigenvectors_symmetric_Jacobi(r, reduced_for_solver.data(), eigenvalues.data(), eigenvectors.data());
    require_finite(eigenvalues, "eigenvalues");
    require_finite(eigenvectors, "eigenvectors");
    check_eigensystem(r, reduced, eigenvalues, eigenvectors);
    const double max_eigenvalue = *std::max_element(eigenvalues.begin(), eigenvalues.end());
    if (!(max_eigenvalue > 0.0)) throw std::runtime_error("RPMD-JA reference has no positive modes");
    for (double value : eigenvalues) {
      if (!(value > 0.0))
        throw std::runtime_error("RPMD-JA reference has an additional nonpositive non-translation mode");
    }
    std::vector<double> omega(r);
    for (int i = 0; i < r; ++i) omega[i] = std::sqrt(eigenvalues[i]);
    const double beta_hbar = HBAR / (K_B * temperature);
    for (int alpha = 0; alpha < 3; ++alpha)
      transform_site_matrix(
        d, r, q, result.masses, dmat, site[alpha], omega, eigenvalues, beta_hbar, blas, eigenvectors, alpha == 0,
        result.delta_h[alpha]);
  } catch (...) {
    gpublasDestroy(blas);
    throw;
  }
  gpublasDestroy(blas);
  write_reference(path, result);
}

void generate_rpmd_ja_sparse_reference(
  const std::string& path,
  const double temperature,
  const double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!(temperature > 0.0) || !std::isfinite(temperature) || !(fd_step > 0.0) || !std::isfinite(fd_step))
    throw std::invalid_argument("sparse rpmd_ja reference requires positive finite temperature and fd_step");
  if (path.empty() || kernel_table_path.empty()) throw std::invalid_argument("sparse rpmd_ja requires output and kernel-table paths");
  std::ifstream existing(path, std::ios::binary), existing_temporary(path + ".tmp", std::ios::binary);
  if (existing.good() || existing_temporary.good()) throw std::runtime_error("RPMD-JA sparse destination or temporary already exists");
  if (atom.number_of_atoms <= 1 || atom.cpu_type.size() != static_cast<std::size_t>(atom.number_of_atoms) ||
      atom.cpu_mass.size() != static_cast<std::size_t>(atom.number_of_atoms))
    throw std::runtime_error("sparse rpmd_ja reference requires initialized atom types and masses");
  if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
    throw std::runtime_error("sparse rpmd_ja reference supports exactly one ordinary short-range NEP potential");
  const int n = atom.number_of_atoms, d = reference_dimension(atom.number_of_atoms);
  const std::string& model = force.primary_nep_model_path();
  RpmdJAReference result;
  result.backend = 1;
  result.number_of_atoms = n;
  result.temperature = temperature;
  result.fd_step = fd_step;
  result.model_fingerprint = rpmd_ja_model_fingerprint(model);
  result.types = atom.cpu_type;
  result.masses = atom.cpu_mass;
  for (int type : result.types)
    if (type < 0 || type >= NUM_ELEMENTS) throw std::runtime_error("sparse rpmd_ja found an invalid atom type");
  for (double mass : result.masses)
    if (!(mass > 0.0) || !std::isfinite(mass)) throw std::runtime_error("sparse rpmd_ja requires positive finite masses");
  result.positions.resize(static_cast<std::size_t>(d));
  atom.position_per_atom.copy_to_host(result.positions.data());
  require_finite(result.positions, "sparse reference positions");
  wrap_positions_once(box, result.positions, n);
  std::copy(box.cpu_h, box.cpu_h + 9, result.cell);
  result.pbc[0] = box.pbc_x;
  result.pbc[1] = box.pbc_y;
  result.pbc[2] = box.pbc_z;
  for (int pbc : result.pbc) if (pbc != 0 && pbc != 1) throw std::runtime_error("invalid sparse RPMD-JA PBC flag");
  for (double value : result.cell) if (!std::isfinite(value)) throw std::runtime_error("non-finite sparse RPMD-JA cell");
  load_kernel_table(kernel_table_path, result);

  Evaluator evaluator(model, n, force.get_run_input(), box, result.types);
  const Evaluation reference = evaluator.evaluate(result.positions);
  result.reference_edge_count = static_cast<std::uint64_t>(reference.edges.size());
  std::uint64_t edge_hash = 14695981039346656037ULL;
  const auto hash_edge_bytes = [&](const void* data, const std::size_t count) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < count; ++i) edge_hash = (edge_hash ^ bytes[i]) * 1099511628211ULL;
  };
  for (const auto& entry : reference.edges) {
    const EdgeKey& key = entry.first;
    const NEP_Local_Edge& edge = entry.second;
    for (int field : {std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), std::get<4>(key)})
      hash_edge_bytes(&field, sizeof(field));
    for (double value : edge.displacement) hash_edge_bytes(&value, sizeof(value));
  }
  result.reference_edge_fingerprint = edge_hash;
  double max_force = 0.0;
  for (double f : reference.force) max_force = std::max(max_force, std::abs(f));
  std::printf("    rpmd_ja sparse fixed-reference maximum force: %.3e eV/A\n", max_force);
  if (reference.edges.empty()) throw std::runtime_error("sparse RPMD-JA reference has no local NEP edges");
  check_cutoff_margin(box, evaluator.nep, result.types, result.positions, n, fd_step);

  std::vector<std::vector<EdgeKey>> keys_by_center(n);
  std::vector<std::vector<int>> centers_by_atom(n);
  for (const auto& entry : reference.edges) {
    const NEP_Local_Edge& edge = entry.second;
    keys_by_center[edge.center].push_back(entry.first);
    centers_by_atom[edge.neighbor].push_back(edge.center);
  }
  for (auto& centers : centers_by_atom) {
    std::sort(centers.begin(), centers.end());
    centers.erase(std::unique(centers.begin(), centers.end()), centers.end());
  }

  std::vector<std::vector<std::pair<int, double>>> d_columns(d);
  std::vector<std::vector<std::pair<int, double>>> b_columns[3] = {
    std::vector<std::vector<std::pair<int, double>>>(d),
    std::vector<std::vector<std::pair<int, double>>>(d),
    std::vector<std::vector<std::pair<int, double>>>(d)};
  double d_diff2 = 0.0, d_fine2 = 0.0;
  double b_diff2[3] = {}, b_fine2[3] = {};
  double max_unaffected_edge_residual = 0.0, reference_edge_gradient_scale = 0.0;
  for (const auto& entry : reference.edges)
    for (double derivative : entry.second.derivative)
      reference_edge_gradient_scale = std::max(reference_edge_gradient_scale, std::abs(derivative));
  std::vector<double> plus = result.positions, minus = result.positions;
  std::vector<double> plus_half = result.positions, minus_half = result.positions;
  const double tau = HBAR / (K_B * temperature);

  std::printf("    rpmd_ja sparse reference: N=%d, D=%d, kernel U=%.6g, generation calls=%d NEP evaluations; stability requires offline sparse checker\n",
    n, d, result.kernel_u, 4 * d + 1);
  for (int input = 0; input < d; ++input) {
    const int atom_index = input % n;
    std::vector<int> centers = centers_by_atom[atom_index];
    centers.push_back(atom_index);
    std::sort(centers.begin(), centers.end());
    centers.erase(std::unique(centers.begin(), centers.end()), centers.end());
    std::vector<unsigned char> support_d_atom(n, 0), support_b_atom(n, 0);
    for (int center : centers) {
      support_d_atom[center] = 1;
      for (const EdgeKey& key : keys_by_center[center]) {
        const int neighbor = std::get<1>(key);
        support_d_atom[neighbor] = 1;
        support_b_atom[neighbor] = 1;
      }
    }
    plus[input] = result.positions[input] + fd_step;
    minus[input] = result.positions[input] - fd_step;
    plus_half[input] = result.positions[input] + 0.5 * fd_step;
    minus_half[input] = result.positions[input] - 0.5 * fd_step;
    const SparseProbe ep = evaluate_sparse_probe(evaluator, plus, reference.edges, keys_by_center, centers, n);
    const SparseProbe em = evaluate_sparse_probe(evaluator, minus, reference.edges, keys_by_center, centers, n);
    const SparseProbe ehp = evaluate_sparse_probe(evaluator, plus_half, reference.edges, keys_by_center, centers, n);
    const SparseProbe ehm = evaluate_sparse_probe(evaluator, minus_half, reference.edges, keys_by_center, centers, n);
    max_unaffected_edge_residual = std::max(max_unaffected_edge_residual, std::max(
      std::max(ep.unaffected_edge_gradient_residual, em.unaffected_edge_gradient_residual),
      std::max(ehp.unaffected_edge_gradient_residual, ehm.unaffected_edge_gradient_residual)));
    plus[input] = minus[input] = plus_half[input] = minus_half[input] = result.positions[input];

    std::vector<double> fine_d(d), coarse_d(d);
    double supported_d_scale = 0.0;
    for (int output = 0; output < d; ++output) {
      const double mass_scale = 1.0 / std::sqrt(result.masses[output % n] * result.masses[atom_index]);
      fine_d[output] = -(ehp.force[output] - ehm.force[output]) / fd_step * mass_scale;
      coarse_d[output] = -(ep.force[output] - em.force[output]) / (2.0 * fd_step) * mass_scale;
      if (support_d_atom[output % n]) supported_d_scale = std::max(supported_d_scale,
        std::max(std::abs(fine_d[output]), std::abs(coarse_d[output])));
    }
    const double d_leak_limit = kSupportLeakTolerance * std::max(1.0, supported_d_scale);
    for (int output = 0; output < d; ++output) {
      if (!support_d_atom[output % n]) {
        if (std::abs(fine_d[output]) > d_leak_limit || std::abs(coarse_d[output]) > d_leak_limit)
          throw std::runtime_error("force derivative leaked outside the reference graph support");
        continue;
      }
      d_columns[input].emplace_back(output, fine_d[output]);
      const double diff = fine_d[output] - coarse_d[output];
      d_diff2 += diff * diff;
      d_fine2 += fine_d[output] * fine_d[output];
    }

    for (int alpha = 0; alpha < 3; ++alpha) {
      std::vector<double> fine_b(d), coarse_b(d);
      double supported_b_scale = 0.0;
      for (int output = 0; output < d; ++output) {
        const double mass_scale = 1.0 / std::sqrt(result.masses[output % n] * result.masses[atom_index]);
        fine_b[output] = (ehp.site_flow[alpha][output] - ehm.site_flow[alpha][output]) / fd_step * mass_scale;
        coarse_b[output] = (ep.site_flow[alpha][output] - em.site_flow[alpha][output]) / (2.0 * fd_step) * mass_scale;
        if (support_b_atom[output % n]) supported_b_scale = std::max(supported_b_scale,
          std::max(std::abs(fine_b[output]), std::abs(coarse_b[output])));
      }
      const double b_leak_limit = kSupportLeakTolerance * std::max(1.0, supported_b_scale);
      for (int output = 0; output < d; ++output) {
        if (!support_b_atom[output % n]) {
          if (std::abs(fine_b[output]) > b_leak_limit || std::abs(coarse_b[output]) > b_leak_limit)
            throw std::runtime_error("site-flow derivative leaked outside the reference graph support");
          continue;
        }
        b_columns[alpha][input].emplace_back(output, fine_b[output]);
        const double diff = fine_b[output] - coarse_b[output];
        b_diff2[alpha] += diff * diff;
        b_fine2[alpha] += fine_b[output] * fine_b[output];
      }
    }
  }
  result.fd_relative_d = std::sqrt(d_diff2 / std::max(d_fine2, 1.0e-30));
  for (int alpha = 0; alpha < 3; ++alpha)
    result.fd_relative_b[alpha] = std::sqrt(b_diff2[alpha] / std::max(b_fine2[alpha], 1.0e-30));
  if (result.fd_relative_d > kDifferenceTolerance)
    throw std::runtime_error("sparse RPMD-JA Hessian h versus h/2 diagnostic exceeded tolerance");
  for (int alpha = 0; alpha < 3; ++alpha)
    if (result.fd_relative_b[alpha] > kDifferenceTolerance)
      throw std::runtime_error("sparse RPMD-JA site-flow h versus h/2 diagnostic exceeded tolerance");
  std::printf("    rpmd_ja sparse h/h2 diagnostic (not a strict derivative error bound): D %.3e, B^T xyz %.3e %.3e %.3e\n",
    result.fd_relative_d, result.fd_relative_b[0], result.fd_relative_b[1], result.fd_relative_b[2]);
  std::printf("    rpmd_ja sparse graph-external NEP edge-gradient residual %.3e (allowed %.3e)\n",
    max_unaffected_edge_residual, kSupportLeakTolerance * std::max(1.0, reference_edge_gradient_scale));
  std::printf("    rpmd_ja fixed-edge policy %s: %llu keys, fingerprint %016llx\n",
    result.edge_policy.c_str(), static_cast<unsigned long long>(result.reference_edge_count),
    static_cast<unsigned long long>(result.reference_edge_fingerprint));

  std::size_t pair_stage_bytes = 0;
  for (int column = 0; column < d; ++column) {
    pair_stage_bytes += d_columns[column].capacity() * sizeof(std::pair<int, double>);
    for (int alpha = 0; alpha < 3; ++alpha)
      pair_stage_bytes += b_columns[alpha][column].capacity() * sizeof(std::pair<int, double>);
  }

  columns_to_csr(d, d_columns, result.dynamical);
  d_columns.clear();
  d_columns.shrink_to_fit();
  const double asymmetry = symmetrize_csr(result.dynamical, d);
  if (asymmetry > kHessianSymmetryTolerance)
    throw std::runtime_error("sparse RPMD-JA dynamical matrix antisymmetry exceeds tolerance");
  const double translation_residual = csr_translation_residual(result.dynamical, result.masses, n);
  if (translation_residual > kHessianSymmetryTolerance)
    throw std::runtime_error("sparse RPMD-JA dynamical matrix fails translation residual check");
  result.spectral_bound = csr_max_abs_row_sum(result.dynamical);
  if (!(result.spectral_bound > 0.0) || !std::isfinite(result.spectral_bound))
    throw std::runtime_error("sparse RPMD-JA dynamical matrix has no finite positive row-sum bound");
  for (int alpha = 0; alpha < 3; ++alpha) {
    columns_to_csr(d, b_columns[alpha], result.site_transpose[alpha]);
    b_columns[alpha].clear();
    b_columns[alpha].shrink_to_fit();
  }
  const double required_kernel_u = tau * std::sqrt(result.spectral_bound);
  const bool kernel_covers_spectrum = required_kernel_u <= result.kernel_u;
  result.stability_checked = false;
  const auto bytes = [](const RpmdJASparseMatrix& matrix) {
    return matrix.row_offsets.size() * sizeof(std::uint64_t) + matrix.columns.size() * sizeof(int) +
      matrix.values.size() * sizeof(double);
  };
  const std::size_t csr_bytes = bytes(result.dynamical) + bytes(result.site_transpose[0]) +
    bytes(result.site_transpose[1]) + bytes(result.site_transpose[2]);
  const std::size_t base_arrays = result.positions.size() * sizeof(double) + result.masses.size() * sizeof(double) +
    result.types.size() * sizeof(int);
  std::printf("    rpmd_ja sparse CSR bytes: %zu (%.3f GiB); input/base arrays %zu bytes; spectral row-sum bound %.6g; tau*sqrt(bound)=%.6g <= U %.6g\n",
    csr_bytes, static_cast<double>(csr_bytes) / (1024.0 * 1024.0 * 1024.0), base_arrays,
    result.spectral_bound, required_kernel_u, result.kernel_u);
  const auto csr_capacity_bytes = [](const RpmdJASparseMatrix& matrix) {
    return matrix.row_offsets.capacity() * sizeof(std::uint64_t) + matrix.columns.capacity() * sizeof(int) +
      matrix.values.capacity() * sizeof(double);
  };
  std::size_t graph_vector_bytes = 0, csr_capacity_total = 0, max_csr_row = 0;
  for (int center = 0; center < n; ++center) {
    graph_vector_bytes += keys_by_center[center].capacity() * sizeof(EdgeKey) +
      centers_by_atom[center].capacity() * sizeof(int);
  }
  csr_capacity_total += csr_capacity_bytes(result.dynamical);
  const auto update_max_row = [&](const RpmdJASparseMatrix& matrix) {
    for (std::size_t row = 0; row + 1 < matrix.row_offsets.size(); ++row)
      max_csr_row = std::max(max_csr_row, static_cast<std::size_t>(matrix.row_offsets[row + 1] - matrix.row_offsets[row]));
  };
  update_max_row(result.dynamical);
  for (int alpha = 0; alpha < 3; ++alpha) {
    csr_capacity_total += csr_capacity_bytes(result.site_transpose[alpha]);
    update_max_row(result.site_transpose[alpha]);
  }
  const std::size_t csr_conversion_scratch = (static_cast<std::size_t>(d) + 1) * sizeof(std::uint64_t) +
    max_csr_row * sizeof(std::pair<int, double>);
  const std::size_t probe_payload = static_cast<std::size_t>(25) * d * sizeof(double); // Four results, fine/coarse D/B vectors, and five position arrays.
  const std::size_t edge_map_estimate = 3 * reference.edges.size() *
    (sizeof(EdgeKey) + sizeof(NEP_Local_Edge) + 3 * sizeof(void*)); // Reference, current map, and local edge-vector/by-image staging.
  const std::size_t symmetrize_copy = result.dynamical.values.capacity() * sizeof(double);
  const std::size_t generation_storage_estimate = base_arrays + pair_stage_bytes + graph_vector_bytes +
    csr_capacity_total + probe_payload + edge_map_estimate + symmetrize_copy + csr_conversion_scratch;
  std::printf("    rpmd_ja estimated generation peak tracked-storage %.3f GiB (CSR capacity %zu, pair-column capacity %zu, graph vectors %zu, probe payload %zu, 3x edge-map/vector estimate %zu, CSR conversion scratch %zu, D symmetry copy %zu; not OS/GPU measured and excludes opaque NEP private buffers/allocator overhead)\n",
    static_cast<double>(generation_storage_estimate) / (1024.0 * 1024.0 * 1024.0), csr_capacity_total,
    pair_stage_bytes, graph_vector_bytes, probe_payload, edge_map_estimate, csr_conversion_scratch, symmetrize_copy);
  write_reference(path, result);
  if (!kernel_covers_spectrum) {
    std::ostringstream message;
    message << "sparse RPMD-JA reference pending: required kernel U=" << std::setprecision(17)
            << required_kernel_u << " exceeds table U=" << result.kernel_u
            << "; fixed D/B and edge metadata were saved to " << path
            << "; rebind a wider table offline, then run the stability checker";
    throw std::runtime_error(message.str());
  }
}

static void generate_rpmd_ja_qnep_raw_reference(
  const std::string& path,
  const double temperature,
  const double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force,
  const bool require_zero_net_gradient)
{
  if (!(temperature > 0.0) || !std::isfinite(temperature) || !(fd_step > 0.0) || !std::isfinite(fd_step))
    throw std::invalid_argument("qNEP rpmd_ja reference requires positive finite temperature and fd_step");
  if (path.empty() || kernel_table_path.empty())
    throw std::invalid_argument("qNEP rpmd_ja reference requires raw-output and kernel-table paths");
  RpmdJAReference validated_kernel;
  load_kernel_table(kernel_table_path, validated_kernel);
  if (atom.number_of_atoms <= 1 || atom.cpu_type.size() != static_cast<size_t>(atom.number_of_atoms) ||
      atom.cpu_mass.size() != static_cast<size_t>(atom.number_of_atoms))
    throw std::runtime_error("qNEP rpmd_ja reference requires initialized atom types and masses");
  if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
    throw std::runtime_error("qNEP rpmd_ja reference supports exactly one qNEP potential");
  auto* active_qnep = dynamic_cast<NEP_Charge*>(&force.get_potential(0));
  if (active_qnep == nullptr)
    throw std::runtime_error("qNEP rpmd_ja reference rejects non-qNEP and mixed potentials");
  if (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2)
    throw std::runtime_error("qNEP rpmd_ja reference supports charge modes 1 and 2 only");
  if (!active_qnep->uses_pppm())
    throw std::runtime_error("qNEP rpmd_ja reference currently supports kspace_method pppm only");
  for (int pbc : {box.pbc_x, box.pbc_y, box.pbc_z})
    if (pbc != 1) throw std::runtime_error("qNEP rpmd_ja reference currently requires a fully periodic fixed cell");

  const int n = atom.number_of_atoms, d = reference_dimension(atom.number_of_atoms);
  const std::string model = force.primary_nep_model_path();
  const std::uint64_t model_fingerprint = rpmd_ja_model_fingerprint(model);
  const std::uint64_t config_fingerprint = rpmd_ja_qnep_config_fingerprint(force);
  std::vector<double> positions(static_cast<std::size_t>(d));
  atom.position_per_atom.copy_to_host(positions.data());
  require_finite(positions, "qNEP reference positions");
  for (double mass : atom.cpu_mass)
    if (!(mass > 0.0) || !std::isfinite(mass)) throw std::runtime_error("qNEP rpmd_ja requires positive finite masses");
  for (double value : box.cpu_h)
    if (!std::isfinite(value)) throw std::runtime_error("qNEP rpmd_ja requires a finite cell");

  const std::string temporary = path + ".tmp";
  std::ifstream existing(path, std::ios::binary), existing_temporary(temporary, std::ios::binary);
  if (existing.good() || existing_temporary.good())
    throw std::runtime_error("qNEP rpmd_ja raw output or temporary file already exists");
  const auto generation_start = std::chrono::steady_clock::now();

  Box observer_box = box;
  observer_box.get_inverse();
  observer_box.set_is_orthogonal();
  QEvaluator evaluator(model, n, force.get_run_input(), observer_box, atom.cpu_type, active_qnep->get_pppm_mesh_spacing());
  QEvaluation reference = evaluator.evaluate(positions);
  std::vector<double> reference_gradient = evaluator.analytic_gradient();
  double precheck_force_gradient_diff2 = 0.0;
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    const double error = reference_gradient[coordinate] + reference.force[coordinate];
    precheck_force_gradient_diff2 += error * error;
  }
  const double precheck_force_gradient_abs = std::sqrt(precheck_force_gradient_diff2 / d);
  std::vector<int> precheck_coordinates;
  std::vector<int> representative_types;
  for (int atom_index = 0; atom_index < n; ++atom_index) {
    if (std::find(representative_types.begin(), representative_types.end(), atom.cpu_type[atom_index]) ==
        representative_types.end()) {
      representative_types.push_back(atom.cpu_type[atom_index]);
      for (int alpha = 0; alpha < 3; ++alpha) precheck_coordinates.push_back(alpha * n + atom_index);
    }
  }
  double precheck_v_diff2 = 0.0, precheck_v_fine2 = 0.0;
  double precheck_k_diff2 = 0.0, precheck_k_fine2 = 0.0;
  double precheck_c_diff2[3] = {}, precheck_c_fine2[3] = {};
  double precheck_identity_diff2 = 0.0, precheck_identity_scale2 = 0.0;
  double precheck_identity_max_abs = 0.0;
  int precheck_identity_max_coordinate = precheck_coordinates.empty() ? 0 : precheck_coordinates.front();
  std::vector<double> unit_direction(static_cast<std::size_t>(d), 0.0);
  for (int coordinate : precheck_coordinates) {
    unit_direction[coordinate] = 1.0;
    const std::vector<double> site_jvp = evaluator.analytic_site_jvp(unit_direction);
    const double summed_jvp = std::accumulate(site_jvp.begin(), site_jvp.end(), 0.0);
    const double error = summed_jvp - reference_gradient[coordinate];
    precheck_identity_diff2 += error * error;
    precheck_identity_scale2 += reference_gradient[coordinate] * reference_gradient[coordinate];
    if (std::abs(error) > precheck_identity_max_abs) {
      precheck_identity_max_abs = std::abs(error);
      precheck_identity_max_coordinate = coordinate;
    }
    unit_direction[coordinate] = 0.0;
  }
  for (std::size_t sample = 0; sample < precheck_coordinates.size(); ++sample) {
    const int coordinate = precheck_coordinates[sample];
    std::array<std::vector<double>, 4> probe_positions = {positions, positions, positions, positions};
    probe_positions[0][coordinate] += fd_step;
    probe_positions[1][coordinate] -= fd_step;
    probe_positions[2][coordinate] += 2.0 * fd_step;
    probe_positions[3][coordinate] -= 2.0 * fd_step;
    std::array<QEvaluation, 4> stencil;
    std::array<std::vector<double>, 4> gradient_stencil;
    for (int lane = 0; lane < 4; ++lane) {
      stencil[lane] = evaluator.evaluate(probe_positions[lane]);
      gradient_stencil[lane] = evaluator.analytic_gradient();
    }
    for (int i = 0; i < n; ++i) {
      const double fine = rpmd_ja_reference_math::central_difference_4th(
        stencil[0].energy[i], stencil[1].energy[i], stencil[2].energy[i], stencil[3].energy[i], fd_step);
      const double coarse = rpmd_ja_reference_math::central_difference_2nd(
        stencil[0].energy[i], stencil[1].energy[i], fd_step);
      const double difference = 3.0 * (fine - coarse);
      precheck_v_diff2 += difference * difference;
      precheck_v_fine2 += fine * fine;
    }
    for (int r = 0; r < d; ++r) {
      const double fine = rpmd_ja_reference_math::central_difference_4th(
        gradient_stencil[0][r], gradient_stencil[1][r], gradient_stencil[2][r], gradient_stencil[3][r], fd_step);
      const double coarse = rpmd_ja_reference_math::central_difference_2nd(
        gradient_stencil[0][r], gradient_stencil[1][r], fd_step);
      const double difference = 3.0 * (fine - coarse);
      precheck_k_diff2 += difference * difference;
      precheck_k_fine2 += fine * fine;
      for (int flux = 0; flux < 3; ++flux) {
        const int component[3][3] = {{0, 3, 4}, {6, 1, 5}, {7, 8, 2}};
        const int offset = component[flux][r / n] * n + r % n;
        const double c_fine = rpmd_ja_reference_math::central_difference_4th(
          stencil[0].virial[offset], stencil[1].virial[offset], stencil[2].virial[offset], stencil[3].virial[offset], fd_step);
        const double c_coarse = rpmd_ja_reference_math::central_difference_2nd(
          stencil[0].virial[offset], stencil[1].virial[offset], fd_step);
        const double c_difference = 3.0 * (c_fine - c_coarse);
        precheck_c_diff2[flux] += c_difference * c_difference;
        precheck_c_fine2[flux] += c_fine * c_fine;
      }
    }
  }
  const double precheck_identity_abs = std::sqrt(precheck_identity_diff2 / precheck_coordinates.size());
  const double precheck_identity_relative = std::sqrt(precheck_identity_diff2 / std::max(precheck_identity_scale2, 1.0e-300));
  const double precheck_v = std::sqrt(precheck_v_diff2 / std::max(precheck_v_fine2, 1.0e-300));
  const double precheck_k = std::sqrt(precheck_k_diff2 / std::max(precheck_k_fine2, 1.0e-300));
  const double precheck_c[3] = {
    std::sqrt(precheck_c_diff2[0] / std::max(precheck_c_fine2[0], 1.0e-300)),
    std::sqrt(precheck_c_diff2[1] / std::max(precheck_c_fine2[1], 1.0e-300)),
    std::sqrt(precheck_c_diff2[2] / std::max(precheck_c_fine2[2], 1.0e-300))};
  const bool precheck_failed = !std::isfinite(precheck_force_gradient_abs) ||
    !std::isfinite(precheck_identity_abs) || !std::isfinite(precheck_identity_relative) ||
    !std::isfinite(precheck_v) || !std::isfinite(precheck_k) ||
    !std::isfinite(precheck_c[0]) || !std::isfinite(precheck_c[1]) || !std::isfinite(precheck_c[2]) ||
    precheck_force_gradient_abs > kForceTolerance || precheck_identity_abs > kForceTolerance ||
    precheck_k > kDifferenceTolerance || precheck_c[0] > kDifferenceTolerance ||
    precheck_c[1] > kDifferenceTolerance || precheck_c[2] > kDifferenceTolerance;
  const bool precheck_gradient_pass = std::isfinite(precheck_force_gradient_abs) && precheck_force_gradient_abs <= kForceTolerance;
  const bool precheck_jvp_pass = std::isfinite(precheck_identity_abs) && std::isfinite(precheck_identity_relative) &&
    precheck_identity_abs <= kForceTolerance;
  const bool precheck_k_pass = std::isfinite(precheck_k) && precheck_k <= kDifferenceTolerance;
  const bool precheck_c_pass[3] = {
    std::isfinite(precheck_c[0]) && precheck_c[0] <= kDifferenceTolerance,
    std::isfinite(precheck_c[1]) && precheck_c[1] <= kDifferenceTolerance,
    std::isfinite(precheck_c[2]) && precheck_c[2] <= kDifferenceTolerance};
  const bool precheck_v_finite = std::isfinite(precheck_v);
  print_qnep_precheck_table(fd_step, "Generation", precheck_force_gradient_abs, precheck_gradient_pass,
    precheck_identity_abs, precheck_jvp_pass, precheck_k, precheck_k_pass, precheck_c, precheck_c_pass,
    precheck_v_finite, precheck_failed, false);
  std::fflush(stdout);
  if (precheck_failed) {
    std::ostringstream failure;
    failure << "qNEP rpmd_ja sampled stencil-step consistency check failed; raw generation stopped early: fd_step="
      << fd_step << ", analytic-gradient/native-force abs RMS " << precheck_force_gradient_abs
      << " (limit " << kForceTolerance << "), site-JVP/gradient identity abs RMS " << precheck_identity_abs
      << " (relative " << precheck_identity_relative << ", limit " << kForceTolerance
      << " eV/A absolute), V D2(h)-D2(2h) " << precheck_v
      << " (finite required; diagnostic only), K " << precheck_k << " (limit " << kDifferenceTolerance
      << "), Cxyz " << precheck_c[0] << ' ' << precheck_c[1] << ' ' << precheck_c[2]
      << " (each limit " << kDifferenceTolerance << "), site-JVP max|e| " << precheck_identity_max_abs
      << " at atom " << precheck_identity_max_coordinate % n << " type "
      << atom.cpu_type[precheck_identity_max_coordinate % n] << " axis "
      << "xyz"[precheck_identity_max_coordinate / n] << "; FAIL=";
    bool first_failure = true;
    const auto add_failure = [&](const char* name) {
      if (!first_failure) failure << ',';
      failure << name; first_failure = false;
    };
    if (!std::isfinite(precheck_force_gradient_abs) || !std::isfinite(precheck_identity_abs) ||
        !std::isfinite(precheck_identity_relative) || !std::isfinite(precheck_v) || !std::isfinite(precheck_k) || !std::isfinite(precheck_c[0]) ||
        !std::isfinite(precheck_c[1]) || !std::isfinite(precheck_c[2])) add_failure("non_finite");
    if (precheck_force_gradient_abs > kForceTolerance) add_failure("gradient_native_force");
    if (precheck_identity_abs > kForceTolerance) add_failure("site_jvp_gradient");
    if (!std::isfinite(precheck_v)) add_failure("V_non_finite");
    if (precheck_k > kDifferenceTolerance) add_failure("K_stencil");
    if (precheck_c[0] > kDifferenceTolerance) add_failure("Cx_stencil");
    if (precheck_c[1] > kDifferenceTolerance) add_failure("Cy_stencil");
    if (precheck_c[2] > kDifferenceTolerance) add_failure("Cz_stencil");
    std::fprintf(stderr, "%s\n", failure.str().c_str());
    throw std::runtime_error(failure.str());
  }
  std::printf("    qNEP rpmd_ja precheck: analytic-gradient/native-force abs RMS %.3e; site-JVP/gradient identity abs RMS %.3e (relative %.3e); V D2(h)-D2(2h) %.3e diagnostic, K/C stencil-step consistency %.3e %.3e %.3e %.3e\n",
    precheck_force_gradient_abs, precheck_identity_abs, precheck_identity_relative, precheck_v, precheck_k,
    precheck_c[0], precheck_c[1], precheck_c[2]);
  reference = evaluator.evaluate(positions);
  reference_gradient = evaluator.analytic_gradient();
  double native_gradient_diff2 = 0.0, native_gradient_force_scale2 = 0.0;
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    const double error = reference_gradient[coordinate] + reference.force[coordinate];
    native_gradient_diff2 += error * error;
    native_gradient_force_scale2 += reference.force[coordinate] * reference.force[coordinate];
  }
  const double native_gradient_abs = std::sqrt(native_gradient_diff2 / d);
  const double native_gradient_relative = std::sqrt(native_gradient_diff2 / std::max(native_gradient_force_scale2, 1.0e-300));
  if (!std::isfinite(native_gradient_abs) || !std::isfinite(native_gradient_relative) || native_gradient_abs > kForceTolerance)
    throw std::runtime_error("qNEP rpmd_ja analytic energy gradient differs from native force beyond 1e-4 eV/A");
  double max_force = 0.0, force_norm2 = 0.0, energy0 = 0.0;
  for (double value : reference.force) {
    max_force = std::max(max_force, std::abs(value));
    force_norm2 += value * value;
  }
  for (double value : reference.energy) energy0 += value;

  struct RemoveRawTemporary
  {
    explicit RemoveRawTemporary(const std::string& value) : path(value) {}
    std::string path;
    bool active = true;
    ~RemoveRawTemporary() { if (active) std::remove(path.c_str()); }
  } remove_temporary(temporary);
  std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot create qNEP rpmd_ja raw file: " + temporary);
  constexpr char raw_magic[8] = {'G','P','J','Q','R','A','W','\0'};
  constexpr char raw_layout[] = "xyz_soa;derivative_input_rows_output_columns";
  const std::uint32_t raw_version = 3, endian = kEndian;
  const int charge_mode = active_qnep->get_charge_mode();
  const int uses_pppm = active_qnep->uses_pppm() ? 1 : 0;
  const double mesh_spacing = active_qnep->get_pppm_mesh_spacing();
  out.write(raw_magic, sizeof(raw_magic));
  write_value(out, raw_version); write_value(out, endian); write_value(out, n); write_value(out, d);
  write_value(out, temperature); write_value(out, fd_step); write_value(out, model_fingerprint);
  write_value(out, config_fingerprint); write_value(out, charge_mode); write_value(out, uses_pppm);
  write_value(out, mesh_spacing); out.write(raw_layout, sizeof(raw_layout));
  out.write(reinterpret_cast<const char*>(box.cpu_h), sizeof(box.cpu_h));
  const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
  out.write(reinterpret_cast<const char*>(pbc), sizeof(pbc));
  write_vector(out, atom.cpu_type); write_vector(out, atom.cpu_mass); write_vector(out, positions);
  write_value(out, energy0); write_vector(out, reference.energy); write_vector(out, reference.force);
  write_vector(out, reference.virial);
  if (!out) throw std::runtime_error("failed writing qNEP rpmd_ja raw header");

  const std::streampos data_start = out.tellp();
  const std::uint64_t v_count = static_cast<std::uint64_t>(d) * n;
  const std::uint64_t c_count = static_cast<std::uint64_t>(3) * d * d;
  const std::uint64_t k_count = static_cast<std::uint64_t>(d) * d;
  const std::streamoff v_bytes = static_cast<std::streamoff>(v_count * sizeof(double));
  const std::streamoff c_bytes = static_cast<std::streamoff>(c_count * sizeof(double));
  const std::streamoff k_bytes = static_cast<std::streamoff>(k_count * sizeof(double));
  const std::streamoff stats_bytes = static_cast<std::streamoff>(18 * sizeof(double));
  if (data_start < 0 || 2 * v_bytes > std::numeric_limits<std::streamoff>::max() - 2 * c_bytes ||
      2 * v_bytes + 2 * c_bytes > std::numeric_limits<std::streamoff>::max() - k_bytes - stats_bytes)
    throw std::runtime_error("qNEP rpmd_ja raw output size overflows platform offsets");
  const std::streampos coarse_v_start = data_start + v_bytes + c_bytes + k_bytes;
  const std::streampos coarse_c_start = coarse_v_start + v_bytes;
  const std::streampos stats_start = coarse_c_start + c_bytes;
  out.seekp(stats_start + stats_bytes - 1);
  out.put('\0');
  out.flush();
  if (!out) throw std::runtime_error("cannot reserve qNEP rpmd_ja raw matrix storage");

  double jvp_identity_diff2 = 0.0, jvp_identity_scale2 = 0.0;
  double k_diff2 = 0.0, k_fine2 = 0.0;
  double c_diff2[3] = {}, c_fine2[3] = {};
  double energy_gradient_diff2 = 0.0, energy_gradient_scale2 = 0.0;
  std::vector<double> plus = positions, minus = positions, plus_twice = positions, minus_twice = positions;
  std::vector<double> v_fine(n), v_coarse(n), k_fine(d), k_coarse(d);
  std::vector<double> full_v_gradient(static_cast<std::size_t>(d), 0.0);
  std::vector<std::vector<double>> probe_directions(3, std::vector<double>(d));
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    probe_directions[0][coordinate] = std::sin((coordinate + 1) * 0.7548776662466927) +
      0.5 * std::cos((coordinate + 1) * 0.5698402909980532);
    probe_directions[1][coordinate] = std::cos((coordinate + 1) * 0.438579021) -
      0.25 * std::sin((coordinate + 1) * 0.812681);
    probe_directions[2][coordinate] = std::sin((coordinate + 1) * 0.327491) +
      0.75 * std::cos((coordinate + 1) * 0.639137);
  }
  double relative_com[3] = {}, total_mass = 0.0;
  for (int i = 0; i < n; ++i) {
    total_mass += atom.cpu_mass[i];
    for (int alpha = 0; alpha < 3; ++alpha)
      relative_com[alpha] += atom.cpu_mass[i] * probe_directions[2][alpha * n + i];
  }
  for (int alpha = 0; alpha < 3; ++alpha) relative_com[alpha] /= total_mass;
  double probe_norm2[3] = {};
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    const int alpha = coordinate / n;
    probe_directions[2][coordinate] -= relative_com[alpha];
    for (int probe = 0; probe < 3; ++probe)
      probe_norm2[probe] += probe_directions[probe][coordinate] * probe_directions[probe][coordinate];
  }
  for (int probe = 0; probe < 3; ++probe) {
    const double inverse = 1.0 / std::sqrt(probe_norm2[probe]);
    for (double& value : probe_directions[probe]) value *= inverse;
  }
  double matrix_quadratic[3] = {};
  std::array<std::vector<double>, 3> c_fine, c_coarse;
  for (int alpha = 0; alpha < 3; ++alpha) { c_fine[alpha].resize(d); c_coarse[alpha].resize(d); }
  constexpr int virial_component[3][3] = {{0, 3, 4}, {6, 1, 5}, {7, 8, 2}};
  std::uint64_t next_decile = 1;
  const auto v_phase_start = std::chrono::steady_clock::now();
  const auto save_row = [&](const std::streampos start, const int row, const int width, const std::vector<double>& values) {
    out.seekp(start + static_cast<std::streamoff>(row) * width * sizeof(double));
    write_vector(out, values);
  };
  std::fill(unit_direction.begin(), unit_direction.end(), 0.0);
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    unit_direction[coordinate] = 1.0;
    v_fine = evaluator.analytic_site_jvp(unit_direction);
    const double summed_jvp = std::accumulate(v_fine.begin(), v_fine.end(), 0.0);
    full_v_gradient[coordinate] = summed_jvp;
    const double identity_error = summed_jvp - reference_gradient[coordinate];
    jvp_identity_diff2 += identity_error * identity_error;
    jvp_identity_scale2 += reference_gradient[coordinate] * reference_gradient[coordinate];
    v_coarse = v_fine;
    save_row(data_start, coordinate, n, v_fine);
    save_row(coarse_v_start, coordinate, n, v_coarse);
    unit_direction[coordinate] = 0.0;
    const std::uint64_t completed = static_cast<std::uint64_t>(coordinate) + 1;
    if (next_decile <= 10 && completed * 10 >= next_decile * static_cast<std::uint64_t>(2) * d) {
      while (next_decile <= 10 && completed * 10 >= next_decile * static_cast<std::uint64_t>(2) * d) ++next_decile;
      const int percent = static_cast<int>(10 * (next_decile - 1));
      std::printf("    qNEP rpmd_ja phase V column %d/%d (%d%% overall)\n", coordinate + 1, d, percent);
      std::fflush(stdout);
    }
  }
  const double v_phase_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - v_phase_start).count();
  const auto net_stats = rpmd_ja_reference_math::net_force_stats(full_v_gradient, n);
  std::printf("    exact full-column V gradient net xyz=(%.17g, %.17g, %.17g), norm=%.17g, vector_norm=%.17g, limit=%.17g, status=%s (eV/A); computed from every V column in qraw atom order.\n",
    net_stats.net[0], net_stats.net[1], net_stats.net[2], net_stats.net_norm, net_stats.vector_norm,
    net_stats.limit, net_stats.within_limit ? "within-limit" : "nonzero-net-or-nonfinite");
  const double reference_net_difference = [&]() {
    double norm2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      double reference_net = 0.0;
      for (int i = 0; i < n; ++i) reference_net += reference_gradient[axis * n + i];
      const double difference = net_stats.net[axis] - reference_net;
      norm2 += difference * difference;
    }
    return std::sqrt(norm2);
  }();
  std::printf("    exact full-column V net minus reference analytic-gradient net norm=%.17g eV/A.\n", reference_net_difference);
  std::fflush(stdout);
  const double jvp_identity_abs = std::sqrt(jvp_identity_diff2 / d);
  const double jvp_identity_relative = std::sqrt(jvp_identity_diff2 / std::max(jvp_identity_scale2, 1.0e-300));
  if (!std::isfinite(jvp_identity_abs) || !std::isfinite(jvp_identity_relative) || jvp_identity_abs > kForceTolerance)
    throw std::runtime_error("qNEP rpmd_ja site-JVP/analytic-gradient identity exceeds 1e-4 eV/A");
  if (require_zero_net_gradient && !net_stats.within_limit) {
    std::ostringstream message;
    message << std::setprecision(17) << "native fit raw qNEP reference gradient has a nonzero net force"
      << "; net_xyz=(" << net_stats.net[0] << ',' << net_stats.net[1] << ',' << net_stats.net[2]
      << "), net_norm=" << net_stats.net_norm << ", vector_norm=" << net_stats.vector_norm
      << ", limit=" << net_stats.limit << ", status=nonzero-net-or-nonfinite";
    throw std::runtime_error(message.str());
  }
  const auto kc_phase_start = std::chrono::steady_clock::now();
  for (int coordinate = 0; coordinate < d; ++coordinate) {
    plus = minus = plus_twice = minus_twice = positions;
    plus[coordinate] += fd_step; minus[coordinate] -= fd_step;
    plus_twice[coordinate] += 2.0 * fd_step; minus_twice[coordinate] -= 2.0 * fd_step;
    std::array<QEvaluation, 4> stencil;
    std::array<std::vector<double>, 4> gradient_stencil;
    const std::array<std::vector<double>*, 4> xyz = {&plus, &minus, &plus_twice, &minus_twice};
    for (int lane = 0; lane < 4; ++lane) {
      stencil[lane] = evaluator.evaluate(*xyz[lane]);
      gradient_stencil[lane] = evaluator.analytic_gradient();
    }
    const QEvaluation& ep = stencil[0];
    const QEvaluation& em = stencil[1];
    const QEvaluation& ehp = stencil[2];
    const QEvaluation& ehm = stencil[3];
    double energy_grad = 0.0;
    for (int i = 0; i < n; ++i) {
      energy_grad += rpmd_ja_reference_math::central_difference_4th(
        ep.energy[i], em.energy[i], ehp.energy[i], ehm.energy[i], fd_step);
    }
    const double grad_error = energy_grad + reference.force[coordinate];
    energy_gradient_diff2 += grad_error * grad_error;
    energy_gradient_scale2 += reference.force[coordinate] * reference.force[coordinate] + energy_grad * energy_grad;
    for (int r = 0; r < d; ++r) {
      k_fine[r] = rpmd_ja_reference_math::central_difference_4th(
        gradient_stencil[0][r], gradient_stencil[1][r], gradient_stencil[2][r], gradient_stencil[3][r], fd_step);
      k_coarse[r] = rpmd_ja_reference_math::central_difference_2nd(
        gradient_stencil[0][r], gradient_stencil[1][r], fd_step);
      const double difference = 3.0 * (k_fine[r] - k_coarse[r]);
      k_diff2 += difference * difference;
      k_fine2 += k_fine[r] * k_fine[r];
    }
    for (int probe = 0; probe < 3; ++probe) {
      double output_projection = 0.0;
      for (int r = 0; r < d; ++r) output_projection += k_fine[r] * probe_directions[probe][r];
      matrix_quadratic[probe] += probe_directions[probe][coordinate] * output_projection;
    }
    for (int alpha = 0; alpha < 3; ++alpha) {
      for (int r = 0; r < d; ++r) {
        const int site = r % n, mu = r / n;
        const int offset = virial_component[alpha][mu] * n + site;
        c_fine[alpha][r] = rpmd_ja_reference_math::central_difference_4th(
          ep.virial[offset], em.virial[offset], ehp.virial[offset], ehm.virial[offset], fd_step);
        c_coarse[alpha][r] = rpmd_ja_reference_math::central_difference_2nd(
          ep.virial[offset], em.virial[offset], fd_step);
        const double difference = 3.0 * (c_fine[alpha][r] - c_coarse[alpha][r]);
        c_diff2[alpha] += difference * difference;
        c_fine2[alpha] += c_fine[alpha][r] * c_fine[alpha][r];
      }
      const std::streampos c_start = data_start + v_bytes +
        static_cast<std::streamoff>(alpha) * d * d * sizeof(double);
      save_row(c_start, coordinate, d, c_fine[alpha]);
      const std::streampos coarse_c_matrix_start = coarse_c_start +
        static_cast<std::streamoff>(alpha) * d * d * sizeof(double);
      save_row(coarse_c_matrix_start, coordinate, d, c_coarse[alpha]);
    }
    save_row(data_start + v_bytes + c_bytes, coordinate, d, k_fine);
    const std::uint64_t completed = static_cast<std::uint64_t>(d) + coordinate + 1;
    if (next_decile <= 10 && completed * 10 >= next_decile * static_cast<std::uint64_t>(2) * d) {
      while (next_decile <= 10 && completed * 10 >= next_decile * static_cast<std::uint64_t>(2) * d) ++next_decile;
      const int percent = static_cast<int>(std::min<std::uint64_t>(10 * (next_decile - 1), 100));
      std::printf("    qNEP rpmd_ja phase K/C column %d/%d (%d%% overall)\n", coordinate + 1, d, percent);
      std::fflush(stdout);
    }
  }
  const double kc_phase_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - kc_phase_start).count();

  const double scalar_energy_gradient_error = std::sqrt(energy_gradient_diff2 / std::max(energy_gradient_scale2, 1.0e-300));
  const double k_relative = std::sqrt(k_diff2 / std::max(k_fine2, 1.0e-300));
  const double c_relative[3] = {
    std::sqrt(c_diff2[0] / std::max(c_fine2[0], 1.0e-300)),
    std::sqrt(c_diff2[1] / std::max(c_fine2[1], 1.0e-300)),
    std::sqrt(c_diff2[2] / std::max(c_fine2[2], 1.0e-300))};
  auto total_energy = [](const QEvaluation& evaluation) {
    double value = 0.0;
    for (double site_energy : evaluation.energy) value += site_energy;
    return value;
  };
  double second_consistency = 0.0, second_convergence = 0.0;
  double scalar_energy_curvature_error = 0.0;
  for (int probe = 0; probe < 3; ++probe) {
    double max_component = 0.0;
    for (double value : probe_directions[probe]) max_component = std::max(max_component, std::abs(value));
    const double probe_step = fd_step / max_component;
    const auto position_at = [&](const double scale) {
      std::vector<double> value = positions;
      for (int coordinate = 0; coordinate < d; ++coordinate)
        value[coordinate] += scale * probe_step * probe_directions[probe][coordinate];
      return value;
    };
    const QEvaluation probe_plus = evaluator.evaluate(position_at(1.0));
    const std::vector<double> gradient_plus = evaluator.analytic_gradient();
    const QEvaluation probe_minus = evaluator.evaluate(position_at(-1.0));
    const std::vector<double> gradient_minus = evaluator.analytic_gradient();
    const QEvaluation probe_twice_plus = evaluator.evaluate(position_at(2.0));
    const std::vector<double> gradient_twice_plus = evaluator.analytic_gradient();
    const QEvaluation probe_twice_minus = evaluator.evaluate(position_at(-2.0));
    const std::vector<double> gradient_twice_minus = evaluator.analytic_gradient();
    const auto gradient_direction = [&](const std::vector<double>& gradient) {
      double value = 0.0;
      for (int coordinate = 0; coordinate < d; ++coordinate)
        value += probe_directions[probe][coordinate] * gradient[coordinate];
      return value;
    };
    const double energy_h = (total_energy(probe_plus) + total_energy(probe_minus) - 2.0 * energy0) /
      (probe_step * probe_step);
    const double gradient_h2 = rpmd_ja_reference_math::central_difference_2nd(
      gradient_direction(gradient_plus), gradient_direction(gradient_minus), probe_step);
    const double gradient_d4 = rpmd_ja_reference_math::central_difference_4th(
      gradient_direction(gradient_plus), gradient_direction(gradient_minus),
      gradient_direction(gradient_twice_plus), gradient_direction(gradient_twice_minus), probe_step);
    const double energy_h2 = (total_energy(probe_twice_plus) + total_energy(probe_twice_minus) - 2.0 * energy0) /
      (4.0 * probe_step * probe_step);
    const double gradient_scale = std::max({std::abs(gradient_d4), std::abs(gradient_h2), std::abs(matrix_quadratic[probe]), 1.0e-12});
    second_consistency = std::max(second_consistency, std::abs(gradient_d4 - matrix_quadratic[probe]) / gradient_scale);
    second_convergence = std::max(second_convergence, 3.0 * std::abs(gradient_d4 - gradient_h2) /
      std::max({std::abs(gradient_d4), std::abs(gradient_h2), 1.0e-12}));
    scalar_energy_curvature_error = std::max(scalar_energy_curvature_error,
      std::abs(energy_h - energy_h2) / std::max({std::abs(energy_h), std::abs(energy_h2), 1.0e-12}));
  }
  double stats[18] = {};
  stats[0] = native_gradient_relative; stats[1] = native_gradient_abs;
  stats[2] = jvp_identity_relative; stats[3] = jvp_identity_abs;
  stats[4] = k_relative; stats[5] = std::sqrt(k_diff2 / (static_cast<double>(d) * d));
  for (int alpha = 0; alpha < 3; ++alpha) {
    stats[6 + alpha] = c_relative[alpha];
    stats[9 + alpha] = std::sqrt(c_diff2[alpha] / (static_cast<double>(d) * d));
  }
  stats[12] = max_force; stats[13] = std::sqrt(force_norm2 / d);
  stats[14] = second_consistency; stats[15] = second_convergence;
  stats[16] = fd_step; stats[17] = 3.0; // Analytic site JVP and fourth-order gradient/virial derivatives.
  out.seekp(stats_start);
  out.write(reinterpret_cast<const char*>(stats), sizeof(stats));
  out.flush();
  if (!out) throw std::runtime_error("failed writing qNEP rpmd_ja raw diagnostic footer");
  out.close();
  if (!out) throw std::runtime_error("failed closing qNEP rpmd_ja raw file");
  if (std::rename(temporary.c_str(), path.c_str()) != 0)
    throw std::runtime_error("cannot finalize qNEP rpmd_ja raw file");
  remove_temporary.active = false;
  std::printf("    qNEP rpmd_ja raw v3: maximum force %.3e eV/A; analytic-gradient/native-force abs RMS %.3e; site-JVP/gradient identity abs RMS %.3e; V analytic, coarse V repeats it; K uses fourth-order analytic-gradient FD; C uses fourth-order native nine-component virial FD.\n",
    max_force, native_gradient_abs, jvp_identity_abs);
  std::printf("    qNEP rpmd_ja raw checks: K/C stencil-step consistency D2(h)-D2(2h), normalized by D4: K %.3e; Cxyz %.3e %.3e %.3e; directional gradient D2(h)-D2(2h) %.3e; scalar energy-gradient FD %.3e and energy-curvature h/2h %.3e are diagnostic only.\n",
    k_relative, c_relative[0], c_relative[1], c_relative[2], second_convergence, scalar_energy_gradient_error, scalar_energy_curvature_error);
  std::printf("    qNEP rpmd_ja work: serial force evaluations %llu, gradient-only calls %llu, site JVPs %d; matrix storage %.3f GiB; tracked QEvaluator GPU buffers %.3f MiB (excludes NEP/PPPM private and host buffers).\n",
    static_cast<unsigned long long>(4ULL * d + 4ULL * precheck_coordinates.size() + 14ULL),
    static_cast<unsigned long long>(4ULL * d + 4ULL * precheck_coordinates.size() + 14ULL),
    d + static_cast<int>(precheck_coordinates.size()),
    static_cast<double>(2 * v_bytes + 2 * c_bytes + k_bytes) / (1024.0 * 1024.0 * 1024.0),
    static_cast<double>(23ULL * n * sizeof(double) + static_cast<std::uint64_t>(n) * sizeof(int)) /
      (1024.0 * 1024.0));
  const double elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - generation_start).count();
  std::printf("    qNEP rpmd_ja raw phases: V %.3f s; K/C %.3f s\n", v_phase_seconds, kc_phase_seconds);
  std::printf("    qNEP rpmd_ja raw generation elapsed %.3f s\n", elapsed_seconds);
  const bool finite_diagnostics = std::isfinite(native_gradient_abs) && std::isfinite(jvp_identity_abs) &&
    std::isfinite(scalar_energy_gradient_error) && std::isfinite(scalar_energy_curvature_error) &&
    std::isfinite(k_relative) && std::isfinite(c_relative[0]) && std::isfinite(c_relative[1]) &&
    std::isfinite(c_relative[2]) && std::isfinite(second_consistency) && std::isfinite(second_convergence);
  if (!finite_diagnostics || native_gradient_abs > kForceTolerance || jvp_identity_abs > kForceTolerance ||
      k_relative > kDifferenceTolerance || second_consistency > kDifferenceTolerance ||
      second_convergence > kDifferenceTolerance || c_relative[0] > kDifferenceTolerance ||
      c_relative[1] > kDifferenceTolerance || c_relative[2] > kDifferenceTolerance) {
    std::fflush(stdout);
    std::fprintf(stderr,
      "qNEP rpmd_ja raw v3 stencil-step checks failed for %s at fd_step=%.9g: analytic-gradient/native-force abs RMS %.6g, site-JVP/gradient identity abs RMS %.6g (limit %.3g), K D2(h)-D2(2h) %.6g, Cxyz %.6g %.6g %.6g, gradient-curvature D4/matrix consistency %.6g, gradient-curvature D2(h)-D2(2h) %.6g (limits %.3g)\n",
      path.c_str(), fd_step, native_gradient_abs, jvp_identity_abs, kForceTolerance, k_relative,
      c_relative[0], c_relative[1], c_relative[2], second_consistency, second_convergence, kDifferenceTolerance);
    throw std::runtime_error("qNEP rpmd_ja raw stencil-step finite-difference consistency check failed; inspect recorded diagnostics");
  }
}

void generate_rpmd_ja_qnep_raw(
  const std::string& raw_path,
  const double temperature,
  const double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force,
  const bool require_zero_net_gradient)
{
  generate_rpmd_ja_qnep_raw_reference(raw_path, temperature, fd_step, kernel_table_path, atom, box, force,
    require_zero_net_gradient);
}

void generate_rpmd_ja_qnep_reference(
  const std::string& path,
  const double temperature,
  const double fd_step,
  const std::string& kernel_table_path,
  Atom& atom,
  Box& box,
  Force& force)
{
  const std::string raw_path = path + ".qraw";
  const std::string sidecar_path = path + ".stability";
  std::ifstream existing(path, std::ios::binary), existing_sidecar(sidecar_path), existing_raw(raw_path, std::ios::binary);
  if (existing.good() || existing_sidecar.good() || existing_raw.good())
    throw std::runtime_error("qNEP rpmd_ja final output, sidecar, or scratch file already exists");
  generate_rpmd_ja_qnep_raw_reference(
    raw_path, temperature, fd_step, kernel_table_path, atom, box, force, false);
  prepare_rpmd_ja_qnep_reference(raw_path, path, kernel_table_path, make_rpmd_ja_qnep_mode_validator(atom, box, force));
  if (std::remove(raw_path.c_str()) != 0)
    throw std::runtime_error("qNEP rpmd_ja finalized but could not remove raw scratch file: " + raw_path);
}

void diagnose_rpmd_ja_qnep_reference(const double fd_step, Atom& atom, Box& box, Force& force, const bool full)
{
  const auto oracle_reason_name = [](const PPPMReferenceTranslationReason reason) {
    switch (reason) {
    case PPPMReferenceTranslationReason::none: return "none";
    case PPPMReferenceTranslationReason::invalid_frame: return "invalid_frame";
    case PPPMReferenceTranslationReason::unsupported_box: return "unsupported_box";
    case PPPMReferenceTranslationReason::unsupported_mesh: return "unsupported_mesh";
    case PPPMReferenceTranslationReason::insufficient_memory: return "insufficient_memory";
    case PPPMReferenceTranslationReason::invalid_precision_target: return "invalid_precision_target";
    case PPPMReferenceTranslationReason::nonfinite_input: return "nonfinite_input";
    case PPPMReferenceTranslationReason::coefficient_copy_failed: return "coefficient_copy_failed";
    case PPPMReferenceTranslationReason::fft_plan_failed: return "fft_plan_failed";
    case PPPMReferenceTranslationReason::fft_execute_failed: return "fft_execute_failed";
    }
    return "unknown";
  };
  if (!(fd_step > 0.0) || !std::isfinite(fd_step))
    throw std::invalid_argument("qNEP rpmd_ja diagnose requires a positive finite fd_step");
  if (atom.number_of_atoms <= 1 || atom.cpu_type.size() != static_cast<std::size_t>(atom.number_of_atoms) ||
      atom.cpu_mass.size() != static_cast<std::size_t>(atom.number_of_atoms))
    throw std::runtime_error("qNEP rpmd_ja diagnose requires initialized atom count and types");
  if (force.get_number_of_potentials() != 1 || force.primary_nep_model_path().empty())
    throw std::runtime_error("qNEP rpmd_ja diagnose requires exactly one qNEP potential");
  auto* active_qnep = dynamic_cast<NEP_Charge*>(&force.get_potential(0));
  if (active_qnep == nullptr || (active_qnep->get_charge_mode() != 1 && active_qnep->get_charge_mode() != 2) ||
      !active_qnep->uses_pppm())
    throw std::runtime_error("qNEP rpmd_ja diagnose supports charge mode 1 or 2 with PPPM only");
  if (box.pbc_x != 1 || box.pbc_y != 1 || box.pbc_z != 1)
    throw std::runtime_error("qNEP rpmd_ja diagnose requires a fully periodic fixed cell");

  const int n = atom.number_of_atoms, d = reference_dimension(atom.number_of_atoms);
  const char* axis_name[3] = {"x", "y", "z"};
  std::vector<double> positions(static_cast<std::size_t>(d));
  atom.position_per_atom.copy_to_host(positions.data());
  require_finite(positions, "qNEP diagnosis positions");
  for (double value : box.cpu_h)
    if (!std::isfinite(value)) throw std::runtime_error("qNEP rpmd_ja diagnose requires a finite cell");
  std::map<int, int> first_atom_by_type;
  for (int i = 0; i < n; ++i) first_atom_by_type.emplace(atom.cpu_type[i], i);
  std::vector<int> coordinates;
  for (int i = 0; i < n; ++i)
    if (first_atom_by_type.at(atom.cpu_type[i]) == i)
      for (int alpha = 0; alpha < 3; ++alpha) coordinates.push_back(alpha * n + i);

  const auto start = std::chrono::steady_clock::now();
  Box observer_box = box;
  observer_box.get_inverse();
  observer_box.set_is_orthogonal();
  if (full) std::printf("[mesh initialization: active target %.6g A; external model initialization output follows]\n",
    active_qnep->get_pppm_mesh_spacing());
  QEvaluator evaluator(force.primary_nep_model_path(), n, force.get_run_input(), observer_box,
    atom.cpu_type, active_qnep->get_pppm_mesh_spacing());
  if (full) std::printf("[1/4] reference/net + exact V\n");
  std::printf("rpmd_ja diagnose only: qNEP charge mode %d, PPPM mesh spacing %.6g A, N=%d, sampled coordinates=%zu; no reference will be produced.\n",
    active_qnep->get_charge_mode(), active_qnep->get_pppm_mesh_spacing(), n, coordinates.size());
  if (!full) {
    std::printf("  This quick diagnose does not evaluate the exact full-column V sum; uniform-translation JVPs below are directional proxies only.\n");
  } else {
    std::printf("  Full includes q/charge, real-space, and PPPM terms; shortmode=diagnostic label excludes electrostatics. Their error difference does not isolate PPPM.\n");
    std::printf("  Displayed thresholds: gradient abs RMS %.3g eV/A, K/C relative %.3g; sampled precheck PASS is not reference acceptance.\n",
      kForceTolerance, kDifferenceTolerance);
  }

  const auto total_energy = [](const QEvaluation& evaluation) {
    double total = 0.0;
    for (double value : evaluation.energy) total += value;
    return total;
  };
  const int virial_component[3][3] = {{0, 3, 4}, {6, 1, 5}, {7, 8, 2}};
  std::uint64_t evaluations = 0;
  const bool include_electro[2] = {true, false};
  const char* mode_name[2] = {"full-qNEP", "short-range/no-electrostatic"};
  PPPMReferenceTranslationReport active_translation_oracle;
  bool active_translation_oracle_frame = false;
  double active_pppm_explicit_xyz[3] = {}, active_pppm_ik_xyz[3] = {}, active_native_force_xyz[3] = {};
  double active_uniform_jvp_xyz[3] = {};
  double active_full_v_net_norm = 0.0, active_full_v_limit = 0.0, active_identity_rms = 0.0;
  bool active_full_v_within_limit = false;
  const auto print_translation_sources = [&](const char* block, const PPPMReferenceTranslationReport& report,
                                               const bool frame_valid, const double explicit_xyz[3],
                                               const double site_jvp_xyz[3], const double ik_xyz[3],
                                               const double native_xyz[3]) {
    std::printf("  %s translation oracle (eV/A); A=frozen float coefficients, B=exact W/frozen G, C=exact W/FP64 evaluation of production G formula (not true Ewald).\n", block);
    std::printf("    FD validation: mesh axis phase model AD FD |AD-FD| FD-unc roundoff limit result; source FD steps fixed at 0.01/0.005/0.0025 A (input h controls sampled stencil checks).\n");
    if (frame_valid && report.source_log_confirmation) {
      std::printf("    oracle values: max|G-even error|=%.3e G0=%.3e charge-closure-error=%.3e.\n",
        report.max_even_G_error, report.mesh_zero_mode_value, report.assignment_charge_error);
      std::printf("    axis integer-grid energy delta (eV) / assignment-cycle error (charge) / half-grid energy delta (eV):\n");
      for (int axis = 0; axis < 3; ++axis) {
        const auto& a = report.axis[axis];
        std::printf("      %c % .3e / % .3e / % .3e\n", "xyz"[axis], a.integer_shift_energy_error,
          a.integer_shift_assignment_error, a.half_shift_energy_change);
      }
    } else {
      std::printf("    oracle values: N/A (frame or source log not confirmed); max|G-even error| G0 charge-closure-error=N/A.\n");
      std::printf("    axis integer-grid energy delta (eV) / assignment-cycle error (charge) / half-grid energy delta (eV): N/A\n");
    }
    if (!frame_valid || !report.source_log_confirmation || !report.mesh_invariant_pass) {
      std::printf("    source stages NOT_RUN/INCONCLUSIVE (frame=%s reason=%s shared-mesh-invariants=%s).\n",
        frame_valid ? "valid" : "invalid", oracle_reason_name(report.reason),
        report.mesh_invariant_pass ? "PASS" : report.source_log_confirmation ? "FAIL" : "NOT_RUN");
      return;
    }
    for (int axis = 0; axis < 3; ++axis) {
      const auto& a = report.axis[axis];
      for (int phase = 0; phase < 2; ++phase) {
        for (int model = 0; model < 3; ++model) {
          const auto& p = a.source[model].phase[phase];
          const char* state = !p.valid || p.fd_uncertainty + p.roundoff > report.precision_target
            ? "INCONCLUSIVE" : p.pass ? "PASS" : "MISMATCH";
          std::printf("    %-7s %c %-5s %c ", block, "xyz"[axis], phase == 0 ? "orig" : "half", "ABC"[model]);
          if (!p.valid) std::printf("%10s %10s %10s %10s %10s %10.2e %s\n", "N/A", "N/A", "N/A", "N/A", "N/A", report.precision_target, state);
          else std::printf("%10.3e %10.3e %10.3e %10.3e %10.3e %10.2e %s\n",
            p.analytic_derivative, p.fd_derivative, std::abs(p.analytic_fd_difference),
            p.fd_uncertainty, p.roundoff, report.precision_target, state);
        }
      }
    }
    std::printf("    Signed contributions: mesh axis phase native-A(host/FP32 proxy) A-B B-C C-residual(resolved only)\n");
    for (int axis = 0; axis < 3; ++axis) for (int phase = 0; phase < 2; ++phase) {
      const auto& a = report.axis[axis];
      const auto& A = a.source[0].phase[phase];
      const auto& B = a.source[1].phase[phase];
      const auto& C = a.source[2].phase[phase];
      std::printf("    %-7s %c %-5s ", block, "xyz"[axis], phase == 0 ? "orig" : "half");
      if (phase == 0 && A.valid) std::printf("%10.3e", a.native_energy_derivative - A.analytic_derivative); else std::printf("%10s", "N/A");
      if (A.valid && B.valid) std::printf(" %10.3e", A.analytic_derivative - B.analytic_derivative); else std::printf(" %10s", "N/A");
      if (B.valid && C.valid) std::printf(" %10.3e", B.analytic_derivative - C.analytic_derivative); else std::printf(" %10s", "N/A");
      if (C.valid) std::printf(" %10.3e ", C.analytic_derivative); else std::printf(" %10s ", "N/A");
      if (C.valid && C.pass && std::abs(C.analytic_derivative) > report.precision_target +
          3.0 * (C.fd_uncertainty + C.roundoff + C.analytic_fd_difference))
        std::printf("NONZERO_ABOVE_LIMIT\n");
      else if (C.valid && C.pass) std::printf("BELOW_RESOLUTION_OR_LIMIT\n");
      else std::printf("INCONCLUSIVE\n");
    }
    std::printf("    GPU tangent components (original phase only; eV/A): mesh axis explicit-gradient site-JVP IK native-force\n");
    for (int axis = 0; axis < 3; ++axis)
      std::printf("    %-7s %c %12.4e %12.4e %12.4e %12.4e\n", block, "xyz"[axis],
        explicit_xyz[axis], site_jvp_xyz[axis], ik_xyz[axis], native_xyz[axis]);
  };
  std::vector<double> full_r0_gradient;
  double r0_gradient_native_abs = 0.0, r0_site_jvp_abs = 0.0;
  bool r0_jvp_pass = false;
  int r0_jvp_worst_coordinate = 0;
  int precheck_steps_passed = 0, precheck_steps_total = 0;
  bool requested_h_seen = false, requested_h_pass = false;
  for (int mode = 0; mode < 2; ++mode) {
    std::array<QEvaluation, 3> repeated_reference;
    for (int repeat = 0; repeat < 3; ++repeat) {
      repeated_reference[repeat] = evaluator.evaluate(positions, include_electro[mode]);
      ++evaluations;
    }
    const QEvaluation& reference = repeated_reference[2];
    const double energy0 = total_energy(reference);
    double energy_min = energy0, energy_max = energy0;
    double force_repeat_diff2 = 0.0, virial_repeat_diff2 = 0.0;
    for (int repeat = 0; repeat < 2; ++repeat) {
      const double energy = total_energy(repeated_reference[repeat]);
      energy_min = std::min(energy_min, energy);
      energy_max = std::max(energy_max, energy);
      for (std::size_t i = 0; i < reference.force.size(); ++i) {
        const double delta = repeated_reference[repeat].force[i] - reference.force[i];
        force_repeat_diff2 += delta * delta;
      }
      for (std::size_t i = 0; i < reference.virial.size(); ++i) {
        const double delta = repeated_reference[repeat].virial[i] - reference.virial[i];
        virial_repeat_diff2 += delta * delta;
      }
    }
    const double force_repeat_rms = std::sqrt(force_repeat_diff2 / (2.0 * reference.force.size()));
    const double virial_repeat_rms = std::sqrt(virial_repeat_diff2 / (2.0 * reference.virial.size()));
    std::printf("  %s R0 repeatability: total-E span %.6g eV, force-difference RMS %.6g eV/A, 9W-difference RMS %.6g eV\n",
      mode_name[mode], energy_max - energy_min, force_repeat_rms, virial_repeat_rms);

    if (mode == 0) {
      GPU_Vector<double> direction(3 * n), site_jvp(n), analytic_gradient(3 * n);
      direction.fill(0.0);
      if (!evaluator.qnep.compute_reference_site_energy_derivative(
            observer_box, evaluator.type, evaluator.position, evaluator.force, nullptr, nullptr,
            analytic_gradient))
        throw std::runtime_error("qNEP rpmd_ja analytic gradient rejected the current PPPM force frame");
      std::vector<double> host_gradient(static_cast<std::size_t>(3) * n);
      analytic_gradient.copy_to_host(host_gradient.data());
      require_finite(host_gradient, "qNEP diagnostic analytic gradient");
      full_r0_gradient = host_gradient;
      if (full) {
        std::vector<double> full_v_gradient(static_cast<std::size_t>(d), 0.0);
        std::vector<double> axis_site_columns(static_cast<std::size_t>(d), 0.0);
        std::vector<double> direction_host(static_cast<std::size_t>(d), 0.0), site_values(n);
        GPU_Vector<double> full_direction(d), full_site_jvp(n), full_gradient(d);
        double gradient_diff2 = 0.0, gradient_max = 0.0;
        double component_diff2 = 0.0, component_max = 0.0;
        double ordered_compensated_diff2 = 0.0, ordered_extended_diff2 = 0.0;
        double ordered_compensated_max = 0.0, ordered_extended_max = 0.0;
        double pppm_ik_xyz[3] = {}, pppm_explicit_xyz[3] = {}, native_force_xyz[3] = {};
        double local_xyz[3] = {}, real_xyz[3] = {}, pppm_site_xyz[3] = {};
        QNEPReferenceTangentReduction raw_charge_qdot[3], projected_charge_qdot[3];
        for (int coordinate = 0; coordinate < d; ++coordinate) {
          direction_host[coordinate] = 1.0;
          site_values = evaluator.analytic_site_jvp(direction_host);
          require_finite(site_values, "qNEP full diagnostic site JVP");
          double ordered = 0.0, compensated_running_sum = 0.0, compensation = 0.0;
          long double extended = 0.0L;
          for (int site = 0; site < n; ++site) {
            const double value = site_values[site];
            ordered += value;
            const double next = compensated_running_sum + value;
            compensation += std::abs(compensated_running_sum) >= std::abs(value)
              ? (compensated_running_sum - next) + value : (value - next) + compensated_running_sum;
            compensated_running_sum = next;
            extended += static_cast<long double>(value);
            axis_site_columns[(coordinate / n) * n + site] += value;
          }
          const double compensated = compensated_running_sum + compensation;
          const double extended_double = static_cast<double>(extended);
          const double compensated_error = ordered - compensated;
          const double extended_error = ordered - extended_double;
          ordered_compensated_diff2 += compensated_error * compensated_error;
          ordered_extended_diff2 += extended_error * extended_error;
          ordered_compensated_max = std::max(ordered_compensated_max, std::abs(compensated_error));
          ordered_extended_max = std::max(ordered_extended_max, std::abs(extended_error));
          full_v_gradient[coordinate] = ordered;
          const double gradient_error = ordered - host_gradient[coordinate];
          gradient_diff2 += gradient_error * gradient_error;
          gradient_max = std::max(gradient_max, std::abs(gradient_error));
          direction_host[coordinate] = 0.0;
          if ((coordinate + 1) % std::max(1, d / 10) == 0 || coordinate + 1 == d) {
            const int percent = static_cast<int>(100LL * (coordinate + 1) / d);
            std::printf("    full V columns %d/%d (%d%%)\n", coordinate + 1, d, percent);
            std::fflush(stdout);
          }
        }
        double uniform_diff2 = 0.0, uniform_max = 0.0;
        double uniform_net[3] = {};
        std::vector<double> uniform_direction(static_cast<std::size_t>(d), 0.0);
        for (int axis = 0; axis < 3; ++axis) {
          for (int i = 0; i < n; ++i) uniform_direction[axis * n + i] = 1.0;
          full_direction.copy_from_host(uniform_direction.data());
          QNEPReferenceTangentDiagnostics tangent;
          if (!evaluator.qnep.compute_reference_site_energy_derivative(
                observer_box, evaluator.type, evaluator.position, evaluator.force, &full_direction,
                &full_site_jvp, full_gradient, &tangent) || !tangent.valid)
            throw std::runtime_error("qNEP full diagnose uniform tangent components rejected the current PPPM force frame");
          if (tangent.local_site_tangent.size() != static_cast<std::size_t>(n) ||
              tangent.real_site_tangent.size() != static_cast<std::size_t>(n) ||
              tangent.pppm_site_tangent.size() != static_cast<std::size_t>(n))
            throw std::runtime_error("qNEP full diagnose returned incomplete uniform tangent components");
          full_site_jvp.copy_to_host(site_values.data());
          const std::vector<double>& direct = site_values;
          raw_charge_qdot[axis] = tangent.raw_charge_direction;
          projected_charge_qdot[axis] = tangent.projected_charge_direction;
          pppm_ik_xyz[axis] = tangent.pppm_ik_force_sum[axis];
          pppm_explicit_xyz[axis] = tangent.pppm_explicit_gradient_sum[axis];
          native_force_xyz[axis] = tangent.native_force_sum[axis];
          for (int i = 0; i < n; ++i) {
            local_xyz[axis] += tangent.local_site_tangent[i];
            real_xyz[axis] += tangent.real_site_tangent[i];
            pppm_site_xyz[axis] += tangent.pppm_site_tangent[i];
            const double component = tangent.local_site_tangent[i] + tangent.real_site_tangent[i] +
              tangent.pppm_site_tangent[i];
            const double component_error = component - direct[i];
            component_diff2 += component_error * component_error;
            component_max = std::max(component_max, std::abs(component_error));
          }
          uniform_net[axis] = std::accumulate(direct.begin(), direct.end(), 0.0);
          for (int i = 0; i < n; ++i) {
            const double error = axis_site_columns[axis * n + i] - direct[i];
            uniform_diff2 += error * error;
            uniform_max = std::max(uniform_max, std::abs(error));
          }
          for (int i = 0; i < n; ++i) uniform_direction[axis * n + i] = 0.0;
        }
        const auto full_net = rpmd_ja_reference_math::net_force_stats(full_v_gradient, n);
        active_full_v_net_norm = full_net.net_norm;
        active_full_v_limit = full_net.limit;
        active_full_v_within_limit = full_net.within_limit;
        PPPMReferenceTranslationReport oracle;
        const bool oracle_frame_valid = evaluator.qnep.diagnose_reference_translation_energy(
          observer_box, evaluator.type, evaluator.position, evaluator.force, oracle, full_net.limit);
        if (!oracle_frame_valid) {
          oracle.status = PPPMReferenceTranslationStatus::inconclusive;
          oracle.reason = PPPMReferenceTranslationReason::invalid_frame;
        }
        active_translation_oracle = oracle;
        active_translation_oracle_frame = oracle_frame_valid;
        for (int axis = 0; axis < 3; ++axis) {
          active_pppm_explicit_xyz[axis] = pppm_explicit_xyz[axis];
          active_pppm_ik_xyz[axis] = pppm_ik_xyz[axis];
          active_native_force_xyz[axis] = native_force_xyz[axis];
          active_uniform_jvp_xyz[axis] = uniform_net[axis];
        }
        const double gradient_rms = std::sqrt(gradient_diff2 / d);
        const double component_rms = std::sqrt(component_diff2 / d);
        const double uniform_rms = std::sqrt(uniform_diff2 / d);
        active_identity_rms = uniform_rms;
        std::printf("  CODE_INVARIANT: full V column sum vs baseline analytic gradient RMS %.9g max %.9g eV/A (%s); local+real+PPPM uniform-site decomposition RMS %.9g max %.9g (%s).\n",
          gradient_rms, gradient_max, gradient_rms <= kForceTolerance ? "PASS" : "FAIL",
          component_rms, component_max, component_rms <= kForceTolerance ? "PASS" : "FAIL");
        std::printf("  CODE_INVARIANT: axis column-site sums vs direct uniform JVP RMS %.9g max %.9g eV/A; direct net xyz=(%.9g, %.9g, %.9g). Shared algebraic identities are not independent oracle proof.\n",
          uniform_rms, uniform_max,
          uniform_net[0], uniform_net[1], uniform_net[2]);
        std::printf("  PRECISION reductions (RMS/max eV/A): ordered-Neumaier %.3e/%.3e; ordered-long-double(%s) %.3e/%.3e.\n",
          std::sqrt(ordered_compensated_diff2 / d), ordered_compensated_max,
          std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits ? "extended" : "double",
          std::sqrt(ordered_extended_diff2 / d), ordered_extended_max);
        std::printf("  signed tangent components (energy columns eV/A; qdot columns charge/A): axis local real PPPM direct site-res -F+IK+explicit force-res raw-qdot(sum/max/RMS) projected-qdot(sum/max/RMS)\n");
        for (int axis = 0; axis < 3; ++axis) {
          const double reconstructed = -native_force_xyz[axis] + pppm_ik_xyz[axis] + pppm_explicit_xyz[axis];
          const auto& raw = raw_charge_qdot[axis];
          const auto& projected = projected_charge_qdot[axis];
          std::printf("    %c %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e/%.3e/%.3e %.3e/%.3e/%.3e\n",
            "xyz"[axis], local_xyz[axis], real_xyz[axis], pppm_site_xyz[axis], uniform_net[axis],
            local_xyz[axis] + real_xyz[axis] + pppm_site_xyz[axis] - uniform_net[axis],
            reconstructed, reconstructed - uniform_net[axis],
            raw.ordered_sum, raw.max_abs, raw.rms, projected.ordered_sum, projected.max_abs, projected.rms);
        }
        std::printf("  EXACT_V_NET: xyz=(%.9g, %.9g, %.9g) norm=%.9g vector_norm=%.9g limit=%.9g status=%s.\n",
          full_net.net[0], full_net.net[1], full_net.net[2], full_net.net_norm, full_net.vector_norm,
          full_net.limit, full_net.within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE");
        std::fflush(stdout);
      }
      const auto gradient_net = rpmd_ja_reference_math::net_force_stats(host_gradient, n);
      const auto native_net = rpmd_ja_reference_math::net_force_stats(reference.force, n);
      std::array<rpmd_ja_reference_math::NetForceStats, 3> repeated_gradient_net;
      repeated_gradient_net[0] = gradient_net;
      for (int repeat = 1; repeat < 3; ++repeat)
        repeated_gradient_net[repeat] = rpmd_ja_reference_math::net_force_stats(evaluator.analytic_gradient(), n);
      double total_mass = 0.0;
      for (double mass : atom.cpu_mass) {
        if (!(mass > 0.0) || !std::isfinite(mass)) throw std::runtime_error("qNEP rpmd_ja diagnose requires positive finite masses");
        total_mass += mass;
      }
      if (!(total_mass > 0.0) || !std::isfinite(total_mass))
        throw std::runtime_error("qNEP rpmd_ja diagnose total mass is not finite and positive");
      const double sqrt_total_mass = std::sqrt(total_mass);
      std::array<double, 3> uniform_jvp_net = {}, mass_translation_jvp_net = {};
      std::vector<double> translation_direction(static_cast<std::size_t>(d), 0.0);
      for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < n; ++i) translation_direction[axis * n + i] = 1.0;
        const auto uniform_site_jvp = evaluator.analytic_site_jvp(translation_direction);
        uniform_jvp_net[axis] = std::accumulate(uniform_site_jvp.begin(), uniform_site_jvp.end(), 0.0);
        for (int i = 0; i < n; ++i) translation_direction[axis * n + i] = 1.0 / sqrt_total_mass;
        const auto mass_site_jvp = evaluator.analytic_site_jvp(translation_direction);
        mass_translation_jvp_net[axis] = sqrt_total_mass *
          std::accumulate(mass_site_jvp.begin(), mass_site_jvp.end(), 0.0);
        for (int i = 0; i < n; ++i) translation_direction[axis * n + i] = 0.0;
      }
      double jvp_analytic_diff2 = 0.0, native_gradient_net_diff2 = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        const double jvp_error = uniform_jvp_net[axis] - gradient_net.net[axis];
        const double net_force_residual = native_net.net[axis] + gradient_net.net[axis];
        jvp_analytic_diff2 += jvp_error * jvp_error;
        native_gradient_net_diff2 += net_force_residual * net_force_residual;
      }
        if (!full) std::printf("  full-qNEP net-force quick diagnostic (eV/A): native force xyz=(%.9g, %.9g, %.9g), norm=%.9g, vector_norm=%.9g, limit=%.9g, status=PROXY_%s; analytic gradient xyz=(%.9g, %.9g, %.9g), norm=%.9g, vector_norm=%.9g, limit=%.9g, status=PROXY_%s.\n",
        native_net.net[0], native_net.net[1], native_net.net[2], native_net.net_norm, native_net.vector_norm,
        native_net.limit, native_net.within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE",
        gradient_net.net[0], gradient_net.net[1], gradient_net.net[2], gradient_net.net_norm,
        gradient_net.vector_norm, gradient_net.limit,
        gradient_net.within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE");
      if (!full) std::printf("    uniform physical translation JVP direction=1 xyz net=(%.9g, %.9g, %.9g) eV/A; mass-weighted translation direction=1/sqrt(total_mass), converted back to xyz net=(%.9g, %.9g, %.9g) eV/A; JVP-analytic net difference norm=%.9g; native_F+gradient net difference norm=%.9g. These directional derivatives are proxies, not the full-column V sum.\n",
        uniform_jvp_net[0], uniform_jvp_net[1], uniform_jvp_net[2], mass_translation_jvp_net[0],
        mass_translation_jvp_net[1], mass_translation_jvp_net[2], std::sqrt(jvp_analytic_diff2),
        std::sqrt(native_gradient_net_diff2));
      for (int repeat = 0; !full && repeat < 3; ++repeat) {
        const auto repeat_net = rpmd_ja_reference_math::net_force_stats(repeated_reference[repeat].force, n);
        std::printf("    baseline repeated force net[%d] xyz=(%.9g, %.9g, %.9g), norm=%.9g eV/A.\n",
          repeat + 1, repeat_net.net[0], repeat_net.net[1], repeat_net.net[2], repeat_net.net_norm);
        const auto& repeat_gradient = repeated_gradient_net[repeat];
        std::printf("    same-force-frame analytic gradient net[%d] xyz=(%.9g, %.9g, %.9g), norm=%.9g eV/A.\n",
          repeat + 1, repeat_gradient.net[0], repeat_gradient.net[1], repeat_gradient.net[2], repeat_gradient.net_norm);
      }
      double force_gradient_diff2 = 0.0, force_gradient_scale2 = 0.0;
      double all_coordinate_diff2 = 0.0, native_force_scale2 = 0.0;
      double axis_diff2[3] = {};
      double max_abs_error = 0.0;
      int max_coordinate = 0, coordinates_over_tolerance = 0;
      double jvp_identity_diff2 = 0.0, jvp_identity_scale2 = 0.0, jvp_identity_max_abs = 0.0;
      for (int coordinate = 0; coordinate < d; ++coordinate) {
        const double error = host_gradient[coordinate] + reference.force[coordinate];
        const double abs_error = std::abs(error);
        if (!std::isfinite(error))
          throw std::runtime_error("qNEP rpmd_ja all-coordinate gradient diagnostic produced a non-finite residual");
        all_coordinate_diff2 += error * error;
        native_force_scale2 += reference.force[coordinate] * reference.force[coordinate];
        axis_diff2[coordinate / n] += error * error;
        if (abs_error > max_abs_error) {
          max_abs_error = abs_error;
          max_coordinate = coordinate;
        }
        if (abs_error > kForceTolerance) ++coordinates_over_tolerance;
      }
      const double all_coordinate_abs = std::sqrt(all_coordinate_diff2 / d);
      const double all_coordinate_relative = std::sqrt(all_coordinate_diff2 / std::max(native_force_scale2, 1.0e-300));
      const bool all_coordinate_pass = all_coordinate_abs <= kForceTolerance;
      r0_gradient_native_abs = all_coordinate_abs;
      if (!std::isfinite(all_coordinate_abs) || !std::isfinite(all_coordinate_relative))
        throw std::runtime_error("qNEP rpmd_ja all-coordinate gradient diagnostic produced non-finite RMS values");
      const int max_atom = max_coordinate % n;
      const int max_axis = max_coordinate / n;
      std::printf("  full-qNEP all-coordinate gradient check: coordinates=%d, abs RMS %.6g eV/A, relative %.6g, limit=%.1e eV/A, %s (this check only; not reference-generation acceptance).\n",
        d, all_coordinate_abs, all_coordinate_relative, kForceTolerance,
        all_coordinate_pass ? "PASS" : "FAIL");
      std::printf("    axis abs RMS x/y/z %.6g %.6g %.6g eV/A; max|e| %.6g eV/A at atom %d (0-based) axis %s type %d; coordinates |e|>tol %d.\n",
        std::sqrt(axis_diff2[0] / n), std::sqrt(axis_diff2[1] / n), std::sqrt(axis_diff2[2] / n),
        max_abs_error, max_atom, axis_name[max_axis], atom.cpu_type[max_atom], coordinates_over_tolerance);
      std::vector<double> unit_direction(static_cast<std::size_t>(3) * n, 0.0);
      std::vector<double> host_jvp(n);
      for (int coordinate : coordinates) {
        unit_direction[coordinate] = 1.0;
        direction.copy_from_host(unit_direction.data());
        if (!evaluator.qnep.compute_reference_site_energy_derivative(
              observer_box, evaluator.type, evaluator.position, evaluator.force, &direction, &site_jvp,
              analytic_gradient))
          throw std::runtime_error("qNEP rpmd_ja analytic site JVP rejected the current PPPM force frame");
        site_jvp.copy_to_host(host_jvp.data());
        require_finite(host_jvp, "qNEP diagnostic site JVP");
        double summed_jvp = 0.0;
        for (double value : host_jvp) summed_jvp += value;
        const double identity_error = summed_jvp - host_gradient[coordinate];
        if (!std::isfinite(summed_jvp) || !std::isfinite(identity_error))
          throw std::runtime_error("qNEP rpmd_ja analytic diagnostic produced a non-finite identity residual");
        jvp_identity_diff2 += identity_error * identity_error;
        jvp_identity_scale2 += host_gradient[coordinate] * host_gradient[coordinate];
        if (std::abs(identity_error) > jvp_identity_max_abs) {
          jvp_identity_max_abs = std::abs(identity_error);
          r0_jvp_worst_coordinate = coordinate;
        }
        const double force_error = host_gradient[coordinate] + reference.force[coordinate];
        force_gradient_diff2 += force_error * force_error;
        force_gradient_scale2 += host_gradient[coordinate] * host_gradient[coordinate] +
                                 reference.force[coordinate] * reference.force[coordinate];
        unit_direction[coordinate] = 0.0;
      }
      const double jvp_identity_abs = std::sqrt(jvp_identity_diff2 / coordinates.size());
      r0_site_jvp_abs = jvp_identity_abs;
      r0_jvp_pass = std::isfinite(jvp_identity_max_abs) && jvp_identity_max_abs <= kForceTolerance;
      const double jvp_identity_relative = std::sqrt(jvp_identity_diff2 / std::max(jvp_identity_scale2, 1.0e-300));
      const double force_gradient_abs = std::sqrt(force_gradient_diff2 / coordinates.size());
      const double force_gradient_relative = std::sqrt(force_gradient_diff2 / std::max(force_gradient_scale2, 1.0e-300));
      if (!std::isfinite(jvp_identity_abs) || !std::isfinite(jvp_identity_relative) ||
          !std::isfinite(force_gradient_abs) || !std::isfinite(force_gradient_relative))
        throw std::runtime_error("qNEP rpmd_ja analytic diagnostic produced non-finite RMS values");
      if (!full) std::printf("  full-qNEP analytic checks: JVP-gradient identity abs RMS %.6g, relative %.6g; sampled analytic-gradient/native-force difference RMS %.6g eV/A, relative %.6g.\n",
        jvp_identity_abs, jvp_identity_relative, force_gradient_abs, force_gradient_relative);
      if (!full) std::printf("    R0 site-JVP identity max|e| %.6g eV/A at atom %d type %d axis %s; limit %.3g %s.\n",
        jvp_identity_max_abs, r0_jvp_worst_coordinate % n,
        atom.cpu_type[r0_jvp_worst_coordinate % n], axis_name[r0_jvp_worst_coordinate / n],
        kForceTolerance, r0_jvp_pass ? "PASS" : "FAIL");
      for (const auto& entry : first_atom_by_type) {
        if (full) break;
        const int type = entry.first;
        double diff2 = 0.0, force2 = 0.0, max_error = 0.0;
        int max_coordinate = entry.second;
        int count = 0;
        for (int coordinate = 0; coordinate < d; ++coordinate) {
          const int atom_index = coordinate % n;
          if (atom.cpu_type[atom_index] != type) continue;
          const double error = host_gradient[coordinate] + reference.force[coordinate];
          diff2 += error * error; force2 += reference.force[coordinate] * reference.force[coordinate]; ++count;
          if (std::abs(error) > max_error) { max_error = std::abs(error); max_coordinate = coordinate; }
        }
        std::printf("    type %d R0 gradient/native-force: abs RMS %.6g eV/A, relative %.6g, max|e| %.6g at atom %d axis %s\n",
          type, std::sqrt(diff2 / count), std::sqrt(diff2 / std::max(force2, 1.0e-300)), max_error,
          max_coordinate % n, axis_name[max_coordinate / n]);
      }
      std::fflush(stdout);
    }

    if (full) {
      if (mode == 0) std::printf("[2/4] stencil compact/short; * marks requested fd_step\n");
      std::printf(mode == 0
        ? "  full-qNEP h table (5 rows): h* grad-native site-JVP K/Cx/Cy/Cz Vfinite gate | energyFD(abs/rel) nativeK\n"
        : "  shortmode=diagnostic label h table (5 rows): h* energyFD(abs/rel) V K Cx/Cy/Cz nativeK [no generation gate]\n");
    }
    for (const double step_scale : {0.5, 1.0, 2.0, 5.0, 10.0}) {
      const double h = fd_step * step_scale;
      if (!(h > 0.0) || !std::isfinite(h))
        throw std::invalid_argument("scaled qNEP rpmd_ja diagnosis fd_step is not finite");
      double grad_diff2 = 0.0, grad_reference2 = 0.0;
      double v_diff2 = 0.0, v_fine2 = 0.0;
      double k_diff2 = 0.0, k_fine2 = 0.0;
      double native_k_d4_norm2 = 0.0, native_k_delta2 = 0.0;
      double c_diff2[3] = {}, c_fine2[3] = {};
      double c_d4_norm2[3] = {}, c_delta2[3] = {}, c_worst_column[3] = {-1.0, -1.0, -1.0};
      int c_worst_input[3] = {}, c_worst_output[3] = {};
      double analytic_d4_norm2 = 0.0, analytic_delta2 = 0.0;
      double analytic_precheck_d4_norm2 = 0.0, analytic_precheck_delta2 = 0.0;
      double c_precheck_d4_norm2[3] = {}, c_precheck_delta2[3] = {};
      double v_precheck_d4_norm2 = 0.0, v_precheck_delta2 = 0.0;
      bool v_stencil_finite = true;
      double worst_column_relative = -1.0;
      int worst_input = 0, worst_output = 0;
      for (int coordinate : coordinates) {
        std::array<std::vector<double>, 4> xyz = {positions, positions, positions, positions};
        xyz[0][coordinate] += h; xyz[1][coordinate] -= h;
        xyz[2][coordinate] += 2.0 * h; xyz[3][coordinate] -= 2.0 * h;
        std::array<QEvaluation, 4> q;
        std::array<std::vector<double>, 4> g;
        for (int lane = 0; lane < 4; ++lane) {
          q[lane] = evaluator.evaluate(xyz[lane], include_electro[mode]);
          ++evaluations;
          if (mode == 0) g[lane] = evaluator.analytic_gradient();
        }
        const QEvaluation& ep = q[0]; const QEvaluation& em = q[1];
        const QEvaluation& ehp = q[2]; const QEvaluation& ehm = q[3];
        double energy_gradient = 0.0;
        for (int i = 0; i < n; ++i) {
          const double fine = rpmd_ja_reference_math::central_difference_2nd(ep.energy[i], em.energy[i], h);
          const double coarse = rpmd_ja_reference_math::central_difference_2nd(ehp.energy[i], ehm.energy[i], 2.0 * h);
          const double v_d4 = rpmd_ja_reference_math::central_difference_4th(
            ep.energy[i], em.energy[i], ehp.energy[i], ehm.energy[i], h);
          v_stencil_finite = v_stencil_finite && std::isfinite(v_d4) && std::isfinite(fine) &&
            std::isfinite(coarse) && std::isfinite(fine - coarse);
          const double v_precheck_delta = 3.0 * (v_d4 - fine);
          v_precheck_d4_norm2 += v_d4 * v_d4;
          v_precheck_delta2 += v_precheck_delta * v_precheck_delta;
          energy_gradient += fine;
          const double difference = fine - coarse;
          v_diff2 += difference * difference;
          v_fine2 += fine * fine;
        }
        const double gradient_error = energy_gradient + reference.force[coordinate];
        grad_diff2 += gradient_error * gradient_error;
        grad_reference2 += reference.force[coordinate] * reference.force[coordinate];
        double c_column_d4[3] = {}, c_column_delta[3] = {}, c_column_max_delta[3] = {};
        int c_column_output[3] = {};
        for (int r = 0; r < d; ++r) {
          const double fine_k = -rpmd_ja_reference_math::central_difference_2nd(ep.force[r], em.force[r], h);
          const double coarse_k = -rpmd_ja_reference_math::central_difference_2nd(ehp.force[r], ehm.force[r], 2.0 * h);
          const double k_difference = fine_k - coarse_k;
          k_diff2 += k_difference * k_difference;
          k_fine2 += fine_k * fine_k;
          const double native_k_d4 = -rpmd_ja_reference_math::central_difference_4th(
            ep.force[r], em.force[r], ehp.force[r], ehm.force[r], h);
          native_k_d4_norm2 += native_k_d4 * native_k_d4; native_k_delta2 += k_difference * k_difference;
          for (int alpha = 0; alpha < 3; ++alpha) {
            const int offset = virial_component[alpha][r / n] * n + r % n;
            const double fine_c = rpmd_ja_reference_math::central_difference_2nd(ep.virial[offset], em.virial[offset], h);
            const double coarse_c = rpmd_ja_reference_math::central_difference_2nd(ehp.virial[offset], ehm.virial[offset], 2.0 * h);
            const double c_difference = fine_c - coarse_c;
            c_diff2[alpha] += c_difference * c_difference;
            c_fine2[alpha] += fine_c * fine_c;
            const double c_d4 = rpmd_ja_reference_math::central_difference_4th(
              ep.virial[offset], em.virial[offset], ehp.virial[offset], ehm.virial[offset], h);
            c_d4_norm2[alpha] += c_d4 * c_d4; c_delta2[alpha] += c_difference * c_difference;
            if (mode == 0) {
              const double c2h = rpmd_ja_reference_math::central_difference_2nd(
                ep.virial[offset], em.virial[offset], h);
              const double precheck_delta = 3.0 * (c_d4 - c2h);
              c_precheck_d4_norm2[alpha] += c_d4 * c_d4;
              c_precheck_delta2[alpha] += precheck_delta * precheck_delta;
            }
            c_column_d4[alpha] += c_d4 * c_d4; c_column_delta[alpha] += c_difference * c_difference;
            if (std::abs(c_difference) > c_column_max_delta[alpha]) {
              c_column_max_delta[alpha] = std::abs(c_difference); c_column_output[alpha] = r;
            }
          }
        }
        for (int alpha = 0; alpha < 3; ++alpha) {
          const double column_relative = std::sqrt(c_column_delta[alpha] / std::max(c_column_d4[alpha], 1.0e-300));
          if (column_relative > c_worst_column[alpha]) {
            c_worst_column[alpha] = column_relative; c_worst_input[alpha] = coordinate;
            c_worst_output[alpha] = c_column_output[alpha];
          }
        }
        if (mode == 0) {
          double column_d4_2 = 0.0, column_delta2 = 0.0;
          for (int r = 0; r < d; ++r) {
            const double d4 = rpmd_ja_reference_math::central_difference_4th(g[0][r], g[1][r], g[2][r], g[3][r], h);
            const double d2h = rpmd_ja_reference_math::central_difference_2nd(g[0][r], g[1][r], h);
            const double d22h = rpmd_ja_reference_math::central_difference_2nd(g[2][r], g[3][r], 2.0 * h);
            const double delta = d2h - d22h;
            analytic_d4_norm2 += d4 * d4; analytic_delta2 += delta * delta;
            const double precheck_delta = 3.0 * (d4 - d2h);
            analytic_precheck_d4_norm2 += d4 * d4;
            analytic_precheck_delta2 += precheck_delta * precheck_delta;
            column_d4_2 += d4 * d4; column_delta2 += delta * delta;
          }
          const double column_relative = std::sqrt(column_delta2 / std::max(column_d4_2, 1.0e-300));
          if (column_relative > worst_column_relative) {
            worst_column_relative = column_relative; worst_input = coordinate;
            double largest_residual = -1.0;
            for (int r = 0; r < d; ++r) {
              const double delta = rpmd_ja_reference_math::central_difference_2nd(g[0][r], g[1][r], h) -
                rpmd_ja_reference_math::central_difference_2nd(g[2][r], g[3][r], 2.0 * h);
              if (std::abs(delta) > largest_residual) { largest_residual = std::abs(delta); worst_output = r; }
            }
          }
        }
      }
      const double grad_abs = std::sqrt(grad_diff2 / coordinates.size());
      const double grad_rel = std::sqrt(grad_diff2 / std::max(grad_reference2, 1.0e-300));
      const double v_relative = std::sqrt(v_diff2 / std::max(v_fine2, 1.0e-300));
      const double k_relative = std::sqrt(k_diff2 / std::max(k_fine2, 1.0e-300));
      const double c_relative[3] = {
        std::sqrt(c_diff2[0] / std::max(c_fine2[0], 1.0e-300)),
        std::sqrt(c_diff2[1] / std::max(c_fine2[1], 1.0e-300)),
        std::sqrt(c_diff2[2] / std::max(c_fine2[2], 1.0e-300))};
      if (!std::isfinite(grad_abs) || !std::isfinite(grad_rel) || !std::isfinite(v_relative) ||
          !std::isfinite(k_relative) || !std::isfinite(c_relative[0]) ||
          !std::isfinite(c_relative[1]) || !std::isfinite(c_relative[2]))
        throw std::runtime_error("qNEP rpmd_ja diagnose produced a non-finite finite-difference diagnostic");
      const double native_k_abs = std::sqrt(native_k_delta2 / (d * coordinates.size()));
      const double native_k_relative = std::sqrt(native_k_delta2 / std::max(native_k_d4_norm2, 1.0e-300));
      if (full && mode == 1)
        std::printf("    h=%.6g%s energyFD=%.3e/%.3e V=%.3e K=%.3e Cxyz=%.3e/%.3e/%.3e nativeK=%.3e diagnostic-only\n",
          h, h == fd_step ? "*" : "", grad_abs, grad_rel, v_relative, k_relative,
          c_relative[0], c_relative[1], c_relative[2], native_k_relative);
      if (!full) {
        std::printf("  %s h=%.6g A: energy-gradient abs RMS %.6g, relative to sampled |F0| norm %.6g; V %.6g K %.6g Cxyz %.6g %.6g %.6g\n",
          mode_name[mode], h, grad_abs, grad_rel, v_relative, k_relative,
          c_relative[0], c_relative[1], c_relative[2]);
        std::printf("    %s native-force K D4 norm %.6g eV/A2, D2(h)-D2(2h) abs RMS %.6g eV/A2, relative %.6g (limit %.3g) %s.\n",
          mode_name[mode], std::sqrt(native_k_d4_norm2), native_k_abs, native_k_relative,
          kDifferenceTolerance, std::isfinite(native_k_relative) && native_k_relative <= kDifferenceTolerance ? "PASS" : "FAIL");
        std::printf("    virial C D2(h)-D2(2h), D4 denominator; Cxyz: ");
        for (int alpha = 0; alpha < 3; ++alpha) {
          const double c_abs = std::sqrt(c_delta2[alpha] / (d * coordinates.size()));
          const double c_rel = std::sqrt(c_delta2[alpha] / std::max(c_d4_norm2[alpha], 1.0e-300));
          const int ci = c_worst_input[alpha], co = c_worst_output[alpha];
          std::printf("%s%c D4norm %.6g eV/A absRMS %.6g eV/A rel %.6g %s worst input %d type %d axis %s -> output %d type %d axis %s (column %.5g)", alpha ? ", " : "", "xyz"[alpha],
            std::sqrt(c_d4_norm2[alpha]), c_abs, c_rel,
            std::isfinite(c_rel) && c_rel <= kDifferenceTolerance ? "PASS" : "FAIL",
            ci % n, atom.cpu_type[ci % n], axis_name[ci / n], co % n, atom.cpu_type[co % n],
            axis_name[co / n], c_worst_column[alpha]);
        }
        std::printf("\n");
      }
      if (mode == 0) {
        const double abs_rms = std::sqrt(analytic_delta2 / (d * coordinates.size()));
        const double relative = std::sqrt(analytic_delta2 / std::max(analytic_d4_norm2, 1.0e-300));
        const double precheck_k_relative = std::sqrt(analytic_precheck_delta2 / std::max(analytic_precheck_d4_norm2, 1.0e-300));
        const double precheck_c_relative[3] = {
          std::sqrt(c_precheck_delta2[0] / std::max(c_precheck_d4_norm2[0], 1.0e-300)),
          std::sqrt(c_precheck_delta2[1] / std::max(c_precheck_d4_norm2[1], 1.0e-300)),
          std::sqrt(c_precheck_delta2[2] / std::max(c_precheck_d4_norm2[2], 1.0e-300))};
        const double v_precheck_relative = std::sqrt(v_precheck_delta2 / std::max(v_precheck_d4_norm2, 1.0e-300));
        v_stencil_finite = v_stencil_finite && std::isfinite(v_precheck_relative);
        const bool gradient_pass = std::isfinite(r0_gradient_native_abs) && r0_gradient_native_abs <= kForceTolerance;
        const bool jvp_pass = std::isfinite(r0_site_jvp_abs) && r0_site_jvp_abs <= kForceTolerance;
        const bool k_pass = std::isfinite(precheck_k_relative) && precheck_k_relative <= kDifferenceTolerance;
        const bool c_pass[3] = {
          std::isfinite(precheck_c_relative[0]) && precheck_c_relative[0] <= kDifferenceTolerance,
          std::isfinite(precheck_c_relative[1]) && precheck_c_relative[1] <= kDifferenceTolerance,
          std::isfinite(precheck_c_relative[2]) && precheck_c_relative[2] <= kDifferenceTolerance};
        const bool precheck_failed = !gradient_pass || !jvp_pass || !k_pass || !c_pass[0] || !c_pass[1] ||
          !c_pass[2] || !v_stencil_finite;
        ++precheck_steps_total;
        if (!precheck_failed) ++precheck_steps_passed;
        if (h == fd_step) { requested_h_seen = true; requested_h_pass = !precheck_failed; }
        if (full) {
          std::printf("    h=%.6g%s grad-native=%.3e site-JVP=%.3e K=%.3e Cxyz=%.3e/%.3e/%.3e Vfinite=%s gate=%s | energyFD=%.3e/%.3e nativeK=%.3e\n",
            h, h == fd_step ? "*" : "", r0_gradient_native_abs, r0_site_jvp_abs,
            precheck_k_relative, precheck_c_relative[0], precheck_c_relative[1], precheck_c_relative[2],
            v_stencil_finite ? "yes" : "no", precheck_failed ? "FAIL" : "PASS",
            grad_abs, grad_rel, native_k_relative);
        }
        const int ia = worst_input % n, ix = worst_input / n, oa = worst_output % n, ox = worst_output / n;
        if (!full || precheck_failed || !std::isfinite(relative) || relative > kDifferenceTolerance)
          std::printf("    detail: analytic Hessian sampled D4 norm %.6g, D2 delta abs RMS %.6g, relative-to-D4 %.6g (limit %.3g); worst input %d type %d axis %s -> output %d type %d axis %s, column-relative %.6g.\n",
            std::sqrt(analytic_d4_norm2), abs_rms, relative, kDifferenceTolerance,
            ia, atom.cpu_type[ia], axis_name[ix], oa, atom.cpu_type[oa], axis_name[ox], worst_column_relative);
        if (!full || precheck_failed)
          std::printf("    precheck detail: h=%.6g gradient/native=%.6g JVP/gradient=%.6g K=%.6g Cxyz=%.6g/%.6g/%.6g Vfinite=%s; generation limits gradient/JVP=%.3g K/C=%.3g; PASS is sampled-column evidence only.\n",
            h, r0_gradient_native_abs, r0_site_jvp_abs, precheck_k_relative,
            precheck_c_relative[0], precheck_c_relative[1], precheck_c_relative[2], v_stencil_finite ? "yes" : "no",
            kForceTolerance, kDifferenceTolerance);
      }
      std::fflush(stdout);
    }
  }
  const double spacing = active_qnep->get_pppm_mesh_spacing();
  if (full) std::printf("[3/4] mesh translation oracle; compare mesh targets first and the saved active oracle last\n");
  for (const double factor : {0.75, 0.5}) {
    const double target_spacing = spacing * factor;
    if (full) std::printf("[mesh initialization: target %.6g A; external model initialization output follows]\n", target_spacing);
    QEvaluator mesh_evaluator(force.primary_nep_model_path(), n, force.get_run_input(), observer_box,
      atom.cpu_type, target_spacing);
    const QEvaluation mesh_eval = mesh_evaluator.evaluate(positions, true); ++evaluations;
    const std::vector<double> mesh_gradient = mesh_evaluator.analytic_gradient();
    if (full) {
      std::vector<double> direction(static_cast<std::size_t>(d), 0.0), site(n);
      double mesh_explicit_xyz[3] = {}, mesh_ik_xyz[3] = {}, mesh_native_xyz[3] = {}, mesh_site_jvp_xyz[3] = {};
      GPU_Vector<double> device_direction(d), device_site(n), device_gradient(d);
      for (int axis = 0; axis < 3; ++axis) {
        for (int i = 0; i < n; ++i) direction[axis * n + i] = 1.0;
        device_direction.copy_from_host(direction.data());
        QNEPReferenceTangentDiagnostics tangent;
        if (!mesh_evaluator.qnep.compute_reference_site_energy_derivative(
              observer_box, mesh_evaluator.type, mesh_evaluator.position, mesh_evaluator.force,
              &device_direction, &device_site, device_gradient, &tangent) || !tangent.valid)
          throw std::runtime_error("qNEP full diagnose mesh translation components rejected the current force frame");
        device_site.copy_to_host(site.data());
        require_finite(site, "qNEP full mesh translation site JVP");
        double local = 0.0, real = 0.0, pppm = 0.0;
        for (int i = 0; i < n; ++i) {
          local += tangent.local_site_tangent[i];
          real += tangent.real_site_tangent[i];
          pppm += tangent.pppm_site_tangent[i];
        }
        const double ik = tangent.pppm_ik_force_sum[axis];
        const double explicit_gradient = tangent.pppm_explicit_gradient_sum[axis];
        const double native_force = tangent.native_force_sum[axis];
        const double direct = std::accumulate(site.begin(), site.end(), 0.0);
        mesh_explicit_xyz[axis] = explicit_gradient;
        mesh_ik_xyz[axis] = ik;
        mesh_native_xyz[axis] = native_force;
        mesh_site_jvp_xyz[axis] = direct;
        const auto& raw_qdot = tangent.raw_charge_direction;
        const auto& projected_qdot = tangent.projected_charge_direction;
        std::printf("    components %c local=%.3e real=%.3e PPPM=%.3e siteJVP=%.3e site-res=%.3e IK=%.3e explicit=%.3e nativeF=%.3e force-res=%.3e; qdot raw(sum/max/RMS)=%.3e/%.3e/%.3e projected=%.3e/%.3e/%.3e\n",
          "xyz"[axis], local, real, pppm, direct, local + real + pppm - direct, ik, explicit_gradient,
          native_force, -native_force + ik + explicit_gradient - direct,
          raw_qdot.ordered_sum, raw_qdot.max_abs, raw_qdot.rms,
          projected_qdot.ordered_sum, projected_qdot.max_abs, projected_qdot.rms);
        for (int i = 0; i < n; ++i) direction[axis * n + i] = 0.0;
      }
      const auto mesh_gradient_net_for_target = rpmd_ja_reference_math::net_force_stats(mesh_gradient, n);
      PPPMReferenceTranslationReport mesh_oracle;
      const bool mesh_oracle_frame = mesh_evaluator.qnep.diagnose_reference_translation_energy(
        observer_box, mesh_evaluator.type, mesh_evaluator.position, mesh_evaluator.force,
        mesh_oracle, mesh_gradient_net_for_target.limit);
      if (!mesh_oracle_frame) {
        mesh_oracle.status = PPPMReferenceTranslationStatus::inconclusive;
        mesh_oracle.reason = PPPMReferenceTranslationReason::invalid_frame;
      }
      const char* oracle_calc_state = mesh_oracle.source_log_confirmation ? nullptr : "NOT_RUN";
      std::printf("    oracle mesh=%.6g frame=%s reason=%s target=%.3e assignment=%s mesh-invariants=%s FD-platform=%s source-log=%s\n",
        target_spacing, mesh_oracle_frame ? "valid" : "invalid", oracle_reason_name(mesh_oracle.reason),
        mesh_oracle.precision_target,
        oracle_calc_state ? oracle_calc_state : mesh_oracle.assignment_closure_pass ? "PASS" : "FAIL",
        oracle_calc_state ? oracle_calc_state : mesh_oracle.mesh_invariant_pass ? "PASS" : "FAIL",
        oracle_calc_state ? oracle_calc_state : mesh_oracle.fd_platform_pass ? "PASS" : "FAIL",
        mesh_oracle.source_log_confirmation ? mesh_oracle.source_protocol_version : "NOT_RUN");
      print_translation_sources("refined", mesh_oracle, mesh_oracle_frame, mesh_explicit_xyz,
        mesh_site_jvp_xyz, mesh_ik_xyz, mesh_native_xyz);
      std::fflush(stdout);
    }
    const auto mesh_native_net = rpmd_ja_reference_math::net_force_stats(mesh_eval.force, n);
    const auto mesh_gradient_net = rpmd_ja_reference_math::net_force_stats(mesh_gradient, n);
    std::array<double, 3> mesh_uniform_jvp = {};
    std::vector<double> mesh_direction(static_cast<std::size_t>(d), 0.0);
    for (int axis = 0; axis < 3; ++axis) {
      for (int i = 0; i < n; ++i) mesh_direction[axis * n + i] = 1.0;
      const auto site_jvp = mesh_evaluator.analytic_site_jvp(mesh_direction);
      mesh_uniform_jvp[axis] = std::accumulate(site_jvp.begin(), site_jvp.end(), 0.0);
      for (int i = 0; i < n; ++i) mesh_direction[axis * n + i] = 0.0;
    }
    double mesh_jvp_analytic_diff2 = 0.0, mesh_native_gradient_net_diff2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double jvp_error = mesh_uniform_jvp[axis] - mesh_gradient_net.net[axis];
      const double net_force_residual = mesh_native_net.net[axis] + mesh_gradient_net.net[axis];
      mesh_jvp_analytic_diff2 += jvp_error * jvp_error;
      mesh_native_gradient_net_diff2 += net_force_residual * net_force_residual;
    }
    double delta2 = 0.0, gradient2 = 0.0, native_mismatch2 = 0.0;
    for (int i = 0; i < d; ++i) {
      const double delta = mesh_gradient[i] - full_r0_gradient[i];
      delta2 += delta * delta; gradient2 += mesh_gradient[i] * mesh_gradient[i];
      const double mismatch = mesh_gradient[i] + mesh_eval.force[i]; native_mismatch2 += mismatch * mismatch;
    }
    std::printf("  R0 PPPM mesh-spacing trend configured target %.6g A: gradient delta-from-active RMS %.6g, analytic-gradient RMS %.6g, native-force mismatch RMS %.6g eV/A.\n",
      target_spacing, std::sqrt(delta2 / d), std::sqrt(gradient2 / d), std::sqrt(native_mismatch2 / d));
    if (!full) {
      std::printf("    mesh net-force quick diagnostic (eV/A): native force xyz=(%.9g, %.9g, %.9g), norm=%.9g, vector_norm=%.9g, limit=%.9g, status=PROXY_%s; analytic gradient xyz=(%.9g, %.9g, %.9g), norm=%.9g, vector_norm=%.9g, limit=%.9g, status=PROXY_%s; uniform translation JVP xyz=(%.9g, %.9g, %.9g); JVP-analytic net difference norm=%.9g; native_F+gradient net difference norm=%.9g.\n",
        mesh_native_net.net[0], mesh_native_net.net[1], mesh_native_net.net[2], mesh_native_net.net_norm,
        mesh_native_net.vector_norm, mesh_native_net.limit,
        mesh_native_net.within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE",
        mesh_gradient_net.net[0], mesh_gradient_net.net[1], mesh_gradient_net.net[2], mesh_gradient_net.net_norm,
        mesh_gradient_net.vector_norm, mesh_gradient_net.limit,
        mesh_gradient_net.within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE",
        mesh_uniform_jvp[0], mesh_uniform_jvp[1], mesh_uniform_jvp[2],
        std::sqrt(mesh_jvp_analytic_diff2), std::sqrt(mesh_native_gradient_net_diff2));
    } else {
      std::printf("    mesh net proxy: native=%.3e/%.3e limit=%.3e; analytic=%.3e/%.3e limit=%.3e; native-force mismatch RMS=%.3e eV/A.\n",
        mesh_native_net.net_norm, mesh_native_net.vector_norm, mesh_native_net.limit,
        mesh_gradient_net.net_norm, mesh_gradient_net.vector_norm, mesh_gradient_net.limit,
        std::sqrt(native_mismatch2 / d));
    }
    std::fflush(stdout);
  }
  if (full) {
    print_translation_sources("active", active_translation_oracle, active_translation_oracle_frame,
      active_pppm_explicit_xyz, active_uniform_jvp_xyz, active_pppm_ik_xyz, active_native_force_xyz);
    std::printf("[4/4] final summary\n");
    std::printf("  requested h precheck=%s; sampled full-qNEP prechecks=%d/%d; sampled PASS is not reference acceptance.\n",
      requested_h_seen && requested_h_pass ? "PASS" : "FAIL", precheck_steps_passed, precheck_steps_total);
    std::printf("  full V columns=%d; net norm=%.6g limit=%.6g status=%s; column-site/direct-JVP identity RMS=%.6g limit=%.3g eV/A status=%s (shared algebraic identity, not oracle proof).\n",
      d, active_full_v_net_norm, active_full_v_limit,
      active_full_v_within_limit ? "WITHIN_LIMIT" : "OUTSIDE_LIMIT_OR_NONFINITE", active_identity_rms,
      kForceTolerance, active_identity_rms <= kForceTolerance ? "PASS" : "FAIL");
    std::printf("  source diagnostic version=pppm_translation_sources_v2; active source log=%s; source contributions and refined mesh rows are above.\n",
      active_translation_oracle.source_log_confirmation ? active_translation_oracle.source_protocol_version : "NOT_RUN");
  } else {
    std::printf("rpmd_ja requested-h precheck: %s; sampled full-qNEP h steps PASS=%d/%d; SAMPLED PRECHECK ONLY / full reference acceptance still required.\n",
      requested_h_seen && requested_h_pass ? "PASS" : "FAIL", precheck_steps_passed, precheck_steps_total);
  }
  std::fflush(stdout);
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("rpmd_ja diagnose completed: evaluations=%llu, elapsed %.3f s; diagnostics only, no reference produced%s.\n",
    static_cast<unsigned long long>(evaluations), elapsed, full ? " (evaluation counter excludes the 3N JVPs)" : "");
}
