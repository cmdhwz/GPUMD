#include "rpmd_ja_qnep_prepare.cuh"
#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_additive.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifndef USE_HIP
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

namespace
{
constexpr int tile_size = 128;
constexpr double raw_difference_tolerance = 5.0e-2;
constexpr double tile_tolerance = 1.0e-8;
constexpr char raw_magic[8] = {'G','P','J','Q','R','A','W','\0'};
constexpr char raw_layout[] = "xyz_soa;derivative_input_rows_output_columns";

template <typename T> void read_value(std::istream& in, T& value)
{
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!in) throw std::runtime_error("truncated qNEP rpmd_ja raw header");
}

template <typename T> void read_vector(std::istream& in, std::vector<T>& value, const std::size_t count)
{
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::runtime_error("invalid qNEP rpmd_ja raw vector length");
  value.resize(count);
  in.read(reinterpret_cast<char*>(value.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error("truncated qNEP rpmd_ja raw vector");
}

struct Raw
{
  RpmdJAReference reference;
  int version = 0;
  std::uint64_t config = 0;
  int dimension = 0;
  int charge = 0;
  double spacing = 0.0;
  std::streamoff data = 0, coarse_v = 0, coarse_c = 0, footer = 0;
  double stats[18] = {};
};

Raw read_raw_header(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open qNEP rpmd_ja raw file: " + path);
  char magic[8], layout[sizeof(raw_layout)];
  std::uint32_t version = 0, endian = 0;
  int n = 0, d = 0, uses_pppm = 0;
  Raw raw;
  in.read(magic, sizeof(magic)); read_value(in, version); read_value(in, endian);
  read_value(in, n); read_value(in, d); read_value(in, raw.reference.temperature);
  read_value(in, raw.reference.fd_step); read_value(in, raw.reference.model_fingerprint);
  read_value(in, raw.config); read_value(in, raw.charge); read_value(in, uses_pppm);
  read_value(in, raw.spacing); in.read(layout, sizeof(layout));
  if (!in || std::memcmp(magic, raw_magic, sizeof(magic)) || (version != 1 && version != 2 && version != 3) || endian != 0x01020304 ||
      n <= 1 || n > std::numeric_limits<int>::max()/3 || d != 3 * n ||
      !(raw.reference.temperature > 0.0) || !std::isfinite(raw.reference.temperature) ||
      !(raw.reference.fd_step > 0.0) || !std::isfinite(raw.reference.fd_step) ||
      std::memcmp(layout, raw_layout, sizeof(layout)) || raw.charge < 1 || raw.charge > 2 ||
      uses_pppm != 1 || !(raw.spacing > 0.0) || !std::isfinite(raw.spacing) || raw.config == 0)
    throw std::runtime_error("unsupported or invalid qNEP rpmd_ja raw header");
  raw.version = static_cast<int>(version);
  raw.dimension = d;
  RpmdJAReference& r = raw.reference;
  r.backend = 2; r.number_of_atoms = n; r.q_charge_mode = raw.charge; r.q_uses_pppm = true;
  r.q_mesh_spacing = raw.spacing;
  r.mechanical_policy = version == 1 ? "native_reference_transport" :
    (version == 2 ? "native_reference_transport;analytic_site_gradient_v1" :
      "native_reference_transport;analytic_site_gradient_fd4_v1");
  r.mechanical_config_fingerprint = raw.config;
  double raw_cell[18];
  in.read(reinterpret_cast<char*>(raw_cell), sizeof(raw_cell));
  if (!in || !std::all_of(raw_cell,raw_cell+18,[](double x){return std::isfinite(x);}))
    throw std::runtime_error("qNEP raw cell contains non-finite values");
  std::copy(raw_cell, raw_cell + 9, r.cell);
  in.read(reinterpret_cast<char*>(r.pbc), sizeof(r.pbc));
  read_vector(in, r.types, n); read_vector(in, r.masses, n); read_vector(in, r.positions, d);
  double energy = 0.0;
  std::vector<double> site_energy, force, virial;
  read_value(in, energy); read_vector(in, site_energy, n); read_vector(in, force, d); read_vector(in, virial, static_cast<std::size_t>(9)*n);
  if (!std::isfinite(energy) || !std::all_of(site_energy.begin(),site_energy.end(),[](double x){return std::isfinite(x);}) ||
      !std::all_of(force.begin(),force.end(),[](double x){return std::isfinite(x);}) ||
      !std::all_of(virial.begin(),virial.end(),[](double x){return std::isfinite(x);}))
    throw std::runtime_error("qNEP raw reference contains non-finite values");
  r.force_balance_residual = 0.0;
  double max_force = 0.0;
  for (double x : force) max_force = std::max(max_force, std::abs(x));
  r.force_balance_residual = max_force;
  r.energy_gradient_relative_error = 0.0;
  for (double x : r.masses) if (!(x > 0.0) || !std::isfinite(x)) throw std::runtime_error("invalid qNEP raw mass");
  const double mass_sum=std::accumulate(r.masses.begin(),r.masses.end(),0.0);
  if(!(mass_sum>0.0)||!std::isfinite(mass_sum))throw std::runtime_error("invalid qNEP total mass");
  if(!std::all_of(r.positions.begin(),r.positions.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("qNEP raw positions contain non-finite values");
  for(double x:r.cell)if(!std::isfinite(x))throw std::runtime_error("qNEP raw cell contains non-finite values");
  for (int pbc : r.pbc) if (pbc != 1) throw std::runtime_error("qNEP rpmd_ja requires full periodicity");
  raw.data = in.tellg();
  const std::uint64_t vd = static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(n);
  const std::uint64_t dd = static_cast<std::uint64_t>(d) * static_cast<std::uint64_t>(d);
  const std::uint64_t max_elements=static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()/sizeof(double));
  if (dd > max_elements/7 || vd > (max_elements-7*dd)/2 || 2*vd+7*dd > max_elements-18)
    throw std::runtime_error("qNEP raw matrix offsets overflow");
  const std::uint64_t max_bytes=static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max());
  const std::uint64_t payload_bytes=(2*vd+7*dd+18)*sizeof(double);
  if(raw.data<0||static_cast<std::uint64_t>(raw.data)>max_bytes-payload_bytes)
    throw std::runtime_error("qNEP raw file offset overflows platform stream offsets");
  const std::streamoff vb = static_cast<std::streamoff>(vd * sizeof(double));
  const std::streamoff cb = static_cast<std::streamoff>(3 * dd * sizeof(double));
  const std::streamoff kb = static_cast<std::streamoff>(dd * sizeof(double));
  raw.coarse_v = raw.data + vb + cb + kb;
  raw.coarse_c = raw.coarse_v + vb;
  raw.footer = raw.coarse_c + cb;
  in.seekg(0, std::ios::end);
  if (in.tellg() != raw.footer + 18 * static_cast<std::streamoff>(sizeof(double)))
    throw std::runtime_error("qNEP raw file size does not match its matrix layout");
  in.seekg(raw.footer);
  in.read(reinterpret_cast<char*>(raw.stats), sizeof(raw.stats));
  const bool finite_stats = std::all_of(raw.stats, raw.stats + 18, [](double x) { return std::isfinite(x) && x >= 0.0; });
  const bool legacy_checks = raw.stats[17] == 1.0 && raw.stats[1] <= 1.0e-4 &&
    std::max({raw.stats[2], raw.stats[4], raw.stats[6], raw.stats[7], raw.stats[8], raw.stats[14], raw.stats[15]}) <= raw_difference_tolerance;
  const bool analytic_checks = raw.stats[17] == 2.0 && raw.stats[1] <= 1.0e-4 && raw.stats[3] <= 1.0e-4 &&
    std::max({raw.stats[4], raw.stats[6], raw.stats[7], raw.stats[8], raw.stats[14], raw.stats[15]}) <= raw_difference_tolerance;
  const bool analytic_fd4_checks = raw.stats[17] == 3.0 && raw.stats[1] <= 1.0e-4 && raw.stats[3] <= 1.0e-4 &&
    std::max({raw.stats[4], raw.stats[6], raw.stats[7], raw.stats[8], raw.stats[14], raw.stats[15]}) <= raw_difference_tolerance;
  if (!in || !finite_stats || raw.stats[16] != r.fd_step ||
      (version == 1 ? !legacy_checks : (version == 2 ? !analytic_checks : !analytic_fd4_checks)))
    throw std::runtime_error("qNEP raw finite-difference diagnostics fail accepted limits");
  r.energy_gradient_relative_error = raw.stats[0];
  r.force_gradient_relative_error = raw.stats[4];
  r.energy_gradient_absolute_rms = raw.stats[1];
  r.force_gradient_absolute_rms = raw.stats[5];
  r.energy_second_probe_relative_error = std::max(raw.stats[14], raw.stats[15]);
  r.force_balance_residual = raw.stats[12];
  for (int a = 0; a < 3; ++a) r.site_derivative_absolute_rms[a] = raw.stats[9 + a];
  return raw;
}

#ifndef USE_HIP
void cuda_check(const cudaError_t status, const char* where)
{ if (status != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status)); }
void solver_check(const cusolverStatus_t status, const char* where)
{ if (status != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + " failed"); }
void blas_check(const cublasStatus_t status, const char* where)
{ if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(where) + " failed"); }
__global__ void compact_physical(const double* a, double* p, int d, int r, int n);

template <typename T> struct SpectrumBuffer
{
  T* pointer = nullptr;
  SpectrumBuffer() = default;
  SpectrumBuffer(const std::size_t count, const char* where)
  {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), where);
  }
  SpectrumBuffer(const SpectrumBuffer&) = delete;
  SpectrumBuffer& operator=(const SpectrumBuffer&) = delete;
  ~SpectrumBuffer() { if (pointer) cudaFree(pointer); }
  T* data() { return pointer; }
  void copy_from_host(const T* source, const std::size_t count) { cuda_check(cudaMemcpy(pointer, source, count * sizeof(T), cudaMemcpyHostToDevice), "copy spectrum buffer to device"); }
  void copy_to_host(T* target, const std::size_t count) const { cuda_check(cudaMemcpy(target, pointer, count * sizeof(T), cudaMemcpyDeviceToHost), "copy spectrum buffer to host"); }
};

std::string low_spectrum_diagnostic(const double* a, const int d, const int r, const int n,
                                   cusolverDnHandle_t solver, cublasHandle_t blas, const double known_negative_rho,
                                   std::vector<RpmdJADiagnosticMode>& modes)
{
  constexpr int maximum_basis = 1024;
  constexpr int wanted = 4;
  const int limit = std::min(r, maximum_basis);
  const std::size_t basis_bytes = static_cast<std::size_t>(d) * limit * sizeof(double);
  std::size_t free_bytes = 0, total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "spectrum cudaMemGetInfo");
  (void)total_bytes;
  const std::size_t small_bytes = static_cast<std::size_t>(limit) * limit * sizeof(double) * 3 +
    static_cast<std::size_t>(limit) * sizeof(double) * 4 + 64ULL * 1024 * 1024;
  if (basis_bytes > free_bytes || small_bytes > free_bytes - basis_bytes)
    throw std::runtime_error("insufficient free GPU memory for bounded low-spectrum basis");

  SpectrumBuffer<double> basis(static_cast<std::size_t>(d) * limit, "allocate Lanczos basis");
  SpectrumBuffer<double> work_vector(d, "allocate Lanczos work vector");
  SpectrumBuffer<double> coefficients(limit, "allocate Lanczos coefficients");
  SpectrumBuffer<double> projected(static_cast<std::size_t>(limit) * limit, "allocate projected spectrum matrix");
  SpectrumBuffer<double> eigenvalues(limit, "allocate spectrum eigenvalues");
  SpectrumBuffer<double> ritz(d, "allocate Ritz vector");
  SpectrumBuffer<double> product(d, "allocate Ritz residual product");
  SpectrumBuffer<int> info(1, "allocate spectrum solver info");
  std::vector<double> initial(d), h(static_cast<std::size_t>(limit) * limit, 0.0);
  for (int i = 0; i < d; ++i)
    if (i != 0 && i != n && i != 2 * n) initial[i] = std::sin((i + 1) * 0.7548776662466927) +
      0.25 * std::cos((i + 1) * 0.5698402909980532);
  double norm = std::sqrt(std::inner_product(initial.begin(), initial.end(), initial.begin(), 0.0));
  if (!(norm > 0.0) || !std::isfinite(norm)) throw std::runtime_error("invalid deterministic Lanczos start vector");
  for (double& x : initial) x /= norm;
  basis.copy_from_host(initial.data(), d);

  int lwork = 0;
  solver_check(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
    limit, projected.data(), limit, eigenvalues.data(), &lwork), "query spectrum Dsyevd workspace");
  if (lwork <= 0) throw std::runtime_error("cuSOLVER returned invalid spectrum workspace size");
  SpectrumBuffer<double> eig_work(lwork, "allocate spectrum Dsyevd workspace");

  const double one = 1.0, zero = 0.0, minus_one = -1.0;
  const auto started = std::chrono::steady_clock::now();
  const double lanczos_scratch_gib = static_cast<double>(basis_bytes +
    (static_cast<std::size_t>(limit) * limit * 2 + lwork) * sizeof(double) +
    static_cast<std::size_t>(3) * d * sizeof(double)) / (1024.0 * 1024.0 * 1024.0);
  if (r <= 32) {
    SpectrumBuffer<double> compact(static_cast<std::size_t>(r) * r, "allocate exact small spectrum matrix");
    SpectrumBuffer<double> exact_values(r, "allocate exact small spectrum eigenvalues");
    dim3 block(16, 16), grid((r + 15) / 16, (r + 15) / 16);
    compact_physical<<<grid, block>>>(a, compact.data(), d, r, n);
    cuda_check(cudaGetLastError(), "extract exact small spectrum matrix");
    int exact_lwork = 0;
    solver_check(cusolverDnDsyevd_bufferSize(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      r, compact.data(), r, exact_values.data(), &exact_lwork), "query exact small spectrum workspace");
    if (exact_lwork <= 0) throw std::runtime_error("invalid exact small spectrum workspace size");
    SpectrumBuffer<double> exact_work(exact_lwork, "allocate exact small spectrum workspace");
    SpectrumBuffer<int> exact_info(1, "allocate exact small spectrum info");
    const double scratch_gib = static_cast<double>(basis_bytes +
      (2ULL * r * r + 3ULL * d + 3ULL * r + lwork + exact_lwork) * sizeof(double) + 2 * sizeof(int)) / (1024.0 * 1024.0 * 1024.0);
    std::printf("    qNEP rpmd_ja low-spectrum diagnostic starting: exact dimension %d, GPU scratch estimate %.3f GiB\n", r, scratch_gib);
    std::fflush(stdout);
    solver_check(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      r, compact.data(), r, exact_values.data(), exact_work.data(), exact_lwork, exact_info.data()), "exact small spectrum Dsyevd");
    cuda_check(cudaDeviceSynchronize(), "synchronize exact small spectrum");
    int host_info = 0; exact_info.copy_to_host(&host_info, 1);
    if (host_info != 0) throw std::runtime_error("exact small spectrum Dsyevd did not converge");
    std::vector<double> eig(r), vectors(static_cast<std::size_t>(r) * r);
    exact_values.copy_to_host(eig.data(), eig.size()); compact.copy_to_host(vectors.data(), vectors.size());
    if (!std::all_of(eig.begin(), eig.end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("exact small spectrum returned non-finite eigenvalues");
    std::ostringstream out;
    out << std::scientific << std::setprecision(17)
        << "\nlow_spectrum_status: EXACT_DENSE_SMALL\nlow_spectrum_method: exact_dense_small_projected_matrix"
        << "\nlow_spectrum_basis_dimension: " << r << "\nlow_spectrum_maximum_basis: " << r
        << "\nlow_spectrum_checkpoints: 1\nlow_spectrum_breakdown: no"
        << "\nlow_spectrum_gpu_scratch_estimate_GiB: "
        << scratch_gib << "\nlow_spectrum_elapsed_seconds: "
        << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << '\n'
        << "Exact eigenvalues of the compact translation-complement matrix; reports at most four values.\n";
    out << "known_negative_rayleigh_upper_bound: ";
    if (std::isfinite(known_negative_rho) && known_negative_rho < 0.0) out << known_negative_rho;
    else out << "unavailable";
    out << '\n';
    for (int q = 0; q < std::min(wanted, r); ++q) {
      std::vector<double> vector(d, 0.0);
      for (int i = 0; i < r; ++i) {
        const int original = i + 1 + (i >= n - 1) + (i >= 2 * n - 2);
        vector[original] = vectors[static_cast<std::size_t>(q) * r + i];
      }
      ritz.copy_from_host(vector.data(), d);
      blas_check(cublasDsymv(blas, CUBLAS_FILL_MODE_LOWER, d, &one, a, d, ritz.data(), 1,
        &zero, product.data(), 1), "exact small spectrum residual product");
      const double neg_lambda = -eig[q];
      blas_check(cublasDaxpy(blas, d, &neg_lambda, ritz.data(), 1, product.data(), 1), "exact small spectrum residual shift");
      double residual = 0.0; blas_check(cublasDnrm2(blas, d, product.data(), 1, &residual), "exact small spectrum residual norm");
      if (!std::isfinite(residual)) throw std::runtime_error("exact small spectrum residual is non-finite");
      modes.push_back({eig[q], residual, vector});
      const double frequency = std::sqrt(std::abs(eig[q])) * 1000.0 / (2.0 * PI * TIME_UNIT_CONVERSION);
      out << "ritz_" << q + 1 << "_eV_per_A2_per_amu: " << eig[q]
          << "\nritz_" << q + 1 << "_residual_norm: " << residual << "\nritz_" << q + 1 << "_status: exact\n";
      if (eig[q] < 0.0) out << "ritz_" << q + 1 << "_imaginary_frequency_THz: " << frequency
        << "\nritz_" << q + 1 << "_negative_sign_resolved: " << (eig[q] + residual < 0.0 ? "yes" : "no_residual_comparable") << '\n';
    }
    return out.str();
  }
  std::printf("    qNEP rpmd_ja low-spectrum diagnostic starting: max basis %d, GPU scratch estimate %.3f GiB\n", limit, lanczos_scratch_gib);
  std::fflush(stdout);
  std::vector<std::vector<double>> report_directions;
  auto analyze = [&](const int k, std::vector<double>& report_values,
                     std::vector<double>& report_residuals,
                     std::vector<std::vector<double>>& directions) {
    std::vector<double> square(static_cast<std::size_t>(k) * k);
    for (int j = 0; j < k; ++j) for (int i = 0; i < k; ++i)
      square[static_cast<std::size_t>(j) * k + i] = h[static_cast<std::size_t>(j) * limit + i];
    projected.copy_from_host(square.data(), square.size());
    solver_check(cusolverDnDsyevd(solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      k, projected.data(), k, eigenvalues.data(), eig_work.data(), lwork, info.data()), "spectrum Dsyevd");
    cuda_check(cudaDeviceSynchronize(), "synchronize spectrum Dsyevd");
    int host_info = 0;
    info.copy_to_host(&host_info, 1);
    if (host_info != 0) throw std::runtime_error("spectrum Dsyevd did not converge");
    std::vector<double> eig(k), vectors(static_cast<std::size_t>(k) * k);
    eigenvalues.copy_to_host(eig.data(), k); projected.copy_to_host(vectors.data(), vectors.size());
      if (!std::all_of(eig.begin(), eig.end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("spectrum Dsyevd returned non-finite eigenvalues");
    report_values.assign(eig.begin(), eig.begin() + std::min(wanted, k));
    report_residuals.clear();
    directions.clear();
    for (int q = 0; q < static_cast<int>(report_values.size()); ++q) {
      std::vector<double> y(k);
      for (int i = 0; i < k; ++i) y[i] = vectors[static_cast<std::size_t>(q) * k + i];
      coefficients.copy_from_host(y.data(), k);
      blas_check(cublasDgemv(blas, CUBLAS_OP_N, d, k, &one, basis.data(), d,
        coefficients.data(), 1, &zero, ritz.data(), 1), "reconstruct spectrum Ritz vector");
      blas_check(cublasDsymv(blas, CUBLAS_FILL_MODE_LOWER, d, &one, a, d, ritz.data(), 1,
        &zero, product.data(), 1), "spectrum residual product");
      const double lambda = eig[q], shift = -lambda;
      blas_check(cublasDaxpy(blas, d, &shift, ritz.data(), 1, product.data(), 1), "spectrum residual shift");
      double residual = 0.0;
      blas_check(cublasDnrm2(blas, d, product.data(), 1, &residual), "spectrum residual norm");
      if (!std::isfinite(residual)) throw std::runtime_error("spectrum Ritz residual is non-finite");
      report_residuals.push_back(residual);
      std::vector<double> direction(d);
      ritz.copy_to_host(direction.data(), direction.size());
      directions.push_back(std::move(direction));
    }
  };

  std::vector<double> report_values, report_residuals;
  int k = 1, next_check = std::min(limit, 32), checks = 0;
  bool converged = false, breakdown = false;
  while (k <= limit) {
    const double* current = basis.data() + static_cast<std::size_t>(k - 1) * d;
    blas_check(cublasDsymv(blas, CUBLAS_FILL_MODE_LOWER, d, &one, a, d, current, 1,
      &zero, work_vector.data(), 1), "Lanczos Hessian product");
    std::vector<double> coeff(k);
    blas_check(cublasDgemv(blas, CUBLAS_OP_T, d, k, &one, basis.data(), d,
      work_vector.data(), 1, &zero, coefficients.data(), 1), "Lanczos projection");
    coefficients.copy_to_host(coeff.data(), k);
    for (int i = 0; i < k; ++i) h[static_cast<std::size_t>(k - 1) * limit + i] = coeff[i];
    blas_check(cublasDgemv(blas, CUBLAS_OP_N, d, k, &one, basis.data(), d,
      coefficients.data(), 1, &zero, ritz.data(), 1), "Lanczos orthogonalization");
    blas_check(cublasDaxpy(blas, d, &minus_one, ritz.data(), 1, work_vector.data(), 1), "Lanczos orthogonalization");
    for (int pass = 0; pass < 2; ++pass) {
      blas_check(cublasDgemv(blas, CUBLAS_OP_T, d, k, &one, basis.data(), d,
        work_vector.data(), 1, &zero, coefficients.data(), 1), "Lanczos reorthogonalization");
      std::vector<double> correction(k); coefficients.copy_to_host(correction.data(), k);
      for (int i = 0; i < k; ++i) h[static_cast<std::size_t>(k - 1) * limit + i] += correction[i];
      blas_check(cublasDgemv(blas, CUBLAS_OP_N, d, k, &one, basis.data(), d,
        coefficients.data(), 1, &zero, ritz.data(), 1), "Lanczos reorthogonalization");
      blas_check(cublasDaxpy(blas, d, &minus_one, ritz.data(), 1, work_vector.data(), 1), "Lanczos reorthogonalization");
    }
    for (int i = 0; i < k - 1; ++i)
      h[static_cast<std::size_t>(i) * limit + k - 1] = h[static_cast<std::size_t>(k - 1) * limit + i];
    double beta = 0.0;
    blas_check(cublasDnrm2(blas, d, work_vector.data(), 1, &beta), "Lanczos residual norm");
    if (!std::isfinite(beta)) throw std::runtime_error("Lanczos residual norm is non-finite");
    if (k >= next_check || beta <= 64.0 * std::numeric_limits<double>::epsilon() || k == limit) {
      analyze(k, report_values, report_residuals, report_directions); ++checks;
      converged = report_values.size() == wanted;
      for (std::size_t q = 0; q < report_values.size(); ++q)
        converged = converged && report_residuals[q] <= 1.0e-10 + 1.0e-8 * std::abs(report_values[q]);
      if (std::isfinite(known_negative_rho) && known_negative_rho < 0.0)
        converged = converged && !report_values.empty() && report_values[0] <= known_negative_rho + 1.0e-10 + 1.0e-8 * std::abs(known_negative_rho);
      if (converged || beta <= 64.0 * std::numeric_limits<double>::epsilon() || k == limit) {
        breakdown = beta <= 64.0 * std::numeric_limits<double>::epsilon() && k < r;
        break;
      }
      next_check = std::min(limit, next_check + 64);
    }
    if (beta <= 64.0 * std::numeric_limits<double>::epsilon()) { breakdown = k < r; break; }
    if (k < limit) {
      const double inv = 1.0 / beta;
      blas_check(cublasDscal(blas, d, &inv, work_vector.data(), 1), "normalize Lanczos vector");
      cuda_check(cudaMemcpy(basis.data() + static_cast<std::size_t>(k) * d, work_vector.data(),
        static_cast<std::size_t>(d) * sizeof(double), cudaMemcpyDeviceToDevice), "append Lanczos basis vector");
      ++k;
    } else break;
  }

  std::ostringstream out;
  out << std::scientific << std::setprecision(17)
      << "\nlow_spectrum_status: " << (converged ? "CONVERGED_RITZ" : "UNCONVERGED_RITZ")
      << "\nlow_spectrum_method: bounded_full_reorthogonalization_Lanczos"
      << "\nlow_spectrum_basis_dimension: " << k << "\nlow_spectrum_maximum_basis: " << limit
      << "\nlow_spectrum_checkpoints: " << checks << "\nlow_spectrum_breakdown: " << (breakdown ? "yes_partial_subspace" : "no")
      << "\nlow_spectrum_gpu_scratch_estimate_GiB: "
      << lanczos_scratch_gib << "\nlow_spectrum_elapsed_seconds: "
      << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
      << "\nRitz values are approximate unless the exact small projected matrix path is used; this reports at most four values and does not certify the full negative spectrum.\n";
  out << "known_negative_rayleigh_upper_bound: ";
  if (std::isfinite(known_negative_rho) && known_negative_rho < 0.0) out << known_negative_rho;
  else out << "unavailable";
  out << '\n';
  for (std::size_t q = 0; q < report_values.size(); ++q) {
    const double lambda = report_values[q], residual = report_residuals[q];
    const double frequency = std::sqrt(std::abs(lambda)) * 1000.0 / (2.0 * PI * TIME_UNIT_CONVERSION);
    out << "ritz_" << q + 1 << "_eV_per_A2_per_amu: " << lambda
        << "\nritz_" << q + 1 << "_residual_norm: " << residual
        << "\nritz_" << q + 1 << "_status: "
        << (residual <= 1.0e-10 + 1.0e-8 * std::abs(lambda) ? "converged" : "UNCONVERGED") << '\n';
    if (lambda < 0.0) out << "ritz_" << q + 1 << "_imaginary_frequency_THz: " << frequency
      << "\nritz_" << q + 1 << "_negative_sign_resolved: " << (lambda + residual < 0.0 ? "yes" : "no_residual_comparable") << '\n';
    modes.push_back({lambda, residual, report_directions[q]});
  }
  return out.str();
}

__global__ void make_mass_hessian(const double* k, const double* masses, double* a, int d, int n)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < d && y < d) a[static_cast<std::size_t>(x) * d + y] =
    0.5 * (k[static_cast<std::size_t>(y) * d + x] + k[static_cast<std::size_t>(x) * d + y]) /
      sqrt(masses[x % n] * masses[y % n]);
}

__global__ void matvec_kernel(const double* a, const double* x, double* y, int n, int ld)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { double s = 0.0; for (int j = 0; j < n; ++j) s += a[static_cast<std::size_t>(i) * ld + j] * x[j]; y[i] = s; }
}

__global__ void householder_kernel(double* a, const double* w, const double* aw, int n, double gamma)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < n && y < n) a[static_cast<std::size_t>(x) * n + y] +=
    -2.0 * w[x] * aw[y] - 2.0 * aw[x] * w[y] + 4.0 * gamma * w[x] * w[y];
}

__global__ void compact_physical(const double* a, double* p, int d, int r, int n)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < r && y < r) {
    const int i = x + 1 + (x >= n - 1) + (x >= 2 * n - 2);
    const int j = y + 1 + (y >= n - 1) + (y >= 2 * n - 2);
    p[static_cast<std::size_t>(x) * r + y] = a[static_cast<std::size_t>(i) * d + j];
  }
}

__global__ void subtract_diagonal(double* a, int n, double value)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[static_cast<std::size_t>(i) * n + i] -= value;
}

__global__ void zero_upper_column_major(double* a, int n)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n*n && index % n < index / n) a[index] = 0.0;
}

__global__ void project_translation_rows(double* a, int d, int n)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < d && y < d && (x == 0 || x == n || x == 2*n || y == 0 || y == n || y == 2*n))
    a[static_cast<std::size_t>(x) * d + y] = 0.0;
}

__global__ void add_matrix(double* target, const double* increment, int d)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < d && y < d) target[static_cast<std::size_t>(x)*d+y] += increment[static_cast<std::size_t>(x)*d+y];
}

__global__ void scatter_sparse(double* target, const std::uint64_t* indices,
                               const double* values, const int count)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) target[indices[i]] = values[i];
}

__global__ void matrix_projection_sums(const double* a, double* sums, int d, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    double total = 0.0, removed = 0.0;
    for (int j = 0; j < d; ++j) {
      const double x = a[static_cast<std::size_t>(i) * d + j];
      total += x*x;
      if (i == 0 || i == n || i == 2*n || j == 0 || j == n || j == 2*n) removed += x*x;
    }
    sums[i] = total; sums[d+i] = removed;
  }
}

__global__ void asymmetry_sums(const double* a,const double* masses,double* sums,int d,int n)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<d){double diff=0.0,norm=0.0;for(int j=0;j<d;++j){const double scale=sqrt(masses[i%n]*masses[j%n]);const double x=a[static_cast<std::size_t>(i)*d+j]/scale,y=a[static_cast<std::size_t>(j)*d+i]/scale,s=0.5*(x+y);diff+=(x-y)*(x-y);norm+=s*s;}sums[i]=diff;sums[d+i]=norm;}
}

__global__ void diagonal_values(const double* a,double* d,int n)
{const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=a[static_cast<std::size_t>(i)*n+i];}

__global__ void failure_column(const double* a,double* b,int d,int n,int column,int count)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){const int row=i+1+(i>=n-1)+(i>=2*n-2),col=column+1+(column>=n-1)+(column>=2*n-2);b[i]=a[static_cast<std::size_t>(row)*d+col];}
}

__global__ void translation_product(const double* a,const double* mass_translation,double* y,int d,int n,int axis)
{
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<d){double x=0.0;for(int j=0;j<n;++j)x+=a[static_cast<std::size_t>(i)*d+axis*n+j]*mass_translation[j];y[i]=x;}
}

__global__ void vectorize_tile(const double* a, double* tile, int d, int r0, int c0, int nr, int nc)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nr && j < nc) tile[static_cast<std::size_t>(i) * nc + j] = a[static_cast<std::size_t>(r0 + i) * d + c0 + j];
}

__global__ void row_sums(const double* a, double* sums, int d)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) { double s = 0.0; for (int j = 0; j < d; ++j) s += fabs(a[static_cast<std::size_t>(i) * d + j]); sums[i] = s; }
}

__global__ void probe_product(const double* a, const double* x, double* y, int n, int ld)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { double s = 0.0; for (int j = 0; j < n; ++j) s += a[static_cast<std::size_t>(i) * ld + j] * x[j]; y[i] = s; }
}

void write_value(std::ostream& out, const int value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }
void write_value(std::ostream& out, const std::uint64_t value) { out.write(reinterpret_cast<const char*>(&value), sizeof(value)); }

void write_tile(std::ostream& out, const RpmdJAMatrixTile& tile)
{
  write_value(out, tile.row); write_value(out, tile.column); write_value(out, tile.rows); write_value(out, tile.columns); write_value(out, tile.rank);
  write_value(out, static_cast<std::uint64_t>(tile.left.size())); write_value(out, static_cast<std::uint64_t>(tile.right.size()));
  out.write(reinterpret_cast<const char*>(tile.left.data()), static_cast<std::streamsize>(tile.left.size() * sizeof(double)));
  out.write(reinterpret_cast<const char*>(tile.right.data()), static_cast<std::streamsize>(tile.right.size() * sizeof(double)));
  if (!out) throw std::runtime_error("failed writing qNEP rpmd_ja block tile");
}

RpmdJAMatrixTile dense_tile(const int row, const int col, const int rows, const int cols, const double* data)
{
  RpmdJAMatrixTile tile;
  tile.row = row; tile.column = col; tile.rows = rows; tile.columns = cols; tile.rank = 0;
  tile.left.assign(data, data + static_cast<std::size_t>(rows) * cols);
  return tile;
}

void svd_tile(cusolverDnHandle_t solver, cublasHandle_t blas, const double* input, const int rows, const int cols, RpmdJAMatrixTile& result,
  double* work, const int lwork, double* matrix, double* singular, double* u, double* vt, int* info,
  double* dleft, double* dright, double* dproduct, double& relative_error)
{
  relative_error = 0.0;
  const bool transpose = rows < cols;
  const int m = std::max(rows, cols), n = std::min(rows, cols), rankmax = n;
  const int ld = m;
  std::vector<double> a(static_cast<std::size_t>(m) * n);
  if (!transpose) for(int i=0;i<m;++i)for(int j=0;j<n;++j)a[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*m]=input[static_cast<std::size_t>(i)*cols+j];
  else for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) a[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * m] = input[static_cast<std::size_t>(i) * cols + j];
  cuda_check(cudaMemcpy(matrix, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), "copy SVD tile");
  solver_check(cusolverDnDgesvd(solver, 'S', 'S', m, n, matrix, ld, singular, u, ld, vt, rankmax, work, lwork, nullptr, info), "cusolverDnDgesvd");
  cuda_check(cudaDeviceSynchronize(), "synchronize SVD");
  int host_info = 0;
  cuda_check(cudaMemcpy(&host_info, info, sizeof(int), cudaMemcpyDeviceToHost), "read SVD info");
  if (host_info != 0) throw std::runtime_error("qNEP H tile SVD did not converge");
  std::vector<double> s(rankmax), hu(static_cast<std::size_t>(m) * rankmax), hvt(static_cast<std::size_t>(rankmax) * n);
  cuda_check(cudaMemcpy(s.data(), singular, s.size()*sizeof(double), cudaMemcpyDeviceToHost), "read SVD singular values");
  cuda_check(cudaMemcpy(hu.data(), u, hu.size()*sizeof(double), cudaMemcpyDeviceToHost), "read SVD U");
  cuda_check(cudaMemcpy(hvt.data(), vt, hvt.size()*sizeof(double), cudaMemcpyDeviceToHost), "read SVD VT");
  double norm2 = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(rows) * cols; ++i) norm2 += input[i] * input[i];
  if(!std::isfinite(norm2))throw std::runtime_error("qNEP H tile norm overflows");
  int k = rankmax;
  double discarded = 0.0;
  while (k > 0) { const double candidate = discarded + s[k - 1] * s[k - 1]; if (candidate > tile_tolerance * tile_tolerance * std::max(norm2, 1.0e-300)) break; discarded = candidate; --k; }
  if (k == 0 || k == rankmax || static_cast<std::size_t>(k) * (rows + cols) >= static_cast<std::size_t>(rows) * cols) {
    result = dense_tile(result.row, result.column, rows, cols, input);
    return;
  }
  result.rank = k; result.left.resize(static_cast<std::size_t>(rows) * k); result.right.resize(static_cast<std::size_t>(k) * cols);
  if (!transpose) {
    for (int i = 0; i < rows; ++i) for (int q = 0; q < k; ++q) result.left[static_cast<std::size_t>(i)*k+q] = hu[static_cast<std::size_t>(i)+static_cast<std::size_t>(q)*ld] * std::sqrt(s[q]);
    for (int q = 0; q < k; ++q) for (int j = 0; j < cols; ++j) result.right[static_cast<std::size_t>(q)*cols+j] = std::sqrt(s[q]) * hvt[static_cast<std::size_t>(q)+static_cast<std::size_t>(j)*rankmax];
  } else {
    for (int i = 0; i < rows; ++i) for (int q = 0; q < k; ++q) result.left[static_cast<std::size_t>(i)*k+q] = hvt[static_cast<std::size_t>(q)+static_cast<std::size_t>(i)*rankmax] * std::sqrt(s[q]);
    for (int q = 0; q < k; ++q) for (int j = 0; j < cols; ++j) result.right[static_cast<std::size_t>(q)*cols+j] = std::sqrt(s[q]) * hu[static_cast<std::size_t>(j)+static_cast<std::size_t>(q)*m];
  }
  cuda_check(cudaMemcpy(dleft,result.left.data(),result.left.size()*sizeof(double),cudaMemcpyHostToDevice),"upload H left factor");
  cuda_check(cudaMemcpy(dright,result.right.data(),result.right.size()*sizeof(double),cudaMemcpyHostToDevice),"upload H right factor");
  // Row-major C=L*R is column-major C^T=R^T*L^T.
  const double one=1.0, zero=0.0;
  blas_check(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_N,cols,rows,k,&one,dright,cols,dleft,k,&zero,dproduct,cols),"reconstruct H tile");
  cuda_check(cudaDeviceSynchronize(),"reconstruct H tile");
  std::vector<double> reconstructed(static_cast<std::size_t>(rows)*cols);
  cuda_check(cudaMemcpy(reconstructed.data(),dproduct,reconstructed.size()*sizeof(double),cudaMemcpyDeviceToHost),"read reconstructed H tile");
  double err2=0.0; for(std::size_t i=0;i<reconstructed.size();++i){const double e=reconstructed[i]-input[i];err2+=e*e;}
  relative_error = std::sqrt(err2 / std::max(norm2, 1.0e-300));
  if (!(relative_error <= tile_tolerance) || result.left.size() + result.right.size() >= static_cast<std::size_t>(rows)*cols) {
    result = dense_tile(result.row, result.column, rows, cols, input);
    relative_error = 0.0;
  }
}
#endif
} // namespace

void prepare_rpmd_ja_qnep_reference(const std::string& raw_path, const std::string& output_path, const std::string& kernel_table_path,
                                    const RpmdJAModeValidator& mode_validator, const std::string& additive_path)
{
#ifdef USE_HIP
  (void)raw_path; (void)output_path; (void)kernel_table_path; (void)mode_validator; (void)additive_path;
  throw std::runtime_error("qNEP rpmd_ja reference preparation currently requires CUDA cuSOLVER");
#else
  if (raw_path.empty() || output_path.empty() || kernel_table_path.empty()) throw std::invalid_argument("qNEP rpmd_ja prepare requires raw, output, and kernel-table paths");
  const std::string sidecar_path = output_path + ".stability", tmp_path = output_path + ".tmp", side_tmp = sidecar_path + ".tmp", failure_path = output_path + ".failure.txt";
  std::ifstream ex1(output_path), ex2(sidecar_path), ex3(tmp_path), ex4(side_tmp), ex5(failure_path);
  if (ex1.good() || ex2.good() || ex3.good() || ex4.good() || ex5.good()) throw std::runtime_error("qNEP rpmd_ja output, temporary, or failure diagnostic already exists: " + (ex5.good()?failure_path:output_path));
  Raw raw = read_raw_header(raw_path);
  RpmdJAReference& ref = raw.reference;
  std::printf("    qNEP rpmd_ja raw derivative policy: %s; fd_step %.9g A; %s\n",
    ref.mechanical_policy.c_str(), ref.fd_step,
    raw.version == 3 ?
      "D4 +/-h,+/-2h with D2(h)-D2(2h) checks" : "legacy stencil checks");
  load_rpmd_ja_kernel_table(kernel_table_path, ref);
  const int n = ref.number_of_atoms, d = raw.dimension, r = d - 3;
  const std::size_t dd = static_cast<std::size_t>(d) * d;
  const std::size_t rr = static_cast<std::size_t>(r) * r;
  std::ifstream raw_in(raw_path, std::ios::binary);
  if (!raw_in) throw std::runtime_error("cannot reopen qNEP raw file");
  const std::streamoff k_start = raw.data + static_cast<std::streamoff>(1ULL*d*n + 3ULL*d*d) * sizeof(double);
  double *dk = nullptr, *a = nullptr, *physical = nullptr, *shifted = nullptr, *reconstructed_device = nullptr, *dv = nullptr, *dw = nullptr, *dm = nullptr;
  double *work=nullptr,*query_matrix=nullptr,*pivot_device=nullptr,*tilebuf=nullptr;
  double *sm=nullptr,*su=nullptr,*svt=nullptr,*sw=nullptr,*dleft=nullptr,*dright=nullptr,*dproduct=nullptr,*dsingular=nullptr;
  int* dinfo = nullptr;
  int* sinfo=nullptr;
  cusolverDnHandle_t solver = nullptr;
  cublasHandle_t blas = nullptr;
  std::string created;
  try {
    RpmdJAAdditiveData additive;
    const bool use_additive = !additive_path.empty();
    if (use_additive) {
      std::vector<double> raw_gradient(static_cast<std::size_t>(d), 0.0), site_energy(static_cast<std::size_t>(n));
      for (int coordinate = 0; coordinate < d; ++coordinate) {
        raw_in.clear(); raw_in.seekg(raw.data + static_cast<std::streamoff>(coordinate) * n * sizeof(double));
        raw_in.read(reinterpret_cast<char*>(site_energy.data()), static_cast<std::streamsize>(n * sizeof(double)));
        if (!raw_in) throw std::runtime_error("truncated qNEP raw V while checking additive gradient");
        raw_gradient[coordinate] = std::accumulate(site_energy.begin(), site_energy.end(), 0.0);
      }
      additive = read_rpmd_ja_additive(additive_path, raw_path, ref, -1, raw_gradient);
      ref.mechanical_policy = std::string("native_reference_transport;") +
        (additive.internal_mass_com ? "internal_mass_com_pullback_v1;" : "") +
        "finite_temperature_additive_v1;beads=" +
        std::to_string(additive.beads) + ";derivative=" + std::to_string(raw.version);
      ref.additive_beads = additive.beads;
      ref.additive_epsilon = additive.epsilon;
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    (void)total_bytes;
    if(dd>std::numeric_limits<std::size_t>::max()/sizeof(double)||rr>std::numeric_limits<std::size_t>::max()/sizeof(double))
      throw std::runtime_error("qNEP rpmd_ja matrix byte count overflows host address space");
    const std::size_t matrix_bytes=dd*sizeof(double), physical_bytes=rr*sizeof(double);
    const std::size_t safety_bytes=64ULL*1024*1024;
    if(matrix_bytes>(std::numeric_limits<std::size_t>::max()-safety_bytes)/2)
      throw std::runtime_error("qNEP rpmd_ja matrix preflight size overflows");
    const std::size_t minimum_required=2*matrix_bytes+safety_bytes;
    if(minimum_required>free_bytes)throw std::runtime_error("qNEP rpmd_ja matrix preflight exceeds free GPU memory");
    solver_check(cusolverDnCreate(&solver), "cusolverDnCreate");
    blas_check(cublasCreate(&blas),"cublasCreate");
    int potrf_work = 0;
    // Query POTRF before allocating either dense GPU matrix.
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&query_matrix), rr*sizeof(double)), "allocate POTRF query matrix");
    solver_check(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, r, query_matrix, r, &potrf_work), "cusolverDnDpotrf_bufferSize");
    if(potrf_work<=0)throw std::runtime_error("cuSOLVER returned a nonpositive POTRF workspace size");
    cuda_check(cudaFree(query_matrix), "free POTRF query matrix");
    query_matrix=nullptr;
    const std::size_t workspace_bytes=static_cast<std::size_t>(potrf_work)*sizeof(double);
    if(workspace_bytes>std::numeric_limits<std::size_t>::max()-matrix_bytes-physical_bytes-safety_bytes)
      throw std::runtime_error("qNEP rpmd_ja Cholesky workspace size overflows");
    const std::size_t peak_base=std::numeric_limits<std::size_t>::max()-matrix_bytes-safety_bytes;
    if(workspace_bytes>peak_base||physical_bytes>(peak_base-workspace_bytes)/(use_additive?3:1))
      throw std::runtime_error("qNEP rpmd_ja Cholesky peak size overflows");
    const std::size_t factor_peak=matrix_bytes+(use_additive?3:1)*physical_bytes+workspace_bytes+safety_bytes;
    if(use_additive){
      if(matrix_bytes>(std::numeric_limits<std::size_t>::max()-2*physical_bytes-safety_bytes)/2)
        throw std::runtime_error("qNEP additive certificate peak size overflows");
    }
    const std::size_t certificate_peak=use_additive?2*matrix_bytes+2*physical_bytes+safety_bytes:0;
    if(use_additive&&(additive.k.values.size()>std::numeric_limits<std::size_t>::max()/(sizeof(std::uint64_t)+sizeof(double)) ||
       additive.k.values.size()*(sizeof(std::uint64_t)+sizeof(double))>std::numeric_limits<std::size_t>::max()-safety_bytes))
      throw std::runtime_error("sparse additive K transfer size overflows host address space");
    const std::size_t additive_sparse_bytes=use_additive ? additive.k.values.size()*(sizeof(std::uint64_t)+sizeof(double)) : 0;
    if(2*matrix_bytes>std::numeric_limits<std::size_t>::max()-additive_sparse_bytes-safety_bytes)
      throw std::runtime_error("qNEP rpmd_ja GPU preflight size overflows");
    const std::size_t memory_required=std::max({2*matrix_bytes+additive_sparse_bytes+safety_bytes,factor_peak,certificate_peak});
    std::printf("    qNEP rpmd_ja prepare GPU preflight: %.3f GiB conservative peak, %.3f GiB free\n",
      static_cast<double>(memory_required)/(1024.0*1024.0*1024.0),static_cast<double>(free_bytes)/(1024.0*1024.0*1024.0));
    if (memory_required > free_bytes) throw std::runtime_error("qNEP rpmd_ja Cholesky preflight exceeds free GPU memory");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dk), dd*sizeof(double)), "allocate raw Hessian");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&a), dd*sizeof(double)), "allocate mass Hessian");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dv), static_cast<std::size_t>(d)*sizeof(double)), "allocate probe vector");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dw), 2*static_cast<std::size_t>(d)*sizeof(double)), "allocate probe result");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dinfo), sizeof(int)), "allocate solver info");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&tilebuf),static_cast<std::size_t>(tile_size)*tile_size*sizeof(double)),"allocate D tile buffer");
    std::vector<double> row(d);
    for (int i = 0; i < d; ++i) {
      raw_in.seekg(k_start + static_cast<std::streamoff>(i)*d*sizeof(double));
      raw_in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(d*sizeof(double)));
      if (!raw_in || !std::all_of(row.begin(), row.end(), [](double x){return std::isfinite(x);})) throw std::runtime_error("qNEP raw Hessian contains invalid values");
      cuda_check(cudaMemcpy(dk + static_cast<std::size_t>(i)*d, row.data(), d*sizeof(double), cudaMemcpyHostToDevice), "upload raw Hessian row");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dm), n*sizeof(double)), "allocate mass vector");
    cuda_check(cudaMemcpy(dm, ref.masses.data(), n*sizeof(double), cudaMemcpyHostToDevice), "upload masses");
    asymmetry_sums<<<(d+255)/256,256>>>(dk,dm,dw,d,n);cuda_check(cudaGetLastError(),"measure qNEP raw Hessian asymmetry");cuda_check(cudaDeviceSynchronize(),"measure qNEP raw Hessian asymmetry");
    std::vector<double> asym_sums(2*static_cast<std::size_t>(d));cuda_check(cudaMemcpy(asym_sums.data(),dw,asym_sums.size()*sizeof(double),cudaMemcpyDeviceToHost),"read raw Hessian asymmetry");
    double asym2=0.0,sym2=0.0;for(int i=0;i<d;++i){asym2+=asym_sums[i];sym2+=asym_sums[d+i];}
    ref.hessian_symmetry_relative_error=std::sqrt(asym2/std::max(sym2,1.0e-300));
    if(!std::isfinite(ref.hessian_symmetry_relative_error)||!(ref.hessian_symmetry_relative_error<=raw_difference_tolerance))throw std::runtime_error("qNEP raw force Jacobian antisymmetry exceeds 5e-2");
    dim3 block(16,16), grid((d+15)/16,(d+15)/16);
    make_mass_hessian<<<grid,block>>>(dk, dm, a, d, n);
    cuda_check(cudaGetLastError(), "build mass Hessian"); cuda_check(cudaDeviceSynchronize(), "build mass Hessian");
    cuda_check(cudaFree(dk), "free raw Hessian"); dk = nullptr;
    std::vector<double> mass_translation(n), house(d), result_host(d);
    std::vector<std::vector<double>> householder(3, std::vector<double>(d));
    double mass_sum = 0.0; for (double x : ref.masses) mass_sum += x;
    for (int i = 0; i < n; ++i) mass_translation[i] = std::sqrt(ref.masses[i]/mass_sum);
    for (int axis=0; axis<3; ++axis) {
      std::fill(house.begin(),house.end(),0.0);
      double norm2 = 0.0;
      for (int i=0;i<n;++i) { house[axis*n+i]=mass_translation[i]; if (i==0) house[axis*n+i]-=1.0; norm2+=house[axis*n+i]*house[axis*n+i]; }
      if (!(norm2>0.0)) throw std::runtime_error("degenerate qNEP translation Householder");
      const double inv=1.0/std::sqrt(norm2); for(double& x:house)x*=inv;
      householder[axis]=house;
      cuda_check(cudaMemcpy(dv,house.data(),d*sizeof(double),cudaMemcpyHostToDevice),"upload Householder vector");
      // Each axis translation is transformed independently and maps to indices 0, N, 2N.
      matvec_kernel<<<(d+255)/256,256>>>(a,dv,dw,d,d);
      cuda_check(cudaGetLastError(),"apply Householder matvec"); cuda_check(cudaDeviceSynchronize(),"apply Householder matvec");
      cuda_check(cudaMemcpy(result_host.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost),"read Householder matvec");
      double gamma=0.0; for(int i=0;i<d;++i)gamma+=house[i]*result_host[i];
      householder_kernel<<<grid,block>>>(a,dv,dw,d,gamma);
      cuda_check(cudaGetLastError(),"apply Householder congruence"); cuda_check(cudaDeviceSynchronize(),"apply Householder congruence");
    }
    matrix_projection_sums<<<(d+255)/256,256>>>(a,dw,d,n);
    cuda_check(cudaGetLastError(),"measure translation projection"); cuda_check(cudaDeviceSynchronize(),"measure translation projection");
    std::vector<double> projection_sums(2*static_cast<std::size_t>(d));
    cuda_check(cudaMemcpy(projection_sums.data(),dw,projection_sums.size()*sizeof(double),cudaMemcpyDeviceToHost),"read translation projection measure");
    double total2=0.0,removed2=0.0;for(int i=0;i<d;++i){total2+=projection_sums[i];removed2+=projection_sums[d+i];}
    ref.projection_relative_change=std::sqrt(removed2/std::max(total2,1.0e-300));
    if(!std::isfinite(ref.projection_relative_change)||!(ref.projection_relative_change<=5.0e-2))throw std::runtime_error("translation projection changes D by more than 5e-2");
    project_translation_rows<<<grid,block>>>(a,d,n);cuda_check(cudaGetLastError(),"remove translation modes");cuda_check(cudaDeviceSynchronize(),"remove translation modes");
    if (use_additive) {
      const std::size_t count=additive.k.values.size();
      if(count>static_cast<std::size_t>(std::numeric_limits<int>::max())||count>std::numeric_limits<std::size_t>::max()/sizeof(std::uint64_t))
        throw std::runtime_error("sparse additive K exceeds GPU scatter index limits");
      std::vector<std::uint64_t> indices(count);
      std::vector<double> weighted_values(count);
      for(int row=0;row<d;++row)for(std::size_t p=additive.k.offsets[row];p<additive.k.offsets[row+1];++p)
      {
        indices[p]=static_cast<std::uint64_t>(row)*d+additive.k.columns[p];
        const double value=additive.k.values[p];
        const double scale=std::sqrt(ref.masses[row%n])*std::sqrt(ref.masses[additive.k.columns[p]%n]);
        const double weighted=value/scale;
        if(!std::isfinite(value)||!std::isfinite(scale)||!(scale>0.0)||!std::isfinite(weighted)||
           (value!=0.0&&(weighted==0.0||std::abs(weighted)<std::numeric_limits<double>::min())))
          throw std::runtime_error("invalid mass-weighted sparse additive Kadd");
        weighted_values[p]=weighted;
      }
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&dk),dd*sizeof(double)), "allocate weighted additive Kadd");
      cuda_check(cudaMemset(dk,0,dd*sizeof(double)), "zero weighted additive Kadd");
      if(count){
        SpectrumBuffer<std::uint64_t> dindices(count,"allocate sparse K indices");
        SpectrumBuffer<double> dvalues(count,"allocate sparse K values");
        dindices.copy_from_host(indices.data(),count);
        dvalues.copy_from_host(weighted_values.data(),count);
        scatter_sparse<<<static_cast<unsigned>((count+255)/256),256>>>(dk,dindices.data(),dvalues.data(),static_cast<int>(count));
        cuda_check(cudaGetLastError(),"scatter mass-weighted sparse Kadd");cuda_check(cudaDeviceSynchronize(),"scatter mass-weighted sparse Kadd");
      }
      for (int axis=0;axis<3;++axis) {
        cuda_check(cudaMemcpy(dv,householder[axis].data(),d*sizeof(double),cudaMemcpyHostToDevice), "upload additive Householder vector");
        matvec_kernel<<<(d+255)/256,256>>>(dk,dv,dw,d,d);
        cuda_check(cudaGetLastError(), "apply additive Householder matvec"); cuda_check(cudaDeviceSynchronize(), "apply additive Householder matvec");
        cuda_check(cudaMemcpy(result_host.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost), "read additive Householder matvec");
        double gamma=0.0; for(int i=0;i<d;++i) gamma+=householder[axis][i]*result_host[i];
        householder_kernel<<<grid,block>>>(dk,dv,dw,d,gamma);
        cuda_check(cudaGetLastError(), "apply additive Householder congruence"); cuda_check(cudaDeviceSynchronize(), "apply additive Householder congruence");
      }
      project_translation_rows<<<grid,block>>>(dk,d,n); cuda_check(cudaGetLastError(), "project additive translation rows"); cuda_check(cudaDeviceSynchronize(), "project additive translation rows");
      add_matrix<<<grid,block>>>(a,dk,d); cuda_check(cudaGetLastError(), "add projected mass-weighted Kadd"); cuda_check(cudaDeviceSynchronize(), "add projected mass-weighted Kadd");
      cuda_check(cudaFree(dk), "free weighted additive Kadd"); dk=nullptr;
      std::printf("    additive K/H coefficients loaded: beads=%d, epsilon=%.9g; baseline Hessian diagnostics retained\n", additive.beads, additive.epsilon);
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&physical),rr*sizeof(double)),"allocate physical Hessian");
    dim3 pgrid((r+15)/16,(r+15)/16);
    compact_physical<<<pgrid,block>>>(a,physical,d,r,n);
    cuda_check(cudaGetLastError(),"compact translation complement"); cuda_check(cudaDeviceSynchronize(),"compact translation complement");
    const double potrf_shift=use_additive?0.5*additive.epsilon:0.0;
    if (use_additive) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&shifted), rr*sizeof(double)), "allocate shifted D certificate reference");
      cuda_check(cudaMemcpy(shifted, physical, rr*sizeof(double), cudaMemcpyDeviceToDevice), "save shifted D certificate reference");
      subtract_diagonal<<<(r+255)/256,256>>>(shifted,r,potrf_shift);
      subtract_diagonal<<<(r+255)/256,256>>>(physical,r,potrf_shift);
      cuda_check(cudaGetLastError(), "apply additive epsilon/2 stability shift"); cuda_check(cudaDeviceSynchronize(), "apply additive epsilon/2 stability shift");
    }
    // Save exact D operator probes before POTRF; the stored D remains lossless.
    std::vector<std::vector<double>> probes(3,std::vector<double>(r)), products(3,std::vector<double>(r));
    for(int p=0;p<3;++p){
      double norm=0.0; for(int i=0;i<r;++i){probes[p][i]=std::sin((i+1)*(0.7548776662466927+p*0.173));norm+=probes[p][i]*probes[p][i];}
      for(double& x:probes[p])x/=std::sqrt(norm);
      cuda_check(cudaMemcpy(dv,probes[p].data(),r*sizeof(double),cudaMemcpyHostToDevice),"upload D probe");
      probe_product<<<(r+255)/256,256>>>(physical,dv,dw,r,r);
      cuda_check(cudaGetLastError(),"D probe product"); cuda_check(cudaDeviceSynchronize(),"D probe product");
      cuda_check(cudaMemcpy(products[p].data(),dw,r*sizeof(double),cudaMemcpyDeviceToHost),"read D probe product");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(potrf_work)*sizeof(double)),"allocate Cholesky workspace");
    solver_check(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,r,physical,r,work,potrf_work,dinfo),"cusolverDnDpotrf");
    cuda_check(cudaDeviceSynchronize(),"synchronize Cholesky");
    int info=0; cuda_check(cudaMemcpy(&info,dinfo,sizeof(info),cudaMemcpyDeviceToHost),"read Cholesky info");
    if(info!=0){
      if(info>0){
        const int j=info-1;
        const int original=j+1+(j>=n-1)+(j>=2*n-2);
        double original_diagonal=std::numeric_limits<double>::quiet_NaN(),failed_slot=std::numeric_limits<double>::quiet_NaN();
        cuda_check(cudaMemcpy(&original_diagonal,a+static_cast<std::size_t>(original)*d+original,sizeof(double),cudaMemcpyDeviceToHost),"read failed Hessian diagonal");
        cuda_check(cudaMemcpy(&failed_slot,physical+static_cast<std::size_t>(j)*r+j,sizeof(double),cudaMemcpyDeviceToHost),"read failed Cholesky slot");
        bool candidate_available=false;double candidate_rho=std::numeric_limits<double>::quiet_NaN(),candidate_residual=std::numeric_limits<double>::quiet_NaN();
        const bool fd4_policy = ref.mechanical_policy == "native_reference_transport;analytic_site_gradient_fd4_v1";
        std::ofstream report(failure_path,std::ios::out|std::ios::trunc);
        if(report){
          report<<std::setprecision(17)<<"qNEP rpmd_ja Cholesky failure\nraw_path: "<<raw_path<<"\nleading_minor_1based: "<<info<<"\ndimension: "<<r
            <<"\nPOTRF_operator: unshifted projected D minus diagonal shift\nPOTRF_diagonal_shift_eV_per_A2_per_amu: "<<potrf_shift
            <<"\noriginal_projected_diagonal: "<<original_diagonal<<"\ncholesky_failure_slot_value: "<<failed_slot
            <<"\nhessian_symmetry_relative_error: "<<ref.hessian_symmetry_relative_error<<"\nprojection_relative_change: "<<ref.projection_relative_change<<"\nraw_stats:";
          for(int i=0;i<18;++i)report<<"\n  ["<<i<<"] "<<raw.stats[i];
          report<<"\nraw_stencil_policy: "<<ref.mechanical_policy
            <<"\nraw_fd_stats: K_"<<(fd4_policy?"D2h_D22h":"h_h2")<<"_relative="<<raw.stats[4]
            <<" Cx_"<<(fd4_policy?"D2h_D22h":"h_h2")<<"_relative="<<raw.stats[6]
            <<" Cy_"<<(fd4_policy?"D2h_D22h":"h_h2")<<"_relative="<<raw.stats[7]
            <<" Cz_"<<(fd4_policy?"D2h_D22h":"h_h2")<<"_relative="<<raw.stats[8]
            <<" projected_D_consistency="<<raw.stats[14]<<" projected_D_"<<(fd4_policy?"D2h_D22h":"h_h2")
            <<"_step_consistency="<<raw.stats[15];
          try{
            std::vector<double> candidate(static_cast<std::size_t>(r),0.0);
            if(j>0){
              failure_column<<<(j+255)/256,256>>>(a,dv,d,n,j,j);cuda_check(cudaGetLastError(),"extract failure candidate column");cuda_check(cudaDeviceSynchronize(),"extract failure candidate column");
              blas_check(cublasDtrsv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,j,physical,r,dv,1),"failure candidate forward solve");
              blas_check(cublasDtrsv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,j,physical,r,dv,1),"failure candidate transpose solve");
              std::vector<double> prefix(static_cast<std::size_t>(j));cuda_check(cudaMemcpy(prefix.data(),dv,prefix.size()*sizeof(double),cudaMemcpyDeviceToHost),"read failure candidate prefix");
              for(int i=0;i<j;++i)candidate[i]=-prefix[i];
            }
            candidate[j]=1.0;
            double norm2=0.0;for(double x:candidate){if(!std::isfinite(x))throw std::runtime_error("non-finite prefix solve");norm2+=x*x;}
            if(!(norm2>0.0)||!std::isfinite(norm2))throw std::runtime_error("invalid candidate norm");
            const double inv=1.0/std::sqrt(norm2);for(double& x:candidate)x*=inv;
            std::vector<double> direction(static_cast<std::size_t>(d),0.0);
            for(int i=0;i<r;++i){const int k=i+1+(i>=n-1)+(i>=2*n-2);direction[k]=candidate[i];}
            cuda_check(cudaMemcpy(dv,direction.data(),direction.size()*sizeof(double),cudaMemcpyHostToDevice),"upload failure candidate");
            matvec_kernel<<<(d+255)/256,256>>>(a,dv,dw,d,d);cuda_check(cudaGetLastError(),"candidate direction product");cuda_check(cudaDeviceSynchronize(),"candidate direction product");
            std::vector<double> product(static_cast<std::size_t>(d));cuda_check(cudaMemcpy(product.data(),dw,product.size()*sizeof(double),cudaMemcpyDeviceToHost),"read candidate direction product");
            double rho=0.0, product2=0.0;for(int i=0;i<d;++i){rho+=direction[i]*product[i];product2+=product[i]*product[i];}
            double residual2=0.0;for(int i=0;i<d;++i){const double x=product[i]-rho*direction[i];residual2+=x*x;}
            if(!std::isfinite(rho)||!std::isfinite(residual2)||!std::isfinite(product2))throw std::runtime_error("non-finite candidate product");
            candidate_available=true;candidate_rho=rho;candidate_residual=std::sqrt(residual2);
            for(int axis=2;axis>=0;--axis){const auto& h=householder[axis];double dot=0.0;for(int i=0;i<d;++i)dot+=h[i]*direction[i];for(int i=0;i<d;++i)direction[i]-=2.0*dot*h[i];}
            std::vector<int> dominant;for(int i=0;i<d;++i)dominant.push_back(i);std::partial_sort(dominant.begin(),dominant.begin()+std::min(8,d),dominant.end(),[&](int x,int y){return std::abs(direction[x])>std::abs(direction[y]);});
            report<<"candidate_direction_status: available\nrho_eV_per_A2_per_amu: "<<rho<<"\ncandidate_direction_residual_norm: "<<candidate_residual
              <<"\noperator_product_norm: "<<std::sqrt(product2)<<"\nnormalization: unit Euclidean norm in translation-complement mass-weighted coordinates\n"
              <<"candidate_curvature_operator: "<<(use_additive?"unshifted projected D = raw Hessian plus additive fit":"unshifted projected D = raw Hessian")<<"; Householder-coordinate Rayleigh quotient, invariant under the orthogonal back-transform\n"
              <<"rho is the Rayleigh quotient of this candidate; residual is not an eigenvalue certificate.\n"
              <<"negative rho indicates a negative direction in the generated projected Hessian; it does not establish instability of the physical structure.\ncandidate_direction_original_mass_weighted "<<direction.size();
            for(double x:direction)report<<' '<<x;
            report<<"\ndominant_original_mass_weighted_components:";
            for(int q=0;q<std::min(8,d);++q){const int index=dominant[q];report<<"\n  atom "<<index%n<<" type "<<ref.types[index%n]<<" axis "<<"xyz"[index/n]<<" value "<<direction[index];}
          }catch(const std::exception& e){report<<"candidate_direction_status: unavailable ("<<e.what()<<")\n";}
          report.flush();
          if(!report)std::fprintf(stderr,"qNEP rpmd_ja: failed writing Cholesky diagnostic %s\n",failure_path.c_str());
        }else{
          std::fprintf(stderr,"qNEP rpmd_ja: cannot create Cholesky diagnostic %s\n",failure_path.c_str());
        }
        std::printf("    qNEP rpmd_ja Cholesky failure: leading minor %d/%d; original projected diagonal %.9g; failed factor slot %.9g; Hessian asymmetry %.3e; projection change %.3e; diagnostic %s\n",
          info,r,original_diagonal,failed_slot,ref.hessian_symmetry_relative_error,ref.projection_relative_change,failure_path.c_str());
        std::fflush(stdout);
        std::string spectrum;
        std::vector<RpmdJADiagnosticMode> modes;
        try {
          if (physical) { cuda_check(cudaFree(physical), "free failed Cholesky factor before spectrum diagnostic"); physical = nullptr; }
          if (work) { cuda_check(cudaFree(work), "free POTRF workspace before spectrum diagnostic"); work = nullptr; }
          spectrum = low_spectrum_diagnostic(a, d, r, n, solver, blas,
            candidate_available ? candidate_rho : std::numeric_limits<double>::quiet_NaN(), modes);
          {
            std::ofstream append(failure_path, std::ios::out | std::ios::app);
            if (append) { append << spectrum; append.flush(); if (!append) std::fprintf(stderr,"qNEP rpmd_ja: failed appending low-spectrum diagnostic %s\n",failure_path.c_str()); }
            std::printf("%s", spectrum.c_str());
          }
          std::vector<double> mode_norms, translation_residuals;
          for (auto& mode : modes) {
            for (int axis = 2; axis >= 0; --axis) {
              const auto& hvec = householder[axis];
              const double dot = std::inner_product(hvec.begin(), hvec.end(), mode.mass_weighted_direction.begin(), 0.0);
              for (int i = 0; i < d; ++i) mode.mass_weighted_direction[i] -= 2.0 * dot * hvec[i];
            }
            const double norm = std::sqrt(std::inner_product(mode.mass_weighted_direction.begin(), mode.mass_weighted_direction.end(), mode.mass_weighted_direction.begin(), 0.0));
            if (!(norm > 0.0) || !std::isfinite(norm)) throw std::runtime_error("invalid unrotated Ritz mode norm");
            mode_norms.push_back(norm);
            for (double& x : mode.mass_weighted_direction) x /= norm;
            double overlap2 = 0.0;
            double mass_sum = std::accumulate(ref.masses.begin(), ref.masses.end(), 0.0);
            for (int axis = 0; axis < 3; ++axis) {
              double overlap = 0.0;
              for (int i = 0; i < n; ++i) overlap += std::sqrt(ref.masses[i] / mass_sum) * mode.mass_weighted_direction[axis * n + i];
              overlap2 += overlap * overlap;
            }
            translation_residuals.push_back(std::sqrt(overlap2));
            if (translation_residuals.back() > 1.0e-8) throw std::runtime_error("unrotated Ritz mode has excessive translation overlap");
          }
          std::ofstream append(failure_path, std::ios::out | std::ios::app);
          if (append) {
            for (std::size_t q = 0; q < modes.size(); ++q) {
              const auto& mode = modes[q];
              append << "mode_" << q + 1 << "_vector_basis: original_cartesian_mass_weighted\n"
                << "mode_" << q + 1 << "_mass_weighted_norm_before_normalization: " << mode_norms[q] << '\n'
                << "mode_" << q + 1 << "_mass_weighted_norm_after_normalization: 1\n"
                << "mode_" << q + 1 << "_translation_overlap_norm: " << translation_residuals[q] << '\n';
              std::vector<int> dominant(d); std::iota(dominant.begin(), dominant.end(), 0);
              std::partial_sort(dominant.begin(), dominant.begin() + std::min(8, d), dominant.end(), [&](int x, int y) { return std::abs(mode.mass_weighted_direction[x]) > std::abs(mode.mass_weighted_direction[y]); });
              double fourth = 0.0, square = 0.0;
              for (double x : mode.mass_weighted_direction) { square += x*x; fourth += x*x*x*x; }
              append << "mode_" << q + 1 << "_IPR: " << fourth / (square*square) << "\nmode_" << q + 1 << "_dominant_components:\n";
              for (int j = 0; j < std::min(8, d); ++j) { const int ix=dominant[j]; append << "  atom " << ix%n << " type " << ref.types[ix%n] << " axis " << "xyz"[ix/n] << " E " << mode.mass_weighted_direction[ix] << '\n'; }
            }
            append.flush(); if (!append) std::fprintf(stderr,"qNEP rpmd_ja: failed appending low-spectrum diagnostic %s\n",failure_path.c_str());
          }
          for (const auto& mode : modes) std::printf("    qNEP rpmd_ja mode: lambda %.9g, residual %.3e, original Cartesian mass-weighted norm %.12g\n", mode.eigenvalue, mode.residual, std::sqrt(std::inner_product(mode.mass_weighted_direction.begin(),mode.mass_weighted_direction.end(),mode.mass_weighted_direction.begin(),0.0)));
          if (mode_validator && !use_additive && !modes.empty()) {
            try {
              const std::string validation = mode_validator(ref, modes);
              std::ofstream append(failure_path, std::ios::out | std::ios::app);
              if (append) { append << validation; append.flush(); }
              std::printf("%s", validation.c_str());
            } catch (const std::exception& e) {
              const std::string message = e.what();
              const std::string unavailable = std::string("\nmode_validation: unavailable (") + message + ")\nmode_validation_flags: " +
                (message.find("INPUT_MISMATCH:") == 0 ? "INPUT_MISMATCH\n" : "NUMERICAL_OR_NONLINEAR_UNRESOLVED\n");
              std::ofstream append(failure_path, std::ios::out | std::ios::app); if (append) append << unavailable;
              std::printf("%s", unavailable.c_str());
            }
          } else if (!mode_validator && !use_additive) {
            const std::string unavailable = "\nmode_validation: unavailable (no qNEP evaluator supplied)\n";
            std::ofstream append(failure_path, std::ios::out | std::ios::app); if (append) append << unavailable;
            std::printf("%s", unavailable.c_str());
          }
        } catch (const std::exception& e) {
          const std::string unavailable = spectrum.empty() ?
            std::string("\nlow_spectrum_status: unavailable (") + e.what() + ")\n" :
            std::string("\nmode_validation: unavailable (mode vector conversion/evaluator error: ") + e.what() + ")\n";
          std::ofstream append(failure_path, std::ios::out | std::ios::app);
          if (append) { append << unavailable; append.flush(); }
          std::printf("%s", unavailable.c_str());
        }
        if(candidate_available)std::printf("    qNEP rpmd_ja candidate direction: rho %.9g eV/A^2/amu; candidate-direction residual %.3e\n",candidate_rho,candidate_residual);
        else std::printf("    qNEP rpmd_ja candidate direction: unavailable\n");
        std::fflush(stdout);
      }
      throw std::runtime_error(info>0?"qNEP translation-complement Hessian is not positive definite (POTRF leading minor "+std::to_string(info)+" of dimension "+std::to_string(r)+", raw: "+raw_path+")":"cuSOLVER Cholesky parameter failure (info="+std::to_string(info)+", invalid parameter index "+std::to_string(-info)+")");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&pivot_device),r*sizeof(double)),"allocate Cholesky pivots");
    diagonal_values<<<(r+255)/256,256>>>(physical,pivot_device,r);cuda_check(cudaGetLastError(),"read Cholesky pivots");cuda_check(cudaDeviceSynchronize(),"read Cholesky pivots");
    std::vector<double> pivots(r);cuda_check(cudaMemcpy(pivots.data(),pivot_device,r*sizeof(double),cudaMemcpyDeviceToHost),"copy Cholesky pivots");
    cuda_check(cudaFree(pivot_device),"free Cholesky pivots");pivot_device=nullptr;
    double min_pivot=std::numeric_limits<double>::infinity();for(double x:pivots){if(!(x>0.0)||!std::isfinite(x))throw std::runtime_error("invalid Cholesky pivot");min_pivot=std::min(min_pivot,x);}
    double additive_eta=0.0;
    if (use_additive) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&reconstructed_device),rr*sizeof(double)), "allocate full shifted reconstruction");
      zero_upper_column_major<<<(static_cast<int>(rr)+255)/256,256>>>(physical,r);
      cuda_check(cudaGetLastError(), "clear unused Cholesky upper triangle"); cuda_check(cudaDeviceSynchronize(), "clear unused Cholesky upper triangle");
      const double one=1.0, zero=0.0;
      blas_check(cublasDgemm(blas,CUBLAS_OP_N,CUBLAS_OP_T,r,r,r,&one,physical,r,physical,r,&zero,reconstructed_device,r), "reconstruct shifted Cholesky factor");
      std::vector<double> expected(static_cast<std::size_t>(tile_size)*tile_size), reconstructed(expected.size());
      double error2=0.0;
      for(int i=0;i<r;i+=tile_size)for(int j=0;j<r;j+=tile_size){
        const int nr=std::min(tile_size,r-i),nc=std::min(tile_size,r-j);
        cuda_check(cudaMemcpy2D(expected.data(),tile_size*sizeof(double),shifted+static_cast<std::size_t>(i)*r+j,r*sizeof(double),nc*sizeof(double),nr,cudaMemcpyDeviceToHost),"read shifted certificate tile");
        cuda_check(cudaMemcpy2D(reconstructed.data(),tile_size*sizeof(double),reconstructed_device+static_cast<std::size_t>(i)*r+j,r*sizeof(double),nc*sizeof(double),nr,cudaMemcpyDeviceToHost),"read Cholesky reconstruction tile");
        for(int x=0;x<nr;++x)for(int y=0;y<nc;++y){const std::size_t q=static_cast<std::size_t>(x)*tile_size+y;const double e=expected[q]-reconstructed[q];error2+=e*e;}
      }
      additive_eta=std::sqrt(error2);
      if(!std::isfinite(additive_eta)||!(additive_eta<0.5*additive.epsilon))
        throw std::runtime_error("additive shifted full-Frobenius Cholesky reconstruction bound must be below epsilon/2");
      cuda_check(cudaFree(shifted), "free shifted D certificate reference"); shifted=nullptr;
    }
    // Reconstruct only three vectors with triangular BLAS calls; no full factor reaches the host.
    double reconstruction=0.0;
    std::vector<double> reconstructed(r);
    for(int p=0;p<3;++p){cuda_check(cudaMemcpy(dv,probes[p].data(),r*sizeof(double),cudaMemcpyHostToDevice),"upload Cholesky probe");blas_check(cublasDtrmv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,r,physical,r,dv,1),"Cholesky probe L transpose");blas_check(cublasDtrmv(blas,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,r,physical,r,dv,1),"Cholesky probe L");cuda_check(cudaMemcpy(reconstructed.data(),dv,r*sizeof(double),cudaMemcpyDeviceToHost),"read Cholesky reconstruction");double e2=0.0,n2=0.0;for(int i=0;i<r;++i){e2+=(reconstructed[i]-products[p][i])*(reconstructed[i]-products[p][i]);n2+=products[p][i]*products[p][i];}reconstruction=std::max(reconstruction,std::sqrt(e2/std::max(n2,1.0e-300)));}
    if(!(reconstruction<=1.0e-8))throw std::runtime_error("Cholesky probe reconstruction residual exceeds 1e-8");
    cuda_check(cudaFree(work),"free Cholesky workspace"); work=nullptr;
    cuda_check(cudaFree(physical),"free Cholesky factor after probes");physical=nullptr;
    // Restore the projected Cartesian operator for v3; only its complement copy was factored.
    for(int axis=2;axis>=0;--axis){cuda_check(cudaMemcpy(dv,householder[axis].data(),d*sizeof(double),cudaMemcpyHostToDevice),"restore Cartesian D basis");matvec_kernel<<<(d+255)/256,256>>>(a,dv,dw,d,d);cuda_check(cudaGetLastError(),"restore D basis matvec");cuda_check(cudaDeviceSynchronize(),"restore D basis matvec");cuda_check(cudaMemcpy(result_host.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost),"read restore D matvec");double gamma=0.0;for(int i=0;i<d;++i)gamma+=householder[axis][i]*result_host[i];householder_kernel<<<grid,block>>>(a,dv,dw,d,gamma);cuda_check(cudaGetLastError(),"restore Cartesian D basis");cuda_check(cudaDeviceSynchronize(),"restore Cartesian D basis");}
    if (use_additive) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&dk),dd*sizeof(double)), "allocate stored-D certificate transform");
      cuda_check(cudaMemcpy(dk,a,dd*sizeof(double),cudaMemcpyDeviceToDevice), "copy final unshifted stored D");
      for(int axis=0;axis<3;++axis){
        cuda_check(cudaMemcpy(dv,householder[axis].data(),d*sizeof(double),cudaMemcpyHostToDevice), "upload stored-D Householder vector");
        matvec_kernel<<<(d+255)/256,256>>>(dk,dv,dw,d,d);cuda_check(cudaGetLastError(), "stored-D Householder matvec");cuda_check(cudaDeviceSynchronize(), "stored-D Householder matvec");
        cuda_check(cudaMemcpy(result_host.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost), "read stored-D Householder matvec");
        double gamma=0.0;for(int i=0;i<d;++i)gamma+=householder[axis][i]*result_host[i];
        householder_kernel<<<grid,block>>>(dk,dv,dw,d,gamma);cuda_check(cudaGetLastError(), "stored-D Householder congruence");cuda_check(cudaDeviceSynchronize(), "stored-D Householder congruence");
      }
      project_translation_rows<<<grid,block>>>(dk,d,n);cuda_check(cudaGetLastError(), "project stored-D certificate translation");cuda_check(cudaDeviceSynchronize(), "project stored-D certificate translation");
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&shifted),rr*sizeof(double)), "allocate stored-D internal certificate");
      compact_physical<<<pgrid,block>>>(dk,shifted,d,r,n);cuda_check(cudaGetLastError(), "compact stored-D certificate");cuda_check(cudaDeviceSynchronize(), "compact stored-D certificate");
      subtract_diagonal<<<(r+255)/256,256>>>(shifted,r,0.5*additive.epsilon);cuda_check(cudaGetLastError(), "shift stored-D certificate");cuda_check(cudaDeviceSynchronize(), "shift stored-D certificate");
      std::vector<double> stored_internal(static_cast<std::size_t>(tile_size)*tile_size), reconstructed(static_cast<std::size_t>(tile_size)*tile_size);
      double error2=0.0;
      for(int i=0;i<r;i+=tile_size)for(int j=0;j<r;j+=tile_size){
        const int nr=std::min(tile_size,r-i),nc=std::min(tile_size,r-j);
        cuda_check(cudaMemcpy2D(stored_internal.data(),tile_size*sizeof(double),shifted+static_cast<std::size_t>(i)*r+j,r*sizeof(double),nc*sizeof(double),nr,cudaMemcpyDeviceToHost),"read stored-D certificate tile");
        cuda_check(cudaMemcpy2D(reconstructed.data(),tile_size*sizeof(double),reconstructed_device+static_cast<std::size_t>(i)*r+j,r*sizeof(double),nc*sizeof(double),nr,cudaMemcpyDeviceToHost),"read retained Cholesky reconstruction tile");
        for(int x=0;x<nr;++x)for(int y=0;y<nc;++y){const std::size_t q=static_cast<std::size_t>(x)*tile_size+y;const double e=stored_internal[q]-reconstructed[q];error2+=e*e;}
      }
      additive_eta=std::sqrt(error2);
      if(!std::isfinite(additive_eta)||!(additive_eta<0.5*additive.epsilon))
        throw std::runtime_error("stored additive D shifted full-Frobenius bound must be below epsilon/2");
      const double certificate_shift=0.5*additive.epsilon;
      const double certified_lower_bound=certificate_shift-additive_eta;
      if(!std::isfinite(certified_lower_bound)||!(certified_lower_bound>0.0))
        throw std::runtime_error("stored additive D certificate does not prove positive definiteness");
      std::printf("    additive stored-D certificate: Cholesky shift=%.17g, Frobenius reconstruction bound eta=%.17g, certified lambda_min lower bound shift-eta=%.17g (SPD only; not an epsilon eigenvalue floor).\n",
        certificate_shift,additive_eta,certified_lower_bound);
      cuda_check(cudaFree(dk), "free stored-D certificate transform");dk=nullptr;
      cuda_check(cudaFree(shifted), "free stored-D internal certificate");shifted=nullptr;
      cuda_check(cudaFree(reconstructed_device), "free retained Cholesky reconstruction");reconstructed_device=nullptr;
    }
    matrix_projection_sums<<<(d+255)/256,256>>>(a,dw,d,n);cuda_check(cudaGetLastError(),"measure projected D norm");cuda_check(cudaDeviceSynchronize(),"measure projected D norm");cuda_check(cudaMemcpy(projection_sums.data(),dw,projection_sums.size()*sizeof(double),cudaMemcpyDeviceToHost),"read projected D norm");
    total2=0.0;for(int i=0;i<d;++i)total2+=projection_sums[i];
    cuda_check(cudaMemcpy(dm,mass_translation.data(),n*sizeof(double),cudaMemcpyHostToDevice),"upload normalized mass translation");
    double translation2=0.0;for(int axis=0;axis<3;++axis){translation_product<<<(d+255)/256,256>>>(a,dm,dw,d,n,axis);cuda_check(cudaGetLastError(),"measure projected translation");cuda_check(cudaDeviceSynchronize(),"measure projected translation");cuda_check(cudaMemcpy(result_host.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost),"read projected translation");for(double x:result_host)translation2+=x*x;}
    const double translation_residual=std::sqrt(translation2/std::max(total2,1.0e-300));if(!std::isfinite(translation_residual)||!(translation_residual<=1.0e-8))throw std::runtime_error("projected D translation residual exceeds 1e-8");
    cuda_check(cudaFree(dm),"free mass vector");dm=nullptr;
    row_sums<<<(d+255)/256,256>>>(a,dw,d); cuda_check(cudaGetLastError(),"compute D row bounds"); cuda_check(cudaDeviceSynchronize(),"compute D row bounds");
    std::vector<double> bounds(d); cuda_check(cudaMemcpy(bounds.data(),dw,d*sizeof(double),cudaMemcpyDeviceToHost),"read D row bounds");
    ref.spectral_bound=*std::max_element(bounds.begin(),bounds.end());
    const double kernel_required=HBAR/(K_B*ref.temperature)*std::sqrt(ref.spectral_bound);
    if(!std::isfinite(ref.spectral_bound)||!std::isfinite(kernel_required)||!(kernel_required<=ref.kernel_u*(1.0+32.0*std::numeric_limits<double>::epsilon())))throw std::runtime_error("qNEP kernel table does not cover lossless D row bound");
    ref.minimum_cholesky_pivot=min_pivot; ref.relative_operator_bound=0.0;
    ref.stability_certificate=use_additive?"additive_shifted_frobenius_v1":"cholesky_relative_bound"; ref.stability_checked=true;
    ref.additive_reconstruction_bound=additive_eta;
    ref.block_relative_residual[0]=0.0;
    for(int aidx=0;aidx<3;++aidx)ref.block_relative_residual[aidx+1]=0.0;
    ref.site_transport_relative_error[0]=ref.site_transport_relative_error[1]=ref.site_transport_relative_error[2]=0.0;

    std::ofstream out(tmp_path,std::ios::binary|std::ios::trunc); if(!out)throw std::runtime_error("cannot create qNEP v3 temporary"); created=tmp_path;
    const std::streampos diagnostics_position=write_rpmd_ja_qnep_v3_stream_prefix(out,ref);
    const std::uint64_t count=static_cast<std::uint64_t>((d+tile_size-1)/tile_size)*((d+tile_size-1)/tile_size);
    write_value(out,tile_size);write_value(out,count);
    std::vector<double> tile(static_cast<std::size_t>(tile_size)*tile_size);
    for(int i=0;i<d;i+=tile_size)for(int j=0;j<d;j+=tile_size){const int nr=std::min(tile_size,d-i),nc=std::min(tile_size,d-j);vectorize_tile<<<dim3((nr+15)/16,(nc+15)/16),block>>>(a,tilebuf,d,i,j,nr,nc);cuda_check(cudaGetLastError(),"read D tile");cuda_check(cudaDeviceSynchronize(),"read D tile");cuda_check(cudaMemcpy(tile.data(),tilebuf,static_cast<std::size_t>(nr)*nc*sizeof(double),cudaMemcpyDeviceToHost),"copy D tile");write_tile(out,dense_tile(i,j,nr,nc,tile.data()));}

    // Free the dense D GPU workspace before processing H tiles.
    cuda_check(cudaFree(a),"free D matrix"); a=nullptr;if(physical){cuda_check(cudaFree(physical),"free physical D");physical=nullptr;}cuda_check(cudaFree(tilebuf),"free D tile buffer");tilebuf=nullptr;
    if(!out)throw std::runtime_error("failed writing qNEP D tiles");
    // H tiles are appended in matrix order; row/column sizes stay bounded by 128.
    std::ifstream vf(raw_path,std::ios::binary),cf(raw_path,std::ios::binary),vcf(raw_path,std::ios::binary),ccf(raw_path,std::ios::binary);
    const std::streamoff vstart=raw.data, cstart=raw.data+static_cast<std::streamoff>(static_cast<std::uint64_t>(d)*n)*sizeof(double);
    const std::streamoff vcstart=raw.coarse_v, ccstart=raw.coarse_c;
    const std::uint64_t ddoubles=static_cast<std::uint64_t>(d)*d;
    std::vector<double> energies(n), fenergy(d), coarse_energy(n), fcoarse(d);
    for(int i=0;i<d;++i){vf.seekg(vstart+static_cast<std::streamoff>(i)*n*sizeof(double));vf.read(reinterpret_cast<char*>(energies.data()),n*sizeof(double));if(!vf)throw std::runtime_error("read fine site derivative");fenergy[i]=-std::accumulate(energies.begin(),energies.end(),0.0);vcf.seekg(vcstart+static_cast<std::streamoff>(i)*n*sizeof(double));vcf.read(reinterpret_cast<char*>(coarse_energy.data()),n*sizeof(double));if(!vcf)throw std::runtime_error("read coarse site derivative");fcoarse[i]=-std::accumulate(coarse_energy.begin(),coarse_energy.end(),0.0);}
    std::vector<double> transport_diff2(3,0.0),transport_norm2(3,0.0); std::vector<double> crows(static_cast<std::size_t>(tile_size)*d), ccrows(static_cast<std::size_t>(tile_size)*d);
    int lwork=0, maxdim=tile_size; solver_check(cusolverDnDgesvd_bufferSize(solver,maxdim,maxdim,&lwork),"cusolverDnDgesvd_bufferSize");
    if(lwork<=0)throw std::runtime_error("cuSOLVER returned a nonpositive SVD workspace size");

    cuda_check(cudaMalloc(reinterpret_cast<void**>(&sm),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD matrix");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&su),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD U");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&svt),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD VT");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&sw),static_cast<std::size_t>(lwork)*sizeof(double)),"allocate SVD workspace");cuda_check(cudaMalloc(reinterpret_cast<void**>(&sinfo),sizeof(int)),"allocate SVD info");cuda_check(cudaMalloc(reinterpret_cast<void**>(&dsingular),static_cast<std::size_t>(maxdim)*sizeof(double)),"allocate singular values");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dleft),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD left verify buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dright),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD right verify buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&dproduct),static_cast<std::size_t>(maxdim)*maxdim*sizeof(double)),"allocate SVD reconstruction buffer");
    double tile_error[3]={};
    std::uint64_t h_v_bytes=0;
    for(int alpha=0;alpha<3;++alpha){
      write_value(out,tile_size);write_value(out,count);
      for(int j=0;j<d;j+=tile_size){const int nc=std::min(tile_size,d-j);
        for(int y=0;y<nc;++y){cf.clear();cf.seekg(cstart+static_cast<std::streamoff>(alpha*ddoubles+static_cast<std::uint64_t>(j+y)*d)*sizeof(double));cf.read(reinterpret_cast<char*>(crows.data()+static_cast<std::size_t>(y)*d),d*sizeof(double));ccf.clear();ccf.seekg(ccstart+static_cast<std::streamoff>(alpha*ddoubles+static_cast<std::uint64_t>(j+y)*d)*sizeof(double));ccf.read(reinterpret_cast<char*>(ccrows.data()+static_cast<std::size_t>(y)*d),d*sizeof(double));if(!cf||!ccf)throw std::runtime_error("read qNEP virial derivative rows");}
      const int atom_start=std::max(0,j-alpha*n),atom_end=std::min(n,j+nc-alpha*n);
      const int atom_width=std::max(0,atom_end-atom_start);
      std::vector<double> vrows(static_cast<std::size_t>(tile_size)*atom_width), vcrows(vrows.size());
      // Internal policy keeps raw ambient Bt/fenergy; cached E^T Bt E realizes the P pullback.
      for(int i=0;i<d;i+=tile_size){const int nr=std::min(tile_size,d-i);
        if(atom_width>0){vf.clear();vf.seekg(vstart+static_cast<std::streamoff>(i)*n*sizeof(double)+static_cast<std::streamoff>(atom_start)*sizeof(double));for(int x=0;x<nr;++x){vf.read(reinterpret_cast<char*>(vrows.data()+static_cast<std::size_t>(x)*atom_width),static_cast<std::streamsize>(atom_width)*sizeof(double));if(!vf)throw std::runtime_error("read H fine V slice");vf.seekg(static_cast<std::streamoff>(n-atom_width)*sizeof(double),std::ios::cur);}vcf.clear();vcf.seekg(vcstart+static_cast<std::streamoff>(i)*n*sizeof(double)+static_cast<std::streamoff>(atom_start)*sizeof(double));for(int x=0;x<nr;++x){vcf.read(reinterpret_cast<char*>(vcrows.data()+static_cast<std::size_t>(x)*atom_width),static_cast<std::streamsize>(atom_width)*sizeof(double));if(!vcf)throw std::runtime_error("read H coarse V slice");vcf.seekg(static_cast<std::streamoff>(n-atom_width)*sizeof(double),std::ios::cur);}h_v_bytes+=static_cast<std::uint64_t>(2)*nr*atom_width*sizeof(double);}
        std::vector<double> h(static_cast<std::size_t>(nr)*nc),hc(h.size());
        for(int x=0;x<nr;++x)for(int y=0;y<nc;++y){const int rr0=i+x,cc0=j+y,site=rr0%n,atom=cc0%n,nu=cc0/n;const double mass=std::sqrt(ref.masses[site]*ref.masses[atom]);double v=0,vc=0;if(nu==alpha){v=vrows[static_cast<std::size_t>(x)*atom_width+atom-atom_start];vc=vcrows[static_cast<std::size_t>(x)*atom_width+atom-atom_start];}double q=crows[static_cast<std::size_t>(y)*d+rr0],qc=ccrows[static_cast<std::size_t>(y)*d+rr0];if(nu==alpha&&site==atom){q-=fenergy[rr0];qc-=fcoarse[rr0];}h[static_cast<std::size_t>(x)*nc+y]=(q-v)/mass;hc[static_cast<std::size_t>(x)*nc+y]=(qc-vc)/mass;}
        if(!std::all_of(h.begin(),h.end(),[](double x){return std::isfinite(x);})||!std::all_of(hc.begin(),hc.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("qNEP raw H tile contains non-finite values");
        for(std::size_t k=0;k<h.size();++k){const double dx=h[k]-hc[k];transport_diff2[alpha]+=dx*dx;transport_norm2[alpha]+=h[k]*h[k];}
        if(use_additive) for(int x=0;x<nr;++x) {
          const int row=i+x;
          const auto& sparse=additive.h[alpha];
          auto first=std::lower_bound(sparse.columns.begin()+sparse.offsets[row],
                                      sparse.columns.begin()+sparse.offsets[row+1],j);
          auto last=std::lower_bound(first,sparse.columns.begin()+sparse.offsets[row+1],j+nc);
          for(auto it=first;it!=last;++it){
            const std::size_t p=static_cast<std::size_t>(it-sparse.columns.begin());
            const int col=*it;
            const double increment=sparse.values[p]/std::sqrt(ref.masses[row%n]*ref.masses[col%n]);
            h[static_cast<std::size_t>(x)*nc+col-j]+=increment;
            hc[static_cast<std::size_t>(x)*nc+col-j]+=increment;
          }
        }
        RpmdJAMatrixTile result;result.row=i;result.column=j;result.rows=nr;result.columns=nc;
        double tile_residual=0.0;
        svd_tile(solver,blas,h.data(),nr,nc,result,sw,lwork,sm,dsingular,su,svt,sinfo,dleft,dright,dproduct,tile_residual);
        tile_error[alpha]=std::max(tile_error[alpha],tile_residual);
        write_tile(out,result);
      }
      }
    }
    const std::uint64_t fenergy_v_bytes=static_cast<std::uint64_t>(2)*d*n*sizeof(double);
    printf("    qNEP H V logical reads: slices=%llu bytes, fenergy=%llu bytes, total=%llu bytes.\n",static_cast<unsigned long long>(h_v_bytes),static_cast<unsigned long long>(fenergy_v_bytes),static_cast<unsigned long long>(h_v_bytes+fenergy_v_bytes));
    cuda_check(cudaFree(sm),"free SVD matrix");sm=nullptr;cuda_check(cudaFree(su),"free SVD U");su=nullptr;cuda_check(cudaFree(svt),"free SVD VT");svt=nullptr;cuda_check(cudaFree(sw),"free SVD workspace");sw=nullptr;cuda_check(cudaFree(sinfo),"free SVD info");sinfo=nullptr;cuda_check(cudaFree(dsingular),"free singular values");dsingular=nullptr;
    cuda_check(cudaFree(dleft),"free SVD verify left");dleft=nullptr;cuda_check(cudaFree(dright),"free SVD verify right");dright=nullptr;cuda_check(cudaFree(dproduct),"free SVD verify product");dproduct=nullptr;
    for(int alpha=0;alpha<3;++alpha){ref.site_transport_relative_error[alpha]=std::sqrt(transport_diff2[alpha]/std::max(transport_norm2[alpha],1.0e-300));ref.site_transport_difference_absolute_rms[alpha]=std::sqrt(transport_diff2[alpha]/ddoubles);ref.block_relative_residual[alpha+1]=tile_error[alpha];if(!std::isfinite(ref.site_transport_relative_error[alpha])||!(ref.site_transport_relative_error[alpha]<=raw_difference_tolerance)||!std::isfinite(ref.site_transport_difference_absolute_rms[alpha])||!std::isfinite(tile_error[alpha])||!(tile_error[alpha]<=tile_tolerance))throw std::runtime_error("qNEP native H fine/coarse or tile reconstruction check exceeds tolerance");}
    reconstruction=std::max(reconstruction,*std::max_element(tile_error,tile_error+3));
    const double diagnostics[21]={ref.energy_gradient_relative_error,ref.force_gradient_relative_error,ref.hessian_symmetry_relative_error,ref.energy_second_probe_relative_error,ref.force_balance_residual,ref.projection_relative_change,ref.energy_gradient_absolute_rms,ref.force_gradient_absolute_rms,ref.site_derivative_absolute_rms[0],ref.site_derivative_absolute_rms[1],ref.site_derivative_absolute_rms[2],ref.site_transport_difference_absolute_rms[0],ref.site_transport_difference_absolute_rms[1],ref.site_transport_difference_absolute_rms[2],ref.site_transport_relative_error[0],ref.site_transport_relative_error[1],ref.site_transport_relative_error[2],ref.block_relative_residual[0],ref.block_relative_residual[1],ref.block_relative_residual[2],ref.block_relative_residual[3]};
    out.seekp(diagnostics_position);out.write(reinterpret_cast<const char*>(diagnostics),sizeof(diagnostics));out.seekp(0,std::ios::end);
    write_value(out,1);out.flush();if(!out)throw std::runtime_error("failed flushing qNEP v3 output");out.close();if(!out)throw std::runtime_error("failed closing qNEP v3 output");
    if(std::rename(tmp_path.c_str(),output_path.c_str())!=0)throw std::runtime_error("cannot finalize qNEP v3 output");created=output_path;
    cuda_check(cudaFree(dv),"free probe vector");dv=nullptr;cuda_check(cudaFree(dw),"free probe result");dw=nullptr;cuda_check(cudaFree(dinfo),"free solver info");dinfo=nullptr;
    std::ofstream side(side_tmp,std::ios::trunc);if(!side)throw std::runtime_error("cannot create qNEP stability sidecar");
    if(use_additive) side<<"GPUMDJA_QNEP_STABILITY 4\nfingerprint "<<std::hex<<rpmd_ja_model_fingerprint(output_path)<<"\nconfig_fingerprint "<<raw.config<<std::dec<<"\natoms "<<n<<"\nderivative_policy "<<ref.mechanical_policy<<"\ncertificate additive_shifted_frobenius_v1\nepsilon_num "<<std::setprecision(17)<<additive.epsilon<<"\nreconstruction_bound "<<additive_eta<<"\nminimum_cholesky_pivot "<<min_pivot<<"\nrelative_operator_bound 0\ntranslation_residual "<<translation_residual<<"\nreconstruction_residual "<<reconstruction<<"\nsoftmode_relative_error 0\n";
    else side<<"GPUMDJA_QNEP_STABILITY 3\nfingerprint "<<std::hex<<rpmd_ja_model_fingerprint(output_path)<<"\nconfig_fingerprint "<<raw.config<<std::dec<<"\natoms "<<n<<"\nderivative_policy "<<ref.mechanical_policy<<"\ncertificate cholesky_relative_bound\nminimum_cholesky_pivot "<<std::setprecision(17)<<min_pivot<<"\nrelative_operator_bound 0\ntranslation_residual "<<translation_residual<<"\nreconstruction_residual "<<reconstruction<<"\nsoftmode_relative_error 0\n";
    side.flush();if(!side)throw std::runtime_error("failed writing qNEP stability sidecar");side.close();if(std::rename(side_tmp.c_str(),sidecar_path.c_str())!=0)throw std::runtime_error("cannot finalize qNEP stability sidecar");created.clear();
  } catch (...) {
    if(dk)cudaFree(dk);if(a)cudaFree(a);if(physical)cudaFree(physical);if(shifted)cudaFree(shifted);if(reconstructed_device)cudaFree(reconstructed_device);if(dv)cudaFree(dv);if(dw)cudaFree(dw);if(dm)cudaFree(dm);if(dinfo)cudaFree(dinfo);if(work)cudaFree(work);if(query_matrix)cudaFree(query_matrix);if(pivot_device)cudaFree(pivot_device);if(tilebuf)cudaFree(tilebuf);if(sm)cudaFree(sm);if(su)cudaFree(su);if(svt)cudaFree(svt);if(sw)cudaFree(sw);if(dleft)cudaFree(dleft);if(dright)cudaFree(dright);if(dproduct)cudaFree(dproduct);if(dsingular)cudaFree(dsingular);if(sinfo)cudaFree(sinfo);if(blas)cublasDestroy(blas);if(solver)cusolverDnDestroy(solver);
    std::remove(tmp_path.c_str());std::remove(side_tmp.c_str());if(created==output_path)std::remove(output_path.c_str());throw;
  }
  if(blas)blas_check(cublasDestroy(blas),"cublasDestroy");
  if(solver)solver_check(cusolverDnDestroy(solver),"cusolverDnDestroy");
#endif
}
