#include "interface_shc.cuh"
#include "interface_geometry.cuh"
#include "interface_sampling_time.cuh"
#include "interface_shc_stats.cuh"
#include "force/force.cuh"
#include "force/nep.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr double EV_PER_PS_A2_TO_W_PER_M2 = 1.602176634e13;

size_t checked_product(const size_t a, const size_t b)
{
  if (a && b > std::numeric_limits<size_t>::max() / a) {
    PRINT_INPUT_ERROR("compute_interface_shc allocation overflows size_t.\n");
  }
  return a * b;
}

size_t checked_sum(const size_t a, const size_t b)
{
  if (b > std::numeric_limits<size_t>::max() - a) {
    PRINT_INPUT_ERROR("compute_interface_shc allocation overflows size_t.\n");
  }
  return a + b;
}

bool output_exists(const std::string& path)
{
  FILE* file = std::fopen(path.c_str(), "r");
  if (!file) return false;
  std::fclose(file);
  return true;
}

__global__ void gpu_store_interface_shc_sample(
  const int selected_count,
  const int plane_count,
  const int total_atoms,
  const int ring_slot,
  const int* selected_atoms,
  const int* slot_planes,
  const double* coefficients,
  const double* velocity,
  const double velocity_conversion,
  int* error_flag,
  int* first_error_step,
  const int step,
  double* coefficient_history,
  double* velocity_history,
  double* plane_flux_row)
{
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= selected_count) return;
  const int atom = selected_atoms[slot];
  const int plane = slot_planes[slot];
  if (plane < 0 || plane >= plane_count) return;
  double flux = 0.0;
  for (int d = 0; d < 3; ++d) {
    const size_t history_index =
      (static_cast<size_t>(ring_slot) * selected_count + slot) * 3 + d;
    const double coefficient = coefficients[static_cast<size_t>(slot) * 3 + d];
    const double speed = velocity[static_cast<size_t>(d) * total_atoms + atom] * velocity_conversion;
    if (!isfinite(coefficient) || !isfinite(speed)) {
      nep_plane_flux_record_error(
        error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
      return;
    }
    coefficient_history[history_index] = coefficient;
    velocity_history[history_index] = speed;
    flux += coefficient * speed;
  }
  if (!isfinite(flux)) {
    nep_plane_flux_record_error(
      error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
    return;
  }
  const double old_flux = atomicAdd(&plane_flux_row[plane], flux);
  if (!isfinite(old_flux + flux)) {
    nep_plane_flux_record_error(
      error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
  }
}

__global__ void gpu_clear_interface_shc_flux_row(const int plane_count, double* plane_flux_row)
{
  const int plane = blockIdx.x * blockDim.x + threadIdx.x;
  if (plane < plane_count) plane_flux_row[plane] = 0.0;
}

__global__ void gpu_reset_interface_shc_error(int* error_info)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    nep_plane_flux_reset_error(error_info, error_info + 1);
  }
}

__global__ void gpu_accumulate_interface_shc_correlation(
  const int selected_count,
  const int plane_count,
  const int correlation_samples,
  const int ring_slot,
  const int valid_lags,
  const int* slot_planes,
  const double* coefficient_history,
  const double* velocity_history,
  int* error_flag,
  int* first_error_step,
  const int step,
  double* correlation_plus,
  double* correlation_minus)
{
  const int selected = blockIdx.x * blockDim.x + threadIdx.x;
  if (selected >= selected_count) return;
  const int plane = slot_planes[selected];
  if (plane < 0 || plane >= plane_count) return;
  const size_t current_base =
    (static_cast<size_t>(ring_slot) * selected_count + selected) * 3;
  double zero = 0.0;
  for (int d = 0; d < 3; ++d) {
    zero += coefficient_history[current_base + d] * velocity_history[current_base + d];
  }
  if (!isfinite(zero)) {
    nep_plane_flux_record_error(
      error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
    return;
  }
  const size_t zero_index = static_cast<size_t>(plane) * correlation_samples;
  const double old_zero = atomicAdd(&correlation_plus[zero_index], zero);
  if (!isfinite(old_zero + zero)) {
    nep_plane_flux_record_error(
      error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
  }

  for (int lag = 1; lag <= valid_lags; ++lag) {
    int old_slot = ring_slot - lag;
    if (old_slot < 0) old_slot += correlation_samples;
    const size_t old_base = (static_cast<size_t>(old_slot) * selected_count + selected) * 3;
    double positive = 0.0;
    double negative = 0.0;
    for (int d = 0; d < 3; ++d) {
      positive += coefficient_history[old_base + d] * velocity_history[current_base + d];
      negative += coefficient_history[current_base + d] * velocity_history[old_base + d];
    }
    if (!isfinite(positive) || !isfinite(negative)) {
      nep_plane_flux_record_error(
        error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
      return;
    }
    const size_t output = static_cast<size_t>(plane) * correlation_samples + lag;
    const double old_positive = atomicAdd(&correlation_plus[output], positive);
    const double old_negative = atomicAdd(&correlation_minus[output], negative);
    if (!isfinite(old_positive + positive) || !isfinite(old_negative + negative)) {
      nep_plane_flux_record_error(
        error_flag, first_error_step, NEP_PLANE_FLUX_ERROR_NONFINITE, step);
    }
  }
}
} // namespace

ComputeInterfaceSHC::ComputeInterfaceSHC(
  const std::vector<std::string>& tokens,
  const std::shared_ptr<InterfaceGeometry>& interface_geometry)
  : interface_geometry_(interface_geometry)
{
  action_name = "compute_interface_shc";
  if (tokens.size() != 8 ||
      !is_valid_int(tokens[1], &sample_interval_) ||
      !is_valid_int(tokens[2], &correlation_samples_) ||
      !is_valid_int(tokens[3], &block_samples_) ||
      !is_valid_int(tokens[4], &num_omega_) ||
      !is_valid_real(tokens[5], &max_omega_rad_ps_) ||
      !is_valid_real(tokens[6], &max_displacement_A_) || tokens[7].empty()) {
    PRINT_INPUT_ERROR(
      "compute_interface_shc syntax: <sample_interval> <Nc> <block_samples> <numomega> "
      "<max_omega_rad_ps> <max_reference_displacement_A> <prefix>.\n");
  }
  if (sample_interval_ <= 0 || correlation_samples_ <= 0 ||
      static_cast<long long>(block_samples_) < 2LL * correlation_samples_ || num_omega_ < 2 ||
      !std::isfinite(max_omega_rad_ps_) || max_omega_rad_ps_ <= 0.0 ||
      !std::isfinite(max_displacement_A_) || max_displacement_A_ < 0.0) {
    PRINT_INPUT_ERROR("Invalid compute_interface_shc sampling or displacement parameter.\n");
  }
  prefix_ = tokens[7];
}

void ComputeInterfaceSHC::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!interface_geometry_ || !interface_geometry_->initialized() ||
      interface_geometry_->interfaces_A().empty()) {
    PRINT_INPUT_ERROR("compute_interface_shc requires shared geometry with at least one interface plane.\n");
  }
  const EnsembleType ensemble = integrate.get_type();
  if (ensemble != EnsembleType::NVE && ensemble != EnsembleType::HEAT_LAN &&
      ensemble != EnsembleType::HEAT_NHC && ensemble != EnsembleType::HEAT_BDP) {
    PRINT_INPUT_ERROR(
      "compute_interface_shc supports only NVE, heat_lan, heat_nhc, and heat_bdp ensembles.\n");
  }
  if (integrate.get_move_group() >= 0) {
    PRINT_INPUT_ERROR("compute_interface_shc rejects move groups because they add external work.\n");
  }
  if (force.get_number_of_potentials() != 1) {
    PRINT_INPUT_ERROR("compute_interface_shc requires exactly one potential.\n");
  }
  nep_ = dynamic_cast<NEP*>(&force.get_potential(0));
  if (!nep_ || !nep_->supports_interface_plane_flux()) {
    PRINT_INPUT_ERROR("compute_interface_shc supports only one regular NEP potential without DFTD3.\n");
  }
  if (!nep_->uses_large_box_for_interface_flux(box)) {
    PRINT_INPUT_ERROR("compute_interface_shc rejects the NEP small-box expansion path.\n");
  }
  if (nep_->nep_model_type != 0) {
    PRINT_INPUT_ERROR("compute_interface_shc rejects temperature-dependent NEP models.\n");
  }
  const double cutoff = nep_->get_interface_plane_flux_cutoff();
  const std::vector<int>& pbc = interface_geometry_->pbc();
  const double lengths[3] = {box.cpu_h[0], box.cpu_h[4], box.cpu_h[8]};
  for (int d = 0; d < 3; ++d) {
    if (!nep_plane_flux_bounds_valid(cutoff, max_displacement_A_, lengths[d], pbc[d] != 0)) {
      PRINT_INPUT_ERROR("compute_interface_shc requires rc+2*max_displacement < L/2 in every periodic direction.\n");
    }
  }

  nominal_step_size_ = time_step;
  double nominal_sample_dt_ps = 0.0;
  if (!interface_sample_interval_ps(
        time_step, TIME_UNIT_CONVERSION, sample_interval_, nominal_sample_dt_ps) ||
      max_omega_rad_ps_ > PI / nominal_sample_dt_ps) {
    PRINT_INPUT_ERROR("compute_interface_shc max omega exceeds the nominal Nyquist limit.\n");
  }
  if (!interface_shc_frequency_grid_resolved(
        nominal_sample_dt_ps, max_omega_rad_ps_, num_omega_, correlation_samples_)) {
    PRINT_INPUT_ERROR("compute_interface_shc rejects an underresolved frequency grid.\n");
  }
  if (block_samples_ > number_of_steps / sample_interval_) {
    PRINT_INPUT_ERROR("compute_interface_shc block_samples exceeds available samples in this run.\n");
  }

  const int N = atom.number_of_atoms;
  const std::vector<double>& interfaces = interface_geometry_->interfaces_A();
  if (interfaces.size() > static_cast<size_t>(std::numeric_limits<int>::max() - 127)) {
    PRINT_INPUT_ERROR("compute_interface_shc interface plane count exceeds supported indexing.\n");
  }
  const int planes = static_cast<int>(interfaces.size());
  const size_t map_count = checked_product(static_cast<size_t>(planes), N);
  if (map_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("compute_interface_shc atom-to-slot map exceeds supported int indexing.\n");
  }
  const std::vector<double>& reference_interleaved = interface_geometry_->reference_positions();
  std::vector<double> reference_soa(checked_product(static_cast<size_t>(N), 3));
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < 3; ++d) reference_soa[static_cast<size_t>(d) * N + n] = reference_interleaved[3 * n + d];
  }
  plane_offsets_.reserve(static_cast<size_t>(planes) + 1);
  plane_offsets_.push_back(0);
  atom_slots_.assign(map_count, -1);
  const double selection_cutoff = cutoff + 2.0 * max_displacement_A_;
  for (int p = 0; p < planes; ++p) {
    for (int n = 0; n < N; ++n) {
      const double x = reference_soa[static_cast<size_t>(interface_geometry_->axis()) * N + n];
      if (nep_plane_atom_selected(
            x, interfaces[p], lengths[interface_geometry_->axis()],
            pbc[interface_geometry_->axis()] != 0, selection_cutoff)) {
        atom_slots_[static_cast<size_t>(p) * N + n] = static_cast<int>(selected_atoms_.size());
        selected_atoms_.push_back(n);
        slot_planes_.push_back(p);
      }
    }
    if (selected_atoms_.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
      PRINT_INPUT_ERROR("compute_interface_shc selected atom count exceeds supported int indexing.\n");
    }
    plane_offsets_.push_back(static_cast<int>(selected_atoms_.size()));
  }

  const size_t selected_count = selected_atoms_.size();
  const size_t vector_values = checked_product(selected_count, 3);
  const size_t ring_values = checked_product(vector_values, correlation_samples_);
  const size_t correlation_values = checked_product(static_cast<size_t>(planes), correlation_samples_);
  const size_t spectrum_values = checked_product(static_cast<size_t>(planes), num_omega_);
  sample_capacity_ = static_cast<size_t>(number_of_steps / sample_interval_);
  complete_block_capacity_ = static_cast<size_t>(interface_shc_complete_block_count(
    static_cast<std::uint64_t>(sample_capacity_), static_cast<std::uint64_t>(block_samples_)));
  const size_t block_flux_values = checked_product(static_cast<size_t>(block_samples_), planes);
  const size_t cached_flux_values = checked_product(sample_capacity_, planes);
  const size_t cached_correlation_values = checked_product(
    checked_product(complete_block_capacity_, correlation_values), 2);
  const size_t cached_sample_metadata_bytes = checked_product(
    sample_capacity_, sizeof(int) + 3 * sizeof(double));
  const size_t cached_block_time_bytes = checked_product(
    complete_block_capacity_, 2 * sizeof(double));
  const size_t transfer_stage_bytes = checked_sum(
    checked_product(block_flux_values, sizeof(double)),
    checked_product(checked_product(correlation_values, 2), sizeof(double)));
  const size_t cached_cpu_bytes = checked_sum(
    checked_sum(checked_product(cached_flux_values, sizeof(double)),
      checked_product(cached_correlation_values, sizeof(double))),
    checked_sum(checked_sum(cached_sample_metadata_bytes, cached_block_time_bytes),
      transfer_stage_bytes));
  const size_t fixed_stat_values = checked_sum(
    checked_product(spectrum_values, 2), checked_product(static_cast<size_t>(planes), 6));
  const size_t transform_workspace_values = checked_sum(
    checked_product(static_cast<size_t>(correlation_samples_), 2),
    static_cast<size_t>(num_omega_));
  const size_t all_cpu_buffer_bytes = checked_sum(
    checked_sum(cached_cpu_bytes, checked_product(fixed_stat_values, sizeof(double))),
    checked_product(transform_workspace_values, sizeof(double)));
  const size_t gpu_fill_limit = checked_product(
    static_cast<size_t>(std::numeric_limits<int>::max()), 128);
  size_t gpu_buffer_bytes = checked_product(map_count, sizeof(int));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(selected_count, 2 * sizeof(int)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(vector_values, sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(checked_product(ring_values, 2), sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(checked_product(correlation_values, 2), sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(reference_soa.size(), sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(block_flux_values, sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(interfaces.size(), sizeof(double)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes,
    checked_product(plane_offsets_.size(), sizeof(int)));
  gpu_buffer_bytes = checked_sum(gpu_buffer_bytes, 2 * sizeof(int));
  if (spectrum_values > static_cast<size_t>(std::numeric_limits<int>::max())) {
    PRINT_INPUT_ERROR("compute_interface_shc spectrum exceeds supported int indexing.\n");
  }
  if (ring_values > std::numeric_limits<size_t>::max() / sizeof(double) ||
      correlation_values > std::numeric_limits<size_t>::max() / sizeof(double) ||
      spectrum_values > std::numeric_limits<size_t>::max() / sizeof(double) ||
      vector_values > gpu_fill_limit || ring_values > gpu_fill_limit ||
      correlation_values > gpu_fill_limit || block_flux_values > gpu_fill_limit) {
    PRINT_INPUT_ERROR("compute_interface_shc buffer size overflows or exceeds GPU fill-grid indexing.\n");
  }
  if (sample_capacity_ > cached_steps_.max_size() ||
      sample_capacity_ > cached_times_ps_.max_size() ||
      sample_capacity_ > cached_bath_source_eV_.max_size() ||
      sample_capacity_ > cached_bath_sink_eV_.max_size() ||
      cached_flux_values > cached_plane_flux_.max_size() ||
      block_flux_values > cpu_plane_flux_block_.max_size() ||
      correlation_values > cpu_correlation_plus_.max_size() ||
      correlation_values > cpu_correlation_minus_.max_size() ||
      cached_correlation_values > cached_correlations_.max_size() ||
      complete_block_capacity_ > cached_block_start_ps_.max_size() ||
      complete_block_capacity_ > cached_block_end_ps_.max_size() ||
      spectrum_values > spectrum_mean_.max_size() ||
      spectrum_values > spectrum_m2_.max_size() ||
      static_cast<size_t>(planes) > c0_mean_.max_size() ||
      static_cast<size_t>(planes) > c0_m2_.max_size() ||
      static_cast<size_t>(planes) > integral_mean_.max_size() ||
      static_cast<size_t>(planes) > integral_m2_.max_size() ||
      static_cast<size_t>(planes) > grid_integral_mean_.max_size() ||
      static_cast<size_t>(planes) > grid_integral_m2_.max_size()) {
    PRINT_INPUT_ERROR("compute_interface_shc cache exceeds host vector capacity.\n");
  }
  allocated_bytes_ = gpu_buffer_bytes;
  printf("Interface SHC GPU buffers require %zu bytes; buffered CPU data require %zu bytes.\n",
    allocated_bytes_, all_cpu_buffer_bytes);

  gpu_reference_positions_.resize(reference_soa.size());
  gpu_reference_positions_.copy_from_host(reference_soa.data(), reference_soa.size());
  gpu_plane_positions_.resize(interfaces.size());
  gpu_plane_positions_.copy_from_host(interfaces.data(), interfaces.size());
  gpu_plane_offsets_.resize(plane_offsets_.size());
  gpu_plane_offsets_.copy_from_host(plane_offsets_.data(), plane_offsets_.size());
  gpu_atom_slots_.resize(atom_slots_.size());
  gpu_atom_slots_.copy_from_host(atom_slots_.data(), atom_slots_.size());
  if (selected_count) {
    gpu_selected_atoms_.resize(selected_count);
    gpu_selected_atoms_.copy_from_host(selected_atoms_.data(), selected_count);
    gpu_slot_planes_.resize(selected_count);
    gpu_slot_planes_.copy_from_host(slot_planes_.data(), selected_count);
  }
  if (vector_values) gpu_coefficients_.resize(vector_values, 0.0);
  gpu_error_info_.resize(2, 0);
  if (ring_values) {
    gpu_coefficient_history_.resize(ring_values, 0.0);
    gpu_velocity_history_.resize(ring_values, 0.0);
  }
  gpu_correlation_plus_.resize(correlation_values, 0.0);
  gpu_correlation_minus_.resize(correlation_values, 0.0);
  gpu_plane_flux_.resize(block_flux_values, 0.0);
  cpu_correlation_plus_.resize(correlation_values);
  cpu_correlation_minus_.resize(correlation_values);
  cpu_plane_flux_block_.resize(block_flux_values);
  cached_steps_.resize(sample_capacity_);
  cached_times_ps_.resize(sample_capacity_);
  cached_bath_source_eV_.resize(sample_capacity_);
  cached_bath_sink_eV_.resize(sample_capacity_);
  cached_plane_flux_.resize(cached_flux_values);
  cached_correlations_.resize(cached_correlation_values);
  cached_block_start_ps_.resize(complete_block_capacity_);
  cached_block_end_ps_.resize(complete_block_capacity_);
  spectrum_mean_.resize(spectrum_values, 0.0);
  spectrum_m2_.resize(spectrum_values, 0.0);
  c0_mean_.resize(planes, 0.0);
  c0_m2_.resize(planes, 0.0);
  integral_mean_.resize(planes, 0.0);
  integral_m2_.resize(planes, 0.0);
  grid_integral_mean_.resize(planes, 0.0);
  grid_integral_m2_.resize(planes, 0.0);

  const std::string plane_flux_name = prefix_ + "_plane_flux.out";
  const std::string correlation_name = prefix_ + "_shc_correlation.out";
  const std::string shc_name = prefix_ + "_shc.out";
  if (output_exists(plane_flux_name) || output_exists(correlation_name) || output_exists(shc_name)) {
    PRINT_INPUT_ERROR("compute_interface_shc output already exists; choose a new prefix.\n");
  }
  fid_plane_flux_ = my_fopen(plane_flux_name.c_str(), "w");
  fid_correlation_ = my_fopen(correlation_name.c_str(), "w");
  fid_shc_ = my_fopen(shc_name.c_str(), "w");
  std::fprintf(fid_plane_flux_, "# compute_interface_shc_plane_flux version 1; Q=sum_j a_j dot v_j; sign follows the configured +axis crossing convention\n");
  std::fprintf(fid_plane_flux_, "# geometry_source %s_geometry.json; area_A2 %.12g; bath counters are cumulative eV with system_to_bath sign\n",
    interface_geometry_->output_prefix().c_str(), interface_geometry_->cross_section_A2());
  std::fprintf(fid_plane_flux_, "# columns step time_ps plane_id Q_eV_ps J_W_m2 bath_source_eV bath_sink_eV\n");
  std::fprintf(fid_correlation_, "# compute_interface_shc_correlation version 1; C(+lag)=a(t) dot v(t+lag); C(-lag)=a(t+lag) dot v(t)\n");
  std::fprintf(fid_correlation_, "# columns block plane_id lag time_ps correlation_W_m2 origins\n");
  std::fprintf(fid_shc_, "# compute_interface_shc_spectrum version 3; j(omega)=dt/pi*[C0+sum_lag>0(C+ + C-)cos(omega*lag*dt)]\n");
  std::fprintf(fid_shc_, "# max_omega_rad_ps %.12g nominal_nyquist_rad_ps %.12g band_limited %d frequency_grid_endpoints_included 1\n",
    max_omega_rad_ps_, PI / nominal_sample_dt_ps, max_omega_rad_ps_ < PI / nominal_sample_dt_ps);
  std::fprintf(fid_shc_, "# window rectangular; analytic full-Nyquist integral equals C0; both analytic and exported-grid trapezoid integrals are reported\n");
  std::fprintf(fid_shc_, "# columns block block_start_ps block_end_ps block_samples accumulation_start_ps accumulation_end_ps accumulation_samples block_count plane_id omega_rad_ps C0_mean_W_m2 C0_SE_W_m2 analytic_integral_0_max_W_m2 analytic_integral_SE_W_m2 exported_grid_trapezoid_0_max_W_m2 exported_grid_trapezoid_SE_W_m2 jomega_mean_W_m2_per_rad_ps jomega_SE_W_m2_per_rad_ps\n");

  view_.enabled = 0;
  view_.atom_count = N;
  view_.plane_count = planes;
  view_.axis = interface_geometry_->axis();
  for (int d = 0; d < 3; ++d) {
    view_.pbc[d] = pbc[d];
    view_.box_lengths[d] = lengths[d];
  }
  view_.max_reference_displacement = max_displacement_A_;
  view_.reference_positions = gpu_reference_positions_.data();
  view_.plane_positions = gpu_plane_positions_.data();
  view_.plane_offsets = gpu_plane_offsets_.data();
  view_.atom_slots = gpu_atom_slots_.data();
  view_.coefficients = vector_values ? gpu_coefficients_.data() : nullptr;
  view_.error_flag = gpu_error_info_.data();
  view_.first_error_step = gpu_error_info_.data() + 1;
  nep_->set_plane_flux_view(view_);
  samples_in_block_ = 0;
  block_index_ = 0;
  spectrum_blocks_ = 0;
  cached_sample_count_ = 0;
  accumulation_start_ps_ = 0.0;
  accumulation_end_ps_ = 0.0;
  accumulation_samples_ = 0;
  have_accumulation_range_ = false;
  last_step_time_ps_ = 0.0;
  sample_dt_ps_ = 0.0;
  have_step_time_ = false;
  active_ = true;
}

void ComputeInterfaceSHC::pre_force(
  const int step,
  const double time_step,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box&,
  Force&)
{
  if (!active_) return;
  if (!interface_time_step_matches(time_step, nominal_step_size_)) {
    PRINT_INPUT_ERROR("compute_interface_shc rejects adaptive or changing time steps.\n");
  }
  if (step == 0) {
    if (!interface_sample_interval_ps(
          time_step, TIME_UNIT_CONVERSION, sample_interval_, sample_dt_ps_) ||
        max_omega_rad_ps_ > PI / sample_dt_ps_ ||
        !interface_shc_frequency_grid_resolved(
          sample_dt_ps_, max_omega_rad_ps_, num_omega_, correlation_samples_)) {
      PRINT_INPUT_ERROR("compute_interface_shc observed sampling interval or frequency grid is invalid.\n");
    }
  } else if (sample_dt_ps_ <= 0.0) {
    PRINT_INPUT_ERROR("compute_interface_shc did not observe the first integration step.\n");
  }
  view_.enabled = (step + 1) % sample_interval_ == 0;
  if (view_.enabled) {
    view_.sample_step = step + 1;
    if (samples_in_block_ == 0) {
      gpu_reset_interface_shc_error<<<1, 1>>>(gpu_error_info_.data());
      GPU_CHECK_KERNEL
    }
    if (!selected_atoms_.empty()) gpu_coefficients_.fill(0.0);
    armed_ = true;
  } else {
    armed_ = false;
  }
  nep_->set_plane_flux_view(view_);
}

void ComputeInterfaceSHC::post_force(
  const int,
  const double,
  Integrate&,
  std::vector<Group>&,
  Atom&,
  Box&,
  Force&)
{
  if (!active_ || !armed_) return;
  view_.enabled = 0;
  nep_->set_plane_flux_view(view_);
  armed_ = false;
}

void ComputeInterfaceSHC::end_of_step(
  const int,
  int step,
  const int,
  const int,
  const double global_time,
  const double,
  Integrate& integrate,
  Box&,
  std::vector<Group>&,
  GPU_Vector<double>&,
  Atom& atom,
  Force&)
{
  if (!active_) return;
  const double now_ps = global_time * TIME_UNIT_CONVERSION / 1000.0;
  if (!interface_time_ps_is_increasing(now_ps, last_step_time_ps_, have_step_time_)) {
    PRINT_INPUT_ERROR("compute_interface_shc requires finite increasing time.\n");
  }
  last_step_time_ps_ = now_ps;
  have_step_time_ = true;
  if ((step + 1) % sample_interval_ != 0) return;
  if (sample_dt_ps_ <= 0.0) PRINT_INPUT_ERROR("compute_interface_shc has no observed sample interval.\n");
  capture_sample(step + 1, now_ps, integrate, atom);
}

void ComputeInterfaceSHC::capture_sample(
  const int step,
  const double time_ps,
  Integrate& integrate,
  Atom& atom)
{
  const int planes = view_.plane_count;
  const int selected_count = static_cast<int>(selected_atoms_.size());
  const int ring_slot = samples_in_block_ % correlation_samples_;
  const int valid_lags = std::min(samples_in_block_, correlation_samples_ - 1);
  std::uint64_t sample_cache_index = 0;
  if (!interface_shc_cache_sample_index(
        static_cast<std::uint64_t>(block_index_),
        static_cast<std::uint64_t>(samples_in_block_),
        static_cast<std::uint64_t>(block_samples_),
        static_cast<std::uint64_t>(sample_capacity_), sample_cache_index) ||
      sample_cache_index != cached_sample_count_) {
    PRINT_INPUT_ERROR("compute_interface_shc sample cache cursor is inconsistent.\n");
  }
  const size_t block_flux_offset = static_cast<size_t>(samples_in_block_) * planes;
  constexpr int clear_threads = 128;
  gpu_clear_interface_shc_flux_row<<<(planes + clear_threads - 1) / clear_threads, clear_threads>>>(
    planes, gpu_plane_flux_.data() + block_flux_offset);
  GPU_CHECK_KERNEL
  if (selected_count) {
    const int threads = 128;
    const int blocks = (selected_count - 1) / threads + 1;
    gpu_store_interface_shc_sample<<<blocks, threads>>>(
      selected_count,
      planes,
      atom.number_of_atoms,
      ring_slot,
      gpu_selected_atoms_.data(),
      gpu_slot_planes_.data(),
      gpu_coefficients_.data(),
      atom.velocity_per_atom.data(),
      1000.0 / TIME_UNIT_CONVERSION,
      gpu_error_info_.data(),
      gpu_error_info_.data() + 1,
      step,
      gpu_coefficient_history_.data(),
      gpu_velocity_history_.data(),
      gpu_plane_flux_.data() + block_flux_offset);
    GPU_CHECK_KERNEL
    gpu_accumulate_interface_shc_correlation<<<blocks, threads>>>(
      selected_count,
      planes,
      correlation_samples_,
      ring_slot,
      valid_lags,
      gpu_slot_planes_.data(),
      gpu_coefficient_history_.data(),
      gpu_velocity_history_.data(),
      gpu_error_info_.data(),
      gpu_error_info_.data() + 1,
      step,
      gpu_correlation_plus_.data(),
      gpu_correlation_minus_.data());
    GPU_CHECK_KERNEL
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();
  double bath_source = nan;
  double bath_sink = nan;
  const EnsembleType ensemble = integrate.get_type();
  if (ensemble == EnsembleType::HEAT_LAN || ensemble == EnsembleType::HEAT_NHC ||
      ensemble == EnsembleType::HEAT_BDP) {
    const double* energy = integrate.get_energy_transferred();
    bath_source = energy[0];
    bath_sink = energy[1];
  }
  cached_steps_[sample_cache_index] = step;
  cached_times_ps_[sample_cache_index] = time_ps;
  cached_bath_source_eV_[sample_cache_index] = bath_source;
  cached_bath_sink_eV_[sample_cache_index] = bath_sink;
  ++cached_sample_count_;
  if (samples_in_block_ == 0) block_start_ps_ = time_ps;
  block_end_ps_ = time_ps;
  ++samples_in_block_;
  if (samples_in_block_ == block_samples_) {
    flush_sample_block(true);
    ++block_index_;
    samples_in_block_ = 0;
    gpu_correlation_plus_.fill(0.0);
    gpu_correlation_minus_.fill(0.0);
  }
}

void ComputeInterfaceSHC::check_block_error()
{
  int error_info[2] = {0, -1};
  gpu_error_info_.copy_to_host(error_info, 2);
  if (!error_info[0]) return;
  std::fprintf(stderr, "compute_interface_shc: first sampled error occurred at step %d.\n", error_info[1]);
  if (error_info[0] & NEP_PLANE_FLUX_ERROR_DISPLACEMENT)
    std::fprintf(stderr, "compute_interface_shc: an atom exceeded max_reference_displacement_A.\n");
  if (error_info[0] & NEP_PLANE_FLUX_ERROR_MISSING_SLOT)
    std::fprintf(stderr, "compute_interface_shc: a crossing force bond was absent from its compact plane map.\n");
  if (error_info[0] & NEP_PLANE_FLUX_ERROR_NONFINITE)
    std::fprintf(stderr, "compute_interface_shc: a sampled coordinate, derivative, coefficient, or velocity was non-finite.\n");
  PRINT_INPUT_ERROR("compute_interface_shc failed closed at the end of its buffered sample block.\n");
}

void ComputeInterfaceSHC::flush_sample_block(const bool complete_block)
{
  if (samples_in_block_ <= 0) return;
  check_block_error();

  const int planes = view_.plane_count;
  const size_t block_flux_values = static_cast<size_t>(block_samples_) * planes;
  gpu_plane_flux_.copy_to_host(cpu_plane_flux_block_.data(), block_flux_values);
  const size_t sample_offset = cached_sample_count_ - samples_in_block_;
  for (int sample = 0; sample < samples_in_block_; ++sample) {
    const size_t output_sample = sample_offset + sample;
    const size_t block_offset = static_cast<size_t>(sample) * planes;
    const size_t cached_offset = output_sample * planes;
    for (int p = 0; p < planes; ++p) {
      const double flux = cpu_plane_flux_block_[block_offset + p];
      if (!std::isfinite(flux)) {
        std::fprintf(stderr, "compute_interface_shc: non-finite plane flux at step %d, plane %d.\n",
          cached_steps_[output_sample], p);
        PRINT_INPUT_ERROR("compute_interface_shc rejected a non-finite buffered flux.\n");
      }
      cached_plane_flux_[cached_offset + p] = flux;
    }
  }

  if (!complete_block) return;
  if (samples_in_block_ != block_samples_ || block_index_ >= complete_block_capacity_) {
    PRINT_INPUT_ERROR("compute_interface_shc complete-block cache cursor is inconsistent.\n");
  }
  const size_t correlation_values = static_cast<size_t>(planes) * correlation_samples_;
  const size_t cache_offset = static_cast<size_t>(block_index_) * 2 * correlation_values;
  gpu_correlation_plus_.copy_to_host(cpu_correlation_plus_.data(), correlation_values);
  gpu_correlation_minus_.copy_to_host(cpu_correlation_minus_.data(), correlation_values);
  std::copy(cpu_correlation_plus_.begin(), cpu_correlation_plus_.end(),
    cached_correlations_.begin() + cache_offset);
  std::copy(cpu_correlation_minus_.begin(), cpu_correlation_minus_.end(),
    cached_correlations_.begin() + cache_offset + correlation_values);
  cached_block_start_ps_[block_index_] = block_start_ps_;
  cached_block_end_ps_[block_index_] = block_end_ps_;
}

void ComputeInterfaceSHC::output_cached_samples()
{
  const int planes = view_.plane_count;
  const double flux_conversion = EV_PER_PS_A2_TO_W_PER_M2 /
                                 interface_geometry_->cross_section_A2();
  for (size_t sample = 0; sample < cached_sample_count_; ++sample) {
    for (int p = 0; p < planes; ++p) {
      const double raw_flux = cached_plane_flux_[sample * planes + p];
      const double flux = raw_flux * flux_conversion;
      if (!std::isfinite(flux)) {
        std::fprintf(stderr, "compute_interface_shc: non-finite converted flux at step %d, plane %d.\n",
          cached_steps_[sample], p);
        PRINT_INPUT_ERROR("compute_interface_shc rejected a non-finite buffered flux.\n");
      }
      std::fprintf(
        fid_plane_flux_, "%d %.12g %d %.12g %.12g %.12g %.12g\n",
        cached_steps_[sample], cached_times_ps_[sample], p, raw_flux, flux,
        cached_bath_source_eV_[sample], cached_bath_sink_eV_[sample]);
    }
  }
}

void ComputeInterfaceSHC::output_complete_block(const size_t block_index)
{
  const int planes = view_.plane_count;
  const size_t values = static_cast<size_t>(planes) * correlation_samples_;
  const size_t cache_offset = block_index * 2 * values;
  const double* block_correlation_plus = cached_correlations_.data() + cache_offset;
  const double* block_correlation_minus = block_correlation_plus + values;
  const double area = interface_geometry_->cross_section_A2();
  const double flux_conversion = EV_PER_PS_A2_TO_W_PER_M2 / area;
  const int block_number = spectrum_blocks_ + 1;
  std::vector<double> correlation_plus(correlation_samples_);
  std::vector<double> correlation_minus(correlation_samples_, 0.0);
  std::vector<double> block_spectrum(static_cast<size_t>(num_omega_));
  const double block_start_ps = cached_block_start_ps_[block_index];
  const double block_end_ps = cached_block_end_ps_[block_index];
  for (int p = 0; p < planes; ++p) {
    for (int lag = 0; lag < correlation_samples_; ++lag) {
      const int origins = block_samples_ - lag;
      const size_t index = static_cast<size_t>(p) * correlation_samples_ + lag;
      correlation_plus[lag] = block_correlation_plus[index] / origins * flux_conversion;
      correlation_minus[lag] = lag == 0 ? 0.0 : block_correlation_minus[index] / origins * flux_conversion;
      if (!std::isfinite(correlation_plus[lag]) || !std::isfinite(correlation_minus[lag])) {
        PRINT_INPUT_ERROR("compute_interface_shc produced a non-finite block correlation.\n");
      }
      std::fprintf(
        fid_correlation_, "%d %d %d %.12g %.12g %d\n",
        static_cast<int>(block_index), p, lag, lag * sample_dt_ps_, correlation_plus[lag], origins);
      if (lag > 0) {
        std::fprintf(
          fid_correlation_, "%d %d %d %.12g %.12g %d\n",
          static_cast<int>(block_index), p, -lag, -lag * sample_dt_ps_, correlation_minus[lag], origins);
      }
    }
    const double plane_c0 = correlation_plus[0];
    const double plane_integral = interface_shc_frequency_integral(
      sample_dt_ps_, max_omega_rad_ps_, correlation_plus.data(), correlation_minus.data(), correlation_samples_);
    if (!std::isfinite(plane_integral)) {
      PRINT_INPUT_ERROR("compute_interface_shc produced a non-finite analytic spectrum integral.\n");
    }
    for (int omega_id = 0; omega_id < num_omega_; ++omega_id) {
      const double omega = max_omega_rad_ps_ * omega_id / (num_omega_ - 1);
      block_spectrum[omega_id] = interface_shc_spectrum_value(
        sample_dt_ps_, omega, correlation_plus.data(), correlation_minus.data(), correlation_samples_);
      if (!std::isfinite(block_spectrum[omega_id])) {
        PRINT_INPUT_ERROR("compute_interface_shc produced a non-finite spectrum row.\n");
      }
    }
    const double plane_grid_integral = interface_shc_trapezoid_integral(
      block_spectrum.data(), num_omega_, max_omega_rad_ps_);
    if (!std::isfinite(plane_grid_integral)) {
      PRINT_INPUT_ERROR("compute_interface_shc produced a non-finite exported-grid integral.\n");
    }
    const size_t plane_index = static_cast<size_t>(p);
    interface_shc_update_mean_m2(
      plane_c0, block_number, c0_mean_[plane_index], c0_m2_[plane_index]);
    interface_shc_update_mean_m2(
      plane_integral, block_number, integral_mean_[plane_index], integral_m2_[plane_index]);
    interface_shc_update_mean_m2(
      plane_grid_integral, block_number,
      grid_integral_mean_[plane_index], grid_integral_m2_[plane_index]);
    for (int omega_id = 0; omega_id < num_omega_; ++omega_id) {
      const double value = block_spectrum[omega_id];
      const size_t spectrum_id = static_cast<size_t>(p) * num_omega_ + omega_id;
      interface_shc_update_mean_m2(
        value, block_number, spectrum_mean_[spectrum_id], spectrum_m2_[spectrum_id]);
    }
  }
  if (!interface_shc_accumulate_range(
        block_start_ps, block_end_ps, static_cast<std::uint64_t>(block_samples_),
        have_accumulation_range_, accumulation_start_ps_, accumulation_end_ps_, accumulation_samples_)) {
    PRINT_INPUT_ERROR("compute_interface_shc failed to extend the cumulative block range.\n");
  }
  spectrum_blocks_ = block_number;
  for (int p = 0; p < planes; ++p) {
    for (int omega_id = 0; omega_id < num_omega_; ++omega_id) {
      const size_t spectrum_id = static_cast<size_t>(p) * num_omega_ + omega_id;
      const double omega = max_omega_rad_ps_ * omega_id / (num_omega_ - 1);
      std::fprintf(
        fid_shc_, "%d %.12g %.12g %d %.12g %.12g %llu %d %d %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g %.12g\n",
        static_cast<int>(block_index), block_start_ps, block_end_ps, block_samples_,
        accumulation_start_ps_, accumulation_end_ps_,
        static_cast<unsigned long long>(accumulation_samples_), spectrum_blocks_, p, omega,
        c0_mean_[p], interface_shc_standard_error(spectrum_blocks_, c0_m2_[p]),
        integral_mean_[p], interface_shc_standard_error(spectrum_blocks_, integral_m2_[p]),
        grid_integral_mean_[p], interface_shc_standard_error(spectrum_blocks_, grid_integral_m2_[p]),
        spectrum_mean_[spectrum_id], interface_shc_standard_error(spectrum_blocks_, spectrum_m2_[spectrum_id]));
    }
  }
}

void ComputeInterfaceSHC::post_run(
  Atom&,
  Box&,
  Integrate&,
  const int,
  const double,
  const double)
{
  if (samples_in_block_ > 0) flush_sample_block(false);
  output_cached_samples();
  for (size_t block = 0; block < static_cast<size_t>(block_index_); ++block) {
    output_complete_block(block);
  }
  std::fflush(fid_plane_flux_);
  std::fflush(fid_correlation_);
  std::fflush(fid_shc_);
  if (samples_in_block_ > 0) {
    printf("compute_interface_shc dropped %d incomplete tail samples after %d complete blocks.\n",
      samples_in_block_, block_index_);
  }
  if (nep_) nep_->clear_plane_flux_view();
  if (fid_plane_flux_) std::fclose(fid_plane_flux_);
  if (fid_correlation_) std::fclose(fid_correlation_);
  if (fid_shc_) std::fclose(fid_shc_);
  fid_plane_flux_ = nullptr;
  fid_correlation_ = nullptr;
  fid_shc_ = nullptr;
  active_ = false;
}
