#include "rpmd_ja_fit.cuh"
#include "rpmd_ja_process_lock.h"

#include "rpmd_ja_fit_sampling.cuh"
#include "rpmd_ja_native_fit.cuh"
#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_sample_provenance.h"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <stdexcept>

namespace
{

double parse_positive(const std::string& value, const char* name)
{
  char* end = nullptr;
  const double result = std::strtod(value.c_str(), &end);
  if (end == value.c_str() || *end != '\0' || !std::isfinite(result) || !(result > 0.0)) {
    throw std::invalid_argument(std::string("rpmd_ja fit ") + name + " must be positive and finite");
  }
  return result;
}

bool file_exists(const std::string& path)
{
  rpmd_ja_file_io::ifstream in(path, std::ios::binary);
  return in.good();
}

bool apply_fit_mic(const Box& box, double& x, double& y, double& z)
{
  return rpmd_ja_fit_mic(box.cpu_h + 9, box.cpu_h, x, y, z);
}

void write_or_throw(rpmd_ja_file_io::ofstream& out, const void* data, const std::size_t bytes)
{
  if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("RPMD-JA sample spool write exceeds stream limits");
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) throw std::runtime_error("failed writing RPMD-JA sample spool");
}

void write_exclusive_file(const std::string& path, const std::string& contents)
{
  std::FILE* output = rpmd_ja_file_io::open_file(path, "wx");
  if (!output) throw std::runtime_error(std::string("RPMD-JA provenance output already exists or cannot be created: ") + path);
  const std::size_t written = std::fwrite(contents.data(), 1, contents.size(), output);
  const bool flushed = std::fflush(output) == 0;
  const bool closed = std::fclose(output) == 0;
  if (written != contents.size() || !flushed || !closed) {
    rpmd_ja_file_io::remove_file(path);
    throw std::runtime_error(std::string("failed writing RPMD-JA provenance output: ") + path);
  }
}

std::string build_provenance_json(
  const std::string& sample_sha256, const std::uint64_t sample_bytes, const std::uint64_t sample_frames,
  const std::string& source_job_id, const std::string& source_job_id_status, const std::string& source_run_id,
  const int first_sample_step, const int last_sample_step, const double timestep_internal, const double timestep_fs,
  const double temperature, const int beads, const bool internal_mass_com, const double* sampling_com_shift,
  const std::string& thermostat_tokens_json,const std::string& active_configuration_scope,const unsigned active_run_ordinal,
  const std::string& model_path,
  const std::string& model_sha256_begin, const std::string& model_sha256_end,
  const std::string& run_input_sha256_begin, const std::string& run_input_sha256_end,
  const std::string& physical_force_configuration_sha256, const std::string& integrator_configuration_sha256,
  const std::string& atom_signature_sha256,
  const bool configuration_stable)
{
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << std::setprecision(17);
  output << "{\"schema\":\"RPJAPROV2\",\"sample_sha256\":\"" << sample_sha256
    << "\",\"sample_bytes\":" << sample_bytes << ",\"sample_frames\":" << sample_frames
    << ",\"source_job_id\":\"" << rpmd_ja_sample_provenance::json_escape(source_job_id)
    << "\",\"source_job_id_status\":\"" << rpmd_ja_sample_provenance::json_escape(source_job_id_status)
    << "\",\"source_run_id\":\"" << rpmd_ja_sample_provenance::json_escape(source_run_id)
    << "\",\"step_start\":" << first_sample_step << ",\"step_end\":" << last_sample_step
    << ",\"provenance_status\":\"PRODUCER_OBSERVED\",\"independence_attestation\":\"NOT_ASSERTED\""
    << ",\"metadata\":{\"integrator_type\":\"PIMD\",\"timestep_internal\":" << timestep_internal
    << ",\"timestep_internal_unit\":\"GPUMD_INTERNAL_TIME\",\"timestep_fs\":" << timestep_fs
    << ",\"timestep_fs_unit\":\"fs\",\"temperature_K\":" << temperature
    << ",\"number_of_beads\":" << beads << ",\"thermostat_configuration_tokens\":" << thermostat_tokens_json
    << ",\"active_configuration_scope\":\"" << rpmd_ja_sample_provenance::json_escape(active_configuration_scope)
    << "\",\"active_run_ordinal\":" << active_run_ordinal
    << ",\"model_path\":\"" << rpmd_ja_sample_provenance::json_escape(model_path)
    << "\",\"model_sha256\":\"" << model_sha256_begin
    << "\",\"model_sha256_begin\":\"" << model_sha256_begin
    << "\",\"model_sha256_end\":\"" << model_sha256_end
    << "\",\"run_input_sha256\":\"" << run_input_sha256_begin
    << "\",\"run_input_sha256_begin\":\"" << run_input_sha256_begin
    << "\",\"run_input_sha256_end\":\"" << run_input_sha256_end
    << "\",\"physical_force_configuration_sha256\":\"" << physical_force_configuration_sha256
    << "\",\"integrator_configuration_sha256\":\"" << integrator_configuration_sha256
    << "\",\"atom_signature_sha256\":\"" << atom_signature_sha256
    << "\",\"physical_force_source\":\"MEAN_OF_PHYSICAL_PIMD_BEAD_FORCES\""
    << ",\"internal_mass_com\":" << (internal_mass_com ? "true" : "false")
    << ",\"sampling_com_method\":\"MASS_WEIGHTED_CUMULATIVE_CENTROID_DRIFT_REMOVAL\""
    << ",\"sampling_com_shift_final\":[" << sampling_com_shift[0] << ',' << sampling_com_shift[1] << ','
    << sampling_com_shift[2] << "],\"configuration_stable_during_run\":"
    << (configuration_stable ? "true" : "false") << "}}\n";
  return output.str();
}

} // namespace

RpmdJA_Fit::RpmdJA_Fit(const std::vector<std::string>& tokens)
{
  action_name = "rpmd_ja_fit";
  if (tokens.size() < 2 || tokens[0] != "rpmd_ja") {
    throw std::invalid_argument("rpmd_ja fit requires valid command tokens");
  }
  sample_only_ = tokens[1] == "sample";
  if (sample_only_) {
    if (tokens.size() != 4) {
      throw std::invalid_argument("rpmd_ja sample requires <basename> <sample_interval>");
    }
  } else if ((tokens.size() != 9 && tokens.size() != 11) || tokens[1] != "fit") {
    throw std::invalid_argument(
      "rpmd_ja fit requires <outfile> <sample_interval> <cutoff> <epsilon> "
      "<response_tolerance> <fd_step> <kernel_table> [max_rounds <N>]");
  }
  output_path_ = tokens[2];
  char* end = nullptr;
  const long long interval = std::strtoll(tokens[3].c_str(), &end, 10);
  if (end == tokens[3].c_str() || *end != '\0' || interval <= 0 || interval > INT_MAX) {
    throw std::invalid_argument("rpmd_ja sample_interval must be a positive integer");
  }
  sample_interval_ = static_cast<int>(interval);
  if (!sample_only_ && tokens.size() == 11) {
    if (tokens[9] != "max_rounds") {
      throw std::invalid_argument("rpmd_ja fit optional tail must be max_rounds <N>");
    }
    errno = 0;
    const long long rounds = std::strtoll(tokens[10].c_str(), &end, 10);
    if (errno == ERANGE || end == tokens[10].c_str() || *end != '\0' || rounds <= 0 || rounds > INT_MAX) {
      throw std::invalid_argument("rpmd_ja fit max_rounds must be a positive integer");
    }
    max_stability_rounds_ = static_cast<int>(rounds);
  }
  if (!sample_only_) {
    cutoff_ = parse_positive(tokens[4], "cutoff");
    epsilon_ = parse_positive(tokens[5], "epsilon");
    response_tolerance_ = parse_positive(tokens[6], "response_tolerance");
    fd_step_ = parse_positive(tokens[7], "fd_step");
    kernel_table_ = tokens[8];
  }
  if (output_path_.empty() || (!sample_only_ && kernel_table_.empty())) {
    throw std::invalid_argument("rpmd_ja output and kernel table paths must be nonempty");
  }
  spool_path_ = output_path_ + ".samples.tmp";
  provenance_path_ = spool_path_ + ".provenance.json";
  provenance_temp_path_ = provenance_path_ + ".tmp";
  lock_path_ = output_path_ + ".fit.lock";
}

RpmdJA_Fit::~RpmdJA_Fit() { release_lock_(); }

void RpmdJA_Fit::release_lock_()
{
  if (!lock_file_) return;
  rpmd_ja_process_lock::unlock(lock_file_);
  std::fclose(lock_file_);
  lock_file_ = nullptr;
}

void RpmdJA_Fit::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
  timestep_internal_ = time_step;
  timestep_fs_ = time_step * TIME_UNIT_CONVERSION;
  internal_mass_com_ = integrate.get_pimd_fix_com();
  if (integrate.get_type() != EnsembleType::PIMD || integrate.get_number_of_beads() < 1 ||
      atom.number_of_beads != integrate.get_number_of_beads()) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires canonical PIMD sampling with initialized bead positions; RPMD and TRPMD do not thermostat the centroid and cannot guarantee canonical sampling.");
  }
  if (integrate.get_num_target_pressure_components() != 0 ||
      integrate.get_deform_x() || integrate.get_deform_y() || integrate.get_deform_z() ||
      integrate.get_deform_xy() || integrate.get_deform_xz() || integrate.get_deform_yz()) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires a fixed cell without pressure control or deformation.");
  }
  if (integrate.get_use_scr_barostat() || integrate.get_use_eco_pimd()) {
    PRINT_INPUT_ERROR("rpmd_ja fit does not support SCR or Eco-PIMD.");
  }
  if (integrate.get_fixed_group() != -1 || integrate.get_move_group() != -1) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires all atoms to remain mobile.");
  }
  temperature_ = integrate.get_temperature2();
  if (!(temperature_ > 0.0) || !std::isfinite(temperature_) ||
      std::abs(integrate.get_temperature1() - temperature_) >
        1.0e-12 * std::max(1.0, std::abs(temperature_))) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires a constant positive ring-polymer temperature.");
  }
  if (atom.number_of_atoms < 2 || atom.number_of_atoms > INT_MAX / 3 ||
      atom.mass.size() != static_cast<std::size_t>(atom.number_of_atoms) ||
      atom.cpu_type.size() != static_cast<std::size_t>(atom.number_of_atoms) ||
      atom.cpu_mass.size() != static_cast<std::size_t>(atom.number_of_atoms)) {
    PRINT_INPUT_ERROR("rpmd_ja fit could not read per-atom masses and types.");
  }
  const std::size_t coordinate_count = static_cast<std::size_t>(atom.number_of_atoms) * 3;
  if (atom.position_beads.size() != static_cast<std::size_t>(atom.number_of_beads) ||
      atom.force_beads.size() != static_cast<std::size_t>(atom.number_of_beads)) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires physical position and force arrays for every bead.");
  }
  for (int bead = 0; bead < atom.number_of_beads; ++bead) {
    if (atom.position_beads[bead].size() != coordinate_count ||
        atom.force_beads[bead].size() != coordinate_count) {
      PRINT_INPUT_ERROR("rpmd_ja fit found an invalid per-bead position or force array.");
    }
  }
  const long long frame_count = number_of_steps / sample_interval_;
  const std::set<int> unique_types(atom.cpu_type.begin(), atom.cpu_type.end());
  const long long expected_probes = std::min<long long>(coordinate_count - 3,
    16 + std::min<std::size_t>(unique_types.size(), 4) + 4);
  const long long holdout_count = frame_count - 2 * frame_count / 3;
  if (sample_only_ && frame_count < 3) {
    PRINT_INPUT_ERROR("rpmd_ja sample needs at least three sampled frames.");
  }
  if (!sample_only_ && (frame_count < 3 || holdout_count <= expected_probes)) {
    const std::string message = "rpmd_ja fit needs at least " +
      std::to_string(3 * expected_probes + 1) +
      " sampled frames for its 2/3 training split and " +
      std::to_string(expected_probes + 1) + " held-out probe frames.";
    PRINT_INPUT_ERROR(message.c_str());
  }
  for (int i = 0; i < 9; ++i) {
    cell_[i] = box.cpu_h[i];
    if (!std::isfinite(cell_[i])) PRINT_INPUT_ERROR("rpmd_ja fit requires a finite fixed cell.");
  }
  pbc_[0] = box.pbc_x; pbc_[1] = box.pbc_y; pbc_[2] = box.pbc_z;
  if (pbc_[0] != 1 || pbc_[1] != 1 || pbc_[2] != 1) {
    PRINT_INPUT_ERROR("rpmd_ja fit currently requires periodic boundaries in all three directions.");
  }
  const unsigned current_run_ordinal=active_run_ordinal_++;
  const rpmd_ja_sample_provenance::ConfigurationSnapshot provenance_snapshot=rpmd_ja_sample_provenance::snapshot(
    force,atom,box,number_of_steps,current_run_ordinal,time_step,temperature_,number_of_beads_,internal_mass_com_);
  if(!provenance_snapshot_initialized_){
    run_input_sha256_=provenance_snapshot.run_input_sha256;
    physical_force_configuration_sha256_=provenance_snapshot.physical_force_configuration_sha256;
    integrator_configuration_sha256_=provenance_snapshot.integrator_configuration_sha256;
    active_configuration_scope_=provenance_snapshot.active_configuration_scope;
    model_sha256_=provenance_snapshot.model_sha256;atom_signature_sha256_=provenance_snapshot.atom_signature_sha256;
    thermostat_tokens_json_=provenance_snapshot.thermostat_tokens_json;model_path_=provenance_snapshot.model_path;
    provenance_snapshot_initialized_=true;
  }else if(!provenance_snapshot.active_configuration_identified||
      provenance_snapshot.run_input_sha256!=run_input_sha256_||
      provenance_snapshot.physical_force_configuration_sha256!=physical_force_configuration_sha256_||
      provenance_snapshot.integrator_configuration_sha256!=integrator_configuration_sha256_||
      provenance_snapshot.model_sha256!=model_sha256_||provenance_snapshot.atom_signature_sha256!=atom_signature_sha256_){
    configuration_stable_during_sampling_=false;
  }
  if(!provenance_snapshot.active_configuration_identified)configuration_stable_during_sampling_=false;
  const std::vector<std::string> outputs = sample_only_ ?
    std::vector<std::string>{spool_path_, provenance_path_, provenance_temp_path_} :
    std::vector<std::string>{output_path_, spool_path_, provenance_path_, provenance_temp_path_,
      output_path_ + ".stability", output_path_ + ".qraw",
      output_path_ + ".additive.tmp", output_path_ + ".fit.txt", output_path_ + ".fit.txt.tmp",
      output_path_ + ".fit_trace.txt", output_path_ + ".cg_witness.txt", output_path_ + ".qp_state.txt",
      output_path_ + ".failure.txt", output_path_ + ".tmp", output_path_ + ".stability.tmp"};
  for (const auto& path : outputs) {
    if (file_exists(path)) PRINT_INPUT_ERROR("rpmd_ja fit will not overwrite an existing output or temporary file.");
  }
  if (!sample_only_) {
    try {
      RpmdJAReference kernel_check;
      load_rpmd_ja_kernel_table(kernel_table_, kernel_check);
    } catch (const std::exception& error) {
      const std::string message = std::string("rpmd_ja fit kernel table is invalid: ") + error.what();
      PRINT_INPUT_ERROR(message.c_str());
    }
  }
#ifdef USE_HIP
  if (!sample_only_) PRINT_INPUT_ERROR("rpmd_ja fit native reference preparation is currently unavailable in HIP builds.");
#endif
  for (const double mass : atom.cpu_mass) {
    if (!(mass > 0.0) || !std::isfinite(mass)) PRINT_INPUT_ERROR("rpmd_ja fit requires positive finite masses.");
  }

  number_of_atoms_ = atom.number_of_atoms;
  number_of_beads_ = atom.number_of_beads;
  masses_ = atom.cpu_mass;
  const std::size_t coordinates = static_cast<std::size_t>(number_of_atoms_) * 3;
  previous_centroid_.resize(coordinates);
  centroid_.resize(coordinates);
  mean_force_.resize(coordinates);
  frame_buffer_.resize(1 + 2 * coordinates);
  bead_position_ptrs_.resize(number_of_beads_);
  bead_force_ptrs_.resize(number_of_beads_);
  for (int bead = 0; bead < number_of_beads_; ++bead) {
    bead_position_ptrs_[bead] = atom.position_beads[bead].data();
    bead_force_ptrs_[bead] = atom.force_beads[bead].data();
  }
  bead_position_ptrs_gpu_.resize(number_of_beads_);
  bead_force_ptrs_gpu_.resize(number_of_beads_);
  bead_position_ptrs_gpu_.copy_from_host(bead_position_ptrs_.data());
  bead_force_ptrs_gpu_.copy_from_host(bead_force_ptrs_.data());
  sample_output_.resize(static_cast<std::size_t>(10) * number_of_atoms_);
  sample_output_gpu_.resize(sample_output_.size());
  force_ = &force;
  const rpmd_ja_sample_provenance::SourceIdentity source_identity =
    rpmd_ja_sample_provenance::source_identity();
  source_job_id_ = source_identity.job_id;
  source_job_id_status_ = source_identity.job_id_status;
  source_run_id_ = source_identity.run_id;

  lock_file_ = rpmd_ja_file_io::open_file(lock_path_, "a+b");
  if (!lock_file_||!rpmd_ja_process_lock::try_lock(lock_file_)) {
    if(lock_file_){std::fclose(lock_file_);lock_file_=nullptr;}
    const std::string message = "rpmd_ja fit could not acquire its exclusive basename lock: " + lock_path_;
    PRINT_INPUT_ERROR(message.c_str());
  }
  spool_.open(spool_path_, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!spool_) {
    release_lock_();
    PRINT_INPUT_ERROR("rpmd_ja fit could not create its sample spool.");
  }
  const char magic[8] = {'G', 'P', 'J', 'A', 'S', 'M', 'P', '1'};
  const std::uint32_t version = 1;
  const std::uint32_t endian = 0x01020304;
  const std::int32_t n = number_of_atoms_;
  const std::int32_t p = number_of_beads_;
  try {
    write_or_throw(spool_, magic, sizeof(magic));
    write_or_throw(spool_, &version, sizeof(version));
    write_or_throw(spool_, &endian, sizeof(endian));
    write_or_throw(spool_, &n, sizeof(n));
    write_or_throw(spool_, &p, sizeof(p));
    write_or_throw(spool_, &temperature_, sizeof(temperature_));
    for (int i = 0; i < 9; ++i) write_or_throw(spool_, &cell_[i], sizeof(double));
    write_or_throw(spool_, masses_.data(), masses_.size() * sizeof(double));
    for (const std::int32_t type : atom.cpu_type) write_or_throw(spool_, &type, sizeof(type));
  } catch (...) {
    spool_.close();
    rpmd_ja_file_io::remove_file(spool_path_);
    release_lock_();
    throw;
  }
  printf("rpmd_ja %s sampling current physical bead forces at %g K, P=%d; spool %s\n",
    sample_only_ ? "sample" : "fit", temperature_, number_of_beads_, spool_path_.c_str());
}

void RpmdJA_Fit::post_force(
  const int step,
  const double,
  const double,
  Integrate&,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force&)
{
  if ((step + 1) % sample_interval_ != 0) return;
  const std::size_t coordinates = static_cast<std::size_t>(number_of_atoms_) * 3;
  for (int i = 0; i < 9; ++i) {
    if (std::abs(box.cpu_h[i] - cell_[i]) > 1.0e-12 * std::max(1.0, std::abs(cell_[i]))) {
      PRINT_INPUT_ERROR("rpmd_ja fit detected a changing cell during sampling.");
    }
  }
  if (pbc_[0] != box.pbc_x || pbc_[1] != box.pbc_y || pbc_[2] != box.pbc_z) {
    PRINT_INPUT_ERROR("rpmd_ja fit detected changing periodic boundary conditions during sampling.");
  }
  bool refresh_pointers = false;
  for (int bead = 0; bead < number_of_beads_; ++bead) {
    const double* position = atom.position_beads[bead].data();
    const double* force = atom.force_beads[bead].data();
    if (position != bead_position_ptrs_[bead] || force != bead_force_ptrs_[bead]) {
      bead_position_ptrs_[bead] = position;
      bead_force_ptrs_[bead] = force;
      refresh_pointers = true;
    }
  }
  if (refresh_pointers) {
    bead_position_ptrs_gpu_.copy_from_host(bead_position_ptrs_.data());
    bead_force_ptrs_gpu_.copy_from_host(bead_force_ptrs_.data());
  }
  RpmdJAFitCell sampling_box;
  for (int i = 0; i < 9; ++i) {
    sampling_box.cell[i] = box.cpu_h[i];
    sampling_box.inverse[i] = box.cpu_h[9 + i];
  }
  constexpr int block_size = 128;
  rpmd_ja_fit_sampling_kernel<<<(number_of_atoms_ + block_size - 1) / block_size, block_size>>>(
    number_of_atoms_, number_of_beads_, bead_position_ptrs_gpu_.data(),
    bead_force_ptrs_gpu_.data(), sampling_box, sample_output_gpu_.data());
  GPU_CHECK_KERNEL
  sample_output_gpu_.copy_to_host(sample_output_.data());

  const std::size_t atom_count = static_cast<std::size_t>(number_of_atoms_);
  std::copy(sample_output_.begin(), sample_output_.begin() + coordinates, centroid_.begin());
  std::copy(sample_output_.begin() + coordinates, sample_output_.begin() + 2 * coordinates, mean_force_.begin());
  double coordinate_scale = 1.0;
  for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
    const unsigned int error = static_cast<unsigned int>(sample_output_[9 * atom_count + atom_id]);
    if (error & 3) PRINT_INPUT_ERROR("rpmd_ja fit encountered a nonfinite bead position or physical force.");
    if (error & (4 | 8)) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite displacement.");
    if (error & 16) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite ring closure.");
    if (error & (32 | 64)) PRINT_INPUT_ERROR("rpmd_ja fit produced a non-finite winding or branch norm.");
    coordinate_scale = std::max(coordinate_scale, sample_output_[6 * atom_count + atom_id]);
  }
  double box_scale = 1.0;
  for (int i = 0; i < 9; ++i) box_scale = std::max(box_scale, std::abs(box.cpu_h[i]));
  if (coordinate_scale > 1.0e12 * box_scale) {
    PRINT_INPUT_ERROR("rpmd_ja fit rejected extreme coordinate offsets that make image branches ambiguous.");
  }
  const double ring_tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
    number_of_beads_ * std::max(box_scale, coordinate_scale);
  for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
    if (sample_output_[7 * atom_count + atom_id] > ring_tolerance) {
      PRINT_INPUT_ERROR("rpmd_ja fit rejected a ring path with nonzero periodic winding.");
    }
    if (sample_output_[8 * atom_count + atom_id] > ring_tolerance) {
      PRINT_INPUT_ERROR("rpmd_ja fit ring-unwrapped centroid differs from the bead-0 MIC centroid.");
    }
  }

  if (has_previous_centroid_) {
    for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
      const std::size_t x = static_cast<std::size_t>(atom_id);
      const std::size_t y = x + number_of_atoms_;
      const std::size_t z = y + number_of_atoms_;
      double dx = centroid_[x] - previous_centroid_[x];
      double dy = centroid_[y] - previous_centroid_[y];
      double dz = centroid_[z] - previous_centroid_[z];
      if (!apply_fit_mic(box, dx, dy, dz)) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite centroid displacement.");
      centroid_[x] = previous_centroid_[x] + dx;
      centroid_[y] = previous_centroid_[y] + dy;
      centroid_[z] = previous_centroid_[z] + dz;
    }
  }
  double total_mass = 0.0;
  double delta_com[3] = {0.0, 0.0, 0.0};
  if (has_previous_centroid_) {
    for (int i = 0; i < number_of_atoms_; ++i) {
      const double mass = masses_[i];
      total_mass += mass;
      for (int d = 0; d < 3; ++d) {
        delta_com[d] += mass * (centroid_[i + d * number_of_atoms_] - previous_centroid_[i + d * number_of_atoms_]);
      }
    }
    for (double& value : delta_com) value /= total_mass;
    for (int d = 0; d < 3; ++d) com_shift_[d] += delta_com[d];
  } else {
    for (const double mass : masses_) total_mass += mass;
  }
  if (!(total_mass > 0.0) || !std::isfinite(total_mass) ||
      !std::isfinite(delta_com[0]) || !std::isfinite(delta_com[1]) || !std::isfinite(delta_com[2])) {
    PRINT_INPUT_ERROR("rpmd_ja fit encountered an invalid total mass or COM displacement.");
  }
  previous_centroid_ = centroid_;

  for (int i = 0; i < number_of_atoms_; ++i) {
    for (int d = 0; d < 3; ++d) {
      centroid_[i + d * number_of_atoms_] -= com_shift_[d];
      if (!std::isfinite(centroid_[i + d * number_of_atoms_]) ||
          !std::isfinite(mean_force_[i + d * number_of_atoms_])) {
        PRINT_INPUT_ERROR("rpmd_ja fit produced a nonfinite centroid or bead-mean force.");
      }
    }
  }
  has_previous_centroid_ = true;

  const double sample_step = static_cast<double>(step + 1);
  frame_buffer_[0] = sample_step;
  std::copy(centroid_.begin(), centroid_.end(), frame_buffer_.begin() + 1);
  std::copy(mean_force_.begin(), mean_force_.end(), frame_buffer_.begin() + 1 + coordinates);
  write_or_throw(spool_, frame_buffer_.data(), frame_buffer_.size() * sizeof(double));
  if (frame_count_ == 0) first_sample_step_ = step + 1;
  last_sample_step_ = step + 1;
  ++frame_count_;
}

void RpmdJA_Fit::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double)
{
  if (sample_only_ && frame_count_ < 3) {
    PRINT_INPUT_ERROR("rpmd_ja sample requires at least three sampled frames; sample spool retained.");
  }
  if (!sample_only_ && frame_count_ < 3) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires at least three sampled frames; sample spool retained.");
  }
  spool_.flush();
  if (!spool_) PRINT_INPUT_ERROR("rpmd_ja fit failed to flush its sample spool.");
  spool_.close();
  if (spool_.fail()) PRINT_INPUT_ERROR("rpmd_ja fit failed to close its sample spool.");
  rpmd_ja_file_io::ifstream size_stream(spool_path_, std::ios::binary | std::ios::ate);
  if (!size_stream) {
    PRINT_INPUT_ERROR("rpmd_ja fit could not inspect its closed sample spool.");
  }
  const std::ifstream::pos_type spool_end = size_stream.tellg();
  if (spool_end == std::ifstream::pos_type(-1)) {
    PRINT_INPUT_ERROR("rpmd_ja fit could not inspect its closed sample spool.");
  }
  size_stream.close();
  if (size_stream.fail()) PRINT_INPUT_ERROR("rpmd_ja fit could not close its sample spool size reader.");
  const std::uint64_t bytes = static_cast<std::uint64_t>(spool_end);
  if(!provenance_snapshot_initialized_)PRINT_INPUT_ERROR("rpmd_ja fit has no active run configuration provenance snapshot.");
  const rpmd_ja_sample_provenance::ConfigurationSnapshot final_snapshot=
    rpmd_ja_sample_provenance::snapshot(*force_,atom,box,number_of_steps,
      active_run_ordinal_?active_run_ordinal_-1:0,time_step,temperature_,number_of_beads_,internal_mass_com_);
  const bool configuration_stable = configuration_stable_during_sampling_&&final_snapshot.active_configuration_identified&&
    final_snapshot.run_input_sha256 == run_input_sha256_ &&
    final_snapshot.physical_force_configuration_sha256 == physical_force_configuration_sha256_ &&
    final_snapshot.integrator_configuration_sha256 == integrator_configuration_sha256_ &&
    final_snapshot.model_sha256 == model_sha256_ && final_snapshot.atom_signature_sha256 == atom_signature_sha256_;
  const std::string sample_sha256 = rpmd_ja_sample_provenance::sha256_file(spool_path_);
  if (sample_sha256.empty()) PRINT_INPUT_ERROR("rpmd_ja fit could not hash its closed sample spool.");
  const std::string sidecar = build_provenance_json(sample_sha256, bytes, frame_count_, source_job_id_,
    source_job_id_status_, source_run_id_, first_sample_step_, last_sample_step_, timestep_internal_,
    timestep_fs_, temperature_, number_of_beads_, internal_mass_com_, com_shift_, thermostat_tokens_json_,
    final_snapshot.active_configuration_scope,final_snapshot.active_run_ordinal,
    model_path_, model_sha256_, final_snapshot.model_sha256, run_input_sha256_, final_snapshot.run_input_sha256,
    physical_force_configuration_sha256_, integrator_configuration_sha256_, atom_signature_sha256_, configuration_stable);
  try {
    write_exclusive_file(provenance_temp_path_, sidecar);
    write_exclusive_file(provenance_path_, sidecar);
    if (rpmd_ja_file_io::remove_file(provenance_temp_path_) != 0)
      throw std::runtime_error("could not remove completed RPMD-JA provenance temporary file");
  } catch (const std::exception& error) {
    const std::string message = std::string("rpmd_ja fit could not write sample provenance: ") + error.what();
    PRINT_INPUT_ERROR(message.c_str());
  }
  if (sample_only_) {
    release_lock_();
    std::printf("rpmd_ja sample saved %llu frames (%llu spool bytes) to %s; provenance %s; fitting was not run.\n",
      static_cast<unsigned long long>(frame_count_), static_cast<unsigned long long>(bytes), spool_path_.c_str(),
      provenance_path_.c_str());
    return;
  }
  RpmdJANativeFitOptions options;
  options.output_path = output_path_;
  options.kernel_table = kernel_table_;
  options.temperature = temperature_;
  options.cutoff = cutoff_;
  options.epsilon = epsilon_;
  options.response_tolerance = response_tolerance_;
  options.fd_step = fd_step_;
  options.sample_interval = sample_interval_;
  options.max_stability_rounds = max_stability_rounds_;
  options.internal_mass_com = integrate.get_pimd_fix_com();
  bool preserve_replay_inputs = false;
  try {
    preserve_replay_inputs = fit_rpmd_ja_native_reference(options, spool_path_, frame_count_, atom, box, *force_);
  } catch (...) {
    release_lock_();
    std::cerr << "rpmd_ja fit failed after " << frame_count_ << " frames; sample spool retained at "
              << spool_path_ << "\n";
    throw;
  }
  if (preserve_replay_inputs) {
    std::printf("rpmd_ja fit retained sample spool and provenance for CG replay: %s\n", spool_path_.c_str());
  } else if (rpmd_ja_file_io::remove_file(spool_path_) != 0) {
    std::cerr << "rpmd_ja fit produced the reference but could not remove its spool: " << spool_path_ << "\n";
  } else if (rpmd_ja_file_io::remove_file(provenance_path_) != 0) {
    std::cerr << "rpmd_ja fit removed its sample spool but could not remove its provenance sidecar: "
              << provenance_path_ << "\n";
  }
  std::printf("rpmd_ja fit sampled %llu frames (%llu spool bytes); reference written to %s\n",
    static_cast<unsigned long long>(frame_count_), static_cast<unsigned long long>(bytes), output_path_.c_str());
  release_lock_();
}
