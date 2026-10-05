#include "rpmd_ja_fit.cuh"

#include "rpmd_ja_native_fit.cuh"
#include "rpmd_ja_reference.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
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
  std::ifstream in(path, std::ios::binary);
  return in.good();
}

void write_or_throw(std::ofstream& out, const void* data, const std::size_t bytes)
{
  if (bytes > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::runtime_error("RPMD-JA sample spool write exceeds stream limits");
  }
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!out) throw std::runtime_error("failed writing RPMD-JA sample spool");
}

bool apply_fit_mic(const Box& box, double& x, double& y, double& z)
{
  double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  if (!std::isfinite(sx) || !std::isfinite(sy) || !std::isfinite(sz)) return false;
  if (box.pbc_x) sx -= std::nearbyint(sx);
  if (box.pbc_y) sy -= std::nearbyint(sy);
  if (box.pbc_z) sz -= std::nearbyint(sz);
  x = box.cpu_h[0] * sx + box.cpu_h[1] * sy + box.cpu_h[2] * sz;
  y = box.cpu_h[3] * sx + box.cpu_h[4] * sy + box.cpu_h[5] * sz;
  z = box.cpu_h[6] * sx + box.cpu_h[7] * sy + box.cpu_h[8] * sz;
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

} // namespace

RpmdJA_Fit::RpmdJA_Fit(const std::vector<std::string>& tokens)
{
  action_name = "rpmd_ja_fit";
  if (tokens.size() != 9 || tokens[0] != "rpmd_ja" || tokens[1] != "fit") {
    throw std::invalid_argument(
      "rpmd_ja fit requires <outfile> <sample_interval> <cutoff> <epsilon> "
      "<response_tolerance> <fd_step> <kernel_table>");
  }
  output_path_ = tokens[2];
  char* end = nullptr;
  const long long interval = std::strtoll(tokens[3].c_str(), &end, 10);
  if (end == tokens[3].c_str() || *end != '\0' || interval <= 0 || interval > INT_MAX) {
    throw std::invalid_argument("rpmd_ja fit sample_interval must be a positive integer");
  }
  sample_interval_ = static_cast<int>(interval);
  cutoff_ = parse_positive(tokens[4], "cutoff");
  epsilon_ = parse_positive(tokens[5], "epsilon");
  response_tolerance_ = parse_positive(tokens[6], "response_tolerance");
  fd_step_ = parse_positive(tokens[7], "fd_step");
  kernel_table_ = tokens[8];
  if (output_path_.empty() || kernel_table_.empty()) {
    throw std::invalid_argument("rpmd_ja fit output and kernel table paths must be nonempty");
  }
  spool_path_ = output_path_ + ".samples.tmp";
  lock_path_ = output_path_ + ".fit.lock";
}

RpmdJA_Fit::~RpmdJA_Fit() { release_lock_(); }

void RpmdJA_Fit::release_lock_()
{
  if (!lock_file_) return;
  std::fclose(lock_file_);
  lock_file_ = nullptr;
  std::remove(lock_path_.c_str());
}

void RpmdJA_Fit::pre_run(
  const int number_of_steps,
  const double,
  Integrate& integrate,
  std::vector<Group>&,
  Atom& atom,
  Box& box,
  Force& force)
{
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
  if (frame_count < 3 || holdout_count <= expected_probes) {
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
  const std::string outputs[] = {output_path_, spool_path_, output_path_ + ".stability",
    output_path_ + ".qraw", output_path_ + ".additive.tmp", output_path_ + ".fit.txt",
    output_path_ + ".fit.txt.tmp", output_path_ + ".failure.txt", output_path_ + ".tmp",
    output_path_ + ".stability.tmp"};
  for (const auto& path : outputs) {
    if (file_exists(path)) PRINT_INPUT_ERROR("rpmd_ja fit will not overwrite an existing output or temporary file.");
  }
  try {
    RpmdJAReference kernel_check;
    load_rpmd_ja_kernel_table(kernel_table_, kernel_check);
  } catch (const std::exception& error) {
    const std::string message = std::string("rpmd_ja fit kernel table is invalid: ") + error.what();
    PRINT_INPUT_ERROR(message.c_str());
  }
#ifdef USE_HIP
  PRINT_INPUT_ERROR("rpmd_ja fit native reference preparation is currently unavailable in HIP builds.");
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
  mic_centroid_.resize(coordinates);
  ring_first_.resize(coordinates);
  bead_position_.resize(coordinates);
  previous_bead_.resize(coordinates);
  bead_force_.resize(coordinates);
  mean_force_.resize(coordinates);
  frame_buffer_.resize(1 + 2 * coordinates);
  force_ = &force;

  lock_file_ = std::fopen(lock_path_.c_str(), "wx");
  if (!lock_file_) {
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
    std::remove(spool_path_.c_str());
    release_lock_();
    throw;
  }
  printf("rpmd_ja fit sampling current physical bead forces at %g K, P=%d; spool %s\n",
    temperature_, number_of_beads_, spool_path_.c_str());
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
  std::fill(centroid_.begin(), centroid_.end(), 0.0);
  std::fill(mic_centroid_.begin(), mic_centroid_.end(), 0.0);
  std::fill(mean_force_.begin(), mean_force_.end(), 0.0);
  double coordinate_scale = 1.0;

  for (int bead = 0; bead < number_of_beads_; ++bead) {
    atom.position_beads[bead].copy_to_host(bead_position_.data());
    atom.force_beads[bead].copy_to_host(bead_force_.data());
    if (bead == 0) {
      previous_bead_ = bead_position_;
      ring_first_ = bead_position_;
      for (std::size_t i = 0; i < coordinates; ++i) centroid_[i] = bead_position_[i] / number_of_beads_;
    } else {
      for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
        const std::size_t x = static_cast<std::size_t>(atom_id);
        const std::size_t y = x + number_of_atoms_;
        const std::size_t z = y + number_of_atoms_;
        double dx = bead_position_[x] - previous_bead_[x];
        double dy = bead_position_[y] - previous_bead_[y];
        double dz = bead_position_[z] - previous_bead_[z];
        if (!apply_fit_mic(box, dx, dy, dz)) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite displacement.");
        previous_bead_[x] += dx;
        previous_bead_[y] += dy;
        previous_bead_[z] += dz;
        centroid_[x] += previous_bead_[x] / number_of_beads_;
        centroid_[y] += previous_bead_[y] / number_of_beads_;
        centroid_[z] += previous_bead_[z] / number_of_beads_;
      }
    }
    for (std::size_t i = 0; i < coordinates; ++i) {
      if (!std::isfinite(bead_position_[i]) || !std::isfinite(bead_force_[i])) {
        PRINT_INPUT_ERROR("rpmd_ja fit encountered a nonfinite bead position or physical force.");
      }
      coordinate_scale = std::max(coordinate_scale, std::abs(bead_position_[i]));
      mean_force_[i] += bead_force_[i] / number_of_beads_;
    }
    for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
      const std::size_t x = static_cast<std::size_t>(atom_id);
      const std::size_t y = x + number_of_atoms_;
      const std::size_t z = y + number_of_atoms_;
      double dx = bead_position_[x] - ring_first_[x];
      double dy = bead_position_[y] - ring_first_[y];
      double dz = bead_position_[z] - ring_first_[z];
      if (!apply_fit_mic(box, dx, dy, dz)) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite displacement.");
      mic_centroid_[x] += dx / number_of_beads_;
      mic_centroid_[y] += dy / number_of_beads_;
      mic_centroid_[z] += dz / number_of_beads_;
    }
  }
  double box_scale = 1.0;
  for (int i = 0; i < 9; ++i) box_scale = std::max(box_scale, std::abs(box.cpu_h[i]));
  if (coordinate_scale > 1.0e12 * box_scale) {
    PRINT_INPUT_ERROR("rpmd_ja fit rejected extreme coordinate offsets that make image branches ambiguous.");
  }
  const double ring_tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
    number_of_beads_ * std::max(box_scale, coordinate_scale);
  for (int atom_id = 0; atom_id < number_of_atoms_; ++atom_id) {
    const std::size_t x = static_cast<std::size_t>(atom_id);
    const std::size_t y = x + number_of_atoms_;
    const std::size_t z = y + number_of_atoms_;
    double close_x = ring_first_[x] - previous_bead_[x];
    double close_y = ring_first_[y] - previous_bead_[y];
    double close_z = ring_first_[z] - previous_bead_[z];
    if (!apply_fit_mic(box, close_x, close_y, close_z)) PRINT_INPUT_ERROR("rpmd_ja fit MIC produced a non-finite ring closure.");
    const double winding_x = previous_bead_[x] - ring_first_[x] + close_x;
    const double winding_y = previous_bead_[y] - ring_first_[y] + close_y;
    const double winding_z = previous_bead_[z] - ring_first_[z] + close_z;
    if (std::sqrt(winding_x * winding_x + winding_y * winding_y + winding_z * winding_z) > ring_tolerance) {
      PRINT_INPUT_ERROR("rpmd_ja fit rejected a ring path with nonzero periodic winding.");
    }
    double branch_x = centroid_[x] - (ring_first_[x] + mic_centroid_[x]);
    double branch_y = centroid_[y] - (ring_first_[y] + mic_centroid_[y]);
    double branch_z = centroid_[z] - (ring_first_[z] + mic_centroid_[z]);
    if (std::sqrt(branch_x * branch_x + branch_y * branch_y + branch_z * branch_z) > ring_tolerance) {
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
  ++frame_count_;
}

void RpmdJA_Fit::post_run(
  Atom& atom,
  Box& box,
  Integrate&,
  const int,
  const double,
  const double)
{
  if (frame_count_ < 3) {
    PRINT_INPUT_ERROR("rpmd_ja fit requires at least three sampled frames; sample spool retained.");
  }
  spool_.flush();
  if (!spool_) PRINT_INPUT_ERROR("rpmd_ja fit failed to flush its sample spool.");
  spool_.close();
  if (spool_.fail()) PRINT_INPUT_ERROR("rpmd_ja fit failed to close its sample spool.");
  std::ifstream size_stream(spool_path_, std::ios::binary | std::ios::ate);
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
  RpmdJANativeFitOptions options;
  options.output_path = output_path_;
  options.kernel_table = kernel_table_;
  options.temperature = temperature_;
  options.cutoff = cutoff_;
  options.epsilon = epsilon_;
  options.response_tolerance = response_tolerance_;
  options.fd_step = fd_step_;
  options.sample_interval = sample_interval_;
  try {
    fit_rpmd_ja_native_reference(options, spool_path_, frame_count_, atom, box, *force_);
  } catch (...) {
    release_lock_();
    std::cerr << "rpmd_ja fit failed after " << frame_count_ << " frames; sample spool retained at "
              << spool_path_ << "\n";
    throw;
  }
  if (std::remove(spool_path_.c_str()) != 0) {
    std::cerr << "rpmd_ja fit produced the reference but could not remove its spool: " << spool_path_ << "\n";
  }
  std::printf("rpmd_ja fit sampled %llu frames (%llu spool bytes); reference written to %s\n",
    static_cast<unsigned long long>(frame_count_), static_cast<unsigned long long>(bytes), output_path_.c_str());
  release_lock_();
}
