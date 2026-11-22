#pragma once
#include "force/nep_charge.cuh"
#include "rpmd_ja_sha256.h"
#include "r164_ja_reference.h"
#include "model/box.cuh"
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
class Atom;
class Box;
class Force;
struct R164_PPPM_Trace;
class R164_RPMD {
public:
  bool enabled() const { return enabled_; }
  void parse(const std::vector<std::string>& tokens);
  void parse_hac(const std::vector<std::string>& tokens);
  void validate_hac(int total);
  void reset() {
    hac_nc_=0; hac_interval_=0; hac_sample_stride_=0; hac_correlation_only_=false;
    compression_.reset(); shadow_.reset(); centroid_virial_current_.reset(); centroid_conv_.clear(); compressed_ = false;
    enabled_ = false; observer_requested_ = false; drift_charge_requested_ = false; buffered_ = false; centroid_requested_ = false;
    sample_stride_ = 0; checkpoint_stride_ = 0; trace_step_ = -1;
    series_.clear(); series_.shrink_to_fit(); snapshots_.clear(); snapshots_.shrink_to_fit();
    ja_reference_path_.clear(); metadata_text_.clear(); identity_text_.clear(); observer_text_.clear(); observer_.reset();
  }
  void begin(Atom&, const Box&, double temperature, double dt, double start_time);
  void configure_observer(Force&, Atom&, const std::string& model_xyz_sha256, const std::string& restart_sha256);
  void validate_trace(const Force&, int total, const Atom&) const;
  bool trace_for_step(int step, R164_PPPM_Trace&) const;
  void capture_drift(Atom&);
  void check_force_positions(Atom&);
  void output(int step, int total, Atom&);
  void prepare_buffered(int total, const Atom&);
  void finish_buffered(Atom&);
private:
  using Clock = std::chrono::steady_clock;
  struct TransferTiming { uint64_t calls=0, bytes=0; double seconds=0; };
  TransferTiming d2h_, h2d_;
  double geometry_wall_=0, output_host_wall_=0, observer_wall_=0, finalization_wall_=0;
  template <typename T> void copy_to_host(const GPU_Vector<T>& source, T* dest) {
    if (!buffered_) { source.copy_to_host(dest); return; }
    const auto start=Clock::now(); source.copy_to_host(dest);
    d2h_.seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    ++d2h_.calls; d2h_.bytes+=source.size()*sizeof(T);
  }
  template <typename T> void copy_from_host(GPU_Vector<T>& dest, const T* source) {
    if (!buffered_) { dest.copy_from_host(source); return; }
    const auto start=Clock::now(); dest.copy_from_host(source);
    h2d_.seconds+=std::chrono::duration<double>(Clock::now()-start).count();
    ++h2d_.calls; h2d_.bytes+=dest.size()*sizeof(T);
  }
  bool enabled_ = false;
  int stride_ = 0, checkpoint_stride_ = 0, n_ = 0, p_ = 0;
  int trace_step_ = -1, trace_bead_ = -1, trace_atom_ = -1;
  double temperature_ = 0, dt_ = 0, lengths_[3] = {};
  std::string directory_, ja_reference_path_;
  R164JAReference ja_reference_;
  bool observer_requested_ = false, drift_charge_requested_ = false, buffered_ = false, compressed_ = false, centroid_requested_ = false;
  int sample_stride_ = 0, hac_sample_stride_ = 0, hac_nc_ = 0, hac_interval_ = 0;
  bool hac_correlation_only_ = false;
  struct SeriesRow {
    int step;
    double time, potential, kinetic, spring, energy, momentum[3], flows[4][3];
    bool b_valid, d2_valid, j2_valid;
    double b[3], d2[3], j2[3];
  };
  std::vector<SeriesRow> series_;
  std::unique_ptr<GPU_Vector<double>> centroid_virial_current_;
  std::vector<double> centroid_conv_;
  struct Snapshot {
    int step;
    std::vector<double> q, v, f, energy;
    int image_width;
    std::vector<unsigned char> images;
  };
  struct CompressionWorker;
  std::shared_ptr<CompressionWorker> compression_;
  std::vector<Snapshot> snapshots_;
  std::string metadata_text_, identity_text_, observer_text_;
  int buffered_total_ = 0;
  void write_text(const std::string&, const std::string&);
  void write_snapshot(int, const std::vector<double>&, const std::vector<double>&,
      const std::vector<double>&, const std::vector<double>&, const std::vector<long long>&, const Atom&);
  void write_complete(int step);
  rpmd_ja_sha256::Digest scalar_prefix_digest_;
  std::unique_ptr<NEP_Charge> observer_;
  Box box_for_observer_;
  GPU_Vector<double> observer_q_, observer_u_, observer_f_, observer_virial_, observer_v_, observer_site_rate_, observer_gradient_, observer_local_, observer_local_image_;
  GPU_Vector<double> observer_drift_v_;
  struct BufferedShadow {
    GPU_Vector<double> q, offset, staging;
    GPU_Vector<double*> sample_pointers;
    std::vector<double*> sample_pointers_host;
    std::vector<double> staging_host;
    GPU_Vector<long long> images;
    GPU_Vector<int> error;
    GPU_Vector<double*> positions;
    std::vector<double*> positions_host;
  };
  std::unique_ptr<BufferedShadow> shadow_;
  void shadow_prepare(Atom&);
  void shadow_stage(Atom&, int phase, bool save_images=false);
  std::vector<double> q_, offset_, position_scratch_;
  std::vector<int> species_;
  double number_increment_[4][3] = {};
  long long next_output_step_ = 0;
  bool drift_pending_ = false;
  std::vector<long long> force_images_, previous_force_images_;
  void read_positions(Atom&, std::vector<double>&);
  void check_branch(const std::vector<double>&) const;
};
