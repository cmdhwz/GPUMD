#include "r164_rpmd.cuh"
#include "r164_r27.h"
#include "model/atom.cuh"
#include "force/force.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/run_input.cuh"
#ifdef GPUMD_WPE_ENABLED
#include "wisevolve_potential_api.h"
#endif
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include "rpmd_ja_utf8_file_io.h"
#include "rpmd_ja_sample_provenance.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <limits>
#include <new>
#include <utility>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

void R164_RPMD::parse(const std::vector<std::string>& t) {
  if (enabled_ || t.size() < 3 || !is_valid_int(t[2], &stride_) || stride_ < 0 || t[1].empty())
    PRINT_INPUT_ERROR("r164_rpmd requires a fresh output directory and nonnegative debug stride.");
  size_t next=3;
  observer_requested_ = next<t.size() && t[next]=="observer";
  if (observer_requested_) ++next;
  drift_charge_requested_ = observer_requested_ && next<t.size() && t[next]=="drift_charge";
  if (drift_charge_requested_) ++next;
  buffered_=false; sample_stride_=0; series_.clear();
  checkpoint_stride_=0; trace_step_=-1;
  auto decimal = [](const std::string& text, int& value) {
    if (text.empty()) return false;
    value=0;
    for (char c : text) {
      if (c<'0' || c>'9' || value>(std::numeric_limits<int>::max()-(c-'0'))/10) return false;
      value=value*10+(c-'0');
    }
    return true;
  };
  if (next<t.size() && t[next]=="buffered") {
    if (!observer_requested_ || drift_charge_requested_ || next+1>=t.size() ||
        !decimal(t[next+1],sample_stride_) || sample_stride_<=0)
      PRINT_INPUT_ERROR("r164_rpmd buffered requires observer and a positive decimal sample stride.");
    buffered_=true; next+=2;
  }
  if (buffered_ && next<t.size())
    PRINT_INPUT_ERROR("r164_rpmd buffered cannot be combined with checkpoint, drift_charge or pppm_trace; use legacy diagnostic mode.");
  if (next<t.size() && t[next]=="checkpoint") {
    if (next+1>=t.size() || !decimal(t[next+1],checkpoint_stride_) || checkpoint_stride_<=0)
      PRINT_INPUT_ERROR("r164_rpmd checkpoint requires a positive decimal int stride.");
    next+=2;
  }
  if (next<t.size() && t[next]=="pppm_trace") {
    if (next+4!=t.size() || !decimal(t[next+1],trace_step_) || !decimal(t[next+2],trace_bead_) || !(t[next+3]=="all" ? (trace_atom_=-2,true) : decimal(t[next+3],trace_atom_)))
      PRINT_INPUT_ERROR("r164_rpmd pppm_trace requires nonnegative decimal step and bead, and a nonnegative decimal atom or all.");
    next+=4;
  }
  if (next!=t.size()) PRINT_INPUT_ERROR("r164_rpmd invalid options; use observer, drift_charge, checkpoint, pppm_trace in order.");
  directory_ = t[1]; enabled_ = true;
}
void R164_RPMD::validate_trace(const Force& force, int total, const Atom& atom) const {
  if (trace_step_<0) return;
  if (trace_step_>=total || trace_bead_>=atom.number_of_beads || trace_atom_>=atom.number_of_atoms)
    PRINT_INPUT_ERROR("r164_rpmd pppm_trace step, bead or atom is outside the run.");
  force.validate_r164_pppm_trace();
}
bool R164_RPMD::trace_for_step(int step, R164_PPPM_Trace& trace) const {
  if (trace_step_<0 || (step!=trace_step_ && step!=trace_step_+1)) return false;
  trace.prefix=directory_+"/pppm_trace_step_"+std::to_string(step)+"_bead_"+std::to_string(trace_bead_);
  trace.step=step; trace.bead=trace_bead_; trace.atom=trace_atom_;
  return true;
}
void R164_RPMD::read_positions(Atom& atom, std::vector<double>& q) {
  q.resize(p_ * n_ * 3);
  for (int s=0; s<p_; ++s) atom.position_beads[s].copy_to_host(q.data() + s*n_*3);
}
void R164_RPMD::check_branch(const std::vector<double>& q) const {
  for (int i=0; i<n_; ++i) for (int d=0; d<3; ++d) for (int s=1; s<p_; ++s) {
    double diff=q[s*n_*3+d*n_+i]-q[d*n_+i];
    if (!std::isfinite(diff) || std::abs(diff) >= lengths_[d]*0.5 - 1e-9)
      PRINT_INPUT_ERROR("r164_rpmd BLOCKED: first-bead MIC branch is ambiguous or differs from the loaded lift.");
  }
}
void R164_RPMD::begin(Atom& atom, const Box& box, double temperature, double dt, double start_time) {
  if (!box.is_orthogonal || !box.pbc_x || !box.pbc_y || !box.pbc_z || start_time != 0 ||
      !std::isfinite(temperature) || temperature <= 0 || !std::isfinite(dt) || dt <= 0)
    PRINT_INPUT_ERROR("r164_rpmd requires an orthogonal periodic cell, positive temperature and dt, and a fresh run.");
  n_=atom.number_of_atoms; p_=atom.number_of_beads; temperature_=temperature; dt_=dt; box_for_observer_=box;
  previous_force_images_.clear(); force_images_.clear();
  if (checkpoint_stride_) scalar_prefix_digest_=rpmd_ja_sha256::Digest();
  if (n_ <= 0 || p_ < 2) PRINT_INPUT_ERROR("r164_rpmd requires loaded bead arrays.");
  for (int d=0; d<3; ++d) { lengths_[d]=box.cpu_h[d*4]; if (!std::isfinite(lengths_[d]) || !(lengths_[d]>0)) PRINT_INPUT_ERROR("r164_rpmd invalid box."); }
  species_.resize(n_);
  for (int i=0; i<n_; ++i) {
    const auto& symbol=atom.cpu_atom_symbol[i];
    if (symbol != "O" && symbol != "H" && symbol != "Na" && symbol != "Cl")
      PRINT_INPUT_ERROR("r164_rpmd supports only O, H, Na and Cl.");
    species_[i]=symbol=="O"?0:symbol=="H"?1:symbol=="Na"?2:3;
  }
  next_output_step_=0; drift_pending_=false;
  read_positions(atom, q_); check_branch(q_);
#ifdef _WIN32
  int created = _wmkdir(rpmd_ja_file_io::utf8_to_wide_path(directory_).c_str());
#else
  int created = mkdir(directory_.c_str(), 0777);
#endif
  if (created != 0) PRINT_INPUT_ERROR("r164_rpmd output directory must not already exist and its parent must exist.");
  std::ostringstream meta;
  meta << std::setprecision(17) << "schema r164_diagnostic_v1\nstorage binary64_text_17_digits; force_backend_precision not_certified\n"
       << "layout q_p_F: bead,atom,xyz; U: bead\nreference_id R27_fixed_2OH_R157\nreference_source_sha256 1d46365935f48e390d08e69e8564a3adc692d439b9472b4250914b22144a2eaa\nreference_atom_identity_and_environment UNVERIFIED\nunits q=angstrom p=eV_fs_per_angstrom F=eV_per_angstrom U=eV time=fs\n"
       << "flow_b UNVERIFIED_OR_UNAVAILABLE\nd2 R27_CANDIDATE_UNVERIFIED_ROW_ID_AND_ENVIRONMENT\nJ2 UNVERIFIED_OR_UNAVAILABLE\nevents BLOCKED\n"
       << "checkpoint " << (checkpoint_stride_ ? "FRESH_REBUILD_ONLY" : "BLOCKED") << "\n"
       << "P " << p_ << "\nN " << n_ << "\ntemperature_K " << temperature_ << "\ndt_fs " << dt*TIME_UNIT_CONVERSION << "\n"
       << "time_convention step_times_fixed_dt_start_zero\n"
       << "number_increment " << (buffered_ ? "UNAVAILABLE_buffered_mode" : "lift_angstrom O_H_Na_Cl_xyz right_closed_initial_boundary_no_increment") << "\n"
       << "cell_angstrom " << lengths_[0] << ' ' << lengths_[1] << ' ' << lengths_[2] << "\n";
  if (buffered_) meta << "buffered_sample_stride " << sample_stride_ << "\n";
  if (drift_charge_requested_) meta << "drift_charge observer_frame_natural_units\n";
  if (checkpoint_stride_) meta << "checkpoint_stride " << checkpoint_stride_ << "\n";
  for (int i=0; i<n_; ++i) meta << "atom " << i << ' ' << atom.cpu_atom_symbol[i] << ' ' << atom.cpu_mass[i] << '\n';
  if (buffered_) metadata_text_=meta.str(); else write_text("metadata.txt",meta.str());
  offset_.resize(n_*3);
}
void R164_RPMD::configure_observer(Force& force, Atom& atom, const std::string& model_xyz_sha256, const std::string& restart_sha256) {
  observer_.reset();
  const std::string source_model=get_first_potential_filename(force.get_run_input());
  const std::string source_hash=rpmd_ja_sample_provenance::sha256_file(source_model);
  const std::string input_hash=rpmd_ja_sample_provenance::sha256_file("run.in");
  std::ostringstream identity;
#ifdef GPUMD_WPE_ENABLED
  identity << "bead_force_backend WPE\n";
#else
  identity << "bead_force_backend native_GPUMD\n";
#endif
  identity << "machine_scope GPU_and_driver_not_recorded\nlogical_bead_force_call_mapping initial_1_step_n_is_n_plus_1_backend_internal_calls_unrecorded\n";
#ifdef GPUMD_GIT_COMMIT_FULL
  identity << "build_git_commit " << GPUMD_GIT_COMMIT_FULL << "\n";
#else
  identity << "build_git_commit UNAVAILABLE\n";
#endif
#ifdef GPUMD_GIT_DIRTY
  identity << "build_git_dirty " << GPUMD_GIT_DIRTY << "\n";
#else
  identity << "build_git_dirty UNAVAILABLE\n";
#endif
  identity << "model_xyz_sha256 " << (model_xyz_sha256.empty()?"UNAVAILABLE":model_xyz_sha256)
           << "\nrestart_input_sha256 " << (restart_sha256.empty()?"UNAVAILABLE":restart_sha256)
           << "\ninput_hash_scope after_successful_read_unverified_against_concurrent_changes\n";
  identity << "model_sha256 " << (source_hash.empty()?"UNAVAILABLE":source_hash)
           << "\nrun_in_sha256 " << (input_hash.empty()?"UNAVAILABLE":input_hash) << "\n";
  if (buffered_) identity_text_=identity.str(); else write_text("source_identity.txt",identity.str());
  if (!observer_requested_) return;
  std::string model; double spacing=0;
#ifdef GPUMD_WPE_ENABLED
  const auto& stage=force.get_wpe_stage_snapshot();
  if (!stage.valid || stage.stage_kind != WPE_STAGE_RUN || stage.number_of_atoms != n_ ||
      stage.pimd_beads != p_ || stage.dftd3_enabled != 0 ||
      (stage.facts & WPE_STAGE_PIMD_ECO_CONFIGURATION) != 0 ||
      stage.qnep_kspace_method != WPE_QNEP_KSPACE_PPPM) {
    if (drift_charge_requested_ || buffered_) PRINT_INPUT_ERROR("r164_rpmd drift_charge or buffered requires an available private observer: incompatible WPE stage.");
    printf("r164_rpmd centroid observer BLOCKED: incompatible WPE stage.\n"); return;
  }
  model=get_first_potential_filename(force.get_run_input()); spacing=stage.qnep_pppm_spacing;
#else
  if (force.potentials.size()!=1 || force.get_run_input().contains("dftd3")) {
    if (drift_charge_requested_ || buffered_) PRINT_INPUT_ERROR("r164_rpmd drift_charge or buffered requires an available private observer: incompatible potential configuration.");
    printf("r164_rpmd centroid observer BLOCKED: incompatible potential configuration.\n"); return;
  }
  auto* active=dynamic_cast<NEP_Charge*>(force.potentials[0].get());
  if (!active || active->get_charge_mode()!=2 || !active->uses_pppm()) {
    if (drift_charge_requested_ || buffered_) PRINT_INPUT_ERROR("r164_rpmd drift_charge or buffered requires an available private observer: needs charge2 PPPM.");
    printf("r164_rpmd centroid observer BLOCKED: needs charge2 PPPM.\n"); return;
  }
  model=force.primary_nep_model_path(); spacing=active->get_pppm_mesh_spacing();
#endif
  rpmd_ja_file_io::ifstream in(model); std::string kind; in >> kind;
  if (!in || kind != "nep4_charge2" || !std::isfinite(spacing) || spacing < 0.2 || spacing > 2.0) {
    if (drift_charge_requested_ || buffered_) PRINT_INPUT_ERROR("r164_rpmd drift_charge or buffered requires an available private observer: unsupported model or PPPM spacing.");
    printf("r164_rpmd centroid observer BLOCKED: unsupported model or PPPM spacing.\n"); return;
  }
  std::ostringstream kspace; kspace << std::setprecision(17) << "kspace pppm " << spacing;
  const RunInput input(std::vector<std::string>{"potential " + model,kspace.str()});
  observer_.reset(new NEP_Charge(model.c_str(),n_,input));
  observer_->N1=0; observer_->N2=n_; observer_->set_pppm_mesh_spacing(spacing);
  observer_->configure_mechanical_observer(); observer_->enable_charge_diagnostics();
  observer_q_.resize(3*n_); observer_u_.resize(n_); observer_f_.resize(3*n_); observer_virial_.resize(9*n_); observer_v_.resize(3*n_);
  if (drift_charge_requested_) observer_drift_v_.resize(3*n_);
  const std::string model_hash=rpmd_ja_sample_provenance::sha256_file(model);
  std::ostringstream meta;
  meta << std::setprecision(17) << "native_private_qnep_pppm_observer\nmodel " << model
       << "\nmodel_sha256 " << (model_hash.empty()?"UNAVAILABLE":model_hash)
       << "\nrun_in_sha256 " << (input_hash.empty()?"UNAVAILABLE":input_hash)
       << "\npppm_spacing " << spacing << "\nframe_call_identity "
       << (buffered_ ? "sampled_step; observer_force_call ordinal step_div_sample_stride_plus_one" : "step; observer_force_call ordinal step_plus_one") << "\n";
  if (buffered_) observer_text_=meta.str(); else write_text("observer.txt",meta.str());
}
void R164_RPMD::capture_drift(Atom& atom) {
  read_positions(atom,position_scratch_); const auto& current=position_scratch_;
  if (next_output_step_<=0 || drift_pending_) PRINT_INPUT_ERROR("r164_rpmd invalid drift capture sequence.");
  for (auto& species : number_increment_) for (double& value : species) value=0;
  for (int s=0; s<p_; ++s) for (int d=0; d<3; ++d) for (int i=0; i<n_; ++i) {
    const int j=s*n_*3+d*n_+i;
    const double next=current[j]+offset_[d*n_+i];
    const double delta=next-q_[j];
    if (!std::isfinite(next) || !std::isfinite(delta)) PRINT_INPUT_ERROR("r164_rpmd non-finite drift increment.");
    number_increment_[species_[i]][d]+=delta/p_;
    q_[j]=next;
  }
  for (const auto& species : number_increment_) for (double value : species)
    if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd non-finite number increment.");
  check_branch(q_);
  drift_pending_=true;
}
void R164_RPMD::check_force_positions(Atom& atom) {
  read_positions(atom,position_scratch_); const auto& current=position_scratch_;
  force_images_.resize(q_.size());
  for (int s=0; s<p_; ++s) for (int d=0; d<3; ++d) for (int i=0; i<n_; ++i) {
    int j=s*n_*3+d*n_+i;
    double image=std::round((q_[j]-current[j])/lengths_[d]);
    if (!std::isfinite(q_[j]) || !std::isfinite(current[j]) || !std::isfinite(image) ||
        std::abs(image) > 9.0e15 || current[j] < -1e-9 || current[j] > lengths_[d]+1e-9 ||
        std::abs(q_[j]-current[j]-image*lengths_[d]) > 1e-8)
      PRINT_INPUT_ERROR("r164_rpmd BLOCKED: force position is not a canonical image of the lifted state.");
    force_images_[j]=static_cast<long long>(image);
  }
}
void R164_RPMD::write_text(const std::string& name, const std::string& text) {
  const std::string path=directory_+"/"+name;
  rpmd_ja_file_io::ofstream out(path+".tmp"); out << text; out.close();
  if (!out || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
    PRINT_INPUT_ERROR("r164_rpmd failed committing output metadata.");
}

void R164_RPMD::write_snapshot(int step, const std::vector<double>& q, const std::vector<double>& v,
  const std::vector<double>& f, const std::vector<double>& energy,
  const std::vector<long long>& force_images, const Atom& atom) {
  const std::string frame_path=directory_+"/frame_"+std::to_string(step)+".txt";
  rpmd_ja_file_io::ofstream frame(frame_path+".tmp"); frame << std::setprecision(17);
  for (int s=0;s<p_;++s) { frame << "bead " << s << ' ' << energy[s] << '\n';
    for (int i=0;i<n_;++i) { frame << i; for (int d=0;d<3;++d) frame << ' ' << q[s*n_*3+d*n_+i];
      for (int d=0;d<3;++d) frame << ' ' << atom.cpu_mass[i]*TIME_UNIT_CONVERSION*v[s*n_*3+d*n_+i];
      for (int d=0;d<3;++d) frame << ' ' << f[s*n_*3+d*n_+i]; frame << '\n'; }
  }
  frame.close(); if (!frame || rpmd_ja_file_io::atomic_commit_new_file(frame_path+".tmp",frame_path)!=0)
    PRINT_INPUT_ERROR("r164_rpmd failed committing frame.");
  const std::string image_path=directory_+"/force_images_"+std::to_string(step)+".txt";
  rpmd_ja_file_io::ofstream images(image_path+".tmp");
  images << "# bead atom image_x image_y image_z; q_lift=q_force+cell*image\n";
  for (int bead=0;bead<p_;++bead) for (int i=0;i<n_;++i)
    images << bead << ' ' << i << ' ' << force_images[bead*3*n_+i] << ' ' << force_images[bead*3*n_+n_+i] << ' ' << force_images[bead*3*n_+2*n_+i] << '\n';
  images.close(); if (!images || rpmd_ja_file_io::atomic_commit_new_file(image_path+".tmp",image_path)!=0)
    PRINT_INPUT_ERROR("r164_rpmd failed committing force images.");
}

void R164_RPMD::prepare_buffered(int total) {
  if (!buffered_) return;
  buffered_total_=total;
  const size_t sample_count=static_cast<size_t>(total)/sample_stride_+1;
  const size_t frame_count=stride_>0 ? 1+static_cast<size_t>(total)/stride_+(total%stride_!=0 ? 1 : 0) : 0;
  const double bytes=static_cast<double>(frame_count)*(static_cast<double>(p_)*n_*(9*sizeof(double)+3*sizeof(long long))+static_cast<double>(p_)*sizeof(double))
      +static_cast<double>(sample_count)*sizeof(SeriesRow);
  if (stride_>0) printf("r164_rpmd buffered output estimates %.3f GiB host memory for %zu snapshots and %zu samples; choose sparse frames for large runs.\n",bytes/(1024.0*1024.0*1024.0),frame_count,sample_count);
  try { series_.reserve(sample_count); snapshots_.reserve(frame_count); }
  catch (const std::bad_alloc&) { PRINT_INPUT_ERROR("r164_rpmd cannot allocate buffered output; reduce frame or sample frequency."); }
}

void R164_RPMD::finish_buffered(Atom& atom) {
  if (!buffered_) return;
  if (next_output_step_!=static_cast<long long>(buffered_total_)+1 || drift_pending_)
    PRINT_INPUT_ERROR("r164_rpmd cannot finish incomplete buffered output.");
  write_text("metadata.txt",metadata_text_);
  write_text("source_identity.txt",identity_text_);
  write_text("observer.txt",observer_text_);
  for (const auto& snapshot:snapshots_) write_snapshot(snapshot.step,snapshot.q,snapshot.v,snapshot.f,snapshot.energy,snapshot.images,atom);
  const std::string path=directory_+"/series.txt";
  rpmd_ja_file_io::ofstream out(path+".tmp"); out << std::setprecision(17);
  out << "# schema r164_buffered_series_v1\n# sample_stride " << sample_stride_
      << "\n# units time=fs energy=eV momentum=eV_fs_per_angstrom number_flow=angstrom_per_fs b_d2_J2=eV_angstrom_per_fs\n"
      << "# validity 1=UNVERIFIED 0=UNAVAILABLE; unavailable values nan\n"
      << "# columns step time_fs U_eV K_eV spring_eV H_per_PN_eV Px Py Pz NOx NOy NOz NHx NHy NHz NNax NNay NNaz NClx NCly NClz b_valid bx by bz d2_valid d2x d2y d2z J2_valid J2x J2y J2z\n";
  const double missing=std::numeric_limits<double>::quiet_NaN();
  for (const auto& row:series_) {
    out << row.step << ' ' << row.time << ' ' << row.potential << ' ' << row.kinetic << ' ' << row.spring << ' ' << row.energy;
    for (double value:row.momentum) out << ' ' << value;
    for (const auto& flow:row.flows) for (double value:flow) out << ' ' << value;
    out << ' ' << row.b_valid; for (double value:row.b) out << ' ' << (row.b_valid ? value : missing);
    out << ' ' << row.d2_valid; for (double value:row.d2) out << ' ' << (row.d2_valid ? value : missing);
    out << ' ' << row.j2_valid; for (double value:row.j2) out << ' ' << (row.j2_valid ? value : missing);
    out << '\n';
  }
  out.close();
  if (!out || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
    PRINT_INPUT_ERROR("r164_rpmd failed committing buffered series.");
  series_.clear(); series_.shrink_to_fit();
  snapshots_.clear(); snapshots_.shrink_to_fit();
  write_complete(buffered_total_);
}

void R164_RPMD::write_complete(int step) {
  const std::string marker=directory_+"/COMPLETE";
  rpmd_ja_file_io::ofstream done(marker+".tmp");
  done << "r164_diagnostic_v1 final_step " << step << "\n"; done.close();
  if (!done || rpmd_ja_file_io::atomic_commit_new_file(marker+".tmp",marker)!=0)
    PRINT_INPUT_ERROR("r164_rpmd failed committing completion marker.");
}

void R164_RPMD::output(int step, int total, Atom& atom) {
  if (step!=next_output_step_ || (step==0 ? drift_pending_ : !drift_pending_))
    PRINT_INPUT_ERROR("r164_rpmd invalid completed-step output sequence.");
  const double time_fs=step*dt_*TIME_UNIT_CONVERSION;
  const bool sample=!buffered_ || step%sample_stride_==0;
  const bool snapshot=stride_>0 && (step==0 || step==total || step%stride_==0);
  // The next drift starts after the integrator maps every bead relative to bead zero.
  read_positions(atom,position_scratch_); const auto& current=position_scratch_;
  for (int i=0;i<n_;++i) for (int d=0;d<3;++d) {
    int j=d*n_+i; offset_[j]=q_[j]-current[j];
    double image=std::round(offset_[j]/lengths_[d]);
    if (!std::isfinite(offset_[j]) || !std::isfinite(image) || std::abs(offset_[j]-image*lengths_[d])>1e-8)
      PRINT_INPUT_ERROR("r164_rpmd BLOCKED: bead zero has no verified integer image.");
    for (int s=1;s<p_;++s) {
      double delta=current[s*n_*3+j]-current[j];
      if (delta < -lengths_[d]*0.5) delta+=lengths_[d];
      else if (delta > lengths_[d]*0.5) delta-=lengths_[d];
      double expected=q_[s*n_*3+j]-q_[j];
      if (!std::isfinite(delta) || !std::isfinite(expected) || std::abs(std::abs(delta)-lengths_[d]*0.5)<1e-9 || std::abs(delta-expected)>1e-8)
        PRINT_INPUT_ERROR("r164_rpmd BLOCKED: propagation MIC branch differs from continuous lift.");
    }
  }
  if (buffered_ && !sample && !snapshot) {
    drift_pending_=false; ++next_output_step_;
    return;
  }
  const bool need_force=!buffered_ || snapshot;
  std::vector<double> v, f, u;
  try { v.resize(p_*n_*3); if (need_force) f.resize(p_*n_*3); u.resize(p_*n_); }
  catch (const std::bad_alloc&) {
    if (!buffered_) throw;
    PRINT_INPUT_ERROR("r164_rpmd cannot allocate buffered frame workspace; reduce frame or sample frequency.");
  }
  for (int s=0; s<p_; ++s) {
    atom.velocity_beads[s].copy_to_host(v.data()+s*n_*3);
    if (need_force) atom.force_beads[s].copy_to_host(f.data()+s*n_*3);
    atom.potential_beads[s].copy_to_host(u.data()+s*n_);
  }
  double Q[2]={}, Pi[2]={}, d2[3]={};
  const double mass_factor=TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION;
  const bool r27_candidate = n_ == 5896 && std::abs(temperature_-300.0)<1e-8 &&
    std::abs(atom.cpu_mass[715]*mass_factor-1658.1794888315474)<1e-8 &&
    std::abs(atom.cpu_mass[3382]*mass_factor-104.47183728621788)<1e-8 &&
    std::abs(atom.cpu_mass[5674]*mass_factor-104.47183728621788)<1e-8 && atom.cpu_atom_symbol[715] == "O" &&
    atom.cpu_atom_symbol[3382] == "H" && atom.cpu_atom_symbol[5674] == "H" &&
    std::abs(lengths_[0]-29.3104)<1e-8 && std::abs(lengths_[1]-29.3104)<1e-8 &&
    std::abs(lengths_[2]-29.3104)<1e-8;
  if (r27_candidate) {
    for (int a=0; a<3; ++a) {
      int i=r164_r27::row[a];
      for (int d=0; d<3; ++d) {
        double qc=0, pc=0;
        for (int bead=0; bead<p_; ++bead) {
          int j=bead*n_*3+d*n_+i;
          qc+=q_[j]/p_;
          pc+=atom.cpu_mass[i]*TIME_UNIT_CONVERSION*v[j]/p_;
        }
        for (int k=0; k<2; ++k) {
          Q[k]+=atom.cpu_mass[i]*TIME_UNIT_CONVERSION*TIME_UNIT_CONVERSION *
            r164_r27::basis[a][d][k]*(qc-r164_r27::reference[a][d]);
          Pi[k]+=r164_r27::basis[a][d][k]*pc;
        }
      }
    }
    for (int d=0;d<3;++d) for (int r=0;r<2;++r) for (int j=0;j<2;++j)
      d2[d]+=Pi[r]*r164_r27::D[d][r][j]*Q[j];
    for (double value:Q) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd nonfinite R27 Q.");
    for (double value:Pi) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd nonfinite R27 Pi.");
    for (double value:d2) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd nonfinite R27 correction.");
  }
  for (double value : q_) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd non-finite q before observer.");
  for (double value : v) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd non-finite velocity before observer.");
  for (double value : f) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd non-finite force before observer.");
  for (double value : u) if (!std::isfinite(value)) PRINT_INPUT_ERROR("r164_rpmd non-finite potential before observer.");
  for (double mass : atom.cpu_mass) if (!std::isfinite(mass) || mass <= 0) PRINT_INPUT_ERROR("r164_rpmd invalid mass.");
  double observer_conv[3]={}, local_current[3]={}, local_image[3]={}, mechanical[3]={}, mechanical_image[3]={}, relay[3]={}, relay_image[3]={}, projection[3]={}, projection_image[3]={}, chart[3]={}, site_rate_sum=0, gradient_power=0, native_power=0, force_difference_max=0;
  const bool projection_frame=!buffered_ && (step==0 || step==total || (stride_ && step%stride_==0));
  bool tangent_valid=false, mechanical_valid=false, relay_valid=false, projection_valid=false;
  std::vector<double> centroid;
  std::vector<float> raw_D, raw_rate;
  const char* projection_status="charge_diagnostics_unavailable";
  int projection_guard_axis=-1, projection_guard_index=-1;
  if (observer_ && sample) {
    centroid.resize(3*n_); std::vector<double> cv(3*n_);
    for (int bead=0; bead<p_; ++bead) for (int j=0; j<3*n_; ++j) {
      centroid[j]+=q_[bead*3*n_+j]/p_; cv[j]+=v[bead*3*n_+j]/p_;
    }
    for (int d=0; d<3; ++d) for (int i=0; i<n_; ++i) {
      int j=d*n_+i; centroid[j]-=std::floor(centroid[j]/lengths_[d])*lengths_[d];
    }
    observer_q_.copy_from_host(centroid.data()); observer_u_.fill(0); observer_f_.fill(0); observer_virial_.fill(0);
    observer_->request_peratom_virial_for_next_force(); observer_->request_charge_diagnostics_for_next_force();
    observer_->compute(box_for_observer_,atom.type,observer_q_,observer_u_,observer_f_,observer_virial_);
    std::vector<double> site(n_), native_f(3*n_), site_rate(n_), gradient(3*n_);
    observer_u_.copy_to_host(site.data()); observer_f_.copy_to_host(native_f.data());
    for (double x:site) if (!std::isfinite(x)) PRINT_INPUT_ERROR("r164_rpmd observer nonfinite energy.");
    for (double x:native_f) if (!std::isfinite(x)) PRINT_INPUT_ERROR("r164_rpmd observer nonfinite force.");
    observer_v_.copy_from_host(cv.data());
    tangent_valid=observer_->compute_reference_site_energy_derivative(box_for_observer_, atom.type,
      observer_q_, observer_f_, &observer_v_, &observer_site_rate_, observer_gradient_, nullptr, &observer_local_, &observer_local_image_);
    if (tangent_valid) {
      observer_site_rate_.copy_to_host(site_rate.data()); observer_gradient_.copy_to_host(gradient.data());
      std::vector<double> local(3*n_), images(3*n_); observer_local_.copy_to_host(local.data()); observer_local_image_.copy_to_host(images.data());
      for (int d=0;d<3;++d) for (int i=0;i<n_;++i) {
        if (!std::isfinite(local[d*n_+i])) PRINT_INPUT_ERROR("r164_rpmd nonfinite local current.");
        local_current[d]+=local[d*n_+i]/TIME_UNIT_CONVERSION;
        if (!std::isfinite(images[d*n_+i])) PRINT_INPUT_ERROR("r164_rpmd nonfinite local image.");
        local_image[d]+=images[d*n_+i]/TIME_UNIT_CONVERSION;
      }
      for (double x:site_rate) { if (!std::isfinite(x)) PRINT_INPUT_ERROR("r164_rpmd nonfinite site tangent."); site_rate_sum+=x/TIME_UNIT_CONVERSION; }
      for (int i=0;i<n_;++i) {
        double force_power=0;
        for (int d=0;d<3;++d) force_power-=gradient[d*n_+i]*cv[d*n_+i];
        for (int d=0;d<3;++d) chart[d]+=centroid[d*n_+i]*(site_rate[i]+force_power)/TIME_UNIT_CONVERSION;
      }
      for (int j=0;j<3*n_;++j) {
        if (!std::isfinite(gradient[j])) PRINT_INPUT_ERROR("r164_rpmd nonfinite energy gradient.");
        gradient_power+=gradient[j]*cv[j]/TIME_UNIT_CONVERSION;
        native_power+=native_f[j]*cv[j]/TIME_UNIT_CONVERSION;
        force_difference_max=std::max(force_difference_max,std::abs(gradient[j]+native_f[j]));
      }
    }
    mechanical_valid=observer_->compute_reference_mechanical_current(box_for_observer_,observer_q_,observer_v_,mechanical,mechanical_image);
    if (mechanical_valid) for (int d=0;d<3;++d) { mechanical[d]/=TIME_UNIT_CONVERSION; mechanical_image[d]/=TIME_UNIT_CONVERSION; if (!std::isfinite(mechanical[d]) || !std::isfinite(mechanical_image[d])) mechanical_valid=false; }
    if (observer_->has_charge_diagnostics_for_current_force_frame()) {
      observer_->compute_charge_rate_for_current_force_frame(box_for_observer_,atom.type,observer_q_,observer_v_);
      relay_valid=observer_->compute_reference_relay_current(box_for_observer_,observer_q_,observer_v_,relay,relay_image);
      if (relay_valid) for (int d=0;d<3;++d) { relay[d]/=TIME_UNIT_CONVERSION; relay_image[d]/=TIME_UNIT_CONVERSION; if (!std::isfinite(relay[d]) || !std::isfinite(relay_image[d])) relay_valid=false; }
      raw_D.resize(n_); raw_rate.resize(n_);
      observer_->get_raw_D_reference().copy_to_host(raw_D.data());
      observer_->get_raw_charge_rate_reference().copy_to_host(raw_rate.data());
      double mean_D=0, mean_rate=0;
      projection_valid=true; projection_status="valid";
      for (int i=0;i<n_;++i) { if (!std::isfinite(raw_D[i]) || !std::isfinite(raw_rate[i])) { projection_valid=false; projection_status="nonfinite"; } mean_D+=raw_D[i]/n_; mean_rate+=raw_rate[i]/n_; }
      for (int d=0;d<3 && projection_valid;++d) {
        std::vector<double> ordered(n_); double mean_x=0, L=lengths_[d];
        for (int i=0;i<n_;++i) { ordered[i]=centroid[d*n_+i]; mean_x+=ordered[i]/n_; }
        std::sort(ordered.begin(),ordered.end());
        for (int i=0;i<n_;++i) {
          double x=centroid[d*n_+i], left=x-L/2, right=x+L/2;
          auto li=std::lower_bound(ordered.begin(),ordered.end(),left);
          auto ri=std::upper_bound(ordered.begin(),ordered.end(),right);
          auto left_near=std::lower_bound(ordered.begin(),ordered.end(),left-1e-9);
          auto right_near=std::lower_bound(ordered.begin(),ordered.end(),right-1e-9);
          if ((left_near!=ordered.end() && *left_near<=left+1e-9) ||
              (right_near!=ordered.end() && *right_near<=right+1e-9)) { projection_valid=false; projection_status="half_box_guard"; projection_guard_axis=d; projection_guard_index=i; break; }
          double left_count=li-ordered.begin(), right_count=ordered.end()-ri;
          double eta=x-mean_x+L*(right_count-left_count)/n_;
          double c=mean_D*raw_rate[i]-raw_D[i]*mean_rate;
          projection[d]+=eta*c/TIME_UNIT_CONVERSION;
          projection_image[d]+=L*(right_count-left_count)/n_*c/TIME_UNIT_CONVERSION;
        }
      }
      for (double x:projection) if (!std::isfinite(x)) { projection_valid=false; projection_status="nonfinite"; projection_guard_axis=-1; projection_guard_index=-1; }
    }
    if (projection_frame) {
      const std::string path=directory_+"/observer_frame_"+std::to_string(step)+".txt";
      rpmd_ja_file_io::ofstream out(path+".tmp"); out << std::setprecision(17);
      const int* mesh=observer_->get_pppm_mesh();
      out << "# step " << step << " force_call " << step+1 << " mesh " << mesh[0] << ' ' << mesh[1] << ' ' << mesh[2] << " tangent " << (tangent_valid?"VALID":"UNAVAILABLE") << "\n";
      for (int i=0;i<n_;++i) {
        out << i << ' ' << site[i]; for (int d=0;d<3;++d) out << ' ' << native_f[d*n_+i];
        if (tangent_valid) { out << ' ' << site_rate[i]/TIME_UNIT_CONVERSION; for (int d=0;d<3;++d) out << ' ' << gradient[d*n_+i]; }
        out << '\n';
      }
      out.close(); if (!out || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
        PRINT_INPUT_ERROR("r164_rpmd failed committing observer frame.");
    }
    for (int i=0;i<n_;++i) {
      double v2=0; for (int d=0;d<3;++d) v2+=cv[d*n_+i]*cv[d*n_+i];
      double energy=site[i]+0.5*atom.cpu_mass[i]*v2;
      for (int d=0;d<3;++d) observer_conv[d]+=energy*cv[d*n_+i]/TIME_UNIT_CONVERSION;
    }
  }
  double kinetic=0, spring=0, potential=0, momentum[3]={}, flows[4][3]={};
  const double omega=p_*K_B*temperature_/HBAR;
  for (int s=0; s<p_; ++s) for (int i=0; i<n_; ++i) {
    const double mass=atom.cpu_mass[i]; potential+=u[s*n_+i];
    int sp=species_[i];
    for (int d=0; d<3; ++d) {
      int j=s*n_*3+d*n_+i, next=((s+1)%p_)*n_*3+d*n_+i;
      double velocity=v[j], delta=q_[next]-q_[j];
      if (!std::isfinite(velocity) || !std::isfinite(delta) || (need_force && !std::isfinite(f[j])) || !std::isfinite(u[s*n_+i]) || !std::isfinite(mass) || !(mass>0))
        PRINT_INPUT_ERROR("r164_rpmd non-finite or invalid state.");
      kinetic+=0.5*mass*velocity*velocity; spring+=0.5*mass*omega*omega*delta*delta;
      momentum[d]+=mass*TIME_UNIT_CONVERSION*velocity/p_;
      flows[sp][d]+=velocity/TIME_UNIT_CONVERSION/p_;
    }
  }
  const bool b_valid=observer_ && tangent_valid && mechanical_valid && relay_valid && projection_valid;
  double candidate[3]={};
  if (b_valid) for (int d=0;d<3;++d) {
    candidate[d]=observer_conv[d]+local_current[d]+mechanical[d]+relay[d]+projection[d];
    if (!std::isfinite(candidate[d])) PRINT_INPUT_ERROR("r164_rpmd nonfinite b candidate.");
  }
  const bool checkpoint = checkpoint_stride_ && (step==0 || step==total || step%checkpoint_stride_==0);
  R164_PPPM_Trace trace;
  const bool trace_here=trace_for_step(step,trace);
  const bool save_snapshot=snapshot || (!buffered_ && (step==0 || step==total || checkpoint || trace_here));
  if (save_snapshot) {
    try {
      std::vector<double> energy(p_,0.0);
      for (int bead=0;bead<p_;++bead) for (int i=0;i<n_;++i) energy[bead]+=u[bead*n_+i];
      if (buffered_) {
        Snapshot frame; frame.step=step; frame.q=q_; frame.images=force_images_; frame.energy=std::move(energy);
        frame.v=std::move(v); frame.f=std::move(f); snapshots_.push_back(std::move(frame));
      } else write_snapshot(step,q_,v,f,energy,force_images_,atom);
    } catch (const std::bad_alloc&) {
      if (!buffered_) throw;
      PRINT_INPUT_ERROR("r164_rpmd cannot allocate buffered snapshot; reduce frame frequency.");
    }
  }
  if (buffered_) {
    if (sample) {
      SeriesRow row{};
      row.step=step; row.time=time_fs; row.potential=potential; row.kinetic=kinetic;
      row.spring=spring; row.energy=(potential+kinetic+spring)/(p_*n_);
      row.b_valid=b_valid;
      row.d2_valid=r27_candidate; row.j2_valid=row.b_valid && row.d2_valid;
      for (int d=0;d<3;++d) {
        row.momentum[d]=momentum[d];
        for (int sp=0;sp<4;++sp) row.flows[sp][d]=flows[sp][d];
        if (row.b_valid) {
          row.b[d]=candidate[d];
        }
        if (row.d2_valid) row.d2[d]=d2[d];
        if (row.j2_valid) {
          row.j2[d]=row.b[d]+d2[d];
          if (!std::isfinite(row.j2[d])) PRINT_INPUT_ERROR("r164_rpmd nonfinite J2 candidate.");
        }
      }
      series_.push_back(row);
    }
    drift_pending_=false; ++next_output_step_;
    return;
  }
  if (!previous_force_images_.empty()) {
    size_t changed=0; for (size_t i=0;i<force_images_.size();++i) if (force_images_[i]!=previous_force_images_[i]) ++changed;
    if (changed) {
      rpmd_ja_file_io::ofstream crossings(directory_+"/geometric_chart_changes.txt",std::ios::app);
      crossings << step-1 << ' ' << step << ' ' << changed << " ENDPOINT_ONLY_NOT_P11\n";
      crossings.close(); if (!crossings) PRINT_INPUT_ERROR("r164_rpmd failed closing geometric changes.");
    }
  }
  previous_force_images_=force_images_;

  const std::string scalar_path=directory_+"/scalar.txt";
  rpmd_ja_file_io::ofstream scalar_file(scalar_path, std::ios::app);
  if (step == 0) scalar_file << "# step time_fs U_eV K_eV spring_eV H_per_PN_eV Px Py Pz NOx NOy NOz NHx NHy NHz NNax NNay NNaz NClx NCly NClz observer_status [conv_xyz local_status local_xyz local_image_status local_image_xyz mechanical_image_status mechanical_image_xyz relay_image_status relay_image_xyz projection_image_status projection_image_xyz mechanical_status mechanical_xyz relay_status relay_xyz projection_status projection_xyz tangent_status site_rate_sum gradient_power native_power max_force_residual] b_status b_xyz J2_status J2_xyz R27_status [Q0 Q1 Pi0 Pi1 d2x d2y d2z] number_increment_status [O_H_Na_Cl_xyz_angstrom]\n";
  std::ostringstream scalar;
  scalar << std::setprecision(17) << step << ' ' << time_fs << ' ' << potential << ' ' << kinetic << ' ' << spring << ' ' << (potential+kinetic+spring)/(p_*n_);
  for (double x:momentum) scalar << ' ' << x;
  for (auto& species:flows) for (double x:species) scalar << ' ' << x;
  if (observer_) {
    for (double x:observer_conv) if (!std::isfinite(x)) PRINT_INPUT_ERROR("r164_rpmd nonfinite observer convective diagnostic.");
    scalar << " OBSERVER_CONV_UNVERIFIED " << observer_conv[0] << ' ' << observer_conv[1] << ' ' << observer_conv[2];
    if (tangent_valid) scalar << " LOCAL_UNVERIFIED " << local_current[0] << ' ' << local_current[1] << ' ' << local_current[2];
    else scalar << " LOCAL_UNAVAILABLE";
    if (tangent_valid) scalar << " LOCAL_IMAGE_UNVERIFIED " << local_image[0] << ' ' << local_image[1] << ' ' << local_image[2];
    else scalar << " LOCAL_IMAGE_UNAVAILABLE";
    if (mechanical_valid) scalar << " MECHANICAL_IMAGE_UNVERIFIED " << mechanical_image[0] << ' ' << mechanical_image[1] << ' ' << mechanical_image[2];
    else scalar << " MECHANICAL_IMAGE_UNAVAILABLE";
    if (relay_valid) scalar << " RELAY_IMAGE_UNVERIFIED " << relay_image[0] << ' ' << relay_image[1] << ' ' << relay_image[2];
    else scalar << " RELAY_IMAGE_UNAVAILABLE";
    if (projection_valid) scalar << " PROJECTION_IMAGE_UNVERIFIED " << projection_image[0] << ' ' << projection_image[1] << ' ' << projection_image[2];
    else scalar << " PROJECTION_IMAGE_UNAVAILABLE";
    if (mechanical_valid) scalar << " MECHANICAL_UNVERIFIED " << mechanical[0] << ' ' << mechanical[1] << ' ' << mechanical[2];
    else scalar << " MECHANICAL_UNAVAILABLE";
    if (relay_valid) scalar << " RELAY_UNVERIFIED " << relay[0] << ' ' << relay[1] << ' ' << relay[2];
    else scalar << " RELAY_UNAVAILABLE";
    if (projection_valid) scalar << " PROJECTION_UNVERIFIED " << projection[0] << ' ' << projection[1] << ' ' << projection[2];
    else scalar << " PROJECTION_UNAVAILABLE";
    if (tangent_valid) scalar << " TANGENT_VALID " << site_rate_sum << ' ' << gradient_power << ' ' << native_power << ' ' << force_difference_max;
    else scalar << " TANGENT_UNAVAILABLE";
  }
  else scalar << " OBSERVER_UNAVAILABLE";
  if (b_valid) {
    scalar << " B_UNVERIFIED " << candidate[0] << ' ' << candidate[1] << ' ' << candidate[2];
    double regular[3], residual[3];
    for (int d=0;d<3;++d) {
      regular[d]=observer_conv[d]+chart[d]+local_image[d]+mechanical_image[d]+relay_image[d]+projection_image[d];
      residual[d]=regular[d]-candidate[d];
      if (!std::isfinite(regular[d]) || !std::isfinite(residual[d])) PRINT_INPUT_ERROR("r164_rpmd nonfinite regular closure.");
    }
    scalar << " CHART_UNVERIFIED " << chart[0] << ' ' << chart[1] << ' ' << chart[2]
           << " B_REGULAR_UNVERIFIED " << regular[0] << ' ' << regular[1] << ' ' << regular[2]
           << " REGULAR_MINUS_B " << residual[0] << ' ' << residual[1] << ' ' << residual[2]
           << " POWER_RESIDUALS " << site_rate_sum-gradient_power << ' ' << site_rate_sum+native_power;
    if (r27_candidate) scalar << " J2_UNVERIFIED " << candidate[0]+d2[0] << ' ' << candidate[1]+d2[1] << ' ' << candidate[2]+d2[2];
    else scalar << " J2_UNAVAILABLE";
  } else scalar << " B_UNAVAILABLE J2_UNAVAILABLE";
  if (r27_candidate) scalar << " R27_UNVERIFIED " << Q[0] << ' ' << Q[1] << ' ' << Pi[0] << ' ' << Pi[1] << ' ' << d2[0] << ' ' << d2[1] << ' ' << d2[2];
  else scalar << " R27_NOT_APPLICABLE";
  if (step==0) scalar << " NUMBER_INCREMENT_INITIAL_BOUNDARY";
  else {
    scalar << " NUMBER_INCREMENT_LIFT_A";
    for (const auto& species : number_increment_) for (double value : species) scalar << ' ' << value;
  }
  scalar << '\n';
  const std::string scalar_line=scalar.str();
  scalar_file << scalar_line;
  scalar_file.close(); if (!scalar_file) PRINT_INPUT_ERROR("r164_rpmd failed closing scalar output.");
  drift_pending_=false; ++next_output_step_;
  if (checkpoint_stride_) scalar_prefix_digest_.update(reinterpret_cast<const unsigned char*>(scalar_line.data()),scalar_line.size());
  if (observer_ && projection_frame) {
    const std::string suffix=std::to_string(step)+".txt";
    const std::string files[]={directory_+"/metadata.txt", directory_+"/source_identity.txt", directory_+"/observer.txt",
                               directory_+"/frame_"+suffix, directory_+"/observer_frame_"+suffix};
    const char* names[]={"metadata", "source_identity", "observer", "frame", "observer_frame"};
    std::string hashes[5];
    for (int k=0;k<5;++k) {
      hashes[k]=rpmd_ja_sample_provenance::sha256_file(files[k]);
      if (hashes[k].size()!=64) PRINT_INPUT_ERROR("r164_rpmd failed hashing observer projection inputs.");
    }
    const std::string scalar_hash=rpmd_ja_sample_provenance::sha256_text(scalar_line);
    if (scalar_hash.size()!=64) PRINT_INPUT_ERROR("r164_rpmd failed hashing observer projection scalar.");
    const bool have_rows=raw_D.size()==static_cast<size_t>(n_) && raw_rate.size()==static_cast<size_t>(n_);
    const std::string path=directory_+"/observer_projection_"+suffix;
    rpmd_ja_file_io::ofstream out(path+".tmp"); out << std::setprecision(17);
    out << "# schema r164_observer_projection_v1\n# step " << step << "\n# time_fs " << time_fs
        << "\n# observer_force_call " << step+1 << "\n# N " << n_ << "\n# status " << projection_status
        << "\n# guard_axis " << projection_guard_axis << "\n# guard_index " << projection_guard_index
        << "\n# rows " << (have_rows?n_:0) << "\n# time_unit_conversion " << TIME_UNIT_CONVERSION
        << "\n# half_box_guard_A 1e-9\n# columns atom centroid_canonical_xyz_A raw_D_fp32_eV_per_e raw_rate_fp32_e_per_natural_time\n";
    for (int k=0;k<5;++k) out << "# " << names[k] << "_sha256 " << hashes[k] << '\n';
    out << "# scalar_line_sha256 " << scalar_hash << '\n';
    if (have_rows) for (int i=0;i<n_;++i) {
      out << i; for (int d=0;d<3;++d) out << ' ' << centroid[d*n_+i];
      out << ' ' << raw_D[i] << ' ' << raw_rate[i] << '\n';
    }
    out.close(); if (!out || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
      PRINT_INPUT_ERROR("r164_rpmd failed committing observer projection.");
    if (drift_charge_requested_) {
      const std::string projection_hash=rpmd_ja_sample_provenance::sha256_file(path);
      if (projection_hash.size()!=64) PRINT_INPUT_ERROR("r164_rpmd failed hashing observer projection.");
      const bool available=observer_->has_charge_diagnostics_for_current_force_frame();
      const char* status[2]={step>0 ? "charge_diagnostics_unavailable" : "absent_boundary",
                             step<total ? "charge_diagnostics_unavailable" : "absent_boundary"};
      std::vector<float> rates[2];
      if (available) for (int side=0;side<2;++side) {
        if ((side==0 && step==0) || (side==1 && step==total)) continue;
        std::vector<double> velocity(3*n_,0.0);
        const double sign=side==0 ? -1.0 : 1.0;
        for (int bead=0;bead<p_;++bead) for (int d=0;d<3;++d) for (int i=0;i<n_;++i) {
          const int j=d*n_+i, k=bead*3*n_+j;
          velocity[j]+=(v[k]+sign*0.5*dt_*f[k]/atom.cpu_mass[i])/p_;
        }
        bool finite=true;
        for (double value:velocity) if (!std::isfinite(value)) finite=false;
        if (!finite) { status[side]="nonfinite"; continue; }
        observer_->notify_velocity_update();
        observer_drift_v_.copy_from_host(velocity.data());
        observer_->compute_charge_rate_for_current_force_frame(box_for_observer_,atom.type,observer_q_,observer_drift_v_);
        rates[side].resize(n_);
        observer_->get_raw_charge_rate_reference().copy_to_host(rates[side].data());
        for (float value:rates[side]) if (!std::isfinite(value)) finite=false;
        if (!finite) { rates[side].clear(); status[side]="nonfinite"; }
        else status[side]="valid";
      }
      observer_->notify_velocity_update();
      const std::string drift_path=directory_+"/observer_drift_charge_"+suffix;
      rpmd_ja_file_io::ofstream drift_out(drift_path+".tmp"); drift_out << std::setprecision(17);
      drift_out << "# schema r164_observer_drift_charge_v1\n# step " << step << "\n# time_fs " << time_fs
                << "\n# observer_force_call " << step+1 << "\n# N " << n_
                << "\n# time_unit_conversion " << TIME_UNIT_CONVERSION
                << "\n# arrival_status " << status[0] << "\n# departure_status " << status[1]
                << "\n# columns side atom raw_charge_rate_fp32_e_per_natural_time\n# observer_projection_sha256 " << projection_hash << '\n';
      for (int side=0;side<2;++side) for (int i=0;i<static_cast<int>(rates[side].size());++i)
        drift_out << (side==0 ? "arrival " : "departure ") << i << ' ' << rates[side][i] << '\n';
      drift_out.close();
      if (!drift_out || rpmd_ja_file_io::atomic_commit_new_file(drift_path+".tmp",drift_path)!=0)
        PRINT_INPUT_ERROR("r164_rpmd failed committing observer drift charge.");
    }
  }
  if (checkpoint) {
    const std::string suffix=std::to_string(step)+".txt";
    const std::string metadata_hash=rpmd_ja_sample_provenance::sha256_file(directory_+"/metadata.txt");
    const std::string identity_hash=rpmd_ja_sample_provenance::sha256_file(directory_+"/source_identity.txt");
    const std::string frame_hash=rpmd_ja_sample_provenance::sha256_file(directory_+"/frame_"+suffix);
    const std::string image_hash=rpmd_ja_sample_provenance::sha256_file(directory_+"/force_images_"+suffix);
    const std::string scalar_hash=rpmd_ja_sample_provenance::sha256_text(scalar_line);
    auto prefix_digest=scalar_prefix_digest_;
    const std::string prefix_hash=prefix_digest.finish_hex();
    if (metadata_hash.size()!=64 || identity_hash.size()!=64 || frame_hash.size()!=64 || image_hash.size()!=64 || scalar_hash.size()!=64 || prefix_hash.size()!=64)
      PRINT_INPUT_ERROR("r164_rpmd failed hashing checkpoint inputs.");
    const std::string path=directory_+"/checkpoint_"+suffix;
    rpmd_ja_file_io::ofstream receipt(path+".tmp");
    receipt << std::setprecision(17) << "schema r164_checkpoint_v1\nstep " << step << "\ntime_fs " << time_fs
            << "\nmetadata_sha256 " << metadata_hash << "\nsource_identity_sha256 " << identity_hash
            << "\nframe_sha256 " << frame_hash << "\nforce_images_sha256 " << image_hash
            << "\nscalar_line_sha256 " << scalar_hash << "\nscalar_prefix_sha256 " << prefix_hash << "\ncache_policy fresh_rebuild_only\nphysical_event_history unavailable\n";
    receipt.close();
    if (!receipt || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
      PRINT_INPUT_ERROR("r164_rpmd failed committing checkpoint.");
  }
  if (trace_here) {
    const std::string suffix=std::to_string(step)+".txt";
    const std::string files[]={directory_+"/metadata.txt",directory_+"/source_identity.txt",directory_+"/frame_"+suffix,directory_+"/force_images_"+suffix,
      trace.prefix+"_mesh.out",trace.prefix+"_kspace.out",trace.prefix+"_atom.out",trace.prefix+"_assignment_atom.out",
      trace.prefix+"_assignment_stencil_atom"+std::to_string(trace_atom_==-2?0:trace_atom_)+".out",trace.prefix+"_pre_electro.out",trace.prefix+(trace_atom_==-2?"_all_meta.out":"_descriptor_neighbors.out"),trace.prefix+(trace_atom_==-2?"_all_edges.bin":"_local_edge_derivatives.out")};
    const char* names[]={"metadata","source_identity","frame","force_images","mesh","kspace","atom","assignment_atom","assignment_stencil","pre_electro","descriptor_neighbors","local_edge_derivatives"};
    const std::string path=directory_+"/pppm_trace_step_"+std::to_string(step)+"_receipt.txt";
    rpmd_ja_file_io::ifstream debug_atom_file(trace.prefix+"_atom.out");
    std::string line; unsigned long long evaluation_id=0; bool found=false;
    while (std::getline(debug_atom_file,line)) {
      const std::string key="# force_evaluation_id ";
      if (line.compare(0,key.size(),key)==0) {
        std::istringstream value(line.substr(key.size()));
        if (found || !(value >> evaluation_id) || evaluation_id==0) PRINT_INPUT_ERROR("r164_rpmd invalid trace force evaluation id.");
        found=true;
      }
    }
    if (!debug_atom_file.eof() || !found) PRINT_INPUT_ERROR("r164_rpmd missing trace force evaluation id.");
    rpmd_ja_file_io::ifstream neighbors(trace.prefix+(trace_atom_==-2?"_all_meta.out":"_descriptor_neighbors.out"));
    bool neighbor_id_found=false;
    while (std::getline(neighbors,line)) {
      const std::string key="# force_evaluation_id ";
      if (line.compare(0,key.size(),key)==0) {
        unsigned long long neighbor_id=0; std::istringstream value(line.substr(key.size()));
        if (neighbor_id_found || !(value >> neighbor_id) || neighbor_id!=evaluation_id)
          PRINT_INPUT_ERROR("r164_rpmd descriptor neighbor force id mismatch.");
        neighbor_id_found=true;
      }
    }
    if (!neighbors.eof() || !neighbor_id_found) PRINT_INPUT_ERROR("r164_rpmd missing descriptor neighbor force id.");
    if (trace_atom_!=-2) {
    rpmd_ja_file_io::ifstream derivatives(trace.prefix+"_local_edge_derivatives.out");
    bool derivative_id_found=false;
    while (std::getline(derivatives,line)) {
      const std::string key="# force_evaluation_id ";
      if (line.compare(0,key.size(),key)==0) {
        unsigned long long id=0; std::istringstream value(line.substr(key.size()));
        if (derivative_id_found || !(value >> id) || id!=evaluation_id)
          PRINT_INPUT_ERROR("r164_rpmd derivative force id mismatch.");
        derivative_id_found=true;
      }
    }
    if (!derivatives.eof() || !derivative_id_found) PRINT_INPUT_ERROR("r164_rpmd missing derivative force id.");
    }
    rpmd_ja_file_io::ofstream receipt(path+".tmp");
    receipt << (trace_atom_==-2?"schema r164_pppm_trace_all_v1\nstep ":"schema r164_pppm_trace_v3\nstep ") << step << "\nbead " << trace_bead_ << "\natom " << (trace_atom_==-2?"all":std::to_string(trace_atom_)) << "\nforce_evaluation_id " << evaluation_id << "\nneighbor_history " << (trace_atom_==-2?"all_centers_selected_bead_endpoint":"partial_target_descriptor_only") << "\nphysical_event unavailable\n";
    for (int i=0;i<12;++i) {
      const auto hash=rpmd_ja_sample_provenance::sha256_file(files[i]);
      if (hash.size()!=64) PRINT_INPUT_ERROR("r164_rpmd failed hashing PPPM trace.");
      const char* name=trace_atom_==-2 && i==10 ? "all_meta" : trace_atom_==-2 && i==11 ? "all_edges" : names[i];
      receipt << name << "_sha256 " << hash << "\n";
    }
    if (trace_atom_>=0) {
      const auto hash=rpmd_ja_sample_provenance::sha256_file(trace.prefix+"_assignment_join.out");
      if (hash.size()!=64) PRINT_INPUT_ERROR("r164_rpmd failed hashing PPPM assignment join.");
      receipt << "assignment_join_sha256 " << hash << "\n";
    }
    receipt << "scalar_line_sha256 " << rpmd_ja_sample_provenance::sha256_text(scalar_line) << "\n";
    receipt.close();
    if (!receipt || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
      PRINT_INPUT_ERROR("r164_rpmd failed committing PPPM trace receipt.");
  }
  if (step == total) write_complete(step);

}
