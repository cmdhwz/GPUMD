#pragma once
#include "force/nep_charge.cuh"
#include "model/box.cuh"
#include <memory>
#include <string>
#include <vector>
class Atom;
class Box;
class Force;
class R164_RPMD {
public:
  bool enabled() const { return enabled_; }
  void parse(const std::vector<std::string>& tokens);
  void reset() { enabled_ = false; observer_requested_ = false; checkpoint_stride_ = 0; observer_.reset(); }
  void begin(Atom&, const Box&, double temperature, double dt, double start_time);
  void configure_observer(Force&, Atom&, const std::string& model_xyz_sha256, const std::string& restart_sha256);
  void capture_drift(Atom&);
  void check_force_positions(Atom&);
  void output(int step, int total, Atom&);
private:
  bool enabled_ = false;
  int stride_ = 0, checkpoint_stride_ = 0, n_ = 0, p_ = 0;
  double temperature_ = 0, dt_ = 0, lengths_[3] = {};
  std::string directory_;
  bool observer_requested_ = false;
  std::unique_ptr<NEP_Charge> observer_;
  Box box_for_observer_;
  GPU_Vector<double> observer_q_, observer_u_, observer_f_, observer_virial_, observer_v_, observer_site_rate_, observer_gradient_, observer_local_, observer_local_image_;
  std::vector<double> q_, offset_;
  std::vector<long long> force_images_, previous_force_images_;
  void read_positions(Atom&, std::vector<double>&);
  void check_branch(const std::vector<double>&) const;
};
