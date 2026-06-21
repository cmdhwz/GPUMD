#pragma once
#include "property.cuh"
#include <vector>

class Box;

class Dump_PIMD_Restart : public Property
{
public:
  Dump_PIMD_Restart(const char** param, int num_param);
  void parse(const char** param, int num_param);

  virtual void preprocess(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void process(
    const int number_of_steps,
    int step,
    const int fixed_group,
    const int move_group,
    const double global_time,
    const double temperature,
    Integrate& integrate,
    Box& box,
    std::vector<Group>& group,
    GPU_Vector<double>& thermo,
    Atom& atom,
    Force& force);

  virtual void postprocess(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

private:
  bool dump_ = false;
  int dump_interval_ = 1;
  std::vector<double> cpu_position_;
  std::vector<double> cpu_velocity_;

  void output_line_2(FILE* fid, const Box& box, int number_of_beads, int bead_index);
};
