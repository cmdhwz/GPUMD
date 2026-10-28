#pragma once
#include "action.cuh"
#include "u9_oho_math.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <string>

class HAC;
class U9OHO : public Action {
public:
  explicit U9OHO(const std::vector<std::string>& tokens);
  ~U9OHO() override;
  void set_hac(HAC* hac) { hac_=hac; }
  void set_run_ordinal(unsigned ordinal) { ordinal_=ordinal; }
  void pre_run(int,double,Integrate&,std::vector<Group>&,Atom&,Box&,Force&) override;
  void end_of_step(int,int,int,int,double,double,Integrate&,Box&,std::vector<Group>&,GPU_Vector<double>&,Atom&,Force&) override;
  void post_run(Atom&,Box&,Integrate&,int,double,double) override;
private:
  void gather(Atom& atom, double data[18]);
  void advance(const double raw[18],const Box& box,bool initialize);
  std::string path_;
  u9_oho::Config config_{};
  HAC* hac_=nullptr;
  GPU_Vector<int> indices_;
  GPU_Vector<double> gather_;
  double last_[9]{}, continuous_[9]{}, initial_cell_[9]{};
  double dt_=0, last_time_=0;
  bool have_time_=false;
  unsigned ordinal_=0;
  FILE* output_=nullptr;
};
