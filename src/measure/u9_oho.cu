#include "u9_oho.cuh"
#include "hac.cuh"
#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_utf8_file_io.h"
#include "force/force.cuh"
#include "force/nep_charge.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <sstream>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>

namespace {
__global__ void gather_oho(int n,const int* indices,const double* r,const double* v,double* out) {
  int i=threadIdx.x;
  if(i<9) { int atom=i/3, d=i%3, index=indices[atom]+d*n; out[i]=r[index]; out[9+i]=v[index]; }
}
void fail(const std::string& message) { PRINT_INPUT_ERROR(message.c_str()); }
void check_cell(const u9_oho::Config& c,const Box& box) {
  if(box.pbc_x!=1 || box.pbc_y!=1 || box.pbc_z!=1) fail("U9 diagnostic requires 3D PBC.");
  for(int i=0;i<9;i++) if(!std::isfinite(box.cpu_h[i]) || !u9_oho::near(c.cell[i],box.cpu_h[i])) fail("U9 diagnostic cell changed or differs from reference.");
}
bool compatible_output() {
  errno=0;
  FILE* probe=std::fopen("u9_oho_diagnostic.out","rb");
  if(!probe) return errno==ENOENT;
  if(std::fseek(probe,-1,SEEK_END)!=0 || std::fgetc(probe)!='\n') { std::fclose(probe); return false; }
  std::fclose(probe);
  std::ifstream in("u9_oho_diagnostic.out");
  if(!in) return false;
  return u9_oho::valid_output(in);
}
}
U9OHO::U9OHO(const std::vector<std::string>& tokens) {
  action_name="compute_u9_oho";
  if(tokens.size()!=3) fail("compute_u9_oho requires diagnostic <config> or candidate <config>.");
  if(tokens[1]=="candidate") fail("U9 candidate BLOCKED: material Cartesian operator and physical acceptance are unavailable.");
  if(tokens[1]!="diagnostic") fail("compute_u9_oho mode must be diagnostic or candidate.");
  path_=tokens[2];
}
U9OHO::~U9OHO() { if(output_) std::fclose(output_); }
void U9OHO::gather(Atom& atom,double data[18]) {
  gather_oho<<<1,9>>>(atom.number_of_atoms,indices_.data(),atom.position_per_atom.data(),atom.velocity_per_atom.data(),gather_.data());
  GPU_CHECK_KERNEL
  gather_.copy_to_host(data);
  for(int i=0;i<18;i++) if(!std::isfinite(data[i])) fail("U9 diagnostic encountered nonfinite centroid data.");
}
void U9OHO::advance(const double raw[18],const Box& box,bool initialize) {
  try { u9_oho::advance(config_,box.cpu_h,raw,last_,continuous_,initialize); }
  catch(const std::exception& e) { fail(e.what()); }
}
void U9OHO::pre_run(int steps,double dt,Integrate& integrate,std::vector<Group>&,Atom& atom,Box& box,Force& force) {
  rpmd_ja_file_io::ifstream input;
  try { input.open(path_); } catch(const std::exception& e) { fail(e.what()); }
  if(!input) fail("U9 diagnostic cannot open configuration file.");
  input.seekg(0,std::ios::end);
  const std::streamoff size=input.tellg();
  if(size<0 || size>65536) fail("U9 diagnostic configuration exceeds 65536 bytes or cannot be measured.");
  input.seekg(0,std::ios::beg);
  if(!input) fail("U9 diagnostic cannot seek configuration file.");
  try { config_=u9_oho::read_config(input); } catch(const std::exception& e) { fail(e.what()); }
  if(!hac_ || !hac_->u9_compatible() || hac_->sample_interval>steps) fail("U9 diagnostic requires one immediate, unsplit centroid HAC with samples.");
  if(integrate.get_type()!=EnsembleType::RPMD && integrate.get_type()!=EnsembleType::PIMD) fail("U9 diagnostic requires RPMD or PIMD.");
  if(atom.number_of_beads<2 || atom.position_beads.size()!=static_cast<size_t>(atom.number_of_beads) || atom.velocity_beads.size()!=static_cast<size_t>(atom.number_of_beads) || integrate.get_deform_x()!=0 || integrate.get_deform_y()!=0 || integrate.get_deform_z()!=0 || integrate.get_deform_xy()!=0 || integrate.get_deform_xz()!=0 || integrate.get_deform_yz()!=0 || integrate.get_num_target_pressure_components()>0 || integrate.get_use_scr_barostat() || integrate.get_use_eco_pimd()) fail("U9 diagnostic requires fixed-cell primitive ring polymer.");
  if(!std::isfinite(dt)||dt<=0 || !u9_oho::near(integrate.get_temperature1(),config_.temperature) || !u9_oho::near(integrate.get_temperature2(),config_.temperature)) fail("U9 diagnostic requires fixed positive time step and matching temperature.");
  std::uint64_t model_fnv=0, config_fnv=0;
  try {
    if(!hac_->u9_qnep_identity(force,atom,model_fnv,config_fnv)) fail("U9 diagnostic requires a verified charge1/2 qNEP PPPM stage without DFTD3 or Eco-PIMD.");
    if(model_fnv!=config_.model_fnv || config_fnv!=config_.config_fnv) fail("U9 diagnostic model or qNEP configuration fingerprint mismatch.");
  } catch(const std::exception& e) { fail(e.what()); }
  if(atom.number_of_atoms<=0 || atom.number_of_atoms>std::numeric_limits<int>::max()/3) fail("U9 diagnostic atom count is invalid.");
  if(atom.position_per_atom.size()!=static_cast<size_t>(3)*atom.number_of_atoms || atom.velocity_per_atom.size()!=static_cast<size_t>(3)*atom.number_of_atoms) fail("U9 diagnostic centroid arrays have invalid lengths.");
  if(atom.cpu_type.size()!=static_cast<size_t>(atom.number_of_atoms) || atom.cpu_mass.size()!=static_cast<size_t>(atom.number_of_atoms)) fail("U9 diagnostic requires atom type and mass metadata.");
  if(atom.cpu_atom_symbol.size()!=static_cast<size_t>(atom.number_of_atoms)) fail("U9 diagnostic requires O-H-O element metadata.");
  for(int i=0;i<3;i++) {
    int index=config_.indices[i];
    if(index>=atom.number_of_atoms || atom.cpu_type[index]!=config_.types[i] || !u9_oho::near(atom.cpu_mass[index],config_.masses[i])) fail("U9 diagnostic atom index, type or mass mismatch.");
    if(atom.cpu_atom_symbol[index]!=(i==1 ? "H" : "O")) fail("U9 diagnostic requires O-H-O element identities; D uses symbol H with isotope mass.");
  }
  check_cell(config_,box); box.get_inverse();
  for(int i=0;i<9;i++) initial_cell_[i]=box.cpu_h[i];
  dt_=dt; have_time_=false;
  indices_.resize(3); indices_.copy_from_host(config_.indices.data()); gather_.resize(18);
  double data[18]; gather(atom,data); advance(data,box,true);
  std::uint64_t reference_file_fnv=0;
  try { reference_file_fnv=rpmd_ja_model_fingerprint(path_); } catch(const std::exception& e) { fail(e.what()); }
  if(!compatible_output()) fail("U9 diagnostic existing output schema is incompatible or unreadable.");
  std::ifstream existing("u9_oho_diagnostic.out",std::ios::binary);
  const bool new_file=!existing.good(); existing.close();
  output_=std::fopen("u9_oho_diagnostic.out","a"); if(!output_) fail("U9 diagnostic cannot open output for append.");
  if(new_file) std::fprintf(output_,"# U9_OHO_DIAGNOSTIC 1\n");
  const auto stamp=std::chrono::system_clock::now().time_since_epoch().count();
  std::fprintf(output_,"# segment %lld_%u reference %s source %s reference_file_fnv %llu config_fnv %llu model_fnv %llu temperature %.17g dt_natural %.17g sample_interval %d beads %d mode %s\n",static_cast<long long>(stamp),ordinal_,config_.reference.c_str(),config_.source.c_str(),static_cast<unsigned long long>(reference_file_fnv),static_cast<unsigned long long>(config_.config_fnv),static_cast<unsigned long long>(config_.model_fnv),config_.temperature,dt,hac_->sample_interval,atom.number_of_beads,integrate.get_type()==EnsembleType::RPMD?"RPMD":"PIMD_diagnostic");
  std::fprintf(output_,"%s\n",u9_oho::columns());
}
void U9OHO::end_of_step(int,int step,int,int,double global_time,double,Integrate&,Box& box,std::vector<Group>&,GPU_Vector<double>&,Atom& atom,Force&) {
  check_cell(config_,box);
  for(int i=0;i<9;i++) if(box.cpu_h[i]!=initial_cell_[i]) fail("U9 diagnostic cell changed during the run.");
  if(!std::isfinite(global_time) || (have_time_ && std::fabs(global_time-last_time_-dt_)>1e-10*std::max(1.,std::fabs(global_time)))) fail("U9 diagnostic time clock is not fixed.");
  last_time_=global_time; have_time_=true;
  double data[18]; gather(atom,data); advance(data,box,false);
  if((step+1)%hac_->sample_interval) return;
  double jc[3]; if(!hac_->centroid_force_ready(step+1) || !hac_->get_current_for_step(step+1,jc)) fail("U9 diagnostic HAC current is not ready at matching step.");
  const auto projection=u9_oho::project(config_,continuous_,data+9);
  const double *z=projection.z,*v=projection.velocity,*p=projection.momentum,*physical=projection.physical;
  const double posres=projection.position_residual,velres=projection.velocity_residual;
  auto u=u9_oho::evaluate(config_,z[0],z[1]);
  for(double x:{z[0],z[1],v[0],v[1],p[0],p[1],u.u,u.g[0],u.g[1],u.h[0],u.h[1],u.h[3],physical[0],physical[1],posres,velres,jc[0],jc[1],jc[2],global_time}) if(!std::isfinite(x)) fail("U9 diagnostic numerical overflow.");
  std::fprintf(output_,"%d %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %.16e %d %.16e %.16e %.16e\n",step+1,(step+1)*dt_*TIME_UNIT_CONVERSION/1000,global_time*TIME_UNIT_CONVERSION/1000,z[0],z[1],v[0],v[1],p[0],p[1],u.u,u.g[0],u.g[1],u.h[0],u.h[1],u.h[3],physical[0],physical[1],std::sqrt(posres),std::sqrt(velres),posres<=1e-16 && velres<=1e-16 ? 1 : 0,jc[0],jc[1],jc[2]);
  if(std::ferror(output_)) fail("U9 diagnostic write failed.");
}
void U9OHO::post_run(Atom&,Box&,Integrate&,int,double,double) {
  if(output_) { std::fprintf(output_,"# segment_end\n"); if(std::fclose(output_)!=0) fail("U9 diagnostic output close failed."); output_=nullptr; }
}
