#include "../../src/measure/u9_oho_math.cuh"
#include <cassert>
#include <sstream>
#include <fstream>
#include <string>
#include <utility>
#include <cmath>

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream f(argv[1]); std::stringstream buffer; buffer << f.rdbuf();
  const std::string text = buffer.str();
  std::istringstream input(text); auto cfg = u9_oho::read_config(input);
  auto value = u9_oho::evaluate(cfg, 1., 0.);
  std::string shifted=text;
  shifted.replace(shifted.find("coefficients 1 0 -2"),std::string("coefficients 1 0 -2").size(),"coefficients 1e20 0 -2");
  std::istringstream shifted_input(shifted); u9_oho::read_config(shifted_input);
  shifted.replace(shifted.find("left -1 0"),std::string("left -1 0").size(),"left -0.8 0");
  std::istringstream nonstationary(shifted); bool bad_center=false;
  try { u9_oho::read_config(nonstationary); } catch(const std::exception&) { bad_center=true; } assert(bad_center);
  assert(std::abs(value.u) < 1e-12 && std::abs(value.g[0]) < 1e-12);
  assert(std::abs(value.h[0]-8.) < 1e-12);
  double displaced[9], velocity[9];
  for(int i=0;i<9;i++) { displaced[i]=cfg.r0[i]+cfg.A[2*i]*.2+cfg.A[2*i+1]*-.3; velocity[i]=cfg.A[2*i]*.4+cfg.A[2*i+1]*.1; }
  auto projection=u9_oho::project(cfg,displaced,velocity);
  assert(std::fabs(projection.z[0]-.2)<1e-12 && std::fabs(projection.z[1]+.3)<1e-12);
  assert(std::fabs(projection.velocity[0]-.4)<1e-12 && std::fabs(projection.velocity[1]-.1)<1e-12);
  assert(projection.position_residual<1e-24 && projection.velocity_residual<1e-24);
  auto unequal=cfg; unequal.masses[2]=17.;
  double G[4]={}; for(int i=0;i<9;i++) for(int a=0;a<2;a++) for(int b=0;b<2;b++) G[2*a+b]+=unequal.masses[i/3]*unequal.A[2*i+a]*unequal.A[2*i+b];
  const double determinant=G[0]*G[3]-G[1]*G[2];
  for(int i=0;i<4;i++) unequal.G[i]=G[i];
  unequal.C={{G[3]/determinant,-G[1]/determinant,-G[2]/determinant,G[0]/determinant}};
  assert(std::fabs(G[1])>1e-6);
  auto unequal_projection=u9_oho::project(unequal,displaced,velocity);
  double next[9]; for(int i=0;i<9;i++) next[i]=displaced[i]+1e-6*velocity[i];
  auto next_projection=u9_oho::project(unequal,next,velocity);
  for(int a=0;a<2;a++) assert(std::fabs((next_projection.z[a]-unequal_projection.z[a])/1e-6-unequal_projection.velocity[a])<1e-9);
  assert(!u9_oho::near(INFINITY, 1.));
  double h[18]={10,0,0,0,10,0,0,0,10,.1,0,0,0,.1,0,0,0,.1};
  double raw[9]={9,0,0,0,0,0,1,0,0},last[9]={},continuous[9]={};
  u9_oho::advance(cfg,h,raw,last,continuous,true);
  assert(std::abs(continuous[0]+1.)<1e-12 && std::abs(continuous[6]-1.)<1e-12);
  raw[0]=.1; u9_oho::advance(cfg,h,raw,last,continuous,false);
  assert(std::abs(continuous[0]-.1)<1e-12 && std::abs(continuous[6]-1.)<1e-12);
  raw[0]=5.; bool branch=false; try { u9_oho::advance(cfg,h,raw,last,continuous,false); } catch(const std::exception&) { branch=true; } assert(branch);
  std::string head="# U9_OHO_DIAGNOSTIC 1\n# segment one\n"+std::string(u9_oho::columns())+"\n";
  std::string row; for(int i=0;i<23;i++) row+="0 "; row+="\n";
  std::istringstream valid(head+row+"# segment_end\n"); assert(u9_oho::valid_output(valid));
  std::istringstream shortrow(head+"0 1\n# segment_end\n"); assert(!u9_oho::valid_output(shortrow));
  std::istringstream truncated(head+row); assert(!u9_oho::valid_output(truncated));
  std::istringstream header("# U9_OHO_DIAGNOSTIC 1\n"); assert(!u9_oho::valid_output(header));
  const std::pair<std::string,std::string> bad[] = {
    {"U9_OHO 1", "U9_OHO 2"}, {"coefficients 1 0 -2", "coefficients nan 0 -2"},
    {"indices 0 1 2", "indices 0 1 1"}, {"masses 16 1 16", "masses 16 -1 16"},
    {"G 0.9696969696969697 0 0 8", "G -1 0 0 8"},
    {"temperature 300", "temperature -1"}, {"pbc 1 1 1", "pbc 1 0 1"},
    {"left -1 0", "left 0 0"}, {"status diagnostic", "status candidate"}, {"d0 2", "d0 -2"},
    {"axis 1 0 0", "axis 0 0 0"}, {"images -1 0 0 0 0 0 0 0 0", "images 0 0 0 0 0 0 0 0 0"}, {"indices 0 1 2", "indices -1 1 2"},
    {"C 1.03125 0 0 0.125", "C 1 0 0 0.125"},
    {"C 1.03125 0 0 0.125", "C 1e308 1e308 1e308 1e308"},
    {"model_fnv 1", "model_fnv -1"}, {"config_fnv 1", "config_fnv 0"},
    {"site_gauge constrained_v1", "site_gauge unknown"},
    {"reference test_fixed", "unknown test_fixed"}, {"source synthetic_fixture", "source " + std::string(129,'x')}, {"end", "missing_end"}
  };
  for (const auto& item : bad) {
    std::string changed = text; auto i = changed.find(item.first); assert(i != std::string::npos);
    changed.replace(i,item.first.size(),item.second); std::istringstream stream(changed);
    bool rejected=false; try { u9_oho::read_config(stream); } catch (const std::exception&) { rejected=true; }
    assert(rejected);
  }
  std::istringstream extra(text+" extra"); bool rejected=false;
  try { u9_oho::read_config(extra); } catch (const std::exception&) { rejected=true; } assert(rejected);
}
