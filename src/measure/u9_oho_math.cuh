#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <istream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

namespace u9_oho {
struct Config {
  std::array<double,9> coefficients, cell, r0;
  std::array<double,18> A;
  std::array<double,4> G, C;
  std::array<double,3> masses, axis;
  std::array<double,2> z0, left, right;
  std::array<int,3> indices, types, pbc;
  std::array<int,9> images;
  double d0, temperature;
  std::uint64_t model_fnv, config_fnv;
  std::string source, reference;
};
struct Value { double u; std::array<double,2> g; std::array<double,4> h; };
inline Value evaluate(const Config& c, double q, double Q) {
  const auto& a=c.coefficients;
  Value v;
  v.u=a[0]+q*(a[1]+q*(a[2]+q*(a[3]+q*a[4])))+0.5*a[5]*Q*Q+(a[6]+a[7]*q+a[8]*q*q)*Q;
  v.g={{a[1]+2*a[2]*q+3*a[3]*q*q+4*a[4]*q*q*q+(a[7]+2*a[8]*q)*Q,
        a[5]*Q+a[6]+a[7]*q+a[8]*q*q}};
  v.h={{2*a[2]+6*a[3]*q+12*a[4]*q*q+2*a[8]*Q,a[7]+2*a[8]*q,a[7]+2*a[8]*q,a[5]}};
  return v;
}
inline bool near(double a,double b) { return std::isfinite(a) && std::isfinite(b) && std::fabs(a-b)<=1e-9*std::max(1.,std::max(std::fabs(a),std::fabs(b))); }
inline void need(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
inline void key(std::istream& in,const char* name) { std::string s; need(bool(in>>s)&&s==name,"U9 config: missing, duplicate, reordered, or unknown field"); }
template<class T, size_t N> inline void values(std::istream& in,std::array<T,N>& a) {
  for(auto& x:a) need(bool(in>>x),"U9 config: invalid or missing value");
}
template<size_t N> inline void finite(const std::array<double,N>& a) {
  for(double x:a) need(std::isfinite(x),"U9 config: nonfinite value");
}
inline Config read_config(std::istream& in) {
  Config c{}; std::string s; int version;
  key(in,"U9_OHO"); need(bool(in>>version)&&version==1,"U9 config: unsupported version");
  key(in,"status"); need(bool(in>>s)&&s=="diagnostic","U9 config: diagnostic status required; candidate is blocked");
  key(in,"units"); need(bool(in>>s)&&s=="angstrom_eV_amu_K_natural_time","U9 config: units mismatch");
  key(in,"coefficients"); values(in,c.coefficients);
  key(in,"d0"); need(bool(in>>c.d0),"U9 config: d0 missing");
  key(in,"indices"); values(in,c.indices); key(in,"types"); values(in,c.types);
  key(in,"masses"); values(in,c.masses); key(in,"cell"); values(in,c.cell);
  key(in,"pbc"); values(in,c.pbc); key(in,"r0"); values(in,c.r0);
  key(in,"images"); values(in,c.images); key(in,"axis"); values(in,c.axis);
  key(in,"z0"); values(in,c.z0); key(in,"A"); values(in,c.A);
  key(in,"G"); values(in,c.G); key(in,"C"); values(in,c.C);
  key(in,"temperature"); need(bool(in>>c.temperature),"U9 config: temperature missing");
  auto read_hash=[&in](std::uint64_t& hash) {
    std::string token; need(bool(in>>token)&&!token.empty(),"U9 config: fingerprint missing");
    for(char ch:token) need(ch>='0'&&ch<='9',"U9 config: fingerprint must be unsigned decimal");
    std::size_t used=0;
    try { hash=std::stoull(token,&used,10); } catch(const std::exception&) { throw std::runtime_error("U9 config: fingerprint overflow"); }
    need(used==token.size()&&hash!=0,"U9 config: invalid fingerprint");
  };
  key(in,"model_fnv"); read_hash(c.model_fnv);
  key(in,"config_fnv"); read_hash(c.config_fnv);
  key(in,"source"); need(bool(in>>c.source),"U9 config: source missing");
  key(in,"reference"); need(bool(in>>c.reference),"U9 config: reference missing");
  need(c.source.size()<=128 && c.reference.size()<=128,"U9 config: identity exceeds 128 bytes");
  key(in,"site_gauge"); need(bool(in>>s)&&s=="constrained_v1","U9 config: unsupported site gauge");
  key(in,"left"); values(in,c.left); key(in,"right"); values(in,c.right); key(in,"end");
  need(!(in>>s) && !in.bad(),"U9 config: trailing tokens or read error");
  finite(c.coefficients); finite(c.cell); finite(c.r0); finite(c.A); finite(c.G); finite(c.C);
  finite(c.masses); finite(c.axis); finite(c.z0); finite(c.left); finite(c.right);
  const auto& coeff=c.coefficients;
  need(coeff[5]>0,"U9 config: K must be positive");
  double effective4=coeff[4]-coeff[8]*coeff[8]/(2*coeff[5]);
  double effective3=coeff[3]-coeff[7]*coeff[8]/coeff[5];
  double effective2=coeff[2]-(coeff[7]*coeff[7]+2*coeff[6]*coeff[8])/(2*coeff[5]);
  need(std::isfinite(effective4)&&std::isfinite(effective3)&&std::isfinite(effective2),"U9 config: boundedness overflow");
  need(effective4>0 || (effective4==0 && effective3==0 && effective2>0),"U9 config: potential not proven confining");
  need(std::isfinite(c.d0)&&c.d0>0&&std::isfinite(c.temperature)&&c.temperature>0,"U9 config: invalid d0 or temperature");
  for(int image:c.images) need(image>=-1000000 && image<=1000000,"U9 config: reference image out of range");
  for(int i=0;i<3;i++) {
    need(c.indices[i]>=0&&c.types[i]>=0&&c.masses[i]>0&&c.pbc[i]==1,"U9 config: invalid index, type, mass or PBC");
    for(int j=0;j<i;j++) need(c.indices[i]!=c.indices[j],"U9 config: duplicate indices");
  }
  need(near(c.axis[0]*c.axis[0]+c.axis[1]*c.axis[1]+c.axis[2]*c.axis[2],1),"U9 config: axis is not unit length");
  double det=c.cell[0]*(c.cell[4]*c.cell[8]-c.cell[5]*c.cell[7])-c.cell[1]*(c.cell[3]*c.cell[8]-c.cell[5]*c.cell[6])+c.cell[2]*(c.cell[3]*c.cell[7]-c.cell[4]*c.cell[6]);
  need(std::isfinite(det)&&det>1e-12,"U9 config: invalid cell");
  double inv[9]={
    (c.cell[4]*c.cell[8]-c.cell[5]*c.cell[7])/det,(c.cell[2]*c.cell[7]-c.cell[1]*c.cell[8])/det,(c.cell[1]*c.cell[5]-c.cell[2]*c.cell[4])/det,
    (c.cell[5]*c.cell[6]-c.cell[3]*c.cell[8])/det,(c.cell[0]*c.cell[8]-c.cell[2]*c.cell[6])/det,(c.cell[2]*c.cell[3]-c.cell[0]*c.cell[5])/det,
    (c.cell[3]*c.cell[7]-c.cell[4]*c.cell[6])/det,(c.cell[1]*c.cell[6]-c.cell[0]*c.cell[7])/det,(c.cell[0]*c.cell[4]-c.cell[1]*c.cell[3])/det};
  for(int atom=0;atom<3;atom++) for(int d=0;d<3;d++) {
    double wrapped=-c.images[3*atom+d];
    for(int k=0;k<3;k++) wrapped+=inv[3*d+k]*c.r0[3*atom+k];
    need(std::isfinite(wrapped) && wrapped>=-1e-10 && wrapped<1.0-1e-10,"U9 config: image does not identify the canonical wrapped R0");
  }
  double G[4]={};
  for(int i=0;i<9;i++) for(int a=0;a<2;a++) for(int b=0;b<2;b++) G[2*a+b]+=c.masses[i/3]*c.A[2*i+a]*c.A[2*i+b];
  for(int i=0;i<4;i++) need(near(G[i],c.G[i]),"U9 config: G does not equal A transpose M A");
  double gd=G[0]*G[3]-G[1]*G[2];
  need(std::isfinite(gd) && G[0]>0&&G[3]>0&&gd>1e-12*G[0]*G[3],"U9 config: G is singular or indefinite");
  for(int a=0;a<2;a++) for(int b=0;b<2;b++) {
    double x=0; for(int j=0;j<2;j++) x+=G[2*a+j]*c.C[2*j+b];
    need(near(x,a==b?1.:0.),"U9 config: C is not inverse G");
  }
  for(int a=0;a<2;a++) for(int b=0;b<2;b++) {
    double x=0; for(int d=0;d<3;d++) {
      double l=c.A[2*d+b],h=c.A[2*(3+d)+b],r=c.A[2*(6+d)+b];
      x+=c.axis[d]*(a==0 ? h-.5*(l+r) : r-l);
    }
    need(near(x,a==b?1.:0.),"U9 config: physical coordinate Jacobian A is not identity");
  }
  double z[2]={0,-c.d0};
  for(int d=0;d<3;d++) { z[0]+=c.axis[d]*(c.r0[3+d]-.5*(c.r0[d]+c.r0[6+d])); z[1]+=c.axis[d]*(c.r0[6+d]-c.r0[d]); }
  need(near(z[0],c.z0[0])&&near(z[1],c.z0[1]),"U9 config: R0 and z0 disagree");
  for(const auto& point:{c.left,c.right}) {
    auto v=evaluate(c,point[0],point[1]);
    need(std::isfinite(v.u)&&std::isfinite(v.g[0])&&std::isfinite(v.g[1])&&std::isfinite(v.h[0])&&std::isfinite(v.h[1])&&std::isfinite(v.h[3]),"U9 config: reference overflow");
    const double q=point[0], Q=point[1];
    const double sq=std::max(1.0,std::fabs(coeff[1])+std::fabs(2*coeff[2]*q)+std::fabs(3*coeff[3]*q*q)+std::fabs(4*coeff[4]*q*q*q)+std::fabs(coeff[7]*Q)+std::fabs(2*coeff[8]*q*Q));
    const double sQ=std::max(1.0,std::fabs(coeff[5]*Q)+std::fabs(coeff[6])+std::fabs(coeff[7]*q)+std::fabs(coeff[8]*q*q));
    const double determinant=v.h[0]*v.h[3]-v.h[1]*v.h[2];
    need(std::isfinite(sq)&&std::isfinite(sQ)&&std::isfinite(determinant)&&std::fabs(v.g[0])<1e-8*sq&&std::fabs(v.g[1])<1e-8*sQ&&v.h[0]>0&&v.h[3]>0&&determinant>1e-12*v.h[0]*v.h[3],"U9 config: reference is not a stable stationary point");
  }
  need(c.left[0]<c.right[0],"U9 config: left/right reference order invalid");
  return c;
}
struct Projection { double z[2], velocity[2], momentum[2], physical[2], position_residual, velocity_residual; };
inline Projection project(const Config& c,const double r[9],const double velocity[9]) {
  Projection out{}; out.z[0]=c.z0[0]; out.z[1]=c.z0[1]; out.physical[1]=-c.d0;
  double b[2]={},w[2]={};
  for(int i=0;i<9;i++) for(int a=0;a<2;a++) { b[a]+=c.A[2*i+a]*c.masses[i/3]*(r[i]-c.r0[i]); w[a]+=c.A[2*i+a]*c.masses[i/3]*velocity[i]; }
  for(int a=0;a<2;a++) for(int j=0;j<2;j++) { out.z[a]+=c.C[2*a+j]*b[j]; out.velocity[a]+=c.C[2*a+j]*w[j]; }
  for(int a=0;a<2;a++) for(int j=0;j<2;j++) out.momentum[a]+=c.G[2*a+j]*out.velocity[j];
  for(int i=0;i<9;i++) { double dr=r[i]-c.r0[i],dv=velocity[i]; for(int a=0;a<2;a++) { dr-=c.A[2*i+a]*(out.z[a]-c.z0[a]); dv-=c.A[2*i+a]*out.velocity[a]; } out.position_residual+=dr*dr; out.velocity_residual+=dv*dv; }
  for(int d=0;d<3;d++) { out.physical[0]+=c.axis[d]*(r[3+d]-.5*(r[d]+r[6+d])); out.physical[1]+=c.axis[d]*(r[6+d]-r[d]); }
  return out;
}
inline void advance(const Config& c,const double h[18],const double raw[9],double last[9],double continuous[9],bool initialize) {
  for(int atom=0;atom<3;atom++) {
    double delta[3],fraction[3];
    for(int d=0;d<3;d++) {
      double wrapped=c.r0[3*atom+d];
      for(int k=0;k<3;k++) wrapped-=h[3*d+k]*c.images[3*atom+k];
      delta[d]=raw[3*atom+d]-(initialize?wrapped:last[3*atom+d]);
    }
    for(int d=0;d<3;d++) {
      fraction[d]=0;
      for(int k=0;k<3;k++) fraction[d]+=h[9+3*d+k]*delta[k];
      fraction[d]-=std::nearbyint(fraction[d]);
      need(std::isfinite(fraction[d])&&std::fabs(fraction[d])<.45,"U9 diagnostic branch step exceeds 0.45 fractional cell");
    }
    for(int d=0;d<3;d++) {
      double step=0; for(int k=0;k<3;k++) step+=h[3*d+k]*fraction[k];
      continuous[3*atom+d]=(initialize?c.r0[3*atom+d]:continuous[3*atom+d])+step;
      last[3*atom+d]=raw[3*atom+d];
    }
    for(int d=0;d<3;d++) {
      double x=0; for(int k=0;k<3;k++) x+=h[9+3*d+k]*(continuous[3*atom+k]-c.r0[3*atom+k]);
      need(std::isfinite(x)&&std::fabs(x)<.45,"U9 diagnostic displacement exceeds fixed reference branch");
    }
  }
}
inline const char* columns() {
  return "# columns step local_time_ps global_time_ps q Q dq dQ p_q p_Q U U_q U_Q H_qq H_qQ H_QQ q_physical Q_physical position_residual_A velocity_residual_A_per_natural_time on_slice Jc_x Jc_y Jc_z";
}
inline bool valid_output(std::istream& in) {
  std::string line; if(!std::getline(in,line)||line!="# U9_OHO_DIAGNOSTIC 1") return false;
  bool segment=false,have_columns=false,have_segment=false;
  while(std::getline(in,line)) {
    if(line.compare(0,10,"# segment ")==0) { if(segment) return false; segment=true; have_columns=false; have_segment=true; }
    else if(line.compare(0,10,"# columns ")==0) { if(!segment||have_columns||line!=columns()) return false; have_columns=true; }
    else if(line=="# segment_end") { if(!segment||!have_columns) return false; segment=false; }
    else {
      if(!segment||!have_columns) return false;
      std::istringstream row(line); double x; int count=0;
      while(row>>x) { if(!std::isfinite(x)) return false; ++count; }
      if(!row.eof()||count!=23) return false;
    }
  }
  return in.eof()&&!in.bad()&&!segment&&have_segment;
}

}
