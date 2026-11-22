#pragma once
#include "rpmd_ja_utf8_file_io.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace r164_hac_ja {
using Complex=std::complex<double>;
inline void fft(std::vector<Complex>& a, bool inverse) {
  const size_t n=a.size();
  for (size_t i=1,j=0;i<n;++i) {
    size_t bit=n>>1;
    for (;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if (i<j) std::swap(a[i],a[j]);
  }
  const double pi=std::acos(-1.0);
  for (size_t len=2;len<=n;len<<=1) {
    const Complex root=std::polar(1.0,(inverse ? 2 : -2)*pi/len);
    for (size_t i=0;i<n;i+=len) {
      Complex w(1,0);
      for (size_t j=0;j<len/2;++j) {
        Complex u=a[i+j],v=a[i+j+len/2]*w;
        a[i+j]=u+v; a[i+j+len/2]=u-v; w*=root;
      }
    }
    if (len==n) break;
  }
  if (inverse) for (auto& x:a) x/=static_cast<double>(n);
}
inline void commit(rpmd_ja_file_io::ofstream& out, const std::string& path) {
  out.close();
  if (!out || rpmd_ja_file_io::atomic_commit_new_file(path+".tmp",path)!=0)
    throw std::runtime_error("compute_hac_ja failed committing output");
}
inline void write(const std::string& dir, const std::vector<double>& v,
                  const std::vector<unsigned char>& mask, int components, int nc,
                  int interval, double dt_fs, double factor, int stride) {
  const size_t n=mask.size();
  size_t size=1;
  if (n>(std::numeric_limits<size_t>::max()/2)) throw std::runtime_error("compute_hac_ja sample count overflow");
  while (size<2*n) {
    if (size>std::numeric_limits<size_t>::max()/2) throw std::runtime_error("compute_hac_ja FFT size overflow");
    size*=2;
  }
  size_t count=0; for (auto m:mask) count+=m!=0;
  std::vector<Complex> work(size);
  for (size_t i=0;i<n;++i) work[i]=mask[i] ? 1.0 : 0.0;
  fft(work,false); for (auto& x:work) x=std::norm(x); fft(work,true);
  std::vector<size_t> pairs(nc);
  for (int l=0;l<nc;++l) pairs[l]=static_cast<size_t>(std::max(0.0,std::round(work[l].real())));
  const char* names[]={"ja","hardy","centroid"};
  const double nan=std::numeric_limits<double>::quiet_NaN();
  for (int kind=0;kind<components/3;++kind) {
    std::vector<std::vector<Complex>> spectra(3,std::vector<Complex>(size));
    for (int d=0;d<3;++d) {
      for (size_t i=0;i<n;++i) if (mask[i]) spectra[d][i]=v[i*components+3*kind+d];
      fft(spectra[d],false);
    }
    std::vector<double> corr(static_cast<size_t>(nc)*9,nan);
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
      for (size_t k=0;k<size;++k) work[k]=std::conj(spectra[a][k])*spectra[b][k];
      fft(work,true);
      for (int l=0;l<nc;++l) if (pairs[l]) corr[static_cast<size_t>(l)*9+3*a+b]=work[l].real()/pairs[l];
    }
    const std::string path=dir+"/hac_"+names[kind]+".out";
    rpmd_ja_file_io::ofstream out(path+".tmp"); out<<std::setprecision(17);
    out<<"# sample_stride "<<stride<<" samples "<<n<<" valid_samples "<<count<<" Nc "<<nc<<" output_interval "<<interval<<"\n";
    out<<"# lag_ps pairs C_xx C_xy C_xz C_yx C_yy C_yz C_zx C_zy C_zz K_xx K_xy K_xz K_yx K_yy K_yz K_zx K_zy K_zz\n"
       <<"# C uses J_a(t) J_b(t+lag), (eV Angstrom/fs)^2; K W/(m K); common mask, mean_subtraction none, trapezoid all lags; complete bins, pairs is the mean valid pair count in each bin; UNVERIFIED\n";
    double integral[9]={}, corr_sum[9]={}, integral_sum[9]={}, pair_sum=0;
    bool supported=pairs[0]>0;
    for (int l=0;l<nc;++l) {
      if (!pairs[l]) supported=false;
      if (l && supported) for (int j=0;j<9;++j)
        integral[j]+=0.5*dt_fs*factor*(corr[static_cast<size_t>(l-1)*9+j]+corr[static_cast<size_t>(l)*9+j]);
      pair_sum+=pairs[l];
      for (int j=0;j<9;++j) {
        corr_sum[j]+=corr[static_cast<size_t>(l)*9+j];
        integral_sum[j]+=supported ? integral[j] : nan;
      }
      if ((l+1)%interval==0) {
        out<<(l-0.5*(interval-1))*dt_fs/1000<<' '<<pair_sum/interval;
        for (int j=0;j<9;++j) out<<' '<<corr_sum[j]/interval;
        for (int j=0;j<9;++j) out<<' '<<integral_sum[j]/interval;
        out<<'\n';
        pair_sum=0;
        for (int j=0;j<9;++j) corr_sum[j]=integral_sum[j]=0;
      }
    }
    commit(out,path);
  }
}
}
