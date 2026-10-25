#include "measure/rpmd_ja_qraw_v4.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

int main()
{
  constexpr int n=3,d=9;
  constexpr std::uint64_t dd=static_cast<std::uint64_t>(d)*d,vd=static_cast<std::uint64_t>(d)*n;
  constexpr std::size_t header_bytes=173;
  const std::vector<double> masses={1.0,2.5,7.0};
  std::vector<double> v(vd),fine_c(3*dd),coarse_c(3*dd),vtranspose(vd),fenergy(d);
  for(int row=0;row<d;++row){double sum=0.0;for(int atom=0;atom<n;++atom){
    const double value=0.071*(row+1)-0.019*(atom+2)*(atom+1)+0.003*(row+1)*(atom+1);
    v[static_cast<std::size_t>(row)*n+atom]=value;vtranspose[static_cast<std::size_t>(atom)*d+row]=value;sum+=value;}
    fenergy[row]=-sum;}
  for(int alpha=0;alpha<3;++alpha)for(int coordinate=0;coordinate<d;++coordinate)for(int row=0;row<d;++row){
    const std::size_t ix=static_cast<std::size_t>(alpha)*dd+static_cast<std::size_t>(coordinate)*d+row;
    fine_c[ix]=0.4+0.017*(alpha+1)*(coordinate+1)-0.011*(row+2)*(row+1)+0.002*(coordinate-row);
    coarse_c[ix]=fine_c[ix]+0.0007*(1+alpha+coordinate%3)*(row+1);}
  const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory=std::filesystem::temp_directory_path()/
    ("rpmd_ja_qraw_v4_transport_"+std::to_string(stamp));
  std::filesystem::create_directory(directory);
  struct Cleanup { std::filesystem::path path; ~Cleanup(){std::error_code error;std::filesystem::remove_all(path,error);} } cleanup{directory};
  const std::filesystem::path raw_path=directory/"fixture.qraw",transpose_path=directory/"fixture.vtranspose";
  {
    std::ofstream out(raw_path,std::ios::binary|std::ios::trunc);assert(out.good());
    const std::vector<unsigned char> header(header_bytes,0x5a);out.write(reinterpret_cast<const char*>(header.data()),header.size());
    out.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double)));
    out.write(reinterpret_cast<const char*>(fine_c.data()),static_cast<std::streamsize>(fine_c.size()*sizeof(double)));
    const std::vector<double> k(dd,0.125),footer(24,19.0);
    out.write(reinterpret_cast<const char*>(k.data()),static_cast<std::streamsize>(k.size()*sizeof(double)));
    out.write(reinterpret_cast<const char*>(footer.data()),static_cast<std::streamsize>(footer.size()*sizeof(double)));
    out.write(reinterpret_cast<const char*>(coarse_c.data()),static_cast<std::streamsize>(coarse_c.size()*sizeof(double)));
    out.close();assert(out.good());
  }
  {
    std::ofstream out(transpose_path,std::ios::binary|std::ios::trunc);assert(out.good());
    out.write(reinterpret_cast<const char*>(vtranspose.data()),static_cast<std::streamsize>(vtranspose.size()*sizeof(double)));
    out.close();assert(out.good());
  }
  std::ifstream raw(raw_path,std::ios::binary),transpose(transpose_path,std::ios::binary);assert(raw.good()&&transpose.good());
  const std::streamoff data=static_cast<std::streamoff>(header_bytes);
  const std::streamoff fine_c_start=data+static_cast<std::streamoff>(vd)*sizeof(double);
  const std::streamoff coarse_c_start=fine_c_start+static_cast<std::streamoff>(4*dd+24)*sizeof(double);
  const auto actual=rpmd_ja_qraw_v4::transport_statistics_from_streams(
    n,d,masses,fenergy,raw,transpose,fine_c_start,coarse_c_start);
  double expected_diff[3]={},expected_norm[3]={};
  for(int alpha=0;alpha<3;++alpha)for(int coordinate=0;coordinate<d;++coordinate){
    const int component=coordinate/n,atom=coordinate%n;
    for(int row=0;row<d;++row){const std::size_t ix=static_cast<std::size_t>(alpha)*dd+static_cast<std::size_t>(coordinate)*d+row;
      double fine=fine_c[ix],coarse=coarse_c[ix];
      if(component==alpha&&row%n==atom){fine-=fenergy[row];coarse-=fenergy[row];}
      if(component==alpha){fine-=v[static_cast<std::size_t>(row)*n+atom];coarse-=v[static_cast<std::size_t>(row)*n+atom];}
      const double scale=std::sqrt(masses[row%n]*masses[atom]);fine/=scale;coarse/=scale;
      const double difference=fine-coarse;expected_diff[alpha]+=difference*difference;expected_norm[alpha]+=fine*fine;}}
  for(int alpha=0;alpha<3;++alpha){assert(actual.difference_squared[alpha]>0.0&&actual.fine_norm_squared[alpha]>0.0);
    assert(std::abs(actual.difference_squared[alpha]-expected_diff[alpha])<1e-13*expected_diff[alpha]);
    assert(std::abs(actual.fine_norm_squared[alpha]-expected_norm[alpha])<1e-13*expected_norm[alpha]);
    assert(std::sqrt(actual.difference_squared[alpha]/actual.fine_norm_squared[alpha])<0.05);}
  raw.close();transpose.close();assert(!raw.is_open()&&!transpose.is_open()&&raw.good()&&transpose.good());
  return 0;
}
