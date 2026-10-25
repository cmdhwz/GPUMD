#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <istream>
#include <stdexcept>
#include <vector>

namespace rpmd_ja_qraw_v4
{
struct TransportStatistics
{
  std::array<double, 3> difference_squared{};
  std::array<double, 3> fine_norm_squared{};
};

template<class ReadCRows, class ReadVColumn>
TransportStatistics transport_statistics(const int n, const int d, const std::vector<double>& masses,
  const std::vector<double>& fenergy, ReadCRows read_c_rows, ReadVColumn read_v_column)
{
  if(n<=1||d!=3*n||masses.size()!=static_cast<std::size_t>(n)||fenergy.size()!=static_cast<std::size_t>(d))
    throw std::invalid_argument("qraw v4 transport-statistic dimensions do not match");
  if(!std::all_of(masses.begin(),masses.end(),[](double x){return std::isfinite(x)&&x>0.0;})||
     !std::all_of(fenergy.begin(),fenergy.end(),[](double x){return std::isfinite(x);}))
    throw std::runtime_error("qraw v4 transport-statistic masses or gradient are invalid");
  std::vector<double> fine(d),coarse(d),v_column(d);
  TransportStatistics result;
  for(int alpha=0;alpha<3;++alpha)for(int coordinate=0;coordinate<d;++coordinate){
    read_c_rows(alpha,coordinate,fine.data(),coarse.data());
    const int component=coordinate/n,atom=coordinate%n;
    if(component==alpha)read_v_column(atom,v_column.data());
    for(int row=0;row<d;++row){
      double hf=fine[row],hc=coarse[row];
      if(component==alpha&&row%n==atom){hf-=fenergy[row];hc-=fenergy[row];}
      if(component==alpha){hf-=v_column[row];hc-=v_column[row];}
      const double denominator=std::sqrt(masses[row%n]*masses[atom]);
      hf/=denominator;hc/=denominator;
      if(!std::isfinite(hf)||!std::isfinite(hc))
        throw std::runtime_error("qraw v4 transported-H value is non-finite");
      const double difference=hf-hc;
      result.difference_squared[alpha]+=difference*difference;
      result.fine_norm_squared[alpha]+=hf*hf;
    }
  }
  for(int alpha=0;alpha<3;++alpha)
    if(!std::isfinite(result.difference_squared[alpha])||!std::isfinite(result.fine_norm_squared[alpha]))
      throw std::runtime_error("qraw v4 transport statistics overflow");
  return result;
}

inline TransportStatistics transport_statistics_from_streams(const int n, const int d, const std::vector<double>& masses,
  const std::vector<double>& fenergy, std::istream& raw, std::istream& vtranspose,
  const std::streamoff fine_c_start, const std::streamoff coarse_c_start)
{
  return transport_statistics(n,d,masses,fenergy,
    [&](const int alpha,const int coordinate,double* fine,double* coarse){
      const std::streamoff row_offset=(static_cast<std::streamoff>(alpha)*d*d+
        static_cast<std::streamoff>(coordinate)*d)*sizeof(double);
      raw.clear();raw.seekg(fine_c_start+row_offset);
      raw.read(reinterpret_cast<char*>(fine),static_cast<std::streamsize>(d*sizeof(double)));
      if(!raw)throw std::runtime_error("truncated qNEP v4 fine C while building transport statistics");
      raw.clear();raw.seekg(coarse_c_start+row_offset);
      raw.read(reinterpret_cast<char*>(coarse),static_cast<std::streamsize>(d*sizeof(double)));
      if(!raw)throw std::runtime_error("truncated qNEP v4 coarse C while building transport statistics");
    },
    [&](const int site,double* values){
      vtranspose.clear();vtranspose.seekg(static_cast<std::streamoff>(site)*d*sizeof(double));
      vtranspose.read(reinterpret_cast<char*>(values),static_cast<std::streamsize>(d*sizeof(double)));
      if(!vtranspose)throw std::runtime_error("truncated qNEP v4 V-transpose while building transport statistics");
    });
}
} // namespace rpmd_ja_qraw_v4
