#include "rpmd_ja_native_fit.cuh"

#ifndef USE_HIP
#include "rpmd_ja_qnep_prepare.cuh"
#include "rpmd_ja_reference.cuh"
#include "rpmd_ja_reference_math.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "force/force.cuh"
#include "force/nep_charge.cuh"
#include "utilities/common.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace
{
constexpr char sample_magic[8] = {'G','P','J','A','S','M','P','1'};
constexpr char additive_magic[8] = {'G','P','J','A','A','D','D','1'};
constexpr std::uint32_t native_endian = 0x01020304;

template <typename T> T read(std::istream& in, const char* what)
{
  T value{};
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error(std::string("truncated native fit spool while reading ") + what);
  return value;
}

template <typename T> void read_array(std::istream& in, T* out, std::size_t count, const char* what)
{
  if (count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T))
    throw std::runtime_error("native fit spool array size overflows stream limits");
  in.read(reinterpret_cast<char*>(out), static_cast<std::streamsize>(count * sizeof(T)));
  if (!in) throw std::runtime_error(std::string("truncated native fit spool while reading ") + what);
}

template <typename T> void write(std::ostream& out, const T& value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!out) throw std::runtime_error("failed writing native additive package");
}

struct SampleHeader
{
  int n = 0, beads = 0;
  double temperature = 0.0;
  double cell[9] = {};
  std::vector<double> masses;
  std::vector<int> types;
  std::streamoff frames = 0;
  std::uint64_t frame_count = 0;
};

SampleHeader read_header(std::istream& in, const std::uint64_t frame_count, const Atom& atom, const Box& box,
                         const double temperature, const bool diagnostic = false,
                         const bool sample_beads_from_header = false)
{
  char magic[8]; in.read(magic, sizeof(magic));
  const auto version = read<std::uint32_t>(in, "version");
  const auto endian = read<std::uint32_t>(in, "byte order");
  SampleHeader h;
  h.n = read<std::int32_t>(in, "atom count");
  h.beads = read<std::int32_t>(in, "bead count");
  h.temperature = read<double>(in, "temperature");
  if (h.n != atom.number_of_atoms || h.n < 2 || h.n > std::numeric_limits<int>::max()/3 || h.beads < 1 ||
      (!diagnostic && !sample_beads_from_header && h.beads != atom.number_of_beads))
    throw std::runtime_error("native fit spool has invalid atom or bead count");
  read_array(in, h.cell, 9, "cell");
  h.masses.resize(h.n); h.types.resize(h.n);
  read_array(in, h.masses.data(), h.masses.size(), "masses");
  read_array(in, h.types.data(), h.types.size(), "types");
  h.frames = in.tellg();
  if (std::memcmp(magic, sample_magic, sizeof(magic)) || version != 1 || endian != native_endian || h.n != atom.number_of_atoms ||
      h.n < 2 || h.beads < 1 || (!diagnostic && !sample_beads_from_header && h.beads != atom.number_of_beads) || !std::isfinite(h.temperature) || h.temperature <= 0.0 ||
      (!diagnostic && std::abs(h.temperature - temperature) > 1.0e-10 * temperature) ||
      h.masses != atom.cpu_mass || h.types != atom.cpu_type || (!diagnostic && frame_count < 3))
    throw std::runtime_error("native fit spool identity, temperature, or dimensions do not match the initialized system");
  if(!std::all_of(h.masses.begin(),h.masses.end(),[](double m){return std::isfinite(m)&&m>0.0;})||
     !std::all_of(h.cell,h.cell+9,[](double x){return std::isfinite(x);}))
    throw std::runtime_error("native fit spool has non-finite cell or invalid masses");
  const auto here=in.tellg();in.seekg(0,std::ios::end);const std::streamoff end=in.tellg();in.seekg(here);
  const std::uint64_t dimension=3ULL*static_cast<std::uint64_t>(h.n),frame_bytes=sizeof(double)+2*dimension*sizeof(double);
  if(h.frames<0||end<h.frames||frame_bytes==0)
    throw std::runtime_error("native fit spool has an invalid byte length");
  const std::uint64_t payload=static_cast<std::uint64_t>(end-h.frames);
  if(payload%frame_bytes!=0)
    throw std::runtime_error("native fit spool has a truncated frame or extra partial data");
  h.frame_count=payload/frame_bytes;
  if(h.frame_count<3 || (frame_count!=0 && h.frame_count!=frame_count) ||
     h.frame_count>(std::numeric_limits<std::uint64_t>::max()-static_cast<std::uint64_t>(h.frames))/frame_bytes||
     static_cast<std::uint64_t>(h.frames)+h.frame_count*frame_bytes>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())||
     end!=static_cast<std::streamoff>(static_cast<std::uint64_t>(h.frames)+h.frame_count*frame_bytes))
    throw std::runtime_error("native fit spool byte length does not match frame_count");
  for (int i = 0; i < 9; ++i)
    if (!std::isfinite(h.cell[i]) || std::abs(h.cell[i] - box.cpu_h[i]) > 1.0e-9 * std::max(1.0, std::abs(box.cpu_h[i])))
      throw std::runtime_error("native fit spool cell does not match the fixed simulation cell");
  return h;
}

struct QrawIdentity
{
  std::streamoff data_offset=0,k_offset=0;
  std::uint32_t version=0;
  std::uint64_t model=0,config=0;
  int charge=0,kspace=0;
  double mesh=0.0,active_mesh=0.0;
};

QrawIdentity validate_qraw_identity(std::istream& raw,const RpmdJANativeFitOptions& options,
  const SampleHeader& header,const std::vector<double>& r0,const Box& box,Force& force)
{
  const int n=header.n,d=3*n;
  char magic[8];raw.read(magic,sizeof(magic));const auto version=read<std::uint32_t>(raw,"qraw version");
  const auto endian=read<std::uint32_t>(raw,"qraw byte order");const int raw_n=read<std::int32_t>(raw,"qraw atom count");
  const int raw_d=read<std::int32_t>(raw,"qraw dimension");const double temperature=read<double>(raw,"qraw temperature");
  const double fd_step=read<double>(raw,"qraw fd step");const auto model=read<std::uint64_t>(raw,"qraw model fingerprint");
  const auto config=read<std::uint64_t>(raw,"qraw config fingerprint");const int charge=read<std::int32_t>(raw,"qraw charge mode");
  const int kspace=read<std::int32_t>(raw,"qraw kspace flag");const double mesh=read<double>(raw,"qraw mesh spacing");
  char layout[sizeof("xyz_soa;derivative_input_rows_output_columns")];raw.read(layout,sizeof(layout));
  if(!raw||std::memcmp(magic,"GPJQRAW\0",8)||version!=3||endian!=native_endian||raw_n!=n||raw_d!=d||
     temperature!=options.temperature||fd_step!=options.fd_step||
     std::memcmp(layout,"xyz_soa;derivative_input_rows_output_columns",sizeof(layout)))
    throw std::runtime_error("qraw version, dimensions, temperature, fd step, or matrix layout does not match native fit assumptions");
  double raw_cell[18];raw.read(reinterpret_cast<char*>(raw_cell),sizeof(raw_cell));int raw_pbc[3];
  raw.read(reinterpret_cast<char*>(raw_pbc),sizeof(raw_pbc));std::vector<std::int32_t> raw_types(n);
  std::vector<double> raw_masses(n),raw_positions(d);read_array(raw,raw_types.data(),raw_types.size(),"qraw atom types");
  read_array(raw,raw_masses.data(),raw_masses.size(),"qraw atom masses");read_array(raw,raw_positions.data(),raw_positions.size(),"qraw reference positions");
  const std::streamoff skip=static_cast<std::streamoff>(sizeof(double))+static_cast<std::streamoff>(n)*sizeof(double)+
    static_cast<std::streamoff>(d)*sizeof(double)+static_cast<std::streamoff>(9*n)*sizeof(double);
  raw.seekg(skip,std::ios::cur);const std::streamoff data=raw.tellg();
  if(!raw||data<0)throw std::runtime_error("qraw payload offset is invalid");
  const std::uint64_t dd=static_cast<std::uint64_t>(d)*d,vd=static_cast<std::uint64_t>(d)*n;
  const std::uint64_t max_u64=std::numeric_limits<std::uint64_t>::max();
  if(dd>(max_u64-18)/7||vd>(max_u64-(7*dd+18))/2)throw std::runtime_error("qraw byte length overflows");
  const std::uint64_t payload_elements=2*vd+7*dd+18;
  if(payload_elements>max_u64/sizeof(double))throw std::runtime_error("qraw byte length overflows");
  const std::uint64_t payload_bytes=payload_elements*sizeof(double);
  if(payload_bytes>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())-static_cast<std::uint64_t>(data))
    throw std::runtime_error("qraw byte length exceeds file-offset limits");
  const std::streamoff expected_end=data+static_cast<std::streamoff>(payload_bytes);
  const std::streamoff k_offset=data+static_cast<std::streamoff>(vd+3*dd)*sizeof(double);
  auto* active_qnep=force.get_number_of_potentials()==1?dynamic_cast<NEP_Charge*>(&force.get_potential(0)):nullptr;
  if(active_qnep==nullptr||!active_qnep->uses_pppm())
    throw std::runtime_error("native fit qraw validation requires exactly one qNEP potential using PPPM");
  const double active_mesh=active_qnep->get_pppm_mesh_spacing();
  const auto expected_model=rpmd_ja_model_fingerprint(force.primary_nep_model_path());
  const auto expected_config=rpmd_ja_qnep_config_fingerprint(force);
  bool geometry_matches=std::all_of(raw_cell,raw_cell+18,[](double x){return std::isfinite(x);})&&
    std::all_of(raw_positions.begin(),raw_positions.end(),[](double x){return std::isfinite(x);});
  for(int i=0;i<18&&geometry_matches;++i)geometry_matches=std::abs(raw_cell[i]-box.cpu_h[i])<=1e-10*std::max(1.0,std::abs(box.cpu_h[i]));
  for(int i=0;i<9&&geometry_matches;++i)geometry_matches=std::abs(raw_cell[i]-header.cell[i])<=1e-10*std::max(1.0,std::abs(header.cell[i]));
  for(int i=0;i<3&&geometry_matches;++i)geometry_matches=raw_pbc[i]==1;
  for(int i=0;i<n&&geometry_matches;++i)geometry_matches=raw_types[i]==header.types[i]&&raw_masses[i]==header.masses[i];
  for(int i=0;i<d&&geometry_matches;++i)geometry_matches=std::abs(raw_positions[i]-r0[i])<=1e-10*std::max(1.0,std::abs(r0[i]));
  if(!raw||model!=expected_model||config!=expected_config||charge!=active_qnep->get_charge_mode()||kspace!=1||
     !std::isfinite(mesh)||std::abs(mesh-active_mesh)>1e-12*std::max(1.0,std::abs(active_mesh))||!geometry_matches)
    throw std::runtime_error("qNEP qraw model, PPPM configuration, temperature, step, atom identity, cell, or sampled training R0 does not match current fit inputs");
  raw.clear();raw.seekg(0,std::ios::end);const std::streamoff actual_end=raw.tellg();
  if(actual_end!=expected_end)throw std::runtime_error("qNEP raw file byte length is invalid");
  raw.seekg(data+static_cast<std::streamoff>(2*vd+7*dd)*sizeof(double));double stats[18]={};
  raw.read(reinterpret_cast<char*>(stats),sizeof(stats));
  const bool finite_stats=raw&&std::all_of(stats,stats+18,[](double value){return std::isfinite(value)&&value>=0.0;});
  const bool raw_v1=stats[17]==1.0&&stats[1]<=1e-4&&std::max({stats[2],stats[4],stats[6],stats[7],stats[8],stats[14],stats[15]})<=5e-2;
  const bool raw_v2=stats[17]==2.0&&stats[1]<=1e-4&&stats[3]<=1e-4&&std::max({stats[4],stats[6],stats[7],stats[8],stats[14],stats[15]})<=5e-2;
  const bool raw_v3=stats[17]==3.0&&stats[1]<=1e-4&&stats[3]<=1e-4&&std::max({stats[4],stats[6],stats[7],stats[8],stats[14],stats[15]})<=5e-2;
  if(!finite_stats||stats[16]!=options.fd_step||(version==1?!raw_v1:(version==2?!raw_v2:!raw_v3)))
    throw std::runtime_error("qNEP raw diagnostics fail accepted limits");
  raw.clear();return {data,k_offset,version,model,config,charge,kspace,mesh,active_mesh};
}

void read_frame(std::istream& in, std::vector<double>& x, std::vector<double>& f, double& step)
{
  step = read<double>(in, "frame step");
  if (!std::isfinite(step)) throw std::runtime_error("native fit spool has non-finite step");
  read_array(in, x.data(), x.size(), "centroid");
  read_array(in, f.data(), f.size(), "mean physical force");
  if (!std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); }) ||
      !std::all_of(f.begin(), f.end(), [](double v) { return std::isfinite(v); }))
    throw std::runtime_error("native fit spool frame contains non-finite values");
}

double read_training_r0(std::istream& in, const SampleHeader& header, const std::uint64_t frame_count,
                        const double expected_interval, std::vector<double>& r0)
{
  const std::uint64_t train = 2 * frame_count / 3;
  std::vector<double> x(r0.size()), f(r0.size());
  double step = 0.0, previous = -std::numeric_limits<double>::infinity(), interval = 0.0;
  in.clear(); in.seekg(header.frames);
  for (std::uint64_t frame = 0; frame < frame_count; ++frame) {
    read_frame(in, x, f, step);
    if (frame == 1) {
      interval = step - previous;
      if (!(interval > 0.0) || !std::isfinite(interval))
        throw std::runtime_error("native fit spool sample interval must be positive and finite");
    }
    if (frame && (!(step > previous) || !std::isfinite(step - previous) ||
        std::abs((step - previous) - interval) > 1.0e-9 ||
        (std::isfinite(expected_interval) && std::abs((step - previous) - expected_interval) > 1.0e-9)))
      throw std::runtime_error("native fit sample steps must be strictly increasing with a uniform interval");
    previous = step;
    if (frame < train)
      for (std::size_t k = 0; k < r0.size(); ++k) r0[k] += x[k] / static_cast<double>(train);
  }
  if (!std::all_of(r0.begin(), r0.end(), [](double v) { return std::isfinite(v); }))
    throw std::runtime_error("native fit training R0 is non-finite");
  return interval;
}

void validate_fit_frame_branch(const std::vector<double>& x,const int n,const Box& box,const std::vector<double>& r0)
{
  for(int i=0;i<n;++i){const double delta[3]={x[i]-r0[i],x[n+i]-r0[n+i],x[2*n+i]-r0[2*n+i]};
    for(int a=0;a<3;++a){double fractional=0.0;for(int b=0;b<3;++b)fractional+=box.cpu_h[9+3*a+b]*delta[b];
      if(!std::isfinite(fractional)||std::abs(fractional)>=0.45)throw std::runtime_error("native fit sample centroid exceeds the fixed-reference branch limit 0.45");}}
}

void validate_fit_branches(std::istream& in, const SampleHeader& header, const std::uint64_t frame_count,
                           const Box& box, const std::vector<double>& r0)
{
  std::vector<double> x(r0.size()), f(r0.size());
  in.clear(); in.seekg(header.frames);
  for (std::uint64_t frame = 0; frame < frame_count; ++frame) {
    double step = 0.0;
    read_frame(in, x, f, step);
    validate_fit_frame_branch(x,header.n,box,r0);
  }
}

struct Edge
{
  int i = 0, j = 0, group = 0, image[3] = {};
};

struct SiteEdge
{
  int j = 0, group = 0, edge_index = 0, image[3] = {};
  double sign = 1.0;
};

struct Graph
{
  std::vector<Edge> edges;
  std::vector<std::vector<SiteEdge>> sites;
  std::vector<std::pair<int,int>> type_pairs;
  std::map<std::pair<int,int>,int> group_by_type;
};

__global__ void mass_symmetrize(double* k, const double* sqrt_mass, int* error, const int d, const int n)
{
  const std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const std::size_t dd=static_cast<std::size_t>(d)*d;
  if(q>=dd)return;
  const int i=static_cast<int>(q/d),j=static_cast<int>(q%d);
  if(i>j)return;
  const std::size_t ij=static_cast<std::size_t>(i)*d+j,ji=static_cast<std::size_t>(j)*d+i;
  const double scale=sqrt_mass[i%n]*sqrt_mass[j%n];const double a=k[ij]/scale;
  const double b=k[ji]/scale;
  const double v=0.5*a+0.5*b;if(!(scale>0.0)||!isfinite(scale)||!isfinite(a)||!isfinite(b)||!isfinite(v)){atomicExch(error,1);k[ij]=k[ji]=0.0;return;}k[ij]=v;k[ji]=v;
}

__global__ void project_internal(double* k, const double* t, const double* kt, const int d, const double* tkt)
{
  const std::size_t q=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  const std::size_t dd=static_cast<std::size_t>(d)*d;
  if(q>=dd)return;
  const int i=static_cast<int>(q/d),j=static_cast<int>(q%d);
  double v=k[q];
  for(int a=0;a<3;++a)v-=t[a*d+i]*kt[a*d+j]+kt[a*d+i]*t[a*d+j];
  for(int a=0;a<3;++a)for(int b=0;b<3;++b)v+=t[a*d+i]*tkt[3*a+b]*t[b*d+j];
  k[q]=v;
}

__global__ void lanczos_translation_coefficients(const double* v,const double* sqrt_atom,double* coefficient,
  const int n,const double inverse_mass_sum)
{
  __shared__ double partial[256];
  const int axis=static_cast<int>(blockIdx.x);double sum=0.0;
  for(int i=threadIdx.x;i<n;i+=blockDim.x)sum+=sqrt_atom[i]*v[axis*n+i];
  partial[threadIdx.x]=sum;__syncthreads();
  for(int stride=blockDim.x/2;stride>0;stride>>=1){if(threadIdx.x<stride)partial[threadIdx.x]+=partial[threadIdx.x+stride];__syncthreads();}
  if(threadIdx.x==0)coefficient[axis]=partial[0]*inverse_mass_sum;
}

__global__ void lanczos_remove_translation(double* v,const double* sqrt_atom,const double* coefficient,const int d,const int n)
{
  const int q=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;if(q>=d)return;
  v[q]-=sqrt_atom[q%n]*coefficient[q/n];
}

__global__ void lanczos_additive_action(const double* q,double* result,const double* sqrt_mass,const double* theta,
  const int* offsets,const int* neighbors,const int* groups,const signed char* signs,const int n,const int d)
{
  const int index=static_cast<int>(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=d)return;
  const int axis=index/n,atom=index-axis*n;double value=0.0;
  for(int edge=offsets[atom];edge<offsets[atom+1];++edge){
    const int neighbor=neighbors[edge],group=groups[edge],i=signs[edge]>0?atom:neighbor,j=signs[edge]>0?neighbor:atom;
    const std::size_t p=6*static_cast<std::size_t>(group);const double c0=theta[p],c1=theta[p+1],c2=theta[p+2],c3=theta[p+3],c4=theta[p+4],c5=theta[p+5];
    const double dx0=q[j]/sqrt_mass[j]-q[i]/sqrt_mass[i];
    const double dx1=q[n+j]/sqrt_mass[n+j]-q[n+i]/sqrt_mass[n+i];
    const double dx2=q[2*n+j]/sqrt_mass[2*n+j]-q[2*n+i]/sqrt_mass[2*n+i];
    const double y=axis==0?c0*dx0+c1*dx1+c2*dx2:(axis==1?c1*dx0+c3*dx1+c4*dx2:c2*dx0+c4*dx1+c5*dx2);
    value-=static_cast<double>(signs[edge])*y/sqrt_mass[index];
  }
  result[index]=value;
}

__device__ inline void add_compensated_value(double& sum,double& correction,const double value)
{
  const double next=sum+value;
  correction+=fabs(sum)>=fabs(value)?(sum-next)+value:(value-next)+sum;
  const double merged=next+correction;
  correction-=merged-next;
  sum=merged;
}

__device__ inline void add_compensated_pair(double& sum,double& correction,const double other_sum,const double other_correction)
{
  add_compensated_value(sum,correction,other_sum);
  add_compensated_value(sum,correction,other_correction);
}

__global__ void compensated_baseline_matvec(const double* matrix,const double* input,double* output_high,
  double* output_low,const int dimension)
{
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  const int row=static_cast<int>(blockIdx.x)*8+warp;
  if(row>=dimension)return;
  double sum=0.0,correction=0.0;
  for(int column=lane;column<dimension;column+=32){
    const double a=matrix[static_cast<std::size_t>(row)*dimension+column];
    const double b=input[column],product=a*b;
    add_compensated_value(sum,correction,product);
    add_compensated_value(sum,correction,fma(a,b,-product));
  }
  for(int offset=16;offset>0;offset>>=1){
    const double other_sum=__shfl_down_sync(0xffffffff,sum,offset);
    const double other_correction=__shfl_down_sync(0xffffffff,correction,offset);
    if(lane<offset)add_compensated_pair(sum,correction,other_sum,other_correction);
  }
  if(lane==0){output_high[row]=sum;output_low[row]=correction;}
}

struct DeviceBaseline
{
  static constexpr int batch_capacity=32;
  double* k=nullptr; double* q=nullptr; double* out=nullptr; double* compensated_high=nullptr; double* compensated_low=nullptr; double* batch_q=nullptr; double* batch_out=nullptr; double* t=nullptr; double* kt=nullptr;double* g=nullptr;int* error=nullptr;
  cublasHandle_t blas=nullptr; int d=0;std::uint64_t dense_matrix_actions=0;
  void release(){if(blas){cublasDestroy(blas);blas=nullptr;}if(k){cudaFree(k);k=nullptr;}if(q){cudaFree(q);q=nullptr;}if(out){cudaFree(out);out=nullptr;}if(compensated_high){cudaFree(compensated_high);compensated_high=nullptr;}if(compensated_low){cudaFree(compensated_low);compensated_low=nullptr;}if(batch_q){cudaFree(batch_q);batch_q=nullptr;}if(batch_out){cudaFree(batch_out);batch_out=nullptr;}if(t){cudaFree(t);t=nullptr;}if(kt){cudaFree(kt);kt=nullptr;}if(g){cudaFree(g);g=nullptr;}if(error){cudaFree(error);error=nullptr;}d=0;dense_matrix_actions=0;}
  ~DeviceBaseline(){release();}
  void initialize(std::istream& raw,const std::streamoff offset,const int dimension,const int n,
                  const std::vector<double>& masses,const std::vector<double>& sqrt_mass)
  {
    d=dimension;std::size_t free_bytes=0,total_bytes=0;
    if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)throw std::runtime_error("native fit cudaMemGetInfo failed");
    const std::size_t dim_size=static_cast<std::size_t>(d),max_size=std::numeric_limits<std::size_t>::max();
    if(d<=0||dim_size>max_size/dim_size/sizeof(double)||dim_size>max_size/batch_capacity/sizeof(double))throw std::runtime_error("native fit raw Hessian GPU workspace size overflows");
    const std::size_t bytes=dim_size*dim_size*sizeof(double),batch_bytes=dim_size*batch_capacity*sizeof(double);
    if(bytes>free_bytes||batch_bytes>(free_bytes-bytes)/2||free_bytes-bytes-2*batch_bytes<256ULL*1024*1024)throw std::runtime_error("native fit raw Hessian GPU preflight leaves less than 256 MiB safety margin");
    if(cudaMalloc(reinterpret_cast<void**>(&k),bytes)!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&q),dim_size*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&out),static_cast<std::size_t>(d)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&t),static_cast<std::size_t>(3)*d*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&compensated_high),dim_size*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&compensated_low),dim_size*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&batch_q),batch_bytes)!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&batch_out),batch_bytes)!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&kt),static_cast<std::size_t>(3)*d*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&g),9*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&error),sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit could not allocate raw Hessian GPU workspace");
    if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit cublasCreate failed");
    raw.clear();raw.seekg(offset);const std::size_t count=static_cast<std::size_t>(d)*d,chunk=std::min<std::size_t>(count,8ULL*1024*1024);std::vector<double> host(chunk);
    for(std::size_t off=0;off<count;){const std::size_t m=std::min(chunk,count-off);read_array(raw,host.data(),m,"raw qNEP Hessian");if(!std::all_of(host.begin(),host.begin()+m,[](double x){return std::isfinite(x);}))throw std::runtime_error("native fit raw qNEP Hessian contains non-finite values");if(cudaMemcpy(k+off,host.data(),m*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit raw Hessian upload failed");off+=m;}
    // sqrt_mass is repeated in xyz SoA order for row/column scaling.
    if(cudaMemcpy(q,sqrt_mass.data(),static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit mass upload failed");
    const std::size_t dd=static_cast<std::size_t>(d)*d;
    if(cudaMemset(error,0,sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit Hessian finite-check initialization failed");mass_symmetrize<<<static_cast<unsigned>((dd+255)/256),256>>>(k,q,error,d,n);int host_error=0;if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess||cudaMemcpy(&host_error,error,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_error)throw std::runtime_error("native fit mass-weighted Hessian symmetrization failed or produced non-finite values");
    std::vector<double> translation(3*static_cast<std::size_t>(d),0.0);double mass_sum=std::accumulate(masses.begin(),masses.end(),0.0);if(!std::isfinite(mass_sum)||!(mass_sum>0.0))throw std::runtime_error("native fit total mass is invalid");
    for(int a=0;a<3;++a)for(int i=0;i<n;++i)translation[a*d+a*n+i]=sqrt_mass[i]/std::sqrt(mass_sum);
    if(cudaMemcpy(t,translation.data(),translation.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit translation upload failed");
    const double one=1.0,zero=0.0;double tkt[9]={};
    for(int a=0;a<3;++a){if(cublasDgemv(blas,CUBLAS_OP_T,d,d,&one,k,d,t+a*d,1,&zero,kt+a*d,1)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit translation projection matvec failed");
      for(int b=0;b<3;++b)if(cublasDdot(blas,d,t+b*d,1,kt+a*d,1,&tkt[3*b+a])!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit translation cross scalar failed");}
    if(cudaMemcpy(g,tkt,sizeof(tkt),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit translation projection scalar upload failed");
    project_internal<<<static_cast<unsigned>((dd+255)/256),256>>>(k,t,kt,d,g);if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native fit internal Hessian projection failed");
  }
  void apply(const std::vector<double>& input,std::vector<double>& result)
  {
    if(cudaMemcpy(q,input.data(),static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       apply_device(q,out)!=CUBLAS_STATUS_SUCCESS||
       cudaMemcpy(result.data(),out,static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("native fit Hessian matvec failed");
  }
  void apply_compensated(const std::vector<double>& input,std::vector<double>& high,std::vector<double>& low)
  {
    if(input.size()!=static_cast<std::size_t>(d)||high.size()!=input.size()||low.size()!=input.size())
      throw std::invalid_argument("native fit compensated Hessian matvec dimensions are invalid");
    if(cudaMemcpy(q,input.data(),input.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw std::runtime_error("native fit compensated Hessian input upload failed");
    compensated_baseline_matvec<<<static_cast<unsigned>((static_cast<std::size_t>(d)+7)/8),256>>>(k,q,compensated_high,compensated_low,d);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess||
       cudaMemcpy(high.data(),compensated_high,input.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||
       cudaMemcpy(low.data(),compensated_low,input.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("native fit compensated Hessian matvec failed");
    ++dense_matrix_actions;
  }
  void apply_batch(const std::vector<double>& input,const int columns,std::vector<double>& result)
  {
    if(columns<=0||columns>batch_capacity||input.size()<static_cast<std::size_t>(d)*columns||result.size()<static_cast<std::size_t>(d)*columns)
      throw std::invalid_argument("native fit Hessian batch dimensions are invalid");
    const std::size_t count=static_cast<std::size_t>(d)*columns;const double one=1.0,zero=0.0;
    if(cudaMemcpy(batch_q,input.data(),count*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       cublasDgemm(blas,CUBLAS_OP_T,CUBLAS_OP_N,d,columns,d,&one,k,d,batch_q,d,&zero,batch_out,d)!=CUBLAS_STATUS_SUCCESS||
       cudaMemcpy(result.data(),batch_out,count*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("native fit Hessian batch matmul failed");
  }
  cublasStatus_t apply_device(const double* input,double* result)
  {
    const double one=1.0,zero=0.0;
    const cublasStatus_t status=cublasDgemv(blas,CUBLAS_OP_T,d,d,&one,k,d,input,1,&zero,result,1);
    if(status==CUBLAS_STATUS_SUCCESS)++dense_matrix_actions;
    return status;
  }
};

struct DeviceLanczosWorkspace
{
  double *basis=nullptr,*v=nullptr,*previous=nullptr,*w=nullptr,*add=nullptr,*ritz=nullptr,*coefficient=nullptr,*theta=nullptr,*sqrt_mass=nullptr,*sqrt_atom=nullptr,*translation=nullptr;
  int *offsets=nullptr,*neighbors=nullptr,*groups=nullptr; signed char* signs=nullptr;
  int d=0,n=0,p=0,max_steps=0;double inverse_mass_sum=0.0;
  void release(){double** values[]={&basis,&v,&previous,&w,&add,&ritz,&coefficient,&theta,&sqrt_mass,&sqrt_atom,&translation};for(double** value:values)if(*value){cudaFree(*value);*value=nullptr;}if(offsets){cudaFree(offsets);offsets=nullptr;}if(neighbors){cudaFree(neighbors);neighbors=nullptr;}if(groups){cudaFree(groups);groups=nullptr;}if(signs){cudaFree(signs);signs=nullptr;}d=n=p=max_steps=0;inverse_mass_sum=0.0;}
  ~DeviceLanczosWorkspace(){release();}
  void initialize(const Graph& graph,const std::vector<double>& sqrt_mass_host,const std::vector<double>& sqrt_atom_host,
                  const int steps,const std::size_t parameter_count)
  {
    if(sqrt_atom_host.empty()||sqrt_atom_host.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()/3)||parameter_count>static_cast<std::size_t>(std::numeric_limits<int>::max())||steps<=0||graph.edges.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()/2))
      throw std::runtime_error("native fit Lanczos workspace dimensions are invalid");
    d=static_cast<int>(sqrt_mass_host.size());n=static_cast<int>(sqrt_atom_host.size());p=static_cast<int>(parameter_count);max_steps=steps;
    if(d!=3*n)throw std::runtime_error("native fit Lanczos workspace dimensions are invalid");
    std::vector<int> host_offsets(static_cast<std::size_t>(n)+1,0);
    for(const auto& edge:graph.edges){if(edge.i<0||edge.i>=n||edge.j<0||edge.j>=n||edge.group<0||6ULL*static_cast<std::size_t>(edge.group)+5>=parameter_count)throw std::runtime_error("native fit Lanczos graph edge is invalid");++host_offsets[edge.i+1];++host_offsets[edge.j+1];}
    for(int i=0;i<n;++i)host_offsets[i+1]+=host_offsets[i];
    std::vector<int> host_neighbors(graph.edges.size()*2),host_groups(graph.edges.size()*2),cursor=host_offsets;
    std::vector<signed char> host_signs(graph.edges.size()*2);
    for(const auto& edge:graph.edges){int at=cursor[edge.i]++;host_neighbors[at]=edge.j;host_groups[at]=edge.group;host_signs[at]=1;at=cursor[edge.j]++;host_neighbors[at]=edge.i;host_groups[at]=edge.group;host_signs[at]=-1;}
    std::size_t free_bytes=0,total_bytes=0;if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)throw std::runtime_error("native fit Lanczos workspace cudaMemGetInfo failed");
    const std::size_t basis_bytes=static_cast<std::size_t>(d)*steps*sizeof(double);
    const std::size_t vector_bytes=(7ULL*static_cast<std::size_t>(d)+steps+std::max(1,p)+static_cast<std::size_t>(n)+3)*sizeof(double);
    const std::size_t graph_bytes=(host_offsets.size()+std::max<std::size_t>(1,host_neighbors.size())+std::max<std::size_t>(1,host_groups.size()))*sizeof(int)+std::max<std::size_t>(1,host_signs.size())*sizeof(signed char);
    if(basis_bytes>free_bytes||vector_bytes>free_bytes-basis_bytes||graph_bytes>free_bytes-basis_bytes-vector_bytes||free_bytes-basis_bytes-vector_bytes-graph_bytes<128ULL*1024*1024)
      throw std::runtime_error("native fit Lanczos workspace preflight leaves less than 128 MiB safety margin");
    auto alloc_double=[](double** pointer,const std::size_t count){if(cudaMalloc(reinterpret_cast<void**>(pointer),std::max<std::size_t>(1,count)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native fit Lanczos GPU allocation failed");};
    alloc_double(&basis,static_cast<std::size_t>(d)*steps);alloc_double(&v,d);alloc_double(&previous,d);alloc_double(&w,d);alloc_double(&add,d);alloc_double(&ritz,d);alloc_double(&coefficient,steps);alloc_double(&theta,parameter_count);alloc_double(&sqrt_mass,d);alloc_double(&sqrt_atom,n);alloc_double(&translation,3);
    if(cudaMalloc(reinterpret_cast<void**>(&offsets),host_offsets.size()*sizeof(int))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&neighbors),std::max<std::size_t>(1,host_neighbors.size())*sizeof(int))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&groups),std::max<std::size_t>(1,host_groups.size())*sizeof(int))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&signs),std::max<std::size_t>(1,host_signs.size())*sizeof(signed char))!=cudaSuccess)
      throw std::runtime_error("native fit Lanczos graph allocation failed");
    if(cudaMemcpy(offsets,host_offsets.data(),host_offsets.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(sqrt_mass,sqrt_mass_host.data(),sqrt_mass_host.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(sqrt_atom,sqrt_atom_host.data(),sqrt_atom_host.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw std::runtime_error("native fit Lanczos static data upload failed");
    if(!host_neighbors.empty()&&(cudaMemcpy(neighbors,host_neighbors.data(),host_neighbors.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(groups,host_groups.data(),host_groups.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(signs,host_signs.data(),host_signs.size()*sizeof(signed char),cudaMemcpyHostToDevice)!=cudaSuccess))
      throw std::runtime_error("native fit Lanczos graph upload failed");
    inverse_mass_sum=1.0/std::inner_product(sqrt_atom_host.begin(),sqrt_atom_host.end(),sqrt_atom_host.begin(),0.0);
    if(!std::isfinite(inverse_mass_sum)||!(inverse_mass_sum>0.0))throw std::runtime_error("native fit Lanczos translation norm is invalid");
  }
  void set_theta(const std::vector<double>& values){if(values.size()!=static_cast<std::size_t>(p))throw std::runtime_error("native fit Lanczos parameter dimension changed");if(p&&cudaMemcpy(theta,values.data(),values.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit Lanczos parameter upload failed");}
  void project(double* x){lanczos_translation_coefficients<<<3,256>>>(x,sqrt_atom,translation,n,inverse_mass_sum);lanczos_remove_translation<<<static_cast<unsigned>((static_cast<std::size_t>(d)+255)/256),256>>>(x,sqrt_atom,translation,d,n);if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("native fit Lanczos translation projection kernel failed");}
  void additive_action(const double* input){lanczos_additive_action<<<static_cast<unsigned>((static_cast<std::size_t>(d)+255)/256),256>>>(input,add,sqrt_mass,theta,offsets,neighbors,groups,signs,n,d);if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("native fit Lanczos additive-action kernel failed");project(add);}
};

struct DeviceQR
{
  cusolverDnHandle_t solver=nullptr;double* a=nullptr;double* y=nullptr;double* tau=nullptr;double* work=nullptr;int* info=nullptr;
  int rows=0,input_rows=0,cols=0,lwork=0;
  void release(){if(solver){cusolverDnDestroy(solver);solver=nullptr;}if(a){cudaFree(a);a=nullptr;}if(y){cudaFree(y);y=nullptr;}if(tau){cudaFree(tau);tau=nullptr;}if(work){cudaFree(work);work=nullptr;}if(info){cudaFree(info);info=nullptr;}}
  ~DeviceQR(){release();}
  void initialize(int m,int p){input_rows=m;rows=std::max(m,p);cols=p;std::size_t free_bytes=0,total_bytes=0;const std::uint64_t base_bytes=static_cast<std::uint64_t>(rows)*p*sizeof(double)+static_cast<std::uint64_t>(rows)*sizeof(double)+static_cast<std::uint64_t>(p)*sizeof(double)+sizeof(int);if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess||base_bytes>free_bytes||free_bytes-base_bytes<256ULL*1024*1024)throw std::runtime_error("native fit QR input workspace preflight leaves less than 256 MiB safety margin");if(cusolverDnCreate(&solver)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit cusolver create failed");
    if(cudaMalloc(reinterpret_cast<void**>(&a),static_cast<std::size_t>(rows)*p*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&y),static_cast<std::size_t>(rows)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&tau),static_cast<std::size_t>(p)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&info),sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit QR GPU allocation failed");
    if(cusolverDnDgeqrf_bufferSize(solver,rows,p,a,rows,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0)throw std::runtime_error("native fit QR workspace query failed");
    int ormqr_work=0;if(cusolverDnDormqr_bufferSize(solver,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,rows,1,p,a,rows,tau,y,rows,&ormqr_work)!=CUSOLVER_STATUS_SUCCESS||ormqr_work<=0)throw std::runtime_error("native fit Q transpose workspace query failed");lwork=std::max(lwork,ormqr_work);
    if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess||static_cast<std::uint64_t>(lwork)*sizeof(double)>free_bytes||free_bytes-static_cast<std::uint64_t>(lwork)*sizeof(double)<256ULL*1024*1024)throw std::runtime_error("native fit QR library workspace preflight leaves less than 256 MiB safety margin");if(cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(lwork)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native fit QR workspace allocation failed");}
  void compress(const std::vector<double>& colmajor,const std::vector<double>& rhs,std::vector<double>& r,std::vector<double>& z,double& discarded2){
    if(cudaMemset(a,0,static_cast<std::size_t>(rows)*cols*sizeof(double))!=cudaSuccess||cudaMemset(y,0,static_cast<std::size_t>(rows)*sizeof(double))!=cudaSuccess||
       cudaMemcpy2D(a,static_cast<std::size_t>(rows)*sizeof(double),colmajor.data(),static_cast<std::size_t>(input_rows)*sizeof(double),static_cast<std::size_t>(input_rows)*sizeof(double),cols,cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(y,rhs.data(),rhs.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit QR block upload failed");
    if(cusolverDnDgeqrf(solver,rows,cols,a,rows,tau,work,lwork,info)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit block QR failed");int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("native fit block QR numerical failure");
    if(cusolverDnDormqr(solver,CUBLAS_SIDE_LEFT,CUBLAS_OP_T,rows,1,cols,a,rows,tau,y,rows,work,lwork,info)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit Q transpose apply failed");if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("native fit Q transpose numerical failure");
    std::vector<double> tail(static_cast<std::size_t>(rows-cols));if(!tail.empty()&&cudaMemcpy(tail.data(),y+cols,tail.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit QR residual read failed");for(double v:tail)discarded2+=v*v;if(!std::isfinite(discarded2))throw std::runtime_error("native fit block QR residual is non-finite");
    std::vector<double> packed(static_cast<std::size_t>(cols)*cols),head(cols),top(static_cast<std::size_t>(cols)*cols);if(cudaMemcpy2D(top.data(),static_cast<std::size_t>(cols)*sizeof(double),a,static_cast<std::size_t>(rows)*sizeof(double),static_cast<std::size_t>(cols)*sizeof(double),cols,cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit QR factor read failed");for(int i=0;i<cols;++i)for(int j=0;j<cols;++j)packed[static_cast<std::size_t>(i)*cols+j]=i<=j?top[static_cast<std::size_t>(j)*cols+i]:0.0;
    if(cudaMemcpy(head.data(),y,static_cast<std::size_t>(cols)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit QR rhs read failed");r=std::move(packed);z=std::move(head);
  }
};

struct SmallSVD
{
  int p=0;std::vector<double> singular,u,vt,eta;
};

SmallSVD svd_small(cusolverDnHandle_t solver,const std::vector<double>& rowmajor,const std::vector<double>& rhs,const int p)
{
  if(!std::all_of(rowmajor.begin(),rowmajor.end(),[](double v){return std::isfinite(v);})||!std::all_of(rhs.begin(),rhs.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit SVD input is non-finite");
  SmallSVD out;out.p=p;std::vector<double> a(rowmajor.size());for(int i=0;i<p;++i)for(int j=0;j<p;++j)a[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*p]=rowmajor[static_cast<std::size_t>(i)*p+j];
  double *da=nullptr,*ds=nullptr,*du=nullptr,*dvt=nullptr,*work=nullptr,*rwork=nullptr;int* info=nullptr;int lwork=0;
  auto cleanup=[&](){if(da)cudaFree(da);if(ds)cudaFree(ds);if(du)cudaFree(du);if(dvt)cudaFree(dvt);if(work)cudaFree(work);if(rwork)cudaFree(rwork);if(info)cudaFree(info);};
  try{
    if(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&ds),static_cast<std::size_t>(p)*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&du),a.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&dvt),a.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&rwork),static_cast<std::size_t>(5*p)*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&info),sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit small-SVD allocation failed");
    if(cusolverDnDgesvd_bufferSize(solver,p,p,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0||cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(lwork)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native fit small-SVD workspace query/allocation failed");
    if(cudaMemcpy(da,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||cusolverDnDgesvd(solver,'A','A',p,p,da,p,ds,du,p,dvt,p,work,lwork,rwork,info)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit small SVD failed");
    int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("native fit small SVD did not converge");
    out.singular.resize(p);out.u.resize(a.size());out.vt.resize(a.size());if(cudaMemcpy(out.singular.data(),ds,p*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out.u.data(),du,a.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out.vt.data(),dvt,a.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit small SVD readback failed");
    if(!std::all_of(out.singular.begin(),out.singular.end(),[](double v){return std::isfinite(v);})||!std::all_of(out.u.begin(),out.u.end(),[](double v){return std::isfinite(v);})||!std::all_of(out.vt.begin(),out.vt.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit small SVD returned non-finite factors");const double top=out.singular.front();if(!(top>0.0)||out.singular.back()<=top*1e-12)throw std::runtime_error("native fit parameter family is rank deficient on training frames");
    out.eta.assign(p,0.0);for(int k=0;k<p;++k)for(int i=0;i<p;++i)out.eta[k]+=out.u[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*p]*rhs[i];
  }catch(...){cleanup();throw;}cleanup();return out;
}

std::vector<double> theta_from_eta(const SmallSVD& svd,const std::vector<double>& xi)
{
  std::vector<double> theta(svd.p,0.0);for(int k=0;k<svd.p;++k){const double c=xi[k]/svd.singular[k];for(int i=0;i<svd.p;++i)theta[i]+=svd.vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(i)*svd.p]*c;}return theta;
}

struct NeumaierAccumulator
{
  double sum=0.0,correction=0.0;
  void add(const double value)
  {
    const double next=sum+value;
    if(std::abs(sum)>=std::abs(value))correction+=(sum-next)+value;
    else correction+=(value-next)+sum;
    sum=next;
  }
  void add_product(const double left,const double right,const double sign=1.0)
  {
    const double product=left*right;
    add(sign*product);
    add(sign*std::fma(left,right,-product));
  }
  double value() const { return sum+correction; }
};

double compensated_dot(const std::vector<double>& left,const std::vector<double>& right)
{
  if(left.size()!=right.size())throw std::runtime_error("native fit compensated dot-product dimensions do not match");
  NeumaierAccumulator sum;for(std::size_t i=0;i<left.size();++i)sum.add_product(left[i],right[i]);return sum.value();
}

double compensated_stationarity_residual(const double z,const int component,const std::vector<std::vector<double>>& a,
  const std::vector<double>& lambda,const std::vector<std::size_t>* active=nullptr)
{
  NeumaierAccumulator sum;sum.add(z);
  for(std::size_t i=0;i<lambda.size();++i){const std::size_t row=active?(*active)[i]:i;sum.add_product(a[row][component],lambda[i],-1.0);}
  return sum.value();
}

double compensated_constraint_residual(const std::vector<double>& row,const std::vector<double>& z,const double rhs)
{
  NeumaierAccumulator sum;sum.add(rhs);for(std::size_t k=0;k<row.size();++k)sum.add_product(row[k],z[k],-1.0);return sum.value();
}

double qp_equality_tolerance(const std::vector<double>& row,const std::vector<double>& z,const double rhs,double& scale)
{
  scale=std::max(1.0,std::abs(rhs));for(std::size_t k=0;k<row.size();++k)scale+=std::abs(row[k]*z[k]);
  return 256.0*std::numeric_limits<double>::epsilon()*scale;
}

struct QPCertificateSummary
{
  double max_primal_violation=0.0,max_primal_excess=0.0,max_complementarity=0.0,max_stationarity=0.0;
  double minimum_primal_slack=std::numeric_limits<double>::infinity(),minimum_slack_allowed_error=0.0;
  double worst_primal_slack=0.0,worst_primal_allowed_error=0.0;
  double worst_complementarity_value=0.0,worst_complementarity_lambda=0.0,worst_complementarity_primal_slack=0.0,worst_complementarity_dual_slack=0.0;
  double worst_stationarity_residual=0.0,max_lambda=0.0;
  std::size_t minimum_slack_constraint_index=0,worst_primal_excess_index=0,worst_complementarity_index=0,worst_stationarity_index=0,max_lambda_index=0;
  bool finite=true,diagnostic_finite=true,multipliers_nonnegative=true,primal_constraints_pass=true,accepted=false;
};

struct QPCertificate:QPCertificateSummary
{
  std::vector<double> theta,primal_slack,allowed_error,dual_slack;
};

constexpr double qp_complementarity_tolerance=1e-7,qp_stationarity_tolerance=1e-9;
constexpr double stability_cut_guard_fraction=1.0e-3;

struct QPPolishDiagnostic
{
  int iteration=0;
  std::size_t active_constraints=0,updates=0;
  std::string failure_reason="none";
  bool certificate_computed=false;
  QPCertificateSummary certificate;
};

void write_qp_certificate_fields(std::ostream& out,const QPCertificateSummary& certificate,const double primal_tolerance)
{
  out<<"finite="<<(certificate.finite?1:0)
    <<" diagnostic_finite="<<(certificate.diagnostic_finite?1:0)
    <<" primal_pass="<<((certificate.primal_constraints_pass&&certificate.max_primal_violation<=primal_tolerance)?1:0)
    <<" multipliers_pass="<<(certificate.multipliers_nonnegative?1:0)
    <<" complementarity_pass="<<((std::isfinite(certificate.worst_complementarity_value)&&certificate.max_complementarity<=qp_complementarity_tolerance)?1:0)
    <<" stationarity_pass="<<((std::isfinite(certificate.worst_stationarity_residual)&&certificate.max_stationarity<=qp_stationarity_tolerance)?1:0)
    <<" accepted="<<(certificate.accepted?1:0)
    <<" max_primal_violation="<<certificate.max_primal_violation
    <<" max_primal_excess="<<certificate.max_primal_excess
    <<" worst_primal_constraint="<<certificate.worst_primal_excess_index
    <<" worst_primal_slack="<<certificate.worst_primal_slack
    <<" worst_primal_allowed_error="<<certificate.worst_primal_allowed_error
    <<" max_complementarity="<<certificate.max_complementarity
    <<" worst_complementarity_constraint="<<certificate.worst_complementarity_index
    <<" worst_complementarity_value="<<certificate.worst_complementarity_value
    <<" worst_complementarity_lambda="<<certificate.worst_complementarity_lambda
    <<" worst_complementarity_primal_slack="<<certificate.worst_complementarity_primal_slack
    <<" worst_complementarity_dual_slack="<<certificate.worst_complementarity_dual_slack
    <<" max_stationarity="<<certificate.max_stationarity
    <<" worst_stationarity_component="<<certificate.worst_stationarity_index
    <<" worst_stationarity_residual="<<certificate.worst_stationarity_residual
    <<" max_lambda="<<certificate.max_lambda<<" max_lambda_index="<<certificate.max_lambda_index;
}

void write_uncomputed_qp_certificate_fields(std::ostream& out)
{
  out<<"finite=NOT_COMPUTED diagnostic_finite=NOT_COMPUTED primal_pass=NOT_COMPUTED multipliers_pass=NOT_COMPUTED complementarity_pass=NOT_COMPUTED stationarity_pass=NOT_COMPUTED";
}

QPCertificate check_cut_qp_kkt(const SmallSVD& svd,const std::vector<std::vector<double>>& rows,const std::vector<double>& rhs,
  const std::vector<std::vector<double>>& a,const std::vector<double>& b,const std::vector<double>& lambda,
  const std::vector<double>& xi,const double primal_tolerance)
{
  if(rows.size()!=rhs.size()||rows.size()!=a.size()||rows.size()!=b.size()||rows.size()!=lambda.size()||xi.size()!=static_cast<std::size_t>(svd.p))
    throw std::runtime_error("native fit stability QP certificate dimensions do not match");
  QPCertificate out;out.theta=theta_from_eta(svd,xi);out.primal_slack.resize(rows.size());out.allowed_error.resize(rows.size());out.dual_slack.resize(rows.size());
  std::vector<double> z(static_cast<std::size_t>(svd.p));for(int k=0;k<svd.p;++k)z[k]=xi[k]-svd.eta[k];
  for(std::size_t i=0;i<lambda.size();++i){const double value=lambda[i];if(!std::isfinite(value)||value<0.0){out.finite=false;out.multipliers_nonnegative=false;}
    if(std::isfinite(value)&&value>out.max_lambda){out.max_lambda=value;out.max_lambda_index=i;}}
  bool diagnostic_finite=true,have_worst_complementarity=false,worst_complementarity_nonfinite=false;
  for(std::size_t i=0;i<rows.size();++i){
    if(rows[i].size()!=static_cast<std::size_t>(svd.p)||a[i].size()!=static_cast<std::size_t>(svd.p))throw std::runtime_error("native fit stability QP certificate row width does not match");
    const double primal=std::inner_product(rows[i].begin(),rows[i].end(),out.theta.begin(),0.0)-rhs[i];
    double scale=std::max(1.0,std::abs(rhs[i]));for(int k=0;k<svd.p;++k)scale+=std::abs(rows[i][k]*out.theta[k]);
    if(!std::isfinite(scale))diagnostic_finite=false;
    const double allowed=std::min(primal_tolerance,256.0*std::numeric_limits<double>::epsilon()*scale);
    const double slack=-compensated_constraint_residual(a[i],z,b[i]),complementarity=lambda[i]*slack;
    out.primal_slack[i]=primal;out.allowed_error[i]=allowed;out.dual_slack[i]=slack;
    if(!std::isfinite(primal)||!std::isfinite(allowed)){out.finite=false;out.primal_constraints_pass=false;}
    if(!std::isfinite(slack)||!std::isfinite(complementarity))out.finite=false;
    out.max_primal_violation=std::max(out.max_primal_violation,std::max(0.0,-primal));
    const double excess=std::max(0.0,-primal-allowed);
    if(i==0||excess>out.max_primal_excess){out.worst_primal_excess_index=i;out.worst_primal_slack=primal;out.worst_primal_allowed_error=allowed;}
    if(excess>out.max_primal_excess)out.max_primal_excess=excess;
    out.max_complementarity=std::max(out.max_complementarity,std::abs(complementarity));
    const double complementarity_abs=std::abs(complementarity);
    if(!std::isfinite(complementarity_abs)){
      if(!worst_complementarity_nonfinite){out.worst_complementarity_index=i;out.worst_complementarity_value=complementarity;out.worst_complementarity_lambda=lambda[i];
        out.worst_complementarity_primal_slack=primal;out.worst_complementarity_dual_slack=slack;have_worst_complementarity=true;worst_complementarity_nonfinite=true;}
    }else if(!worst_complementarity_nonfinite&&(!have_worst_complementarity||complementarity_abs>std::abs(out.worst_complementarity_value))){
      out.worst_complementarity_index=i;out.worst_complementarity_value=complementarity;out.worst_complementarity_lambda=lambda[i];
      out.worst_complementarity_primal_slack=primal;out.worst_complementarity_dual_slack=slack;have_worst_complementarity=true;}
    if(i==0||primal<out.minimum_primal_slack){out.minimum_slack_constraint_index=i;out.minimum_primal_slack=primal;out.minimum_slack_allowed_error=allowed;}
    if(primal < -allowed)out.primal_constraints_pass=false;
  }
  bool have_worst_stationarity=false,worst_stationarity_nonfinite=false;
  for(int k=0;k<svd.p;++k){const double stationarity=compensated_stationarity_residual(z[k],k,a,lambda);if(!std::isfinite(stationarity))out.finite=false;out.max_stationarity=std::max(out.max_stationarity,std::abs(stationarity));
    if(!std::isfinite(stationarity)){if(!worst_stationarity_nonfinite){out.worst_stationarity_index=k;out.worst_stationarity_residual=stationarity;have_worst_stationarity=true;worst_stationarity_nonfinite=true;}}
    else if(!worst_stationarity_nonfinite&&(!have_worst_stationarity||std::abs(stationarity)>std::abs(out.worst_stationarity_residual))){out.worst_stationarity_index=k;out.worst_stationarity_residual=stationarity;have_worst_stationarity=true;}}
  out.diagnostic_finite=out.finite&&diagnostic_finite;
  out.accepted=out.finite&&out.multipliers_nonnegative&&out.primal_constraints_pass&&out.max_primal_violation<=primal_tolerance&&
    out.max_complementarity<=qp_complementarity_tolerance&&out.max_stationarity<=qp_stationarity_tolerance;
  return out;
}

struct QPSolution
{
  std::vector<double> xi,lambda;
  QPCertificate certificate;
  int iterations=0;
  double last_multiplier_change=0.0;
  double coordinate_seconds=0.0,polish_seconds=0.0;
  const char* method="coordinate";
  int polish_attempts=0;
  std::size_t polish_updates=0;
  std::size_t active_constraints=0;
  std::string polish_failure_reason="none";
  std::vector<QPPolishDiagnostic> polish_diagnostics;
};

void write_cut_qp_state(const std::string& path,const int outer,const QPSolution& solution,const double primal_tolerance,
  const SmallSVD& svd,const std::vector<std::vector<double>>& rows,const std::vector<double>& rhs,
  const std::vector<std::vector<double>>& a,const std::vector<double>& b,const std::vector<double>& gram,
  const std::vector<double>& initial_lambda)
{
  if(path.empty())return;
  std::ifstream existing(path,std::ios::binary);if(existing.good())throw std::runtime_error("native fit refuses to overwrite an existing QP state snapshot: "+path);
  std::ofstream out(path,std::ios::out|std::ios::trunc);if(!out)throw std::runtime_error("native fit cannot create QP state snapshot: "+path);
  out<<std::setprecision(17)<<"outer_index "<<outer<<"\niterations "<<solution.iterations<<"\nmethod "<<solution.method
    <<"\npolish_attempts "<<solution.polish_attempts<<"\nactive_constraints "<<solution.active_constraints
    <<"\npolish_updates "<<solution.polish_updates<<"\npolish_failure_reason "<<solution.polish_failure_reason
    <<"\ncoordinate_seconds "<<solution.coordinate_seconds<<"\npolish_seconds "<<solution.polish_seconds
    <<"\nparameters "<<svd.p<<"\nconstraints "<<rows.size()
    <<"\nprimal_tolerance "<<primal_tolerance<<"\ncomplementarity_tolerance "<<qp_complementarity_tolerance<<"\nstationarity_tolerance "<<qp_stationarity_tolerance
    <<"\nlast_multiplier_change "<<solution.last_multiplier_change<<"\nmax_primal_violation "<<solution.certificate.max_primal_violation
    <<"\nmax_complementarity "<<solution.certificate.max_complementarity<<"\nmax_stationarity "<<solution.certificate.max_stationarity
    <<"\nprimal_constraints_pass "<<(solution.certificate.primal_constraints_pass?1:0)<<"\nmax_primal_excess "<<solution.certificate.max_primal_excess
    <<"\nworst_primal_excess_index "<<solution.certificate.worst_primal_excess_index
    <<"\nminimum_slack_constraint_index "<<solution.certificate.minimum_slack_constraint_index<<"\nminimum_primal_slack "<<solution.certificate.minimum_primal_slack
    <<"\nminimum_slack_allowed_error "<<solution.certificate.minimum_slack_allowed_error<<"\nkkt_accepted 0\n";
  auto write_vector=[&](const char* name,const std::vector<double>& values){out<<name<<' '<<values.size();for(double value:values)out<<' '<<value;out<<'\n';};
  write_vector("svd_singular",svd.singular);write_vector("svd_eta",svd.eta);write_vector("initial_lambda",initial_lambda);write_vector("lambda",solution.lambda);write_vector("xi",solution.xi);write_vector("theta",solution.certificate.theta);
  out<<"svd_vt "<<svd.p<<' '<<svd.p<<'\n';for(int i=0;i<svd.p;++i){for(int j=0;j<svd.p;++j)out<<svd.vt[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*svd.p]<<(j+1==svd.p?'\n':' ');}
  out<<"original_rows "<<rows.size()<<' '<<svd.p<<'\n';for(std::size_t i=0;i<rows.size();++i){out<<i;for(double value:rows[i])out<<' '<<value;out<<" rhs "<<rhs[i]<<'\n';}
  out<<"transformed_a "<<a.size()<<' '<<svd.p<<'\n';for(std::size_t i=0;i<a.size();++i){out<<i;for(double value:a[i])out<<' '<<value;out<<" b "<<b[i]<<'\n';}
  out<<"gram "<<rows.size()<<' '<<rows.size()<<'\n';for(std::size_t i=0;i<rows.size();++i){for(std::size_t j=0;j<rows.size();++j)out<<gram[i*rows.size()+j]<<(j+1==rows.size()?'\n':' ');}
  out<<"constraint_state index primal_slack allowed_error primal_excess dual_slack lambda complementarity\n";
  for(std::size_t i=0;i<rows.size();++i)out<<i<<' '<<solution.certificate.primal_slack[i]<<' '<<solution.certificate.allowed_error[i]<<' '<<std::max(0.0,-solution.certificate.primal_slack[i]-solution.certificate.allowed_error[i])<<' '<<solution.certificate.dual_slack[i]<<' '<<solution.lambda[i]<<' '<<solution.lambda[i]*solution.certificate.dual_slack[i]<<'\n';
  out<<"QP_COORDINATE_DIAGNOSTIC iteration="<<solution.iterations<<" certificate_status=COMPUTED ";
  write_qp_certificate_fields(out,solution.certificate,primal_tolerance);out<<'\n';
  for(const auto& diagnostic:solution.polish_diagnostics){
    out<<"QP_POLISH_DIAGNOSTIC iteration="<<diagnostic.iteration<<" failure_reason="<<diagnostic.failure_reason
      <<" active_constraints="<<diagnostic.active_constraints<<" updates="<<diagnostic.updates
      <<" certificate_status="<<(diagnostic.certificate_computed?"COMPUTED":"NOT_COMPUTED")<<' ';
    if(diagnostic.certificate_computed)write_qp_certificate_fields(out,diagnostic.certificate,primal_tolerance);
    else write_uncomputed_qp_certificate_fields(out);
    out<<'\n';
  }
  out.close();if(!out)throw std::runtime_error("native fit failed writing QP state snapshot: "+path);
}

struct ActiveSetPolishResult
{
  std::vector<double> lambda,xi;
  QPCertificate certificate;
  std::size_t active_constraints=0,updates=0;
  bool certificate_computed=false;
  bool accepted=false;
  std::string failure_reason="none";
};

bool solve_active_equalities(cusolverDnHandle_t solver,const std::vector<std::vector<double>>& a,const std::vector<double>& b,
  const std::vector<std::size_t>& active,const int p,const std::vector<double>& eta,
  std::vector<double>& y,std::vector<double>& lambda_star,std::string& failure_reason)
{
  const std::size_t q=active.size();
  if(q==0||p<=0||q>static_cast<std::size_t>(std::numeric_limits<int>::max())||static_cast<std::size_t>(p)>std::numeric_limits<std::size_t>::max()/q){failure_reason="ACTIVE_MATRIX_DIMENSION";return false;}
  if(q>static_cast<std::size_t>(p)){failure_reason="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT";return false;}
  const int n=static_cast<int>(q),rank=std::min(p,n);std::vector<double> matrix(static_cast<std::size_t>(p)*q);
  for(int j=0;j<n;++j){const std::size_t row=active[j];if(row>=a.size()||row>=b.size()||a[row].size()!=static_cast<std::size_t>(p)){failure_reason="ACTIVE_MATRIX_DIMENSION";return false;}for(int i=0;i<p;++i)matrix[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*p]=a[row][i];}
  double *da=nullptr,*ds=nullptr,*du=nullptr,*dvt=nullptr,*work=nullptr,*rwork=nullptr;int* info=nullptr;int lwork=0;
  auto cleanup=[&](){if(da){cudaFree(da);da=nullptr;}if(ds){cudaFree(ds);ds=nullptr;}if(du){cudaFree(du);du=nullptr;}if(dvt){cudaFree(dvt);dvt=nullptr;}if(work){cudaFree(work);work=nullptr;}if(rwork){cudaFree(rwork);rwork=nullptr;}if(info){cudaFree(info);info=nullptr;}};
  try{
    if(cudaMalloc(reinterpret_cast<void**>(&da),matrix.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&ds),static_cast<std::size_t>(rank)*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&du),static_cast<std::size_t>(p)*rank*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&dvt),static_cast<std::size_t>(rank)*q*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&rwork),static_cast<std::size_t>(5)*rank*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&info),sizeof(int))!=cudaSuccess)
      throw std::runtime_error("allocation");
    if(cusolverDnDgesvd_bufferSize(solver,p,n,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0||
       cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(lwork)*sizeof(double))!=cudaSuccess)throw std::runtime_error("workspace");
    if(cudaMemcpy(da,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       cusolverDnDgesvd(solver,'S','S',p,n,da,p,ds,du,p,dvt,rank,work,lwork,rwork,info)!=CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("factorization");
    int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("convergence");
    std::vector<double> singular(rank),u(static_cast<std::size_t>(p)*rank),vt(static_cast<std::size_t>(rank)*q);
    if(cudaMemcpy(singular.data(),ds,singular.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||
       cudaMemcpy(u.data(),du,u.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||
       cudaMemcpy(vt.data(),dvt,vt.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("readback");
    cleanup();
    if(!std::all_of(singular.begin(),singular.end(),[](double x){return std::isfinite(x);})||
       !std::all_of(u.begin(),u.end(),[](double x){return std::isfinite(x);})||
       !std::all_of(vt.begin(),vt.end(),[](double x){return std::isfinite(x);})) {failure_reason="ACTIVE_SVD_NONFINITE";return false;}
    if(!(singular.front()>0.0)||singular.back()<=singular.front()*1e-12){failure_reason="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT";return false;}
    std::vector<double> projected_b(rank,0.0);for(int k=0;k<rank;++k)for(int j=0;j<n;++j)projected_b[k]+=vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*rank]*b[active[j]];
    y.assign(p,0.0);lambda_star.assign(q,0.0);
    for(int k=0;k<rank;++k){const double scaled=projected_b[k]/singular[k];for(int i=0;i<p;++i)y[i]+=u[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*p]*scaled;}
    for(int j=0;j<n;++j)for(int k=0;k<rank;++k)lambda_star[j]+=vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*rank]*(projected_b[k]/singular[k])/singular[k];
    if(!std::all_of(y.begin(),y.end(),[](double x){return std::isfinite(x);})||
       !std::all_of(lambda_star.begin(),lambda_star.end(),[](double x){return std::isfinite(x);})) {failure_reason="ACTIVE_SVD_NONFINITE";return false;}
    if(std::all_of(lambda_star.begin(),lambda_star.end(),[](double x){return x>=0.0;})){
      // Reuse this SVD to jointly reduce active equalities, complementarity, and stationarity.
      std::vector<double> xi(p),z(p),stationarity(p),equality(q),equality_tolerance(q),h(rank),delta_y(p),delta_lambda(q),trial_y(p),trial_lambda(q),best_y,best_lambda;
      bool have_best=false;double best_score=std::numeric_limits<double>::infinity();
      for(int pass=0;pass<=3;++pass){
        if(!std::all_of(y.begin(),y.end(),[](double x){return std::isfinite(x);})||
           !std::all_of(lambda_star.begin(),lambda_star.end(),[](double x){return std::isfinite(x)&&x>=0.0;}))break;
        for(int k=0;k<p;++k){xi[k]=eta[k]+y[k];z[k]=xi[k]-eta[k];}
        if(!std::all_of(xi.begin(),xi.end(),[](double x){return std::isfinite(x);})||!std::all_of(z.begin(),z.end(),[](double x){return std::isfinite(x);}))break;
        double max_complementarity=0.0,max_stationarity=0.0,max_scaled_equality=0.0;bool residuals_finite=true;
        for(int k=0;k<p;++k){stationarity[k]=compensated_stationarity_residual(z[k],k,a,lambda_star,&active);if(!std::isfinite(stationarity[k]))residuals_finite=false;else max_stationarity=std::max(max_stationarity,std::abs(stationarity[k]));}
        for(int j=0;j<n;++j){const std::size_t row=active[j];equality[j]=compensated_constraint_residual(a[row],z,b[row]);double scale=0.0;
          equality_tolerance[j]=qp_equality_tolerance(a[row],z,b[row],scale);
          const double product=lambda_star[j]*equality[j];
          if(!std::isfinite(equality[j])||!std::isfinite(scale)||!std::isfinite(equality_tolerance[j])||!(equality_tolerance[j]>0.0)||!std::isfinite(product)){residuals_finite=false;continue;}
          max_complementarity=std::max(max_complementarity,std::abs(product));max_scaled_equality=std::max(max_scaled_equality,std::abs(equality[j])/equality_tolerance[j]);}
        if(!residuals_finite)break;
        const double score=std::max({max_complementarity/qp_complementarity_tolerance,max_stationarity/qp_stationarity_tolerance,max_scaled_equality});
        if(!have_best||score<best_score){best_score=score;best_y=y;best_lambda=lambda_star;have_best=true;}
        if(score<=1.0||pass==3)break;
        bool correction_finite=true;
        for(int k=0;k<rank;++k){NeumaierAccumulator projected_c,projected_s;
          for(int j=0;j<n;++j)projected_c.add_product(vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*rank],equality[j]);
          for(int i=0;i<p;++i)projected_s.add_product(u[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*p],stationarity[i]);
          h[k]=projected_c.value()/singular[k]+projected_s.value();if(!std::isfinite(h[k]))correction_finite=false;}
        if(!correction_finite)break;
        for(int i=0;i<p;++i){NeumaierAccumulator correction;correction.add(-stationarity[i]);for(int k=0;k<rank;++k)correction.add_product(u[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*p],h[k]);delta_y[i]=correction.value();if(!std::isfinite(delta_y[i]))correction_finite=false;}
        for(int j=0;j<n;++j){NeumaierAccumulator correction;for(int k=0;k<rank;++k)correction.add_product(vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*rank],h[k]/singular[k]);delta_lambda[j]=correction.value();if(!std::isfinite(delta_lambda[j]))correction_finite=false;}
        if(!correction_finite)break;
        for(int i=0;i<p;++i){NeumaierAccumulator updated;updated.add(y[i]);updated.add(delta_y[i]);trial_y[i]=updated.value();if(!std::isfinite(trial_y[i]))correction_finite=false;}
        for(int j=0;j<n;++j){NeumaierAccumulator updated;updated.add(lambda_star[j]);updated.add(delta_lambda[j]);trial_lambda[j]=updated.value();if(!std::isfinite(trial_lambda[j])||trial_lambda[j]<0.0)correction_finite=false;}
        if(!correction_finite)break;
        y=trial_y;lambda_star=trial_lambda;
      }
      if(!have_best){failure_reason="ACTIVE_SVD_NONFINITE";return false;}
      y=std::move(best_y);lambda_star=std::move(best_lambda);
    }
    std::vector<double> xi(p),z(p);for(int k=0;k<p;++k){xi[k]=eta[k]+y[k];z[k]=xi[k]-eta[k];}
    for(int j=0;j<n;++j){const std::size_t row=active[j];const double residual=compensated_constraint_residual(a[row],z,b[row]);double scale=0.0;const double tolerance=qp_equality_tolerance(a[row],z,b[row],scale);if(!std::isfinite(residual)||!std::isfinite(scale)||!std::isfinite(tolerance)||std::abs(residual)>tolerance){failure_reason="ACTIVE_EQUALITY_UNRELIABLE";return false;}}
    return true;
  }catch(...){cleanup();failure_reason="ACTIVE_SVD_FAILURE";return false;}
}

ActiveSetPolishResult polish_cut_qp(cusolverDnHandle_t solver,const SmallSVD& svd,const std::vector<std::vector<double>>& rows,
  const std::vector<double>& rhs,const std::vector<std::vector<double>>& a,const std::vector<double>& b,
  const std::vector<double>& initial_lambda,const double primal_tolerance)
{
  ActiveSetPolishResult out;out.lambda=initial_lambda;const std::size_t m=rows.size();
  if(initial_lambda.size()!=m){out.failure_reason="ACTIVE_INITIAL_MULTIPLIER_DIMENSION";return out;}
  if(m>std::numeric_limits<std::size_t>::max()-static_cast<std::size_t>(svd.p)||(m+static_cast<std::size_t>(svd.p))>std::numeric_limits<std::size_t>::max()/4){out.failure_reason="ACTIVE_UPDATE_LIMIT_OVERFLOW";return out;}
  const std::size_t update_limit=4*(m+static_cast<std::size_t>(svd.p));std::vector<std::size_t> active;std::vector<char> is_active(m,0);bool rank_restart_used=false;
  for(std::size_t i=0;i<m;++i){if(!std::isfinite(out.lambda[i])||out.lambda[i]<0.0){out.failure_reason="ACTIVE_INITIAL_MULTIPLIER_INVALID";return out;}if(out.lambda[i]>0.0){active.push_back(i);is_active[i]=1;}}
  for(;;){out.certificate_computed=false;out.active_constraints=active.size();std::vector<double> candidate_y(static_cast<std::size_t>(svd.p),0.0),lambda_star;
    if(!active.empty()&&!solve_active_equalities(solver,a,b,active,svd.p,svd.eta,candidate_y,lambda_star,out.failure_reason)){
      if(rank_restart_used||out.failure_reason!="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT")return out;
      // Rebuild a dependent working set once; retain every cut for the full KKT check.
      if(out.updates>update_limit-active.size()){out.failure_reason="ACTIVE_UPDATE_LIMIT";return out;}
      out.updates+=active.size();active.clear();std::fill(out.lambda.begin(),out.lambda.end(),0.0);std::fill(is_active.begin(),is_active.end(),0);
      rank_restart_used=true;continue;
    }
    bool has_negative=false;for(double value:lambda_star)has_negative=has_negative||value<0.0;
    if(has_negative){double alpha=1.0;std::vector<double> ratios(active.size(),std::numeric_limits<double>::infinity());
      for(std::size_t j=0;j<active.size();++j)if(lambda_star[j]<0.0){const double current=out.lambda[active[j]],denominator=current-lambda_star[j];if(!(current>0.0)||!(denominator>0.0)||!std::isfinite(denominator)){out.failure_reason="ACTIVE_NEGATIVE_MULTIPLIER_STEP_INVALID";return out;}ratios[j]=current/denominator;if(!std::isfinite(ratios[j])||ratios[j]<0.0||ratios[j]>1.0){out.failure_reason="ACTIVE_NEGATIVE_MULTIPLIER_STEP_INVALID";return out;}alpha=std::min(alpha,ratios[j]);}
      std::vector<std::size_t> remaining,leaving;remaining.reserve(active.size());
      for(std::size_t j=0;j<active.size();++j){const std::size_t row=active[j];
        if(lambda_star[j]<0.0&&ratios[j]<=alpha+64.0*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(alpha))){out.lambda[row]=0.0;is_active[row]=0;leaving.push_back(row);}
        else{const double updated=out.lambda[row]+alpha*(lambda_star[j]-out.lambda[row]);if(!std::isfinite(updated)||updated<0.0){out.failure_reason="ACTIVE_NEGATIVE_MULTIPLIER_STEP_INVALID";return out;}out.lambda[row]=updated;remaining.push_back(row);}}
      if(leaving.empty()||out.updates>update_limit-leaving.size()){out.failure_reason=leaving.empty()?"ACTIVE_NEGATIVE_MULTIPLIER_NO_EXIT":"ACTIVE_UPDATE_LIMIT";return out;}
      out.updates+=leaving.size();active=std::move(remaining);continue;
    }
    std::fill(out.lambda.begin(),out.lambda.end(),0.0);for(std::size_t j=0;j<active.size();++j)out.lambda[active[j]]=lambda_star[j];
    std::vector<std::size_t> remaining;remaining.reserve(active.size());for(std::size_t j=0;j<active.size();++j){const std::size_t row=active[j];if(lambda_star[j]==0.0){is_active[row]=0;if(out.updates==update_limit){out.failure_reason="ACTIVE_UPDATE_LIMIT";return out;}++out.updates;}else remaining.push_back(row);}active=std::move(remaining);
    // Keep the SVD primal solution; large multipliers amplify reconstruction roundoff.
    out.xi=svd.eta;for(int k=0;k<svd.p;++k)out.xi[k]+=candidate_y[k];
    out.certificate=check_cut_qp_kkt(svd,rows,rhs,a,b,out.lambda,out.xi,primal_tolerance);out.certificate_computed=true;if(out.certificate.accepted){out.accepted=true;out.failure_reason="none";out.active_constraints=active.size();return out;}
    std::size_t worst=m;double worst_excess=0.0;
    for(std::size_t i=0;i<m;++i){const double excess=std::max(0.0,-out.certificate.primal_slack[i]-out.certificate.allowed_error[i]);if(excess<=0.0)continue;if(is_active[i]){out.failure_reason="ACTIVE_EQUALITY_KKT_MISMATCH";return out;}if(excess>worst_excess){worst=i;worst_excess=excess;}}
    if(worst<m){if(out.updates==update_limit){out.failure_reason="ACTIVE_UPDATE_LIMIT";return out;}active.push_back(worst);is_active[worst]=1;++out.updates;continue;}
    out.failure_reason=out.certificate.finite?"ACTIVE_FULL_KKT_NOT_ACCEPTED":"ACTIVE_KKT_NONFINITE";return out;
  }
}

QPSolution solve_cut_qp(cusolverDnHandle_t solver,const SmallSVD& svd,const std::vector<std::vector<double>>& rows,const std::vector<double>& rhs,
  const double primal_tolerance=1e-8,const std::vector<double>& initial_lambda={},const std::string& qp_state_path={},const int outer=-1)
{
  if(!std::isfinite(primal_tolerance)||primal_tolerance<0.0)throw std::runtime_error("native fit stability QP has an invalid primal tolerance");
  std::vector<std::vector<double>> a(rows.size(),std::vector<double>(svd.p));std::vector<double> b=rhs;
  for(std::size_t c=0;c<rows.size();++c){for(int k=0;k<svd.p;++k){for(int j=0;j<svd.p;++j)a[c][k]+=rows[c][j]*svd.vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*svd.p]/svd.singular[k];}if(!std::all_of(a[c].begin(),a[c].end(),[](double x){return std::isfinite(x);})||!std::isfinite(b[c]))throw std::runtime_error("native fit stability cut is non-finite");for(int k=0;k<svd.p;++k)b[c]-=a[c][k]*svd.eta[k];}
  std::vector<double> gram(rows.size()*rows.size()),lambda(rows.size(),0.0);if(initial_lambda.size()>rows.size())throw std::runtime_error("native fit stability QP warm start has more multipliers than constraints");std::copy(initial_lambda.begin(),initial_lambda.end(),lambda.begin());
  for(std::size_t i=0;i<rows.size();++i)for(std::size_t j=0;j<rows.size();++j){gram[i*rows.size()+j]=std::inner_product(a[i].begin(),a[i].end(),a[j].begin(),0.0);if(!std::isfinite(gram[i*rows.size()+j]))throw std::runtime_error("native fit stability QP Gram matrix is non-finite");}
  std::vector<double> diag(rows.size());for(std::size_t i=0;i<rows.size();++i){diag[i]=gram[i*rows.size()+i];if(!(diag[i]>1e-24)||!std::isfinite(diag[i]))throw std::runtime_error("native fit stability cut is singular or non-finite");}
  for(double value:lambda)if(!std::isfinite(value)||value<0.0)throw std::runtime_error("native fit stability QP warm start is non-finite or negative");
  const std::vector<double> initial_lambda_full=lambda;QPSolution result;result.lambda=std::move(lambda);constexpr int check_interval=32,max_iterations=200000,first_polish_iteration=64,retry_polish_iteration=4096;
  auto check_current=[&](){result.xi=svd.eta;for(std::size_t i=0;i<rows.size();++i)for(int k=0;k<svd.p;++k)result.xi[k]+=a[i][k]*result.lambda[i];result.certificate=check_cut_qp_kkt(svd,rows,rhs,a,b,result.lambda,result.xi,primal_tolerance);};
  auto coordinate_started=std::chrono::steady_clock::now();
  auto record_coordinate_time=[&](){const auto now=std::chrono::steady_clock::now();result.coordinate_seconds+=std::chrono::duration<double>(now-coordinate_started).count();coordinate_started=now;};
  auto try_polish=[&](){++result.polish_attempts;ActiveSetPolishResult polished;try{polished=polish_cut_qp(solver,svd,rows,rhs,a,b,result.lambda,primal_tolerance);}catch(...){polished.failure_reason="ACTIVE_INTERNAL_FAILURE";}
    QPPolishDiagnostic diagnostic;diagnostic.iteration=result.iterations;diagnostic.active_constraints=polished.active_constraints;diagnostic.updates=polished.updates;
    diagnostic.failure_reason=polished.failure_reason;diagnostic.certificate_computed=polished.certificate_computed;
    if(diagnostic.certificate_computed)diagnostic.certificate=static_cast<const QPCertificateSummary&>(polished.certificate);
    result.polish_diagnostics.push_back(std::move(diagnostic));
    result.active_constraints=polished.active_constraints;result.polish_updates+=polished.updates;result.polish_failure_reason=polished.failure_reason;
    if(!polished.accepted)return false;result.lambda=std::move(polished.lambda);result.xi=std::move(polished.xi);result.certificate=std::move(polished.certificate);result.method="active_set_polish";return true;};
  for(int it=0;it<max_iterations;++it){double change=0.0;for(std::size_t i=0;i<rows.size();++i){double residual=b[i];for(std::size_t j=0;j<rows.size();++j)residual-=gram[i*rows.size()+j]*result.lambda[j];const double next=std::max(0.0,result.lambda[i]+residual/diag[i]);if(!std::isfinite(next))throw std::runtime_error("native fit stability QP iterate is non-finite");change=std::max(change,std::abs(next-result.lambda[i]));result.lambda[i]=next;}result.iterations=it+1;result.last_multiplier_change=change;
    if(result.iterations%check_interval==0||result.iterations==max_iterations){check_current();if(result.certificate.accepted)break;
      if((result.iterations==first_polish_iteration||result.iterations==retry_polish_iteration||result.iterations==max_iterations)&&result.polish_attempts<3){record_coordinate_time();const auto polish_started=std::chrono::steady_clock::now();const bool accepted=try_polish();result.polish_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-polish_started).count();coordinate_started=std::chrono::steady_clock::now();if(accepted)break;}}}
  // Recheck the full KKT certificate without overwriting a polished primal solution.
  record_coordinate_time();const auto final_check_started=std::chrono::steady_clock::now();result.certificate=check_cut_qp_kkt(svd,rows,rhs,a,b,result.lambda,result.xi,primal_tolerance);result.coordinate_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-final_check_started).count();
  if(!result.certificate.accepted){write_cut_qp_state(qp_state_path,outer,result,primal_tolerance,svd,rows,rhs,a,b,gram,initial_lambda_full);throw std::runtime_error("native fit stability QP failed primal, dual, or complementary-slackness check after "+std::to_string(result.iterations)+" iterations"+(qp_state_path.empty()?std::string():"; inspect "+qp_state_path));}
  return result;
}

void project_translation(std::vector<double>& v,const std::vector<double>& sqrt_mass,const int n)
{
  const double sum=std::inner_product(sqrt_mass.begin(),sqrt_mass.end(),sqrt_mass.begin(),0.0);
  for(int a=0;a<3;++a){double dot=0.0;for(int i=0;i<n;++i)dot+=sqrt_mass[i]*v[a*n+i];dot/=sum;for(int i=0;i<n;++i)v[a*n+i]-=sqrt_mass[i]*dot;}
}

std::array<double,3> mic_displacement(const std::vector<double>& r, const int n, const int i, const int j,
                                      const Box& box, int image[3])
{
  double dx[3] = {r[j]-r[i], r[n+j]-r[n+i], r[2*n+j]-r[2*n+i]};
  double s[3] = {};
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) s[a] += box.cpu_h[9+3*a+b]*dx[b];
  for (int a=0;a<3;++a) {const double nearest=std::nearbyint(s[a]);if(!std::isfinite(nearest)||nearest>std::numeric_limits<int>::max()||nearest<std::numeric_limits<int>::min()+1.0)throw std::runtime_error("native fit periodic image exceeds integer range");image[a]=-static_cast<int>(nearest);s[a]+=image[a];}
  std::array<double,3> out{};
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) out[a] += box.cpu_h[3*a+b]*s[b];
  return out;
}

Graph make_graph(const std::vector<double>& r, const std::vector<int>& types, const Box& box, const double cutoff)
{
  const int n = static_cast<int>(types.size());
  Graph g; g.sites.resize(n);
  std::array<int,3> bins{};std::map<std::array<int,3>,std::vector<int>> cells;std::vector<std::array<int,3>> atom_bin(n);
  for(int a=0;a<3;++a){double inverse_norm=0.0;for(int b=0;b<3;++b)inverse_norm+=box.cpu_h[9+3*a+b]*box.cpu_h[9+3*a+b];const double reach=cutoff*std::sqrt(inverse_norm);if(!(reach>0.0)||!std::isfinite(reach))throw std::runtime_error("native fit invalid cell-list reciprocal reach");bins[a]=std::max(1,static_cast<int>(std::min<double>(n,std::floor(1.0/reach))));}
  for(int i=0;i<n;++i){double fractional[3]={};for(int a=0;a<3;++a){for(int b=0;b<3;++b)fractional[a]+=box.cpu_h[9+3*a+b]*r[b*n+i];fractional[a]-=std::floor(fractional[a]);atom_bin[i][a]=std::min(bins[a]-1,static_cast<int>(fractional[a]*bins[a]));}cells[atom_bin[i]].push_back(i);}
  for (int i=0;i<n;++i) {
    std::set<int> candidates;
    for(int a=-1;a<=1;++a)for(int b=-1;b<=1;++b)for(int c=-1;c<=1;++c){std::array<int,3> key={
      (atom_bin[i][0]+a+bins[0])%bins[0],(atom_bin[i][1]+b+bins[1])%bins[1],(atom_bin[i][2]+c+bins[2])%bins[2]};
      auto it=cells.find(key);if(it!=cells.end())candidates.insert(it->second.begin(),it->second.end());}
    for (int j:candidates) {if(j<=i)continue;
    int image[3]; const auto dr = mic_displacement(r,n,i,j,box,image);
    if (dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2] >= cutoff*cutoff) continue;
    const auto type_key = std::minmax(types[i], types[j]);
    auto it = g.group_by_type.find(type_key);
    if (it == g.group_by_type.end()) {
      const int id = static_cast<int>(g.type_pairs.size());
      g.group_by_type.emplace(type_key,id); g.type_pairs.push_back(type_key); it = g.group_by_type.find(type_key);
    }
    Edge e; e.i=i; e.j=j; e.group=it->second;
    std::copy(image,image+3,e.image);
    const int edge_index=static_cast<int>(g.edges.size());g.edges.push_back(e);
    SiteEdge a; a.j=j; a.group=e.group; a.edge_index=edge_index; std::copy(image,image+3,a.image); a.sign=1.0; g.sites[i].push_back(a);
    SiteEdge b; b.j=i; b.group=e.group; b.edge_index=edge_index; for(int k=0;k<3;++k)b.image[k]=-image[k]; b.sign=-1.0; g.sites[j].push_back(b);
    }
  }
  if (g.edges.empty()) throw std::runtime_error("native fit cutoff graph has no edges");
  std::vector<char> seen(n); std::vector<int> todo(1,0); seen[0]=1;
  while (!todo.empty()) { const int i=todo.back(); todo.pop_back(); for (const auto& e:g.sites[i]) if(!seen[e.j]) { seen[e.j]=1; todo.push_back(e.j); } }
  if (std::find(seen.begin(),seen.end(),0)!=seen.end()) throw std::runtime_error("native fit cutoff graph is disconnected");
  return g;
}

void make_design(const Graph& graph,const std::vector<double>& q,const std::vector<double>& sqrt_mass,
                 const int n,const int p,std::vector<double>& design)
{
  std::fill(design.begin(),design.end(),0.0);
  for(const auto&e:graph.edges){
    const double dx[3]={q[e.j]/sqrt_mass[e.j]-q[e.i]/sqrt_mass[e.i],
                        q[n+e.j]/sqrt_mass[n+e.j]-q[n+e.i]/sqrt_mass[n+e.i],
                        q[2*n+e.j]/sqrt_mass[2*n+e.j]-q[2*n+e.i]/sqrt_mass[2*n+e.i]};
    const double values[6][3]={{dx[0],0,0},{dx[1],dx[0],0},{dx[2],0,dx[0]},
                               {0,dx[1],0},{0,dx[2],dx[1]},{0,0,dx[2]}};
    for(int a=0;a<6;++a)for(int axis=0;axis<3;++axis){const int col=6*e.group+a;
      design[static_cast<std::size_t>(col)*3*n+axis*n+e.i]+=values[a][axis]/sqrt_mass[axis*n+e.i];
      design[static_cast<std::size_t>(col)*3*n+axis*n+e.j]-=values[a][axis]/sqrt_mass[axis*n+e.j];}
  }
  (void)p;
}

void apply_additive(const Graph& graph,const std::vector<double>& q,const std::vector<double>& sqrt_mass,
                    const std::vector<double>& sqrt_atom,const std::vector<double>& theta,const int n,std::vector<double>& result)
{
  std::fill(result.begin(),result.end(),0.0);
  for(const auto&e:graph.edges){const std::size_t p=6*static_cast<std::size_t>(e.group);
    const double dx[3]={q[e.j]/sqrt_mass[e.j]-q[e.i]/sqrt_mass[e.i],q[n+e.j]/sqrt_mass[n+e.j]-q[n+e.i]/sqrt_mass[n+e.i],q[2*n+e.j]/sqrt_mass[2*n+e.j]-q[2*n+e.i]/sqrt_mass[2*n+e.i]};
    const double y[3]={theta[p]*dx[0]+theta[p+1]*dx[1]+theta[p+2]*dx[2],theta[p+1]*dx[0]+theta[p+3]*dx[1]+theta[p+4]*dx[2],theta[p+2]*dx[0]+theta[p+4]*dx[1]+theta[p+5]*dx[2]};
    for(int a=0;a<3;++a){result[a*n+e.i]-=y[a]/sqrt_mass[a*n+e.i];result[a*n+e.j]+=y[a]/sqrt_mass[a*n+e.j];}}
  project_translation(result,sqrt_atom,n);
}

struct SmallEigen{int n=0;std::vector<double> values,vectors;};
SmallEigen eigen_small(cusolverDnHandle_t solver,const std::vector<double>& rowmajor,const int n,const int requested=-1)
{
  if(!std::all_of(rowmajor.begin(),rowmajor.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit eigensolver input is non-finite");
  const int returned=requested>0?std::min(requested,n):n;
  SmallEigen out;out.n=n;std::vector<double>a(rowmajor.size());for(int i=0;i<n;++i)for(int j=0;j<n;++j)a[i+static_cast<std::size_t>(j)*n]=rowmajor[static_cast<std::size_t>(i)*n+j];
  double *da=nullptr,*dw=nullptr,*work=nullptr;int *info=nullptr,lwork=0;auto cleanup=[&](){if(da)cudaFree(da);if(dw)cudaFree(dw);if(work)cudaFree(work);if(info)cudaFree(info);};
  try{if(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&dw),static_cast<std::size_t>(n)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&info),sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit small-eigen allocation failed");
    if(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,da,n,dw,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0||cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(lwork)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native fit small-eigen workspace failed");
    if(cudaMemcpy(da,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,da,n,dw,work,lwork,info)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit small symmetric eigensolve failed");
    int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("native fit small symmetric eigensolve did not converge");
     out.values.resize(returned);out.vectors.resize(static_cast<std::size_t>(n)*returned);if(cudaMemcpy(out.values.data(),dw,static_cast<std::size_t>(returned)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out.vectors.data(),da,out.vectors.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit small eigen readback failed");if(!std::all_of(out.values.begin(),out.values.end(),[](double v){return std::isfinite(v);})||!std::all_of(out.vectors.begin(),out.vectors.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit small eigensolver returned non-finite values");
  }catch(...){cleanup();throw;}cleanup();return out;
}

struct RitzMode{double value=0.0,actual_rayleigh=0.0,residual=0.0,base_rayleigh=0.0,additive_rayleigh=0.0,curvature_scale=0.0,curvature_tolerance=0.0;int steps_used=0,dense_matrix_actions=0;bool early_exit=false;std::vector<double> vector;};
bool stability_cut_is_violated(const double rayleigh,const double epsilon,const double tolerance){return rayleigh<epsilon-tolerance;}
std::vector<double> spectral_row(const Graph& graph,const std::vector<double>& v,const std::vector<double>& sqrt_mass,const int n,const int p);
double curvature_rounding_tolerance(double epsilon,double base_rayleigh,const std::vector<double>& row,const std::vector<double>& theta,double& scale);
std::vector<RitzMode> lanczos_low_modes(cusolverDnHandle_t solver,DeviceBaseline& baseline,const Graph& graph,
  const std::vector<double>& theta,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const int n,const int max_steps,const int wanted,const double epsilon,
  const std::vector<double>& initial_vector=std::vector<double>{},DeviceLanczosWorkspace* workspace=nullptr)
{
  const int d=3*n;const int steps=std::min(max_steps,d-3);if(steps<=0)throw std::runtime_error("native fit Lanczos has no internal dimensions");
  DeviceLanczosWorkspace local_workspace;if(workspace==nullptr){local_workspace.initialize(graph,sqrt_mass,sqrt_atom,steps,theta.size());workspace=&local_workspace;}
  if(workspace->d!=d||workspace->n!=n||workspace->max_steps<steps||workspace->p!=static_cast<int>(theta.size()))throw std::runtime_error("native fit Lanczos workspace does not match search dimensions");
  auto blas_check=[](const cublasStatus_t status,const char* what){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(what);};
  auto dot=[&](const double* x,const double* y){double value=0.0;blas_check(cublasDdot(baseline.blas,d,x,1,y,1,&value),"native fit Lanczos dot product failed");return value;};
  auto norm=[&](const double* x){double value=0.0;blas_check(cublasDnrm2(baseline.blas,d,x,1,&value),"native fit Lanczos norm failed");return value;};
  auto axpy=[&](const double scale,const double* x,double* y){blas_check(cublasDaxpy(baseline.blas,d,&scale,x,1,y,1),"native fit Lanczos vector update failed");};
  const double one=1.0,zero=0.0;workspace->set_theta(theta);int dense_matrix_actions=0;
  std::vector<double> initial(d);
  if(initial_vector.empty()){for(int i=0;i<d;++i)initial[i]=std::sin((i+1)*1.6180339887498948)+std::cos((i+1)*0.7548776662466927);}
  else{if(initial_vector.size()!=static_cast<std::size_t>(d))throw std::runtime_error("native fit Lanczos initial vector has an invalid dimension");initial=initial_vector;}
  if(cudaMemcpy(workspace->v,initial.data(),static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)throw std::runtime_error("native fit Lanczos initial vector upload failed");
  workspace->project(workspace->v);double initial_norm=norm(workspace->v);if(!std::isfinite(initial_norm)||!(initial_norm>0.0))throw std::runtime_error("native fit Lanczos initial vector vanished or is non-finite");
  const double inverse_initial_norm=1.0/initial_norm;blas_check(cublasDscal(baseline.blas,d,&inverse_initial_norm,workspace->v,1),"native fit Lanczos initial vector normalization failed");
  std::vector<double> alpha,beta;double beta_prev=0.0;
  auto extract_modes=[&](const int m,const bool early_exit){
    std::vector<double> t(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i){t[static_cast<std::size_t>(i)*m+i]=alpha[i];if(i+1<m)t[static_cast<std::size_t>(i)*m+i+1]=t[static_cast<std::size_t>(i+1)*m+i]=beta[i];}
    const SmallEigen eig=eigen_small(solver,t,m,wanted);std::vector<RitzMode> out;
    for(int mode=0;mode<static_cast<int>(eig.values.size());++mode){RitzMode r;r.value=eig.values[mode];r.steps_used=m;r.early_exit=early_exit;std::vector<double> coefficients(m);for(int k=0;k<m;++k)coefficients[k]=eig.vectors[static_cast<std::size_t>(k)+static_cast<std::size_t>(mode)*m];
      if(cudaMemcpy(workspace->coefficient,coefficients.data(),coefficients.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||cublasDgemv(baseline.blas,CUBLAS_OP_N,d,m,&one,workspace->basis,d,workspace->coefficient,1,&zero,workspace->ritz,1)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit Lanczos Ritz reconstruction failed");
      workspace->project(workspace->ritz);const double rnorm=norm(workspace->ritz);if(!std::isfinite(rnorm)||!(rnorm>0.0))throw std::runtime_error("native fit Lanczos Ritz vector is non-finite or translation-only");const double inverse_rnorm=1.0/rnorm;blas_check(cublasDscal(baseline.blas,d,&inverse_rnorm,workspace->ritz,1),"native fit Lanczos Ritz normalization failed");
      if(baseline.apply_device(workspace->ritz,baseline.out)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit Lanczos Ritz Hessian action failed");++dense_matrix_actions;workspace->additive_action(workspace->ritz);r.base_rayleigh=dot(workspace->ritz,baseline.out);r.additive_rayleigh=dot(workspace->ritz,workspace->add);
      axpy(one,workspace->add,baseline.out);r.actual_rayleigh=dot(workspace->ritz,baseline.out);axpy(-r.actual_rayleigh,workspace->ritz,baseline.out);r.residual=norm(baseline.out);
      r.vector.resize(d);if(cudaMemcpy(r.vector.data(),workspace->ritz,static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit Lanczos Ritz vector readback failed");const auto row=spectral_row(graph,r.vector,sqrt_mass,n,static_cast<int>(theta.size()));double scale=0.0;
      r.curvature_tolerance=curvature_rounding_tolerance(epsilon,r.base_rayleigh,row,theta,scale);r.curvature_scale=scale;if(!std::isfinite(r.residual)||!std::isfinite(r.actual_rayleigh)||!std::isfinite(r.curvature_tolerance)||!std::isfinite(scale)||!std::all_of(row.begin(),row.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("native fit Lanczos actual Ritz evidence is non-finite");
      out.push_back(std::move(r));}
    for(auto& mode:out)mode.dense_matrix_actions=dense_matrix_actions;
    return out;
  };
  for(int k=0;k<steps;++k){
    if(cublasDcopy(baseline.blas,d,workspace->v,1,workspace->basis+static_cast<std::size_t>(k)*d,1)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit Lanczos basis copy failed");
    if(baseline.apply_device(workspace->v,baseline.out)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native fit Lanczos basis Hessian operation failed");++dense_matrix_actions;
    workspace->additive_action(workspace->v);blas_check(cublasDcopy(baseline.blas,d,baseline.out,1,workspace->w,1),"native fit Lanczos operator copy failed");axpy(one,workspace->add,workspace->w);
    const double av_norm=norm(workspace->w);if(!std::isfinite(av_norm))throw std::runtime_error("native fit Lanczos operator norm is non-finite");
    if(k)axpy(-beta_prev,workspace->previous,workspace->w);const double a=dot(workspace->v,workspace->w);if(!std::isfinite(a))throw std::runtime_error("native fit Lanczos diagonal is non-finite");alpha.push_back(a);axpy(-a,workspace->v,workspace->w);
    workspace->project(workspace->w);
    for(int pass=0;pass<2;++pass){blas_check(cublasDgemv(baseline.blas,CUBLAS_OP_T,d,k+1,&one,workspace->basis,d,workspace->w,1,&zero,workspace->coefficient,1),"native fit Lanczos basis projection failed");
      const double minus_one=-1.0;blas_check(cublasDgemv(baseline.blas,CUBLAS_OP_N,d,k+1,&minus_one,workspace->basis,d,workspace->coefficient,1,&one,workspace->w,1),"native fit Lanczos reorthogonalization failed");}
    workspace->project(workspace->w);const double b=norm(workspace->w);if(!std::isfinite(b))throw std::runtime_error("native fit Lanczos residual norm is non-finite");beta.push_back(b);
    const double operator_scale=std::max({av_norm,std::abs(a),std::abs(beta_prev)}),breakdown=64.0*std::numeric_limits<double>::epsilon()*operator_scale;
    const int used=k+1;const bool search_finished=b<=breakdown||k==steps-1;
    const bool checkpoint=search_finished||used==32||used==64||used==128||used==256;
    if(checkpoint){auto modes=extract_modes(used,false);const bool below_threshold=std::any_of(modes.begin(),modes.end(),[&](const RitzMode& mode){return stability_cut_is_violated(mode.actual_rayleigh,epsilon,mode.curvature_tolerance);});
      const bool early_cut=steps<=96&&below_threshold&&!search_finished;
      if(search_finished||early_cut){if(early_cut)for(auto& mode:modes)mode.early_exit=true;return modes;}}
    blas_check(cublasDcopy(baseline.blas,d,workspace->v,1,workspace->previous,1),"native fit Lanczos previous-vector copy failed");blas_check(cublasDcopy(baseline.blas,d,workspace->w,1,workspace->v,1),"native fit Lanczos next-vector copy failed");const double inverse_b=1.0/b;blas_check(cublasDscal(baseline.blas,d,&inverse_b,workspace->v,1),"native fit Lanczos next-vector normalization failed");beta_prev=b;
  }
  return extract_modes(static_cast<int>(alpha.size()),false);
}

std::vector<double> spectral_row(const Graph& graph,const std::vector<double>& v,const std::vector<double>& sqrt_mass,
  const int n,const int p)
{
  std::vector<double> row(p,0.0);
  for(const auto& e:graph.edges){
    if(e.group<0||6*static_cast<std::size_t>(e.group)+5>=row.size())throw std::runtime_error("native fit spectral edge has an invalid type-pair group");
    const double dx[3]={v[e.j]/sqrt_mass[e.j]-v[e.i]/sqrt_mass[e.i],
      v[n+e.j]/sqrt_mass[n+e.j]-v[n+e.i]/sqrt_mass[n+e.i],
      v[2*n+e.j]/sqrt_mass[2*n+e.j]-v[2*n+e.i]/sqrt_mass[2*n+e.i]};
    const std::size_t c=6*static_cast<std::size_t>(e.group);
    row[c]+=dx[0]*dx[0];row[c+1]+=2.0*dx[0]*dx[1];row[c+2]+=2.0*dx[0]*dx[2];
    row[c+3]+=dx[1]*dx[1];row[c+4]+=2.0*dx[1]*dx[2];row[c+5]+=dx[2]*dx[2];
  }
  return row;
}

double curvature_rounding_tolerance(const double epsilon,const double base_rayleigh,const std::vector<double>& row,
  const std::vector<double>& theta,double& scale)
{
  scale=std::max(1.0,std::abs(base_rayleigh));for(std::size_t i=0;i<row.size();++i)scale+=std::abs(row[i]*theta[i]);
  return std::min(epsilon/4.0,256.0*std::numeric_limits<double>::epsilon()*scale);
}

void apply_total(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& q,const std::vector<double>& sqrt_mass,
  const std::vector<double>& sqrt_atom,const std::vector<double>& theta,const int n,std::vector<double>& result)
{
  std::vector<double> base(q.size()),add(q.size());baseline.apply(q,base);apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,add);
  for(std::size_t i=0;i<result.size();++i)result[i]=base[i]+add[i];project_translation(result,sqrt_atom,n);
}

void add_compensated_quotient(NeumaierAccumulator& result,const double numerator_high,const double numerator_low,
  const double denominator,const double sign)
{
  const double high=numerator_high/denominator;
  NeumaierAccumulator remainder;remainder.add(std::fma(-high,denominator,numerator_high));remainder.add(numerator_low);
  const double low=remainder.sum/denominator;
  NeumaierAccumulator tail;tail.add(std::fma(-low,denominator,remainder.sum));tail.add(remainder.correction);
  const double low2=tail.sum/denominator;
  const double low3=(std::fma(-low2,denominator,tail.sum)+tail.correction)/denominator;
  result.add_product(sign,high);result.add_product(sign,low);result.add_product(sign,low2);result.add_product(sign,low3);
}

void apply_additive_compensated(const Graph& graph,const std::vector<double>& q,const std::vector<double>& sqrt_mass,
  const std::vector<double>& theta,const int n,std::vector<double>& high,std::vector<double>& low)
{
  if(q.size()!=static_cast<std::size_t>(3*n)||sqrt_mass.size()!=q.size()||high.size()!=q.size()||low.size()!=q.size())
    throw std::invalid_argument("native fit compensated graph action dimensions are invalid");
  std::vector<NeumaierAccumulator> sums(q.size());
  for(const auto& edge:graph.edges){
    if(edge.i<0||edge.j<0||edge.i>=n||edge.j>=n||edge.group<0||6ULL*static_cast<std::size_t>(edge.group)+5>=theta.size())
      throw std::runtime_error("native fit compensated graph action has an invalid edge");
    const std::size_t p=6*static_cast<std::size_t>(edge.group);
    NeumaierAccumulator dx[3];
    for(int axis=0;axis<3;++axis){const std::size_t offset=static_cast<std::size_t>(axis)*n;
      add_compensated_quotient(dx[axis],q[offset+edge.j],0.0,sqrt_mass[offset+edge.j],1.0);
      add_compensated_quotient(dx[axis],q[offset+edge.i],0.0,sqrt_mass[offset+edge.i],-1.0);}
    NeumaierAccumulator y[3];
    y[0].add_product(theta[p],dx[0].sum);y[0].add_product(theta[p],dx[0].correction);y[0].add_product(theta[p+1],dx[1].sum);y[0].add_product(theta[p+1],dx[1].correction);y[0].add_product(theta[p+2],dx[2].sum);y[0].add_product(theta[p+2],dx[2].correction);
    y[1].add_product(theta[p+1],dx[0].sum);y[1].add_product(theta[p+1],dx[0].correction);y[1].add_product(theta[p+3],dx[1].sum);y[1].add_product(theta[p+3],dx[1].correction);y[1].add_product(theta[p+4],dx[2].sum);y[1].add_product(theta[p+4],dx[2].correction);
    y[2].add_product(theta[p+2],dx[0].sum);y[2].add_product(theta[p+2],dx[0].correction);y[2].add_product(theta[p+4],dx[1].sum);y[2].add_product(theta[p+4],dx[1].correction);y[2].add_product(theta[p+5],dx[2].sum);y[2].add_product(theta[p+5],dx[2].correction);
    for(int axis=0;axis<3;++axis){
      add_compensated_quotient(sums[static_cast<std::size_t>(axis)*n+edge.i],y[axis].sum,y[axis].correction,
        sqrt_mass[static_cast<std::size_t>(axis)*n+edge.i],-1.0);
      add_compensated_quotient(sums[static_cast<std::size_t>(axis)*n+edge.j],y[axis].sum,y[axis].correction,
        sqrt_mass[static_cast<std::size_t>(axis)*n+edge.j],1.0);
    }
  }
  for(std::size_t i=0;i<q.size();++i){high[i]=sums[i].sum;low[i]=sums[i].correction;}
}

void compute_cg_true_residual_compensated(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  const std::vector<double>& x,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,
  const std::vector<double>& theta,const int n,std::vector<double>& residual)
{
  if(rhs.size()!=x.size()||rhs.size()!=static_cast<std::size_t>(3*n)||sqrt_mass.size()!=rhs.size()||sqrt_atom.size()!=static_cast<std::size_t>(n))
    throw std::invalid_argument("native fit compensated CG residual dimensions are invalid");
  std::vector<double> base_high(rhs.size()),base_low(rhs.size()),add_high(rhs.size()),add_low(rhs.size());
  baseline.apply_compensated(x,base_high,base_low);
  apply_additive_compensated(graph,x,sqrt_mass,theta,n,add_high,add_low);
  std::vector<NeumaierAccumulator> values(rhs.size());
  for(std::size_t i=0;i<rhs.size();++i){values[i].add(rhs[i]);values[i].add(-base_high[i]);values[i].add(-base_low[i]);values[i].add(-add_high[i]);values[i].add(-add_low[i]);}
  const double mass_sum=compensated_dot(sqrt_atom,sqrt_atom);
  if(!(mass_sum>0.0)||!std::isfinite(mass_sum))throw std::runtime_error("native fit compensated CG residual has invalid total mass");
  for(int axis=0;axis<3;++axis){
    NeumaierAccumulator projection;
    for(int i=0;i<n;++i){const std::size_t index=static_cast<std::size_t>(axis)*n+i;
      projection.add_product(sqrt_atom[i],values[index].sum);projection.add_product(sqrt_atom[i],values[index].correction);}
    for(int i=0;i<n;++i){const std::size_t index=static_cast<std::size_t>(axis)*n+i;
      add_compensated_quotient(values[index],projection.sum,projection.correction,mass_sum,-sqrt_atom[i]);}
  }
  residual.resize(rhs.size());
  for(std::size_t i=0;i<rhs.size();++i){residual[i]=values[i].value();if(!std::isfinite(residual[i]))throw std::runtime_error("native fit compensated CG residual is non-finite");}
}

struct CGResidualComparison
{
  double fast_relative=0.0,compensated_relative=0.0,difference_norm=0.0;
  std::vector<double> fast,compensated;
};

CGResidualComparison compare_cg_true_residuals(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  const std::vector<double>& x,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,
  const std::vector<double>& theta,const int n)
{
  CGResidualComparison comparison;std::vector<double> projected_rhs=rhs,action(rhs.size());
  project_translation(projected_rhs,sqrt_atom,n);
  const double rhs_norm=std::sqrt(compensated_dot(projected_rhs,projected_rhs));
  if(!(rhs_norm>0.0)||!std::isfinite(rhs_norm))throw std::runtime_error("native fit CG replay has zero or non-finite projected right-hand side");
  apply_total(baseline,graph,x,sqrt_mass,sqrt_atom,theta,n,action);
  comparison.fast.resize(rhs.size());
  for(std::size_t i=0;i<rhs.size();++i)comparison.fast[i]=std::fma(-1.0,action[i],projected_rhs[i]);
  project_translation(comparison.fast,sqrt_atom,n);
  compute_cg_true_residual_compensated(baseline,graph,rhs,x,sqrt_mass,sqrt_atom,theta,n,comparison.compensated);
  comparison.fast_relative=std::sqrt(compensated_dot(comparison.fast,comparison.fast))/rhs_norm;
  comparison.compensated_relative=std::sqrt(compensated_dot(comparison.compensated,comparison.compensated))/rhs_norm;
  for(std::size_t i=0;i<rhs.size();++i)comparison.difference_norm=std::hypot(comparison.difference_norm,comparison.fast[i]-comparison.compensated[i]);
  if(!std::isfinite(comparison.fast_relative)||!std::isfinite(comparison.compensated_relative)||!std::isfinite(comparison.difference_norm))
    throw std::runtime_error("native fit CG replay residual comparison is non-finite");
  return comparison;
}

struct CGResidualCheck{int iteration=0,restarts=0;double recursive_relative=0.0,true_relative=0.0;};
struct CGWitness {
  std::string classification;
  int probe=-1,iteration=-1,residual_restarts=0;
  double p2=0.0,pap=0.0,rayleigh=0.0,base_rayleigh=0.0,add_rayleigh=0.0,repeat_rayleigh=0.0,repeat_diff_norm=0.0,translation_overlap=0.0,relative_residual=0.0,recursive_relative_residual=0.0,curvature_tolerance=0.0;
  bool finite=false,true_residual_computed=false;
  std::vector<double> direction,action,base,add,rhs_input,rhs,solution;
  std::vector<CGResidualCheck> residual_checks;
};
struct CGResult{std::vector<double> x;double relative_residual=0.0,recursive_relative_residual=std::numeric_limits<double>::quiet_NaN();int residual_restarts=0,iterations=0;CGWitness witness;};
struct CGReplayContext
{
  const RpmdJANativeFitOptions& options;
  std::uint64_t frame_count,raw_model,raw_config,active_model,active_config;
  int raw_charge,raw_kspace;
  double raw_mesh,active_mesh;
};
void set_true_residual_failure(CGResult& result,const int probe,const int iteration,const std::vector<double>& rhs_input,const std::vector<double>& rhs,
  const double recursive_relative,const double true_relative,std::vector<CGResidualCheck> checks)
{
  CGWitness& witness=result.witness;witness.classification="TRUE_RESIDUAL_FAILURE";witness.probe=probe;witness.iteration=iteration;
  witness.residual_restarts=result.residual_restarts;witness.relative_residual=true_relative;witness.recursive_relative_residual=recursive_relative;
  result.recursive_relative_residual=recursive_relative;
  witness.true_residual_computed=true;witness.rhs_input=rhs_input;witness.rhs=rhs;witness.solution=result.x;witness.residual_checks=std::move(checks);
  witness.finite=std::isfinite(recursive_relative)&&std::isfinite(true_relative)&&
    std::all_of(witness.rhs_input.begin(),witness.rhs_input.end(),[](double x){return std::isfinite(x);})&&
    std::all_of(witness.rhs.begin(),witness.rhs.end(),[](double x){return std::isfinite(x);})&&
    std::all_of(witness.solution.begin(),witness.solution.end(),[](double x){return std::isfinite(x);})&&
    std::all_of(witness.residual_checks.begin(),witness.residual_checks.end(),[](const CGResidualCheck& check){return std::isfinite(check.recursive_relative)&&std::isfinite(check.true_relative);});
}
void write_cg_witness(const std::string& path,const CGWitness& w,const std::vector<double>& theta,
  const std::string& raw_path,const std::string& spool_path,const CGReplayContext& replay)
{
  std::ofstream out(path,std::ios::out|std::ios::trunc);if(!out)throw std::runtime_error("cannot write native fit CG witness: "+path);
  const auto& options=replay.options;out<<std::setprecision(17)<<"status NOT_ACCEPTED_REFERENCE\nclassification "<<w.classification<<"\nprobe "<<w.probe<<"\niteration "<<w.iteration
    <<"\nresidual_restarts "<<w.residual_restarts<<"\nrecursive_relative_residual "<<w.recursive_relative_residual<<"\ntrue_residual_status "<<(w.true_residual_computed?"COMPUTED":"NOT_COMPUTED")<<"\ntrue_relative_residual ";
  if(w.true_residual_computed)out<<w.relative_residual;else out<<"NOT_COMPUTED";out<<"\nresidual_checks "<<w.residual_checks.size()<<'\n';
  for(const auto& check:w.residual_checks)out<<"residual_check "<<check.iteration<<' '<<check.restarts<<' '<<check.recursive_relative<<' '<<check.true_relative<<'\n';
  out<<"curvature_diagnostic_status "<<(w.direction.empty()?"NOT_COMPUTED":"COMPUTED")<<"\np2 "<<w.p2<<"\npAp "<<w.pap<<"\nrayleigh_quotient "<<w.rayleigh<<"\nbase_rayleigh "<<w.base_rayleigh<<"\nadditive_rayleigh "<<w.add_rayleigh<<"\ncurvature_tolerance "<<w.curvature_tolerance<<"\nrepeat_rayleigh "<<w.repeat_rayleigh<<"\nrepeat_diff_norm "<<w.repeat_diff_norm<<"\ntranslation_overlap "<<w.translation_overlap<<"\nfinite "<<w.finite
    <<"\nqraw_source "<<std::quoted(raw_path)<<"\nsample_spool "<<std::quoted(spool_path)<<"\nsample_frames "<<replay.frame_count<<"\ntraining_frames "<<2*replay.frame_count/3
    <<"\ncutoff_A "<<options.cutoff<<"\ntemperature_K "<<options.temperature<<"\nfd_step "<<options.fd_step<<"\nsample_interval "<<options.sample_interval<<"\nepsilon "<<options.epsilon
    <<"\ninternal_mass_com "<<options.internal_mass_com<<"\nqraw_model_fingerprint "<<replay.raw_model<<"\nqraw_config_fingerprint "<<replay.raw_config
    <<"\nactive_model_fingerprint "<<replay.active_model<<"\nactive_config_fingerprint "<<replay.active_config<<"\nqraw_charge_mode "<<replay.raw_charge
    <<"\nqraw_kspace_flag "<<replay.raw_kspace<<"\nqraw_pppm_mesh_spacing "<<replay.raw_mesh<<"\nactive_pppm_mesh_spacing "<<replay.active_mesh
    <<"\nprobe_source fixed_seeded_random16_then_type_local_then_low_ritz; CG probe index above\ncoordinate_convention "
    <<(options.internal_mass_com?"internal mass-COM projected mass-weighted xyz-SoA":"raw mass-weighted xyz-SoA")<<"\ntheta "<<theta.size();for(double x:theta)out<<' '<<x;out<<'\n';
  auto vector=[&](const char* label,const std::vector<double>& x){out<<label<<' '<<x.size();for(double v:x)out<<' '<<v;out<<'\n';};
  vector("rhs_input_xyz_soa",w.rhs_input);vector("rhs_projected_xyz_soa",w.rhs);vector("solution_x_xyz_soa",w.solution);vector("v_xyz_soa",w.direction);vector("Dv_xyz_soa",w.action);vector("base_Dv_xyz_soa",w.base);vector("additive_Dv_xyz_soa",w.add);
  out.close();if(!out)throw std::runtime_error("failed closing native fit CG witness: "+path);
}

struct CGWitnessSnapshot
{
  std::string status,classification,true_residual_status,qraw_path,sample_path;
  std::uint64_t sample_frames=0,training_frames=0,qraw_model=0,qraw_config=0,active_model=0,active_config=0;
  int sample_interval=0,internal_mass_com=0,qraw_charge=0,qraw_kspace=0,probe=-1,iteration=-1;
  double cutoff=0.0,temperature=0.0,fd_step=0.0,epsilon=0.0,qraw_mesh=0.0,active_mesh=0.0;
  std::vector<double> theta,rhs_input,rhs_projected,solution;
};

std::pair<std::string,std::string> read_cg_witness_paths(const std::string& path)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open native fit CG witness: "+path);
  std::string line,qraw_path,sample_path;
  while(std::getline(in,line)){std::istringstream fields(line);std::string key;if(!(fields>>key))continue;
    if(key=="qraw_source")fields>>std::quoted(qraw_path);
    else if(key=="sample_spool")fields>>std::quoted(sample_path);
  }
  if(qraw_path.empty()||sample_path.empty())throw std::runtime_error("native fit CG witness is missing qraw_source or sample_spool");
  return {qraw_path,sample_path};
}

CGWitnessSnapshot read_cg_witness(const std::string& path,const int dimension,const int parameter_count)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open native fit CG witness: "+path);
  CGWitnessSnapshot snapshot;std::set<std::string> seen;std::string line;
  const std::set<std::string> repeated={"residual_check","curvature_diagnostic_status","p2","pAp","rayleigh_quotient",
    "base_rayleigh","additive_rayleigh","curvature_tolerance","repeat_rayleigh","repeat_diff_norm","translation_overlap",
    "finite","probe_source","coordinate_convention","v_xyz_soa","Dv_xyz_soa","base_Dv_xyz_soa","additive_Dv_xyz_soa"};
  while(std::getline(in,line)){
    std::istringstream fields(line);std::string key;if(!(fields>>key))continue;
    if(!repeated.count(key)&&!seen.insert(key).second)throw std::runtime_error("native fit CG witness contains a duplicate field: "+key);
    if(key=="theta"||key=="rhs_input_xyz_soa"||key=="rhs_projected_xyz_soa"||key=="solution_x_xyz_soa"){
      std::size_t count=0;if(!(fields>>count))throw std::runtime_error("invalid vector header in native fit CG witness");
      const std::size_t expected=key=="theta"?static_cast<std::size_t>(std::max(0,parameter_count)):static_cast<std::size_t>(dimension);
      if(key=="theta"&&parameter_count==0){if(count>1000000||count%6!=0)throw std::runtime_error("native fit CG witness theta dimension is invalid");}
      else if(count!=expected)throw std::runtime_error("native fit CG witness vector dimension does not match sample/qraw inputs: "+key);
      std::vector<double> values(count);for(double& value:values)if(!(fields>>value)||!std::isfinite(value))throw std::runtime_error("native fit CG witness vector is truncated or non-finite: "+key);
      std::string extra;if(fields>>extra)throw std::runtime_error("native fit CG witness vector has trailing data: "+key);
      if(key=="theta")snapshot.theta=std::move(values);
      else if(key=="rhs_input_xyz_soa")snapshot.rhs_input=std::move(values);
      else if(key=="rhs_projected_xyz_soa")snapshot.rhs_projected=std::move(values);
      else snapshot.solution=std::move(values);
      continue; // The trailing-data probe normally sets failbit at end of line.
    }else if(key=="status")fields>>snapshot.status;
    else if(key=="classification")fields>>snapshot.classification;
    else if(key=="true_residual_status")fields>>snapshot.true_residual_status;
    else if(key=="qraw_source")fields>>std::quoted(snapshot.qraw_path);
    else if(key=="sample_spool")fields>>std::quoted(snapshot.sample_path);
    else if(key=="sample_frames")fields>>snapshot.sample_frames;
    else if(key=="training_frames")fields>>snapshot.training_frames;
    else if(key=="sample_interval")fields>>snapshot.sample_interval;
    else if(key=="internal_mass_com")fields>>snapshot.internal_mass_com;
    else if(key=="qraw_model_fingerprint")fields>>snapshot.qraw_model;
    else if(key=="qraw_config_fingerprint")fields>>snapshot.qraw_config;
    else if(key=="active_model_fingerprint")fields>>snapshot.active_model;
    else if(key=="active_config_fingerprint")fields>>snapshot.active_config;
    else if(key=="qraw_charge_mode")fields>>snapshot.qraw_charge;
    else if(key=="qraw_kspace_flag")fields>>snapshot.qraw_kspace;
    else if(key=="probe")fields>>snapshot.probe;
    else if(key=="iteration")fields>>snapshot.iteration;
    else if(key=="cutoff_A")fields>>snapshot.cutoff;
    else if(key=="temperature_K")fields>>snapshot.temperature;
    else if(key=="fd_step")fields>>snapshot.fd_step;
    else if(key=="epsilon")fields>>snapshot.epsilon;
    else if(key=="qraw_pppm_mesh_spacing")fields>>snapshot.qraw_mesh;
    else if(key=="active_pppm_mesh_spacing")fields>>snapshot.active_mesh;
    else continue;
    if(!fields)throw std::runtime_error("invalid native fit CG witness field: "+key);
  }
  const char* required[]={"status","classification","true_residual_status","qraw_source","sample_spool","sample_frames",
    "training_frames","sample_interval","internal_mass_com","qraw_model_fingerprint","qraw_config_fingerprint",
    "active_model_fingerprint","active_config_fingerprint","qraw_charge_mode","qraw_kspace_flag","probe","iteration",
    "cutoff_A","temperature_K","fd_step","epsilon","qraw_pppm_mesh_spacing","active_pppm_mesh_spacing","theta",
    "rhs_input_xyz_soa","rhs_projected_xyz_soa","solution_x_xyz_soa"};
  for(const char* key:required)if(!seen.count(key))throw std::runtime_error(std::string("native fit CG witness is missing field: ")+key);
  if(snapshot.status!="NOT_ACCEPTED_REFERENCE"||snapshot.classification!="TRUE_RESIDUAL_FAILURE"||snapshot.true_residual_status!="COMPUTED"||
     (snapshot.internal_mass_com!=0&&snapshot.internal_mass_com!=1)||snapshot.sample_interval<=0||snapshot.probe<0||snapshot.iteration<0||
     !std::isfinite(snapshot.cutoff)||snapshot.cutoff<=0.0||!std::isfinite(snapshot.temperature)||snapshot.temperature<=0.0||
     !std::isfinite(snapshot.fd_step)||snapshot.fd_step<=0.0||!std::isfinite(snapshot.epsilon)||snapshot.epsilon<0.0||
     !std::isfinite(snapshot.qraw_mesh)||!std::isfinite(snapshot.active_mesh))
    throw std::runtime_error("native fit CG witness is not a complete finite true-residual failure snapshot");
  return snapshot;
}

bool preserve_candidate_package_on_full_spd_failure(const std::string& message,bool& pack_owned)
{
  if(message.find("qNEP translation-complement Hessian is not positive definite")!=0)return false;
  pack_owned=false;return true;
}
CGResult solve_projected_cg(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const std::vector<double>& theta,const int n,const int probe=-1,const double epsilon=0.0,
  const double relative_tolerance=1e-8,const int max_iterations=20000,const bool return_on_iteration_limit=false)
{
  if(!(relative_tolerance>0.0)||!(relative_tolerance<1.0)||!std::isfinite(relative_tolerance)||max_iterations<=0)
    throw std::invalid_argument("native fit projected CG tolerance or iteration limit is invalid");
  CGResult out;out.x.assign(rhs.size(),0.0);std::vector<double> r=rhs,p=rhs,ap(rhs.size());project_translation(r,sqrt_atom,n);p=r;
  const std::vector<double> projected_rhs=r;double rr=compensated_dot(r,r);const double initial=std::sqrt(rr);if(!(initial>0.0)||!std::isfinite(initial))throw std::runtime_error("NONFINITE_CG_ARITHMETIC: native fit response probe has zero or non-finite norm");
  auto record_failure=[&](const char* classification,const int it){CGWitness& w=out.witness;w.classification=classification;w.probe=probe;w.iteration=it;w.residual_restarts=out.residual_restarts;w.p2=compensated_dot(p,p);w.pap=compensated_dot(p,ap);w.relative_residual=std::sqrt(rr)/initial;w.recursive_relative_residual=w.relative_residual;out.recursive_relative_residual=w.relative_residual;if(w.p2>0.0&&std::isfinite(w.p2)){w.direction=p;for(double& x:w.direction)x/=std::sqrt(w.p2);w.base.resize(p.size());w.add.resize(p.size());w.action.resize(p.size());baseline.apply(w.direction,w.base);apply_additive(graph,w.direction,sqrt_mass,sqrt_atom,theta,n,w.add);apply_total(baseline,graph,w.direction,sqrt_mass,sqrt_atom,theta,n,w.action);std::vector<double> repeat(p.size());apply_total(baseline,graph,w.direction,sqrt_mass,sqrt_atom,theta,n,repeat);w.rayleigh=compensated_dot(w.direction,w.action);w.base_rayleigh=compensated_dot(w.direction,w.base);w.add_rayleigh=compensated_dot(w.direction,w.add);w.repeat_rayleigh=compensated_dot(w.direction,repeat);for(std::size_t i=0;i<repeat.size();++i)w.repeat_diff_norm=std::hypot(w.repeat_diff_norm,repeat[i]-w.action[i]);}w.finite=std::isfinite(w.p2)&&std::isfinite(w.pap)&&std::isfinite(w.relative_residual)&&std::all_of(w.base.begin(),w.base.end(),[](double x){return std::isfinite(x);})&&std::all_of(w.add.begin(),w.add.end(),[](double x){return std::isfinite(x);})&&std::all_of(w.action.begin(),w.action.end(),[](double x){return std::isfinite(x);});};
  std::vector<CGResidualCheck> residual_checks;
  for(int it=0;it<max_iterations;++it){out.iterations=it;project_translation(p,sqrt_atom,n);apply_total(baseline,graph,p,sqrt_mass,sqrt_atom,theta,n,ap);const double pap=compensated_dot(p,ap);if(!std::all_of(ap.begin(),ap.end(),[](double x){return std::isfinite(x);} )){
      record_failure("NONFINITE_OPERATOR",it);return out;
    }
     if(!std::isfinite(pap)){record_failure("NONFINITE_CG_ARITHMETIC",it);return out;}
     const double p2=compensated_dot(p,p);if(!(p2>0.0)||!std::isfinite(p2)){record_failure("NONFINITE_CG_ARITHMETIC",it);return out;}
     const double rayleigh=pap/p2,operator_scale=std::max(1.0,std::sqrt(compensated_dot(ap,ap))/std::sqrt(p2));const double curvature_tolerance=std::min(epsilon/4.0,256.0*std::numeric_limits<double>::epsilon()*operator_scale);
     if(!(pap>0.0)||rayleigh<epsilon-curvature_tolerance){
      CGWitness& w=out.witness;w.classification="FINITE_CURVATURE_BELOW_EPSILON";w.probe=probe;w.iteration=it;w.residual_restarts=out.residual_restarts;w.p2=p2;w.pap=pap;w.relative_residual=std::sqrt(rr)/initial;w.recursive_relative_residual=w.relative_residual;out.recursive_relative_residual=w.relative_residual;
      if(!(w.p2>0.0)||!std::isfinite(w.p2)){w.classification="NONFINITE_CG_ARITHMETIC";return out;}
      const double original_pap=pap;w.direction=p;for(double& x:w.direction)x/=std::sqrt(w.p2);project_translation(w.direction,sqrt_atom,n);
      const double norm=std::sqrt(compensated_dot(w.direction,w.direction));if(!(norm>0.0)||!std::isfinite(norm)){w.classification="NONFINITE_CG_ARITHMETIC";return out;}for(double& x:w.direction)x/=norm;
      w.base.resize(p.size());w.add.resize(p.size());w.action.resize(p.size());std::vector<double> repeat(p.size());baseline.apply(w.direction,w.base);apply_additive(graph,w.direction,sqrt_mass,sqrt_atom,theta,n,w.add);apply_total(baseline,graph,w.direction,sqrt_mass,sqrt_atom,theta,n,w.action);apply_total(baseline,graph,w.direction,sqrt_mass,sqrt_atom,theta,n,repeat);
       w.pap=original_pap;w.rayleigh=compensated_dot(w.direction,w.action);
      std::vector<double> projected=w.direction;project_translation(projected,sqrt_atom,n);double removed2=0.0;for(std::size_t i=0;i<projected.size();++i){const double delta=w.direction[i]-projected[i];removed2+=delta*delta;}w.translation_overlap=std::sqrt(removed2);
       w.base_rayleigh=compensated_dot(w.direction,w.base);w.add_rayleigh=compensated_dot(w.direction,w.add);w.repeat_rayleigh=compensated_dot(w.direction,repeat);for(std::size_t i=0;i<repeat.size();++i){w.repeat_diff_norm=std::hypot(w.repeat_diff_norm,repeat[i]-w.action[i]);w.repeat_diff_norm=std::hypot(w.repeat_diff_norm,w.action[i]-w.base[i]-w.add[i]);}double base_norm=0.0,add_norm=0.0;for(double x:w.base)base_norm=std::hypot(base_norm,x);for(double x:w.add)add_norm=std::hypot(add_norm,x);w.curvature_tolerance=std::min(epsilon/4.0,256.0*std::numeric_limits<double>::epsilon()*std::max(1.0,base_norm+add_norm));
       const bool operator_finite=std::all_of(w.base.begin(),w.base.end(),[](double x){return std::isfinite(x);})&&std::all_of(w.add.begin(),w.add.end(),[](double x){return std::isfinite(x);})&&std::all_of(w.action.begin(),w.action.end(),[](double x){return std::isfinite(x);})&&std::all_of(repeat.begin(),repeat.end(),[](double x){return std::isfinite(x);});const bool scalar_finite=std::isfinite(w.p2)&&std::isfinite(w.pap)&&std::isfinite(w.rayleigh)&&std::isfinite(w.base_rayleigh)&&std::isfinite(w.add_rayleigh)&&std::isfinite(w.repeat_rayleigh)&&std::isfinite(w.repeat_diff_norm)&&std::isfinite(w.translation_overlap)&&std::isfinite(w.relative_residual)&&std::isfinite(w.curvature_tolerance)&&std::isfinite(base_norm)&&std::isfinite(add_norm);w.finite=operator_finite&&scalar_finite;
       if(!operator_finite)w.classification="NONFINITE_OPERATOR";
       else if(!scalar_finite)w.classification="NONFINITE_CG_ARITHMETIC";
        else if(w.repeat_diff_norm>256.0*std::numeric_limits<double>::epsilon()*std::max(1.0,base_norm+add_norm))w.classification="NUMERICAL_BREAKDOWN";
       else if(w.rayleigh<=0.0)w.classification="NONPOSITIVE_OPERATOR_DIRECTION";
       else w.classification="UNRESOLVED_SOFT_DIRECTION";
       if(w.finite&&w.rayleigh>=epsilon-w.curvature_tolerance&&original_pap>0.0&&w.classification!="NUMERICAL_BREAKDOWN")out.witness=CGWitness{};
       else {if(w.finite&&w.rayleigh>0.0&&w.rayleigh>=epsilon-w.curvature_tolerance)w.classification="NUMERICAL_BREAKDOWN";return out;}
    }
    const double alpha=rr/pap;for(std::size_t i=0;i<r.size();++i){out.x[i]=std::fma(alpha,p[i],out.x[i]);r[i]=std::fma(-alpha,ap[i],r[i]);}out.iterations=it+1;project_translation(r,sqrt_atom,n);
     const double next=compensated_dot(r,r);out.relative_residual=std::sqrt(next)/initial;out.recursive_relative_residual=out.relative_residual;if(!std::isfinite(out.relative_residual)||!std::isfinite(alpha)||!std::isfinite(next))throw std::runtime_error("NONFINITE_CG_ARITHMETIC: native fit response CG arithmetic became non-finite");if(out.relative_residual<=relative_tolerance){const double recursive_relative=out.relative_residual;apply_total(baseline,graph,out.x,sqrt_mass,sqrt_atom,theta,n,ap);for(std::size_t i=0;i<r.size();++i)r[i]=std::fma(-1.0,ap[i],projected_rhs[i]);project_translation(r,sqrt_atom,n);const double true2=compensated_dot(r,r);out.relative_residual=std::sqrt(true2)/initial;if(!std::isfinite(out.relative_residual))throw std::runtime_error("NONFINITE_CG_ARITHMETIC: native fit response CG true residual is non-finite");residual_checks.push_back({it+1,out.residual_restarts,recursive_relative,out.relative_residual});if(out.relative_residual<=relative_tolerance)return out;if(out.residual_restarts>=2){set_true_residual_failure(out,probe,it+1,rhs,projected_rhs,recursive_relative,out.relative_residual,std::move(residual_checks));return out;}rr=true2;p=r;++out.residual_restarts;continue;}
    const double beta=next/rr;for(std::size_t i=0;i<p.size();++i)p[i]=std::fma(beta,p[i],r[i]);project_translation(p,sqrt_atom,n);rr=next;
  }
  if(return_on_iteration_limit){out.witness.classification="ITERATION_LIMIT";out.witness.probe=probe;out.witness.iteration=max_iterations;
    out.witness.residual_restarts=out.residual_restarts;out.witness.relative_residual=out.relative_residual;
    out.witness.recursive_relative_residual=out.recursive_relative_residual;return out;}
  throw std::runtime_error(relative_tolerance==1e-8?
    "NUMERICAL_BREAKDOWN: native fit response CG failed its 1e-8 residual tolerance":
    "NUMERICAL_BREAKDOWN: native fit response CG failed its requested residual tolerance");
}

struct CGReplayRefinementRound
{
  std::string status,error;
  int iterations=0,restarts=0;
  bool solver_counts_available=false;
  double recursive_relative=std::numeric_limits<double>::quiet_NaN();
  double correction_relative=std::numeric_limits<double>::quiet_NaN();
  double correction_norm=std::numeric_limits<double>::quiet_NaN();
  double fast_relative=std::numeric_limits<double>::quiet_NaN();
  double compensated_relative=std::numeric_limits<double>::quiet_NaN();
  double seconds=0.0;
  std::uint64_t dense_matrix_actions=0;
};

struct CGReplayRefinement
{
  std::vector<double> solution;
  CGResidualComparison residuals;
  std::vector<CGReplayRefinementRound> rounds;
  std::string status;
  std::uint64_t dense_matrix_actions=0;
};

struct CGProbeRecoveryDiagnostic
{
  int probe=-1,outer=-1,refinement_rounds=0,refinement_restarts=0;
  std::uint64_t verification_matrix_actions=0,refinement_matrix_actions=0;
  double initial_fast_residual=std::numeric_limits<double>::quiet_NaN();
  double initial_compensated_residual=std::numeric_limits<double>::quiet_NaN();
  double final_fast_residual=std::numeric_limits<double>::quiet_NaN();
  double final_compensated_residual=std::numeric_limits<double>::quiet_NaN();
  double verification_seconds=0.0,refinement_seconds=0.0;
  std::string initial_cg_status,verification_method="COMPENSATED",refinement_status="NOT_STARTED",witness_snapshot_path,error;
  bool recovered=false;
  CGWitness original_witness;
};

CGReplayRefinement refine_cg_replay_solution(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  const std::vector<double>& saved_solution,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,
  const std::vector<double>& theta,const int n,const int probe,const CGResidualComparison& saved_residuals,
  const int max_rounds=3,const std::uint64_t max_dense_matrix_actions=4096)
{
  constexpr std::uint64_t reserved_matrix_actions=5; // Three CG checks/diagnostics plus two trial-residual actions.
  constexpr double correction_relative_tolerance=1e-2;
  constexpr double final_relative_tolerance=1e-8;
  CGReplayRefinement result;result.solution=saved_solution;result.residuals=saved_residuals;
  const std::uint64_t actions_start=baseline.dense_matrix_actions;
  if(result.residuals.compensated_relative<=final_relative_tolerance){result.status="COMPENSATED_RESIDUAL_ALREADY_WITHIN_GATE";return result;}
  for(int round=0;round<max_rounds;++round){
    CGReplayRefinementRound record;const auto round_started=std::chrono::steady_clock::now();
    const std::uint64_t used=baseline.dense_matrix_actions-actions_start;
    if(used>max_dense_matrix_actions||max_dense_matrix_actions-used<=reserved_matrix_actions){result.status="MATRIX_ACTION_LIMIT";break;}
    const int max_iterations=static_cast<int>(std::min<std::uint64_t>(20000,max_dense_matrix_actions-used-reserved_matrix_actions));
    const bool matrix_action_limited=max_iterations<20000;
    const std::uint64_t actions_before=baseline.dense_matrix_actions;
    const std::vector<double> correction_rhs=result.residuals.compensated;
    CGResult correction;
    try{correction=solve_projected_cg(baseline,graph,correction_rhs,sqrt_mass,sqrt_atom,theta,n,probe,0.0,
        correction_relative_tolerance,max_iterations,true);
      record.status=correction.witness.classification.empty()?"CG_PASS":correction.witness.classification;
      record.iterations=correction.iterations;record.restarts=correction.residual_restarts;record.solver_counts_available=true;
      record.recursive_relative=correction.recursive_relative_residual;
      if(correction.witness.classification.empty())record.correction_relative=correction.relative_residual;
      else if(correction.witness.true_residual_computed)record.correction_relative=correction.witness.relative_residual;
      if(record.status=="ITERATION_LIMIT"&&matrix_action_limited)record.status="MATRIX_ACTION_LIMIT";
    }catch(const std::exception& error){record.error=error.what();record.status="CG_EXCEPTION";}
    if(!record.status.empty()&&record.status!="CG_PASS"){
      record.dense_matrix_actions=baseline.dense_matrix_actions-actions_before;
      record.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-round_started).count();
      result.rounds.push_back(std::move(record));result.status=result.rounds.back().status=="MATRIX_ACTION_LIMIT"?
        "MATRIX_ACTION_LIMIT":"CORRECTION_"+result.rounds.back().status;break;
    }
    if(!std::all_of(correction.x.begin(),correction.x.end(),[](double value){return std::isfinite(value);} )){
      record.status="NONFINITE_CORRECTION";record.dense_matrix_actions=baseline.dense_matrix_actions-actions_before;
      record.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-round_started).count();
      result.rounds.push_back(std::move(record));result.status="CORRECTION_NONFINITE";break;
    }
    record.correction_norm=std::sqrt(compensated_dot(correction.x,correction.x));
    if(!std::isfinite(record.correction_norm)){
      record.status="NONFINITE_CORRECTION_NORM";record.dense_matrix_actions=baseline.dense_matrix_actions-actions_before;
      record.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-round_started).count();
      result.rounds.push_back(std::move(record));result.status="CORRECTION_NONFINITE";break;
    }
    std::vector<double> trial_solution=result.solution;
    for(std::size_t i=0;i<trial_solution.size();++i)trial_solution[i]=std::fma(1.0,correction.x[i],trial_solution[i]);
    CGResidualComparison trial_residuals;
    try{trial_residuals=compare_cg_true_residuals(baseline,graph,rhs,trial_solution,sqrt_mass,sqrt_atom,theta,n);}
    catch(const std::exception& error){record.status="RESIDUAL_CHECK_EXCEPTION";record.error=error.what();
      record.dense_matrix_actions=baseline.dense_matrix_actions-actions_before;record.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-round_started).count();
      result.rounds.push_back(std::move(record));result.status="CORRECTION_RESIDUAL_CHECK_FAILED";break;}
    record.fast_relative=trial_residuals.fast_relative;record.compensated_relative=trial_residuals.compensated_relative;
    record.dense_matrix_actions=baseline.dense_matrix_actions-actions_before;
    record.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-round_started).count();
    if(!(trial_residuals.compensated_relative<result.residuals.compensated_relative)){
      record.status="NO_RESIDUAL_DECREASE";result.rounds.push_back(std::move(record));result.status="NO_RESIDUAL_DECREASE";break;
    }
    result.solution=std::move(trial_solution);result.residuals=std::move(trial_residuals);result.rounds.push_back(std::move(record));
    if(result.residuals.compensated_relative<=final_relative_tolerance){result.status="COMPENSATED_RESIDUAL_TARGET_REACHED";break;}
  }
  if(result.status.empty())result.status=result.rounds.size()>=static_cast<std::size_t>(max_rounds)?"ROUND_LIMIT":"STOPPED";
  result.dense_matrix_actions=baseline.dense_matrix_actions-actions_start;
  return result;
}

bool recover_cg_true_residual_failure(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  CGResult& cg,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const std::vector<double>& theta,
  const int n,CGProbeRecoveryDiagnostic& diagnostic,const int outer=-1)
{
  diagnostic.probe=cg.witness.probe;diagnostic.outer=outer;diagnostic.initial_cg_status=cg.witness.classification;
  diagnostic.original_witness=cg.witness;
  const std::uint64_t verification_actions=baseline.dense_matrix_actions;const auto verification_started=std::chrono::steady_clock::now();
  CGResidualComparison initial;
  try{initial=compare_cg_true_residuals(baseline,graph,rhs,cg.x,sqrt_mass,sqrt_atom,theta,n);}
  catch(const std::exception& error){diagnostic.refinement_status="COMPENSATED_VERIFICATION_EXCEPTION";diagnostic.error=error.what();
    diagnostic.verification_matrix_actions=baseline.dense_matrix_actions-verification_actions;
    diagnostic.verification_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-verification_started).count();throw;}
  diagnostic.verification_matrix_actions=baseline.dense_matrix_actions-verification_actions;
  diagnostic.verification_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-verification_started).count();
  diagnostic.initial_fast_residual=initial.fast_relative;diagnostic.initial_compensated_residual=initial.compensated_relative;
  diagnostic.final_fast_residual=initial.fast_relative;diagnostic.final_compensated_residual=initial.compensated_relative;
  constexpr double final_relative_tolerance=1e-8;
  if(initial.compensated_relative<=final_relative_tolerance){diagnostic.refinement_status="COMPENSATED_RESIDUAL_ALREADY_WITHIN_GATE";
    diagnostic.recovered=true;cg.relative_residual=initial.compensated_relative;return true;}
  const std::uint64_t refinement_actions=baseline.dense_matrix_actions;const auto refinement_started=std::chrono::steady_clock::now();
  try{
    const CGReplayRefinement refinement=refine_cg_replay_solution(baseline,graph,rhs,cg.x,sqrt_mass,sqrt_atom,theta,n,cg.witness.probe,initial);
    diagnostic.refinement_status=refinement.status;diagnostic.refinement_rounds=static_cast<int>(refinement.rounds.size());
    diagnostic.refinement_matrix_actions=refinement.dense_matrix_actions;
    diagnostic.refinement_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-refinement_started).count();
    for(const CGReplayRefinementRound& round:refinement.rounds)if(round.solver_counts_available)diagnostic.refinement_restarts+=round.restarts;
    diagnostic.final_fast_residual=refinement.residuals.fast_relative;
    diagnostic.final_compensated_residual=refinement.residuals.compensated_relative;
    if(refinement.residuals.compensated_relative<=final_relative_tolerance){cg.x=refinement.solution;
      cg.relative_residual=refinement.residuals.compensated_relative;diagnostic.recovered=true;return true;}
    return false;
  }catch(const std::exception& error){diagnostic.refinement_status="REFINEMENT_EXCEPTION";diagnostic.error=error.what();
    diagnostic.refinement_matrix_actions=baseline.dense_matrix_actions-refinement_actions;
    diagnostic.refinement_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-refinement_started).count();throw;}
}

void write_cg_replay_refinement_round(std::ostream& report,const std::size_t index,const CGReplayRefinementRound& round)
{
  const auto write_value=[&](const double value){if(std::isfinite(value))report<<value;else report<<"NOT_COMPUTED";};
  report<<"correction_round "<<index<<" status "<<round.status<<" iterations ";
  if(round.solver_counts_available)report<<round.iterations;else report<<"NOT_AVAILABLE";
  report<<" restarts ";if(round.solver_counts_available)report<<round.restarts;else report<<"NOT_AVAILABLE";
  report<<" recursive_relative ";write_value(round.recursive_relative);
  report<<" correction_relative ";write_value(round.correction_relative);report<<" correction_norm ";write_value(round.correction_norm);
  report<<" fast_relative ";write_value(round.fast_relative);report<<" compensated_relative ";write_value(round.compensated_relative);
  report<<" dense_matrix_actions "<<round.dense_matrix_actions<<" seconds ";write_value(round.seconds);
  report<<" error "<<std::quoted(round.error)<<'\n';
}

struct ProbeOrigin
{
  int seed_atom_index=-1,type_index=-1;
  double mass=std::numeric_limits<double>::quiet_NaN();
};

std::vector<std::vector<double>> make_probes(const std::vector<RitzMode>& soft,const std::vector<double>& sqrt_mass,
  const std::vector<int>& types,const int n,std::vector<std::string>* sources=nullptr,std::vector<ProbeOrigin>* origins=nullptr)
{
  const int d=3*n,internal=d-3;std::vector<std::vector<double>> probes;probes.reserve(std::min(24,internal));std::uint64_t state=0x243f6a8885a308d3ULL;
  auto add=[&](std::vector<double> v,const char* source,const int seed_atom){project_translation(v,sqrt_mass,n);for(const auto&w:probes){const double c=std::inner_product(v.begin(),v.end(),w.begin(),0.0);for(int i=0;i<d;++i)v[i]-=c*w[i];}const double norm=std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));if(norm>1e-10){for(double&x:v)x/=norm;probes.push_back(std::move(v));if(sources)sources->emplace_back(source);if(origins){ProbeOrigin origin;if(seed_atom>=0){origin.seed_atom_index=seed_atom;origin.type_index=types[seed_atom];origin.mass=sqrt_mass[seed_atom]*sqrt_mass[seed_atom];}origins->push_back(origin);}return true;}return false;};
  for(int k=0;k<std::min(16,internal);++k){std::vector<double> v(d);for(double&x:v){state^=state>>12;state^=state<<25;state^=state>>27;x=(static_cast<double>((state*2685821657736338717ULL)>>11)/9007199254740992.0)-0.5;}add(std::move(v),"FIXED_RANDOM",-1);}
  std::set<int> seen_types;for(int atom=0;atom<n&&seen_types.size()<4&&static_cast<int>(probes.size())<internal;++atom)if(seen_types.insert(types[atom]).second){std::vector<double> v(d);for(int a=0;a<3;++a)v[a*n+atom]=sqrt_mass[atom];add(std::move(v),"TYPE_LOCAL",atom);}
  int low_mode_probes=0;for(std::size_t i=0;i<soft.size()&&low_mode_probes<4&&static_cast<int>(probes.size())<internal;++i)if(add(soft[i].vector,"LOW_RITZ",-1))++low_mode_probes;
  if(probes.empty())throw std::runtime_error("native fit could not construct an independent response probe");return probes;
}

struct ProbeMoments
{
  int m=0;
  std::uint64_t frames=0;
  std::vector<double> sum_q,sum_f,sum_f2,sum_qq,sum_fq;
  explicit ProbeMoments(const int count=0):m(count),sum_q(count,0.0),sum_f(count,0.0),sum_f2(count,0.0),sum_qq(static_cast<std::size_t>(count)*count,0.0),sum_fq(sum_qq.size(),0.0){}
  void add(const std::vector<double>& q,const std::vector<double>& f)
  {
    ++frames;
    for(int i=0;i<m;++i){sum_q[i]+=q[i];sum_f[i]+=f[i];sum_f2[i]+=f[i]*f[i];}
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){const std::size_t ij=static_cast<std::size_t>(i)*m+j;sum_qq[ij]+=q[i]*q[j];sum_fq[ij]+=f[i]*q[j];}
  }
  void merge(const ProbeMoments& other)
  {
    if(m!=other.m)throw std::runtime_error("native fit bootstrap moment dimensions do not match");
    frames+=other.frames;
    for(std::size_t i=0;i<sum_q.size();++i){sum_q[i]+=other.sum_q[i];sum_f[i]+=other.sum_f[i];sum_f2[i]+=other.sum_f2[i];}
    for(std::size_t i=0;i<sum_qq.size();++i){sum_qq[i]+=other.sum_qq[i];sum_fq[i]+=other.sum_fq[i];}
  }
  std::vector<double> covariance_q() const
  {
    std::vector<double> out(sum_qq.size());if(!frames)return out;const double count=static_cast<double>(frames);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){const std::size_t ij=static_cast<std::size_t>(i)*m+j;out[ij]=sum_qq[ij]/count-(sum_q[i]/count)*(sum_q[j]/count);}return out;
  }
  std::vector<double> covariance_fq() const
  {
    std::vector<double> out(sum_fq.size());if(!frames)return out;const double count=static_cast<double>(frames);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){const std::size_t ij=static_cast<std::size_t>(i)*m+j;out[ij]=sum_fq[ij]/count-(sum_f[i]/count)*(sum_q[j]/count);}return out;
  }
  double centered_variance(const double sum_square,const double sum) const
  {
    if(!frames||!std::isfinite(sum_square)||!std::isfinite(sum))return std::numeric_limits<double>::quiet_NaN();
    const double count=static_cast<double>(frames),mean=sum/count,second=sum_square/count;
    if(!std::isfinite(mean)||!std::isfinite(second))return std::numeric_limits<double>::quiet_NaN();
    const double value=second-mean*mean,scale=std::abs(second)+mean*mean;
    if(!std::isfinite(value)||!std::isfinite(scale))return std::numeric_limits<double>::quiet_NaN();
    const double roundoff=64.0*std::numeric_limits<double>::epsilon()*scale;
    return value<0.0?(value>=-roundoff?0.0:std::numeric_limits<double>::quiet_NaN()):value;
  }
  std::vector<double> variance_f() const
  {
    std::vector<double> result(m,std::numeric_limits<double>::quiet_NaN());
    for(int i=0;i<m;++i)result[i]=centered_variance(sum_f2[i],sum_f[i]);return result;
  }
  std::vector<double> variance_q() const
  {
    std::vector<double> result(m,std::numeric_limits<double>::quiet_NaN());
    for(int i=0;i<m;++i)result[i]=centered_variance(sum_qq[static_cast<std::size_t>(i)*m+i],sum_q[i]);return result;
  }
  std::vector<double> ibp_matrix(const double kbt) const
  {
    std::vector<double> out(sum_fq.size());if(!frames)return out;const double count=static_cast<double>(frames);
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){const std::size_t ij=static_cast<std::size_t>(i)*m+j;out[ij]=sum_fq[ij]/(count*kbt)-(sum_f[i]/count)*(sum_q[j]/count)/kbt+(i==j?1.0:0.0);}return out;
  }
};

struct SourceAtomMotion
{
  int atom_index=-1,type_index=-1;
  double mass=std::numeric_limits<double>::quiet_NaN();
  std::array<std::array<double,3>,4> block_displacement_sum{};
  std::array<std::uint64_t,4> block_frames{};
  std::uint64_t frames=0;
  double squared_displacement_sum=0.0;
  double maximum_adjacent_sampled_frame_displacement=std::numeric_limits<double>::quiet_NaN();
  std::array<double,3> previous_position{};
  bool has_previous=false;

  void add(const std::vector<double>& position,const std::vector<double>& reference,const int n,const int block)
  {
    ++frames;++block_frames[block];std::array<double,3> current{};double displacement2=0.0,adjacent2=0.0;
    for(int axis=0;axis<3;++axis){const std::size_t index=static_cast<std::size_t>(axis)*n+atom_index;
      current[axis]=position[index];const double delta=position[index]-reference[index];block_displacement_sum[block][axis]+=delta;displacement2+=delta*delta;
      if(has_previous){const double adjacent=current[axis]-previous_position[axis];adjacent2+=adjacent*adjacent;}}
    squared_displacement_sum+=displacement2;
    if(has_previous){const double adjacent=std::sqrt(adjacent2);maximum_adjacent_sampled_frame_displacement=
      std::isfinite(maximum_adjacent_sampled_frame_displacement)?std::max(maximum_adjacent_sampled_frame_displacement,adjacent):adjacent;}
    previous_position=current;has_previous=true;
  }

  double mean_square_displacement() const
  {return frames?squared_displacement_sum/static_cast<double>(frames):std::numeric_limits<double>::quiet_NaN();}

  std::array<double,3> mean_displacement(const int block) const
  {std::array<double,3> mean;mean.fill(std::numeric_limits<double>::quiet_NaN());if(block_frames[block])for(int axis=0;axis<3;++axis)mean[axis]=block_displacement_sum[block][axis]/static_cast<double>(block_frames[block]);return mean;}
};

struct IBPNoiseChannel
{
  int i=0,j=0;
  std::string selection_reason;
  double training_entry=std::numeric_limits<double>::quiet_NaN();
  double validation_entry=std::numeric_limits<double>::quiet_NaN();
  double training_variance_proxy=std::numeric_limits<double>::quiet_NaN();
  double validation_variance_proxy=std::numeric_limits<double>::quiet_NaN();
};

double gaussian_variance_proxy(const double variance_f,const double variance_q,const double covariance,const double kbt)
{
  if(!(kbt>0.0)||!std::isfinite(kbt)||!std::isfinite(variance_f)||!std::isfinite(variance_q)||!std::isfinite(covariance))
    return std::numeric_limits<double>::quiet_NaN();
  const double numerator=variance_f*variance_q+covariance*covariance,denominator=kbt*kbt;
  if(!std::isfinite(numerator)||!std::isfinite(denominator)||denominator==0.0)return std::numeric_limits<double>::quiet_NaN();
  const double proxy=numerator/denominator;return std::isfinite(proxy)?proxy:std::numeric_limits<double>::quiet_NaN();
}

double gaussian_variance_proxy(const ProbeMoments& moments,const int i,const int j,const double kbt)
{
  if(i<0||j<0||i>=moments.m||j>=moments.m||!moments.frames)return std::numeric_limits<double>::quiet_NaN();
  const double count=static_cast<double>(moments.frames),mean_f=moments.sum_f[i]/count,mean_q=moments.sum_q[j]/count;
  const double covariance=moments.sum_fq[static_cast<std::size_t>(i)*moments.m+j]/count-mean_f*mean_q;
  const double variance_f=moments.centered_variance(moments.sum_f2[i],moments.sum_f[i]);
  const double variance_q=moments.centered_variance(moments.sum_qq[static_cast<std::size_t>(j)*moments.m+j],moments.sum_q[j]);
  return gaussian_variance_proxy(variance_f,variance_q,covariance,kbt);
}

std::vector<IBPNoiseChannel> select_ibp_noise_channels(const ProbeMoments& training,const double kbt)
{
  if(!training.frames||!(kbt>0.0)||!std::isfinite(kbt))return {};
  struct Candidate{int i,j;double score;};std::vector<Candidate> proxy_candidates,entry_candidates;
  const std::vector<double> ibp=training.ibp_matrix(kbt);
  for(int i=0;i<training.m;++i)for(int j=0;j<training.m;++j){const std::size_t ij=static_cast<std::size_t>(i)*training.m+j;
    const double proxy=gaussian_variance_proxy(training,i,j,kbt),entry=ibp[ij];
    if(i!=j&&std::isfinite(proxy))proxy_candidates.push_back({i,j,proxy});
    if(std::isfinite(entry))entry_candidates.push_back({i,j,std::abs(entry)});
  }
  const auto order=[](const Candidate& a,const Candidate& b){return a.score!=b.score?a.score>b.score:(a.i!=b.i?a.i<b.i:a.j<b.j);};
  std::sort(proxy_candidates.begin(),proxy_candidates.end(),order);std::sort(entry_candidates.begin(),entry_candidates.end(),order);
  std::vector<IBPNoiseChannel> selected;
  const auto append=[&](const Candidate& candidate,const char* reason){
    auto found=std::find_if(selected.begin(),selected.end(),[&](const IBPNoiseChannel& channel){return channel.i==candidate.i&&channel.j==candidate.j;});
    if(found==selected.end()){IBPNoiseChannel channel;channel.i=candidate.i;channel.j=candidate.j;channel.selection_reason=reason;selected.push_back(std::move(channel));}
    else if(found->selection_reason.find(reason)==std::string::npos)found->selection_reason+="+"+std::string(reason);
  };
  for(std::size_t k=0;k<std::min<std::size_t>(3,proxy_candidates.size());++k)append(proxy_candidates[k],"TOP3_GAUSSIAN_VARIANCE_PROXY");
  for(std::size_t k=0;k<std::min<std::size_t>(3,entry_candidates.size());++k)append(entry_candidates[k],"TOP3_ABS_TRAIN_IBP_ENTRY");
  return selected;
}

ProbeMoments select_probe_moments(const ProbeMoments& moments,const std::vector<int>& probes)
{
  ProbeMoments selected(static_cast<int>(probes.size()));selected.frames=moments.frames;
  for(std::size_t i=0;i<probes.size();++i){const int source_i=probes[i];if(source_i<0||source_i>=moments.m)throw std::runtime_error("native fit selected noise probe index is invalid");
    selected.sum_q[i]=moments.sum_q[source_i];selected.sum_f[i]=moments.sum_f[source_i];selected.sum_f2[i]=moments.sum_f2[source_i];
    for(std::size_t j=0;j<probes.size();++j){const int source_j=probes[j];const std::size_t dst=i*probes.size()+j,src=static_cast<std::size_t>(source_i)*moments.m+source_j;selected.sum_qq[dst]=moments.sum_qq[src];selected.sum_fq[dst]=moments.sum_fq[src];}}
  return selected;
}

std::pair<double,double> signed_channel_interval(const double estimate,const double radius)
{return {estimate-radius,estimate+radius};}

double max_abs_symmetric_eigenvalue(std::vector<double> a,const int n)
{
  if(n<=0||a.size()!=static_cast<std::size_t>(n)*n)return std::numeric_limits<double>::quiet_NaN();
  double scale=0.0;for(double value:a){if(!std::isfinite(value))return std::numeric_limits<double>::quiet_NaN();scale=std::max(scale,std::abs(value));}
  if(!std::isfinite(scale))return std::numeric_limits<double>::quiet_NaN();
  if(scale==0.0)return 0.0;
  for(double& value:a)value/=scale;
  for(int sweep=0;sweep<64;++sweep){double largest=0.0,diagonal=0.0;
    for(int i=0;i<n;++i)diagonal=std::max(diagonal,std::abs(a[static_cast<std::size_t>(i)*n+i]));
    for(int p=0;p<n;++p)for(int q=p+1;q<n;++q){const std::size_t pq=static_cast<std::size_t>(p)*n+q;const double apq=a[pq];largest=std::max(largest,std::abs(apq));
      if(std::abs(apq)<=8.0*std::numeric_limits<double>::epsilon()*std::max(diagonal,1.0e-300))continue;
      const double app=a[static_cast<std::size_t>(p)*n+p],aqq=a[static_cast<std::size_t>(q)*n+q];
      const double tau=(aqq-app)/(2.0*apq);const double t=std::copysign(1.0,tau)/(std::abs(tau)+std::hypot(1.0,tau));
      const double c=1.0/std::sqrt(1.0+t*t),s=t*c;
      a[static_cast<std::size_t>(p)*n+p]=app-t*apq;a[static_cast<std::size_t>(q)*n+q]=aqq+t*apq;
      a[pq]=a[static_cast<std::size_t>(q)*n+p]=0.0;
      for(int k=0;k<n;++k)if(k!=p&&k!=q){const double akp=a[static_cast<std::size_t>(k)*n+p],akq=a[static_cast<std::size_t>(k)*n+q];
        a[static_cast<std::size_t>(k)*n+p]=a[static_cast<std::size_t>(p)*n+k]=c*akp-s*akq;
        a[static_cast<std::size_t>(k)*n+q]=a[static_cast<std::size_t>(q)*n+k]=s*akp+c*akq;}
    }
    if(largest<=32.0*std::numeric_limits<double>::epsilon()*std::max(diagonal,1.0e-300))break;
  }
  double result=0.0;for(int i=0;i<n;++i)result=std::max(result,std::abs(a[static_cast<std::size_t>(i)*n+i]));
  result*=scale;return std::isfinite(result)?result:std::numeric_limits<double>::quiet_NaN();
}

double nonsymmetric_spectral_norm(const std::vector<double>& matrix,const int n)
{
  if(n<=0||matrix.size()!=static_cast<std::size_t>(n)*n||
     !std::all_of(matrix.begin(),matrix.end(),[](double value){return std::isfinite(value);}))
    return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> gram(static_cast<std::size_t>(n)*n,0.0);
  for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k)
    gram[static_cast<std::size_t>(i)*n+j]+=matrix[static_cast<std::size_t>(k)*n+i]*matrix[static_cast<std::size_t>(k)*n+j];
  if(!std::all_of(gram.begin(),gram.end(),[](double value){return std::isfinite(value);}))return std::numeric_limits<double>::quiet_NaN();
  const double eigenvalue=max_abs_symmetric_eigenvalue(std::move(gram),n);
  return std::isfinite(eigenvalue)&&eigenvalue>=0.0?std::sqrt(eigenvalue):std::numeric_limits<double>::quiet_NaN();
}

struct BootstrapBand
{
  double estimate=std::numeric_limits<double>::quiet_NaN();
  double radius=std::numeric_limits<double>::quiet_NaN();
  double lower=std::numeric_limits<double>::quiet_NaN();
  double upper=std::numeric_limits<double>::quiet_NaN();
  std::uint64_t frames=0;
  bool block_lengths_stable=false;
  std::string status="STATISTICS_INCONCLUSIVE";
  std::string diagnostic_stage="NOT_COMPUTED",rejection_reason="NOT_COMPUTED",rejection_detail;
  bool bootstrap_started=false;
  std::vector<std::pair<int,double>> level_radii;
};

bool ibp_statistics_numerically_invalid(const BootstrapBand& band)
{return band.status=="NUMERICAL_FAILURE"||!std::isfinite(band.estimate)||
  (band.diagnostic_stage=="INTERVAL_CHECK"&&(!std::isfinite(band.radius)||!std::isfinite(band.lower)||!std::isfinite(band.upper)));}

std::vector<int> bootstrap_block_factors(const std::size_t block_count)
{
  const int limit=static_cast<int>(std::min<std::size_t>(128,block_count/16));
  std::vector<int> factors;
  for(int factor=1;factor<=limit;factor*=2)factors.push_back(factor);
  if(!factors.empty()&&factors.back()!=limit)factors.push_back(limit);
  return factors;
}

bool block_product_tail_covered(const std::vector<ProbeMoments>& blocks,const ProbeMoments& segment,
  BootstrapBand* diagnostic=nullptr,const bool check_fq=true,const bool check_qq=true)
{
  const int n=static_cast<int>(blocks.size()),m=blocks.empty()?segment.m:blocks.front().m;
  const std::vector<int> factors=bootstrap_block_factors(blocks.size());
  const int max_factor=factors.empty()?0:factors.back();
  int product=-1,element=-1,lag=-1,last_correlated_lag=-1,quiet_lags=-1;
  const double uncomputed=std::numeric_limits<double>::quiet_NaN();double variance=uncomputed,noise_bound=uncomputed,integrated_correlation=uncomputed;
  const auto reject=[&](const char* reason){
    if(diagnostic){diagnostic->rejection_reason=reason;if(std::strcmp(reason,"NONFINITE_PRODUCT_STATISTICS")==0)diagnostic->status="NUMERICAL_FAILURE";std::ostringstream out;out<<std::setprecision(17)
      <<"product="<<(product<0?"NOT_COMPUTED":(product==0?"fq":"qq"))<<" probe_i="<<(element<0||m<=0?-1:element/m)<<" probe_j="<<(element<0||m<=0?-1:element%m)
      <<" base_blocks="<<n<<" base_block_frames="<<(blocks.empty()?0:blocks.front().frames)<<" available_factor="<<max_factor
      <<" product_variance="<<variance<<" lag_blocks="<<std::min(lag,max_factor)<<" acf_noise_bound="<<noise_bound
      <<" last_correlated_lag_blocks="<<last_correlated_lag<<" quiet_lags="<<quiet_lags<<" estimated_iat_blocks="<<integrated_correlation
      <<" required_factor="<<(last_correlated_lag>=0&&std::isfinite(integrated_correlation)?4*std::max(static_cast<double>(last_correlated_lag),integrated_correlation):uncomputed)<<" effective_blocks="<<static_cast<double>(n)/integrated_correlation;
      diagnostic->rejection_detail=out.str();}return false;};
  if(blocks.empty())return reject("EMPTY_BLOCKS");
  if(!check_fq&&!check_qq)return reject("NO_PRODUCT_STATISTICS_SELECTED");
  const bool enough_resolution=n>=64&&max_factor>=8;
  const double family=2.0*m*m*max_factor;
  for(const ProbeMoments& block:blocks)if(block.frames!=blocks.front().frames)return reject("NONUNIFORM_BLOCKS");
  if(!segment.frames)return reject("EMPTY_SEGMENT");
  const double count=static_cast<double>(segment.frames);std::vector<double> mean_q(m),mean_f(m,0.0);
  for(int i=0;i<m;++i){mean_q[i]=segment.sum_q[i]/count;if(check_fq)mean_f[i]=segment.sum_f[i]/count;}
  std::vector<double> values(static_cast<std::size_t>(n));
  for(product=0;product<2;++product)for(element=0;element<m*m;++element){
    if((product==0&&!check_fq)||(product==1&&!check_qq))continue;
    variance=noise_bound=integrated_correlation=uncomputed;lag=last_correlated_lag=quiet_lags=-1;
    const int i=element/m,j=element%m;double mean=0.0;
    for(int b=0;b<n;++b){const ProbeMoments& moment=blocks[b];if(!moment.frames)return reject("EMPTY_BLOCK");
      const double block_count=static_cast<double>(moment.frames);
      const double raw=product==0?
        (moment.sum_fq[element]-mean_f[i]*moment.sum_q[j]-mean_q[j]*moment.sum_f[i]+block_count*mean_f[i]*mean_q[j])/block_count:
        (moment.sum_qq[element]-mean_q[i]*moment.sum_q[j]-mean_q[j]*moment.sum_q[i]+block_count*mean_q[i]*mean_q[j])/block_count;
      if(!std::isfinite(raw))return reject("NONFINITE_PRODUCT_STATISTICS");values[b]=raw;mean+=raw/static_cast<double>(n);}
    variance=0.0;for(double value:values){const double centered=value-mean;variance+=centered*centered/static_cast<double>(n);}
    if(!std::isfinite(variance))return reject("NONFINITE_PRODUCT_STATISTICS");
    if(variance==0.0)return reject("ZERO_PRODUCT_VARIANCE");
    if(!enough_resolution)return reject("INSUFFICIENT_BLOCKS");
    last_correlated_lag=quiet_lags=0;integrated_correlation=1.0;
    for(lag=1;lag<=max_factor;++lag){
      const double pairs=static_cast<double>(n-lag);noise_bound=std::sqrt(2.0*std::log(2.0*family/0.05)/pairs);
      if(noise_bound>0.25)return reject("ACF_NOISE_TOO_LARGE");
      double covariance=0.0;for(int b=lag;b<n;++b)covariance+=(values[b]-mean)*(values[b-lag]-mean)/pairs;
      const double rho=covariance/variance;if(!std::isfinite(rho))return reject("NONFINITE_PRODUCT_STATISTICS");integrated_correlation+=2.0*std::max(0.0,rho);
      if(std::abs(rho)>noise_bound){last_correlated_lag=lag;quiet_lags=0;}else ++quiet_lags;
    }
    // Require a resolved tail, four estimated correlation lengths per longest block,
    // and at least sixteen effective independent product blocks.
    if(quiet_lags<3)return reject("TAIL_UNRESOLVED");
    if(!std::isfinite(integrated_correlation))return reject("NONFINITE_PRODUCT_STATISTICS");
    if(max_factor<4*std::max(static_cast<double>(last_correlated_lag),integrated_correlation))return reject("BLOCK_TOO_SHORT");
    if(static_cast<double>(n)/integrated_correlation<16.0)return reject("TOO_FEW_EFFECTIVE_BLOCKS");
  }
  if(!enough_resolution)return reject("INSUFFICIENT_BLOCKS");
  return true;
}

const char* bootstrap_block_length_status(const BootstrapBand& band)
{return band.block_lengths_stable?"PASS":(band.bootstrap_started?"INCONCLUSIVE":"NOT_COMPUTED");}

void write_bootstrap_diagnostic(std::ostream& out,const char* label,const BootstrapBand& band,const bool newline=true)
{
  out<<label<<" stage="<<band.diagnostic_stage<<" bootstrap_started="<<band.bootstrap_started
    <<" configured_replicates=500 completed_levels="<<band.level_radii.size()<<" reason="<<band.rejection_reason;
  if(!band.rejection_detail.empty())out<<' '<<band.rejection_detail;
  if(newline)out<<'\n';
}

template <typename MatrixMetric,typename Distance>
std::vector<double> resample_percentile_deviations(const std::vector<ProbeMoments>& blocks,const int factor,
  const std::vector<double>& center,const int replicates,const std::uint64_t seed,MatrixMetric matrix_metric,Distance distance)
{
  const int count=static_cast<int>(blocks.size()),m=blocks.front().m;
  std::mt19937_64 random(seed+static_cast<std::uint64_t>(factor));std::uniform_int_distribution<int> pick(0,count-1);
  std::vector<double> deviations;deviations.reserve(replicates);
  for(int r=0;r<replicates;++r){ProbeMoments sample(m);int consumed=0;while(consumed<count){const int start=pick(random),length=std::min(factor,count-consumed);for(int k=0;k<length;++k)sample.merge(blocks[(start+k)%count]);consumed+=length;}
    const double deviation=distance(matrix_metric(sample),center);if(!std::isfinite(deviation))return {std::numeric_limits<double>::quiet_NaN()};deviations.push_back(deviation);}
  std::sort(deviations.begin(),deviations.end());
  return deviations;
}

template <typename MatrixMetric,typename Norm,typename Distance>
BootstrapBand bootstrap_metric_band(const ProbeMoments& full,const std::vector<ProbeMoments>& blocks,
  const int base_block_length,const double tolerance,const std::uint64_t seed,MatrixMetric matrix_metric,
  Norm norm,Distance distance,const char* pass,const char* fail,const char* inconclusive,
  const bool check_fq=true,const bool check_qq=true)
{
  BootstrapBand result;result.status=inconclusive;result.diagnostic_stage="INPUT_CHECK";
  const auto stop=[&](const char* reason){result.rejection_reason=reason;return result;};
  ProbeMoments covered(full.m);for(const ProbeMoments& block:blocks)covered.merge(block);result.frames=covered.frames;
  if(covered.frames>full.frames){result.status="NUMERICAL_FAILURE";return stop("COVERED_FRAMES_EXCEED_TOTAL");}
  const auto finite=[](const std::vector<double>& values){return std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);});};
  if(!finite(full.sum_q)||(check_fq&&(!finite(full.sum_f)||!finite(full.sum_fq)))||(check_qq&&!finite(full.sum_qq))){
    result.status="NUMERICAL_FAILURE";return stop("NONFINITE_INPUT_MOMENTS");
  }
  result.diagnostic_stage="POINT_ESTIMATE";
  const std::vector<double> center=matrix_metric(covered);
  if(!std::all_of(center.begin(),center.end(),[](double value){return std::isfinite(value);})){
    result.status="NUMERICAL_FAILURE";return stop("NONFINITE_POINT_MATRIX");
  }
  result.estimate=norm(center);
  if(!std::isfinite(result.estimate)){result.status="NUMERICAL_FAILURE";return stop("NONFINITE_POINT_NORM");}
  result.diagnostic_stage="INPUT_CHECK";
  for(const ProbeMoments& block:blocks){
    if(!finite(block.sum_q)||(check_fq&&(!finite(block.sum_f)||!finite(block.sum_fq)))||(check_qq&&!finite(block.sum_qq))){
      result.status="NUMERICAL_FAILURE";return stop("NONFINITE_BLOCK_MOMENTS");
    }
  }
  if(std::any_of(blocks.begin(),blocks.end(),[&](const ProbeMoments& block){return block.frames!=static_cast<std::uint64_t>(base_block_length);} ))return stop("BLOCK_LENGTH_MISMATCH");
  if(blocks.empty())return stop("EMPTY_BLOCKS");
  if(base_block_length<=0||!(tolerance>=0.0)||!std::isfinite(tolerance))return stop("INVALID_BOOTSTRAP_ARGUMENTS");
  result.diagnostic_stage="TAIL_CHECK";
  if(!block_product_tail_covered(blocks,covered,&result,check_fq,check_qq))return result;
  result.diagnostic_stage="BOOTSTRAP";result.bootstrap_started=true;
  constexpr int replicates=500;
  for(const int factor:bootstrap_block_factors(blocks.size())){
    auto deviations=resample_percentile_deviations(blocks,factor,center,replicates,seed,matrix_metric,distance);
    if(deviations.empty()||!std::isfinite(deviations.back())){result.status="NUMERICAL_FAILURE";return stop("NONFINITE_BOOTSTRAP_DEVIATION");}
    // The raw 95th percentile under-covered correlated harmonic samples with this finite block count.
    const std::size_t q=static_cast<std::size_t>(std::ceil(0.99*replicates))-1;
    result.level_radii.emplace_back(base_block_length*factor,deviations[q]);
  }
  result.diagnostic_stage="BLOCK_LENGTH_CHECK";
  if(result.level_radii.size()<3)return stop("BOOTSTRAP_LEVELS_TOO_FEW");
  const std::size_t first=result.level_radii.size()-3;
  for(std::size_t i=first+1;i<result.level_radii.size();++i){const double a=result.level_radii[i-1].second,b=result.level_radii[i].second;
    if(std::abs(a-b)>0.25*std::max({a,b,1.0e-12})){
      std::ostringstream detail;detail<<std::setprecision(17)<<"previous_block_frames="<<result.level_radii[i-1].first<<" current_block_frames="<<result.level_radii[i].first
        <<" previous_radius="<<a<<" current_radius="<<b<<" radius_difference="<<std::abs(a-b)<<" allowed_difference="<<0.25*std::max({a,b,1.0e-12});
      result.rejection_detail=detail.str();return stop("BLOCK_RADIUS_UNSTABLE");}}
  result.block_lengths_stable=true;result.radius=result.level_radii.back().second;
  result.lower=std::max(0.0,result.estimate-result.radius);result.upper=result.estimate+result.radius;
  result.diagnostic_stage="INTERVAL_CHECK";
  if(!std::isfinite(result.radius)||!std::isfinite(result.lower)||!std::isfinite(result.upper)){
    result.status="NUMERICAL_FAILURE";return stop("NONFINITE_INTERVAL");}
  result.status=result.upper<=tolerance?pass:(result.lower>tolerance?fail:inconclusive);
  result.rejection_reason=result.upper<=tolerance?"NONE":(result.lower>tolerance?"LOWER_BOUND_EXCEEDS_TOLERANCE":"INTERVAL_OVERLAPS_TOLERANCE");
  return result;
}

double nonsymmetric_matrix_distance(const std::vector<double>& a,const std::vector<double>& b,const int n)
{if(a.size()!=b.size())return std::numeric_limits<double>::quiet_NaN();std::vector<double> difference(a.size());for(std::size_t i=0;i<a.size();++i)difference[i]=a[i]-b[i];return nonsymmetric_spectral_norm(difference,n);}

double symmetric_matrix_distance(const std::vector<double>& a,const std::vector<double>& b,const int n)
{if(a.size()!=b.size())return std::numeric_limits<double>::quiet_NaN();std::vector<double> difference(a.size());for(std::size_t i=0;i<a.size();++i)difference[i]=a[i]-b[i];return max_abs_symmetric_eigenvalue(std::move(difference),n);}

BootstrapBand bootstrap_ibp_band(const ProbeMoments& full,const std::vector<ProbeMoments>& blocks,
  const int base_block_length,const double kbt,const double tolerance,const std::uint64_t seed)
{
  const int m=full.m;
  return bootstrap_metric_band(full,blocks,base_block_length,tolerance,seed,
    [=](const ProbeMoments& moments){return moments.ibp_matrix(kbt);},
    [=](const std::vector<double>& matrix){return nonsymmetric_spectral_norm(matrix,m);},
    [=](const std::vector<double>& a,const std::vector<double>& b){return nonsymmetric_matrix_distance(a,b,m);},
    "IBP_PASS","IBP_FAIL","IBP_INCONCLUSIVE");
}

struct FixedChannelNoiseDiagnostic
{
  int validation_base_block_length=0;
  std::uint64_t validation_frames_total=0,validation_frames_used=0,validation_frames_omitted=0;
  std::vector<IBPNoiseChannel> channels;
  std::vector<int> tail_probe_original_indices;
  BootstrapBand band;
  std::string status="NOT_COMPUTED";
  std::string variance_status="AVAILABLE";
};

FixedChannelNoiseDiagnostic diagnose_fixed_ibp_channels(const ProbeMoments& training,const std::vector<ProbeMoments>& validation_blocks,
  const int base_block_length,const std::uint64_t validation_frames,const double kbt)
{
  FixedChannelNoiseDiagnostic result;result.validation_base_block_length=base_block_length;result.validation_frames_total=validation_frames;
  ProbeMoments validation(training.m);for(const ProbeMoments& block:validation_blocks)validation.merge(block);
  result.validation_frames_used=validation.frames;result.validation_frames_omitted=validation_frames>=validation.frames?validation_frames-validation.frames:0;
  const auto finite=[](const std::vector<double>& values){return std::all_of(values.begin(),values.end(),[](double value){return std::isfinite(value);});};
  const auto variance_diagnostic_bad=[&](const ProbeMoments& moments){if(!moments.frames)return false;
    if(!finite(moments.variance_f())||!finite(moments.variance_q()))return true;
    for(int i=0;i<moments.m;++i)for(int j=0;j<moments.m;++j)if(!std::isfinite(gaussian_variance_proxy(moments,i,j,kbt)))return true;return false;};
  const bool train_variance_bad=variance_diagnostic_bad(training),validation_variance_bad=variance_diagnostic_bad(validation);
  if(train_variance_bad||validation_variance_bad)result.variance_status=train_variance_bad&&validation_variance_bad?"NUMERICAL_FAILURE_TRAIN_AND_VALIDATION":(train_variance_bad?"NUMERICAL_FAILURE_TRAIN":"NUMERICAL_FAILURE_VALIDATION");
  result.channels=select_ibp_noise_channels(training,kbt);
  const std::vector<double> train_ibp=training.ibp_matrix(kbt),validation_ibp=validation.ibp_matrix(kbt);
  for(IBPNoiseChannel& channel:result.channels){const std::size_t ij=static_cast<std::size_t>(channel.i)*training.m+channel.j;
    channel.training_entry=training.frames?train_ibp[ij]:std::numeric_limits<double>::quiet_NaN();
    channel.validation_entry=validation.frames?validation_ibp[ij]:std::numeric_limits<double>::quiet_NaN();
    channel.training_variance_proxy=gaussian_variance_proxy(training,channel.i,channel.j,kbt);
    channel.validation_variance_proxy=gaussian_variance_proxy(validation,channel.i,channel.j,kbt);}
  if(result.channels.empty()){result.status=result.variance_status=="AVAILABLE"?"NO_FINITE_TRAIN_CHANNELS":result.variance_status;result.band.rejection_reason=result.status;return result;}

  std::set<int> probe_set;for(const IBPNoiseChannel& channel:result.channels){probe_set.insert(channel.i);probe_set.insert(channel.j);}
  const std::vector<int> selected_probes(probe_set.begin(),probe_set.end());result.tail_probe_original_indices=selected_probes;std::vector<int> local_index(training.m,-1);
  for(std::size_t i=0;i<selected_probes.size();++i)local_index[selected_probes[i]]=static_cast<int>(i);
  ProbeMoments selected_validation=select_probe_moments(validation,selected_probes);std::vector<ProbeMoments> selected_blocks;selected_blocks.reserve(validation_blocks.size());
  for(const ProbeMoments& block:validation_blocks)selected_blocks.push_back(select_probe_moments(block,selected_probes));
  const auto channel_entries=[channels=result.channels,local_index,kbt](const ProbeMoments& moments){
    const std::vector<double> covariance=moments.covariance_fq();std::vector<double> entries;entries.reserve(channels.size());
    for(const IBPNoiseChannel& channel:channels){const int i=local_index[channel.i],j=local_index[channel.j];
      entries.push_back(covariance[static_cast<std::size_t>(i)*moments.m+j]/kbt+(channel.i==channel.j?1.0:0.0));}
    return entries;};
  const auto norm=[](const std::vector<double>& values){double maximum=0.0;for(double value:values){if(!std::isfinite(value))return std::numeric_limits<double>::quiet_NaN();maximum=std::max(maximum,std::abs(value));}return maximum;};
  const auto distance=[norm](const std::vector<double>& a,const std::vector<double>& b){if(a.size()!=b.size())return std::numeric_limits<double>::quiet_NaN();std::vector<double> difference(a.size());for(std::size_t i=0;i<a.size();++i)difference[i]=a[i]-b[i];return norm(difference);};
  result.band=bootstrap_metric_band(selected_validation,selected_blocks,base_block_length,std::numeric_limits<double>::max(),
    0xa54ff53a5f1d36f1ULL,channel_entries,norm,distance,
    "CHANNEL_INTERVAL_DIAGNOSTIC","CHANNEL_INTERVAL_DIAGNOSTIC","CHANNEL_INTERVAL_NOT_COMPUTED");
  const bool interval_available=result.band.block_lengths_stable&&std::isfinite(result.band.radius)&&
    std::isfinite(result.band.lower)&&std::isfinite(result.band.upper);
  if(result.band.block_lengths_stable&&!interval_available)result.band.rejection_reason="NONFINITE_CHANNEL_INTERVAL";
  result.status=interval_available?"COMPUTED_BLOCK_BOOTSTRAP_INTERVAL":"NOT_COMPUTED:"+result.band.rejection_reason;
  return result;
}

struct FixedProbeStatistics
{
  std::vector<std::vector<double>> probes;
  std::vector<std::string> sources;
  std::vector<ProbeOrigin> origins;
  std::vector<SourceAtomMotion> source_atom_motion;
  std::vector<ProbeMoments> segments;
  int all_base_block_length=0;
  std::vector<ProbeMoments> all_blocks;
  BootstrapBand ibp_band;
  int validation_base_block_length=0;
  std::vector<ProbeMoments> validation_blocks;
  ProbeMoments validation_complete_moments;
  FixedChannelNoiseDiagnostic fixed_channel_noise;
  struct LagMoments
  {
    std::array<int,4> lag_steps{{1,2,5,10}};
    int m=0;
    std::array<std::uint64_t,4> pairs{};
    std::vector<double> sum_product,sum_current,sum_previous;
    std::vector<std::vector<double>> history;
    explicit LagMoments(const int count=0):m(count),sum_product(static_cast<std::size_t>(4)*count,0.0),sum_current(sum_product.size(),0.0),sum_previous(sum_product.size(),0.0){}
    void add(const std::vector<double>& q)
    {
      for(int k=0;k<4;++k)if(history.size()>=static_cast<std::size_t>(lag_steps[k])){
        const auto& previous=history[history.size()-lag_steps[k]];++pairs[k];
        for(int i=0;i<m;++i){const std::size_t ik=static_cast<std::size_t>(k)*m+i;sum_product[ik]+=q[i]*previous[i];sum_current[ik]+=q[i];sum_previous[ik]+=previous[i];}
      }
      history.push_back(q);if(history.size()>10)history.erase(history.begin());
    }
    std::array<double,4> autocorrelation(const ProbeMoments& moments,const std::vector<double>& covariance,const int probe) const
    {
      std::array<double,4> result;result.fill(std::numeric_limits<double>::quiet_NaN());const double variance=covariance[static_cast<std::size_t>(probe)*m+probe];
      if(!(variance>0.0)||!moments.frames)return result;const double mean=moments.sum_q[probe]/static_cast<double>(moments.frames);
      for(int k=0;k<4;++k)if(pairs[k]){const double count=static_cast<double>(pairs[k]);const std::size_t ik=static_cast<std::size_t>(k)*m+probe;const double lag_covariance=sum_product[ik]/count-mean*(sum_current[ik]+sum_previous[ik])/count+mean*mean;result[k]=lag_covariance/variance;}
      return result;
    }
  };
  std::vector<LagMoments> lags;
};

void write_probe_origin_table(std::ostream& out,const std::vector<std::string>& sources,const std::vector<ProbeOrigin>& origins)
{
  out<<"\nPROBE_ORIGINS\nprobe_index source seed_atom_index type_index mass\n";
  for(std::size_t i=0;i<sources.size();++i){const ProbeOrigin origin=i<origins.size()?origins[i]:ProbeOrigin{};
    out<<i<<' '<<sources[i]<<' '<<origin.seed_atom_index<<' '<<origin.type_index<<' '<<origin.mass<<'\n';}
  out<<"probe_direction_note type-local seed vectors are translation-projected and orthogonalized; the seed atom is not the final probe direction\n";
}

void write_probe_origin_diagnostics(std::ostream& out,const FixedProbeStatistics& stats)
{
  write_probe_origin_table(out,stats.sources,stats.origins);
  out<<"\nSOURCE_ATOM_MOTION\natom_index type_index mass block0_frames block0_mean_dx block0_mean_dy block0_mean_dz block1_frames block1_mean_dx block1_mean_dy block1_mean_dz block2_frames block2_mean_dx block2_mean_dy block2_mean_dz block3_frames block3_mean_dx block3_mean_dy block3_mean_dz mean_square_displacement_from_training_R0 maximum_adjacent_sampled_frame_displacement\n";
  for(const SourceAtomMotion& motion:stats.source_atom_motion){out<<motion.atom_index<<' '<<motion.type_index<<' '<<motion.mass;
    for(int block=0;block<4;++block){const auto mean=motion.mean_displacement(block);out<<' '<<motion.block_frames[block]<<' '<<mean[0]<<' '<<mean[1]<<' '<<mean[2];}
    out<<' '<<motion.mean_square_displacement()<<' '<<motion.maximum_adjacent_sampled_frame_displacement<<'\n';}
}

FixedProbeStatistics collect_fixed_probe_statistics(std::istream& spool,const SampleHeader& header,
  const std::uint64_t frames,const std::uint64_t train,const std::vector<double>& r0,const bool collect_lags=false,
  const double ibp_tolerance=0.15,const Box* branch_check_box=nullptr)
{
  const int n=header.n,d=3*n;std::vector<double> sqrt_atom(n),sqrt_mass(d);
  for(int i=0;i<n;++i){sqrt_atom[i]=std::sqrt(header.masses[i]);for(int a=0;a<3;++a)sqrt_mass[a*n+i]=sqrt_atom[i];}
  FixedProbeStatistics result;result.probes=make_probes({},sqrt_atom,header.types,n,&result.sources,&result.origins);const int m=static_cast<int>(result.probes.size());
  for(const ProbeOrigin& origin:result.origins)if(origin.seed_atom_index>=0){
    SourceAtomMotion motion;motion.atom_index=origin.seed_atom_index;motion.type_index=origin.type_index;motion.mass=origin.mass;
    result.source_atom_motion.push_back(std::move(motion));}
  result.segments.reserve(7);for(int i=0;i<7;++i)result.segments.emplace_back(m);if(collect_lags){result.lags.reserve(7);for(int i=0;i<7;++i)result.lags.emplace_back(m);}
  result.all_base_block_length=static_cast<int>(std::max<std::uint64_t>(1,(frames+1023)/1024));
  const std::uint64_t validation_frames=frames-train;
  if(collect_lags){result.validation_base_block_length=static_cast<int>(std::max<std::uint64_t>(1,(validation_frames+1023)/1024));result.validation_complete_moments=ProbeMoments(m);}
  ProbeMoments all_pending(m),validation_pending(m);int all_pending_frames=0,validation_pending_frames=0;
  std::vector<double> x(d),force(d),q(d),f(d);std::vector<double> qp(m),fp(m);spool.clear();spool.seekg(header.frames);
  for(std::uint64_t frame=0;frame<frames;++frame){double step=0.0;read_frame(spool,x,force,step);if(branch_check_box)validate_fit_frame_branch(x,n,*branch_check_box,r0);
    for(int i=0;i<d;++i){q[i]=sqrt_mass[i]*(x[i]-r0[i]);f[i]=force[i]/sqrt_mass[i];}
    project_translation(q,sqrt_atom,n);project_translation(f,sqrt_atom,n);
    for(int p=0;p<m;++p){qp[p]=std::inner_product(result.probes[p].begin(),result.probes[p].end(),q.begin(),0.0);fp[p]=std::inner_product(result.probes[p].begin(),result.probes[p].end(),f.begin(),0.0);}
    const int block=std::min(3,static_cast<int>(frame*4/frames));const int train_segment=frame<train?1:2;
    for(SourceAtomMotion& motion:result.source_atom_motion)motion.add(x,r0,n,block);
    result.segments[0].add(qp,fp);result.segments[train_segment].add(qp,fp);result.segments[3+block].add(qp,fp);
    all_pending.add(qp,fp);if(++all_pending_frames==result.all_base_block_length){result.all_blocks.push_back(std::move(all_pending));all_pending=ProbeMoments(m);all_pending_frames=0;}
    if(collect_lags&&frame>=train){validation_pending.add(qp,fp);if(++validation_pending_frames==result.validation_base_block_length){result.validation_blocks.push_back(std::move(validation_pending));validation_pending=ProbeMoments(m);validation_pending_frames=0;}}
    if(collect_lags){result.lags[0].add(qp);result.lags[train_segment].add(qp);result.lags[3+block].add(qp);}
  }
  result.ibp_band=bootstrap_ibp_band(result.segments[0],result.all_blocks,result.all_base_block_length,
    K_B*header.temperature,ibp_tolerance,0x6a09e667f3bcc909ULL);
  if(collect_lags){for(const ProbeMoments& block:result.validation_blocks)result.validation_complete_moments.merge(block);
    result.fixed_channel_noise=diagnose_fixed_ibp_channels(result.segments[1],result.validation_blocks,
      result.validation_base_block_length,validation_frames,K_B*header.temperature);}
  return result;
}

const char* probe_sample_status(const std::uint64_t frames,const int probes)
{return frames<=static_cast<std::uint64_t>(probes)?"INSUFFICIENT_SAMPLES":"INCONCLUSIVE";}

double ibp_max_abs(const ProbeMoments& moments,const double kbt)
{double maximum=0.0;for(double value:moments.ibp_matrix(kbt))maximum=std::max(maximum,std::abs(value));return maximum;}

void write_matrix(std::ostream& out,const char* name,const std::vector<double>& matrix,const int n)
{
  out<<name<<"\n"<<std::setprecision(17);for(int i=0;i<n;++i){for(int j=0;j<n;++j)out<<(j?" ":"")<<matrix[static_cast<std::size_t>(i)*n+j];out<<'\n';}
}

void write_text_exclusive(const std::string& path,const std::string& contents,const char* description)
{
  std::FILE* file=std::fopen(path.c_str(),"wx");if(!file)throw std::runtime_error(std::string("cannot create ")+description+" without overwriting existing evidence: "+path);
  const std::size_t written=std::fwrite(contents.data(),1,contents.size(),file);const int close_status=std::fclose(file);const bool failed=written!=contents.size()||close_status!=0;
  if(failed){std::remove(path.c_str());throw std::runtime_error(std::string("failed writing ")+description+": "+path);}
}

void print_fixed_probe_summary(const FixedProbeStatistics& stats,const SampleHeader& header,const double interval,
  const std::uint64_t train,const double ibp_tolerance=0.15,const double elapsed_seconds=0.0)
{
  const double kbt=K_B*header.temperature;const ProbeMoments& tr=stats.segments[1];const ProbeMoments& va=stats.segments[2];
  const auto tr_cov=tr.covariance_q(),va_cov=va.covariance_q();double variance_delta=0.0;
  for(int i=0;i<tr.m;++i){const double a=tr_cov[static_cast<std::size_t>(i)*tr.m+i],b=va_cov[static_cast<std::size_t>(i)*tr.m+i];variance_delta=std::max(variance_delta,a>0.0?std::abs(b/a-1.0):std::numeric_limits<double>::infinity());}
  std::printf("SAMPLE_STATISTICS: frames=%llu train=%llu validation=%llu P=%d T=%.9g K interval=%.9g probes=%d\n",
    static_cast<unsigned long long>(header.frame_count),static_cast<unsigned long long>(train),static_cast<unsigned long long>(header.frame_count-train),header.beads,header.temperature,interval,tr.m);
  std::printf("fixed-probe IBP max_abs_entry: all=%.6g train=%.6g validation=%.6g\n",ibp_max_abs(stats.segments[0],kbt),ibp_max_abs(tr,kbt),ibp_max_abs(va,kbt));
  std::printf("train/validation variance comparison: max_relative_delta=%.6g\n",variance_delta);
  std::printf("fixed-probe IBP bootstrap: frames_used=%llu omitted=%llu ||E||2=%.6g q99_radius=%.6g interval=[%.6g, %.6g] status=%s\n",
    static_cast<unsigned long long>(stats.ibp_band.frames),static_cast<unsigned long long>(header.frame_count-stats.ibp_band.frames),
    stats.ibp_band.estimate,stats.ibp_band.radius,stats.ibp_band.lower,stats.ibp_band.upper,stats.ibp_band.status.c_str());
  std::printf("sampling evidence: %s; tolerance=%.6g (configured_bootstrap_replicates=500; block-length stability=%s)\n",
    stats.ibp_band.status.c_str(),ibp_tolerance,bootstrap_block_length_status(stats.ibp_band));
  std::ostringstream diagnostic;write_bootstrap_diagnostic(diagnostic,"IBP_DIAGNOSTIC",stats.ibp_band);std::printf("%s",diagnostic.str().c_str());
  std::printf("sample_statistics_seconds=%.6f\n",elapsed_seconds);
}

std::string format_fixed_probe_report(const FixedProbeStatistics& stats,const SampleHeader& header,const double interval,
  const std::uint64_t train,const double ibp_tolerance=0.15,const double elapsed_seconds=0.0)
{
  const int m=static_cast<int>(stats.probes.size());const double kbt=K_B*header.temperature;const char* names[]={"all","train","validation","block0","block1","block2","block3"};
  std::ostringstream out;out<<std::setprecision(17)<<"rpmd_ja fixed-probe sample statistics\nframes "<<header.frame_count<<"\ntrain_frames "<<train<<"\nvalidation_frames "<<header.frame_count-train
    <<"\nbeads "<<header.beads<<"\ntemperature_K "<<header.temperature<<"\nsample_interval "<<interval<<"\nprobe_count "<<m
    <<"\nprobe_policy fixed random plus type-local only\nLIMITATION: excludes the four low-frequency Ritz probes generated during fitting; this report cannot replace final response validation.\n"
    <<"IBP definition: Cov(f_probe,q_probe)/(kB*T)+I; each segment uses its own means.\n"
    <<"ibp_tolerance "<<ibp_tolerance<<"\nIBP_frames_used "<<stats.ibp_band.frames<<"\nIBP_frames_omitted "<<(header.frame_count-stats.ibp_band.frames)
    <<"\nIBP_estimate_spectral_norm "<<stats.ibp_band.estimate<<"\nIBP_bootstrap_q99_radius "<<stats.ibp_band.radius
    <<"\nIBP_interval "<<stats.ibp_band.lower<<' '<<stats.ibp_band.upper<<"\nIBP_status "<<stats.ibp_band.status
    <<"\nblock_length_stability "<<bootstrap_block_length_status(stats.ibp_band)<<"\nsample_statistics_seconds "<<elapsed_seconds<<'\n';
  write_probe_origin_diagnostics(out,stats);
  write_bootstrap_diagnostic(out,"IBP_DIAGNOSTIC",stats.ibp_band);
  out<<"bootstrap_levels block_length_frames q99_radius\n";
  for(const auto& level:stats.ibp_band.level_radii)out<<level.first<<' '<<level.second<<'\n';
  const auto write_proxy_segment=[&](const char* segment,const ProbeMoments& moments){
    const std::vector<double> variance_f=moments.variance_f(),variance_q=moments.variance_q(),covariance=moments.covariance_fq();
    for(int i=0;i<m;++i)for(int j=0;j<m;++j){const double cfq=covariance[static_cast<std::size_t>(i)*m+j];const double proxy=gaussian_variance_proxy(variance_f[i],variance_q[j],cfq,kbt);
      out<<segment<<' '<<i<<' '<<j<<' '<<variance_f[i]<<' '<<variance_q[j]<<' '<<cfq<<' '<<proxy<<'\n';}
  };
  out<<"\nIBP_GAUSSIAN_VARIANCE_PROXY\n";
  out<<"proxy_is_diagnostic_only_not_a_Gaussian_assumption_or_acceptance_gate\n";
  out<<"segment i j variance_f variance_q covariance_fq variance_proxy\n";
  write_proxy_segment("train",stats.segments[1]);write_proxy_segment("validation_complete_blocks",stats.validation_complete_moments);
  const FixedChannelNoiseDiagnostic& fixed=stats.fixed_channel_noise;
  out<<"\nIBP_FIXED_CHANNEL_VALIDATION\nselection_source TRAIN_ONLY\nvalidation_evidence ADJACENT_SEGMENT_INTERNAL_CHECK\n";
  out<<"validation_total_frames "<<fixed.validation_frames_total<<"\nvalidation_base_block_length_frames "<<fixed.validation_base_block_length<<"\nvalidation_frames_used "<<fixed.validation_frames_used
    <<"\nvalidation_frames_omitted "<<fixed.validation_frames_omitted<<"\nvalidation_q99_radius "<<fixed.band.radius
    <<"\nuncertainty_radius_method upper_0.99_block_bootstrap_percentile_coverage_not_universally_calibrated"
    <<"\nvariance_diagnostic_status "<<fixed.variance_status<<"\ndiagnostic_status "<<fixed.status<<"\n";
  write_bootstrap_diagnostic(out,"fixed_channel_bootstrap_diagnostic",fixed.band);
  out<<"tail_probe_index_space LOCAL_SELECTED_PROBES\ntail_probe_original_indices";
  for(const int original_index:fixed.tail_probe_original_indices)out<<' '<<original_index;
  out<<"\ntail_probe_scope ALL_FQ_AND_QQ_PRODUCTS_IN_SELECTED_PROBE_SUBSPACE\n";
  out<<"i j selection_reason train_entry validation_entry train_variance_proxy validation_variance_proxy validation_frames_used validation_frames_omitted signed_lower signed_upper variance_diagnostic_status diagnostic_status\n";
  const double uncomputed=std::numeric_limits<double>::quiet_NaN();
  for(const IBPNoiseChannel& channel:fixed.channels){const auto interval=fixed.status=="COMPUTED_BLOCK_BOOTSTRAP_INTERVAL"?
      signed_channel_interval(channel.validation_entry,fixed.band.radius):std::make_pair(uncomputed,uncomputed);
    out<<channel.i<<' '<<channel.j<<' '<<channel.selection_reason<<' '<<channel.training_entry<<' '<<channel.validation_entry<<' '
      <<channel.training_variance_proxy<<' '<<channel.validation_variance_proxy<<' '<<fixed.validation_frames_used<<' '
      <<fixed.validation_frames_omitted<<' '<<interval.first<<' '<<interval.second<<' '<<fixed.variance_status<<' '<<fixed.status<<'\n';}
  for(int s=0;s<7;++s){const ProbeMoments& moments=stats.segments[s];const auto cov=moments.covariance_q(),ibp=moments.ibp_matrix(kbt);
    out<<"\nSEGMENT "<<names[s]<<" frames="<<moments.frames<<" status="<<probe_sample_status(moments.frames,m)<<'\n';
    out<<"probe index source mean_q_probe variance_q_probe mean_f_probe ibp_diagonal\n";
    for(int i=0;i<m;++i){const double count=static_cast<double>(moments.frames);out<<i<<' '<<stats.sources[i]<<' '<<(moments.frames?moments.sum_q[i]/count:0.0)<<' '<<cov[static_cast<std::size_t>(i)*m+i]<<' '<<(moments.frames?moments.sum_f[i]/count:0.0)<<' '<<ibp[static_cast<std::size_t>(i)*m+i]<<'\n';}
    write_matrix(out,"displacement_covariance",cov,m);write_matrix(out,"force_displacement_covariance",moments.covariance_fq(),m);
    write_matrix(out,"IBP_residual",ibp,m);
    out<<"lag_autocorrelation lags=1,2,5,10\n";
    for(int i=0;i<m;++i){out<<i;for(double rho:stats.lags[s].autocorrelation(moments,cov,i))out<<' '<<rho;out<<'\n';}
  }
  const auto overall=stats.segments[0].covariance_q();
  out<<"\nblock_variance_relative_delta_vs_all\nprobe";for(int b=0;b<4;++b)out<<" block"<<b;out<<'\n';
  for(int i=0;i<m;++i){out<<i;for(int b=0;b<4;++b){const auto cov=stats.segments[3+b].covariance_q();const double base=overall[static_cast<std::size_t>(i)*m+i],value=cov[static_cast<std::size_t>(i)*m+i];out<<' '<<(base>0.0?value/base-1.0:std::numeric_limits<double>::infinity());}out<<'\n';}
  return out.str();
}

struct ResponseCheck{
  ResponseCheck(){response_band.status="NOT_COMPUTED";ibp_band.status="NOT_COMPUTED";}
  double response=std::numeric_limits<double>::quiet_NaN(),ibp=std::numeric_limits<double>::quiet_NaN(),cg=std::numeric_limits<double>::quiet_NaN(),force_residual=std::numeric_limits<double>::quiet_NaN();
  double response_ratio_min=std::numeric_limits<double>::quiet_NaN(),response_ratio_max=std::numeric_limits<double>::quiet_NaN();
  double observed_cov_min=std::numeric_limits<double>::quiet_NaN(),observed_cov_max=std::numeric_limits<double>::quiet_NaN(),observed_cov_condition=std::numeric_limits<double>::quiet_NaN();
  double predicted_cov_min=std::numeric_limits<double>::quiet_NaN(),predicted_cov_max=std::numeric_limits<double>::quiet_NaN(),predicted_cov_condition=std::numeric_limits<double>::quiet_NaN();
  double baseline_apply_seconds=0.0;
  double block_variance_max_relative_delta=std::numeric_limits<double>::quiet_NaN();
  std::array<double,4> block_variance_relative_delta={std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::quiet_NaN()};
  std::array<int,4> block_frames{};
  std::uint64_t total_frames=0,validation_frames=0,statistical_frames=0;
  int probes=0,cg_residual_restarts=0;
  std::string cg_residual_status="NOT_COMPUTED";
  std::string cg_verification_method="FAST";
  bool observed_cov_available=false,predicted_cov_available=false,cg_residual_available=false,response_ibp_available=false,force_residual_available=false,response_extremes_available=false;
  bool has_underprediction=false,has_overprediction=false;
  std::vector<std::string> probe_sources;
  std::vector<ProbeOrigin> probe_origins;
  std::vector<CGProbeRecoveryDiagnostic> cg_probe_recoveries;
  std::vector<double> observed_matrix,predicted_matrix,ibp_matrix,whitened_matrix,worst_response_coefficients,worst_ibp_coefficients;
  std::vector<double> underpredicted_probe_coefficients,overpredicted_probe_coefficients,worst_underpredicted_direction,worst_overpredicted_direction;
  BootstrapBand response_band,ibp_band;
  CGWitness witness;
};

void write_recovered_cg_witnesses(ResponseCheck& response,const std::string& witness_path,const std::vector<double>& theta,
  const std::string& raw_path,const std::string& spool_path,const CGReplayContext& replay,
  std::vector<std::string>& saved_paths)
{
  for(CGProbeRecoveryDiagnostic& diagnostic:response.cg_probe_recoveries)if(diagnostic.recovered){
    const std::string path=witness_path+".outer_"+std::to_string(diagnostic.outer)+".probe_"+std::to_string(diagnostic.probe)+".txt";
    std::ifstream existing(path,std::ios::binary);if(existing.good())throw std::runtime_error("refusing to overwrite recovered CG witness: "+path);
    write_cg_witness(path,diagnostic.original_witness,theta,raw_path,spool_path,replay);diagnostic.witness_snapshot_path=path;saved_paths.push_back(path);
  }
}

std::vector<double> response_error_matrix(const ProbeMoments& moments,const std::vector<double>& predicted,
  const std::vector<double>& invroot)
{
  const int m=moments.m;std::vector<double> observed=moments.covariance_q(),difference(static_cast<std::size_t>(m)*m),left(difference.size()),whitened(difference.size(),0.0);
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double value=.5*observed[static_cast<std::size_t>(i)*m+j]+.5*observed[static_cast<std::size_t>(j)*m+i];observed[static_cast<std::size_t>(i)*m+j]=observed[static_cast<std::size_t>(j)*m+i]=value;}
  for(std::size_t i=0;i<difference.size();++i)difference[i]=observed[i]-predicted[i];
  for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int a=0;a<m;++a)
    left[static_cast<std::size_t>(i)*m+j]+=invroot[static_cast<std::size_t>(i)*m+a]*difference[static_cast<std::size_t>(a)*m+j];
  for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int a=0;a<m;++a)
    whitened[static_cast<std::size_t>(i)*m+j]+=left[static_cast<std::size_t>(i)*m+a]*invroot[static_cast<std::size_t>(a)*m+j];
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double value=.5*whitened[static_cast<std::size_t>(i)*m+j]+.5*whitened[static_cast<std::size_t>(j)*m+i];whitened[static_cast<std::size_t>(i)*m+j]=whitened[static_cast<std::size_t>(j)*m+i]=value;}
  return whitened;
}

std::vector<double> response_probe_basis_coefficients(const std::vector<double>& invroot,
  const std::vector<double>& whitened_eigenvectors,const int m,const int eigenvector)
{
  std::vector<double> coefficients(m,0.0);
  for(int i=0;i<m;++i)for(int j=0;j<m;++j)
    coefficients[i]+=invroot[static_cast<std::size_t>(i)*m+j]*whitened_eigenvectors[static_cast<std::size_t>(j)+static_cast<std::size_t>(eigenvector)*m];
  return coefficients;
}

std::vector<double> response_direction_in_internal_coordinates(const std::vector<std::vector<double>>& probes,
  const std::vector<double>& coefficients)
{
  if(probes.empty()||probes.size()!=coefficients.size())throw std::runtime_error("native fit response direction dimensions do not match");
  const std::size_t d=probes.front().size();std::vector<double> direction(d,0.0);
  for(std::size_t i=0;i<probes.size();++i){if(probes[i].size()!=d)throw std::runtime_error("native fit response probe dimensions do not match");
    for(std::size_t j=0;j<d;++j)direction[j]=std::fma(coefficients[i],probes[i][j],direction[j]);}
  double norm=0.0;for(const double value:direction)norm=std::hypot(norm,value);
  if(!std::isfinite(norm)||!(norm>0.0)||!std::all_of(direction.begin(),direction.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit response direction is non-finite or zero");
  for(double& value:direction)value/=norm;return direction;
}

void set_response_extremes(ResponseCheck& response,const SmallEigen& eigen,const std::vector<double>& invroot,
  const std::vector<std::vector<double>>& probes,const bool save_full_directions)
{
  const int m=response.probes;if(m<=0||eigen.n!=m||eigen.values.size()!=static_cast<std::size_t>(m)||
     eigen.vectors.size()!=static_cast<std::size_t>(m)*m||probes.size()!=static_cast<std::size_t>(m))
    throw std::runtime_error("native fit response eigenvectors have invalid dimensions");
  response.response_ratio_min=1.0+eigen.values.front();response.response_ratio_max=1.0+eigen.values.back();
  if(!std::isfinite(response.response_ratio_min)||!std::isfinite(response.response_ratio_max))throw std::runtime_error("native fit response ratios are non-finite");
  response.has_underprediction=response.response_ratio_max>1.0;response.has_overprediction=response.response_ratio_min<1.0;
  if(response.has_underprediction)response.underpredicted_probe_coefficients=response_probe_basis_coefficients(invroot,eigen.vectors,m,m-1);
  if(response.has_overprediction)response.overpredicted_probe_coefficients=response_probe_basis_coefficients(invroot,eigen.vectors,m,0);
  if(save_full_directions){
    if(response.has_underprediction)response.worst_underpredicted_direction=response_direction_in_internal_coordinates(probes,response.underpredicted_probe_coefficients);
    if(response.has_overprediction)response.worst_overpredicted_direction=response_direction_in_internal_coordinates(probes,response.overpredicted_probe_coefficients);
  }
  response.response_extremes_available=true;
}

void write_response_stats(std::ostream& out,const ResponseCheck& r)
{
  const auto write_cg_value=[&](const double value){if(std::isfinite(value))out<<value;else out<<"NOT_COMPUTED";};
  out<<"validation_frames="<<r.validation_frames<<" statistical_frames="<<r.statistical_frames<<" probes="<<r.probes
    <<" observed_cov_status="<<(r.observed_cov_available?"AVAILABLE":"NOT_COMPUTED")<<" observed_cov_min="<<r.observed_cov_min
    <<" observed_cov_max="<<r.observed_cov_max<<" observed_cov_condition="<<r.observed_cov_condition
    <<" predicted_cov_status="<<(r.predicted_cov_available?"AVAILABLE":"NOT_COMPUTED")<<" predicted_cov_min="<<r.predicted_cov_min<<" predicted_cov_max="<<r.predicted_cov_max
    <<" predicted_cov_condition="<<r.predicted_cov_condition<<" block_frames=";
  for(int i=0;i<4;++i)out<<(i?",":"")<<r.block_frames[i];
  out<<" block_variance_relative_delta=";
  for(int i=0;i<4;++i)out<<(i?",":"")<<r.block_variance_relative_delta[i];
  out<<" block_variance_max_relative_delta="<<r.block_variance_max_relative_delta
    <<" response_uncertainty_status="<<r.response_band.status<<" response_interval="<<r.response_band.lower<<','<<r.response_band.upper
    <<" response_q99_radius="<<r.response_band.radius<<" ibp_uncertainty_status="<<r.ibp_band.status
    <<" ibp_interval="<<r.ibp_band.lower<<','<<r.ibp_band.upper<<" ibp_q99_radius="<<r.ibp_band.radius<<" response_block_radii=";
  for(const auto& level:r.response_band.level_radii)out<<(level.first==r.response_band.level_radii.front().first?"":",")<<level.first<<':'<<level.second;
  out<<" response_ratio_status="<<(r.response_extremes_available?"AVAILABLE":"NOT_COMPUTED")<<" response_ratio_min=";
  write_cg_value(r.response_ratio_min);out<<" response_ratio_max=";write_cg_value(r.response_ratio_max);
  out<<" underprediction="<<(r.response_extremes_available?(r.has_underprediction?"PRESENT":"NONE"):"NOT_COMPUTED")
    <<" overprediction="<<(r.response_extremes_available?(r.has_overprediction?"PRESENT":"NONE"):"NOT_COMPUTED");
  out<<" ibp_block_radii=";
  for(const auto& level:r.ibp_band.level_radii)out<<(level.first==r.ibp_band.level_radii.front().first?"":",")<<level.first<<':'<<level.second;
  out
    <<" baseline_apply_seconds="<<r.baseline_apply_seconds<<" cg_residual_restarts="<<r.cg_residual_restarts<<" cg_residual_status="<<r.cg_residual_status
    <<" cg_failure_classification="<<(r.witness.classification.empty()?"NONE":r.witness.classification)<<" cg_failure_iteration="<<r.witness.iteration
    <<" cg_recursive_relative_residual=";
  if(r.witness.classification.empty())out<<"NOT_RECORDED";else out<<r.witness.recursive_relative_residual;
  const bool true_residual_computed=r.cg_residual_status=="CG_PASS"||r.witness.true_residual_computed;
  out<<" cg_true_residual_checks="<<r.witness.residual_checks.size()<<" cg_true_residual_status="<<(true_residual_computed?"COMPUTED":"NOT_COMPUTED")<<" cg_true_residual=";
  if(true_residual_computed)out<<r.cg;else out<<"NOT_COMPUTED";
  out<<" cg_verification_method="<<r.cg_verification_method<<" cg_probe_recovery_count="<<r.cg_probe_recoveries.size();
  for(std::size_t i=0;i<r.cg_probe_recoveries.size();++i){const CGProbeRecoveryDiagnostic& diagnostic=r.cg_probe_recoveries[i];
    const std::string prefix=" cg_probe_recovery_"+std::to_string(i)+"_";
    out<<prefix<<"probe="<<diagnostic.probe<<prefix<<"initial_cg_status="<<diagnostic.initial_cg_status
      <<prefix<<"verification_method="<<diagnostic.verification_method<<prefix<<"initial_fast_residual=";write_cg_value(diagnostic.initial_fast_residual);
    out<<prefix<<"initial_compensated_residual=";write_cg_value(diagnostic.initial_compensated_residual);
    out<<prefix<<"final_fast_residual=";write_cg_value(diagnostic.final_fast_residual);
    out<<prefix<<"final_compensated_residual=";write_cg_value(diagnostic.final_compensated_residual);
    out<<prefix<<"refinement_status="<<diagnostic.refinement_status<<prefix<<"refinement_rounds="<<diagnostic.refinement_rounds
      <<prefix<<"refinement_actions="<<diagnostic.refinement_matrix_actions<<prefix<<"refinement_seconds=";write_cg_value(diagnostic.refinement_seconds);
    out<<prefix<<"verification_actions="<<diagnostic.verification_matrix_actions<<prefix<<"verification_seconds=";write_cg_value(diagnostic.verification_seconds);
    out<<prefix<<"recovered="<<(diagnostic.recovered?1:0)<<prefix<<"witness_snapshot="<<std::quoted(diagnostic.witness_snapshot_path);
    if(!diagnostic.error.empty())out<<prefix<<"error="<<std::quoted(diagnostic.error);
  }
  out<<" response_ibp_status="<<(r.response_ibp_available?"AVAILABLE":"NOT_COMPUTED")<<" response="<<r.response<<" ibp="<<r.ibp
    <<" force_residual_status="<<(r.force_residual_available?"AVAILABLE":"NOT_COMPUTED")<<" heldout_force_residual="<<r.force_residual;
  out<<' ';write_bootstrap_diagnostic(out,"response_diagnostic",r.response_band,false);
  out<<' ';write_bootstrap_diagnostic(out,"ibp_diagnostic",r.ibp_band,false);
}
ResponseCheck validate_probes(cusolverDnHandle_t solver,DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& theta,
  const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const std::vector<int>& types,const int n,
  const std::vector<double>& r0,std::ifstream& spool,const std::streamoff frame_start,const std::uint64_t frame_count,const int train,
  const int sample_interval,const double temperature,const std::vector<RitzMode>& soft,const double epsilon=0.0,ResponseCheck* progress=nullptr,
  const double response_tolerance=0.15,const double ibp_tolerance=0.15,const int outer=-1,
  const std::function<CGResult(int,const std::vector<double>&)>& solve_probe_override={})
{
  const int d=3*n;std::vector<std::string> probe_sources;std::vector<ProbeOrigin> probe_origins;const auto probes=make_probes(soft,sqrt_atom,types,n,&probe_sources,&probe_origins);const int m=static_cast<int>(probes.size());ResponseCheck result;result.probes=m;result.probe_sources=probe_sources;result.probe_origins=probe_origins;
  if(frame_count<static_cast<std::uint64_t>(train))throw std::runtime_error("native fit held-out frame count is invalid");
  result.total_frames=frame_count;result.validation_frames=frame_count-train;if(progress)*progress=result;
  if(result.validation_frames<=static_cast<std::uint64_t>(m))throw std::runtime_error("native fit held-out frames must exceed the fixed probe count");
  std::vector<double> predicted(static_cast<std::size_t>(m)*m),rhs(d);double max_cg=0.0;int fast_verified=0,compensated_verified=0;
  const auto update_verification_method=[&](){result.cg_verification_method=compensated_verified?(fast_verified?"MIXED":"COMPENSATED"):"FAST";};
  for(int j=0;j<m;++j){CGResult cg=solve_probe_override?solve_probe_override(j,probes[j]):solve_projected_cg(baseline,graph,probes[j],sqrt_mass,sqrt_atom,theta,n,j,epsilon);result.cg_residual_restarts+=cg.residual_restarts;
    if(!cg.witness.classification.empty()){
      result.cg=cg.relative_residual;result.cg_residual_available=cg.witness.true_residual_computed;result.cg_residual_status=cg.witness.classification;result.witness=cg.witness;
      if(cg.witness.classification!="TRUE_RESIDUAL_FAILURE"){if(progress)*progress=result;return result;}
      result.cg_probe_recoveries.emplace_back();CGProbeRecoveryDiagnostic& diagnostic=result.cg_probe_recoveries.back();diagnostic.probe=j;diagnostic.outer=outer;diagnostic.original_witness=cg.witness;
      ++compensated_verified;update_verification_method();if(progress)*progress=result;
      bool recovered=false;try{recovered=recover_cg_true_residual_failure(baseline,graph,probes[j],cg,sqrt_mass,sqrt_atom,theta,n,diagnostic,outer);}
      catch(...){if(progress)*progress=result;throw;}
      result.cg=diagnostic.final_compensated_residual;result.cg_residual_available=std::isfinite(diagnostic.final_compensated_residual);
      result.cg_residual_restarts+=diagnostic.refinement_restarts;
      if(!recovered){if(progress)*progress=result;return result;}
      result.witness=CGWitness{};result.cg_residual_status="CG_PASS";result.cg_residual_available=true;
    }else{++fast_verified;update_verification_method();}
    max_cg=std::max(max_cg,cg.relative_residual);for(int i=0;i<m;++i)predicted[static_cast<std::size_t>(i)*m+j]=K_B*temperature*std::inner_product(probes[i].begin(),probes[i].end(),cg.x.begin(),0.0);
    if(progress)*progress=result;
  }
  result.cg=max_cg;result.cg_residual_available=true;result.cg_residual_status="CG_PASS";if(progress)*progress=result;
  ProbeMoments validation_moments(m);const int bootstrap_block_length=static_cast<int>(std::max<std::uint64_t>(1,(result.validation_frames+1023)/1024));std::vector<ProbeMoments> validation_bootstrap_blocks;ProbeMoments pending_block(m);int pending_count=0;std::vector<double> block_q(static_cast<std::size_t>(4)*m,0.0),block_q2(block_q.size(),0.0),x(d),f(d),q(d),fw(d),add(d),qp(m),fp(m);
  const std::size_t batch_elements=static_cast<std::size_t>(d)*DeviceBaseline::batch_capacity;std::vector<double> q_batch(batch_elements),force_batch(batch_elements),base_batch(batch_elements);
  double residual2=0.0,force2=0.0;spool.clear();spool.seekg(frame_start);double step=0.0,previous=-std::numeric_limits<double>::infinity();
  const auto read_checked_frame=[&](const std::uint64_t frame){read_frame(spool,x,f,step);if(!(step>previous)||(frame&&std::abs((step-previous)-sample_interval)>1e-9))throw std::runtime_error("native fit validation steps do not match sample_interval");previous=step;};
  for(std::uint64_t frame=0;frame<static_cast<std::uint64_t>(train);++frame)read_checked_frame(frame);
  for(std::uint64_t first=static_cast<std::uint64_t>(train);first<frame_count;){
    const int columns=static_cast<int>(std::min<std::uint64_t>(DeviceBaseline::batch_capacity,frame_count-first));
    for(int column=0;column<columns;++column){const std::uint64_t frame=first+column;read_checked_frame(frame);
      for(int i=0;i<d;++i){q[i]=sqrt_mass[i]*(x[i]-r0[i]);fw[i]=f[i]/sqrt_mass[i];}project_translation(q,sqrt_atom,n);project_translation(fw,sqrt_atom,n);
      std::copy(q.begin(),q.end(),q_batch.begin()+static_cast<std::size_t>(column)*d);std::copy(fw.begin(),fw.end(),force_batch.begin()+static_cast<std::size_t>(column)*d);}
    const auto baseline_started=std::chrono::steady_clock::now();baseline.apply_batch(q_batch,columns,base_batch);
    result.baseline_apply_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-baseline_started).count();if(progress)progress->baseline_apply_seconds=result.baseline_apply_seconds;
    for(int column=0;column<columns;++column){const std::uint64_t frame=first+column;const double* base=base_batch.data()+static_cast<std::size_t>(column)*d;
      std::copy_n(q_batch.begin()+static_cast<std::size_t>(column)*d,d,q.begin());std::copy_n(force_batch.begin()+static_cast<std::size_t>(column)*d,d,fw.begin());
      apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i){const double target=fw[i]+base[i];residual2+=(target+add[i])*(target+add[i]);force2+=fw[i]*fw[i];}
      for(int a=0;a<m;++a){qp[a]=std::inner_product(probes[a].begin(),probes[a].end(),q.begin(),0.0);fp[a]=std::inner_product(probes[a].begin(),probes[a].end(),fw.begin(),0.0);}
      validation_moments.add(qp,fp);pending_block.add(qp,fp);if(++pending_count==bootstrap_block_length){validation_bootstrap_blocks.push_back(std::move(pending_block));pending_block=ProbeMoments(m);pending_count=0;}
      const int block=std::min(3,static_cast<int>((frame-train)*4/result.validation_frames));++result.block_frames[block];for(int a=0;a<m;++a){const std::size_t ba=static_cast<std::size_t>(block)*m+a;block_q[ba]+=qp[a];block_q2[ba]+=qp[a]*qp[a];}}
    first+=columns;
  }
  ProbeMoments statistical_moments(m);for(const ProbeMoments& block:validation_bootstrap_blocks)statistical_moments.merge(block);
  result.statistical_frames=statistical_moments.frames;
  std::vector<double> observed=statistical_moments.covariance_q(),ibp=statistical_moments.ibp_matrix(K_B*temperature);
  // Symmetric whitening on the finite, declared probe subspace.
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double v=.5*predicted[static_cast<std::size_t>(i)*m+j]+.5*predicted[static_cast<std::size_t>(j)*m+i];predicted[static_cast<std::size_t>(i)*m+j]=predicted[static_cast<std::size_t>(j)*m+i]=v;}
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double v=.5*observed[static_cast<std::size_t>(i)*m+j]+.5*observed[static_cast<std::size_t>(j)*m+i];observed[static_cast<std::size_t>(i)*m+j]=observed[static_cast<std::size_t>(j)*m+i]=v;}
  const SmallEigen oe=eigen_small(solver,observed,m);result.observed_cov_min=oe.values.front();result.observed_cov_max=oe.values.back();result.observed_cov_condition=oe.values.front()>0.0?oe.values.back()/oe.values.front():std::numeric_limits<double>::infinity();
  result.block_variance_relative_delta.fill(0.0);result.block_variance_max_relative_delta=0.0;for(int b=0;b<4;++b){for(int a=0;a<m;++a){const double overall=observed[static_cast<std::size_t>(a)*m+a];const std::size_t ba=static_cast<std::size_t>(b)*m+a;const double count=result.block_frames[b];const double block_variance=count>0.0?block_q2[ba]/count-(block_q[ba]/count)*(block_q[ba]/count):0.0;const double delta=overall>0.0?std::abs(block_variance/overall-1.0):std::numeric_limits<double>::infinity();result.block_variance_relative_delta[b]=std::max(result.block_variance_relative_delta[b],delta);}result.block_variance_max_relative_delta=std::max(result.block_variance_max_relative_delta,result.block_variance_relative_delta[b]);}
  result.observed_cov_available=true;if(progress)*progress=result;
  if(!(oe.values.front()>0.0)||oe.values.front()<=oe.values.back()*1e-12)throw std::runtime_error("native fit held-out probe covariance is rank deficient or too small to validate");
  const SmallEigen pe=eigen_small(solver,predicted,m);result.predicted_cov_min=pe.values.front();result.predicted_cov_max=pe.values.back();result.predicted_cov_condition=pe.values.front()>0.0?pe.values.back()/pe.values.front():std::numeric_limits<double>::infinity();result.predicted_cov_available=true;if(progress)*progress=result;
  if(!(pe.values.front()>0.0)||pe.values.front()<=pe.values.back()*1e-12)throw std::runtime_error("native fit predicted probe response is non-positive or ill-conditioned");
  std::vector<double> invroot(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)invroot[static_cast<std::size_t>(i)*m+j]+=pe.vectors[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*m]*pe.vectors[static_cast<std::size_t>(j)+static_cast<std::size_t>(k)*m]/std::sqrt(pe.values[k]);
  std::vector<double> whitened(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j){for(int a=0;a<m;++a)for(int b=0;b<m;++b)whitened[static_cast<std::size_t>(i)*m+j]+=invroot[static_cast<std::size_t>(i)*m+a]*(observed[static_cast<std::size_t>(a)*m+b]-predicted[static_cast<std::size_t>(a)*m+b])*invroot[static_cast<std::size_t>(b)*m+j];}
  const SmallEigen re=eigen_small(solver,whitened,m);double response=0.0;int worst_response=0;for(int i=0;i<m;++i)if(std::abs(re.values[i])>response){response=std::abs(re.values[i]);worst_response=i;}
  std::vector<double> gram(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)gram[static_cast<std::size_t>(i)*m+j]+=ibp[static_cast<std::size_t>(k)*m+i]*ibp[static_cast<std::size_t>(k)*m+j];
  const SmallEigen ie=eigen_small(solver,gram,m);const int worst_ibp=m-1;const double ibp_error=std::sqrt(std::max(0.0,ie.values.back()));result.response=response;result.ibp=ibp_error;result.response_ibp_available=true;result.observed_matrix=observed;result.predicted_matrix=predicted;result.ibp_matrix=ibp;result.whitened_matrix=whitened;result.worst_response_coefficients=response_probe_basis_coefficients(invroot,re.vectors,m,worst_response);result.worst_ibp_coefficients.resize(m);for(int i=0;i<m;++i)result.worst_ibp_coefficients[i]=ie.vectors[static_cast<std::size_t>(i)+static_cast<std::size_t>(worst_ibp)*m];if(progress)*progress=result;if(!std::isfinite(response)||!std::isfinite(ibp_error)||!std::isfinite(max_cg))throw std::runtime_error("native fit response diagnostics are non-finite");
  result.response_band=bootstrap_metric_band(validation_moments,validation_bootstrap_blocks,bootstrap_block_length,response_tolerance,
    0xbb67ae8584caa73bULL,[&](const ProbeMoments& moments){return response_error_matrix(moments,predicted,invroot);},
    [=](const std::vector<double>& matrix){return max_abs_symmetric_eigenvalue(matrix,m);},
    [=](const std::vector<double>& a,const std::vector<double>& b){return symmetric_matrix_distance(a,b,m);},
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE",false,true);
  result.ibp_band=bootstrap_metric_band(validation_moments,validation_bootstrap_blocks,bootstrap_block_length,ibp_tolerance,
    0x3c6ef372fe94f82bULL,[&](const ProbeMoments& moments){return moments.ibp_matrix(K_B*temperature);},
    [=](const std::vector<double>& matrix){return nonsymmetric_spectral_norm(matrix,m);},
    [=](const std::vector<double>& a,const std::vector<double>& b){return nonsymmetric_matrix_distance(a,b,m);},
    "IBP_PASS","IBP_FAIL","IBP_INCONCLUSIVE",true,true);
  set_response_extremes(result,re,invroot,probes,result.response>response_tolerance||result.response_band.status!="RESPONSE_PASS");
  if(progress)*progress=result;
  if(!std::isfinite(residual2)||!std::isfinite(force2)||force2<0.0)throw std::runtime_error("native fit held-out force residual is non-finite");const double heldout_residual=force2>0.0?std::sqrt(residual2/force2):(residual2==0.0?0.0:std::numeric_limits<double>::infinity());if(!std::isfinite(heldout_residual))throw std::runtime_error("native fit held-out residual is nonzero at zero physical-force scale");
  result.force_residual=heldout_residual;result.force_residual_available=true;if(progress)*progress=result;return result;
}

void write_response_state(const std::string& path,const ResponseCheck& response,const std::vector<double>& theta,
  const double temperature,const double response_tolerance,const double ibp_tolerance)
{
  const int m=response.probes;std::ostringstream out;out<<std::setprecision(17)<<"rpmd_ja response failure state\nstatus RESPONSE_CHECK_FAIL\ntemperature_K "<<temperature
    <<"\ntotal_frames "<<response.total_frames<<"\nvalidation_frames "<<response.validation_frames<<"\nstatistical_frames "<<response.statistical_frames<<"\nprobe_count "<<m<<"\nresponse_tolerance "<<response_tolerance
    <<"\nibp_tolerance "<<ibp_tolerance<<"\nresponse_error "<<response.response<<"\nibp_error "<<response.ibp
    <<"\nresponse_ratio_definition observed_over_predicted=1+whitened_eigenvalue\nresponse_ratio_status "<<(response.response_extremes_available?"AVAILABLE":"NOT_COMPUTED")
    <<"\nresponse_ratio_min ";
  if(response.response_extremes_available)out<<response.response_ratio_min;else out<<"NOT_COMPUTED";
  out<<"\nresponse_ratio_max ";if(response.response_extremes_available)out<<response.response_ratio_max;else out<<"NOT_COMPUTED";out<<"\ntheta";
  for(double value:theta)out<<' '<<value;
  out<<"\nresponse_uncertainty "<<response.response_band.status<<' '<<response.response_band.lower<<' '<<response.response_band.upper<<' '<<response.response_band.radius
    <<"\nibp_uncertainty "<<response.ibp_band.status<<' '<<response.ibp_band.lower<<' '<<response.ibp_band.upper<<' '<<response.ibp_band.radius<<"\nuncertainty_radius_method upper_0.99_block_bootstrap_percentile; coverage_is_not_universally_calibrated\n";
  write_bootstrap_diagnostic(out,"RESPONSE_DIAGNOSTIC",response.response_band);
  write_bootstrap_diagnostic(out,"IBP_DIAGNOSTIC",response.ibp_band);
  out<<"response_bootstrap_levels block_length_frames q99_radius\n";for(const auto& level:response.response_band.level_radii)out<<level.first<<' '<<level.second<<'\n';
  out<<"ibp_bootstrap_levels block_length_frames q99_radius\n";for(const auto& level:response.ibp_band.level_radii)out<<level.first<<' '<<level.second<<'\n';
  write_probe_origin_table(out,response.probe_sources,response.probe_origins);
  out<<"probe_variances index observed predicted observed_over_predicted\n";
  for(int i=0;i<m;++i){const double observed=response.observed_matrix[static_cast<std::size_t>(i)*m+i],predicted=response.predicted_matrix[static_cast<std::size_t>(i)*m+i];out<<i<<' '<<observed<<' '<<predicted<<' '<<(predicted!=0.0?observed/predicted:std::numeric_limits<double>::infinity())<<'\n';}
  write_matrix(out,"observed_covariance",response.observed_matrix,m);write_matrix(out,"predicted_covariance",response.predicted_matrix,m);
  write_matrix(out,"IBP_residual_nonsymmetric",response.ibp_matrix,m);write_matrix(out,"whitened_response_difference",response.whitened_matrix,m);
  out<<"worst_response_direction_mapping c=P^(-1/2)*u, where u is the whitened eigenvector\n";
  out<<"worst_response_probe_basis_coefficients\n";for(int i=0;i<m;++i)out<<i<<' '<<response.worst_response_coefficients[i]<<'\n';
  out<<"response_direction_coordinate_convention mass_weighted_internal_xyz_soa unit_euclidean_norm\n";
  out<<"worst_underpredicted_direction_status "<<(response.response_extremes_available?(response.has_underprediction?"PRESENT":"NONE"):"NOT_COMPUTED")<<'\n';
  if(response.response_extremes_available&&response.has_underprediction){out<<"worst_underpredicted_response_ratio "<<response.response_ratio_max<<'\n';
    out<<"worst_underpredicted_probe_basis_coefficients\n";for(int i=0;i<m;++i)out<<i<<' '<<response.underpredicted_probe_coefficients[i]<<'\n';
    out<<"worst_underpredicted_direction_mass_weighted_internal_xyz_soa\n";for(std::size_t i=0;i<response.worst_underpredicted_direction.size();++i)out<<i<<' '<<response.worst_underpredicted_direction[i]<<'\n';}
  out<<"worst_overpredicted_direction_status "<<(response.response_extremes_available?(response.has_overprediction?"PRESENT":"NONE"):"NOT_COMPUTED")<<'\n';
  if(response.response_extremes_available&&response.has_overprediction){out<<"worst_overpredicted_response_ratio "<<response.response_ratio_min<<'\n';
    out<<"worst_overpredicted_probe_basis_coefficients\n";for(int i=0;i<m;++i)out<<i<<' '<<response.overpredicted_probe_coefficients[i]<<'\n';
    out<<"worst_overpredicted_direction_mass_weighted_internal_xyz_soa\n";for(std::size_t i=0;i<response.worst_overpredicted_direction.size();++i)out<<i<<' '<<response.worst_overpredicted_direction[i]<<'\n';}
  out<<"worst_IBP_right_singular_probe_basis_coefficients (from eigenvector of E^T E)\n";for(int i=0;i<m;++i)out<<i<<' '<<response.worst_ibp_coefficients[i]<<'\n';
  write_text_exclusive(path,out.str(),"response state snapshot");
}

void householder_compress(std::vector<double>& a, std::vector<double>& y, const int rows, const int cols,
                          std::vector<double>& r, std::vector<double>& z,double& discarded2)
{
  for (int k=0;k<cols;++k) {
    double norm=0.0; for(int i=k;i<rows;++i) norm=std::hypot(norm,a[static_cast<std::size_t>(i)*cols+k]);
    if (norm == 0.0) continue;
    const double x0=a[static_cast<std::size_t>(k)*cols+k];
    const double alpha=std::copysign(norm,x0);
    const double denom=x0+alpha;
    a[static_cast<std::size_t>(k)*cols+k]=-alpha;
    for(int i=k+1;i<rows;++i)a[static_cast<std::size_t>(i)*cols+k]/=denom;
    const double tau=denom/alpha;
    for(int j=k+1;j<cols;++j){double dot=a[static_cast<std::size_t>(k)*cols+j];for(int i=k+1;i<rows;++i)dot+=a[static_cast<std::size_t>(i)*cols+k]*a[static_cast<std::size_t>(i)*cols+j];dot*=tau;a[static_cast<std::size_t>(k)*cols+j]-=dot;for(int i=k+1;i<rows;++i)a[static_cast<std::size_t>(i)*cols+j]-=a[static_cast<std::size_t>(i)*cols+k]*dot;}
    double dot=y[k];for(int i=k+1;i<rows;++i)dot+=a[static_cast<std::size_t>(i)*cols+k]*y[i];dot*=tau;y[k]-=dot;for(int i=k+1;i<rows;++i)y[i]-=a[static_cast<std::size_t>(i)*cols+k]*dot;
  }
  for(int i=cols;i<rows;++i)discarded2+=y[i]*y[i];
  r.assign(static_cast<std::size_t>(cols)*cols,0.0); z.assign(cols,0.0);
  for(int i=0;i<cols;++i){z[i]=y[i];for(int j=i;j<cols;++j)r[static_cast<std::size_t>(i)*cols+j]=a[static_cast<std::size_t>(i)*cols+j];}
}

void merge_qr(std::vector<double>& total_r, std::vector<double>& total_z, const std::vector<double>& next_r,
              const std::vector<double>& next_z, const int p,double& discarded2)
{
  std::vector<double> a(static_cast<std::size_t>(2*p)*p,0.0), y(2*p,0.0);
  for(int i=0;i<p;++i){y[i]=total_z[i];y[p+i]=next_z[i];for(int j=i;j<p;++j){a[static_cast<std::size_t>(i)*p+j]=total_r[static_cast<std::size_t>(i)*p+j];a[static_cast<std::size_t>(p+i)*p+j]=next_r[static_cast<std::size_t>(i)*p+j];}}
  householder_compress(a,y,2*p,p,total_r,total_z,discarded2);
}

double compressed_training_force_residual(const std::vector<double>& rmat,const std::vector<double>& z,
  const double discarded2,const double training_force2,const std::vector<double>& theta)
{
  const std::size_t p=theta.size();if(p==0||rmat.size()!=p*p||z.size()!=p)throw std::runtime_error("native fit compressed residual dimensions do not match");
  double compressed_residual2=discarded2;
  for(std::size_t i=0;i<p;++i){double value=-z[i];for(std::size_t j=i;j<p;++j)value+=rmat[i*p+j]*theta[j];compressed_residual2+=value*value;}
  if(!std::isfinite(compressed_residual2)||!std::isfinite(training_force2)||training_force2<0.0)throw std::runtime_error("native fit training force residual is non-finite");
  const double residual=training_force2>0.0?std::sqrt(compressed_residual2/training_force2):(compressed_residual2==0.0?0.0:std::numeric_limits<double>::infinity());
  if(!std::isfinite(residual))throw std::runtime_error("native fit training residual is nonzero at zero physical-force scale");return residual;
}

void make_edge_linear(const Graph& graph,const std::vector<double>& raw_gradient,const int n,
                      std::vector<std::array<double,3>>& edge_linear)
{
  const auto net=rpmd_ja_reference_math::net_force_stats(raw_gradient,n);
  if(!net.within_limit){std::ostringstream message;message<<std::setprecision(17)<<"native fit raw qNEP reference gradient has a nonzero net force"
    <<"; net_xyz=("<<net.net[0]<<','<<net.net[1]<<','<<net.net[2]<<"), net_norm="<<net.net_norm
    <<", vector_norm="<<net.vector_norm<<", limit="<<net.limit<<", status=nonzero-net-or-nonfinite";throw std::runtime_error(message.str());}
  std::vector<std::vector<int>> incident(n);
  for(std::size_t e=0;e<graph.edges.size();++e){incident[graph.edges[e].i].push_back(static_cast<int>(e));incident[graph.edges[e].j].push_back(static_cast<int>(e));}
  edge_linear.resize(graph.edges.size());
  for(int axis=0;axis<3;++axis){
    std::vector<double> b(n),x(n,0.0),r(n),p(n),ap(n);
    for(int i=0;i<n;++i)b[i]=-raw_gradient[axis*n+i];
    r=b;p=r;double rr=std::inner_product(r.begin(),r.end(),r.begin(),0.0);const double initial=std::sqrt(rr);
    for(int it=0;it<20000&&std::sqrt(rr)>1e-11*std::max(1.0,initial);++it){
      for(int i=0;i<n;++i){ap[i]=incident[i].size()*p[i];for(int ei:incident[i]){const auto&e=graph.edges[ei];ap[i]-=p[e.i==i?e.j:e.i];}}
      const double pap=std::inner_product(p.begin(),p.end(),ap.begin(),0.0);if(!(pap>0.0))throw std::runtime_error("native fit edge Laplacian CG lost positive curvature");
       const double alpha=rr/pap;for(int i=0;i<n;++i){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
       const double next=std::inner_product(r.begin(),r.end(),r.begin(),0.0);const double beta=next/rr;for(int i=0;i<n;++i)p[i]=r[i]+beta*p[i];rr=next;
    }
    if(std::sqrt(rr)>1e-9*std::max(1.0,initial))throw std::runtime_error("native fit edge Laplacian CG failed gradient-balance tolerance");
    for(std::size_t e=0;e<graph.edges.size();++e){const auto& edge=graph.edges[e];edge_linear[e][axis]=x[edge.j]-x[edge.i];}
  }
  std::vector<double> residual=raw_gradient;
  for(std::size_t e=0;e<graph.edges.size();++e){const auto& edge=graph.edges[e];for(int a=0;a<3;++a){residual[a*n+edge.i]-=edge_linear[e][a];residual[a*n+edge.j]+=edge_linear[e][a];}}
  double error=0.0,scale=0.0;for(int i=0;i<3*n;++i){error=std::hypot(error,residual[i]);scale=std::hypot(scale,raw_gradient[i]);}
  if(!std::isfinite(error)||!std::isfinite(scale)||error>1e-8*std::max(1.0,scale))throw std::runtime_error("native fit edge linear coefficients do not cancel the qNEP reference gradient");
}

__host__ __device__ inline std::size_t edge_parameter_index(const int a,const int b)
{
  const int lo=a<b?a:b,hi=a>b?a:b;
  return static_cast<std::size_t>(lo==0?hi:(lo==1?(hi==1?3:4):5));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double covariance_atomic_add(double* address,const double value)
{
  unsigned long long* bits=reinterpret_cast<unsigned long long*>(address);unsigned long long old=*bits,assumed;
  do{assumed=old;old=atomicCAS(bits,assumed,__double_as_longlong(value+__longlong_as_double(assumed)));}while(assumed!=old);
  return __longlong_as_double(old);
}
#else
static __device__ __inline__ double covariance_atomic_add(double* address,const double value)
{return atomicAdd(address,value);}
#endif

void add_covariance_frame_statistics(const Graph& graph,const std::vector<double>& position,
  const std::vector<double>& r0,const int n,std::vector<double>& sums)
{
  if(position.size()!=static_cast<std::size_t>(3*n)||r0.size()!=position.size()||sums.size()!=6*graph.type_pairs.size())
    throw std::invalid_argument("native covariance fit statistic dimensions do not match");
  for(const Edge& edge:graph.edges){
    const double dx[3]={position[edge.j]-r0[edge.j]-position[edge.i]+r0[edge.i],
      position[n+edge.j]-r0[n+edge.j]-position[n+edge.i]+r0[n+edge.i],
      position[2*n+edge.j]-r0[2*n+edge.j]-position[2*n+edge.i]+r0[2*n+edge.i]};
    const std::size_t p=6*static_cast<std::size_t>(edge.group);
    sums[p]+=dx[0]*dx[0];sums[p+1]+=2.0*dx[0]*dx[1];sums[p+2]+=2.0*dx[0]*dx[2];
    sums[p+3]+=dx[1]*dx[1];sums[p+4]+=2.0*dx[1]*dx[2];sums[p+5]+=dx[2]*dx[2];
  }
}

std::vector<double> collect_covariance_fit_statistics(std::istream& in,const SampleHeader& header,
  const std::uint64_t frame_count,const std::uint64_t training_frames,const Graph& graph,
  const std::vector<double>& r0,const double temperature)
{
  if(training_frames==0||training_frames>frame_count||!(temperature>0.0)||!std::isfinite(temperature))
    throw std::invalid_argument("native covariance fit training metadata is invalid");
  std::vector<double> sums(6*graph.type_pairs.size(),0.0),position(r0.size()),force_frame(r0.size());
  double step=0.0;in.clear();in.seekg(header.frames);
  for(std::uint64_t frame=0;frame<frame_count;++frame){read_frame(in,position,force_frame,step);if(frame<training_frames)add_covariance_frame_statistics(graph,position,r0,header.n,sums);}
  const double beta=1.0/(K_B*temperature*static_cast<double>(training_frames));
  for(double& value:sums){value*=beta;if(!std::isfinite(value))throw std::runtime_error("native covariance fit training statistic is non-finite");}
  return sums;
}

std::pair<double,double> compare_covariance_training_force_residuals(std::istream& in,const SampleHeader& header,
  const std::uint64_t frame_count,const std::uint64_t training_frames,const Graph& graph,const std::vector<double>& r0,
  DeviceBaseline& baseline,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,
  const std::vector<double>& theta,const std::vector<double>& theta_baseline)
{
  if(training_frames==0||training_frames>frame_count||theta.size()!=theta_baseline.size())
    throw std::invalid_argument("native covariance fit force diagnostic dimensions do not match");
  const int n=header.n,d=3*n;std::vector<double> position(d),force_frame(d),q(d),f(d),base(d),candidate(d),reference(d);
  double step=0.0,candidate2=0.0,baseline2=0.0,force2=0.0;in.clear();in.seekg(header.frames);
  for(std::uint64_t frame=0;frame<frame_count;++frame){read_frame(in,position,force_frame,step);if(frame>=training_frames)continue;
    for(int k=0;k<d;++k){q[k]=sqrt_mass[k]*(position[k]-r0[k]);f[k]=force_frame[k]/sqrt_mass[k];}project_translation(q,sqrt_atom,n);project_translation(f,sqrt_atom,n);
    baseline.apply(q,base);apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,candidate);apply_additive(graph,q,sqrt_mass,sqrt_atom,theta_baseline,n,reference);
    for(int k=0;k<d;++k){const double target=f[k]+base[k];candidate2+=(target+candidate[k])*(target+candidate[k]);baseline2+=(target+reference[k])*(target+reference[k]);force2+=f[k]*f[k];}
  }
  if(!std::isfinite(candidate2)||!std::isfinite(baseline2)||!std::isfinite(force2)||candidate2<0.0||baseline2<0.0||force2<0.0)
    throw std::runtime_error("native covariance fit force residual diagnostic is non-finite");
  const double denominator=std::max(force2,1e-300);return {std::sqrt(candidate2/denominator),std::sqrt(baseline2/denominator)};
}

__global__ void covariance_scatter_additive(double* matrix,const int* edge_i,const int* edge_j,const int* edge_group,
  const double* theta,const double* sqrt_mass,const int edge_count,const int n,const int d)
{
  const int edge=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);if(edge>=edge_count)return;
  const int atoms[2]={edge_i[edge],edge_j[edge]},group=edge_group[edge];const double signs[2]={-1.0,1.0};
  for(int u=0;u<2;++u)for(int v=0;v<2;++v)for(int a=0;a<3;++a)for(int b=0;b<3;++b){
    const std::size_t param=6ULL*group+edge_parameter_index(a,b);
    const double scale=sqrt_mass[atoms[u]]*sqrt_mass[atoms[v]];
    const int row=a*n+atoms[u],column=b*n+atoms[v];
    covariance_atomic_add(matrix+static_cast<std::size_t>(row)*d+column,signs[u]*signs[v]*theta[param]/scale);
  }
}

__global__ void covariance_householder_matvec(const double* matrix,const double* vector,double* product,const int d)
{
  const int row=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);if(row>=d)return;
  double value=0.0;for(int column=0;column<d;++column)value+=matrix[static_cast<std::size_t>(row)*d+column]*vector[column];product[row]=value;
}

__global__ void covariance_householder_update(double* matrix,const double* vector,const double* product,
  const double gamma,const int d)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(d)*d)return;
  const int row=static_cast<int>(index/d),column=static_cast<int>(index%d);
  matrix[index]+=-2.0*vector[row]*product[column]-2.0*product[row]*vector[column]+4.0*gamma*vector[row]*vector[column];
}

__global__ void covariance_symmetrize_matrix(double* matrix,const int d)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(d)*d)return;const int row=static_cast<int>(index/d),column=static_cast<int>(index%d);if(row>column)return;
  const std::size_t mirror=static_cast<std::size_t>(column)*d+row;const double value=0.5*matrix[index]+0.5*matrix[mirror];matrix[index]=value;matrix[mirror]=value;
}

__global__ void covariance_compact_matrix(const double* full,double* compact,const int d,const int r,const int n)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;if(index>=static_cast<std::size_t>(r)*r)return;
  const int row=static_cast<int>(index/r),column=static_cast<int>(index%r);
  const int full_row=row+1+(row>=n-1)+(row>=2*n-2),full_column=column+1+(column>=n-1)+(column>=2*n-2);
  compact[index]=full[static_cast<std::size_t>(full_row)*d+full_column];
}

__global__ void covariance_subtract_diagonal(double* matrix,const int n,const double value)
{const int i=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)matrix[static_cast<std::size_t>(i)*n+i]-=value;}

__global__ void covariance_generate_rhs(double* rhs,const int* compact_to_full,const double* householder,
  const int n,const int r,const int source_begin,const int columns)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(r)*columns)return;const int row=static_cast<int>(index%r),column=static_cast<int>(index/r);
  const int full=compact_to_full[row],source=source_begin+column,axis=source/n;
  const double reflection=full/n==axis?householder[static_cast<std::size_t>(axis)*3*n+full]*householder[static_cast<std::size_t>(axis)*3*n+source]:0.0;
  rhs[index]=(full==source?1.0:0.0)-2.0*reflection;
}

__global__ void covariance_expand_solutions(const double* compact,double* ambient,const int* full_to_compact,
  const int d,const int r,const int columns)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(d)*columns)return;const int row=static_cast<int>(index%d),column=static_cast<int>(index/d);
  const int source=full_to_compact[row];ambient[index]=source<0?0.0:compact[static_cast<std::size_t>(column)*r+source];
}

__global__ void covariance_householder_dots(const double* ambient,const double* householder,double* dots,
  const int n,const int d,const int columns)
{
  const int index=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);if(index>=3*columns)return;
  const int axis=index/columns,column=index%columns;double sum=0.0;
  for(int atom=0;atom<n;++atom){const int full=axis*n+atom;sum+=householder[static_cast<std::size_t>(axis)*d+full]*ambient[static_cast<std::size_t>(column)*d+full];}
  dots[index]=sum;
}

__global__ void covariance_apply_householder(double* ambient,const double* householder,const double* dots,
  const int n,const int d,const int columns)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(d)*columns)return;const int row=static_cast<int>(index%d),column=static_cast<int>(index/d),axis=row/n;
  ambient[index]-=2.0*householder[static_cast<std::size_t>(axis)*d+row]*dots[axis*columns+column];
}

__global__ void covariance_capture_edge_covariance(const double* ambient,const double* sqrt_mass,
  const int* edge_i,const int* edge_j,const int edge_count,const int n,const int d,const int source_begin,
  const int columns,double* left,double* right)
{
  const std::size_t index=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
  if(index>=static_cast<std::size_t>(edge_count)*columns)return;const int edge=static_cast<int>(index/columns),column=static_cast<int>(index%columns);
  const int source=source_begin+column,atom=source%n,beta=source/n,i=edge_i[edge],j=edge_j[edge];
  if(atom!=i&&atom!=j)return;const double cross_scale=sqrt_mass[i]*sqrt_mass[j];
  for(int alpha=0;alpha<3;++alpha){const std::size_t at=static_cast<std::size_t>(edge)*9+alpha*3+beta;
    const double ci=ambient[static_cast<std::size_t>(column)*d+alpha*n+i],cj=ambient[static_cast<std::size_t>(column)*d+alpha*n+j];
    if(atom==i)left[at]=ci/(sqrt_mass[i]*sqrt_mass[i])-cj/cross_scale;
    else right[at]=cj/(sqrt_mass[j]*sqrt_mass[j])-ci/cross_scale;
  }
}

__global__ void covariance_log_diagonal(const double* factor,double* values,const int n)
{const int i=static_cast<int>(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)values[i]=factor[static_cast<std::size_t>(i)*n+i];}

int covariance_parameter_count(const Graph& graph)
{if(graph.type_pairs.empty()||graph.type_pairs.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()/6))throw std::runtime_error("native covariance fit parameter count is invalid");return static_cast<int>(6*graph.type_pairs.size());}

int covariance_edge_count(const Graph& graph)
{if(graph.edges.empty()||graph.edges.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::runtime_error("native covariance fit edge count is invalid");return static_cast<int>(graph.edges.size());}

struct DeviceCovarianceSystem
{
  static constexpr int solve_batch=32;
  DeviceBaseline& baseline;const Graph& graph;int n=0,d=0,r=0,p=0,edge_count=0,lwork=0;
  std::vector<double> parameter_basis_norms;
  double *full=nullptr,*compact=nullptr,*shifted=nullptr,*householder=nullptr,*product=nullptr,*theta_device=nullptr;
  double *sqrt_mass_device=nullptr,*rhs=nullptr,*rhs_check=nullptr,*rhs_temp=nullptr,*ambient=nullptr,*dots=nullptr,*diagonal=nullptr;
  double *edge_cov_left=nullptr,*edge_cov_right=nullptr,*solver_work=nullptr;int *edge_i=nullptr,*edge_j=nullptr,*edge_group=nullptr;
  int *compact_to_full=nullptr,*full_to_compact=nullptr,*info=nullptr;cublasHandle_t blas=nullptr;cusolverDnHandle_t solver=nullptr;
  std::uint64_t factorizations=0,solve_rhs_columns=0;double worst_solve_relative_residual=0.0;

  DeviceCovarianceSystem(DeviceBaseline& base,const Graph& fixed_graph,const std::vector<double>& sqrt_mass,
    const std::vector<double>& masses)
    :baseline(base),graph(fixed_graph),n(static_cast<int>(masses.size())),d(3*n),r(d-3),
     p(covariance_parameter_count(fixed_graph)),edge_count(covariance_edge_count(fixed_graph))
  {try{initialize(sqrt_mass,masses);}catch(...){release();throw;}}
  ~DeviceCovarianceSystem(){release();}

  void release()
  {
    if(solver){cusolverDnDestroy(solver);solver=nullptr;}if(blas){cublasDestroy(blas);blas=nullptr;}
    double** values[]={&full,&compact,&shifted,&householder,&product,&theta_device,&sqrt_mass_device,&rhs,&rhs_check,&rhs_temp,&ambient,&dots,&diagonal,&edge_cov_left,&edge_cov_right,&solver_work};
    for(double** value:values)if(*value){cudaFree(*value);*value=nullptr;}
    int** indices[]={&edge_i,&edge_j,&edge_group,&compact_to_full,&full_to_compact,&info};
    for(int** value:indices)if(*value){cudaFree(*value);*value=nullptr;}
  }

  void initialize(const std::vector<double>& sqrt_mass,const std::vector<double>& masses)
  {
    if(n<2||d!=baseline.d||r<=0||p<=0||edge_count<=0||sqrt_mass.size()!=static_cast<std::size_t>(d)||
       graph.edges.size()>static_cast<std::size_t>(std::numeric_limits<int>::max())||
       static_cast<std::size_t>(d)>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(d)/sizeof(double)||
       static_cast<std::size_t>(r)>std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(r)/sizeof(double))
      throw std::runtime_error("native covariance fit dimensions are invalid");
    const std::size_t full_bytes=static_cast<std::size_t>(d)*d*sizeof(double),compact_bytes=static_cast<std::size_t>(r)*r*sizeof(double);
    if(static_cast<std::size_t>(d)>std::numeric_limits<std::size_t>::max()/solve_batch/sizeof(double)||
       graph.edges.size()>std::numeric_limits<std::size_t>::max()/18/sizeof(double))
      throw std::runtime_error("native covariance fit workspace size overflows");
    const std::size_t batch_bytes=static_cast<std::size_t>(d)*solve_batch*sizeof(double);
    const std::size_t edge_bytes=graph.edges.size()*9*sizeof(double);
    std::size_t required=full_bytes;
    for(const std::size_t amount:{compact_bytes,compact_bytes,batch_bytes,batch_bytes,batch_bytes,batch_bytes,2*edge_bytes,
        static_cast<std::size_t>(3)*d*sizeof(double),static_cast<std::size_t>(4)*d*sizeof(double),
        static_cast<std::size_t>(3)*graph.edges.size()*sizeof(int),static_cast<std::size_t>(d+r)*sizeof(int),
        static_cast<std::size_t>(p)*sizeof(double),static_cast<std::size_t>(r)*sizeof(double)}){
      if(amount>std::numeric_limits<std::size_t>::max()-required)throw std::runtime_error("native covariance fit workspace size overflows");required+=amount;}
    std::size_t free_bytes=0,total_bytes=0;if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess||required>free_bytes||free_bytes-required<512ULL*1024*1024)
      throw std::runtime_error("native covariance fit exact-Cholesky workspace preflight leaves less than 512 MiB safety margin");
    std::vector<int> host_ctf(r),host_ftc(d,-1),host_i(edge_count),host_j(edge_count),host_group(edge_count);
    for(int k=0,index=0;k<d;++k)if(k!=0&&k!=n&&k!=2*n){host_ctf[index]=k;host_ftc[k]=index++;}
    for(int e=0;e<edge_count;++e){host_i[e]=graph.edges[e].i;host_j[e]=graph.edges[e].j;host_group[e]=graph.edges[e].group;}
    const std::size_t groups=graph.type_pairs.size();
    std::map<std::pair<int,int>,double> group_diagonal;std::vector<double> group_off_diagonal(groups,0.0);
    for(const Edge& edge:graph.edges){const double inverse_i=1.0/masses[edge.i],inverse_j=1.0/masses[edge.j];
      group_diagonal[{edge.group,edge.i}]+=inverse_i;group_diagonal[{edge.group,edge.j}]+=inverse_j;group_off_diagonal[edge.group]+=inverse_i*inverse_j;}
    // The off-diagonal parameter basis occupies two symmetric coordinate blocks.
    std::vector<NeumaierAccumulator> group_norm_squared(groups);for(const auto& entry:group_diagonal)
      group_norm_squared[entry.first.first].add_product(entry.second,entry.second);
    parameter_basis_norms.resize(static_cast<std::size_t>(p));
    for(std::size_t group=0;group<groups;++group){group_norm_squared[group].add(2.0*group_off_diagonal[group]);const double diagonal_norm=std::sqrt(group_norm_squared[group].value());if(!(diagonal_norm>0.0)||!std::isfinite(diagonal_norm))throw std::runtime_error("native covariance fit parameter basis norm is invalid");
      for(const int index:{0,3,5})parameter_basis_norms[6*group+index]=diagonal_norm;
      for(const int index:{1,2,4})parameter_basis_norms[6*group+index]=std::sqrt(2.0)*diagonal_norm;}
    std::vector<double> host_house(static_cast<std::size_t>(3)*d,0.0);const double mass_sum=std::accumulate(masses.begin(),masses.end(),0.0);
    if(!std::isfinite(mass_sum)||!(mass_sum>0.0))throw std::runtime_error("native covariance fit total mass is invalid");
    for(int axis=0;axis<3;++axis){double norm2=0.0;for(int i=0;i<n;++i){const int k=axis*n+i;host_house[static_cast<std::size_t>(axis)*d+k]=sqrt_mass[k]/std::sqrt(mass_sum);if(i==0)host_house[static_cast<std::size_t>(axis)*d+k]-=1.0;norm2+=host_house[static_cast<std::size_t>(axis)*d+k]*host_house[static_cast<std::size_t>(axis)*d+k];}
      if(!(norm2>0.0)||!std::isfinite(norm2))throw std::runtime_error("native covariance fit translation Householder is degenerate");const double inv=1.0/std::sqrt(norm2);for(int i=0;i<n;++i)host_house[static_cast<std::size_t>(axis)*d+axis*n+i]*=inv;}
    auto alloc_double=[](double** out,const std::size_t count){if(cudaMalloc(reinterpret_cast<void**>(out),std::max<std::size_t>(1,count)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native covariance fit GPU allocation failed");};
    auto alloc_int=[](int** out,const std::size_t count){if(cudaMalloc(reinterpret_cast<void**>(out),std::max<std::size_t>(1,count)*sizeof(int))!=cudaSuccess)throw std::runtime_error("native covariance fit GPU allocation failed");};
    alloc_double(&full,static_cast<std::size_t>(d)*d);alloc_double(&compact,static_cast<std::size_t>(r)*r);alloc_double(&shifted,static_cast<std::size_t>(r)*r);
    alloc_double(&householder,static_cast<std::size_t>(3)*d);alloc_double(&product,d);alloc_double(&theta_device,p);alloc_double(&sqrt_mass_device,n);
    alloc_double(&rhs,static_cast<std::size_t>(r)*solve_batch);alloc_double(&rhs_check,static_cast<std::size_t>(r)*solve_batch);alloc_double(&rhs_temp,static_cast<std::size_t>(r)*solve_batch);
    alloc_double(&ambient,static_cast<std::size_t>(d)*solve_batch);alloc_double(&dots,static_cast<std::size_t>(3)*solve_batch);alloc_double(&diagonal,r);
    alloc_double(&edge_cov_left,graph.edges.size()*9);alloc_double(&edge_cov_right,graph.edges.size()*9);
    alloc_int(&edge_i,edge_count);alloc_int(&edge_j,edge_count);alloc_int(&edge_group,edge_count);alloc_int(&compact_to_full,r);alloc_int(&full_to_compact,d);alloc_int(&info,1);
    if(cudaMemcpy(householder,host_house.data(),host_house.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(sqrt_mass_device,sqrt_mass.data(),static_cast<std::size_t>(n)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(edge_i,host_i.data(),host_i.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(edge_j,host_j.data(),host_j.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(edge_group,host_group.data(),host_group.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(compact_to_full,host_ctf.data(),host_ctf.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||
       cudaMemcpy(full_to_compact,host_ftc.data(),host_ftc.size()*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw std::runtime_error("native covariance fit static workspace upload failed");
    if(cublasCreate(&blas)!=CUBLAS_STATUS_SUCCESS||cusolverDnCreate(&solver)!=CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("native covariance fit cuBLAS/cuSOLVER handle creation failed");
    if(cusolverDnDpotrf_bufferSize(solver,CUBLAS_FILL_MODE_LOWER,r,compact,r,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0)
      throw std::runtime_error("native covariance fit Cholesky workspace query failed");
    std::size_t workspace_free=0,workspace_total=0;const std::size_t workspace_bytes=static_cast<std::size_t>(lwork)*sizeof(double);
    if(cudaMemGetInfo(&workspace_free,&workspace_total)!=cudaSuccess||workspace_bytes>workspace_free||workspace_free-workspace_bytes<512ULL*1024*1024)
      throw std::runtime_error("native covariance fit Cholesky workspace preflight leaves less than 512 MiB safety margin");
    alloc_double(&solver_work,lwork);
  }

  std::vector<double> coordinate_scales(const std::vector<double>& parameters,const double epsilon)
  {
    assemble(parameters);const std::size_t matrix_values=static_cast<std::size_t>(r)*r;double frobenius=0.0;
    for(std::size_t offset=0;offset<matrix_values;){const int count=static_cast<int>(std::min<std::size_t>(matrix_values-offset,static_cast<std::size_t>(std::numeric_limits<int>::max())));double partial=0.0;
      if(cublasDnrm2(blas,count,compact+offset,1,&partial)!=CUBLAS_STATUS_SUCCESS||!std::isfinite(partial))throw std::runtime_error("native covariance fit baseline matrix norm failed");
      frobenius=std::hypot(frobenius,partial);offset+=static_cast<std::size_t>(count);}
    const double matrix_scale=std::max(epsilon,frobenius/std::sqrt(static_cast<double>(r)));
    if(!(matrix_scale>0.0)||!std::isfinite(matrix_scale))throw std::runtime_error("native covariance fit baseline matrix scale is invalid");
    std::vector<double> scales(static_cast<std::size_t>(p));for(int index=0;index<p;++index){scales[index]=matrix_scale/parameter_basis_norms[index];
      if(!(scales[index]>0.0)||!std::isfinite(scales[index]))throw std::runtime_error("native covariance fit parameter scale is invalid");}
    return scales;
  }

  void assemble(const std::vector<double>& parameters)
  {
    if(parameters.size()!=static_cast<std::size_t>(p)||!std::all_of(parameters.begin(),parameters.end(),[](double x){return std::isfinite(x);}))
      throw std::runtime_error("native covariance fit parameter vector is invalid");
    if(cudaMemcpy(full,baseline.k,static_cast<std::size_t>(d)*d*sizeof(double),cudaMemcpyDeviceToDevice)!=cudaSuccess||
       cudaMemcpy(theta_device,parameters.data(),parameters.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess)
      throw std::runtime_error("native covariance fit matrix/parameter copy failed");
    covariance_scatter_additive<<<static_cast<unsigned>((edge_count+255)/256),256>>>(full,edge_i,edge_j,edge_group,theta_device,sqrt_mass_device,edge_count,n,d);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit additive matrix assembly failed");
    const std::size_t dd=static_cast<std::size_t>(d)*d;
    covariance_symmetrize_matrix<<<static_cast<unsigned>((dd+255)/256),256>>>(full,d);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit matrix symmetrization failed");
    for(int axis=0;axis<3;++axis){const double* vector=householder+static_cast<std::size_t>(axis)*d;
      covariance_householder_matvec<<<static_cast<unsigned>((d+255)/256),256>>>(full,vector,product,d);
      if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit Householder matvec failed");
      double gamma=0.0;if(cublasDdot(blas,d,vector,1,product,1,&gamma)!=CUBLAS_STATUS_SUCCESS||!std::isfinite(gamma))throw std::runtime_error("native covariance fit Householder inner product failed");
      covariance_householder_update<<<static_cast<unsigned>((dd+255)/256),256>>>(full,vector,product,gamma,d);
      if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit Householder congruence failed");}
    covariance_compact_matrix<<<static_cast<unsigned>((static_cast<std::size_t>(r)*r+255)/256),256>>>(full,compact,d,r,n);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit translation-complement compaction failed");
  }

  bool factor(double* matrix,double& logdet)
  {
    if(cusolverDnDpotrf(solver,CUBLAS_FILL_MODE_LOWER,r,matrix,r,solver_work,lwork,info)!=CUSOLVER_STATUS_SUCCESS||cudaDeviceSynchronize()!=cudaSuccess)
      throw std::runtime_error("native covariance fit Cholesky call failed");
    ++factorizations;int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native covariance fit Cholesky status read failed");
    if(host_info>0)return false;if(host_info<0)throw std::runtime_error("native covariance fit Cholesky received an invalid parameter");
    covariance_log_diagonal<<<static_cast<unsigned>((r+255)/256),256>>>(matrix,diagonal,r);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit Cholesky diagonal extraction failed");
    std::vector<double> values(r);if(cudaMemcpy(values.data(),diagonal,values.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native covariance fit Cholesky diagonal read failed");
    logdet=0.0;for(const double value:values){if(!(value>0.0)||!std::isfinite(value))throw std::runtime_error("native covariance fit Cholesky diagonal is non-finite");logdet+=2.0*std::log(value);}
    if(!std::isfinite(logdet))throw std::runtime_error("native covariance fit log-determinant is non-finite");return true;
  }

  std::vector<double> trace_derivatives(const double* factor_matrix)
  {
    const std::size_t edge_values=graph.edges.size()*9;if(cudaMemset(edge_cov_left,0,edge_values*sizeof(double))!=cudaSuccess||cudaMemset(edge_cov_right,0,edge_values*sizeof(double))!=cudaSuccess)
      throw std::runtime_error("native covariance fit trace initialization failed");
    const double minus_one=-1.0,one=1.0;
    for(int source=0;source<d;source+=solve_batch){const int columns=std::min(solve_batch,d-source);const std::size_t values=static_cast<std::size_t>(r)*columns;
      covariance_generate_rhs<<<static_cast<unsigned>((values+255)/256),256>>>(rhs,compact_to_full,householder,n,r,source,columns);
      if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit inverse RHS generation failed");
      if(cusolverDnDpotrs(solver,CUBLAS_FILL_MODE_LOWER,r,columns,factor_matrix,r,rhs,r,info)!=CUSOLVER_STATUS_SUCCESS||cudaDeviceSynchronize()!=cudaSuccess)
        throw std::runtime_error("native covariance fit multi-RHS solve failed");
      int solve_info=0;if(cudaMemcpy(&solve_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||solve_info!=0)throw std::runtime_error("native covariance fit multi-RHS solve returned an error");
      covariance_expand_solutions<<<static_cast<unsigned>((static_cast<std::size_t>(d)*columns+255)/256),256>>>(rhs,ambient,full_to_compact,d,r,columns);
      if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("native covariance fit solution expansion failed");
      covariance_householder_dots<<<static_cast<unsigned>((3*columns+255)/256),256>>>(ambient,householder,dots,n,d,columns);
      if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("native covariance fit inverse Householder projection failed");
      covariance_apply_householder<<<static_cast<unsigned>((static_cast<std::size_t>(d)*columns+255)/256),256>>>(ambient,householder,dots,n,d,columns);
      if(cudaGetLastError()!=cudaSuccess)throw std::runtime_error("native covariance fit inverse Householder projection failed");
      covariance_capture_edge_covariance<<<static_cast<unsigned>((static_cast<std::size_t>(edge_count)*columns+255)/256),256>>>(ambient,sqrt_mass_device,edge_i,edge_j,edge_count,n,d,source,columns,edge_cov_left,edge_cov_right);
      if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit graph trace contraction failed");
      if(cudaMemcpy(rhs_check,rhs,values*sizeof(double),cudaMemcpyDeviceToDevice)!=cudaSuccess||
         cublasDtrmm(blas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,r,columns,&one,factor_matrix,r,rhs_check,r,rhs_temp,r)!=CUBLAS_STATUS_SUCCESS||
         cublasDtrmm(blas,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,r,columns,&one,factor_matrix,r,rhs_temp,r,rhs_check,r)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("native covariance fit solve-residual reconstruction failed");
      covariance_generate_rhs<<<static_cast<unsigned>((values+255)/256),256>>>(rhs,compact_to_full,householder,n,r,source,columns);
      if(cudaGetLastError()!=cudaSuccess||cublasDaxpy(blas,static_cast<int>(values),&minus_one,rhs,1,rhs_check,1)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("native covariance fit solve-residual comparison failed");
      double residual=0.0,rhs_norm=0.0;if(cublasDnrm2(blas,static_cast<int>(values),rhs_check,1,&residual)!=CUBLAS_STATUS_SUCCESS||cublasDnrm2(blas,static_cast<int>(values),rhs,1,&rhs_norm)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("native covariance fit solve-residual norm failed");
      const double relative=residual/std::max(rhs_norm,1e-300);if(!std::isfinite(relative))throw std::runtime_error("native covariance fit solve residual is non-finite");
      worst_solve_relative_residual=std::max(worst_solve_relative_residual,relative);solve_rhs_columns+=static_cast<std::uint64_t>(columns);
    }
    if(worst_solve_relative_residual>1e-7)throw std::runtime_error("native covariance fit matrix solve residual exceeds 1e-7");
    std::vector<double> left(edge_values),right(edge_values);if(cudaMemcpy(left.data(),edge_cov_left,edge_values*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||
       cudaMemcpy(right.data(),edge_cov_right,edge_values*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native covariance fit edge trace read failed");
    std::vector<NeumaierAccumulator> sums(static_cast<std::size_t>(p));
    for(std::size_t edge=0;edge<graph.edges.size();++edge){const std::size_t group=6*static_cast<std::size_t>(graph.edges[edge].group);
      double block[9];for(int a=0;a<3;++a)for(int b=0;b<3;++b){const std::size_t ab=edge*9+a*3+b,ba=edge*9+b*3+a;block[a*3+b]=0.5*((left[ab]+right[ab])+(left[ba]+right[ba]));}
      for(int a=0;a<3;++a)sums[group+edge_parameter_index(a,a)].add(block[a*3+a]);
      sums[group+1].add(2.0*block[1]);sums[group+2].add(2.0*block[2]);sums[group+4].add(2.0*block[5]);
    }
    std::vector<double> trace(p);for(int i=0;i<p;++i){trace[i]=sums[i].value();if(!std::isfinite(trace[i]))throw std::runtime_error("native covariance fit inverse trace is non-finite");}return trace;
  }

  bool evaluate_factors(const std::vector<double>& parameters,const double epsilon,double& logdet,
    double& shifted_logdet,std::vector<double>* traces,std::vector<double>* shifted_traces=nullptr)
  {
    assemble(parameters);const std::size_t bytes=static_cast<std::size_t>(r)*r*sizeof(double);
    if(cudaMemcpy(shifted,compact,bytes,cudaMemcpyDeviceToDevice)!=cudaSuccess)throw std::runtime_error("native covariance fit shifted matrix copy failed");
    covariance_subtract_diagonal<<<static_cast<unsigned>((r+255)/256),256>>>(shifted,r,epsilon);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("native covariance fit epsilon shift failed");
    if(!factor(shifted,shifted_logdet))return false;
    if(!factor(compact,logdet))throw std::runtime_error("native covariance fit unshifted matrix is not SPD after shifted Cholesky passed");
    if(traces||shifted_traces)worst_solve_relative_residual=0.0;
    if(traces)*traces=trace_derivatives(compact);if(shifted_traces)*shifted_traces=trace_derivatives(shifted);return true;
  }
};

struct CovarianceBaseline
{
  std::vector<double> theta,trace;
  double mu=0.0,logdet=0.0,shifted_logdet=0.0,boundary_mu=0.0;
  bool epsilon_constraint_active=false;
};

double covariance_dot(const std::vector<double>& left,const std::vector<double>& right)
{
  if(left.size()!=right.size())throw std::invalid_argument("native covariance fit dot-product dimensions do not match");
  NeumaierAccumulator sum;for(std::size_t i=0;i<left.size();++i)sum.add_product(left[i],right[i]);return sum.value();
}

std::vector<double> graph_laplacian_direction(const Graph& graph)
{
  std::vector<double> direction(6*graph.type_pairs.size(),0.0);
  for(std::size_t group=0;group<graph.type_pairs.size();++group){direction[6*group]=1.0;direction[6*group+3]=1.0;direction[6*group+5]=1.0;}
  return direction;
}

CovarianceBaseline build_graph_covariance_baseline(DeviceCovarianceSystem& system,const std::vector<double>& statistics,
  const double epsilon,std::uint64_t& objective_evaluations)
{
  if(statistics.size()!=static_cast<std::size_t>(system.p)||!(epsilon>0.0)||!std::isfinite(epsilon))
    throw std::invalid_argument("native covariance fit baseline inputs are invalid");
  const std::vector<double> direction=graph_laplacian_direction(system.graph);
  const double data_slope=covariance_dot(statistics,direction);
  if(!(data_slope>0.0)||!std::isfinite(data_slope))throw std::runtime_error("BASELINE_SPD_FAIL: training fluctuations do not bound the graph baseline objective");
  auto parameters_at=[&](const double mu){std::vector<double> value(direction.size());for(std::size_t i=0;i<value.size();++i)value[i]=mu*direction[i];return value;};
  auto evaluate_mu=[&](const double mu,double& value){++objective_evaluations;double logdet=0.0,shifted=0.0;
    if(!system.evaluate_factors(parameters_at(mu),epsilon,logdet,shifted,nullptr))return false;
    value=0.5*mu*data_slope-0.5*logdet;if(!std::isfinite(value))throw std::runtime_error("native covariance fit baseline objective is non-finite");return true;};
  const double initial_step=std::max(epsilon,1e-8);
  double floor=0.0,zero_objective=0.0;const bool zero_feasible=evaluate_mu(0.0,zero_objective);
  if(zero_feasible){
    double feasible=0.0,infeasible=0.0,distance=initial_step;bool found=false;
    for(int attempt=0;attempt<64;++attempt){infeasible=feasible-distance;if(!std::isfinite(infeasible))break;double ignored=0.0;
      if(!evaluate_mu(infeasible,ignored)){found=true;break;}feasible=infeasible;distance*=2.0;if(!std::isfinite(distance))break;}
    if(!found)throw std::runtime_error("BASELINE_SPD_FAIL: could not bracket the negative graph-Laplacian feasibility boundary");
    double low=infeasible,high=feasible;for(int iteration=0;iteration<48;++iteration){const double middle=low+0.5*(high-low);double ignored=0.0;
      if(evaluate_mu(middle,ignored))high=middle;else low=middle;}floor=high;
  }else{
    double low=0.0,high=initial_step,high_objective=0.0;bool found=false;
    for(int attempt=0;attempt<64;++attempt){if(evaluate_mu(high,high_objective)){found=true;break;}low=high;high*=2.0;if(!std::isfinite(high))break;}
    if(!found)throw std::runtime_error("BASELINE_SPD_FAIL: could not find a graph-Laplacian shift with D_b-epsilon I positive definite");
    for(int iteration=0;iteration<48;++iteration){const double middle=low+0.5*(high-low);double ignored=0.0;
      if(evaluate_mu(middle,ignored))high=middle;else low=middle;}floor=high;
  }
  const double step=std::max(1e-8,std::abs(floor)*1e-4);
  double center=zero_feasible?0.0:floor+step,center_value=zero_feasible?zero_objective:0.0;
  if(center<=floor)center=std::nextafter(floor,std::numeric_limits<double>::infinity());
  if(!zero_feasible&&!evaluate_mu(center,center_value))throw std::runtime_error("BASELINE_SPD_FAIL: initial graph-Laplacian search point is not interior");
  double left=center-step,right=center+step,left_value=0.0,right_value=0.0;
  if(left<floor)left=floor;else if(!(left<center))left=std::max(floor,std::nextafter(center,-std::numeric_limits<double>::infinity()));
  if(!(right>center))right=std::nextafter(center,std::numeric_limits<double>::infinity());
  if(!evaluate_mu(left,left_value)||!std::isfinite(right)||!evaluate_mu(right,right_value))
    throw std::runtime_error("BASELINE_SPD_FAIL: graph-Laplacian baseline bracket point is not feasible");
  double lo=left,hi=right;bool bracketed=false;
  auto expand_left=[&](){
    if(left<=floor){lo=left;hi=center;bracketed=true;return;}
    double previous=center,point=left,value=left_value,distance=step;
    for(int attempt=0;attempt<64;++attempt){double candidate=point-distance;if(candidate<floor)candidate=floor;
      if(!(candidate<point))candidate=std::nextafter(point,-std::numeric_limits<double>::infinity());double candidate_value=0.0;
      if(candidate<floor||!evaluate_mu(candidate,candidate_value))throw std::runtime_error("BASELINE_SPD_FAIL: negative baseline search left the feasible interval");
      if(candidate_value>=value||candidate<=floor){lo=candidate;hi=previous;bracketed=true;return;}
      previous=point;point=candidate;value=candidate_value;distance*=2.0;if(!std::isfinite(distance))break;}
  };
  auto expand_right=[&](){
    double previous=center,point=right,value=right_value,distance=step;
    for(int attempt=0;attempt<64;++attempt){double candidate=point+distance;if(!(candidate>point))candidate=std::nextafter(point,std::numeric_limits<double>::infinity());double candidate_value=0.0;
      if(!std::isfinite(candidate)||!evaluate_mu(candidate,candidate_value))throw std::runtime_error("BASELINE_SPD_FAIL: positive baseline search left the feasible interval");
      if(candidate_value>=value){lo=previous;hi=candidate;bracketed=true;return;}
      previous=point;point=candidate;value=candidate_value;distance*=2.0;if(!std::isfinite(distance))break;}
  };
  const bool left_descends=left_value<center_value,right_descends=right_value<center_value;
  if(left_descends&&(!right_descends||left_value<=right_value))expand_left();
  else if(right_descends)expand_right();
  else bracketed=true;
  if(!bracketed)throw std::runtime_error("BASELINE_SPD_FAIL: could not bracket the one-dimensional covariance baseline minimum");
  constexpr double golden=0.6180339887498948482;
  double x1=hi-golden*(hi-lo),x2=lo+golden*(hi-lo),f1=0.0,f2=0.0;
  if(!evaluate_mu(x1,f1)||!evaluate_mu(x2,f2))throw std::runtime_error("BASELINE_SPD_FAIL: covariance baseline line search left the feasible interval");
  for(int iteration=0;iteration<48;++iteration){if(f1<=f2){hi=x2;x2=x1;f2=f1;x1=hi-golden*(hi-lo);if(!evaluate_mu(x1,f1))throw std::runtime_error("BASELINE_SPD_FAIL: covariance baseline search lost positive definiteness");}
    else{lo=x1;x1=x2;f1=f2;x2=lo+golden*(hi-lo);if(!evaluate_mu(x2,f2))throw std::runtime_error("BASELINE_SPD_FAIL: covariance baseline search lost positive definiteness");}}
  CovarianceBaseline result;result.mu=0.5*(lo+hi);result.theta=parameters_at(result.mu);result.boundary_mu=floor;
  if(!system.evaluate_factors(result.theta,epsilon,result.logdet,result.shifted_logdet,&result.trace))
    throw std::runtime_error("BASELINE_SPD_FAIL: selected graph-Laplacian baseline did not retain strict epsilon-positive curvature");
  result.epsilon_constraint_active=result.mu-floor<=1e-4*std::max({epsilon,std::abs(result.mu),std::abs(floor),1e-12});
  return result;
}

struct CovarianceFitResult
{
  bool converged=false;std::string status="NOT_STARTED",reason="none",epsilon_constraint_status="NOT_COMPUTED";
  std::string initialization_status="NOT_STARTED",line_search_failure_status="NONE",directional_derivative_status="NOT_RUN",directional_derivative_side="NONE";
  std::vector<double> statistics,theta_baseline,theta_start,theta;
  double baseline_mu=std::numeric_limits<double>::quiet_NaN(),objective_data=std::numeric_limits<double>::quiet_NaN();
  double objective_prior=std::numeric_limits<double>::quiet_NaN(),scaled_kkt_residual=std::numeric_limits<double>::quiet_NaN();
  double raw_gradient_inf_norm=std::numeric_limits<double>::quiet_NaN();
  double start_graph_delta=0.0,start_objective=std::numeric_limits<double>::quiet_NaN(),start_raw_gradient_inf_norm=std::numeric_limits<double>::quiet_NaN();
  double directional_derivative_slope=std::numeric_limits<double>::quiet_NaN(),directional_derivative_finite_difference=std::numeric_limits<double>::quiet_NaN();
  double directional_derivative_step=std::numeric_limits<double>::quiet_NaN();
  double barrier_tau=std::numeric_limits<double>::quiet_NaN(),barrier_gap_estimate=std::numeric_limits<double>::quiet_NaN();
  double worst_solve_relative_residual=std::numeric_limits<double>::quiet_NaN(),optimizer_seconds=0.0;
  std::uint64_t objective_evaluations=0,initialization_evaluations=0,line_search_trials=0,line_search_evaluations=0,shifted_not_spd_trials=0,armijo_reject_trials=0;
  std::uint64_t directional_diagnostic_evaluations=0,factorizations=0,solve_rhs_columns=0;bool barrier_center_condition_pass=false;
  int iterations=0;
};

enum class CovarianceObjectiveStatus { NotEvaluated, ShiftedNotSpd, Feasible };

struct CovarianceObjectiveEvaluation
{
  bool feasible=false;CovarianceObjectiveStatus status=CovarianceObjectiveStatus::NotEvaluated;
  double data=0.0,prior=0.0,barrier=0.0,objective=0.0,logdet=0.0,shifted_logdet=0.0;
  std::vector<double> gradient;
};

const char* covariance_objective_status_name(const CovarianceObjectiveStatus status)
{switch(status){case CovarianceObjectiveStatus::ShiftedNotSpd:return "SHIFTED_NOT_SPD";case CovarianceObjectiveStatus::Feasible:return "FEASIBLE";default:return "NOT_EVALUATED";}}

CovarianceObjectiveEvaluation evaluate_covariance_fit_objective(DeviceCovarianceSystem& system,
  const std::vector<double>& theta,const CovarianceBaseline& baseline,const std::vector<double>& statistics,
  const double rho,const double epsilon,const double tau,std::uint64_t& evaluations,const bool compute_gradient=true)
{
  CovarianceObjectiveEvaluation value;++evaluations;std::vector<double> trace,shifted_trace;
  if(!system.evaluate_factors(theta,epsilon,value.logdet,value.shifted_logdet,compute_gradient?&trace:nullptr,compute_gradient?&shifted_trace:nullptr)){
    value.status=CovarianceObjectiveStatus::ShiftedNotSpd;return value;}
  const double delta_logdet=value.logdet-baseline.logdet;std::vector<double> delta(theta.size());
  for(std::size_t i=0;i<theta.size();++i)delta[i]=theta[i]-baseline.theta[i];
  value.data=0.5*covariance_dot(delta,statistics)-0.5*delta_logdet;
  value.prior=0.5*rho*covariance_dot(delta,baseline.trace)-0.5*rho*delta_logdet;
  value.barrier=-tau*(value.shifted_logdet-baseline.shifted_logdet);
  value.objective=value.data+value.prior+value.barrier;
  if(compute_gradient){value.gradient.resize(theta.size());for(std::size_t i=0;i<theta.size();++i)value.gradient[i]=0.5*(statistics[i]+rho*baseline.trace[i])-
      0.5*(1.0+rho)*trace[i]-tau*shifted_trace[i];}
  value.feasible=std::isfinite(value.objective)&&std::isfinite(value.data)&&std::isfinite(value.prior)&&
    (!compute_gradient||std::all_of(value.gradient.begin(),value.gradient.end(),[](double x){return std::isfinite(x);}));
  if(!value.feasible)throw std::runtime_error("native covariance fit objective or analytic gradient is non-finite");
  value.status=CovarianceObjectiveStatus::Feasible;return value;
}

void complete_covariance_fit_gradient_from_factors(DeviceCovarianceSystem& system,CovarianceObjectiveEvaluation& value,
  const CovarianceBaseline& baseline,const std::vector<double>& statistics,const double rho,const double tau)
{
  system.worst_solve_relative_residual=0.0;
  const std::vector<double> trace=system.trace_derivatives(system.compact),shifted_trace=system.trace_derivatives(system.shifted);
  value.gradient.resize(statistics.size());for(std::size_t i=0;i<value.gradient.size();++i)
    value.gradient[i]=0.5*(statistics[i]+rho*baseline.trace[i])-0.5*(1.0+rho)*trace[i]-tau*shifted_trace[i];
  if(!std::all_of(value.gradient.begin(),value.gradient.end(),[](double x){return std::isfinite(x);}))
    throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance fit gradient from accepted Cholesky factors is non-finite");
}

std::vector<double> covariance_scaled_gradient(const std::vector<double>& gradient,const std::vector<double>& scale)
{if(gradient.size()!=scale.size())throw std::invalid_argument("native covariance fit gradient scaling dimensions do not match");std::vector<double> out(gradient.size());for(std::size_t i=0;i<gradient.size();++i)out[i]=gradient[i]*scale[i];return out;}

double covariance_norm_inf(const std::vector<double>& values)
{double norm=0.0;for(const double value:values)norm=std::max(norm,std::abs(value));return norm;}

bool covariance_build_scaled_trial(const std::vector<double>& theta,const std::vector<double>& z,const std::vector<double>& scale,
  const std::vector<double>& direction,const double step_scale,std::vector<double>& next_z,std::vector<double>& actual_step,
  std::vector<double>& next_theta)
{
  next_z=z;actual_step.resize(z.size());for(std::size_t i=0;i<z.size();++i){next_z[i]=z[i]+step_scale*direction[i];actual_step[i]=next_z[i]-z[i];}
  if(!std::all_of(next_z.begin(),next_z.end(),[](double value){return std::isfinite(value);})||
     !std::all_of(actual_step.begin(),actual_step.end(),[](double value){return std::isfinite(value);}))
    throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance scaled-coordinate trial is non-finite");
  if(next_z==z)return false;
  next_theta.resize(theta.size());for(std::size_t i=0;i<theta.size();++i)next_theta[i]=theta[i]+scale[i]*actual_step[i];
  if(!std::all_of(next_theta.begin(),next_theta.end(),[](double value){return std::isfinite(value);}))
    throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance trial parameter is non-finite");
  return next_theta!=theta;
}

bool covariance_directional_fd_stabilized(const double previous,const double current,int& stable_halvings,double& relative_change)
{
  relative_change=std::abs(current-previous)/std::max({std::abs(current),std::abs(previous),1e-12});
  if(relative_change<=0.01)++stable_halvings;else stable_halvings=0;
  return stable_halvings>=2;
}

const char* covariance_directional_fd_status(const double slope,const double finite_difference,const bool stable,double& relative_error)
{
  if(!stable)return "INCONCLUSIVE_UNSTABLE_FD";
  relative_error=std::abs(finite_difference-slope)/std::max({std::abs(finite_difference),std::abs(slope),1e-12});
  return (slope<0.0&&finite_difference>=0.0)||relative_error>0.5?"MISMATCH":"CONSISTENT";
}

void apply_covariance_directional_diagnostic(CovarianceFitResult& result,const char* status)
{
  result.directional_derivative_status=status;
  if(result.directional_derivative_status=="MISMATCH"){
    result.status="GRADIENT_CONSISTENCY_SUSPECT";
    result.reason+="; stabilized directional finite difference conflicts with the analytic descent prediction";
  }
}

CovarianceFitResult solve_covariance_reference(DeviceCovarianceSystem& system,const std::vector<double>& statistics,
  const double epsilon,const double rho,const int max_iterations,std::ostream* trace=nullptr)
{
  CovarianceFitResult result;result.statistics=statistics;const auto started=std::chrono::steady_clock::now();
  bool initialization_complete=false;
  auto flush_trace=[&](){if(trace){trace->flush();if(!*trace)throw std::runtime_error("failed writing covariance optimizer trace");}};
  try{
    CovarianceBaseline baseline;try{baseline=build_graph_covariance_baseline(system,statistics,epsilon,result.objective_evaluations);}
    catch(const std::exception& error){result.status=std::string(error.what()).find("BASELINE_SPD_FAIL")!=std::string::npos?"BASELINE_SPD_FAIL":"NUMERICAL_SOLVE_FAIL";result.reason=error.what();throw;}
    result.theta_baseline=baseline.theta;result.theta_start=baseline.theta;result.theta=baseline.theta;result.baseline_mu=baseline.mu;
    result.epsilon_constraint_status=baseline.epsilon_constraint_active?"ACTIVE":"INACTIVE";
    if(trace)*trace<<"BASELINE_SPD_PASS baseline_mu "<<baseline.mu<<" epsilon_constraint_status "<<result.epsilon_constraint_status<<'\n';flush_trace();
    const std::vector<double> coordinate_scale=system.coordinate_scales(baseline.theta,epsilon);
    double tau=1.0;const std::uint64_t max_evaluations=result.objective_evaluations+std::max<std::uint64_t>(100,20ULL*static_cast<std::uint64_t>(max_iterations));
    constexpr std::uint64_t max_initialization_evaluations=16;
    auto evaluate_initialization=[&](const std::vector<double>& theta,const bool compute_gradient){++result.initialization_evaluations;
      return evaluate_covariance_fit_objective(system,theta,baseline,statistics,rho,epsilon,tau,result.objective_evaluations,compute_gradient);};
    const bool search_start=baseline.epsilon_constraint_active;
    CovarianceObjectiveEvaluation current=evaluate_initialization(result.theta,!search_start);
    if(!current.feasible)throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance optimizer baseline is not a feasible interior point");
    std::vector<double> start_theta=baseline.theta;double start_delta=0.0,previous_objective=current.objective;
    if(baseline.epsilon_constraint_active){result.initialization_status="NO_IMPROVEMENT";const std::vector<double> direction=graph_laplacian_direction(system.graph);
      double delta_scale=std::numeric_limits<double>::infinity();for(std::size_t i=0;i<direction.size();++i)if(direction[i]>0.0)delta_scale=std::min(delta_scale,coordinate_scale[i]/direction[i]);
      double delta=0.01*delta_scale;std::string stop_status="SEARCH_LIMIT";
      if(!(delta>0.0)||!std::isfinite(delta))stop_status="STEP_UNDERFLOW";
      else for(std::uint64_t trial=1;trial<=max_initialization_evaluations-2&&result.initialization_evaluations+1<max_initialization_evaluations;++trial){
        if(result.objective_evaluations>=max_evaluations){stop_status="EVALUATION_BUDGET_EXHAUSTED";break;}
        std::vector<double> trial_theta(baseline.theta.size());for(std::size_t i=0;i<trial_theta.size();++i)trial_theta[i]=baseline.theta[i]+delta*direction[i];
        if(!std::all_of(trial_theta.begin(),trial_theta.end(),[](double value){return std::isfinite(value);}))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance initialization point is non-finite");
        if(trial_theta==start_theta){stop_status="STEP_UNDERFLOW";break;}
        const CovarianceObjectiveEvaluation trial_value=evaluate_initialization(trial_theta,false);
        if(trace){*trace<<"COVARIANCE_INITIALIZATION_TRIAL trial "<<trial<<" graph_delta "<<delta<<" status "<<covariance_objective_status_name(trial_value.status);
          if(trial_value.feasible)*trace<<" objective "<<trial_value.objective;*trace<<'\n';}flush_trace();
        if(!trial_value.feasible){stop_status="SHIFTED_NOT_SPD";break;}
        if(trial_value.objective<current.objective&&trial_value.objective<previous_objective){start_theta=trial_theta;start_delta=delta;current=trial_value;}
        if(!(trial_value.objective<previous_objective)){stop_status="OBJECTIVE_INCREASED";break;}
        previous_objective=trial_value.objective;delta*=2.0;
        if(!std::isfinite(delta)){stop_status="DELTA_LIMIT_REACHED";break;}
        if(trial==max_initialization_evaluations-2)stop_status="EVALUATION_LIMIT_REACHED";
      }
      if(start_delta>0.0)result.initialization_status=stop_status=="EVALUATION_LIMIT_REACHED"?"IMPROVED_EVALUATION_LIMIT":"IMPROVED";
      else result.initialization_status=stop_status=="OBJECTIVE_INCREASED"?"NO_IMPROVEMENT":"NO_IMPROVEMENT_"+stop_status;
    }else result.initialization_status="SKIPPED_BASELINE_INTERIOR";
    result.theta_start=start_theta;result.theta=result.theta_start;result.start_graph_delta=start_delta;
    if(search_start){current=evaluate_initialization(result.theta,true);
      if(!current.feasible)throw std::runtime_error("NUMERICAL_SOLVE_FAIL: selected covariance optimizer start is not feasible");}
    result.start_objective=current.objective;result.start_raw_gradient_inf_norm=covariance_norm_inf(current.gradient);
    if(trace){*trace<<"COVARIANCE_INITIALIZATION status "<<result.initialization_status<<" tau "<<tau<<" initialization_evaluations "<<result.initialization_evaluations
        <<" start_graph_delta "<<result.start_graph_delta<<" start_objective "<<result.start_objective
        <<" start_raw_gradient_inf_norm "<<result.start_raw_gradient_inf_norm<<"\ntheta_start "<<result.theta_start.size();
      for(const double value:result.theta_start)*trace<<' '<<value;*trace<<'\n';}flush_trace();
    initialization_complete=true;
    std::vector<double> z(result.theta.size());for(std::size_t i=0;i<z.size();++i)z[i]=result.theta[i]/coordinate_scale[i];
    std::vector<double> objective_gradient_scale(statistics.size());for(std::size_t i=0;i<statistics.size();++i)
      objective_gradient_scale[i]=0.5*(statistics[i]+rho*baseline.trace[i])*coordinate_scale[i];
    const double gradient_scale=std::max(1.0,covariance_norm_inf(objective_gradient_scale));
    const double target_gap=1e-6,tau_min=std::nextafter(target_gap/static_cast<double>(system.r),0.0);
    std::vector<double> inverse_hessian(z.size()*z.size(),0.0);auto reset_hessian=[&](){std::fill(inverse_hessian.begin(),inverse_hessian.end(),0.0);for(std::size_t i=0;i<z.size();++i)inverse_hessian[i*z.size()+i]=1.0;};reset_hessian();
    auto scaled_gradient=[&](const CovarianceObjectiveEvaluation& evaluation){return covariance_scaled_gradient(evaluation.gradient,coordinate_scale);};
    auto kkt=[&](const CovarianceObjectiveEvaluation& evaluation){return covariance_norm_inf(scaled_gradient(evaluation))/gradient_scale;};
    result.status="OPTIMIZING";
    while(result.iterations<max_iterations&&result.objective_evaluations<max_evaluations){
      result.scaled_kkt_residual=kkt(current);result.raw_gradient_inf_norm=covariance_norm_inf(current.gradient);
      if(result.scaled_kkt_residual<=1e-6){if(tau<=tau_min){result.converged=true;result.barrier_center_condition_pass=true;result.status="COVARIANCE_OPTIMIZER_CONVERGED";break;}
        tau=std::max(tau*0.2,tau_min);current=evaluate_covariance_fit_objective(system,result.theta,baseline,statistics,rho,epsilon,tau,result.objective_evaluations);reset_hessian();continue;}
      const std::vector<double> g=scaled_gradient(current);std::vector<double> direction(z.size(),0.0);
      for(std::size_t i=0;i<z.size();++i)for(std::size_t j=0;j<z.size();++j)direction[i]-=inverse_hessian[i*z.size()+j]*g[j];
      double slope=covariance_dot(g,direction);if(!std::isfinite(slope))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance BFGS direction is non-finite");
      if(!(slope<0.0)){reset_hessian();for(std::size_t i=0;i<z.size();++i)direction[i]=-g[i];slope=-covariance_dot(g,g);}
      if(!std::isfinite(slope)||!(slope<0.0))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance BFGS direction is not a finite descent direction");
      const double direction_inf_norm=covariance_norm_inf(direction);
      if(!std::isfinite(direction_inf_norm)||!(direction_inf_norm>0.0))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance BFGS direction has an invalid infinity norm");
      double alpha=std::min(1.0,1.0/direction_inf_norm);
      while(alpha*direction_inf_norm>1.0)alpha=std::nextafter(alpha,0.0);
      if(!std::isfinite(alpha)||!(alpha>0.0))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance BFGS initial step is not representable");
      bool accepted=false,step_underflow=false;int line_trials=0;std::vector<double> next_z,next_theta,step;CovarianceObjectiveEvaluation next;
      const int iteration=result.iterations;
      for(int trial=1;trial<=64&&result.objective_evaluations<max_evaluations;++trial){
        ++line_trials;++result.line_search_trials;
        if(!covariance_build_scaled_trial(result.theta,z,coordinate_scale,direction,alpha,next_z,step,next_theta)){step_underflow=true;
          if(trace)*trace<<"LINE_SEARCH iter="<<iteration<<" tau="<<tau<<" trial="<<trial<<" alpha="<<alpha
            <<" direction_inf_norm="<<direction_inf_norm<<" step_inf_norm="<<covariance_norm_inf(step)<<" status=STEP_UNDERFLOW\n";flush_trace();break;}
        const double step_inf_norm=covariance_norm_inf(step);
        ++result.line_search_evaluations;
        next=evaluate_covariance_fit_objective(system,next_theta,baseline,statistics,rho,epsilon,tau,result.objective_evaluations,false);
        const double armijo_required_delta=1e-4*alpha*slope;
        if(!std::isfinite(armijo_required_delta))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance Armijo threshold is non-finite");
        const char* trial_status=nullptr;double objective_delta=std::numeric_limits<double>::quiet_NaN();
        if(!next.feasible){++result.shifted_not_spd_trials;trial_status="SHIFTED_NOT_SPD";}
        else{objective_delta=next.objective-current.objective;const double armijo_bound=current.objective+armijo_required_delta;
          if(!std::isfinite(objective_delta)||!std::isfinite(armijo_bound))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance Armijo comparison is non-finite");
          if(next.objective<=armijo_bound){accepted=true;trial_status="ACCEPTED";}
          else{++result.armijo_reject_trials;trial_status="ARMIJO_REJECT";}}
        if(trace){*trace<<"LINE_SEARCH iter="<<iteration<<" tau="<<tau<<" trial="<<trial<<" alpha="<<alpha
          <<" direction_inf_norm="<<direction_inf_norm<<" step_inf_norm="<<step_inf_norm<<" status="<<trial_status;
          if(next.feasible)*trace<<" objective_delta="<<objective_delta<<" armijo_required_delta="<<armijo_required_delta;
          *trace<<'\n';}flush_trace();
        if(accepted)break;alpha*=0.5;
      }
      if(!accepted){
        if(result.objective_evaluations>=max_evaluations){result.status="EVALUATION_BUDGET_EXHAUSTED";result.line_search_failure_status="EVALUATION_BUDGET_EXHAUSTED";}
        else if(step_underflow){result.status="STEP_UNDERFLOW";result.line_search_failure_status="STEP_UNDERFLOW";}
        else{result.status="LINE_SEARCH_FAILED";result.line_search_failure_status="MAX_BACKTRACKS";}
        std::ostringstream reason;reason<<"covariance backtracking stopped after "<<line_trials<<" trials; SHIFTED_NOT_SPD="<<result.shifted_not_spd_trials
          <<" ARMIJO_REJECT="<<result.armijo_reject_trials;
        if(step_underflow)reason<<"; the actual theta update was not representable";
        else if(result.status=="EVALUATION_BUDGET_EXHAUSTED")reason<<"; the objective evaluation budget was exhausted";
        else reason<<"; no sufficient objective decrease was found";
        result.reason=reason.str();

        const double z_norm=covariance_norm_inf(z);double fd_step=std::cbrt(std::numeric_limits<double>::epsilon())*std::max(1.0,z_norm)/direction_inf_norm;
        if(!std::isfinite(fd_step))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance directional diagnostic step is non-finite");
        if(!(fd_step>0.0))apply_covariance_directional_diagnostic(result,"INCONCLUSIVE_STEP");
        else if(result.objective_evaluations>=max_evaluations)apply_covariance_directional_diagnostic(result,"INCONCLUSIVE_BUDGET");
        else{
          result.directional_derivative_slope=slope;bool derivative_found=false,saw_feasible=false,saw_roundoff=false,saw_step_underflow=false;
          for(int side=1;side>=-1&&!derivative_found;side-=2){double h=fd_step;
            bool have_previous=false;double previous_fd=0.0;int stable_halvings=0;
            for(int probe=0;probe<8&&result.directional_diagnostic_evaluations<16&&result.objective_evaluations<max_evaluations;++probe){
              std::vector<double> probe_z,probe_step,probe_theta;
              if(!covariance_build_scaled_trial(result.theta,z,coordinate_scale,direction,side*h,probe_z,probe_step,probe_theta)){
                saw_step_underflow=true;if(trace)*trace<<"DIRECTIONAL_DIAGNOSTIC side="<<(side>0?"POSITIVE":"NEGATIVE")<<" step="<<h<<" status=INCONCLUSIVE_STEP\n";flush_trace();break;}
              const CovarianceObjectiveEvaluation probe_value=evaluate_covariance_fit_objective(system,probe_theta,baseline,statistics,rho,epsilon,tau,result.objective_evaluations,false);
              ++result.directional_diagnostic_evaluations;const char* side_name=side>0?"POSITIVE":"NEGATIVE";
              if(trace)*trace<<"DIRECTIONAL_DIAGNOSTIC side="<<side_name<<" step="<<h<<" status="<<covariance_objective_status_name(probe_value.status)<<'\n';flush_trace();
              if(probe_value.feasible){
                saw_feasible=true;const double objective_change=probe_value.objective-current.objective;
                const double objective_resolution=32.0*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(probe_value.objective),std::abs(current.objective)});
                if(std::abs(objective_change)<=objective_resolution){saw_roundoff=true;
                  if(trace)*trace<<"DIRECTIONAL_DIAGNOSTIC side="<<side_name<<" step="<<h<<" status=INCONCLUSIVE_ROUNDOFF\n";flush_trace();break;}
                const double finite_difference=side>0?objective_change/h:-objective_change/h;
                if(!std::isfinite(finite_difference))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance directional finite difference is non-finite");
                result.directional_derivative_finite_difference=finite_difference;result.directional_derivative_step=h;result.directional_derivative_side=side_name;
                double relative_change=std::numeric_limits<double>::quiet_NaN();const bool stable=have_previous&&covariance_directional_fd_stabilized(previous_fd,finite_difference,stable_halvings,relative_change);
                double relative_error=std::numeric_limits<double>::quiet_NaN();const char* check_status=covariance_directional_fd_status(slope,finite_difference,stable,relative_error);
                if(trace){*trace<<"DIRECTIONAL_DIAGNOSTIC side="<<side_name<<" step="<<h<<" status=FEASIBLE finite_difference="<<finite_difference
                    <<" stability_change="<<relative_change<<" stable_halvings="<<stable_halvings<<'\n';}flush_trace();
                if(stable){apply_covariance_directional_diagnostic(result,check_status);derivative_found=true;
                  if(!std::isfinite(relative_error))throw std::runtime_error("NUMERICAL_SOLVE_FAIL: covariance directional derivative comparison is non-finite");
                  if(trace)*trace<<"DIRECTIONAL_DIAGNOSTIC_RESULT slope="<<slope<<" finite_difference="<<finite_difference
                    <<" relative_error="<<relative_error<<" status="<<result.directional_derivative_status<<'\n';flush_trace();
                  break;}
                previous_fd=finite_difference;have_previous=true;
              }else{have_previous=false;stable_halvings=0;}
              h*=0.5;
            }
          }
          if(!derivative_found)apply_covariance_directional_diagnostic(result,result.objective_evaluations>=max_evaluations||result.directional_diagnostic_evaluations>=16?"INCONCLUSIVE_BUDGET":
            saw_roundoff?"INCONCLUSIVE_ROUNDOFF":saw_step_underflow?"INCONCLUSIVE_STEP":saw_feasible?"INCONCLUSIVE_UNSTABLE_FD":"INCONCLUSIVE_NO_FEASIBLE_SIDE");
          if(trace&&result.directional_derivative_status.rfind("INCONCLUSIVE_",0)==0)*trace<<"DIRECTIONAL_DIAGNOSTIC_RESULT slope="<<slope
            <<" finite_difference="<<result.directional_derivative_finite_difference<<" step="<<result.directional_derivative_step
            <<" side="<<result.directional_derivative_side<<" status="<<result.directional_derivative_status<<'\n';flush_trace();
        }
        if(trace)*trace<<"LINE_SEARCH_FAILURE iter="<<iteration<<" status="<<result.status<<" reason="<<std::quoted(result.reason)
          <<" line_search_failure_status="<<result.line_search_failure_status<<" shifted_not_spd_trials="<<result.shifted_not_spd_trials
          <<" armijo_reject_trials="<<result.armijo_reject_trials<<" directional_derivative_status="<<result.directional_derivative_status<<'\n';flush_trace();
        break;
      }
      complete_covariance_fit_gradient_from_factors(system,next,baseline,statistics,rho,tau);
      if(trace)*trace<<"OPT_STEP iteration="<<iteration<<" tau="<<tau<<" objective_data="<<next.data<<" objective_prior="<<next.prior
        <<" objective_barrier="<<next.barrier<<" scaled_kkt_residual="<<kkt(next)<<" raw_gradient_inf_norm="<<covariance_norm_inf(next.gradient)
        <<" accepted_alpha="<<alpha<<" line_search_trials="<<line_trials<<'\n';flush_trace();
      const std::vector<double> next_gradient=scaled_gradient(next);std::vector<double> change(z.size());
      for(std::size_t i=0;i<z.size();++i)change[i]=next_gradient[i]-g[i];
      const double sy=covariance_dot(step,change);if(std::isfinite(sy)&&sy>1e-12*std::sqrt(covariance_dot(step,step)*covariance_dot(change,change))){
        const double inverse_sy=1.0/sy;std::vector<double> h_change(z.size());for(std::size_t i=0;i<z.size();++i)for(std::size_t j=0;j<z.size();++j)h_change[i]+=inverse_hessian[i*z.size()+j]*change[j];
        const double change_h_change=covariance_dot(change,h_change),factor=(1.0+change_h_change*inverse_sy)*inverse_sy;
        for(std::size_t i=0;i<z.size();++i)for(std::size_t j=0;j<z.size();++j)inverse_hessian[i*z.size()+j]+=factor*step[i]*step[j]-inverse_sy*(step[i]*h_change[j]+h_change[i]*step[j]);
      }else reset_hessian();
      z=std::move(next_z);result.theta=std::move(next_theta);current=std::move(next);++result.iterations;
    }
    if(!result.converged&&result.status=="OPTIMIZING"){
      result.scaled_kkt_residual=kkt(current);result.raw_gradient_inf_norm=covariance_norm_inf(current.gradient);
      if(result.scaled_kkt_residual<=1e-6&&tau<=tau_min){result.converged=true;result.barrier_center_condition_pass=true;result.status="COVARIANCE_OPTIMIZER_CONVERGED";}
      else{
        if(result.objective_evaluations>=max_evaluations)result.status="EVALUATION_BUDGET_EXHAUSTED";else result.status="ITERATION_BUDGET_EXHAUSTED";
        result.reason="the covariance optimizer exhausted its explicit iteration/evaluation budget before satisfying KKT and barrier-gap criteria";}
    }
    result.objective_data=current.data;result.objective_prior=current.prior;result.barrier_tau=tau;
    result.barrier_gap_estimate=result.barrier_center_condition_pass?system.r*tau:std::numeric_limits<double>::quiet_NaN();
    result.scaled_kkt_residual=kkt(current);result.raw_gradient_inf_norm=covariance_norm_inf(current.gradient);
  }catch(const std::exception& error){
    if(result.reason=="none")result.reason=error.what();
    else if(result.line_search_failure_status!="NONE")result.reason+="; numerical failure during line-search diagnosis: "+std::string(error.what());
    if(!initialization_complete&&!result.theta_baseline.empty())result.initialization_status="NUMERICAL_FAILURE";
    if(result.status!="BASELINE_SPD_FAIL")result.status="NUMERICAL_SOLVE_FAIL";}
  result.factorizations=system.factorizations;result.solve_rhs_columns=system.solve_rhs_columns;
  result.worst_solve_relative_residual=system.worst_solve_relative_residual;
  result.optimizer_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  return result;
}

void write_covariance_state(const std::string& path,const CovarianceFitResult& fit,const double rho,
  const std::string& samples,const std::string& raw)
{
  std::ostringstream out;out<<std::setprecision(17)<<"rpmd_ja Gaussian covariance shrinkage state\nfit_method gaussian_covariance_shrinkage\nstatus "<<fit.status
    <<"\nreason "<<fit.reason<<"\nsamples "<<std::quoted(samples)<<"\nqraw "<<std::quoted(raw)<<"\nshrinkage_strength "<<rho
    <<"\nbaseline_mu "<<fit.baseline_mu<<"\nbaseline_epsilon_constraint_status "<<fit.epsilon_constraint_status
    <<"\nobjective_data "<<fit.objective_data<<"\nobjective_prior "<<fit.objective_prior
    <<"\noptimizer_scaled_kkt_residual "<<fit.scaled_kkt_residual<<"\noptimizer_raw_gradient_inf_norm "<<fit.raw_gradient_inf_norm<<"\nbarrier_tau "<<fit.barrier_tau
    <<"\nbarrier_center_condition_status "<<(fit.barrier_center_condition_pass?"PASS":"NOT_VERIFIED")
    <<"\nbarrier_gap_estimate "<<fit.barrier_gap_estimate<<"\nobjective_evaluations "<<fit.objective_evaluations
    <<"\ninitialization_status "<<fit.initialization_status<<"\ninitialization_evaluations "<<fit.initialization_evaluations
    <<"\nstart_graph_delta "<<fit.start_graph_delta<<"\nstart_objective "<<fit.start_objective
    <<"\nstart_raw_gradient_inf_norm "<<fit.start_raw_gradient_inf_norm
    <<"\nline_search_failure_status "<<fit.line_search_failure_status<<"\nline_search_trials "<<fit.line_search_trials
    <<"\nline_search_evaluations "<<fit.line_search_evaluations
    <<"\nshifted_not_spd_trials "<<fit.shifted_not_spd_trials<<"\narmijo_reject_trials "<<fit.armijo_reject_trials
    <<"\ndirectional_derivative_status "<<fit.directional_derivative_status<<"\ndirectional_derivative_side "<<fit.directional_derivative_side
    <<"\ndirectional_derivative_slope "<<fit.directional_derivative_slope
    <<"\ndirectional_derivative_finite_difference "<<fit.directional_derivative_finite_difference
    <<"\ndirectional_derivative_step "<<fit.directional_derivative_step
    <<"\ndirectional_diagnostic_evaluations "<<fit.directional_diagnostic_evaluations
    <<"\nfactorizations "<<fit.factorizations<<"\nsolve_rhs_columns "<<fit.solve_rhs_columns
    <<"\nworst_solve_relative_residual "<<fit.worst_solve_relative_residual<<"\noptimizer_seconds "<<fit.optimizer_seconds<<"\niterations "<<fit.iterations
    <<"\ntheta_baseline "<<fit.theta_baseline.size();for(double value:fit.theta_baseline)out<<' '<<value;
  out<<"\ntheta_start "<<fit.theta_start.size();for(double value:fit.theta_start)out<<' '<<value;
  out<<"\ntheta "<<fit.theta.size();for(double value:fit.theta)out<<' '<<value;
  out<<"\ntraining_statistics "<<fit.statistics.size();for(double value:fit.statistics)out<<' '<<value;out<<'\n';
  write_text_exclusive(path,out.str(),"covariance optimizer state");
}

} // namespace

static bool fit_rpmd_ja_native_reference_impl(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  const std::uint64_t frame_count,Atom& atom,Box& box,Force& force,const bool sample_beads_from_header=false)
{
  const bool covariance_method=options.method==RpmdJANativeFitMethod::GaussianCovarianceShrinkage;
  if(options.output_path.empty()||options.kernel_table.empty()||spool_path.empty()||!std::isfinite(options.temperature)||options.temperature<=0.0||
     !std::isfinite(options.cutoff)||options.cutoff<=0.0||!std::isfinite(options.epsilon)||options.epsilon<=0.0||
     !std::isfinite(options.response_tolerance)||options.response_tolerance<0.0||!std::isfinite(options.ibp_tolerance)||options.ibp_tolerance<0.0||!std::isfinite(options.fd_step)||options.fd_step<=0.0||
     options.sample_interval<=0||(!covariance_method&&options.max_stability_rounds<=0)||
     (covariance_method&&(!std::isfinite(options.covariance_shrinkage)||options.covariance_shrinkage<=0.0||options.max_covariance_iterations<=0||options.raw_input_path.empty()))||frame_count<3)
    throw std::invalid_argument("invalid native rpmd_ja fit options");
  double cut_guard=0.0,cut_target=0.0;
  if(!covariance_method){cut_guard=options.epsilon*stability_cut_guard_fraction;cut_target=options.epsilon+cut_guard;
    if(!std::isfinite(cut_guard)||!(cut_guard>0.0)||!std::isfinite(cut_target)||!(cut_target>options.epsilon))throw std::invalid_argument("native fit epsilon cannot represent the stability cut guard");}
  const int max_rounds=options.max_stability_rounds;
  if(!covariance_method){std::printf("rpmd_ja fit stability round limit=%d\n",max_rounds);
    std::printf("rpmd_ja fit stability cut epsilon=%.17g cut_guard=%.17g cut_target=%.17g\n",options.epsilon,cut_guard,cut_target);}
  else std::printf("rpmd_ja fit method=gaussian_covariance_shrinkage rho=%.17g max_iter=%d; exact dense Cholesky is enabled\n",options.covariance_shrinkage,options.max_covariance_iterations);
  const int n=atom.number_of_atoms;
  if(frame_count>static_cast<std::uint64_t>(std::numeric_limits<int>::max())||n>std::numeric_limits<int>::max()/3)throw std::runtime_error("native fit dimensions exceed indexing limits");
  const int d=3*n,train=static_cast<int>(2*frame_count/3),validation=static_cast<int>(frame_count)-train;
  if(n<2||atom.cpu_mass.size()!=static_cast<std::size_t>(n)||atom.cpu_type.size()!=static_cast<std::size_t>(n)||validation<1)
    throw std::runtime_error("native rpmd_ja fit requires initialized atom masses/types and at least three frames");
  std::ifstream in(spool_path,std::ios::binary);if(!in)throw std::runtime_error("cannot open native rpmd_ja sample spool: "+spool_path);
  const SampleHeader header=read_header(in,frame_count,atom,box,options.temperature,false,sample_beads_from_header);
  std::set<int> unique_types(header.types.begin(),header.types.end());
  const int expected_probes=std::min(3*n-3,16+std::min(4,static_cast<int>(unique_types.size()))+4);
  if(validation<=expected_probes)throw std::runtime_error("native fit requires at least " + std::to_string(expected_probes+1) + " held-out frames for its fixed probe policy");
  std::vector<double> position(3*n),force_frame(3*n),r0(3*n,0.0);double step=0.0,previous=-std::numeric_limits<double>::infinity();
  read_training_r0(in,header,frame_count,static_cast<double>(options.sample_interval),r0);
  if(covariance_method)validate_fit_branches(in,header,frame_count,box,r0);
  const std::string generated_raw_path=options.output_path+".qraw",raw_path=options.raw_input_path.empty()?generated_raw_path:options.raw_input_path;
  const std::string pack_path=options.output_path+".additive.tmp",fit_path=options.output_path+".fit.txt",fit_tmp=fit_path+".tmp",trace_path=options.output_path+".fit_trace.txt",witness_path=options.output_path+".cg_witness.txt",qp_state_path=options.output_path+".qp_state.txt",response_state_path=options.output_path+".response_state.txt";
  const std::string covariance_state_path=options.output_path+".covariance_state.txt";
  const bool generate_raw=options.raw_input_path.empty();
  {std::ifstream kernel(options.kernel_table),old(options.output_path,std::ios::binary),side(options.output_path+".stability"),raw_old(generated_raw_path,std::ios::binary),pack_old(pack_path,std::ios::binary),fit_old(fit_path),fit_tmp_old(fit_tmp),trace_old(trace_path),witness_old(witness_path),qp_state_old(qp_state_path),response_state_old(response_state_path),covariance_state_old(covariance_state_path),failure(options.output_path+".failure.txt"),tmp(options.output_path+".tmp"),side_tmp(options.output_path+".stability.tmp");
    if(!kernel)throw std::runtime_error("native fit cannot read its qNEP kernel table");kernel.seekg(0,std::ios::end);const auto kernel_bytes=kernel.tellg();if(kernel_bytes<=0||kernel_bytes>std::numeric_limits<std::streamoff>::max())throw std::runtime_error("native fit qNEP kernel table has invalid or overflowing byte length");if(old.good()||side.good()||raw_old.good()||pack_old.good()||fit_old.good()||fit_tmp_old.good()||trace_old.good()||witness_old.good()||qp_state_old.good()||response_state_old.good()||covariance_state_old.good()||failure.good()||tmp.good()||side_tmp.good())throw std::runtime_error("native fit refuses to overwrite an existing output or scratch artifact");}
  const auto sample_statistics_started=std::chrono::steady_clock::now();
  const FixedProbeStatistics sample_statistics=collect_fixed_probe_statistics(in,header,frame_count,static_cast<std::uint64_t>(train),r0,false,options.ibp_tolerance,&box);
  const double sample_statistics_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-sample_statistics_started).count();
  print_fixed_probe_summary(sample_statistics,header,static_cast<double>(options.sample_interval),static_cast<std::uint64_t>(train),options.ibp_tolerance,sample_statistics_seconds);
  std::ofstream trace(trace_path,std::ios::out|std::ios::trunc);if(!trace)throw std::runtime_error("cannot create native fit trace: "+trace_path);
  trace<<std::setprecision(17)<<"evidence SAMPLED; "<<(covariance_method?"exact covariance SPD, held-out response, prepare and stored-D certificate remain required; no stability-cut feedback is used; IBP is diagnostic only":"QP_PASS is current finite cut-set only; Ritz/CG, held-out response and final stored-D Cholesky remain required; IBP is diagnostic only")
    <<"\nfit_method "<<(covariance_method?"gaussian_covariance_shrinkage":"force_least_squares_cut_qp")
    <<"\nreference_class finite_temperature_additive\n";
  if(!covariance_method)trace<<"rpmd_ja fit stability round limit="<<max_rounds<<'\n';
  trace<<"candidate_identity strategy native_additive_fixed_cutoff_graph output "<<std::quoted(options.output_path)<<" samples "<<std::quoted(spool_path)<<" raw "<<std::quoted(raw_path)
    <<"\ncandidate_parameters cutoff_A="<<options.cutoff<<" epsilon="<<options.epsilon<<" response_tolerance="<<options.response_tolerance
    <<" fd_step="<<options.fd_step<<" temperature_K="<<options.temperature<<" sample_interval="<<options.sample_interval;
  if(covariance_method)trace<<" shrinkage_strength="<<options.covariance_shrinkage<<" max_covariance_iterations="<<options.max_covariance_iterations;
  if(covariance_method)trace<<"\nstability_cut_policy NONE covariance_cholesky_enforces_D_minus_epsilon_I_SPD";
  else trace<<"\nstability_cut_policy epsilon="<<options.epsilon<<" cut_guard="<<cut_guard<<" cut_target="<<cut_target<<" guard_fraction="<<stability_cut_guard_fraction;
  trace<<"\nsample_IBP_status "<<sample_statistics.ibp_band.status<<" estimate "<<sample_statistics.ibp_band.estimate<<" q99_radius "<<sample_statistics.ibp_band.radius
    <<" interval "<<sample_statistics.ibp_band.lower<<' '<<sample_statistics.ibp_band.upper<<" tolerance "<<options.ibp_tolerance<<'\n';
  write_bootstrap_diagnostic(trace,"sample_IBP_DIAGNOSTIC",sample_statistics.ibp_band);
  write_probe_origin_diagnostics(trace,sample_statistics);trace.flush();if(!trace)throw std::runtime_error("failed writing native fit evidence header: "+trace_path);
  if(ibp_statistics_numerically_invalid(sample_statistics.ibp_band)){
    trace<<"PREFLIGHT_NUMERICAL_FAILURE sample_IBP_status "<<sample_statistics.ibp_band.status<<"\n";trace.flush();
    std::ostringstream failure;failure<<std::setprecision(17)<<"NATIVE_FIT_SAMPLE_IBP_NUMERICAL_FAILURE candidate_identity=native_additive_fixed_cutoff_graph output="
      <<std::quoted(options.output_path)<<" samples="<<std::quoted(spool_path)<<" status="<<sample_statistics.ibp_band.status
      <<" estimate="<<sample_statistics.ibp_band.estimate<<" q99_radius="<<sample_statistics.ibp_band.radius
      <<" interval="<<sample_statistics.ibp_band.lower<<' '<<sample_statistics.ibp_band.upper<<"; no qraw, QR, QP, or stability search was started ";
    write_bootstrap_diagnostic(failure,"IBP_DIAGNOSTIC",sample_statistics.ibp_band,false);throw std::runtime_error(failure.str());}
  // Rewind to the first frame for streaming block-QR fitting.
  in.clear();in.seekg(header.frames);
  previous=-std::numeric_limits<double>::infinity();
  const Graph graph=make_graph(r0,header.types,box,options.cutoff);
  const std::size_t psize=6*graph.type_pairs.size();
  if(psize==0||psize>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::runtime_error("native fit has an invalid parameter count");
  std::uint64_t design_bytes=0,qr_bytes=0;
  if(!covariance_method){const std::uint64_t frame_design=static_cast<std::uint64_t>(d)*psize,padded_rows=std::max<std::uint64_t>(d,psize);
    if(frame_design>std::numeric_limits<std::size_t>::max()/sizeof(double)||padded_rows>std::numeric_limits<std::size_t>::max()/psize/sizeof(double)||psize>std::numeric_limits<std::size_t>::max()/psize/sizeof(double))
      throw std::runtime_error("native fit parameter count exceeds actual frame-block or QR storage preflight");
    design_bytes=frame_design*sizeof(double);qr_bytes=padded_rows*psize*sizeof(double)+(padded_rows+psize)*sizeof(double)+psize*psize*sizeof(double);
    if(design_bytes>std::numeric_limits<std::size_t>::max()-qr_bytes)throw std::runtime_error("native fit actual host/GPU frame storage estimate overflows");}
  std::printf("    rpmd_ja native fit family: fixed cutoff graph, %zu unordered type pairs, %zu shared lab-frame symmetric parameters; training=%d validation=%d; fit design/QR %.3f MiB\n",
    graph.type_pairs.size(),psize,train,validation,static_cast<double>(design_bytes+qr_bytes)/(1024*1024));
  std::vector<double> saved;
  struct Restore { Atom& atom; std::vector<double>& x; bool active; ~Restore(){if(active)atom.position_per_atom.copy_from_host(x.data());} };
  if(generate_raw){if(atom.position_per_atom.size()!=static_cast<std::size_t>(d))throw std::runtime_error("native fit raw generation requires an initialized centroid position array");saved.resize(static_cast<std::size_t>(d));atom.position_per_atom.copy_to_host(saved.data());atom.position_per_atom.copy_from_host(r0.data());}
  Restore restore{atom,saved,generate_raw};
  const std::uint64_t d64=static_cast<std::uint64_t>(d),n64=static_cast<std::uint64_t>(n),dd64=d64*d64,vd64=d64*n64;
  if(dd64>(std::numeric_limits<std::uint64_t>::max()-18)/7||vd64>(std::numeric_limits<std::uint64_t>::max()-7*dd64-18)/2)throw std::runtime_error("native fit qraw size estimate overflows");
  const std::uint64_t raw_elements=2*vd64+7*dd64+18;if(raw_elements>std::numeric_limits<std::uint64_t>::max()/sizeof(double))throw std::runtime_error("native fit qraw byte estimate overflows");
  const std::uint64_t raw_bytes=raw_elements*sizeof(double);
  if(dd64>std::numeric_limits<std::size_t>::max()/sizeof(double)||raw_bytes>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))throw std::runtime_error("native fit qraw/Hessian exceeds host or file-offset limits");
  std::size_t free_before=0,total_before=0;if(cudaMemGetInfo(&free_before,&total_before)!=cudaSuccess)throw std::runtime_error("native fit CUDA memory preflight failed");
  std::printf("    rpmd_ja native fit preflight: qraw %.2f GiB, GPU baseline %.2f GiB, free GPU %.2f GiB, host design+QR %.2f MiB\n",static_cast<double>(raw_bytes)/(1024.0*1024.0*1024.0),static_cast<double>(dd64*sizeof(double))/(1024.0*1024.0*1024.0),static_cast<double>(free_before)/(1024.0*1024.0*1024.0),static_cast<double>(design_bytes+qr_bytes)/(1024.0*1024.0));
  if(dd64*sizeof(double)>free_before||free_before-dd64*sizeof(double)<256ULL*1024*1024)throw std::runtime_error("native fit baseline matrix does not fit current GPU memory with a 256 MiB safety margin");
  struct OwnScratch{std::string raw,pack,output,sidecar,fit,fit_tmp;bool raw_owned=false,pack_owned=false,output_owned=false,sidecar_owned=false,fit_owned=false,fit_tmp_owned=false;~OwnScratch(){if(raw_owned)std::remove(raw.c_str());if(pack_owned)std::remove(pack.c_str());if(output_owned)std::remove(output.c_str());if(sidecar_owned)std::remove(sidecar.c_str());if(fit_owned)std::remove(fit.c_str());if(fit_tmp_owned)std::remove(fit_tmp.c_str());}} own_scratch{generated_raw_path,pack_path,options.output_path,options.output_path+".stability",fit_path,fit_tmp};
  std::printf("    rpmd_ja fit reference policy: %s; raw qraw remains the unmodified native derivative source\n",
    options.internal_mass_com?"internal_mass_com_pullback_v1 (raw net force diagnostic only)":"strict raw zero-net gradient");
  if(generate_raw){generate_rpmd_ja_qnep_raw(generated_raw_path,options.temperature,options.fd_step,options.kernel_table,atom,box,force,!options.internal_mass_com);own_scratch.raw_owned=true;}
  std::ifstream raw(raw_path,std::ios::binary);if(!raw)throw std::runtime_error("native fit cannot read qNEP raw reference: "+raw_path);
  const QrawIdentity qraw_identity=validate_qraw_identity(raw,options,header,r0,box,force);
  const std::streamoff data=qraw_identity.data_offset,k_offset=qraw_identity.k_offset;
  const auto raw_model=qraw_identity.model,raw_config=qraw_identity.config;
  const std::uint32_t rawver=qraw_identity.version;const auto expected_model=raw_model,expected_config=raw_config;
  const int raw_charge=qraw_identity.charge,raw_kspace=qraw_identity.kspace;
  const double raw_mesh=qraw_identity.mesh,active_mesh=qraw_identity.active_mesh;  const CGReplayContext cg_replay{options,frame_count,raw_model,raw_config,expected_model,expected_config,raw_charge,raw_kspace,raw_mesh,active_mesh};
  std::vector<double> raw_gradient(d),site(n);for(int a=0;a<d;++a){raw.clear();raw.seekg(data+static_cast<std::streamoff>(a)*n*sizeof(double));read_array(raw,site.data(),n,"raw qNEP site gradient");if(!std::all_of(site.begin(),site.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit qNEP site gradient contains non-finite values");raw_gradient[a]=std::accumulate(site.begin(),site.end(),0.0);if(!std::isfinite(raw_gradient[a]))throw std::runtime_error("native fit qNEP reference gradient is non-finite");}
   own_scratch.raw_owned=false;
   const auto raw_net=rpmd_ja_reference_math::net_force_stats(raw_gradient,n);
  const std::vector<double> fitting_gradient=options.internal_mass_com ?
    rpmd_ja_reference_math::mass_com_covector_pullback(raw_gradient,atom.cpu_mass) : raw_gradient;
  const auto internal_net=rpmd_ja_reference_math::net_force_stats(fitting_gradient,n);
  double removed_normal_norm=0.0;
  for(int i=0;i<d;++i)removed_normal_norm=std::hypot(removed_normal_norm,raw_gradient[i]-fitting_gradient[i]);
  std::printf("    rpmd_ja fit reference gradient: policy=%s raw_net=(%.9g,%.9g,%.9g) raw_net_norm=%.9g raw_status=%s internal_net=(%.9g,%.9g,%.9g) internal_net_norm=%.9g internal_limit=%.9g internal_status=%s removed_mass_normal_norm=%.9g\n",
    options.internal_mass_com?"mass_com_internal_pullback_v1":"strict_net_v1",
    raw_net.net[0],raw_net.net[1],raw_net.net[2],raw_net.net_norm,
    options.internal_mass_com?"DIAGNOSTIC_ONLY":(raw_net.within_limit?"PASS":"FAIL"),
    internal_net.net[0],internal_net.net[1],internal_net.net[2],internal_net.net_norm,internal_net.limit,
    internal_net.within_limit?"PASS":"FAIL",removed_normal_norm);
  std::vector<std::array<double,3>> edge_linear;make_edge_linear(graph,fitting_gradient,n,edge_linear);
  std::vector<double> sqrt_mass(d),sqrt_mass_atom(n),q(d),f(d),base(d),target(d);
  for(int i=0;i<n;++i){if(!std::isfinite(atom.cpu_mass[i])||atom.cpu_mass[i]<=0.0)throw std::runtime_error("native fit requires positive finite masses");sqrt_mass_atom[i]=std::sqrt(atom.cpu_mass[i]);}
  for(int a=0;a<3;++a)for(int i=0;i<n;++i)sqrt_mass[a*n+i]=sqrt_mass_atom[i];
  std::vector<double> rmat,z,R,zblock;double training_force2=0.0,discarded2=0.0;
  if(!covariance_method){rmat.assign(psize*psize,0.0);z.assign(psize,0.0);}
  std::vector<double> design;if(!covariance_method)design.resize(static_cast<std::size_t>(d)*psize);
  DeviceBaseline baseline;baseline.initialize(raw,k_offset,d,n,atom.cpu_mass,sqrt_mass);
  DeviceQR qr;if(!covariance_method)qr.initialize(d,static_cast<int>(psize));
  raw.close();
  if(!covariance_method)for(int frame=0;frame<frame_count;++frame){
    read_frame(in,position,force_frame,step);if(!(step>previous)||(frame&&std::abs((step-previous)-options.sample_interval)>1e-9))throw std::runtime_error("native fit sample steps must strictly match sample_interval");previous=step;
    if(frame>=train)continue;
    for(int k=0;k<d;++k){q[k]=sqrt_mass[k]*(position[k]-r0[k]);f[k]=force_frame[k]/sqrt_mass[k];}
    project_translation(q,sqrt_mass_atom,n);
    project_translation(f,sqrt_mass_atom,n);
    baseline.apply(q,base);for(int i=0;i<d;++i)target[i]=f[i]+base[i];project_translation(target,sqrt_mass_atom,n);for(double v:f)training_force2+=v*v;
    make_design(graph,q,sqrt_mass,n,static_cast<int>(psize),design);
    qr.compress(design,target,R,zblock,discarded2);
    if(frame==0){rmat=std::move(R);z=std::move(zblock);}else merge_qr(rmat,z,R,zblock,static_cast<int>(psize),discarded2);
  }
  SmallSVD fit_svd;std::vector<std::vector<double>> cut_rows;std::vector<double> cut_rhs,qp_lambda,theta;
  double unconstrained_training_force_residual=std::numeric_limits<double>::quiet_NaN();
  double svd_sigma_max=std::numeric_limits<double>::quiet_NaN(),svd_sigma_min=std::numeric_limits<double>::quiet_NaN(),design_condition=std::numeric_limits<double>::quiet_NaN();
  if(!covariance_method){fit_svd=svd_small(qr.solver,rmat,z,static_cast<int>(psize));theta=theta_from_eta(fit_svd,fit_svd.eta);
    unconstrained_training_force_residual=compressed_training_force_residual(rmat,z,discarded2,training_force2,theta);
    svd_sigma_max=fit_svd.singular.front();svd_sigma_min=fit_svd.singular.back();design_condition=svd_sigma_max/svd_sigma_min;
    trace<<"FIT_DIAGNOSTIC stage=INITIAL_UNCONSTRAINED reference_class=finite_temperature_additive parameter_count="<<psize
      <<" training_force_residual="<<unconstrained_training_force_residual<<" force_fit_cost_increase=0"
      <<" force_fit_cost_definition=relative_squared_residual_delta_from_unconstrained svd_sigma_min="<<svd_sigma_min
      <<" svd_sigma_max="<<svd_sigma_max<<" design_condition="<<design_condition<<'\n';trace.flush();
    if(!trace)throw std::runtime_error("failed writing native fit unconstrained training diagnostic: "+trace_path);}
  bool fit_converged=false;ResponseCheck response;std::vector<std::string> saved_recovery_witness_paths;
  CovarianceFitResult covariance_fit;double covariance_baseline_force_residual=std::numeric_limits<double>::quiet_NaN();
  double training_force_residual=std::numeric_limits<double>::quiet_NaN(),force_fit_cost_increase=std::numeric_limits<double>::quiet_NaN();
  auto add_cut=[&](const std::vector<double>& v,const double base_value,const int outer,const char* source){
    const auto row=spectral_row(graph,v,sqrt_mass,n,static_cast<int>(psize));const double row_norm=std::sqrt(std::inner_product(row.begin(),row.end(),row.begin(),0.0));const double rhs=cut_target-base_value;const double value=std::inner_product(row.begin(),row.end(),theta.begin(),0.0)+base_value;
    double scale=0.0;const double curvature_tolerance=curvature_rounding_tolerance(options.epsilon,base_value,row,theta,scale);
    if(!std::isfinite(row_norm)||!std::isfinite(rhs)||!std::isfinite(value)||!std::isfinite(scale)||!std::isfinite(curvature_tolerance)||!std::all_of(row.begin(),row.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("NONFINITE_FIT_CURVATURE: stability cut row or scalar is non-finite");
    if(row_norm<=1e-20){if(value<options.epsilon-curvature_tolerance){trace<<"outer "<<outer<<" status STABILITY_CUT_STALLED source "<<source<<" curvature "<<value<<" epsilon "<<options.epsilon<<" cut_guard "<<cut_guard<<" cut_target "<<cut_target<<" curvature_scale "<<scale<<" curvature_tolerance "<<curvature_tolerance<<'\n';trace.flush();throw std::runtime_error("STABILITY_CUT_STALLED: unstable direction lies outside the fitted edge family");}return false;}
    for(std::size_t c=0;c<cut_rows.size();++c){double difference=0.0;for(std::size_t j=0;j<row.size();++j)difference=std::hypot(difference,row[j]-cut_rows[c][j]);const double prior_norm=std::sqrt(std::inner_product(cut_rows[c].begin(),cut_rows[c].end(),cut_rows[c].begin(),0.0));
      if(difference<=64.0*std::numeric_limits<double>::epsilon()*std::max(row_norm,prior_norm)&&std::abs(rhs-cut_rhs[c])<=curvature_tolerance){
        // The multiplier vector covers only cuts accepted by the previous QP.
        if(c>=qp_lambda.size()){trace<<"outer "<<outer<<" status DUPLICATE_PENDING source "<<source<<" constraint_index "<<c<<" solved_constraints "<<qp_lambda.size()<<" epsilon "<<options.epsilon<<" cut_guard "<<cut_guard<<" cut_target "<<cut_target<<" curvature_scale "<<scale<<" curvature_tolerance "<<curvature_tolerance<<'\n';return false;}
        if(value<options.epsilon-curvature_tolerance){trace<<"outer "<<outer<<" status STABILITY_CUT_STALLED source "<<source<<" constraint_index "<<c<<" solved_constraints "<<qp_lambda.size()<<" curvature "<<value<<" epsilon "<<options.epsilon<<" cut_guard "<<cut_guard<<" cut_target "<<cut_target<<" curvature_scale "<<scale<<" curvature_tolerance "<<curvature_tolerance<<'\n';trace.flush();throw std::runtime_error("STABILITY_CUT_STALLED: identical constraint remains violated; inspect fit_trace and cg_witness");}return false;}}
    cut_rows.push_back(row);cut_rhs.push_back(rhs);trace<<"outer "<<outer<<" status RETRY source "<<source<<" cuts "<<cut_rows.size()<<" curvature "<<value<<" epsilon "<<options.epsilon<<" cut_guard "<<cut_guard<<" cut_target "<<cut_target<<" curvature_scale "<<scale<<" curvature_tolerance "<<curvature_tolerance<<" rhs "<<rhs<<" theta";for(double x:theta)trace<<' '<<x;trace<<'\n';return true;
  };
  auto solve_qp_and_trace=[&](const int outer){
    const std::size_t warm_start=qp_lambda.size();auto qp=solve_cut_qp(qr.solver,fit_svd,cut_rows,cut_rhs,options.epsilon/4.0,qp_lambda,qp_state_path,outer);qp_lambda=std::move(qp.lambda);
    std::vector<double> fitted_theta=std::move(qp.certificate.theta);
    const double training_residual=compressed_training_force_residual(rmat,z,discarded2,training_force2,fitted_theta);
    const double force_fit_cost_increase=training_residual*training_residual-unconstrained_training_force_residual*unconstrained_training_force_residual;
    trace<<"QP_PASS outer="<<outer<<" constraints="<<cut_rows.size()<<" iterations="<<qp.iterations<<" warm_start="<<warm_start<<" method="<<qp.method
      <<" polish_attempts="<<qp.polish_attempts<<" active_constraints="<<qp.active_constraints<<" polish_updates="<<qp.polish_updates<<" polish_failure_reason="<<qp.polish_failure_reason
      <<" coordinate_seconds="<<qp.coordinate_seconds<<" polish_seconds="<<qp.polish_seconds
      <<" primal_excess="<<qp.certificate.max_primal_excess<<" complementarity="<<qp.certificate.max_complementarity
      <<" stationarity="<<qp.certificate.max_stationarity<<" lambda_change="<<qp.last_multiplier_change
      <<" parameter_count="<<psize<<" training_force_residual="<<training_residual<<" force_fit_cost_increase="<<force_fit_cost_increase
      <<" svd_sigma_min="<<svd_sigma_min<<" svd_sigma_max="<<svd_sigma_max<<" design_condition="<<design_condition<<'\n';trace.flush();
    if(!trace)throw std::runtime_error("failed writing native fit QP trace: "+trace_path);
    return fitted_theta;
  };
  if(covariance_method){
    const std::vector<double> statistics=collect_covariance_fit_statistics(in,header,frame_count,static_cast<std::uint64_t>(train),graph,r0,options.temperature);
    covariance_fit.statistics=statistics;
    try{DeviceCovarianceSystem covariance_system(baseline,graph,sqrt_mass,atom.cpu_mass);
      covariance_fit=solve_covariance_reference(covariance_system,statistics,options.epsilon,options.covariance_shrinkage,options.max_covariance_iterations,&trace);}
    catch(const std::exception& error){if(covariance_fit.reason=="none"){covariance_fit.status="NUMERICAL_SOLVE_FAIL";covariance_fit.reason=error.what();}}
    if(covariance_fit.theta_baseline.empty())
      trace<<"BASELINE_SPD_FAIL status "<<covariance_fit.status<<" reason "<<std::quoted(covariance_fit.reason)<<'\n';
    trace<<"COVARIANCE_OPTIMIZER status "<<covariance_fit.status<<" reason "<<std::quoted(covariance_fit.reason)
      <<" objective_evaluations "<<covariance_fit.objective_evaluations<<" factorizations "<<covariance_fit.factorizations
      <<" solve_rhs_columns "<<covariance_fit.solve_rhs_columns<<" optimizer_seconds "<<covariance_fit.optimizer_seconds
      <<" scaled_kkt_residual "<<covariance_fit.scaled_kkt_residual<<" raw_gradient_inf_norm "<<covariance_fit.raw_gradient_inf_norm<<" barrier_tau "<<covariance_fit.barrier_tau
      <<" initialization_status "<<covariance_fit.initialization_status<<" initialization_evaluations "<<covariance_fit.initialization_evaluations
      <<" start_graph_delta "<<covariance_fit.start_graph_delta<<" start_objective "<<covariance_fit.start_objective
      <<" line_search_failure_status "<<covariance_fit.line_search_failure_status<<" line_search_trials "<<covariance_fit.line_search_trials
      <<" line_search_evaluations "<<covariance_fit.line_search_evaluations
      <<" shifted_not_spd_trials "<<covariance_fit.shifted_not_spd_trials<<" armijo_reject_trials "<<covariance_fit.armijo_reject_trials
      <<" directional_derivative_status "<<covariance_fit.directional_derivative_status
      <<" barrier_center_condition_status "<<(covariance_fit.barrier_center_condition_pass?"PASS":"NOT_VERIFIED")
      <<" barrier_gap_estimate "<<covariance_fit.barrier_gap_estimate<<'\n';trace.flush();
    if(!covariance_fit.converged){write_covariance_state(covariance_state_path,covariance_fit,options.covariance_shrinkage,spool_path,raw_path);
      throw std::runtime_error(covariance_fit.reason.empty()?"COVARIANCE_OPTIMIZER_FAILED: inspect covariance state":covariance_fit.reason);}
    theta=covariance_fit.theta;
    trace<<"COVARIANCE_OPTIMIZER_CONVERGED scaled_kkt_residual "<<covariance_fit.scaled_kkt_residual
      <<" barrier_tau "<<covariance_fit.barrier_tau<<" barrier_center_condition_status PASS barrier_gap_estimate "<<covariance_fit.barrier_gap_estimate
      <<" objective_data "<<covariance_fit.objective_data<<" objective_prior "<<covariance_fit.objective_prior
      <<" raw_gradient_inf_norm "<<covariance_fit.raw_gradient_inf_norm<<'\n';trace.flush();
    cusolverDnHandle_t response_solver=nullptr;if(cusolverDnCreate(&response_solver)!=CUSOLVER_STATUS_SUCCESS){
      covariance_fit.status="FIT_RESPONSE_NUMERICAL_FAIL";covariance_fit.reason="native covariance fit response cuSOLVER handle creation failed";
      write_covariance_state(covariance_state_path,covariance_fit,options.covariance_shrinkage,spool_path,raw_path);
      throw std::runtime_error(covariance_fit.reason);}
    try{const int steps=std::min(96,d-3);DeviceLanczosWorkspace response_lanczos;
      response_lanczos.initialize(graph,sqrt_mass,sqrt_mass_atom,steps,psize);
      const auto modes=lanczos_low_modes(response_solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4,options.epsilon,{},&response_lanczos);
      response_lanczos.release();
      if(modes.empty())throw std::runtime_error("native covariance fit low-mode response probe search returned no modes");
      const auto residuals=compare_covariance_training_force_residuals(in,header,frame_count,static_cast<std::uint64_t>(train),graph,r0,baseline,sqrt_mass,sqrt_mass_atom,theta,covariance_fit.theta_baseline);
      training_force_residual=residuals.first;covariance_baseline_force_residual=residuals.second;
      ResponseCheck response_progress;
      response=validate_probes(response_solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,header.types,n,r0,in,header.frames,frame_count,train,
          options.sample_interval,options.temperature,modes,options.epsilon,&response_progress,options.response_tolerance,options.ibp_tolerance,0);
      if(!response.witness.classification.empty()){
        write_cg_witness(witness_path,response.witness,theta,raw_path,spool_path,cg_replay);
        trace<<"FIT_RESPONSE_FAIL status CG_FAILURE classification "<<response.witness.classification<<'\n';trace.flush();
        throw std::runtime_error("COVARIANCE_RESPONSE_CG_FAIL: exact covariance fit did not pass held-out probe CG; no stability feedback was applied");}
      const bool numerically_valid=std::isfinite(response.response)&&std::isfinite(response.ibp)&&response.response_band.status!="NUMERICAL_FAILURE"&&
        !ibp_statistics_numerically_invalid(response.ibp_band);
      const bool response_pass=numerically_valid&&response.response<=options.response_tolerance&&response.response_band.status=="RESPONSE_PASS";
      trace<<"FIT_DIAGNOSTIC stage=COVARIANCE_RESPONSE_CHECK reference_class=finite_temperature_additive training_force_residual="<<training_force_residual
        <<" baseline_training_force_residual="<<covariance_baseline_force_residual<<" response_ratio_min="<<response.response_ratio_min
        <<" response_ratio_max="<<response.response_ratio_max<<'\n';
      write_response_stats(trace,response);trace<<'\n';
      if(!response_pass){trace<<"FIT_RESPONSE_FAIL status "<<(numerically_valid?"RESPONSE_CHECK_FAIL":"NUMERICAL_FAILURE")<<'\n';trace.flush();
        write_response_state(response_state_path,response,theta,options.temperature,options.response_tolerance,options.ibp_tolerance);
        throw std::runtime_error("COVARIANCE_RESPONSE_FAIL: Gaussian covariance fit failed existing held-out response acceptance");}
      trace<<"FIT_RESPONSE_PASS response "<<response.response<<" tolerance "<<options.response_tolerance
        <<" response_ratio_min "<<response.response_ratio_min<<" response_ratio_max "<<response.response_ratio_max<<'\n';trace.flush();
      fit_converged=true;
    }catch(const std::exception& error){covariance_fit.status="FIT_RESPONSE_FAIL";covariance_fit.reason=error.what();
      trace<<"FIT_RESPONSE_ABORT detail "<<std::quoted(error.what())<<'\n';trace.flush();
      std::ifstream state_exists(covariance_state_path);if(!state_exists.good())write_covariance_state(covariance_state_path,covariance_fit,options.covariance_shrinkage,spool_path,raw_path);cusolverDnDestroy(response_solver);throw;}
    cusolverDnDestroy(response_solver);
  }else{
  int search_depth=std::min(96,d-3),consecutive_cg_feedback=0;
  std::vector<double> previous_low_ritz,cg_witness_seed;
  const int max_search_depth=std::min(384,d-3);
  DeviceLanczosWorkspace lanczos_workspace;lanczos_workspace.initialize(graph,sqrt_mass,sqrt_mass_atom,max_search_depth,psize);
  auto deepen_search=[&](){if(search_depth<max_search_depth)search_depth=std::min(max_search_depth,2*search_depth);};
  for(int outer=0;outer<max_rounds;++outer){
    const int steps=std::min(search_depth,d-3);
    const std::string seed_source=!cg_witness_seed.empty()?"CG_WITNESS":(!previous_low_ritz.empty()?"LOWEST_RITZ":"FIXED_SEED");
    const std::vector<double> initial_vector=!cg_witness_seed.empty()?cg_witness_seed:previous_low_ritz;
    const auto warm_search_started=std::chrono::steady_clock::now();
    auto primary_modes=lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4,options.epsilon,initial_vector,&lanczos_workspace);
    const double warm_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-warm_search_started).count();
    cg_witness_seed.clear();
    if(primary_modes.empty())throw std::runtime_error("native fit Lanczos returned no Ritz modes");
    const char* search_policy=steps<=96?"FAST_CUT":"FULL_SEARCH";
    auto record_search=[&](const std::string& source,const std::vector<RitzMode>& found){
      const RitzMode& summary=found.front();trace<<"outer "<<outer<<" SEARCH search_policy "<<search_policy<<" seed_source "<<source<<" lanczos_steps "<<steps<<" lanczos_steps_requested "<<steps<<" lanczos_steps_actual "<<summary.steps_used
        <<" dense_matrix_actions "<<summary.dense_matrix_actions<<" early_exit "<<(summary.early_exit?1:0)<<" returned_directions "<<found.size()<<" actual_rayleighs";
      for(const auto& mode:found)trace<<' '<<mode.actual_rayleigh;
      trace<<" curvature_scales";
      for(const auto& mode:found)trace<<' '<<mode.curvature_scale;
      trace<<" curvature_tolerances";
      for(const auto& mode:found)trace<<' '<<mode.curvature_tolerance;
      trace<<" actual_residuals";
      for(const auto& mode:found)trace<<' '<<mode.residual;
      trace<<'\n';
    };
    record_search(seed_source,primary_modes);
    const int primary_steps_used=primary_modes.front().steps_used,primary_dense_matrix_actions=primary_modes.front().dense_matrix_actions;
    std::vector<RitzMode> independent_modes;int independent_steps_used=0,independent_dense_matrix_actions=0;
    double independent_seconds=0.0;
    if(!initial_vector.empty()){
      const auto independent_search_started=std::chrono::steady_clock::now();
      independent_modes=lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,steps,4,options.epsilon,{},&lanczos_workspace);
      independent_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-independent_search_started).count();
      record_search("FIXED_INDEPENDENT",independent_modes);
      if(independent_modes.empty())throw std::runtime_error("native fit independent-seed Lanczos returned no Ritz modes");
      independent_steps_used=independent_modes.front().steps_used;independent_dense_matrix_actions=independent_modes.front().dense_matrix_actions;
    }
    const int dense_matrix_actions_total=primary_dense_matrix_actions+independent_dense_matrix_actions;
    trace<<"outer "<<outer<<" SEARCH search_policy="<<search_policy<<" warm_seconds="<<warm_seconds<<" independent_seconds="<<independent_seconds<<" total_seconds="<<warm_seconds+independent_seconds
      <<" requested_steps="<<steps<<" warm_steps_actual="<<primary_steps_used<<" independent_steps_actual="<<independent_steps_used
      <<" dense_matrix_actions_total="<<dense_matrix_actions_total<<'\n';trace.flush();
    if(!trace)throw std::runtime_error("failed writing native fit search timing: "+trace_path);
    auto inspect_modes=[&](const std::vector<RitzMode>& candidates,const std::string& source,bool& observed_violation,bool& added_cut){
      for(const auto& mode:candidates){if(!std::isfinite(mode.base_rayleigh)||!std::isfinite(mode.additive_rayleigh)||!std::isfinite(mode.actual_rayleigh)||!std::isfinite(mode.curvature_scale)||!std::isfinite(mode.curvature_tolerance))throw std::runtime_error("NONFINITE_FIT_CURVATURE: sampled Ritz evidence is non-finite");if(!stability_cut_is_violated(mode.actual_rayleigh,options.epsilon,mode.curvature_tolerance))continue;observed_violation=true;added_cut|=add_cut(mode.vector,mode.base_rayleigh,outer,("RITZ_"+source).c_str());}
    };
    bool observed_violation=false,added_cut=false;
    inspect_modes(primary_modes,seed_source,observed_violation,added_cut);
    inspect_modes(independent_modes,"FIXED_INDEPENDENT",observed_violation,added_cut);
    std::vector<RitzMode> modes=std::move(primary_modes);modes.insert(modes.end(),std::make_move_iterator(independent_modes.begin()),std::make_move_iterator(independent_modes.end()));
    std::sort(modes.begin(),modes.end(),[](const RitzMode& a,const RitzMode& b){return a.value<b.value;});
    if(modes.empty())throw std::runtime_error("native fit Lanczos returned no Ritz modes");
    previous_low_ritz=modes.front().vector;
    const double min_ritz=modes.front().value,ritz_residual=modes.front().residual;
    const bool near_boundary=std::abs(min_ritz-options.epsilon)<=std::max(options.epsilon,2.0*ritz_residual);
    const bool residual_large=ritz_residual>std::max(1e-12,options.epsilon/4.0);
    if(near_boundary&&residual_large)deepen_search();
    trace<<"outer "<<outer<<" sampled_min_ritz "<<min_ritz<<" sampled_min_rayleigh "<<modes.front().actual_rayleigh<<" ritz_residual "<<ritz_residual<<" ritz_residual_kind actual_Dv_minus_rayleigh_v lanczos_steps "<<steps
      <<" lanczos_steps_requested "<<steps<<" warm_steps_actual "<<primary_steps_used<<" independent_steps_actual "<<independent_steps_used<<" dense_matrix_actions "<<dense_matrix_actions_total
      <<" next_lanczos_steps "<<search_depth<<" observed_directions "<<modes.size()<<" constraints "<<cut_rows.size()<<" cg_feedback none status "<<(observed_violation?"RETRY":"CHECK_CG")<<" theta";for(double x:theta)trace<<' '<<x;trace<<'\n';trace.flush();if(!trace)throw std::runtime_error("failed writing native fit trace: "+trace_path);
    if(observed_violation){consecutive_cg_feedback=0;if(!added_cut)throw std::runtime_error("STABILITY_CUT_STALLED: sampled Ritz violation added no new cut");if(outer==max_rounds-1)throw std::runtime_error("STABILITY_CUT_LIMIT: sampled Ritz violations remain after "+std::to_string(max_rounds)+" rounds");theta=solve_qp_and_trace(outer);continue;}
    ResponseCheck response_progress;
    try{response=validate_probes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,header.types,n,r0,in,header.frames,frame_count,train,options.sample_interval,options.temperature,modes,options.epsilon,&response_progress,options.response_tolerance,options.ibp_tolerance,outer);}
    catch(const std::exception& e){if(!response_progress.witness.classification.empty())write_cg_witness(witness_path,response_progress.witness,theta,raw_path,spool_path,cg_replay);
      write_recovered_cg_witnesses(response_progress,witness_path,theta,raw_path,spool_path,cg_replay,saved_recovery_witness_paths);
      trace<<"outer "<<outer<<" status PROBE_VALIDATION_FAIL detail "<<e.what()<<" ";write_response_stats(trace,response_progress);trace<<'\n';trace.flush();throw;}
    write_recovered_cg_witnesses(response,witness_path,theta,raw_path,spool_path,cg_replay,saved_recovery_witness_paths);
    if(response.witness.classification.empty()){
      if(!std::isfinite(response.response)||response.response_band.status=="NUMERICAL_FAILURE"||!std::isfinite(response.ibp)||ibp_statistics_numerically_invalid(response.ibp_band)){
        trace<<"outer "<<outer<<" status RESPONSE_NUMERICAL_FAILURE ";write_response_stats(trace,response);trace<<'\n';trace.flush();
        throw std::runtime_error("NONFINITE_RESPONSE_VALIDATION: held-out response or IBP arithmetic failed numerically");}
      const bool response_pass=response.response<=options.response_tolerance&&response.response_band.status=="RESPONSE_PASS";
      const double response_training_residual=compressed_training_force_residual(rmat,z,discarded2,training_force2,theta);
      const double response_force_fit_cost_increase=response_training_residual*response_training_residual-unconstrained_training_force_residual*unconstrained_training_force_residual;
      trace<<"FIT_DIAGNOSTIC stage=RESPONSE_CHECK outer="<<outer<<" reference_class=finite_temperature_additive parameter_count="<<psize
        <<" training_force_residual="<<response_training_residual<<" force_fit_cost_increase="<<response_force_fit_cost_increase
        <<" force_fit_cost_definition=relative_squared_residual_delta_from_unconstrained svd_sigma_min="<<svd_sigma_min
        <<" svd_sigma_max="<<svd_sigma_max<<" design_condition="<<design_condition<<" response_ratio_min="<<response.response_ratio_min
        <<" response_ratio_max="<<response.response_ratio_max<<'\n';trace.flush();
      std::printf("    rpmd_ja fit RESPONSE_CHECK %s: response=%.6g/%.6g; IBP_DIAGNOSTIC status=%s value=%.6g/%.6g; CG verification=%s residual=%.3g recovered_probes=%zu; frames=%llu probes=%d covariance condition observed/predicted=%.3g/%.3g block variance delta=%.3g; FULL_CERTIFICATE PENDING\n",
        response_pass?"PASS":"FAIL",response.response,options.response_tolerance,response.ibp_band.status.c_str(),response.ibp,options.ibp_tolerance,response.cg_verification_method.c_str(),response.cg,response.cg_probe_recoveries.size(),
        static_cast<unsigned long long>(response.validation_frames),response.probes,response.observed_cov_condition,response.predicted_cov_condition,response.block_variance_max_relative_delta);
      trace<<"outer "<<outer<<" status CG_PASS response_check "<<(response_pass?"PASS":"FAIL")<<" response_IBP_status "<<response.ibp_band.status<<" constraints "<<cut_rows.size()<<" ";
      write_response_stats(trace,response);trace<<" response_tolerance "<<options.response_tolerance<<" ibp_tolerance "<<options.ibp_tolerance<<" theta";for(double x:theta)trace<<' '<<x;trace<<'\n';trace.flush();
      if(!response_pass){write_response_state(response_state_path,response,theta,options.temperature,options.response_tolerance,options.ibp_tolerance);
        std::printf("    RESPONSE_CHECK_FAIL response=%.6g; IBP_DIAGNOSTIC status=%s value=%.6g; snapshot=%s\n",response.response,response.ibp_band.status.c_str(),response.ibp,response_state_path.c_str());
        throw std::runtime_error("RESPONSE_CHECK_FAIL: native fit held-out probe response exceeds its declared tolerance; inspect "+response_state_path);}
      trace<<"FIT_RESPONSE_PASS outer="<<outer<<" reference_class=finite_temperature_additive response="<<response.response
        <<" tolerance="<<options.response_tolerance<<" response_ratio_min="<<response.response_ratio_min<<" response_ratio_max="<<response.response_ratio_max<<'\n';trace.flush();
      if(!trace)throw std::runtime_error("failed writing native fit response pass status: "+trace_path);
      fit_converged=true;break;}
    write_cg_witness(witness_path,response.witness,theta,raw_path,spool_path,cg_replay);const bool feedback_eligible=response.witness.finite&&(response.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION"||response.witness.classification=="UNRESOLVED_SOFT_DIRECTION");
    if(response.witness.classification=="TRUE_RESIDUAL_FAILURE")std::printf("    rpmd_ja fit CG_FEEDBACK FAIL: %s probe=%d iteration=%d recursive_residual=%.3g true_residual=%.3g restarts=%d; witness=%s\n",response.witness.classification.c_str(),response.witness.probe,response.witness.iteration,response.witness.recursive_relative_residual,response.witness.relative_residual,response.witness.residual_restarts,witness_path.c_str());
    else std::printf("    rpmd_ja fit CG_FEEDBACK %s: %s probe=%d iteration=%d lambda=%.17g base=%.17g add=%.17g tol=%.3g repeat_delta=%.3g recursive_residual=%.3g restarts=%d; witness=%s\n",feedback_eligible?"RETRY":"FAIL",response.witness.classification.c_str(),response.witness.probe,response.witness.iteration,response.witness.rayleigh,response.witness.base_rayleigh,response.witness.add_rayleigh,response.witness.curvature_tolerance,response.witness.repeat_diff_norm,response.witness.recursive_relative_residual,response.witness.residual_restarts,witness_path.c_str());
    if(!feedback_eligible){trace<<"outer "<<outer<<" status CG_FAILURE classification="<<response.witness.classification<<" probe="<<response.witness.probe<<" iteration="<<response.witness.iteration<<" restarts="<<response.witness.residual_restarts<<" snapshot="<<std::quoted(witness_path)<<' ';write_response_stats(trace,response);trace<<'\n';trace.flush();if(!trace)throw std::runtime_error("failed writing native fit CG failure diagnostics: "+trace_path);throw std::runtime_error("CG stability feedback failed with classification "+response.witness.classification+"; inspect "+witness_path);}
    const double base_value=std::inner_product(response.witness.direction.begin(),response.witness.direction.end(),response.witness.base.begin(),0.0);if(response.witness.rayleigh>=options.epsilon-response.witness.curvature_tolerance)throw std::runtime_error("NUMERICAL_BREAKDOWN: normalized CG witness is not below the declared curvature threshold; inspect "+witness_path);const bool added=add_cut(response.witness.direction,base_value,outer,response.witness.classification.c_str());
    if(!added)throw std::runtime_error("STABILITY_CUT_STALLED: CG witness did not add a new violated constraint");
    cg_witness_seed=response.witness.direction;++consecutive_cg_feedback;if(consecutive_cg_feedback>=2)deepen_search();
    trace<<"outer "<<outer<<" cg_feedback "<<response.witness.classification<<" probe "<<response.witness.probe<<" iteration "<<response.witness.iteration<<" rayleigh "<<response.witness.rayleigh<<" lanczos_steps "<<steps<<" lanczos_steps_requested "<<steps<<" warm_steps_actual "<<primary_steps_used<<" independent_steps_actual "<<independent_steps_used<<" dense_matrix_actions "<<dense_matrix_actions_total<<" next_lanczos_steps "<<search_depth<<" consecutive_cg_feedback "<<consecutive_cg_feedback<<" constraints "<<cut_rows.size()<<" status RETRY ";write_response_stats(trace,response);trace<<'\n';trace.flush();
    if(outer==max_rounds-1)throw std::runtime_error("STABILITY_CUT_LIMIT: CG feedback exhausted "+std::to_string(max_rounds)+" rounds");theta=solve_qp_and_trace(outer);
  }
  if(!fit_converged)throw std::runtime_error("STABILITY_CUT_LIMIT: native fit did not pass sampled Ritz and CG checks within "+std::to_string(max_rounds)+" rounds");lanczos_workspace.release();
  training_force_residual=compressed_training_force_residual(rmat,z,discarded2,training_force2,theta);
  force_fit_cost_increase=training_force_residual*training_force_residual-unconstrained_training_force_residual*unconstrained_training_force_residual;
  }
  if(!std::isfinite(response.response)||!std::isfinite(response.ibp)||response.response_band.status=="NUMERICAL_FAILURE"||ibp_statistics_numerically_invalid(response.ibp_band))
    throw std::runtime_error("NONFINITE_RESPONSE_VALIDATION: held-out response or IBP arithmetic failed numerically");
  if(response.response>options.response_tolerance||response.response_band.status!="RESPONSE_PASS")
    throw std::runtime_error("native fit held-out probe response exceeds the declared tolerance");
  std::uint64_t estimated=88;
  for(int i=0;i<n;++i){const std::uint64_t zsite=graph.sites[i].size();estimated+=sizeof(std::int32_t)+zsite*16+zsite*3*sizeof(double)+9*zsite*zsite*sizeof(double);}
  if(estimated>std::numeric_limits<std::streamsize>::max())throw std::runtime_error("native additive package byte estimate overflows");
  {
    std::ofstream out(pack_path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create native additive package");own_scratch.pack_owned=true;
    out.write(additive_magic,8);write(out,std::uint32_t(options.internal_mass_com?2:1));write(out,native_endian);write(out,std::int32_t(n));write(out,std::int32_t(header.beads));write(out,rpmd_ja_model_fingerprint(raw_path));
    write(out,options.temperature);write(out,options.epsilon);write(out,response.response);write(out,options.response_tolerance);write(out,std::uint64_t(train));write(out,std::uint64_t(validation));
    for(int i=0;i<n;++i){const int zs=static_cast<int>(graph.sites[i].size());write(out,std::int32_t(zs));for(const auto&e:graph.sites[i]){write(out,std::int32_t(e.j));for(int a=0;a<3;++a)write(out,std::int32_t(e.image[a]));}
      for(const auto&e:graph.sites[i])for(int a=0;a<3;++a)write(out,.5*e.sign*edge_linear[e.edge_index][a]);
      const int qdim=3*zs;for(int a=0;a<qdim;++a)for(int b=0;b<qdim;++b){double v=0.0;if(a/3==b/3&&a%3<=b%3){const int group=graph.sites[i][a/3].group;const int lo=std::min(a%3,b%3),hi=std::max(a%3,b%3);const int ci=lo==0?(hi==0?0:(hi==1?1:2)):(lo==1?(hi==1?3:4):5);v=.5*theta[static_cast<std::size_t>(6*group+ci)];}else if(a/3==b/3){const int group=graph.sites[i][a/3].group;const int lo=std::min(a%3,b%3),hi=std::max(a%3,b%3);const int ci=lo==0?(hi==0?0:(hi==1?1:2)):(lo==1?(hi==1?3:4):5);v=.5*theta[static_cast<std::size_t>(6*group+ci)];}write(out,v);}
    }
    out.close();if(!out)throw std::runtime_error("failed closing native additive package");
  }
  std::printf("    rpmd_ja fit/probe validation complete; preparing reference: family=shared_lab_frame_typepair_edge_blocks parameters=%zu; restricted family, probes=%d seed=0x243f6a8885a308d3; train/heldout force residual=%.6g/%.6g response=%.6g IBP=%.6g CG residual=%.3g; stability cuts=%zu; estimated additive bytes=%llu\n",psize,response.probes,training_force_residual,response.force_residual,response.response,response.ibp,response.cg,cut_rows.size(),static_cast<unsigned long long>(estimated));
  qr.release();baseline.release();
  try{prepare_rpmd_ja_qnep_reference(raw_path,options.output_path,options.kernel_table,make_rpmd_ja_qnep_mode_validator(atom,box,force),pack_path);}
  catch(const std::exception& e){
    if(preserve_candidate_package_on_full_spd_failure(e.what(),own_scratch.pack_owned)){
      trace<<"FULL_CERTIFICATE FAIL reason stored_D_cholesky candidate_additive_package "<<pack_path<<" qraw "<<raw_path<<"; full candidate direction and unshifted-D curvature are in "<<options.output_path<<".failure.txt; no automatic stability cut was added\n";trace.flush();
    }
    throw;
  }
  own_scratch.output_owned=true;own_scratch.sidecar_owned=true;
  double certificate_epsilon=0.0,certificate_eta=0.0;{std::ifstream stability(options.output_path+".stability");std::string key;while(stability>>key){if(key=="epsilon_num")stability>>certificate_epsilon;else if(key=="reconstruction_bound")stability>>certificate_eta;else{std::string value;stability>>value;}}if(!stability.eof()||!std::isfinite(certificate_epsilon)||!std::isfinite(certificate_eta)||!(certificate_epsilon>0.0)||!(certificate_eta<0.5*certificate_epsilon))throw std::runtime_error("native fit could not read a valid stored-D Cholesky certificate");}
  trace<<"FULL_CERTIFICATE_PASS stored_D_cholesky epsilon_num "<<certificate_epsilon<<" reconstruction_bound "<<certificate_eta<<'\n'
    <<"FULL_CERTIFICATE PASS stored_D_cholesky epsilon_num "<<certificate_epsilon<<" reconstruction_bound "<<certificate_eta<<'\n';trace.flush();
  std::ifstream written_reference(options.output_path,std::ios::binary);if(!written_reference)throw std::runtime_error("native fit prepare returned without writing the reference package");
  written_reference.seekg(0,std::ios::end);if(written_reference.tellg()<=std::streampos(0))throw std::runtime_error("native fit prepare wrote an empty reference package");
  std::printf("    rpmd_ja fit FULL_CERTIFICATE PASS: stored-D Cholesky epsilon_num=%.6g reconstruction_bound=%.6g\n",certificate_epsilon,certificate_eta);
  {std::ofstream fitout(fit_tmp,std::ios::out|std::ios::trunc);if(!fitout)throw std::runtime_error("cannot create native fit diagnostics");own_scratch.fit_tmp_owned=true;
    fitout<<std::setprecision(17)<<"candidate_identity strategy native_additive_fixed_cutoff_graph output "<<std::quoted(options.output_path)<<" samples "<<std::quoted(spool_path)<<" raw "<<std::quoted(raw_path)
      <<"\nreference_class finite_temperature_additive\nfit_response_status FIT_RESPONSE_PASS\nfull_certificate_status FULL_CERTIFICATE_PASS\nreference_write_status REFERENCE_WRITTEN"
      <<"\nparameter_count "<<psize;
    if(covariance_method){fitout<<"\nfit_method gaussian_covariance_shrinkage\nshrinkage_strength "<<options.covariance_shrinkage
      <<"\nbaseline_mu "<<covariance_fit.baseline_mu<<"\nbaseline_epsilon_constraint_status "<<covariance_fit.epsilon_constraint_status
      <<"\ninitialization_status "<<covariance_fit.initialization_status<<"\ninitialization_evaluations "<<covariance_fit.initialization_evaluations
      <<"\nstart_graph_delta "<<covariance_fit.start_graph_delta<<"\nstart_objective "<<covariance_fit.start_objective
      <<"\nstart_raw_gradient_inf_norm "<<covariance_fit.start_raw_gradient_inf_norm
      <<"\nobjective_data "<<covariance_fit.objective_data<<"\nobjective_prior "<<covariance_fit.objective_prior
      <<"\noptimizer_scaled_kkt_residual "<<covariance_fit.scaled_kkt_residual<<"\noptimizer_raw_gradient_inf_norm "<<covariance_fit.raw_gradient_inf_norm<<"\nbarrier_tau "<<covariance_fit.barrier_tau
        <<"\nbarrier_center_condition_status "<<(covariance_fit.barrier_center_condition_pass?"PASS":"NOT_VERIFIED")
        <<"\nbarrier_gap_estimate "<<covariance_fit.barrier_gap_estimate<<"\nobjective_evaluations "<<covariance_fit.objective_evaluations
      <<"\nfactorizations "<<covariance_fit.factorizations<<"\nsolve_rhs_columns "<<covariance_fit.solve_rhs_columns
      <<"\nworst_solve_relative_residual "<<covariance_fit.worst_solve_relative_residual<<"\noptimizer_seconds "<<covariance_fit.optimizer_seconds
      <<"\nline_search_failure_status "<<covariance_fit.line_search_failure_status<<"\nline_search_trials "<<covariance_fit.line_search_trials
      <<"\nline_search_evaluations "<<covariance_fit.line_search_evaluations
      <<"\nshifted_not_spd_trials "<<covariance_fit.shifted_not_spd_trials<<"\narmijo_reject_trials "<<covariance_fit.armijo_reject_trials
      <<"\ndirectional_derivative_status "<<covariance_fit.directional_derivative_status
      <<"\ndirectional_derivative_side "<<covariance_fit.directional_derivative_side
      <<"\ndirectional_derivative_slope "<<covariance_fit.directional_derivative_slope
      <<"\ndirectional_derivative_finite_difference "<<covariance_fit.directional_derivative_finite_difference
      <<"\ndirectional_derivative_step "<<covariance_fit.directional_derivative_step
      <<"\ndirectional_diagnostic_evaluations "<<covariance_fit.directional_diagnostic_evaluations
      <<"\ncandidate_training_force_residual_relative "<<training_force_residual
      <<"\nbaseline_training_force_residual_relative "<<covariance_baseline_force_residual<<"\ntheta_start "<<covariance_fit.theta_start.size();
      for(const double value:covariance_fit.theta_start)fitout<<' '<<value;fitout<<'\n';}
    else fitout<<" svd_sigma_min "<<svd_sigma_min<<" svd_sigma_max "<<svd_sigma_max<<" design_condition "<<design_condition
      <<"\nunconstrained_training_force_residual_relative "<<unconstrained_training_force_residual
      <<" force_fit_cost_definition relative_squared_residual_delta_from_unconstrained"
      <<"\nforce_fit_cost_increase "<<force_fit_cost_increase;
    fitout<<"\ncutoff_A "<<options.cutoff<<" epsilon "<<options.epsilon<<" temperature_K "<<options.temperature<<" sample_interval "<<options.sample_interval
      <<"\nsample_IBP_status "<<sample_statistics.ibp_band.status<<" estimate "<<sample_statistics.ibp_band.estimate<<" q99_radius "<<sample_statistics.ibp_band.radius<<" interval "<<sample_statistics.ibp_band.lower<<' '<<sample_statistics.ibp_band.upper
      <<"\nresponse_IBP_status "<<response.ibp_band.status<<" estimate "<<response.ibp<<" q99_radius "<<response.ibp_band.radius<<" interval "<<response.ibp_band.lower<<' '<<response.ibp_band.upper<<"\nIBP_acceptance_role DIAGNOSTIC_ONLY\n"
      <<"\ncg_recovery_witness_count "<<saved_recovery_witness_paths.size()<<"\ncg_recovery_replay_inputs_retained "<<(!saved_recovery_witness_paths.empty()?1:0)<<'\n';
    if(!saved_recovery_witness_paths.empty()){
      fitout<<"cg_recovery_sample_spool "<<std::quoted(spool_path)<<"\ncg_recovery_qraw "<<std::quoted(raw_path)<<'\n';
      for(const std::string& path:saved_recovery_witness_paths)fitout<<"cg_recovery_witness "<<std::quoted(path)<<'\n';
    }
    fitout<<"family shared_lab_frame_symmetric_typepair_edge_blocks\ncoordinate_convention "<<(options.internal_mass_com?"x=R-R0; raw_eval=R0+A*x; U0(x)=Taylor_raw(A*x)+Uadd(x); P=I-tt^T internal mass-COM coordinates":"raw Cartesian coordinates")<<"\ntransport_convention "<<(options.internal_mass_com?"Bt_internal=P(Bt_raw+Bt_add)P; ambient H tiles retained and interpreted by internal E^T Bt E":"raw native reference transport")<<"\nmechanical_policy "<<(options.internal_mass_com?"native_reference_transport;internal_mass_com_pullback_v1;finite_temperature_additive_v1":"native_reference_transport;finite_temperature_additive_v1")<<";beads="<<header.beads<<";derivative="<<rawver<<"\nraw_gradient_net_xyz "<<raw_net.net[0]<<' '<<raw_net.net[1]<<' '<<raw_net.net[2]<<"\nraw_gradient_net_norm "<<raw_net.net_norm<<"\ninternal_gradient_net_xyz "<<internal_net.net[0]<<' '<<internal_net.net[1]<<' '<<internal_net.net[2]<<"\ninternal_gradient_net_norm "<<internal_net.net_norm<<"\ninternal_gradient_net_limit "<<internal_net.limit<<"\ninternal_gradient_net_status "<<(internal_net.within_limit?"PASS":"FAIL")<<"\nremoved_mass_normal_norm "<<removed_normal_norm<<"\ncoordinate_scope_note finite-grid derivative errors remain; auxiliary reference definition only; pimd_fix_com removes PILE momentum and does not constrain the production Hamiltonian\nparameters "<<psize<<"\ntype_pairs "<<graph.type_pairs.size()<<"\ncutoff_A "<<options.cutoff<<"\ntraining_frames "<<train<<"\nvalidation_frames "<<validation<<"\nprobe_policy fixed_seeded_random16_type_local_up_to4_low_ritz4\nprobe_seed 0x243f6a8885a308d3\nprobe_count "<<response.probes<<"\nprobe_subspace_only 1\nforce_residual_normalization massweighted_projected_physical_force_norm\ntraining_force_residual_relative "<<training_force_residual<<"\nheldout_force_residual_relative "<<response.force_residual<<"\nresponse_ratio_min "<<response.response_ratio_min<<"\nresponse_ratio_max "<<response.response_ratio_max<<"\nprobe_response_error "<<response.response<<"\nforce_position_ibp_error "<<response.ibp<<"\nprobe_cg_max_true_relative_residual "<<response.cg<<"\n";write_response_stats(fitout,response);write_probe_origin_table(fitout,response.probe_sources,response.probe_origins);fitout<<"\nresponse_tolerance "<<options.response_tolerance<<"\nibp_tolerance "<<options.ibp_tolerance<<"\nepsilon_fit "<<options.epsilon<<"\nspectral_cuts "<<cut_rows.size()<<"\ncertificate stored_D_additive_shifted_frobenius_SPD_bound\ncertificate_epsilon_num "<<certificate_epsilon<<"\ncertificate_reconstruction_bound "<<certificate_eta<<"\ncertificate_lambda_min_lower_bound "<<0.5*certificate_epsilon-certificate_eta<<"\ncertificate_note SPD lower bound only; not an epsilon eigenvalue floor\n";fitout.close();if(!fitout)throw std::runtime_error("failed closing native fit diagnostics");}
  if(std::rename(fit_tmp.c_str(),fit_path.c_str())!=0)throw std::runtime_error("native fit could not publish its diagnostics");own_scratch.fit_tmp_owned=false;own_scratch.fit_owned=true;
  in.close();raw.close();if(std::remove(pack_path.c_str())!=0||(generate_raw&&saved_recovery_witness_paths.empty()&&std::remove(generated_raw_path.c_str())!=0))throw std::runtime_error("native fit succeeded but could not remove its own additive/qraw scratch files");own_scratch.pack_owned=false;own_scratch.raw_owned=false;own_scratch.output_owned=false;own_scratch.sidecar_owned=false;own_scratch.fit_owned=false;
  trace<<"REFERENCE_WRITTEN reference_class=finite_temperature_additive output "<<std::quoted(options.output_path)<<" stability "<<std::quoted(options.output_path+".stability")<<" fit_diagnostics "<<std::quoted(fit_path)<<'\n';
  trace<<"FIT_STATUS ACCEPTED_REFERENCE response=PASS full_spd=PASS sample_IBP_DIAGNOSTIC="<<sample_statistics.ibp_band.status
    <<" response_IBP_DIAGNOSTIC="<<response.ibp_band.status<<" recovery_witnesses="<<saved_recovery_witness_paths.size();
  if(!saved_recovery_witness_paths.empty())trace<<" replay_samples="<<std::quoted(spool_path)<<" replay_qraw="<<std::quoted(raw_path);
  trace<<'\n';for(const std::string& path:saved_recovery_witness_paths)trace<<"CG_RECOVERY_WITNESS "<<std::quoted(path)<<'\n';
  trace.flush();trace.close();if(!trace)throw std::runtime_error("failed finalizing native fit trace: "+trace_path);
  std::printf("    rpmd_ja native reference ACCEPTED_REFERENCE; response and full SPD passed; IBP remains diagnostic (sample=%s, validation=%s); stability sidecar and fit diagnostics published.\n",
    sample_statistics.ibp_band.status.c_str(),response.ibp_band.status.c_str());
  if(!saved_recovery_witness_paths.empty())std::printf("    rpmd_ja fit retained CG replay inputs: samples=%s qraw=%s; witnesses=%zu\n",spool_path.c_str(),raw_path.c_str(),saved_recovery_witness_paths.size());
  return !saved_recovery_witness_paths.empty();
}

static bool fit_rpmd_ja_native_reference_checked(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  const std::uint64_t frame_count,Atom& atom,Box& box,Force& force,const bool sample_beads_from_header)
{
  try{return fit_rpmd_ja_native_reference_impl(options,spool_path,frame_count,atom,box,force,sample_beads_from_header);}
  catch(const std::exception& e){
    if(!options.output_path.empty()){
      const std::string failure=options.output_path+".failure.txt";std::ifstream exists(failure);
      if(!exists.good()){std::ofstream out(failure,std::ios::out|std::ios::trunc);if(out)out<<"native finite-temperature reference fit failed\n"<<e.what()<<'\n';}
      const std::string raw_path=options.raw_input_path.empty()?options.output_path+".qraw":options.raw_input_path;std::ifstream trace_exists(options.output_path+".fit_trace.txt"),raw_exists(raw_path,std::ios::binary),qp_state_exists(options.output_path+".qp_state.txt",std::ios::binary),samples_exist(spool_path,std::ios::binary);const std::string trace_path=options.output_path+".fit_trace.txt",qp_state_path=options.output_path+".qp_state.txt";std::printf("    rpmd_ja fit FAIL: %s; failure=%s trace=%s qraw=%s qp_state=%s samples=%s\n",e.what(),failure.c_str(),trace_exists.good()?trace_path.c_str():"unavailable",raw_exists.good()?raw_path.c_str():"unavailable",qp_state_exists.good()?qp_state_path.c_str():"unavailable",samples_exist.good()?spool_path.c_str():"unavailable");
    }
    throw;
  }
}

bool fit_rpmd_ja_native_reference(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  const std::uint64_t frame_count,Atom& atom,Box& box,Force& force)
{
  return fit_rpmd_ja_native_reference_checked(options,spool_path,frame_count,atom,box,force,false);
}

void fit_rpmd_ja_native_reference_from_samples(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  Atom& atom,Box& box,Force& force)
{
  if(options.output_path.empty())throw std::invalid_argument("native fit_samples requires an output path");
  const std::string lock_path=options.output_path+".fit.lock";
  std::FILE* lock=std::fopen(lock_path.c_str(),"wx");
  if(!lock)throw std::runtime_error("rpmd_ja fit_samples could not acquire its output lock: "+lock_path);
  struct ReleaseLock{std::FILE* file;std::string path;~ReleaseLock(){std::fclose(file);std::remove(path.c_str());}} release{lock,lock_path};
  std::ifstream in(spool_path,std::ios::binary);if(!in)throw std::runtime_error("cannot open native rpmd_ja sample spool: "+spool_path);
  const SampleHeader header=read_header(in,0,atom,box,0.0,true);
  std::vector<double> position(static_cast<std::size_t>(3)*header.n),force_frame(position.size());double first_step=0.0,next_step=0.0;
  read_frame(in,position,force_frame,first_step);read_frame(in,position,force_frame,next_step);
  const double sample_interval=next_step-first_step;
  if(!std::isfinite(sample_interval)||!(sample_interval>0.0)||sample_interval>std::numeric_limits<int>::max()||
     std::abs(sample_interval-std::round(sample_interval))>1e-9)
    throw std::runtime_error("fit_samples could not infer a positive integer sample interval from GPJASMP1");
  RpmdJANativeFitOptions derived=options;derived.temperature=header.temperature;derived.sample_interval=static_cast<int>(std::llround(sample_interval));
  std::printf("rpmd_ja fit_samples: frames=%llu P=%d T=%.17g K sample_interval=%d; GPJASMP1 has no potential/PPPM fingerprint, so confirm its sampling configuration is unchanged\n",
    static_cast<unsigned long long>(header.frame_count),header.beads,header.temperature,derived.sample_interval);
  (void)fit_rpmd_ja_native_reference_checked(derived,spool_path,header.frame_count,atom,box,force,true);
}

void diagnose_rpmd_ja_native_fit_samples(const std::string& spool_path, const double fd_step,
                                         Atom& atom, Box& box, Force& force, const bool full)
{
  if (spool_path.empty() || !std::isfinite(fd_step) || fd_step <= 0.0)
    throw std::invalid_argument("native fit sample diagnostic requires a spool path and positive finite fd_step");
  std::ifstream in(spool_path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open native rpmd_ja sample spool: " + spool_path);
  const SampleHeader header = read_header(in, 0, atom, box, 0.0, true);
  if (header.frame_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("native fit sample diagnostic frame count exceeds indexing limits");
  const std::uint64_t train = 2 * header.frame_count / 3;
  const std::uint64_t validation = header.frame_count - train;
  std::vector<double> r0(static_cast<std::size_t>(3) * header.n, 0.0);
  const double sample_interval = read_training_r0(
    in, header, header.frame_count, std::numeric_limits<double>::quiet_NaN(), r0);
  validate_fit_branches(in, header, header.frame_count, box, r0);

  std::printf("rpmd_ja diagnose_samples: spool=%s frames=%llu train=%llu validation=%llu P=%d T=%.17g K sample_interval=%.17g fd_step=%.17g\n",
    spool_path.c_str(), static_cast<unsigned long long>(header.frame_count),
    static_cast<unsigned long long>(train), static_cast<unsigned long long>(validation),
    header.beads, header.temperature, sample_interval, fd_step);
  std::printf("  train R0 is the xyz-SoA arithmetic mean of centroid x from the first floor(2F/3) spool frames.\n");
  std::printf("  Evaluating with the currently loaded potential and PPPM configuration; GPJASMP1 has no potential fingerprint.\n");

  std::vector<double> saved(static_cast<std::size_t>(3) * header.n);
  if (atom.position_per_atom.size() != saved.size())
    throw std::runtime_error("native fit sample diagnostic requires an initialized centroid position array");
  atom.position_per_atom.copy_to_host(saved.data());
  struct RestorePosition {
    Atom& atom;
    std::vector<double>& saved;
    ~RestorePosition() { atom.position_per_atom.copy_from_host(saved.data()); }
  } restore{atom, saved};
  atom.position_per_atom.copy_from_host(r0.data());
  diagnose_rpmd_ja_qnep_reference(fd_step, atom, box, force, full);
}

void check_rpmd_ja_native_fit_samples(const std::string& spool_path,const std::string& report_path,Atom& atom,Box& box)
{
  if(spool_path.empty()||report_path.empty()||spool_path==report_path)
    throw std::invalid_argument("rpmd_ja check_samples requires distinct nonempty input and report paths");
  std::ifstream in(spool_path,std::ios::binary);if(!in)throw std::runtime_error("cannot open native rpmd_ja sample spool: "+spool_path);
  const SampleHeader header=read_header(in,0,atom,box,0.0,true);
  const std::uint64_t train=2*header.frame_count/3;std::vector<double> r0(static_cast<std::size_t>(3)*header.n,0.0);
  const double interval=read_training_r0(in,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  const auto sample_statistics_started=std::chrono::steady_clock::now();
  const FixedProbeStatistics stats=collect_fixed_probe_statistics(in,header,header.frame_count,train,r0,true,0.15,&box);
  const double sample_statistics_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-sample_statistics_started).count();
  write_text_exclusive(report_path,format_fixed_probe_report(stats,header,interval,train,0.15,sample_statistics_seconds),"sample statistics report");
  print_fixed_probe_summary(stats,header,interval,train,0.15,sample_statistics_seconds);
  std::printf("training-selected IBP noise channels:");for(const IBPNoiseChannel& channel:stats.fixed_channel_noise.channels)std::printf(" (%d,%d)",channel.i,channel.j);if(stats.fixed_channel_noise.channels.empty())std::printf(" none");std::printf("\n");
  std::printf("fixed-channel validation=%s variance_diagnostic=%s q99_radius=%.6g frames_used=%llu omitted=%llu reason=%s\n",
    stats.fixed_channel_noise.status.c_str(),stats.fixed_channel_noise.variance_status.c_str(),stats.fixed_channel_noise.band.radius,
    static_cast<unsigned long long>(stats.fixed_channel_noise.validation_frames_used),
    static_cast<unsigned long long>(stats.fixed_channel_noise.validation_frames_omitted),
    stats.fixed_channel_noise.band.rejection_reason.c_str());
  std::printf("sample statistics report=%s\n",report_path.c_str());
  std::printf("sampling evidence: %s; report=%s\n",stats.ibp_band.status.c_str(),report_path.c_str());
  std::printf("  fixed probes omit four low-frequency Ritz probes; this report cannot replace final response validation.\n");
}

void replay_rpmd_ja_native_cg(const std::string& witness_path,const std::string& report_path,const bool internal_mass_com,
  Atom& atom,Box& box,Force& force)
{
  if(witness_path.empty()||report_path.empty()||witness_path==report_path)
    throw std::invalid_argument("rpmd_ja replay_cg requires distinct nonempty witness and report paths");
  std::ifstream existing_report(report_path,std::ios::binary);
  if(existing_report.good())throw std::runtime_error("rpmd_ja replay_cg report already exists: "+report_path);
  const auto paths=read_cg_witness_paths(witness_path);
  std::ifstream spool(paths.second,std::ios::binary);
  if(!spool)throw std::runtime_error("cannot open CG witness sample spool: "+paths.second);
  const SampleHeader header=read_header(spool,0,atom,box,0.0,true,true);
  if(header.frame_count>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    throw std::runtime_error("CG witness sample frame count exceeds supported indexing limits");
  const std::uint64_t train=2*header.frame_count/3;
  std::vector<double> r0(static_cast<std::size_t>(3)*header.n,0.0);
  const double interval=read_training_r0(spool,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  validate_fit_branches(spool,header,header.frame_count,box,r0);
  const int n=header.n,d=3*n;
  const CGWitnessSnapshot snapshot=read_cg_witness(witness_path,d,0);
  if(snapshot.sample_path!=paths.second||snapshot.qraw_path!=paths.first||snapshot.sample_frames!=header.frame_count||
     snapshot.training_frames!=train||std::abs(snapshot.temperature-header.temperature)>1e-10*header.temperature||
     std::abs(static_cast<double>(snapshot.sample_interval)-interval)>1e-9||snapshot.internal_mass_com!=(internal_mass_com?1:0))
    throw std::runtime_error("CG witness sample metadata or COM convention does not match the supplied sample/configuration");
  const Graph candidate_graph=make_graph(r0,header.types,box,snapshot.cutoff);
  const int parameter_count=6*static_cast<int>(candidate_graph.type_pairs.size());
  if(snapshot.theta.size()!=static_cast<std::size_t>(parameter_count))
    throw std::runtime_error("CG witness theta dimension does not match the graph rebuilt from sample R0 and cutoff");
  RpmdJANativeFitOptions options;options.temperature=snapshot.temperature;options.cutoff=snapshot.cutoff;
  options.epsilon=snapshot.epsilon;options.fd_step=snapshot.fd_step;options.sample_interval=snapshot.sample_interval;
  options.internal_mass_com=internal_mass_com;
  std::ifstream raw(paths.first,std::ios::binary);
  if(!raw)throw std::runtime_error("cannot open CG witness qraw source: "+paths.first);
  const QrawIdentity identity=validate_qraw_identity(raw,options,header,r0,box,force);
  if(snapshot.qraw_model!=identity.model||snapshot.qraw_config!=identity.config||snapshot.active_model!=identity.model||
     snapshot.active_config!=identity.config||snapshot.qraw_charge!=identity.charge||snapshot.qraw_kspace!=identity.kspace||
     std::abs(snapshot.qraw_mesh-identity.mesh)>1e-12*std::max(1.0,std::abs(identity.mesh))||
     std::abs(snapshot.active_mesh-identity.active_mesh)>1e-12*std::max(1.0,std::abs(identity.active_mesh)))
    throw std::runtime_error("CG witness qraw/model/PPPM identity does not match the validated source files and active potential");
  std::vector<double> sqrt_mass(static_cast<std::size_t>(d)),sqrt_atom(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i){if(!std::isfinite(header.masses[i])||header.masses[i]<=0.0)throw std::runtime_error("CG replay requires positive finite sample masses");sqrt_atom[i]=std::sqrt(header.masses[i]);}
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)sqrt_mass[static_cast<std::size_t>(axis)*n+i]=sqrt_atom[i];
  std::vector<double> projected_rhs=snapshot.rhs_input;project_translation(projected_rhs,sqrt_atom,n);
  double projected_rhs_diff=0.0,saved_x_translation=0.0;
  for(int i=0;i<d;++i){projected_rhs_diff=std::hypot(projected_rhs_diff,projected_rhs[i]-snapshot.rhs_projected[i]);}
  std::vector<double> projected_x=snapshot.solution;project_translation(projected_x,sqrt_atom,n);
  for(int i=0;i<d;++i)saved_x_translation=std::hypot(saved_x_translation,projected_x[i]-snapshot.solution[i]);
  const double rhs_norm=std::sqrt(compensated_dot(projected_rhs,projected_rhs));
  if(projected_rhs_diff>1e-12*std::max(1.0,rhs_norm)||saved_x_translation>1e-10*std::max(1.0,std::sqrt(compensated_dot(snapshot.solution,snapshot.solution))))
    throw std::runtime_error("CG witness right-hand side or saved solution is inconsistent with the projected coordinate convention");
  DeviceBaseline baseline;baseline.initialize(raw,identity.k_offset,d,n,header.masses,sqrt_mass);raw.close();
  const CGResidualComparison saved=compare_cg_true_residuals(baseline,candidate_graph,snapshot.rhs_input,snapshot.solution,
    sqrt_mass,sqrt_atom,snapshot.theta,n);
  const auto refinement_started=std::chrono::steady_clock::now();
  const CGReplayRefinement refinement=refine_cg_replay_solution(baseline,candidate_graph,snapshot.rhs_input,snapshot.solution,
    sqrt_mass,sqrt_atom,snapshot.theta,n,snapshot.probe,saved);
  const double refinement_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-refinement_started).count();
  const std::uint64_t actions_before=baseline.dense_matrix_actions;
  CGResult replay;std::string replay_status,replay_error;double replay_seconds=0.0;
  const auto replay_started=std::chrono::steady_clock::now();
  try{replay=solve_projected_cg(baseline,candidate_graph,snapshot.rhs_input,sqrt_mass,sqrt_atom,snapshot.theta,n,snapshot.probe,snapshot.epsilon);
    replay_status=replay.witness.classification.empty()?"CG_PASS":replay.witness.classification;}
  catch(const std::exception& error){replay_status="CG_EXCEPTION";replay_error=error.what();}
  replay_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-replay_started).count();
  const std::uint64_t replay_actions=baseline.dense_matrix_actions-actions_before;
  const std::string replay_iterations=replay_status=="CG_EXCEPTION"?"NOT_AVAILABLE":std::to_string(replay.iterations);
  const std::string replay_restarts=replay_status=="CG_EXCEPTION"?"NOT_AVAILABLE":std::to_string(replay.residual_restarts);
  CGProbeRecoveryDiagnostic production_recovery;bool production_recovery_attempted=false,production_recovery_returned=false;
  bool production_recovery_gate_pass=false,production_recovered_solution_finite=false,production_original_witness_preserved=false;
  std::string production_recovery_status="NOT_ATTEMPTED",production_recovery_error;
  double production_recovery_seconds=0.0,production_solution_delta_norm=std::numeric_limits<double>::quiet_NaN();
  std::uint64_t production_recovery_actions=0;
  if(replay_status=="TRUE_RESIDUAL_FAILURE"){
    production_recovery_attempted=true;const std::uint64_t actions_before_recovery=baseline.dense_matrix_actions;
    const auto production_recovery_started=std::chrono::steady_clock::now();
    try{production_recovery_returned=recover_cg_true_residual_failure(baseline,candidate_graph,snapshot.rhs_input,replay,
        sqrt_mass,sqrt_atom,snapshot.theta,n,production_recovery,snapshot.probe);
      production_recovery_status=production_recovery_returned?std::string("RECOVERED"):std::string("REJECTED_")+production_recovery.refinement_status;}
    catch(const std::exception& error){production_recovery_status="RECOVERY_EXCEPTION";production_recovery_error=error.what();}
    production_recovery_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-production_recovery_started).count();
    production_recovery_actions=baseline.dense_matrix_actions-actions_before_recovery;
    production_recovered_solution_finite=replay.x.size()==static_cast<std::size_t>(d)&&
      std::all_of(replay.x.begin(),replay.x.end(),[](double value){return std::isfinite(value);});
    production_original_witness_preserved=replay.witness.classification=="TRUE_RESIDUAL_FAILURE"&&
      replay.witness.classification==production_recovery.original_witness.classification&&
      replay.witness.solution==production_recovery.original_witness.solution;
    if(production_original_witness_preserved&&replay.x.size()==production_recovery.original_witness.solution.size()){
      production_solution_delta_norm=0.0;for(std::size_t i=0;i<replay.x.size();++i)
        production_solution_delta_norm=std::hypot(production_solution_delta_norm,replay.x[i]-production_recovery.original_witness.solution[i]);}
    production_recovery_gate_pass=production_recovery_returned&&production_recovered_solution_finite&&production_original_witness_preserved&&
      std::isfinite(production_recovery.final_fast_residual)&&std::isfinite(production_recovery.final_compensated_residual)&&
      production_recovery.final_compensated_residual<=1e-8;
    if(production_recovery_status!="RECOVERY_EXCEPTION"&&production_recovery_returned&&!production_recovery_gate_pass)
      production_recovery_status="RECOVERY_RESULT_INVALID";
  }
  std::ostringstream report;const auto write_diagnostic_value=[&](const double value){if(std::isfinite(value))report<<value;else report<<"NOT_COMPUTED";};
  report<<std::setprecision(17)
    <<"status NOT_ACCEPTED_REFERENCE\nreplay_source "<<std::quoted(witness_path)<<"\nqraw_source "<<std::quoted(paths.first)
    <<"\nsample_spool "<<std::quoted(paths.second)<<"\nframes "<<header.frame_count<<"\ntraining_frames "<<train
    <<"\nprobe "<<snapshot.probe<<"\ncutoff_A "<<snapshot.cutoff<<"\ntemperature_K "<<snapshot.temperature
    <<"\nfd_step "<<snapshot.fd_step<<"\nepsilon "<<snapshot.epsilon<<"\ninternal_mass_com "<<snapshot.internal_mass_com
    <<"\nqraw_model_fingerprint "<<identity.model<<"\nqraw_config_fingerprint "<<identity.config
    <<"\npppm_mesh_spacing "<<identity.mesh<<"\nprojected_rhs_consistency_difference_norm "<<projected_rhs_diff
    <<"\nsaved_x_translation_removed_norm "<<saved_x_translation
    <<"\nproduction_recovery_attempted "<<(production_recovery_attempted?1:0)
    <<"\nproduction_recovery_status "<<production_recovery_status
    <<"\nproduction_verification_method "<<(production_recovery_attempted?production_recovery.verification_method:"NOT_ATTEMPTED")
    <<"\nproduction_initial_fast_residual ";write_diagnostic_value(production_recovery.initial_fast_residual);
  report<<"\nproduction_initial_compensated_residual ";write_diagnostic_value(production_recovery.initial_compensated_residual);
  report<<"\nproduction_final_fast_residual ";write_diagnostic_value(production_recovery.final_fast_residual);
  report<<"\nproduction_final_compensated_residual ";write_diagnostic_value(production_recovery.final_compensated_residual);
  report<<"\nproduction_refinement_status "<<(production_recovery_attempted?production_recovery.refinement_status:"NOT_ATTEMPTED")
    <<"\nproduction_refinement_rounds "<<(production_recovery_attempted?std::to_string(production_recovery.refinement_rounds):"NOT_ATTEMPTED")
    <<"\nproduction_verification_actions "<<(production_recovery_attempted?std::to_string(production_recovery.verification_matrix_actions):"NOT_ATTEMPTED")
    <<"\nproduction_refinement_actions "<<(production_recovery_attempted?std::to_string(production_recovery.refinement_matrix_actions):"NOT_ATTEMPTED")
    <<"\nproduction_recovery_actions "<<(production_recovery_attempted?std::to_string(production_recovery_actions):"NOT_ATTEMPTED")
    <<"\nproduction_recovery_seconds ";
  if(production_recovery_attempted)write_diagnostic_value(production_recovery_seconds);else report<<"NOT_ATTEMPTED";
  report<<"\nproduction_recovered_solution_finite "<<(production_recovery_attempted?(production_recovered_solution_finite?"1":"0"):"NOT_ATTEMPTED")
    <<"\nproduction_recovery_gate_pass "<<(production_recovery_attempted?(production_recovery_gate_pass?"1":"0"):"NOT_ATTEMPTED")
    <<"\nproduction_original_witness_preserved "<<(production_recovery_attempted?(production_original_witness_preserved?"1":"0"):"NOT_ATTEMPTED")
    <<"\nproduction_solution_delta_norm ";write_diagnostic_value(production_solution_delta_norm);
  report<<"\nproduction_recovery_error "<<std::quoted(production_recovery_error)
    <<"\nfast_true_relative_residual "<<saved.fast_relative
    <<"\ncompensated_true_relative_residual "<<saved.compensated_relative
    <<"\nresidual_vector_difference_norm "<<saved.difference_norm
    <<"\ncorrection_refinement_status "<<refinement.status
    <<"\ncorrection_refinement_rounds "<<refinement.rounds.size()
    <<"\ncorrection_relative_target 0.01"
    <<"\ncorrection_refinement_round_limit 3"
    <<"\ncorrection_matrix_action_limit 4096"
    <<"\ncorrection_dense_matrix_actions "<<refinement.dense_matrix_actions
    <<"\ncorrection_seconds "<<refinement_seconds
    <<"\nrefined_solution_fast_true_relative_residual "<<refinement.residuals.fast_relative
    <<"\nrefined_solution_compensated_true_relative_residual "<<refinement.residuals.compensated_relative
    <<"\nrefined_solution_vector_difference_norm "<<refinement.residuals.difference_norm
    <<"\nreplay_cg_status "<<replay_status<<"\nreplay_cg_iterations "<<replay_iterations
    <<"\nreplay_cg_residual_restarts "<<replay_restarts
    <<"\nreplay_cg_recursive_relative_residual ";
  if(std::isfinite(replay.recursive_relative_residual))report<<replay.recursive_relative_residual;else report<<"NOT_COMPUTED";
  report
    <<"\nreplay_cg_dense_matrix_actions "<<replay_actions<<"\nreplay_cg_seconds "<<replay_seconds
    <<"\nreplay_cg_true_residual_status "<<(replay.witness.true_residual_computed?"COMPUTED":(replay_status=="CG_PASS"?"COMPUTED":"NOT_COMPUTED"))
    <<"\nreplay_cg_true_relative_residual ";
  if(replay.witness.true_residual_computed)report<<replay.witness.relative_residual;
  else if(replay_status=="CG_PASS")report<<replay.relative_residual;
  else report<<"NOT_COMPUTED";
  report<<"\nreplay_cg_error "<<std::quoted(replay_error)<<"\ncorrection_round_records "<<refinement.rounds.size()<<'\n';
  for(std::size_t i=0;i<refinement.rounds.size();++i)write_cg_replay_refinement_round(report,i+1,refinement.rounds[i]);
  report<<"refined_solution_x_xyz_soa "<<refinement.solution.size();for(double value:refinement.solution)report<<' '<<value;
  report<<"\ntheta "<<snapshot.theta.size();
  for(double value:snapshot.theta)report<<' '<<value;
  report<<'\n';
  write_text_exclusive(report_path,report.str(),"CG replay report");
  std::printf("rpmd_ja replay_cg: saved-x fast=%.9g compensated=%.9g; correction=%s fast=%.9g compensated=%.9g actions=%llu; zero-start=%s iterations=%s actions=%llu; report=%s; NOT_ACCEPTED_REFERENCE\n",
    saved.fast_relative,saved.compensated_relative,refinement.status.c_str(),refinement.residuals.fast_relative,refinement.residuals.compensated_relative,
    static_cast<unsigned long long>(refinement.dense_matrix_actions),replay_status.c_str(),replay_iterations.c_str(),
    static_cast<unsigned long long>(replay_actions),report_path.c_str());
  if(production_recovery_attempted)std::printf("    production recovery=%s compensated_residual=%.9g gate_pass=%d original_witness_preserved=%d actions=%llu seconds=%.3f\n",
    production_recovery_status.c_str(),production_recovery.final_compensated_residual,production_recovery_gate_pass?1:0,
    production_original_witness_preserved?1:0,static_cast<unsigned long long>(production_recovery_actions),production_recovery_seconds);
  else std::printf("    production recovery=NOT_ATTEMPTED (zero-start status %s)\n",replay_status.c_str());
}
#else
#include <stdexcept>
bool fit_rpmd_ja_native_reference(const RpmdJANativeFitOptions&, const std::string&,
  const std::uint64_t, Atom&, Box&, Force&)
{
  throw std::runtime_error("native rpmd_ja reference fitting requires CUDA");
}
void fit_rpmd_ja_native_reference_from_samples(const RpmdJANativeFitOptions&, const std::string&,
  Atom&, Box&, Force&)
{
  throw std::runtime_error("native rpmd_ja reference fitting requires CUDA");
}
void diagnose_rpmd_ja_native_fit_samples(const std::string&, double, Atom&, Box&, Force&, bool)
{
  throw std::runtime_error("native rpmd_ja sample diagnostics require CUDA");
}
void check_rpmd_ja_native_fit_samples(const std::string&,const std::string&,Atom&,Box&)
{
  throw std::runtime_error("native rpmd_ja sample checks require CUDA");
}
void replay_rpmd_ja_native_cg(const std::string&,const std::string&,bool,Atom&,Box&,Force&)
{
  throw std::runtime_error("native rpmd_ja CG replay requires CUDA");
}
#endif
