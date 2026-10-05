#include "rpmd_ja_native_fit.cuh"

#ifndef USE_HIP
#include "rpmd_ja_qnep_prepare.cuh"
#include "rpmd_ja_reference.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "force/force.cuh"
#include "utilities/common.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <set>
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
                         const double temperature, const bool diagnostic = false)
{
  char magic[8]; in.read(magic, sizeof(magic));
  const auto version = read<std::uint32_t>(in, "version");
  const auto endian = read<std::uint32_t>(in, "byte order");
  SampleHeader h;
  h.n = read<std::int32_t>(in, "atom count");
  h.beads = read<std::int32_t>(in, "bead count");
  h.temperature = read<double>(in, "temperature");
  if (h.n != atom.number_of_atoms || h.n < 2 || h.n > std::numeric_limits<int>::max()/3 ||
      (diagnostic ? h.beads < 1 : h.beads != atom.number_of_beads))
    throw std::runtime_error("native fit spool has invalid atom or bead count");
  read_array(in, h.cell, 9, "cell");
  h.masses.resize(h.n); h.types.resize(h.n);
  read_array(in, h.masses.data(), h.masses.size(), "masses");
  read_array(in, h.types.data(), h.types.size(), "types");
  h.frames = in.tellg();
  if (std::memcmp(magic, sample_magic, sizeof(magic)) || version != 1 || endian != native_endian || h.n != atom.number_of_atoms ||
      h.n < 2 || (diagnostic ? h.beads < 1 : h.beads != atom.number_of_beads) || !std::isfinite(h.temperature) || h.temperature <= 0.0 ||
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

void validate_fit_branches(std::istream& in, const SampleHeader& header, const std::uint64_t frame_count,
                           const Box& box, const std::vector<double>& r0)
{
  std::vector<double> x(r0.size()), f(r0.size());
  in.clear(); in.seekg(header.frames);
  for (std::uint64_t frame = 0; frame < frame_count; ++frame) {
    double step = 0.0;
    read_frame(in, x, f, step);
    for (int i = 0; i < header.n; ++i) {
      const double delta[3] = {x[i] - r0[i], x[header.n + i] - r0[header.n + i],
                               x[2 * header.n + i] - r0[2 * header.n + i]};
      for (int a = 0; a < 3; ++a) {
        double fractional = 0.0;
        for (int b = 0; b < 3; ++b) fractional += box.cpu_h[9 + 3 * a + b] * delta[b];
        if (!std::isfinite(fractional) || std::abs(fractional) >= 0.45)
          throw std::runtime_error("native fit sample centroid exceeds the fixed-reference branch limit 0.45");
      }
    }
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

struct DeviceBaseline
{
  double* k=nullptr; double* q=nullptr; double* out=nullptr; double* t=nullptr; double* kt=nullptr;double* g=nullptr;int* error=nullptr;
  cublasHandle_t blas=nullptr; int d=0;
  void release(){if(blas){cublasDestroy(blas);blas=nullptr;}if(k){cudaFree(k);k=nullptr;}if(q){cudaFree(q);q=nullptr;}if(out){cudaFree(out);out=nullptr;}if(t){cudaFree(t);t=nullptr;}if(kt){cudaFree(kt);kt=nullptr;}if(g){cudaFree(g);g=nullptr;}if(error){cudaFree(error);error=nullptr;}d=0;}
  ~DeviceBaseline(){release();}
  void initialize(std::istream& raw,const std::streamoff offset,const int dimension,const int n,
                  const std::vector<double>& masses,const std::vector<double>& sqrt_mass)
  {
    d=dimension;std::size_t free_bytes=0,total_bytes=0;
    if(cudaMemGetInfo(&free_bytes,&total_bytes)!=cudaSuccess)throw std::runtime_error("native fit cudaMemGetInfo failed");
    const std::size_t bytes=static_cast<std::size_t>(d)*d*sizeof(double);
    if(bytes>free_bytes||free_bytes-bytes<256ULL*1024*1024)throw std::runtime_error("native fit raw Hessian GPU preflight leaves less than 256 MiB safety margin");
    if(cudaMalloc(reinterpret_cast<void**>(&k),bytes)!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&q),static_cast<std::size_t>(d)*sizeof(double))!=cudaSuccess||
       cudaMalloc(reinterpret_cast<void**>(&out),static_cast<std::size_t>(d)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&t),static_cast<std::size_t>(3)*d*sizeof(double))!=cudaSuccess||
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
    const double one=1.0,zero=0.0;
    if(cudaMemcpy(q,input.data(),static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||
       cublasDgemv(blas,CUBLAS_OP_T,d,d,&one,k,d,q,1,&zero,out,1)!=CUBLAS_STATUS_SUCCESS||
       cudaMemcpy(result.data(),out,static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)
      throw std::runtime_error("native fit Hessian matvec failed");
  }
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

std::vector<double> unconstrained_eta(const SmallSVD& svd){return svd.eta;}

std::vector<double> solve_cut_qp(const SmallSVD& svd,const std::vector<std::vector<double>>& rows,const std::vector<double>& rhs)
{
  std::vector<std::vector<double>> a(rows.size(),std::vector<double>(svd.p));
  std::vector<double> b=rhs;
  for(std::size_t c=0;c<rows.size();++c){for(int k=0;k<svd.p;++k){for(int j=0;j<svd.p;++j)a[c][k]+=rows[c][j]*svd.vt[static_cast<std::size_t>(k)+static_cast<std::size_t>(j)*svd.p]/svd.singular[k];}if(!std::all_of(a[c].begin(),a[c].end(),[](double x){return std::isfinite(x);})||!std::isfinite(b[c]))throw std::runtime_error("native fit stability cut is non-finite");for(int k=0;k<svd.p;++k)b[c]-=a[c][k]*svd.eta[k];}
  std::vector<double> gram(rows.size()*rows.size()),lambda(rows.size(),0.0);for(std::size_t i=0;i<rows.size();++i)for(std::size_t j=0;j<rows.size();++j){gram[i*rows.size()+j]=std::inner_product(a[i].begin(),a[i].end(),a[j].begin(),0.0);if(!std::isfinite(gram[i*rows.size()+j]))throw std::runtime_error("native fit stability QP Gram matrix is non-finite");}
  std::vector<double> diag(rows.size());for(std::size_t i=0;i<rows.size();++i){diag[i]=gram[i*rows.size()+i];if(!(diag[i]>1e-24)||!std::isfinite(diag[i]))throw std::runtime_error("native fit stability cut is singular or non-finite");}
  for(int it=0;it<200000;++it){double change=0.0;for(std::size_t i=0;i<rows.size();++i){double residual=b[i];for(std::size_t j=0;j<rows.size();++j)residual-=gram[i*rows.size()+j]*lambda[j];const double next=std::max(0.0,lambda[i]+residual/diag[i]);if(!std::isfinite(next))throw std::runtime_error("native fit stability QP iterate is non-finite");change=std::max(change,std::abs(next-lambda[i]));lambda[i]=next;}if(change<1e-11)break;if(it==199999)throw std::runtime_error("native fit stability QP failed to converge");}
  std::vector<double> xi=svd.eta;for(std::size_t i=0;i<rows.size();++i)for(int k=0;k<svd.p;++k)xi[k]+=a[i][k]*lambda[i];
  double max_primal=0.0,max_comp=0.0,max_dual=0.0;const std::vector<double> theta=theta_from_eta(svd,xi);
  for(std::size_t i=0;i<rows.size();++i){double shifted=0.0;for(int k=0;k<svd.p;++k)shifted+=a[i][k]*(xi[k]-svd.eta[k]);const double slack=shifted-b[i];const double primal=std::inner_product(rows[i].begin(),rows[i].end(),theta.begin(),0.0)-rhs[i];if(!std::isfinite(slack)||!std::isfinite(primal)||!std::isfinite(lambda[i]))throw std::runtime_error("native fit stability QP produced a non-finite certificate");max_primal=std::max(max_primal,std::max(0.0,-primal));max_comp=std::max(max_comp,std::abs(lambda[i]*slack));}
  for(int k=0;k<svd.p;++k){double stationarity=xi[k]-svd.eta[k];for(std::size_t i=0;i<rows.size();++i)stationarity-=a[i][k]*lambda[i];max_dual=std::max(max_dual,std::abs(stationarity));}if(!std::isfinite(max_dual)||max_primal>1e-8||max_comp>1e-7||max_dual>1e-9)throw std::runtime_error("native fit stability QP failed primal, dual, or complementary-slackness check");
  return xi;
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

void apply_group(const Graph& graph, const std::vector<double>& q, const int n, const int parameters,
                 const double* theta, double* result)
{
  std::fill(result,result+3*n,0.0);
  for (const auto& e : graph.edges) {
    const std::size_t p = static_cast<std::size_t>(6*e.group);
    const double dx[3] = {q[e.j]-q[e.i],q[n+e.j]-q[n+e.i],q[2*n+e.j]-q[2*n+e.i]};
    const double c[6] = {theta[p],theta[p+1],theta[p+2],theta[p+3],theta[p+4],theta[p+5]};
    const double y[3] = {c[0]*dx[0]+c[1]*dx[1]+c[2]*dx[2],
                         c[1]*dx[0]+c[3]*dx[1]+c[4]*dx[2],
                         c[2]*dx[0]+c[4]*dx[1]+c[5]*dx[2]};
    for(int a=0;a<3;++a){result[a*n+e.i]-=y[a];result[a*n+e.j]+=y[a];}
  }
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
SmallEigen eigen_small(cusolverDnHandle_t solver,const std::vector<double>& rowmajor,const int n)
{
  if(!std::all_of(rowmajor.begin(),rowmajor.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit eigensolver input is non-finite");
  SmallEigen out;out.n=n;std::vector<double>a(rowmajor.size());for(int i=0;i<n;++i)for(int j=0;j<n;++j)a[i+static_cast<std::size_t>(j)*n]=rowmajor[static_cast<std::size_t>(i)*n+j];
  double *da=nullptr,*dw=nullptr,*work=nullptr;int *info=nullptr,lwork=0;auto cleanup=[&](){if(da)cudaFree(da);if(dw)cudaFree(dw);if(work)cudaFree(work);if(info)cudaFree(info);};
  try{if(cudaMalloc(reinterpret_cast<void**>(&da),a.size()*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&dw),static_cast<std::size_t>(n)*sizeof(double))!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&info),sizeof(int))!=cudaSuccess)throw std::runtime_error("native fit small-eigen allocation failed");
    if(cusolverDnDsyevd_bufferSize(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,da,n,dw,&lwork)!=CUSOLVER_STATUS_SUCCESS||lwork<=0||cudaMalloc(reinterpret_cast<void**>(&work),static_cast<std::size_t>(lwork)*sizeof(double))!=cudaSuccess)throw std::runtime_error("native fit small-eigen workspace failed");
    if(cudaMemcpy(da,a.data(),a.size()*sizeof(double),cudaMemcpyHostToDevice)!=cudaSuccess||cusolverDnDsyevd(solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,da,n,dw,work,lwork,info)!=CUSOLVER_STATUS_SUCCESS)throw std::runtime_error("native fit small symmetric eigensolve failed");
    int host_info=0;if(cudaMemcpy(&host_info,info,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess||host_info)throw std::runtime_error("native fit small symmetric eigensolve did not converge");
     out.values.resize(n);out.vectors.resize(a.size());if(cudaMemcpy(out.values.data(),dw,static_cast<std::size_t>(n)*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess||cudaMemcpy(out.vectors.data(),da,a.size()*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess)throw std::runtime_error("native fit small eigen readback failed");if(!std::all_of(out.values.begin(),out.values.end(),[](double v){return std::isfinite(v);})||!std::all_of(out.vectors.begin(),out.vectors.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit small eigensolver returned non-finite values");
  }catch(...){cleanup();throw;}cleanup();return out;
}

struct RitzMode{double value=0.0,residual=0.0;std::vector<double> vector;};
std::vector<RitzMode> lanczos_low_modes(cusolverDnHandle_t solver,DeviceBaseline& baseline,const Graph& graph,
  const std::vector<double>& theta,const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const int n,const int max_steps,const int wanted)
{
  const int d=3*n;const int steps=std::min(max_steps,d-3);if(steps<=0)throw std::runtime_error("native fit Lanczos has no internal dimensions");
  std::vector<std::vector<double>> basis;basis.reserve(steps);std::vector<double> v(d),w(d),raw(d),add(d);
  for(int i=0;i<d;++i)v[i]=std::sin((i+1)*1.6180339887498948)+std::cos((i+1)*0.7548776662466927);
  project_translation(v,sqrt_atom,n);
  double norm=std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));if(!std::isfinite(norm)||!(norm>0.0))throw std::runtime_error("native fit Lanczos initial vector vanished or is non-finite");for(double&x:v)x/=norm;
  std::vector<double> alpha,beta;double beta_prev=0.0;std::vector<double> previous(d,0.0);
  for(int k=0;k<steps;++k){basis.push_back(v);baseline.apply(v,raw);apply_additive(graph,v,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i)w[i]=raw[i]+add[i];
    const double av_norm=std::sqrt(std::inner_product(w.begin(),w.end(),w.begin(),0.0));if(!std::isfinite(av_norm))throw std::runtime_error("native fit Lanczos operator norm is non-finite");
    if(k)for(int i=0;i<d;++i)w[i]-=beta_prev*previous[i];const double a=std::inner_product(v.begin(),v.end(),w.begin(),0.0);if(!std::isfinite(a))throw std::runtime_error("native fit Lanczos diagonal is non-finite");alpha.push_back(a);for(int i=0;i<d;++i)w[i]-=a*v[i];
    project_translation(w,sqrt_atom,n);
    for(int pass=0;pass<2;++pass)for(const auto&u:basis){const double dot=std::inner_product(u.begin(),u.end(),w.begin(),0.0);for(int i=0;i<d;++i)w[i]-=dot*u[i];}
    project_translation(w,sqrt_atom,n);
    const double b=std::sqrt(std::inner_product(w.begin(),w.end(),w.begin(),0.0));if(!std::isfinite(b))throw std::runtime_error("native fit Lanczos residual norm is non-finite");beta.push_back(b);
    const double operator_scale=std::max({av_norm,std::abs(a),std::abs(beta_prev)});
    const double breakdown=64.0*std::numeric_limits<double>::epsilon()*operator_scale;
    if(b<=breakdown||k==steps-1)break;previous=std::move(v);v.resize(d);for(int i=0;i<d;++i)v[i]=w[i]/b;beta_prev=b;
  }
  const int m=static_cast<int>(alpha.size());std::vector<double> t(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i){t[static_cast<std::size_t>(i)*m+i]=alpha[i];if(i+1<m)t[static_cast<std::size_t>(i)*m+i+1]=t[static_cast<std::size_t>(i+1)*m+i]=beta[i];}
  const SmallEigen eig=eigen_small(solver,t,m);std::vector<RitzMode> out;const double tail=beta[m-1];
  for(int mode=0;mode<std::min(wanted,m);++mode){RitzMode r;r.value=eig.values[mode];r.vector.assign(d,0.0);
    for(int k=0;k<m;++k){const double c=eig.vectors[static_cast<std::size_t>(k)+static_cast<std::size_t>(mode)*m];for(int i=0;i<d;++i)r.vector[i]+=c*basis[k][i];}
    project_translation(r.vector,sqrt_atom,n);const double rnorm=std::sqrt(std::inner_product(r.vector.begin(),r.vector.end(),r.vector.begin(),0.0));if(!std::isfinite(rnorm)||!(rnorm>0.0))throw std::runtime_error("native fit Lanczos Ritz vector is non-finite or translation-only");for(double&x:r.vector)x/=rnorm;
    r.residual=tail*std::abs(eig.vectors[static_cast<std::size_t>(m-1)+static_cast<std::size_t>(mode)*m])/rnorm;out.push_back(std::move(r));}
  return out;
}

std::vector<double> spectral_row(const Graph& graph,const std::vector<double>& v,const std::vector<double>& sqrt_mass,
  const int n,const int p)
{
  std::vector<double> design(static_cast<std::size_t>(3*n)*p);make_design(graph,v,sqrt_mass,n,p,design);std::vector<double> row(p,0.0);
  for(int c=0;c<p;++c)for(int i=0;i<3*n;++i)row[c]-=v[i]*design[static_cast<std::size_t>(c)*3*n+i];return row;
}

void apply_total(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& q,const std::vector<double>& sqrt_mass,
  const std::vector<double>& sqrt_atom,const std::vector<double>& theta,const int n,std::vector<double>& result)
{
  std::vector<double> base(q.size()),add(q.size());baseline.apply(q,base);apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,add);
  for(std::size_t i=0;i<result.size();++i)result[i]=base[i]+add[i];project_translation(result,sqrt_atom,n);
}

struct CGResult{std::vector<double> x;double relative_residual=0.0;};
CGResult solve_projected_cg(DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& rhs,
  const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const std::vector<double>& theta,const int n)
{
  CGResult out;out.x.assign(rhs.size(),0.0);std::vector<double> r=rhs,p=rhs,ap(rhs.size());project_translation(r,sqrt_atom,n);p=r;
  double rr=std::inner_product(r.begin(),r.end(),r.begin(),0.0);const double initial=std::sqrt(rr);if(!(initial>0.0))throw std::runtime_error("native fit response probe has zero norm");
  for(int it=0;it<20000;++it){apply_total(baseline,graph,p,sqrt_mass,sqrt_atom,theta,n,ap);const double pap=std::inner_product(p.begin(),p.end(),ap.begin(),0.0);if(!(pap>0.0)||!std::isfinite(pap))throw std::runtime_error("native fit response CG found a nonpositive operator direction");
    const double alpha=rr/pap;for(std::size_t i=0;i<r.size();++i){out.x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
     const double next=std::inner_product(r.begin(),r.end(),r.begin(),0.0);out.relative_residual=std::sqrt(next)/initial;if(out.relative_residual<=1e-8){apply_total(baseline,graph,out.x,sqrt_mass,sqrt_atom,theta,n,ap);for(std::size_t i=0;i<r.size();++i)r[i]=rhs[i]-ap[i];project_translation(r,sqrt_atom,n);out.relative_residual=std::sqrt(std::inner_product(r.begin(),r.end(),r.begin(),0.0))/initial;if(!std::isfinite(out.relative_residual)||out.relative_residual>1e-8)throw std::runtime_error("native fit response CG true residual exceeds tolerance");return out;}
    const double beta=next/rr;for(std::size_t i=0;i<p.size();++i)p[i]=r[i]+beta*p[i];rr=next;
  }
  throw std::runtime_error("native fit response CG failed its 1e-8 residual tolerance");
}

std::vector<std::vector<double>> make_probes(const std::vector<RitzMode>& soft,const std::vector<double>& sqrt_mass,
  const std::vector<int>& types,const int n)
{
  const int d=3*n,internal=d-3;std::vector<std::vector<double>> probes;probes.reserve(std::min(24,internal));std::uint64_t state=0x243f6a8885a308d3ULL;
  auto add=[&](std::vector<double> v){project_translation(v,sqrt_mass,n);for(const auto&w:probes){const double c=std::inner_product(v.begin(),v.end(),w.begin(),0.0);for(int i=0;i<d;++i)v[i]-=c*w[i];}const double norm=std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));if(norm>1e-10){for(double&x:v)x/=norm;probes.push_back(std::move(v));}};
  for(int k=0;k<std::min(16,internal);++k){std::vector<double> v(d);for(double&x:v){state^=state>>12;state^=state<<25;state^=state>>27;x=(static_cast<double>((state*2685821657736338717ULL)>>11)/9007199254740992.0)-0.5;}add(std::move(v));}
  std::set<int> seen_types;for(int atom=0;atom<n&&seen_types.size()<4&&static_cast<int>(probes.size())<internal;++atom)if(seen_types.insert(types[atom]).second){std::vector<double> v(d);for(int a=0;a<3;++a)v[a*n+atom]=sqrt_mass[atom];add(std::move(v));}
  for(std::size_t i=0;i<soft.size()&&i<4&&static_cast<int>(probes.size())<internal;++i)add(soft[i].vector);
  if(probes.empty())throw std::runtime_error("native fit could not construct an independent response probe");return probes;
}

struct ResponseCheck{double response=0.0,ibp=0.0,cg=0.0,force_residual=0.0;int probes=0;};
ResponseCheck validate_probes(cusolverDnHandle_t solver,DeviceBaseline& baseline,const Graph& graph,const std::vector<double>& theta,
  const std::vector<double>& sqrt_mass,const std::vector<double>& sqrt_atom,const std::vector<int>& types,const int n,
  const std::vector<double>& r0,std::ifstream& spool,const std::streamoff frame_start,const std::uint64_t frame_count,const int train,
  const int sample_interval,const double temperature,const std::vector<RitzMode>& soft)
{
  const int d=3*n;const auto probes=make_probes(soft,sqrt_atom,types,n);const int m=static_cast<int>(probes.size());if(frame_count-train<=static_cast<std::uint64_t>(m))throw std::runtime_error("native fit held-out frames must exceed the fixed probe count");
  std::vector<double> predicted(static_cast<std::size_t>(m)*m),rhs(d);double max_cg=0.0;
  for(int j=0;j<m;++j){const CGResult cg=solve_projected_cg(baseline,graph,probes[j],sqrt_mass,sqrt_atom,theta,n);max_cg=std::max(max_cg,cg.relative_residual);for(int i=0;i<m;++i)predicted[static_cast<std::size_t>(i)*m+j]=K_B*temperature*std::inner_product(probes[i].begin(),probes[i].end(),cg.x.begin(),0.0);}
  std::vector<double> sumq(m,0.0),sumf(m,0.0),q2(static_cast<std::size_t>(m)*m,0.0),fq(static_cast<std::size_t>(m)*m,0.0),x(d),f(d),q(d),fw(d),base(d),add(d),target(d);double residual2=0.0,force2=0.0;spool.clear();spool.seekg(frame_start);double step=0.0,previous=-std::numeric_limits<double>::infinity();
  for(std::uint64_t frame=0;frame<frame_count;++frame){read_frame(spool,x,f,step);if(!(step>previous)||(frame&&std::abs((step-previous)-sample_interval)>1e-9))throw std::runtime_error("native fit validation steps do not match sample_interval");previous=step;if(frame<static_cast<std::uint64_t>(train))continue;
     for(int i=0;i<d;++i){q[i]=sqrt_mass[i]*(x[i]-r0[i]);fw[i]=f[i]/sqrt_mass[i];}
     project_translation(q,sqrt_atom,n);project_translation(fw,sqrt_atom,n);baseline.apply(q,base);apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i){target[i]=fw[i]+base[i];residual2+=(target[i]+add[i])*(target[i]+add[i]);force2+=fw[i]*fw[i];}std::vector<double> qp(m),fp(m);for(int a=0;a<m;++a){qp[a]=std::inner_product(probes[a].begin(),probes[a].end(),q.begin(),0.0);fp[a]=std::inner_product(probes[a].begin(),probes[a].end(),fw.begin(),0.0);sumq[a]+=qp[a];sumf[a]+=fp[a];}
    for(int a=0;a<m;++a)for(int b=0;b<m;++b){q2[static_cast<std::size_t>(a)*m+b]+=qp[a]*qp[b];fq[static_cast<std::size_t>(a)*m+b]+=fp[a]*qp[b];}
  }
  const int nv=static_cast<int>(frame_count-train);std::vector<double> observed(q2.size()),ibp(q2.size());
  for(int a=0;a<m;++a)for(int b=0;b<m;++b){const std::size_t ij=static_cast<std::size_t>(a)*m+b;observed[ij]=q2[ij]/nv-(sumq[a]/nv)*(sumq[b]/nv);ibp[ij]=fq[ij]/(nv*K_B*temperature)-(sumf[a]/nv)*(sumq[b]/nv)/(K_B*temperature)+(a==b?1.0:0.0);}
  // Symmetric whitening on the finite, declared probe subspace.
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double v=.5*predicted[static_cast<std::size_t>(i)*m+j]+.5*predicted[static_cast<std::size_t>(j)*m+i];predicted[static_cast<std::size_t>(i)*m+j]=predicted[static_cast<std::size_t>(j)*m+i]=v;}
  for(int i=0;i<m;++i)for(int j=i+1;j<m;++j){const double v=.5*observed[static_cast<std::size_t>(i)*m+j]+.5*observed[static_cast<std::size_t>(j)*m+i];observed[static_cast<std::size_t>(i)*m+j]=observed[static_cast<std::size_t>(j)*m+i]=v;}
  const SmallEigen oe=eigen_small(solver,observed,m);if(!(oe.values.front()>0.0)||oe.values.front()<=oe.values.back()*1e-12)throw std::runtime_error("native fit held-out probe covariance is rank deficient or too small to validate");
  const SmallEigen pe=eigen_small(solver,predicted,m);if(!(pe.values.front()>0.0)||pe.values.front()<=pe.values.back()*1e-12)throw std::runtime_error("native fit predicted probe response is non-positive or ill-conditioned");
  std::vector<double> invroot(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)invroot[static_cast<std::size_t>(i)*m+j]+=pe.vectors[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*m]*pe.vectors[static_cast<std::size_t>(j)+static_cast<std::size_t>(k)*m]/std::sqrt(pe.values[k]);
  std::vector<double> whitened(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j){for(int a=0;a<m;++a)for(int b=0;b<m;++b)whitened[static_cast<std::size_t>(i)*m+j]+=invroot[static_cast<std::size_t>(i)*m+a]*(observed[static_cast<std::size_t>(a)*m+b]-predicted[static_cast<std::size_t>(a)*m+b])*invroot[static_cast<std::size_t>(b)*m+j];}
  const SmallEigen re=eigen_small(solver,whitened,m);double response=0.0;for(double e:re.values)response=std::max(response,std::abs(e));
  std::vector<double> gram(static_cast<std::size_t>(m)*m,0.0);for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)gram[static_cast<std::size_t>(i)*m+j]+=ibp[static_cast<std::size_t>(k)*m+i]*ibp[static_cast<std::size_t>(k)*m+j];
  const SmallEigen ie=eigen_small(solver,gram,m);const double ibp_error=std::sqrt(std::max(0.0,ie.values.back()));if(!std::isfinite(response)||!std::isfinite(ibp_error)||!std::isfinite(max_cg))throw std::runtime_error("native fit response diagnostics are non-finite");
  if(!std::isfinite(residual2)||!std::isfinite(force2)||force2<0.0)throw std::runtime_error("native fit held-out force residual is non-finite");const double heldout_residual=force2>0.0?std::sqrt(residual2/force2):(residual2==0.0?0.0:std::numeric_limits<double>::infinity());if(!std::isfinite(heldout_residual))throw std::runtime_error("native fit held-out residual is nonzero at zero physical-force scale");
  return {response,ibp_error,max_cg,heldout_residual,m};
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

std::vector<double> solve_upper(const std::vector<double>& r,const std::vector<double>& z,const int p)
{
  std::vector<double> x=z;
  double largest=0.0,smallest=std::numeric_limits<double>::infinity();
  for(int i=0;i<p;++i){largest=std::max(largest,std::abs(r[static_cast<std::size_t>(i)*p+i]));smallest=std::min(smallest,std::abs(r[static_cast<std::size_t>(i)*p+i]));}
  if (!(smallest>largest*1.0e-12)) throw std::runtime_error("native fit parameter family is rank deficient on training frames");
  for(int i=p-1;i>=0;--i){for(int j=i+1;j<p;++j)x[i]-=r[static_cast<std::size_t>(i)*p+j]*x[j];x[i]/=r[static_cast<std::size_t>(i)*p+i];}
  return x;
}

void make_edge_linear(const Graph& graph,const std::vector<double>& raw_gradient,const int n,
                      std::vector<std::array<double,3>>& edge_linear)
{
  double net2=0.0,gradient2=0.0;for(int a=0;a<3;++a){double net=0.0;for(int i=0;i<n;++i){net+=raw_gradient[a*n+i];gradient2+=raw_gradient[a*n+i]*raw_gradient[a*n+i];}net2+=net*net;}if(!std::isfinite(net2)||!std::isfinite(gradient2)||std::sqrt(net2)>1e-8*std::max(1.0,std::sqrt(gradient2)))throw std::runtime_error("native fit raw qNEP reference gradient has a nonzero net force");
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

} // namespace

static void fit_rpmd_ja_native_reference_impl(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  const std::uint64_t frame_count,Atom& atom,Box& box,Force& force)
{
  if(options.output_path.empty()||options.kernel_table.empty()||spool_path.empty()||!std::isfinite(options.temperature)||options.temperature<=0.0||
     !std::isfinite(options.cutoff)||options.cutoff<=0.0||!std::isfinite(options.epsilon)||options.epsilon<=0.0||
     !std::isfinite(options.response_tolerance)||options.response_tolerance<0.0||!std::isfinite(options.fd_step)||options.fd_step<=0.0||
     options.sample_interval<=0||frame_count<3)throw std::invalid_argument("invalid native rpmd_ja fit options");
  const int n=atom.number_of_atoms;
  if(frame_count>static_cast<std::uint64_t>(std::numeric_limits<int>::max())||n>std::numeric_limits<int>::max()/3)throw std::runtime_error("native fit dimensions exceed indexing limits");
  const int d=3*n,train=static_cast<int>(2*frame_count/3),validation=static_cast<int>(frame_count)-train;
  if(n<2||atom.cpu_mass.size()!=static_cast<std::size_t>(n)||atom.cpu_type.size()!=static_cast<std::size_t>(n)||validation<1)
    throw std::runtime_error("native rpmd_ja fit requires initialized atom masses/types and at least three frames");
  std::ifstream in(spool_path,std::ios::binary);if(!in)throw std::runtime_error("cannot open native rpmd_ja sample spool: "+spool_path);
  const SampleHeader header=read_header(in,frame_count,atom,box,options.temperature);
  std::set<int> unique_types(header.types.begin(),header.types.end());
  const int expected_probes=std::min(3*n-3,16+std::min(4,static_cast<int>(unique_types.size()))+4);
  if(validation<=expected_probes)throw std::runtime_error("native fit requires at least " + std::to_string(expected_probes+1) + " held-out frames for its fixed probe policy");
  std::vector<double> position(3*n),force_frame(3*n),r0(3*n,0.0);double step=0.0,previous=-std::numeric_limits<double>::infinity();
  read_training_r0(in,header,frame_count,static_cast<double>(options.sample_interval),r0);
  validate_fit_branches(in,header,frame_count,box,r0);
  // Rewind to the first frame for streaming block-QR fitting.
  in.clear();in.seekg(header.frames);
  previous=-std::numeric_limits<double>::infinity();
  const Graph graph=make_graph(r0,header.types,box,options.cutoff);
  const std::size_t psize=6*graph.type_pairs.size();
  if(psize==0||psize>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::runtime_error("native fit has an invalid parameter count");
  const std::uint64_t frame_design=static_cast<std::uint64_t>(d)*psize;
  const std::uint64_t padded_rows=std::max<std::uint64_t>(d,psize);
  if(frame_design>std::numeric_limits<std::size_t>::max()/sizeof(double)||padded_rows>std::numeric_limits<std::size_t>::max()/psize/sizeof(double)||
     psize>std::numeric_limits<std::size_t>::max()/psize/sizeof(double))
    throw std::runtime_error("native fit parameter count exceeds actual frame-block or QR storage preflight");
  const std::uint64_t design_bytes=frame_design*sizeof(double),qr_bytes=padded_rows*psize*sizeof(double)+(padded_rows+psize)*sizeof(double)+psize*psize*sizeof(double);
  if(design_bytes>std::numeric_limits<std::size_t>::max()-qr_bytes)throw std::runtime_error("native fit actual host/GPU frame storage estimate overflows");
  std::printf("    rpmd_ja native fit family: fixed cutoff graph, %zu unordered type pairs, %zu shared lab-frame symmetric parameters; training=%d validation=%d; frame design %.3f MiB\n",
    graph.type_pairs.size(),psize,train,validation,static_cast<double>(design_bytes)/(1024*1024));
  std::vector<double> saved(static_cast<std::size_t>(d));atom.position_per_atom.copy_to_host(saved.data());
  struct Restore { Atom& atom; std::vector<double>& x; ~Restore(){atom.position_per_atom.copy_from_host(x.data());} } restore{atom,saved};
  atom.position_per_atom.copy_from_host(r0.data());
  const std::string raw_path=options.output_path+".qraw",pack_path=options.output_path+".additive.tmp",fit_path=options.output_path+".fit.txt",fit_tmp=fit_path+".tmp";
  {std::ifstream kernel(options.kernel_table),old(options.output_path,std::ios::binary),side(options.output_path+".stability"),raw_old(raw_path,std::ios::binary),pack_old(pack_path,std::ios::binary),fit_old(fit_path),fit_tmp_old(fit_tmp),failure(options.output_path+".failure.txt"),tmp(options.output_path+".tmp"),side_tmp(options.output_path+".stability.tmp");
    if(!kernel)throw std::runtime_error("native fit cannot read its qNEP kernel table");kernel.seekg(0,std::ios::end);const auto kernel_bytes=kernel.tellg();if(kernel_bytes<=0||kernel_bytes>std::numeric_limits<std::streamoff>::max())throw std::runtime_error("native fit qNEP kernel table has invalid or overflowing byte length");if(old.good()||side.good()||raw_old.good()||pack_old.good()||fit_old.good()||fit_tmp_old.good()||failure.good()||tmp.good()||side_tmp.good())throw std::runtime_error("native fit refuses to overwrite an existing output or scratch artifact");}
  const std::uint64_t d64=static_cast<std::uint64_t>(d),n64=static_cast<std::uint64_t>(n),dd64=d64*d64,vd64=d64*n64;
  if(dd64>(std::numeric_limits<std::uint64_t>::max()-18)/7||vd64>(std::numeric_limits<std::uint64_t>::max()-7*dd64-18)/2)throw std::runtime_error("native fit qraw size estimate overflows");
  const std::uint64_t raw_elements=2*vd64+7*dd64+18;if(raw_elements>std::numeric_limits<std::uint64_t>::max()/sizeof(double))throw std::runtime_error("native fit qraw byte estimate overflows");
  const std::uint64_t raw_bytes=raw_elements*sizeof(double);
  if(dd64>std::numeric_limits<std::size_t>::max()/sizeof(double)||raw_bytes>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()))throw std::runtime_error("native fit qraw/Hessian exceeds host or file-offset limits");
  std::size_t free_before=0,total_before=0;if(cudaMemGetInfo(&free_before,&total_before)!=cudaSuccess)throw std::runtime_error("native fit CUDA memory preflight failed");
  std::printf("    rpmd_ja native fit preflight: qraw %.2f GiB, GPU baseline %.2f GiB, free GPU %.2f GiB, host design+QR %.2f MiB\n",static_cast<double>(raw_bytes)/(1024.0*1024.0*1024.0),static_cast<double>(dd64*sizeof(double))/(1024.0*1024.0*1024.0),static_cast<double>(free_before)/(1024.0*1024.0*1024.0),static_cast<double>(design_bytes+qr_bytes)/(1024.0*1024.0));
  if(dd64*sizeof(double)>free_before||free_before-dd64*sizeof(double)<256ULL*1024*1024)throw std::runtime_error("native fit baseline matrix does not fit current GPU memory with a 256 MiB safety margin");
  struct OwnScratch{std::string raw,pack,output,sidecar,fit,fit_tmp;bool raw_owned=false,pack_owned=false,output_owned=false,sidecar_owned=false,fit_owned=false,fit_tmp_owned=false;~OwnScratch(){if(raw_owned)std::remove(raw.c_str());if(pack_owned)std::remove(pack.c_str());if(output_owned)std::remove(output.c_str());if(sidecar_owned)std::remove(sidecar.c_str());if(fit_owned)std::remove(fit.c_str());if(fit_tmp_owned)std::remove(fit_tmp.c_str());}} own_scratch{raw_path,pack_path,options.output_path,options.output_path+".stability",fit_path,fit_tmp};
  generate_rpmd_ja_qnep_raw(raw_path,options.temperature,options.fd_step,options.kernel_table,atom,box,force);
  own_scratch.raw_owned=true;
  std::ifstream raw(raw_path,std::ios::binary);if(!raw)throw std::runtime_error("native fit did not find generated qNEP raw reference");
  char rawmagic[8];raw.read(rawmagic,8);const auto rawver=read<std::uint32_t>(raw,"qraw version");const auto endian=read<std::uint32_t>(raw,"qraw byte order");const int raw_n=read<std::int32_t>(raw,"qraw atom count"),raw_d=read<std::int32_t>(raw,"qraw dimension");
  const double rawT=read<double>(raw,"qraw temperature");const double rawfd=read<double>(raw,"qraw fd step");(void)read<std::uint64_t>(raw,"qraw model fingerprint");(void)read<std::uint64_t>(raw,"qraw config fingerprint");
  (void)read<std::int32_t>(raw,"qraw charge mode");(void)read<std::int32_t>(raw,"qraw kspace flag");(void)read<double>(raw,"qraw mesh spacing");
  char layout[sizeof("xyz_soa;derivative_input_rows_output_columns")];raw.read(layout,sizeof(layout));
  raw.seekg(18*sizeof(double)+3*sizeof(std::int32_t),std::ios::cur);
  raw.seekg(static_cast<std::streamoff>(n)*(sizeof(std::int32_t)+sizeof(double))+static_cast<std::streamoff>(d)*sizeof(double),std::ios::cur);
  raw.seekg(sizeof(double)+static_cast<std::streamoff>(n)*sizeof(double)+static_cast<std::streamoff>(d)*sizeof(double)+static_cast<std::streamoff>(9*n)*sizeof(double),std::ios::cur);
  const std::streamoff data=raw.tellg();const std::uint64_t dd=static_cast<std::uint64_t>(d)*d,vd=static_cast<std::uint64_t>(d)*n;
  const std::streamoff k_offset=data+static_cast<std::streamoff>(vd+3*dd)*sizeof(double);
  if(!raw||std::memcmp(rawmagic,"GPJQRAW\0",8)||rawver!=3||endian!=native_endian||raw_n!=n||raw_d!=d||rawT!=options.temperature||rawfd!=options.fd_step||
     std::memcmp(layout,"xyz_soa;derivative_input_rows_output_columns",sizeof(layout)))throw std::runtime_error("generated qNEP raw header does not match native fit assumptions");
  raw.seekg(0,std::ios::end);const std::streamoff raw_end=raw.tellg();const std::uint64_t raw_expected=static_cast<std::uint64_t>(data)+((2*vd+7*dd+18)*sizeof(double));
  if(raw_expected>static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())||raw_end!=static_cast<std::streamoff>(raw_expected))throw std::runtime_error("generated qNEP raw file byte length is invalid");
  double raw_stats[18];raw.seekg(static_cast<std::streamoff>(data)+static_cast<std::streamoff>(2*vd+7*dd)*sizeof(double));raw.read(reinterpret_cast<char*>(raw_stats),sizeof(raw_stats));const bool finite_stats=std::all_of(raw_stats,raw_stats+18,[](double v){return std::isfinite(v)&&v>=0.0;});const bool raw_v1=raw_stats[17]==1.0&&raw_stats[1]<=1e-4&&std::max({raw_stats[2],raw_stats[4],raw_stats[6],raw_stats[7],raw_stats[8],raw_stats[14],raw_stats[15]})<=5e-2;const bool raw_v2=raw_stats[17]==2.0&&raw_stats[1]<=1e-4&&raw_stats[3]<=1e-4&&std::max({raw_stats[4],raw_stats[6],raw_stats[7],raw_stats[8],raw_stats[14],raw_stats[15]})<=5e-2;const bool raw_v3=raw_stats[17]==3.0&&raw_stats[1]<=1e-4&&raw_stats[3]<=1e-4&&std::max({raw_stats[4],raw_stats[6],raw_stats[7],raw_stats[8],raw_stats[14],raw_stats[15]})<=5e-2;if(!raw||!finite_stats||raw_stats[16]!=options.fd_step||(rawver==1?!raw_v1:(rawver==2?!raw_v2:!raw_v3)))throw std::runtime_error("generated qNEP raw diagnostics fail accepted limits");raw.clear();
  std::vector<double> raw_gradient(d),site(n);for(int a=0;a<d;++a){raw.clear();raw.seekg(data+static_cast<std::streamoff>(a)*n*sizeof(double));read_array(raw,site.data(),n,"raw qNEP site gradient");if(!std::all_of(site.begin(),site.end(),[](double v){return std::isfinite(v);}))throw std::runtime_error("native fit qNEP site gradient contains non-finite values");raw_gradient[a]=std::accumulate(site.begin(),site.end(),0.0);if(!std::isfinite(raw_gradient[a]))throw std::runtime_error("native fit qNEP reference gradient is non-finite");}
  std::vector<std::array<double,3>> edge_linear;make_edge_linear(graph,raw_gradient,n,edge_linear);
  std::vector<double> sqrt_mass(d),sqrt_mass_atom(n),q(d),f(d),base(d),target(d);
  for(int i=0;i<n;++i){if(!std::isfinite(atom.cpu_mass[i])||atom.cpu_mass[i]<=0.0)throw std::runtime_error("native fit requires positive finite masses");sqrt_mass_atom[i]=std::sqrt(atom.cpu_mass[i]);}
  for(int a=0;a<3;++a)for(int i=0;i<n;++i)sqrt_mass[a*n+i]=sqrt_mass_atom[i];
  std::vector<double> rmat(psize*psize,0.0),z(psize,0.0),R,zblock;double training_force2=0.0,discarded2=0.0;
  std::vector<double> design(static_cast<std::size_t>(d)*psize);
  DeviceBaseline baseline;baseline.initialize(raw,k_offset,d,n,atom.cpu_mass,sqrt_mass);
  DeviceQR qr;qr.initialize(d,static_cast<int>(psize));
  raw.close();
  for(int frame=0;frame<frame_count;++frame){
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
  const SmallSVD fit_svd=svd_small(qr.solver,rmat,z,static_cast<int>(psize));
  std::vector<std::vector<double>> cut_rows;std::vector<double> cut_rhs;std::vector<double> theta=theta_from_eta(fit_svd,fit_svd.eta);
  bool ritz_converged=false;
  for(int outer=0;outer<=40;++outer){
    const auto modes=lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,96,4);
    if(modes.empty())throw std::runtime_error("native fit Lanczos returned no Ritz modes");
    bool violated=false;
    for(const auto& mode:modes){
      std::vector<double> total_mode(d),base_mode(d);apply_total(baseline,graph,mode.vector,sqrt_mass,sqrt_mass_atom,theta,n,total_mode);
      const double rayleigh=std::inner_product(mode.vector.begin(),mode.vector.end(),total_mode.begin(),0.0);
      if(rayleigh>=options.epsilon)continue;
      baseline.apply(mode.vector,base_mode);const double base_value=std::inner_product(mode.vector.begin(),mode.vector.end(),base_mode.begin(),0.0);
      const auto row=spectral_row(graph,mode.vector,sqrt_mass,n,static_cast<int>(psize));
      const double row_norm=std::sqrt(std::inner_product(row.begin(),row.end(),row.begin(),0.0));
      if(row_norm<=1e-20){if(base_value<options.epsilon)throw std::runtime_error("native fit has an unstable baseline mode outside the fitted edge family");continue;}
      const double rhs=options.epsilon-base_value;bool implied=false;
      for(std::size_t c=0;c<cut_rows.size();++c){const double prior_norm=std::sqrt(std::inner_product(cut_rows[c].begin(),cut_rows[c].end(),cut_rows[c].begin(),0.0));const double cosine=std::inner_product(row.begin(),row.end(),cut_rows[c].begin(),0.0)/(row_norm*prior_norm);
        if(cosine>1.0-1e-8&&rhs/row_norm<=cut_rhs[c]/prior_norm+1e-8){implied=true;break;}}
      if(implied)continue;cut_rows.push_back(row);cut_rhs.push_back(rhs);violated=true;
    }
    if(!violated){ritz_converged=true;break;}
    if(outer==40)break;
    theta=theta_from_eta(fit_svd,solve_cut_qp(fit_svd,cut_rows,cut_rhs));
  }
  if(!ritz_converged)throw std::runtime_error("native fit spectral-cut iteration reached its 40-cut limit; no final reference was written");
  double compressed_residual2=discarded2;for(int i=0;i<static_cast<int>(psize);++i){double v=-z[i];for(int j=i;j<static_cast<int>(psize);++j)v+=rmat[static_cast<std::size_t>(i)*psize+j]*theta[j];compressed_residual2+=v*v;}
  if(!std::isfinite(compressed_residual2)||!std::isfinite(training_force2)||training_force2<0.0)throw std::runtime_error("native fit training force residual is non-finite");const double training_force_residual=training_force2>0.0?std::sqrt(compressed_residual2/training_force2):(compressed_residual2==0.0?0.0:std::numeric_limits<double>::infinity());if(!std::isfinite(training_force_residual))throw std::runtime_error("native fit training residual is nonzero at zero physical-force scale");
  const auto final_modes=lanczos_low_modes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,n,96,4);
  const ResponseCheck response=validate_probes(qr.solver,baseline,graph,theta,sqrt_mass,sqrt_mass_atom,header.types,n,r0,in,header.frames,frame_count,train,options.sample_interval,options.temperature,final_modes);
  if(response.response>options.response_tolerance||response.ibp>options.response_tolerance)
    throw std::runtime_error("native fit held-out probe response or force-position IBP exceeds the declared tolerance");
  std::uint64_t estimated=88;
  for(int i=0;i<n;++i){const std::uint64_t zsite=graph.sites[i].size();estimated+=sizeof(std::int32_t)+zsite*16+zsite*3*sizeof(double)+9*zsite*zsite*sizeof(double);}
  if(estimated>std::numeric_limits<std::streamsize>::max())throw std::runtime_error("native additive package byte estimate overflows");
  {
    std::ofstream out(pack_path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create native additive package");own_scratch.pack_owned=true;
    out.write(additive_magic,8);write(out,std::uint32_t(1));write(out,native_endian);write(out,std::int32_t(n));write(out,std::int32_t(header.beads));write(out,rpmd_ja_model_fingerprint(raw_path));
    write(out,options.temperature);write(out,options.epsilon);write(out,response.response);write(out,options.response_tolerance);write(out,std::uint64_t(train));write(out,std::uint64_t(validation));
    for(int i=0;i<n;++i){const int zs=static_cast<int>(graph.sites[i].size());write(out,std::int32_t(zs));for(const auto&e:graph.sites[i]){write(out,std::int32_t(e.j));for(int a=0;a<3;++a)write(out,std::int32_t(e.image[a]));}
      for(const auto&e:graph.sites[i])for(int a=0;a<3;++a)write(out,.5*e.sign*edge_linear[e.edge_index][a]);
      const int qdim=3*zs;for(int a=0;a<qdim;++a)for(int b=0;b<qdim;++b){double v=0.0;if(a/3==b/3&&a%3<=b%3){const int group=graph.sites[i][a/3].group;const int lo=std::min(a%3,b%3),hi=std::max(a%3,b%3);const int ci=lo==0?(hi==0?0:(hi==1?1:2)):(lo==1?(hi==1?3:4):5);v=.5*theta[static_cast<std::size_t>(6*group+ci)];}else if(a/3==b/3){const int group=graph.sites[i][a/3].group;const int lo=std::min(a%3,b%3),hi=std::max(a%3,b%3);const int ci=lo==0?(hi==0?0:(hi==1?1:2)):(lo==1?(hi==1?3:4):5);v=.5*theta[static_cast<std::size_t>(6*group+ci)];}write(out,v);}
    }
    out.close();if(!out)throw std::runtime_error("failed closing native additive package");
  }
  std::printf("    rpmd_ja fit/probe validation complete; preparing reference: family=shared_lab_frame_typepair_edge_blocks parameters=%zu; restricted family, probes=%d seed=0x243f6a8885a308d3; train/heldout force residual=%.6g/%.6g response=%.6g IBP=%.6g CG residual=%.3g; stability cuts=%zu; estimated additive bytes=%llu\n",psize,response.probes,training_force_residual,response.force_residual,response.response,response.ibp,response.cg,cut_rows.size(),static_cast<unsigned long long>(estimated));
  qr.release();baseline.release();prepare_rpmd_ja_qnep_reference(raw_path,options.output_path,options.kernel_table,make_rpmd_ja_qnep_mode_validator(atom,box,force),pack_path);own_scratch.output_owned=true;own_scratch.sidecar_owned=true;
  double certificate_epsilon=0.0,certificate_eta=0.0;{std::ifstream stability(options.output_path+".stability");std::string key;while(stability>>key){if(key=="epsilon_num")stability>>certificate_epsilon;else if(key=="reconstruction_bound")stability>>certificate_eta;else{std::string value;stability>>value;}}if(!stability.eof()||!std::isfinite(certificate_epsilon)||!std::isfinite(certificate_eta)||!(certificate_epsilon>0.0)||!(certificate_eta<0.5*certificate_epsilon))throw std::runtime_error("native fit could not read a valid stored-D Cholesky certificate");}
  {std::ofstream fitout(fit_tmp,std::ios::out|std::ios::trunc);if(!fitout)throw std::runtime_error("cannot create native fit diagnostics");own_scratch.fit_tmp_owned=true;fitout<<"family shared_lab_frame_symmetric_typepair_edge_blocks\nparameters "<<psize<<"\ntype_pairs "<<graph.type_pairs.size()<<"\ncutoff_A "<<std::setprecision(17)<<options.cutoff<<"\ntraining_frames "<<train<<"\nvalidation_frames "<<validation<<"\nprobe_policy fixed_seeded_random16_type_local_up_to4_low_ritz4\nprobe_seed 0x243f6a8885a308d3\nprobe_count "<<response.probes<<"\nprobe_subspace_only 1\nforce_residual_normalization massweighted_projected_physical_force_norm\ntraining_force_residual_relative "<<training_force_residual<<"\nheldout_force_residual_relative "<<response.force_residual<<"\nprobe_response_error "<<response.response<<"\nforce_position_ibp_error "<<response.ibp<<"\nprobe_cg_max_true_relative_residual "<<response.cg<<"\nresponse_tolerance "<<options.response_tolerance<<"\nepsilon_fit "<<options.epsilon<<"\nspectral_cuts "<<cut_rows.size()<<"\ncertificate stored_D_additive_shifted_frobenius_SPD_bound\ncertificate_epsilon_num "<<certificate_epsilon<<"\ncertificate_reconstruction_bound "<<certificate_eta<<"\ncertificate_lambda_min_lower_bound "<<0.5*certificate_epsilon-certificate_eta<<"\ncertificate_note SPD lower bound only; not an epsilon eigenvalue floor\n";fitout.close();if(!fitout)throw std::runtime_error("failed closing native fit diagnostics");}
  if(std::rename(fit_tmp.c_str(),fit_path.c_str())!=0)throw std::runtime_error("native fit could not publish its diagnostics");own_scratch.fit_tmp_owned=false;own_scratch.fit_owned=true;
  in.close();raw.close();if(std::remove(pack_path.c_str())!=0||std::remove(raw_path.c_str())!=0)throw std::runtime_error("native fit succeeded but could not remove its own additive/qraw scratch files");own_scratch.pack_owned=false;own_scratch.raw_owned=false;own_scratch.output_owned=false;own_scratch.sidecar_owned=false;own_scratch.fit_owned=false;
  std::printf("    rpmd_ja native reference generated and certified; stability sidecar and fit diagnostics published.\n");
}

void fit_rpmd_ja_native_reference(const RpmdJANativeFitOptions& options,const std::string& spool_path,
  const std::uint64_t frame_count,Atom& atom,Box& box,Force& force)
{
  try{fit_rpmd_ja_native_reference_impl(options,spool_path,frame_count,atom,box,force);}
  catch(const std::exception& e){
    if(!options.output_path.empty()){
      const std::string failure=options.output_path+".failure.txt";std::ifstream exists(failure);
      if(!exists.good()){std::ofstream out(failure,std::ios::out|std::ios::trunc);if(out)out<<"native finite-temperature reference fit failed\n"<<e.what()<<'\n';}
    }
    throw;
  }
}

void diagnose_rpmd_ja_native_fit_samples(const std::string& spool_path, const double fd_step,
                                         Atom& atom, Box& box, Force& force)
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
  diagnose_rpmd_ja_qnep_reference(fd_step, atom, box, force);
}
#else
#include <stdexcept>
void fit_rpmd_ja_native_reference(const RpmdJANativeFitOptions&, const std::string&,
  const std::uint64_t, Atom&, Box&, Force&)
{
  throw std::runtime_error("native rpmd_ja reference fitting requires CUDA");
}
void diagnose_rpmd_ja_native_fit_samples(const std::string&, double, Atom&, Box&, Force&)
{
  throw std::runtime_error("native rpmd_ja sample diagnostics require CUDA");
}
#endif
