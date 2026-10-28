#include "measure/rpmd_ja_qnep_prepare.cuh"
#include "measure/rpmd_ja_reference.cuh"
#include "measure/rpmd_ja_sample_provenance.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
template <typename T> void put(std::ostream& out,const T& x){out.write(reinterpret_cast<const char*>(&x),sizeof(x));}
template <typename T> void array(std::ostream& out,const std::vector<T>& x){if(!x.empty())out.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T));}

void make_input(const std::string& raw,const std::string& kernel,const int n,const double stiffness,
                const std::uint32_t version=1,const double stats_marker=-1.0,const double identity_abs=0.0,const int negative_axis=-1,
                const bool multi_spectrum=false,const std::vector<double>* custom_masses=nullptr,const std::vector<int>* custom_types=nullptr,
                const bool full_rank_h=false,const bool zero_h=false,const double h_scale=1.0,const double h_noise=0.0)
{
  const int d=3*n;const double temperature=300.0,step=0.01;
  std::vector<double> mass(n),position(d,0.0),v(static_cast<std::size_t>(d)*n),vc(v.size()),c(static_cast<std::size_t>(3)*d*d),k(static_cast<std::size_t>(d)*d),coarse_c(c.size());
  std::vector<int> types(n,0),pbc(3,1);double mass_sum=0.0;
  if((custom_masses&&custom_masses->size()!=static_cast<std::size_t>(n))||
     (custom_types&&custom_types->size()!=static_cast<std::size_t>(n)))throw std::invalid_argument("custom qraw atom fixture dimensions do not match");
  for(int i=0;i<n;++i){mass[i]=custom_masses?(*custom_masses)[i]:1.0+0.07*i;types[i]=custom_types?(*custom_types)[i]:0;mass_sum+=mass[i];}
  std::vector<std::vector<double>> complement;
  if(multi_spectrum){
    const double inv_mass=1.0/std::sqrt(mass_sum);std::vector<double> translation(n);for(int i=0;i<n;++i)translation[i]=std::sqrt(mass[i])*inv_mass;
    for(int q=0;q<n-1;++q){std::vector<double> u(n,0.0);u[q]=1.0;double dot=0.0;for(int i=0;i<n;++i)dot+=u[i]*translation[i];for(int i=0;i<n;++i)u[i]-=dot*translation[i];for(const auto& vq:complement){dot=0.0;for(int i=0;i<n;++i)dot+=u[i]*vq[i];for(int i=0;i<n;++i)u[i]-=dot*vq[i];}double norm=0.0;for(double x:u)norm+=x*x;norm=std::sqrt(norm);for(double& x:u)x/=norm;complement.push_back(u);}
  }
  for(int row=0;row<d;++row)for(int col=0;col<d;++col){const int ar=row%n,ac=col%n;if(row/n==col/n){double dm=0.0;if(multi_spectrum){for(int q=0;q<n-1;++q){const double value=(row/n)*10.0+ (q==0?-1.0:static_cast<double>(q));dm+=value*complement[q][ar]*complement[q][ac];}}else{const double tr=std::sqrt(mass[ar]/mass_sum),tc=std::sqrt(mass[ac]/mass_sum);const double axis_stiffness=negative_axis<0?stiffness:(row/n==negative_axis?-1.0e-10:1.0e-10);dm=axis_stiffness*((row==col?1.0:0.0)-tr*tc);}k[static_cast<std::size_t>(col)*d+row]=dm*std::sqrt(mass[ar]*mass[ac]);}}
  for(int row=0;row<d;++row)for(int atom=0;atom<n;++atom){const std::size_t ix=static_cast<std::size_t>(row)*n+atom;v[ix]=zero_h?0.0:h_scale*0.03*std::sin((row+1)*(atom+2));vc[ix]=version>=2?v[ix]:0.97*v[ix];}
  for(int alpha=0;alpha<3;++alpha)for(int col=0;col<d;++col)for(int row=0;row<d;++row){const int site=row%n,atom=col%n;const double left=1.0+0.01*std::sin((row+1)*(alpha+1));const double right=1.0+0.01*std::cos((col+1)*(alpha+2));const double phase=(row+1)*12.9898+(col+1)*78.233+(alpha+1)*37.719;const double noise=std::sin(phase)*43758.5453-std::floor(std::sin(phase)*43758.5453);const double target=zero_h?0.0:h_scale*(full_rank_h?noise+(row==col?1.0:0.0):left*right+h_noise*noise);double fine=target,coarse=target;if(col/n==alpha){fine+=v[static_cast<std::size_t>(row)*n+atom];coarse+=vc[static_cast<std::size_t>(row)*n+atom];if(site==atom)for(int a=0;a<n;++a){fine-=v[static_cast<std::size_t>(row)*n+a];coarse-=vc[static_cast<std::size_t>(row)*n+a];}}const std::size_t ix=static_cast<std::size_t>(alpha)*d*d+static_cast<std::size_t>(col)*d+row;c[ix]=fine;coarse_c[ix]=coarse;}
  std::ofstream out(raw,std::ios::binary);if(!out)throw std::runtime_error("cannot create raw fixture");
  const char magic[8]={'G','P','J','Q','R','A','W','\0'};const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::uint32_t endian=0x01020304;const int charge=1,pppm=1;const std::uint64_t model=0x12345678,config=0x87654321;const double spacing=0.5;
  out.write(magic,8);put(out,version);put(out,endian);put(out,n);put(out,d);put(out,temperature);put(out,step);put(out,model);put(out,config);put(out,charge);put(out,pppm);put(out,spacing);out.write(layout,sizeof(layout));
  double cell[18]={1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1};out.write(reinterpret_cast<char*>(cell),sizeof(cell));out.write(reinterpret_cast<const char*>(pbc.data()),sizeof(int)*3);
  array(out,types);array(out,mass);array(out,position);const double energy=0.0;std::vector<double> site_energy(n,0),force(d,0),virial(9*n,0);for(int row=0;row<d;++row)for(int atom=0;atom<n;++atom)force[row]-=v[static_cast<std::size_t>(row)*n+atom];put(out,energy);array(out,site_energy);array(out,force);array(out,virial);
  array(out,v);array(out,c);array(out,k);double stats[24]={};for(double value:force)stats[12]=std::max(stats[12],std::abs(value));stats[3]=identity_abs;stats[16]=step;stats[17]=stats_marker<0.0?(version==4?3.0:static_cast<double>(version)):stats_marker;
  if(version==4){for(int alpha=0;alpha<3;++alpha)for(int col=0;col<d;++col)for(int row=0;row<d;++row){const int site=row%n,atom=col%n,nu=col/n;double h=c[static_cast<std::size_t>(alpha)*d*d+static_cast<std::size_t>(col)*d+row];if(nu==alpha){h-=v[static_cast<std::size_t>(row)*n+atom];if(site==atom){double sum=0;for(int a=0;a<n;++a)sum+=v[static_cast<std::size_t>(row)*n+a];h+=sum;}}h/=std::sqrt(mass[site]*mass[atom]);stats[21+alpha]+=h*h;}}
  if(version==4)array(out,std::vector<double>(stats,stats+24));else{array(out,vc);array(out,coarse_c);array(out,std::vector<double>(stats,stats+18));}
  out.close();if(!out)throw std::runtime_error("failed writing raw fixture");
  std::ofstream kt(kernel);if(!kt)throw std::runtime_error("cannot create kernel fixture");
  kt<<"GPUMDJA_KERNEL 1\nU 100\ndegree 1\nP_rank 1\nQ_rank 1\nP_error 0\nQ_error 0\nP_S2 0\nQ_S2 0\nP_values\n1\nP_vectors\n0 1\nQ_values\n1\nQ_vectors\n0 1\nEND\n";
}

bool has_compressed_h(const RpmdJAReference& r)
{for(int a=0;a<3;++a)for(const auto& t:r.block_site_transpose[a].tiles)if(t.rank>0)return true;return false;}

double tile_value(const RpmdJABlockMatrix& matrix,const int row,const int column)
{
  for(const auto& t:matrix.tiles)if(row>=t.row&&row<t.row+t.rows&&column>=t.column&&column<t.column+t.columns){
    const int i=row-t.row,j=column-t.column;
    if(t.rank==0)return t.left[static_cast<std::size_t>(i)*t.columns+j];
    double x=0.0;for(int q=0;q<t.rank;++q)x+=t.left[static_cast<std::size_t>(i)*t.rank+q]*t.right[static_cast<std::size_t>(q)*t.columns+j];return x;
  }
  throw std::runtime_error("readback matrix tile missing");
}

void verify_matrices(const RpmdJAReference& r,const double stiffness)
{
  const int n=r.number_of_atoms,d=3*n;double total=0.0;for(double m:r.masses)total+=m;
  for(int row=0;row<d;++row)for(int col=0;col<d;++col){
    const double trow=std::sqrt(r.masses[row%n]/total),tcol=std::sqrt(r.masses[col%n]/total);
    const double expected=(row/n==col/n)?stiffness*((row==col?1.0:0.0)-trow*tcol):0.0;
    const double actual=tile_value(r.block_dynamical,row,col);
    if(std::abs(actual-expected)>2.0e-9*std::max(std::abs(stiffness),1.0e-10))throw std::runtime_error("D matrix readback mismatch");
  }
  for(int alpha=0;alpha<3;++alpha){
    for(int row=0;row<d;++row)for(int col=0;col<d;++col){
      const double left=1.0+0.01*std::sin((row+1)*(alpha+1));
      const double right=1.0+0.01*std::cos((col+1)*(alpha+2));
      const double expected=left*right/std::sqrt(r.masses[row%n]*r.masses[col%n]);
      if(std::abs(tile_value(r.block_site_transpose[alpha],row,col)-expected)>2.0e-8*std::max(std::abs(expected),1.0))
        throw std::runtime_error("H tile readback mismatch");
    }
    if(std::abs(tile_value(r.block_site_transpose[alpha],0,1)-tile_value(r.block_site_transpose[alpha],1,0))<1.0e-5)
      throw std::runtime_error("fixture H direction unexpectedly symmetric");
  }
  for(int axis=0;axis<3;++axis)for(int row=axis*n;row<(axis+1)*n;++row){
    double projected=0.0;for(int col=axis*n;col<(axis+1)*n;++col)projected+=tile_value(r.block_dynamical,row,col)*std::sqrt(r.masses[col%n]/total);
    if(std::abs(projected)>1.0e-8)throw std::runtime_error("D translation mode readback mismatch");
  }
}

void clean(const std::string& p){std::remove(p.c_str());std::remove((p+".tmp").c_str());std::remove((p+".stability").c_str());std::remove((p+".stability.tmp").c_str());std::remove((p+".failure.txt").c_str());}

bool exists(const std::string& path){std::ifstream file(path,std::ios::binary);return file.good();}

std::vector<char> read_file_bytes(const std::string& path)
{
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("cannot read fixture bytes: "+path);
  return std::vector<char>(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
}

void run_positive(const std::string& base,const int n,const std::uint32_t version)
{
  const std::string stem=base+"_N"+std::to_string(n),raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja";
  make_input(raw,kernel,n,1.0e-10,version);prepare_rpmd_ja_qnep_reference(raw,out,kernel);
  std::uint64_t fingerprint=0;const RpmdJAReference r=read_rpmd_ja_reference(out,&fingerprint);
  if(fingerprint!=rpmd_ja_model_fingerprint(out))throw std::runtime_error("reference fingerprint mismatch");
  const std::string expected_sha=rpmd_ja_sample_provenance::sha256_file(out);
  if(r.source_file_sha256!=expected_sha||read_rpmd_ja_reference(out).source_file_sha256!=expected_sha)
    throw std::runtime_error("reference SHA-256 mismatch");
  const std::string sidecar=out+".stability";
  const auto sidecar_bytes=read_file_bytes(sidecar);
  std::string altered(sidecar_bytes.begin(),sidecar_bytes.end());
  const auto fingerprint_start=altered.find("fingerprint ");
  if(fingerprint_start==std::string::npos)throw std::runtime_error("sidecar fingerprint missing");
  const auto digit=fingerprint_start+std::string("fingerprint ").size();
  altered[digit]=altered[digit]=='0'?'1':'0';
  { std::ofstream broken(sidecar,std::ios::binary|std::ios::trunc);broken<<altered; }
  bool rejected=false;try{read_rpmd_ja_reference(out,&fingerprint);}catch(const std::exception&){rejected=true;}
  { std::ofstream restore(sidecar,std::ios::binary|std::ios::trunc);restore.write(sidecar_bytes.data(),sidecar_bytes.size()); }
  if(!rejected)throw std::runtime_error("tampered sidecar fingerprint accepted");
  read_rpmd_ja_reference(out,&fingerprint);
  if(fingerprint!=rpmd_ja_model_fingerprint(out))throw std::runtime_error("restored fingerprint mismatch");
  if(r.backend!=2||r.number_of_atoms!=n||r.block_dynamical.tiles.empty()||r.block_site_transpose[0].tiles.empty()||!r.stability_checked||!has_compressed_h(r)||!(r.force_balance_residual>1.0e-4))throw std::runtime_error("prepared qNEP v3 fixture failed nonzero-force readback or H compression");
  const std::string expected_policy=version==1?"native_reference_transport":
    (version==2?"native_reference_transport;analytic_site_gradient_v1":"native_reference_transport;analytic_site_gradient_fd4_v1");
  if(r.mechanical_policy!=expected_policy)throw std::runtime_error("prepared qNEP v3 fixture lost its derivative-source marker");
  if(r.block_dynamical.tiles.size()!=static_cast<std::size_t>((3*n+127)/128)*((3*n+127)/128))throw std::runtime_error("D tile grid readback incomplete");
  std::ifstream serialized(out,std::ios::binary|std::ios::ate);
  const std::uint64_t actual_bytes=static_cast<std::uint64_t>(serialized.tellg());
  if(!serialized||actual_bytes>estimate_rpmd_ja_qnep_v3_max_serialized_bytes(r))
    throw std::runtime_error("qNEP v3 output exceeds its conservative serialized-size bound");
  verify_matrices(r,1.0e-10);
  clean(raw);clean(kernel);clean(out);
}

void run_dense_h_upper_bound(const std::string& base)
{
  constexpr int n=260;const std::string raw=base+"_dense_h.qraw",kernel=base+"_dense_h.kernel",out=base+"_dense_h.ja";
  make_input(raw,kernel,n,1.0e-10,3,-1.0,0.0,-1,false,nullptr,nullptr,true);
  prepare_rpmd_ja_qnep_reference(raw,out,kernel);const RpmdJAReference reference=read_rpmd_ja_reference(out);
  for(const auto& tile:reference.block_dynamical.tiles)
    if(tile.rank!=0)throw std::runtime_error("full-rank D fixture did not exercise dense tile storage");
  for(const auto& matrix:reference.block_site_transpose)for(const auto& tile:matrix.tiles)
    if(tile.rank!=0)throw std::runtime_error("full-rank H fixture did not exercise dense SVD fallback");
  const int d=3*n;
  for(int alpha=0;alpha<3;++alpha)for(int row=0;row<d;++row)for(int col=0;col<d;++col){
    const double phase=(row+1)*12.9898+(col+1)*78.233+(alpha+1)*37.719;
    const double noise=std::sin(phase)*43758.5453-std::floor(std::sin(phase)*43758.5453);
    const double expected=(noise+(row==col?1.0:0.0))/
      std::sqrt(reference.masses[row%n]*reference.masses[col%n]);
    if(std::abs(tile_value(reference.block_site_transpose[alpha],row,col)-expected)>2.0e-12*std::max(1.0,std::abs(expected)))
      throw std::runtime_error("full-rank dense H fallback changed an entry after write/readback");
  }
  std::ifstream serialized(out,std::ios::binary|std::ios::ate);
  const std::uint64_t actual_bytes=static_cast<std::uint64_t>(serialized.tellg());
  if(!serialized||actual_bytes!=estimate_rpmd_ja_qnep_v3_max_serialized_bytes(reference))
    throw std::runtime_error("dense H qNEP v3 output did not meet its exact worst-rank serialized-size bound");
  clean(raw);clean(kernel);clean(out);
}

void run_failure(const std::string& base,const std::string& suffix,const int negative_axis,const int minimum_minor,const double expected_rho,const int n=3,const bool multi_spectrum=false)
{
  const std::string stem=base+suffix,raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja",failure=out+".failure.txt";
  make_input(raw,kernel,n,1.0e-10,1,-1.0,0.0,negative_axis,multi_spectrum);bool rejected=false,validated=false;std::string reason;
  const RpmdJAModeValidator validator=[&](const RpmdJAReference&,const std::vector<RpmdJADiagnosticMode>& modes){
    if(modes.empty())throw std::runtime_error("mode validator received no Ritz vectors");
    for(const auto& mode:modes){
      if(mode.mass_weighted_direction.size()!=static_cast<std::size_t>(3*n)||!std::isfinite(mode.eigenvalue)||!std::isfinite(mode.residual))throw std::runtime_error("Ritz mode vector metadata invalid");
      double norm2=0.0;for(double x:mode.mass_weighted_direction)norm2+=x*x;
      if(std::abs(norm2-1.0)>1.0e-10)throw std::runtime_error("Ritz vector is not unit mass-weighted norm");
      for(int axis=0;axis<3;++axis){double overlap=0.0;for(int i=0;i<n;++i)overlap+=std::sqrt((1.0+0.07*i)/(n+0.07*n*(n-1)/2.0))*mode.mass_weighted_direction[axis*n+i];if(std::abs(overlap)>1.0e-8)throw std::runtime_error("Ritz vector retained translation component");}
    }
    const int expected_axis=multi_spectrum?0:negative_axis;
    if(expected_axis>=0){double outside=0.0;for(int axis=0;axis<3;++axis)if(axis!=expected_axis)for(int i=0;i<n;++i)outside+=modes[0].mass_weighted_direction[axis*n+i]*modes[0].mass_weighted_direction[axis*n+i];if(outside>1.0e-12)throw std::runtime_error("Ritz vector was not restored to original Cartesian axes");}
    validated=true;return std::string("\nmode_validation_test: actual unit translation-free original-basis Ritz vectors received\n");
  };
  try{prepare_rpmd_ja_qnep_reference(raw,out,kernel,validator);}catch(const std::exception& e){reason=e.what();rejected=reason.find("not positive definite")!=std::string::npos;}
  std::ifstream report(failure);std::string text((std::istreambuf_iterator<char>(report)),std::istreambuf_iterator<char>());
  const std::string marker="leading_minor_1based: ";const std::size_t at=text.find(marker);int minor=0;if(at!=std::string::npos)minor=std::atoi(text.c_str()+at+marker.size());
  const std::string rho_marker="rho_eV_per_A2_per_amu: ";const std::size_t rho_at=text.find(rho_marker);const double rho=rho_at==std::string::npos?0.0:std::strtod(text.c_str()+rho_at+rho_marker.size(),nullptr);
  const std::string eigen_marker="ritz_1_eV_per_A2_per_amu: ";const std::size_t eigen_at=text.find(eigen_marker);const double eigen=eigen_at==std::string::npos?0.0:std::strtod(text.c_str()+eigen_at+eigen_marker.size(),nullptr);
  const std::string residual_marker="ritz_1_residual_norm: ";const std::size_t residual_at=text.find(residual_marker);const double residual=residual_at==std::string::npos?1.0:std::strtod(text.c_str()+residual_at+residual_marker.size(),nullptr);
  const std::string frequency_marker="ritz_1_imaginary_frequency_THz: ";const std::size_t frequency_at=text.find(frequency_marker);const double frequency=frequency_at==std::string::npos?0.0:std::strtod(text.c_str()+frequency_at+frequency_marker.size(),nullptr);
  const std::string basis_marker="low_spectrum_basis_dimension: ";const std::size_t basis_at=text.find(basis_marker);const int basis=basis_at==std::string::npos?0:std::atoi(text.c_str()+basis_at+basis_marker.size());
  const double expected_eigenvalue=multi_spectrum?-1.0:-1.0e-10;
  const double expected_frequency=std::sqrt(std::abs(expected_eigenvalue))*1000.0/(2.0*std::acos(-1.0)*10.18051);
  std::ifstream raw_check(raw),out_check(out),sidecar_check(out+".stability");
  const std::string expected_method=n<=10?"exact_dense_small_projected_matrix":"bounded_full_reorthogonalization_Lanczos";
  if(!rejected||!validated||text.find("mode_validation_test: actual unit translation-free original-basis Ritz vectors received")==std::string::npos||!raw_check.good()||out_check.good()||sidecar_check.good()||text.empty()||minor<minimum_minor||text.find("candidate_direction_status: available")==std::string::npos||(std::isfinite(expected_rho)&&(!std::isfinite(rho)||std::abs(rho-expected_rho)>1.0e-20))||
     text.find("low_spectrum_method: "+expected_method)==std::string::npos||text.find("low_spectrum_status: unavailable")!=std::string::npos||
     !std::isfinite(eigen)||!std::isfinite(residual)||!std::isfinite(frequency)||std::abs(eigen-expected_eigenvalue)>(multi_spectrum?1.0e-8:1.0e-15)||residual>(multi_spectrum?1.01e-8:1.0e-14)||std::abs(frequency-expected_frequency)>1.0e-8||text.find("ritz_1_negative_sign_resolved: yes")==std::string::npos||
     (multi_spectrum&&basis<=2))
    throw std::runtime_error("qNEP Cholesky failure diagnostic fixture did not preserve input or report a negative candidate direction");
  clean(raw);clean(kernel);clean(out);
}

void run_zero(const std::string& base)
{
  const std::string stem=base+"_zero",raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja",failure=out+".failure.txt";
  make_input(raw,kernel,3,0.0);bool rejected=false;try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}catch(const std::exception& e){rejected=std::string(e.what()).find("not positive definite")!=std::string::npos;}
  std::ifstream report(failure);std::string text((std::istreambuf_iterator<char>(report)),std::istreambuf_iterator<char>());
  const std::string marker="rho_eV_per_A2_per_amu: ";const std::size_t at=text.find(marker);const double rho=at==std::string::npos?1.0:std::strtod(text.c_str()+at+marker.size(),nullptr);
  const std::string eigen_marker="ritz_1_eV_per_A2_per_amu: ";const std::size_t eigen_at=text.find(eigen_marker);const double eigen=eigen_at==std::string::npos?1.0:std::strtod(text.c_str()+eigen_at+eigen_marker.size(),nullptr);
  const std::string residual_marker="ritz_1_residual_norm: ";const std::size_t residual_at=text.find(residual_marker);const double residual=residual_at==std::string::npos?1.0:std::strtod(text.c_str()+residual_at+residual_marker.size(),nullptr);
  std::ifstream raw_check(raw),out_check(out),sidecar_check(out+".stability");
  if(!rejected||!raw_check.good()||out_check.good()||sidecar_check.good()||text.find("candidate_direction_status: available")==std::string::npos||!std::isfinite(rho)||std::abs(rho)>1.0e-20||
     text.find("low_spectrum_status: EXACT_DENSE_SMALL")==std::string::npos||!std::isfinite(eigen)||!std::isfinite(residual)||std::abs(eigen)>1.0e-20||residual>1.0e-20)
    throw std::runtime_error("zero qNEP projected Hessian failure diagnostic did not report a zero candidate direction");clean(raw);clean(kernel);clean(out);
}

void run_zero_h_tiles(const std::string& base)
{
  const std::string stem=base+"_zero_h_tiles",raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja";
  constexpr int n=260;
  make_input(raw,kernel,n,1.0e-10,4,-1.0,0.0,-1,false,nullptr,nullptr,false,true);
  prepare_rpmd_ja_qnep_reference(raw,out,kernel);
  const RpmdJAReference reference=read_rpmd_ja_reference(out);const int d=3*reference.number_of_atoms;
  for(const auto& matrix:reference.block_site_transpose)for(const auto& tile:matrix.tiles){
    if(tile.rank!=1||tile.left.size()!=static_cast<std::size_t>(tile.rows)||
       tile.right.size()!=static_cast<std::size_t>(tile.columns)||
       tile.left.size()+tile.right.size()>=static_cast<std::size_t>(tile.rows)*tile.columns)
      throw std::runtime_error("exact zero H tile did not use the compact rank-one representation");
    for(double value:tile.left)if(value!=0.0)throw std::runtime_error("zero H tile left factor changed");
    for(double value:tile.right)if(value!=0.0)throw std::runtime_error("zero H tile right factor changed");
  }
  std::vector<double> action(static_cast<std::size_t>(d),0.0);
  for(int row=0;row<d;++row)for(int col=0;col<d;++col)action[row]+=tile_value(reference.block_site_transpose[0],row,col)*(col+1.0);
  for(double value:action)if(value!=0.0)throw std::runtime_error("readback action changed for compact zero H tile");
  clean(raw);clean(kernel);clean(out);
}

void run_ill_conditioned_h_tile_edges(const std::string& base)
{
  constexpr int n=260;const int d=3*n;constexpr double h_scale=1.0e-8,h_noise=5.0e-8;
  const std::string raw=base+"_ill_conditioned_h.qraw",kernel=base+"_ill_conditioned_h.kernel",out=base+"_ill_conditioned_h.ja";
  make_input(raw,kernel,n,1.0e-10,4,-1.0,0.0,-1,false,nullptr,nullptr,false,false,h_scale,h_noise);
  prepare_rpmd_ja_qnep_reference(raw,out,kernel);
  const RpmdJAReference reference=read_rpmd_ja_reference(out);
  bool shape_128x128=false,shape_128x12=false,shape_12x128=false,shape_12x12=false;
  for(int alpha=0;alpha<3;++alpha){
    const auto& matrix=reference.block_site_transpose[alpha];
    if(reference.block_relative_residual[alpha+1]>1.0e-8)
      throw std::runtime_error("ill-conditioned H block error exceeded the stored SVD tolerance");
    for(const auto& tile:matrix.tiles){
      shape_128x128=shape_128x128||(tile.rows==128&&tile.columns==128);
      shape_128x12=shape_128x12||(tile.rows==128&&tile.columns==12);
      shape_12x128=shape_12x128||(tile.rows==12&&tile.columns==128);
      shape_12x12=shape_12x12||(tile.rows==12&&tile.columns==12);
      double error2=0.0,norm2=0.0;
      for(int x=0;x<tile.rows;++x)for(int y=0;y<tile.columns;++y){
        const int row=tile.row+x,column=tile.column+y;
        const double left=1.0+0.01*std::sin((row+1)*(alpha+1));
        const double right=1.0+0.01*std::cos((column+1)*(alpha+2));
        const double phase=(row+1)*12.9898+(column+1)*78.233+(alpha+1)*37.719;
        const double noise=std::sin(phase)*43758.5453-std::floor(std::sin(phase)*43758.5453);
        const double expected=h_scale*(left*right+h_noise*noise)/
          std::sqrt(reference.masses[row%n]*reference.masses[column%n]);
        const double actual=tile_value(matrix,row,column),difference=actual-expected;
        error2+=difference*difference;norm2+=expected*expected;
      }
      if(!std::isfinite(error2)||std::sqrt(error2/std::max(norm2,1.0e-300))>1.0e-8)
        throw std::runtime_error("ill-conditioned H SVD tile failed reconstruction/readback tolerance");
    }
  }
  if(!shape_128x128||!shape_128x12||!shape_12x128||!shape_12x12)
    throw std::runtime_error("ill-conditioned H fixture missed one of the four SVD tile shapes");
  clean(raw);clean(kernel);clean(out);
}

#ifdef GPUMD_RPMD_JA_QNEP_PREPARE_FAULT_INJECTION_TEST
void run_svd_fault_injection_regressions(const std::string& base)
{
  const std::string stem=base+"_svd_fault",raw=stem+".qraw",kernel=stem+".kernel";
  make_input(raw,kernel,3,1.0e-10,4);
  const std::vector<char> raw_input_before=read_file_bytes(raw);
  const std::string unrelated=stem+"_unrelated.keep";
  {std::ofstream file(unrelated,std::ios::binary);file<<"unrelated sentinel";if(!file)throw std::runtime_error("cannot create unrelated cleanup sentinel");}
  const auto raw_unchanged=[&](){return raw_input_before==read_file_bytes(raw);};
  const std::vector<char> unrelated_before=read_file_bytes(unrelated);
  const auto unrelated_unchanged=[&](){return unrelated_before==read_file_bytes(unrelated);};
  const auto no_published_artifacts=[](const std::string& path){
    return !exists(path)&&!exists(path+".stability")&&!exists(path+".tmp")&&
      !exists(path+".stability.tmp")&&!exists(path+".failure.txt");
  };
  auto expect_failure=[&](const char* suffix,const rpmd_ja_qnep_prepare_test::SvdFaultInjection fault,
                          const std::vector<std::string>& expected){
    const std::string out=stem+suffix+".ja";std::string message;
    rpmd_ja_qnep_prepare_test::next_svd_fault=fault;
    try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}
    catch(const std::exception& error){message=error.what();}
    rpmd_ja_qnep_prepare_test::next_svd_fault=rpmd_ja_qnep_prepare_test::SvdFaultInjection::None;
    if(message.empty()||!std::all_of(expected.begin(),expected.end(),[&](const std::string& value){return message.find(value)!=std::string::npos;})||
       !no_published_artifacts(out)||!raw_unchanged()||!unrelated_unchanged())
      throw std::runtime_error(std::string("SVD injected hard failure did not classify/clean up correctly: ")+suffix+"; message="+message);
    clean(out);
  };
  expect_failure("_api_unwritten",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInvalidValueUnwrittenInfo,
    {"CUSOLVER_STATUS_INVALID_VALUE","CUDA state: cudaSuccess","devInfo: not written","H direction 0","tile row 0","tile column 0","lwork="});
  expect_failure("_api_written",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInvalidValueWrittenInfo,
    {"CUSOLVER_STATUS_INVALID_VALUE","devInfo: -7"});
  expect_failure("_api_cuda_unavailable",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInvalidValueCudaStatusUnavailable,
    {"CUSOLVER_STATUS_INVALID_VALUE","CUDA state: cudaErrorInvalidValue","devInfo: unavailable (CUDA state is not safely readable)"});
  expect_failure("_negative_info",rpmd_ja_qnep_prepare_test::SvdFaultInjection::NegativeDevInfo,
    {"devInfo=-2","invalid parameter index"});
  expect_failure("_internal_negative_info",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInternalErrorNegativeInfo,
    {"CUSOLVER_STATUS_INTERNAL_ERROR","devInfo: -2"});
  expect_failure("_internal_cuda_unavailable",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInternalErrorCudaStatusUnavailable,
    {"CUSOLVER_STATUS_INTERNAL_ERROR","CUDA state: cudaErrorInvalidValue","devInfo: unavailable (CUDA state is not safely readable)"});
  expect_failure("_nan_factor",rpmd_ja_qnep_prepare_test::SvdFaultInjection::NonFiniteFactor,
    {"SVD returned non-finite or invalid factors"});
  expect_failure("_nan_reconstruction",rpmd_ja_qnep_prepare_test::SvdFaultInjection::NonFiniteReconstruction,
    {"SVD reconstruction produced a non-finite error"});

  const auto expect_exact_dense_readback=[&](const char* suffix,const rpmd_ja_qnep_prepare_test::SvdFaultInjection fault){
    const std::string out=stem+suffix+".ja";
    rpmd_ja_qnep_prepare_test::next_svd_fault=fault;
    prepare_rpmd_ja_qnep_reference(raw,out,kernel);
    rpmd_ja_qnep_prepare_test::next_svd_fault=rpmd_ja_qnep_prepare_test::SvdFaultInjection::None;
    const RpmdJAReference reference=read_rpmd_ja_reference(out);
    if(!exists(out+".stability")||!reference.stability_checked||reference.block_dynamical.tiles.empty()||
       reference.block_site_transpose[0].tiles.size()!=1||reference.block_site_transpose[0].tiles[0].rank!=0||
       reference.block_relative_residual[1]!=0.0||!raw_unchanged()||!unrelated_unchanged())
      throw std::runtime_error(std::string("SVD exact dense fallback failed to retain its certificate/readback: ")+suffix);
    verify_matrices(reference,1.0e-10);
    clean(out);
  };
  expect_exact_dense_readback("_positive_info",rpmd_ja_qnep_prepare_test::SvdFaultInjection::PositiveDevInfo);
  expect_exact_dense_readback("_corrupt_reconstruction",rpmd_ja_qnep_prepare_test::SvdFaultInjection::CorruptReconstruction);
  const auto expect_api_error_dense_downgrade=[&](const char* suffix,
      const rpmd_ja_qnep_prepare_test::SvdFaultInjection fault){
    const std::string out=stem+suffix+".ja";
    rpmd_ja_qnep_prepare_test::next_svd_fault=fault;
    prepare_rpmd_ja_qnep_reference(raw,out,kernel);
    rpmd_ja_qnep_prepare_test::next_svd_fault=rpmd_ja_qnep_prepare_test::SvdFaultInjection::None;
    const RpmdJAReference reference=read_rpmd_ja_reference(out);
    bool all_dense=true;
    for(const auto& matrix:reference.block_site_transpose)for(const auto& tile:matrix.tiles)all_dense=all_dense&&tile.rank==0;
    if(!exists(out+".stability")||!reference.stability_checked||!all_dense||
       std::any_of(reference.block_relative_residual+1,reference.block_relative_residual+4,
         [](double residual){return residual!=0.0;})||!raw_unchanged()||!unrelated_unchanged())
      throw std::runtime_error(std::string("SVD API exact downgrade did not disable later compression or retain its certificate: ")+suffix);
    verify_matrices(reference,1.0e-10);
    clean(out);
  };
  expect_api_error_dense_downgrade("_internal_dense",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiInternalErrorUnwrittenInfo);
  expect_api_error_dense_downgrade("_execution_dense",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiExecutionFailedZeroInfo);
  expect_api_error_dense_downgrade("_unsupported_dense",rpmd_ja_qnep_prepare_test::SvdFaultInjection::ApiNotSupportedUnwrittenInfo);
  if(!raw_unchanged()||!unrelated_unchanged())throw std::runtime_error("SVD fault regressions modified unrelated input files");
  clean(raw);clean(kernel);std::remove(unrelated.c_str());
}
#endif

void run_small_h_norm_checks(const std::string& base)
{
  const std::string kernel=base+"_small_h.kernel";
  const std::string small_raw=base+"_small_h.qraw",small_out=base+"_small_h.ja";
  make_input(small_raw,kernel,3,1.0e-10,4,-1.0,0.0,-1,false,nullptr,nullptr,false,false,1.0e-8);
  prepare_rpmd_ja_qnep_reference(small_raw,small_out,kernel);
  const RpmdJAReference small=read_rpmd_ja_reference(small_out);
  for(int alpha=0;alpha<3;++alpha){const double value=tile_value(small.block_site_transpose[alpha],0,0);
    if(!(std::isfinite(value)&&std::abs(value)>1.0e-10&&std::abs(value)<1.0e-6))
      throw std::runtime_error("scaled small-V/C fixture did not retain its nonzero native H");}
  clean(small_raw);clean(small_out);

  const std::string bad_small_raw=base+"_small_h_bad_norm.qraw",bad_small_out=base+"_small_h_bad_norm.ja";
  make_input(bad_small_raw,kernel,3,1.0e-10,4,-1.0,0.0,-1,false,nullptr,nullptr,false,false,1.0e-8);
  {std::fstream file(bad_small_raw,std::ios::binary|std::ios::in|std::ios::out);if(!file)throw std::runtime_error("cannot reopen scaled compact qraw norm fixture");
    double compact[6];file.seekg(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.read(reinterpret_cast<char*>(compact),sizeof(compact));
    if(!file||!(compact[3]>0.0))throw std::runtime_error("scaled compact qraw fixture has no positive H norm");compact[3]*=1.1;
    file.seekp(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.write(reinterpret_cast<char*>(compact),sizeof(compact));
    file.close();if(!file)throw std::runtime_error("cannot close scaled compact qraw norm fixture");}
  bool rejected_small=false;try{prepare_rpmd_ja_qnep_reference(bad_small_raw,bad_small_out,kernel);}
  catch(const std::exception& e){rejected_small=std::string(e.what()).find("stored native-H norm")!=std::string::npos;}
  if(!rejected_small)throw std::runtime_error("qNEP prepare accepted a same-scale small-H footer norm mismatch");
  clean(bad_small_raw);clean(bad_small_out);

  const std::string zero_raw=base+"_zero_h_bad_norm.qraw",zero_out=base+"_zero_h_bad_norm.ja";
  make_input(zero_raw,kernel,3,1.0e-10,4,-1.0,0.0,-1,false,nullptr,nullptr,false,true);
  {std::fstream file(zero_raw,std::ios::binary|std::ios::in|std::ios::out);if(!file)throw std::runtime_error("cannot reopen zero-H compact qraw fixture");
    double compact[6];file.seekg(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.read(reinterpret_cast<char*>(compact),sizeof(compact));
    if(!file||compact[3]!=0.0)throw std::runtime_error("zero-H compact qraw fixture norm is not zero");compact[3]=1.0e-13;
    file.seekp(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.write(reinterpret_cast<char*>(compact),sizeof(compact));
    file.close();if(!file)throw std::runtime_error("cannot close zero-H compact qraw fixture");}
  bool rejected_zero=false;try{prepare_rpmd_ja_qnep_reference(zero_raw,zero_out,kernel);}
  catch(const std::exception& e){rejected_zero=std::string(e.what()).find("stored native-H norm")!=std::string::npos;}
  if(!rejected_zero)throw std::runtime_error("qNEP prepare accepted a nonzero footer norm for zero H");
  clean(zero_raw);clean(kernel);clean(zero_out);
}

void patch_additive_source_geometry(const std::string& raw,const int n,const std::uint32_t raw_version)
{
  const int d=3*n;const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::streamoff header=8+4+4+4+4+8+8+8+8+4+4+8+sizeof(layout);
  const std::streamoff positions=header+18*8+12+n*4+n*8;
  const std::streamoff vstart=positions+d*8+8+n*8+d*8+9*n*8;
  const std::streamoff force_start=vstart-9*n*8-d*8, cstart=vstart+static_cast<std::streamoff>(d)*n*8;
  const std::streamoff kstart=cstart+static_cast<std::streamoff>(3)*d*d*8;
  const std::streamoff coarse_v=kstart+static_cast<std::streamoff>(d)*d*8;
  const std::streamoff coarse_c=coarse_v+static_cast<std::streamoff>(d)*n*8;
  std::fstream file(raw,std::ios::binary|std::ios::in|std::ios::out);
  if(!file)throw std::runtime_error("cannot patch additive fixture raw");
  std::vector<double> xyz(d),zero_v(static_cast<std::size_t>(d)*n,0.0),c(static_cast<std::size_t>(3)*d*d);
  const std::streamoff cell_start=header;const double cell[18]={10.,.4,0.,0.,11.,.3,0.,0.,12.,.1,-.4/110.,.4*.3/1320.,0.,1./11.,-.3/132.,0.,0.,1./12.};
  for(int i=0;i<n;++i){xyz[i]=0.2*i;xyz[n+i]=0.1*i*i;xyz[2*n+i]=0.05*i;}
  for(int alpha=0;alpha<3;++alpha)for(int col=0;col<d;++col)for(int row=0;row<d;++row){const double left=1.+.01*std::sin((row+1)*(alpha+1)),right=1.+.01*std::cos((col+1)*(alpha+2));c[static_cast<std::size_t>(alpha)*d*d+static_cast<std::size_t>(col)*d+row]=left*right;}
  file.seekp(cell_start);file.write(reinterpret_cast<const char*>(cell),sizeof(cell));file.seekp(positions);array(file,xyz);file.seekp(force_start);array(file,std::vector<double>(d,0.0));
  file.seekp(vstart);array(file,zero_v);file.seekp(cstart);array(file,c);
  if(raw_version==4){
    const std::streamoff footer=kstart+static_cast<std::streamoff>(d)*d*8;
    double stats[24]={};stats[16]=.01;stats[17]=3.;stats[21]=stats[22]=stats[23]=1.;file.seekp(footer);file.write(reinterpret_cast<char*>(stats),sizeof(stats));
  }else{
    file.seekp(coarse_v);array(file,zero_v);file.seekp(coarse_c);array(file,c);
    const std::streamoff footer=coarse_c+static_cast<std::streamoff>(3)*d*d*8;
    double stats[18]={};stats[16]=.01;stats[17]=3.;file.seekp(footer);file.write(reinterpret_cast<char*>(stats),sizeof(stats));
  }
  file.close();
  if(!file)throw std::runtime_error("failed patching additive raw fixture");
}

void write_additive_package(const std::string& path,const std::string& raw,const int n,const std::uint32_t version=1)
{
  std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot create additive fixture package");
  const char magic[8]={'G','P','J','A','A','D','D','1'};const std::uint32_t endian=0x01020304;
  const int beads=8;const std::uint64_t source=rpmd_ja_model_fingerprint(raw),training=4,validation=2;
  const double temperature=300.,epsilon=1e-8,response=version==7?.05:0.,budget=.02;
  out.write(magic,8);put(out,version);put(out,endian);put(out,n);put(out,beads);put(out,source);
  put(out,temperature);put(out,epsilon);put(out,response);put(out,budget);put(out,training);put(out,validation);
  if(version==7){put(out,std::uint32_t(2));put(out,std::uint32_t(1));put(out,std::uint32_t(1));}
  const int d=3*n,q=6,z=2;
  for(int i=0;i<n;++i){put(out,z);const int neighbors[2]={(i+1)%n,(i+n-1)%n};
    for(int e=0;e<2;++e){put(out,neighbors[e]);int image[3]={0,0,0};if(e==0&&i==n-1)image[0]=1;if(e==1&&i==0)image[0]=-1;array(out,std::vector<int>(image,image+3));}
    std::vector<double> ell(q,0.0);ell[0]=.1;ell[3]=-.1;array(out,ell);
    if(version>=5)put(out,std::uint32_t(0));
    std::vector<double> b(static_cast<std::size_t>(q)*q,0.0);for(int a=0;a<q;++a)b[static_cast<std::size_t>(a)*q+a]=.02;
    b[0*q+4]=b[4*q+0]=.003;b[1*q+5]=b[5*q+1]=.002;array(out,b);
  }
  out.close();if(!out)throw std::runtime_error("failed writing additive fixture package");(void)d;
}

void verify_additive_reference(const RpmdJAReference& r,const int n,const std::uint32_t version)
{
  const int d=3*n,q=6;double mass_sum=0.0;for(double m:r.masses)mass_sum+=m;
  std::vector<double> kadd(static_cast<std::size_t>(d)*d,0.0),hadd[3];for(auto& h:hadd)h.assign(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<n;++i){const int neighbors[2]={(i+1)%n,(i+n-1)%n};double b[q*q]={};for(int x=0;x<q;++x)b[x*q+x]=.02;b[0*q+4]=b[4*q+0]=.003;b[1*q+5]=b[5*q+1]=.002;
    double xyz[2][3]={{.2*neighbors[0]-.2*i,.1*neighbors[0]*neighbors[0]-.1*i*i,.05*neighbors[0]-.05*i},{.2*neighbors[1]-.2*i,.1*neighbors[1]*neighbors[1]-.1*i*i,.05*neighbors[1]-.05*i}};
    int edge_images[2][3]={{0,0,0},{0,0,0}};if(i==n-1)edge_images[0][0]=1;if(i==0)edge_images[1][0]=-1;
    const double h[3][3]={{10.,.4,0.},{0.,11.,.3},{0.,0.,12.}};
    for(int e=0;e<2;++e)for(int a=0;a<3;++a)for(int c=0;c<3;++c)xyz[e][a]+=h[a][c]*edge_images[e][c];
    for(int x=0;x<q;++x)for(int y=0;y<q;++y){const int ex=x/3,ey=y/3,ax=x%3,ay=y%3,ix[2]={ax*n+neighbors[ex],ax*n+i},iy[2]={ay*n+neighbors[ey],ay*n+i};const double signs[2]={1.,-1.};for(int u=0;u<2;++u)for(int v=0;v<2;++v)kadd[static_cast<std::size_t>(ix[u])*d+iy[v]]+=signs[u]*b[x*q+y]*signs[v];}
    for(int e=0;e<2;++e)for(int mu=0;mu<3;++mu)for(int y=0;y<q;++y){const int axis=y%3,edge=y/3,disp[2]={axis*n+neighbors[edge],axis*n+i},vel=mu*n+neighbors[e];const double signs[2]={1.,-1.};for(int a=0;a<3;++a)for(int v=0;v<2;++v)hadd[a][static_cast<std::size_t>(disp[v])*d+vel]-=xyz[e][a]*b[(3*e+mu)*q+y]*signs[v];}
  }
  const std::string expected_policy=version==1?"native_reference_transport;finite_temperature_additive_v1;beads=8;derivative=3":
    version==7?"native_reference_transport;finite_temperature_additive_v1;beads=8;derivative=3;use=DIAGNOSTIC;response=FAIL;numerical=PASS":
    "native_reference_transport;internal_mass_com_pullback_v1;finite_temperature_additive_v1;beads=8;derivative=3";
  if(r.mechanical_policy!=expected_policy||r.additive_beads!=8||
     !r.stability_checked||r.stability_certificate!="additive_shifted_frobenius_v1"||!(r.additive_reconstruction_bound<.5*r.additive_epsilon))
    throw std::runtime_error("additive reference policy/certificate mismatch");
  for(int row=0;row<d;++row)for(int col=0;col<d;++col){const int ax=row/n,ay=col/n;const double tr=std::sqrt(r.masses[row%n]/mass_sum),tc=std::sqrt(r.masses[col%n]/mass_sum);const double baseline=ax==ay?1e-10*((row==col?1.0:0.0)-tr*tc):0.0;const double expected=baseline+kadd[static_cast<std::size_t>(row)*d+col]/std::sqrt(r.masses[row%n]*r.masses[col%n]);if(std::abs(tile_value(r.block_dynamical,row,col)-expected)>2e-9)throw std::runtime_error("additive K readback mismatch");}
  bool nonsymmetric=false;
  for(int a=0;a<3;++a)for(int row=0;row<d;++row)for(int col=0;col<d;++col){const double left=1.+.01*std::sin((row+1)*(a+1)),right=1.+.01*std::cos((col+1)*(a+2));const double expected=left*right/std::sqrt(r.masses[row%n]*r.masses[col%n])+hadd[a][static_cast<std::size_t>(col)*d+row]/std::sqrt(r.masses[row%n]*r.masses[col%n]);const double actual=tile_value(r.block_site_transpose[a],row,col);if(std::abs(actual-expected)>2e-8)throw std::runtime_error("additive transposed H readback mismatch");if(std::abs(actual-tile_value(r.block_site_transpose[a],col,row))>1e-7)nonsymmetric=true;}
  if(!nonsymmetric)throw std::runtime_error("additive H fixture did not exercise non-symmetric cross-axis terms");
}

void run_additive(const std::string& base,const std::uint32_t version=1)
{
  const int n=3;const std::string stem=base+"_additive",raw=stem+".qraw",kernel=stem+".kernel",pack=stem+".gpjaadd",out=stem+".ja";
  make_input(raw,kernel,n,1.0e-10,4);patch_additive_source_geometry(raw,n,4);write_additive_package(pack,raw,n,version);
  prepare_rpmd_ja_qnep_reference(raw,out,kernel,{},pack);const RpmdJAReference prepared=read_rpmd_ja_reference(out);
  verify_additive_reference(prepared,n,version);
  if(version==7&&(!rpmd_ja_reference_policy_is_diagnostic(prepared.mechanical_policy)||
     rpmd_ja_reference_policy_response_status(prepared.mechanical_policy)!="FAIL"||
     rpmd_ja_reference_policy_numerical_status(prepared.mechanical_policy)!="PASS"))
    throw std::runtime_error("diagnostic additive identity did not survive native qNEP prepare/readback");
  const std::string wrong=stem+"_wrong.gpjaadd",truncated=stem+"_truncated.gpjaadd";std::ifstream in(pack,std::ios::binary);std::vector<char> bytes((std::istreambuf_iterator<char>(in)),{}),wrong_bytes=bytes,valid_bytes=bytes;wrong_bytes[24]^=1;
  {std::ofstream f(wrong,std::ios::binary);f.write(wrong_bytes.data(),wrong_bytes.size());}
  bytes.pop_back();{std::ofstream f(truncated,std::ios::binary);f.write(bytes.data(),bytes.size());}
  for(const std::string& bad:{wrong,truncated}){const std::string failed=bad+".ja";bool rejected=false;try{prepare_rpmd_ja_qnep_reference(raw,failed,kernel,{},bad);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("invalid additive package was accepted");
    if(exists(failed)||exists(failed+".tmp")||exists(failed+".stability")||exists(failed+".stability.tmp"))
      throw std::runtime_error("failed additive prepare left a publishable output or temporary file");
    clean(failed);}
  if(version==7){
    for(const auto& mutation:std::array<std::pair<const char*,std::pair<std::size_t,std::uint32_t>>,3>{{
        {"bad_numerical",{84,2U}},{"bad_response",{88,9U}},{"bad_pass_budget",{88,0U}}}}){
      std::vector<char> damaged=valid_bytes;std::memcpy(damaged.data()+mutation.second.first,&mutation.second.second,sizeof(std::uint32_t));
      const std::string bad=stem+"_"+mutation.first+".gpjaadd",failed=bad+".ja";
      {std::ofstream f(bad,std::ios::binary);f.write(damaged.data(),static_cast<std::streamsize>(damaged.size()));}
      bool rejected=false;try{prepare_rpmd_ja_qnep_reference(raw,failed,kernel,{},bad);}catch(const std::exception&){rejected=true;}
      if(!rejected||exists(failed)||exists(failed+".stability"))
        throw std::runtime_error("invalid diagnostic additive status was accepted or published");
      clean(bad);clean(failed);
    }
    for(const auto status:{2U,3U})for(const double response:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()}){
      std::vector<char> damaged=valid_bytes;std::memcpy(damaged.data()+48,&response,sizeof(response));
      std::memcpy(damaged.data()+88,&status,sizeof(status));
      const std::string bad=stem+"_infinite_response_"+std::to_string(status)+"_"+(response>0?"positive":"negative")+".gpjaadd";
      const std::string failed=bad+".ja";
      {std::ofstream f(bad,std::ios::binary);f.write(damaged.data(),static_cast<std::streamsize>(damaged.size()));}
      bool rejected=false;try{prepare_rpmd_ja_qnep_reference(raw,failed,kernel,{},bad);}catch(const std::exception&){rejected=true;}
      if(!rejected||exists(failed)||exists(failed+".stability"))
        throw std::runtime_error("infinite diagnostic response was accepted or published");
      clean(bad);clean(failed);
    }
  }
  clean(raw);clean(kernel);clean(pack);clean(out);clean(wrong);clean(truncated);
}

void run_raw2_rejections(const std::string& base)
{
  const std::string kernel=base+"_raw2.kernel";
  const auto rejected=[&](const std::string& suffix,const double marker,const double identity,const std::uint32_t version) {
    const std::string raw=base+suffix+".qraw",out=base+suffix+".ja";
    make_input(raw,kernel,3,1.0e-10,version,marker,identity);
    bool failed=false;try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}catch(const std::exception&){failed=true;}
    if(!failed)throw std::runtime_error("qNEP raw v2 accepted an invalid stats marker or site-JVP identity");
    clean(raw);clean(kernel);clean(out);
  };
  rejected("_bad_marker_v2",1.0,0.0,2);
  rejected("_bad_identity_v2",2.0,1.1e-4,2);
  rejected("_bad_marker_v1",2.0,0.0,1);
  rejected("_bad_marker_v3",2.0,0.0,3);
  rejected("_bad_identity_v3",3.0,1.1e-4,3);
  const std::string bad_v4=base+"_bad_transport_v4.qraw",bad_v4_out=base+"_bad_transport_v4.ja";
  make_input(bad_v4,kernel,3,1.0e-10,4);
  {std::fstream file(bad_v4,std::ios::binary|std::ios::in|std::ios::out);file.seekp(-6*static_cast<std::streamoff>(sizeof(double)),std::ios::end);double stats[6]={1,0,0,1,1,1};file.write(reinterpret_cast<char*>(stats),sizeof(stats));}
  bool rejected_v4=false;try{prepare_rpmd_ja_qnep_reference(bad_v4,bad_v4_out,kernel);}catch(const std::exception&){rejected_v4=true;}
  if(!rejected_v4)throw std::runtime_error("qNEP raw v4 accepted a failed compact transport statistic");
  clean(bad_v4);clean(kernel);clean(bad_v4_out);
  const std::string bad_v4_norm=base+"_bad_transport_norm_v4.qraw",bad_v4_norm_out=base+"_bad_transport_norm_v4.ja";
  make_input(bad_v4_norm,kernel,3,1.0e-10,4);
  {std::fstream file(bad_v4_norm,std::ios::binary|std::ios::in|std::ios::out);if(!file)throw std::runtime_error("cannot reopen compact qraw norm fixture");
    double compact[6];file.seekg(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.read(reinterpret_cast<char*>(compact),sizeof(compact));
    if(!file)throw std::runtime_error("cannot read compact qraw norm fixture");compact[3]+=1.0;file.seekp(-static_cast<std::streamoff>(sizeof(compact)),std::ios::end);file.write(reinterpret_cast<char*>(compact),sizeof(compact));
    file.close();if(!file)throw std::runtime_error("cannot close compact qraw norm fixture");}
  bool rejected_norm=false;try{prepare_rpmd_ja_qnep_reference(bad_v4_norm,bad_v4_norm_out,kernel);}
  catch(const std::exception& e){rejected_norm=std::string(e.what()).find("stored native-H norm")!=std::string::npos;}
  if(!rejected_norm)throw std::runtime_error("qNEP prepare trusted a footer norm that disagrees with reconstructed H");
  clean(bad_v4_norm);clean(kernel);clean(bad_v4_norm_out);
}
} // namespace

#ifndef GPUMD_RPMD_JA_QNEP_PREPARE_TEST_LIBRARY
int main(int argc,char** argv)
{
  try{const bool integration_smoke=argc>1&&std::string(argv[1])=="--integration-smoke";
    const std::string base=integration_smoke?"rpmd_ja_qnep_prepare_integration_smoke_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()):(argc>1?argv[1]:"rpmd_ja_qnep_prepare_cuda_test");
    if(integration_smoke){run_additive(base,1);run_additive(base+"_internal_com",2);run_additive(base+"_diagnostic",7);return 0;}
    run_positive(base,2,1);run_positive(base,3,1);run_positive(base,3,2);run_positive(base,3,3);run_positive(base,3,4);run_positive(base,44,1);run_positive(base,260,4);
    run_dense_h_upper_bound(base);run_zero_h_tiles(base);run_ill_conditioned_h_tile_edges(base);run_small_h_norm_checks(base);
    run_failure(base,"_negative_first",0,1,-1.0e-10);run_failure(base,"_negative_late",2,3,-1.0e-10);
    run_failure(base,"_negative_lanczos",0,1,std::numeric_limits<double>::quiet_NaN(),44,true);run_zero(base);run_raw2_rejections(base);
    run_additive(base);run_additive(base+"_diagnostic",7);return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"qNEP prepare CUDA fixture failed: %s\n",e.what());return 1;}
}
#endif
