#include "measure/rpmd_ja_qnep_prepare.cuh"
#include "measure/rpmd_ja_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
template <typename T> void put(std::ostream& out,const T& x){out.write(reinterpret_cast<const char*>(&x),sizeof(x));}
template <typename T> void array(std::ostream& out,const std::vector<T>& x){if(!x.empty())out.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T));}

void make_input(const std::string& raw,const std::string& kernel,const int n,const double stiffness,
                const std::uint32_t version=1,const double stats_marker=-1.0,const double identity_abs=0.0,const int negative_axis=-1,const bool multi_spectrum=false)
{
  const int d=3*n;const double temperature=300.0,step=0.01;
  std::vector<double> mass(n),position(d,0.0),v(static_cast<std::size_t>(d)*n),vc(v.size()),c(static_cast<std::size_t>(3)*d*d),k(static_cast<std::size_t>(d)*d),coarse_c(c.size());
  std::vector<int> types(n,0),pbc(3,1);double mass_sum=0.0;
  for(int i=0;i<n;++i){mass[i]=1.0+0.07*i;mass_sum+=mass[i];}
  std::vector<std::vector<double>> complement;
  if(multi_spectrum){
    const double inv_mass=1.0/std::sqrt(mass_sum);std::vector<double> translation(n);for(int i=0;i<n;++i)translation[i]=std::sqrt(mass[i])*inv_mass;
    for(int q=0;q<n-1;++q){std::vector<double> u(n,0.0);u[q]=1.0;double dot=0.0;for(int i=0;i<n;++i)dot+=u[i]*translation[i];for(int i=0;i<n;++i)u[i]-=dot*translation[i];for(const auto& vq:complement){dot=0.0;for(int i=0;i<n;++i)dot+=u[i]*vq[i];for(int i=0;i<n;++i)u[i]-=dot*vq[i];}double norm=0.0;for(double x:u)norm+=x*x;norm=std::sqrt(norm);for(double& x:u)x/=norm;complement.push_back(u);}
  }
  for(int row=0;row<d;++row)for(int col=0;col<d;++col){const int ar=row%n,ac=col%n;if(row/n==col/n){double dm=0.0;if(multi_spectrum){for(int q=0;q<n-1;++q){const double value=(row/n)*10.0+ (q==0?-1.0:static_cast<double>(q));dm+=value*complement[q][ar]*complement[q][ac];}}else{const double tr=std::sqrt(mass[ar]/mass_sum),tc=std::sqrt(mass[ac]/mass_sum);const double axis_stiffness=negative_axis<0?stiffness:(row/n==negative_axis?-1.0e-10:1.0e-10);dm=axis_stiffness*((row==col?1.0:0.0)-tr*tc);}k[static_cast<std::size_t>(col)*d+row]=dm*std::sqrt(mass[ar]*mass[ac]);}}
  for(int row=0;row<d;++row)for(int atom=0;atom<n;++atom){const std::size_t ix=static_cast<std::size_t>(row)*n+atom;v[ix]=0.03*std::sin((row+1)*(atom+2));vc[ix]=version==2?v[ix]:0.97*v[ix];}
  for(int alpha=0;alpha<3;++alpha)for(int col=0;col<d;++col)for(int row=0;row<d;++row){const int site=row%n,atom=col%n;const double left=1.0+0.01*std::sin((row+1)*(alpha+1));const double right=1.0+0.01*std::cos((col+1)*(alpha+2));const double target=left*right;double fine=target,coarse=target;if(col/n==alpha){fine+=v[static_cast<std::size_t>(row)*n+atom];coarse+=vc[static_cast<std::size_t>(row)*n+atom];if(site==atom)for(int a=0;a<n;++a){fine-=v[static_cast<std::size_t>(row)*n+a];coarse-=vc[static_cast<std::size_t>(row)*n+a];}}const std::size_t ix=static_cast<std::size_t>(alpha)*d*d+static_cast<std::size_t>(col)*d+row;c[ix]=fine;coarse_c[ix]=coarse;}
  std::ofstream out(raw,std::ios::binary);if(!out)throw std::runtime_error("cannot create raw fixture");
  const char magic[8]={'G','P','J','Q','R','A','W','\0'};const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::uint32_t endian=0x01020304;const int charge=1,pppm=1;const std::uint64_t model=0x12345678,config=0x87654321;const double spacing=0.5;
  out.write(magic,8);put(out,version);put(out,endian);put(out,n);put(out,d);put(out,temperature);put(out,step);put(out,model);put(out,config);put(out,charge);put(out,pppm);put(out,spacing);out.write(layout,sizeof(layout));
  double cell[18]={1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1};out.write(reinterpret_cast<char*>(cell),sizeof(cell));out.write(reinterpret_cast<const char*>(pbc.data()),sizeof(int)*3);
  array(out,types);array(out,mass);array(out,position);const double energy=0.0;std::vector<double> site_energy(n,0),force(d,0),virial(9*n,0);for(int row=0;row<d;++row)for(int atom=0;atom<n;++atom)force[row]-=v[static_cast<std::size_t>(row)*n+atom];put(out,energy);array(out,site_energy);array(out,force);array(out,virial);
  array(out,v);array(out,c);array(out,k);array(out,vc);array(out,coarse_c);double stats[18]={};for(double value:force)stats[12]=std::max(stats[12],std::abs(value));stats[3]=identity_abs;stats[16]=step;stats[17]=stats_marker<0.0?static_cast<double>(version):stats_marker;out.write(reinterpret_cast<char*>(stats),sizeof(stats));out.close();if(!out)throw std::runtime_error("failed writing raw fixture");
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

void run_positive(const std::string& base,const int n,const std::uint32_t version)
{
  const std::string stem=base+"_N"+std::to_string(n),raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja";
  make_input(raw,kernel,n,1.0e-10,version);prepare_rpmd_ja_qnep_reference(raw,out,kernel);const RpmdJAReference r=read_rpmd_ja_reference(out);
  if(r.backend!=2||r.number_of_atoms!=n||r.block_dynamical.tiles.empty()||r.block_site_transpose[0].tiles.empty()||!r.stability_checked||!has_compressed_h(r)||!(r.force_balance_residual>1.0e-4))throw std::runtime_error("prepared qNEP v3 fixture failed nonzero-force readback or H compression");
  const std::string expected_policy=version==1?"native_reference_transport":"native_reference_transport;analytic_site_gradient_v1";
  if(r.mechanical_policy!=expected_policy)throw std::runtime_error("prepared qNEP v3 fixture lost its derivative-source marker");
  if(r.block_dynamical.tiles.size()!=static_cast<std::size_t>((3*n+127)/128)*((3*n+127)/128))throw std::runtime_error("D tile grid readback incomplete");
  verify_matrices(r,1.0e-10);
  clean(raw);clean(kernel);clean(out);
}

void run_failure(const std::string& base,const std::string& suffix,const int negative_axis,const int minimum_minor,const double expected_rho,const int n=3,const bool multi_spectrum=false)
{
  const std::string stem=base+suffix,raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja",failure=out+".failure.txt";
  make_input(raw,kernel,n,1.0e-10,1,-1.0,0.0,negative_axis,multi_spectrum);bool rejected=false;std::string reason;try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}catch(const std::exception& e){reason=e.what();rejected=reason.find("not positive definite")!=std::string::npos;}
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
  if(!rejected||!raw_check.good()||out_check.good()||sidecar_check.good()||text.empty()||minor<minimum_minor||text.find("candidate_direction_status: available")==std::string::npos||(std::isfinite(expected_rho)&&(!std::isfinite(rho)||std::abs(rho-expected_rho)>1.0e-20))||
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

void run_raw2_rejections(const std::string& base)
{
  const std::string kernel=base+"_raw2.kernel";
  const auto rejected=[&](const std::string& suffix,const double marker,const double identity) {
    const std::string raw=base+suffix+".qraw",out=base+suffix+".ja";
    make_input(raw,kernel,3,1.0e-10,2,marker,identity);
    bool failed=false;try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}catch(const std::exception&){failed=true;}
    if(!failed)throw std::runtime_error("qNEP raw v2 accepted an invalid stats marker or site-JVP identity");
    clean(raw);clean(kernel);clean(out);
  };
  rejected("_bad_marker",1.0,0.0);
  rejected("_bad_identity",2.0,1.1e-4);
}
} // namespace

int main(int argc,char** argv)
{
  try{const std::string base=argc>1?argv[1]:"rpmd_ja_qnep_prepare_cuda_test";run_positive(base,2,1);run_positive(base,3,1);run_positive(base,3,2);run_positive(base,44,1);run_failure(base,"_negative_first",0,1,-1.0e-10);run_failure(base,"_negative_late",2,3,-1.0e-10);run_failure(base,"_negative_lanczos",0,1,std::numeric_limits<double>::quiet_NaN(),44,true);run_zero(base);run_raw2_rejections(base);return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"qNEP prepare CUDA fixture failed: %s\n",e.what());return 1;}
}
