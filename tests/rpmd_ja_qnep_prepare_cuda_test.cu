#include "measure/rpmd_ja_qnep_prepare.cuh"
#include "measure/rpmd_ja_reference.cuh"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
template <typename T> void put(std::ostream& out,const T& x){out.write(reinterpret_cast<const char*>(&x),sizeof(x));}
template <typename T> void array(std::ostream& out,const std::vector<T>& x){if(!x.empty())out.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(T));}

void make_input(const std::string& raw,const std::string& kernel,const int n,const double stiffness)
{
  const int d=3*n;const double temperature=300.0,step=0.01;
  std::vector<double> mass(n),position(d,0.0),v(static_cast<std::size_t>(d)*n),vc(v.size()),c(static_cast<std::size_t>(3)*d*d),k(static_cast<std::size_t>(d)*d),coarse_c(c.size());
  std::vector<int> types(n,0),pbc(3,1);double mass_sum=0.0;
  for(int i=0;i<n;++i){mass[i]=1.0+0.07*i;mass_sum+=mass[i];}
  for(int row=0;row<d;++row)for(int col=0;col<d;++col){const int ar=row%n,ac=col%n;if(row/n==col/n){const double tr=std::sqrt(mass[ar]/mass_sum),tc=std::sqrt(mass[ac]/mass_sum);const double dm=stiffness*((row==col?1.0:0.0)-tr*tc);k[static_cast<std::size_t>(col)*d+row]=dm*std::sqrt(mass[ar]*mass[ac]);}}
  for(int row=0;row<d;++row)for(int atom=0;atom<n;++atom){const std::size_t ix=static_cast<std::size_t>(row)*n+atom;v[ix]=0.03*std::sin((row+1)*(atom+2));vc[ix]=0.97*v[ix];}
  for(int alpha=0;alpha<3;++alpha)for(int col=0;col<d;++col)for(int row=0;row<d;++row){const int site=row%n,atom=col%n;const double left=1.0+0.01*std::sin((row+1)*(alpha+1));const double right=1.0+0.01*std::cos((col+1)*(alpha+2));const double target=left*right;double fine=target,coarse=target;if(col/n==alpha){fine+=v[static_cast<std::size_t>(row)*n+atom];coarse+=vc[static_cast<std::size_t>(row)*n+atom];if(site==atom)for(int a=0;a<n;++a){fine-=v[static_cast<std::size_t>(row)*n+a];coarse-=vc[static_cast<std::size_t>(row)*n+a];}}const std::size_t ix=static_cast<std::size_t>(alpha)*d*d+static_cast<std::size_t>(col)*d+row;c[ix]=fine;coarse_c[ix]=coarse;}
  std::ofstream out(raw,std::ios::binary);if(!out)throw std::runtime_error("cannot create raw fixture");
  const char magic[8]={'G','P','J','Q','R','A','W','\0'};const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::uint32_t version=1,endian=0x01020304;const int charge=1,pppm=1;const std::uint64_t model=0x12345678,config=0x87654321;const double spacing=0.5;
  out.write(magic,8);put(out,version);put(out,endian);put(out,n);put(out,d);put(out,temperature);put(out,step);put(out,model);put(out,config);put(out,charge);put(out,pppm);put(out,spacing);out.write(layout,sizeof(layout));
  double cell[18]={1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1};out.write(reinterpret_cast<char*>(cell),sizeof(cell));out.write(reinterpret_cast<const char*>(pbc.data()),sizeof(int)*3);
  array(out,types);array(out,mass);array(out,position);const double energy=0.0;std::vector<double> site_energy(n,0),force(d,0),virial(9*n,0);put(out,energy);array(out,site_energy);array(out,force);array(out,virial);
  array(out,v);array(out,c);array(out,k);array(out,vc);array(out,coarse_c);double stats[18]={};stats[16]=step;stats[17]=1.0;out.write(reinterpret_cast<char*>(stats),sizeof(stats));out.close();if(!out)throw std::runtime_error("failed writing raw fixture");
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

void clean(const std::string& p){std::remove(p.c_str());std::remove((p+".tmp").c_str());std::remove((p+".stability").c_str());std::remove((p+".stability.tmp").c_str());}

void run_positive(const std::string& base,const int n)
{
  const std::string stem=base+"_N"+std::to_string(n),raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja";
  make_input(raw,kernel,n,1.0e-10);prepare_rpmd_ja_qnep_reference(raw,out,kernel);const RpmdJAReference r=read_rpmd_ja_reference(out);
  if(r.backend!=2||r.number_of_atoms!=n||r.block_dynamical.tiles.empty()||r.block_site_transpose[0].tiles.empty()||!r.stability_checked||!has_compressed_h(r))throw std::runtime_error("prepared qNEP v3 fixture failed readback or H compression");
  if(r.block_dynamical.tiles.size()!=static_cast<std::size_t>((3*n+127)/128)*((3*n+127)/128))throw std::runtime_error("D tile grid readback incomplete");
  verify_matrices(r,1.0e-10);
  clean(raw);clean(kernel);clean(out);
}

void run_negative(const std::string& base)
{
  const std::string stem=base+"_negative",raw=stem+".qraw",kernel=stem+".kernel",out=stem+".ja";
  make_input(raw,kernel,3,-1.0e-10);bool rejected=false;std::string reason;try{prepare_rpmd_ja_qnep_reference(raw,out,kernel);}catch(const std::exception& e){reason=e.what();rejected=reason.find("not positive definite")!=std::string::npos;}
  if(!rejected)throw std::runtime_error("negative qNEP soft mode was accepted or rejected for a different reason");clean(raw);clean(kernel);clean(out);
}
} // namespace

int main(int argc,char** argv)
{
  try{const std::string base=argc>1?argv[1]:"rpmd_ja_qnep_prepare_cuda_test";run_positive(base,2);run_positive(base,3);run_positive(base,44);run_negative(base);return 0;}
  catch(const std::exception& e){std::fprintf(stderr,"qNEP prepare CUDA fixture failed: %s\n",e.what());return 1;}
}
