#include "../src/measure/rpmd_ja_native_fit.cu"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>

static std::vector<double> diagnostic_expected_r0;
static bool diagnostic_should_throw = false;

Force::Force(void) {}
void diagnose_rpmd_ja_qnep_reference(double, Atom& atom, Box&, Force&, bool)
{
  std::vector<double> x(diagnostic_expected_r0.size());
  atom.position_per_atom.copy_to_host(x.data());
  for (std::size_t i = 0; i < x.size(); ++i) assert(std::abs(x[i] - diagnostic_expected_r0[i]) < 1e-12);
  if (diagnostic_should_throw) throw std::runtime_error("diagnostic test failure");
}
void generate_rpmd_ja_qnep_raw(const std::string&, double, double, const std::string&, Atom&, Box&, Force&, bool) {}
void prepare_rpmd_ja_qnep_reference(const std::string&, const std::string&, const std::string&,
                                    const RpmdJAModeValidator&, const std::string&) {}
RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom&, Box&, Force&) { return {}; }
std::uint64_t rpmd_ja_model_fingerprint(const std::string&) { return 0; }

namespace
{
std::string write_sample_spool(const std::string& suffix, const int frames)
{
  const std::string path = "rpmd_ja_native_samples_test_" +
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + suffix;
  std::ofstream out(path, std::ios::binary);
  const char magic[8] = {'G','P','J','A','S','M','P','1'};
  const std::uint32_t version = 1, endian = native_endian;
  const std::int32_t n = 2, beads = 4;
  const double temperature = 300.0;
  const double cell[9] = {10,0,0, 0,10,0, 0,0,10};
  const double masses[2] = {1.0,2.0};
  const std::int32_t types[2] = {0,1};
  out.write(magic, sizeof(magic));
  out.write(reinterpret_cast<const char*>(&version), sizeof(version));
  out.write(reinterpret_cast<const char*>(&endian), sizeof(endian));
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  out.write(reinterpret_cast<const char*>(&beads), sizeof(beads));
  out.write(reinterpret_cast<const char*>(&temperature), sizeof(temperature));
  out.write(reinterpret_cast<const char*>(cell), sizeof(cell));
  out.write(reinterpret_cast<const char*>(masses), sizeof(masses));
  out.write(reinterpret_cast<const char*>(types), sizeof(types));
  for (int frame = 0; frame < frames; ++frame) {
    const double step = 5.0 + 2.5 * frame;
    double x[6], f[6];
    for (int i = 0; i < 6; ++i) { x[i] = 0.1 + 0.01 * frame; f[i] = -0.2; }
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));
    out.write(reinterpret_cast<const char*>(x), sizeof(x));
    out.write(reinterpret_cast<const char*>(f), sizeof(f));
  }
  assert(out.good());
  return path;
}

struct RemoveTestFile { std::string path; ~RemoveTestFile() { std::remove(path.c_str()); } };

void test_saved_sample_diagnostic()
{
  const std::string path = write_sample_spool(".bin", 3);
  RemoveTestFile cleanup{path};
  Atom atom;
  atom.number_of_atoms = 2;
  atom.cpu_mass = {1.0, 2.0};
  atom.cpu_type = {0, 1};
  atom.number_of_beads = 0;
  atom.position_per_atom.resize(6);
  std::vector<double> before = {0.7,0.8,0.9,1.0,1.1,1.2};
  atom.position_per_atom.copy_from_host(before.data());
  Box box{};
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = 10.0;
  box.cpu_h[9] = box.cpu_h[13] = box.cpu_h[17] = 0.1;

  {
    std::ifstream in(path, std::ios::binary);
    const auto header = read_header(in, 0, atom, box, 0.0, true);
    assert(header.frame_count == 3 && header.beads == 4 && header.temperature == 300.0);
  }
  atom.number_of_beads = 4;
  {
    std::ifstream in(path, std::ios::binary);
    bool rejected = false;
    try { (void)read_header(in, 0, atom, box, 300.0); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  atom.number_of_beads = 0;
  atom.cpu_mass[0] = 3.0;
  {
    std::ifstream in(path, std::ios::binary);
    bool rejected = false;
    try { (void)read_header(in, 0, atom, box, 0.0, true); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  atom.cpu_mass[0] = 1.0;

  diagnostic_expected_r0.assign(6, 0.105);
  Force force;
  diagnose_rpmd_ja_native_fit_samples(path, 1e-4, atom, box, force);
  std::vector<double> after(6);
  atom.position_per_atom.copy_to_host(after.data());
  assert(after == before);
  diagnostic_should_throw = true;
  bool failed = false;
  try { diagnose_rpmd_ja_native_fit_samples(path, 1e-4, atom, box, force); }
  catch (const std::runtime_error&) { failed = true; }
  diagnostic_should_throw = false;
  assert(failed);
  atom.position_per_atom.copy_to_host(after.data());
  assert(after == before);

  const std::string truncated_path = write_sample_spool(".truncated", 3);
  RemoveTestFile truncated_cleanup{truncated_path};
  std::ifstream source(truncated_path, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(source)), std::istreambuf_iterator<char>());
  source.close();
  bytes.pop_back();
  std::ofstream truncated(truncated_path, std::ios::binary | std::ios::trunc);
  truncated.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  truncated.close();
  std::ifstream invalid(truncated_path, std::ios::binary);
  bool rejected = false;
  try { (void)read_header(invalid, 0, atom, box, 0.0, true); }
  catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}

void check_close(const double a, const double b, const double tol=2e-9)
{
  assert(std::abs(a-b)<=tol*std::max({1.0,std::abs(a),std::abs(b)}));
}

double energy(const Graph& graph, const std::vector<double>& q, const std::vector<double>& sqrt_mass,
              const int n, const std::vector<double>& theta)
{
  double result=0.0;
  for(const auto& e:graph.edges){
    const double dx[3]={q[e.j]/sqrt_mass[e.j]-q[e.i]/sqrt_mass[e.i],
      q[n+e.j]/sqrt_mass[n+e.j]-q[n+e.i]/sqrt_mass[n+e.i],
      q[2*n+e.j]/sqrt_mass[2*n+e.j]-q[2*n+e.i]/sqrt_mass[2*n+e.i]};
    const std::size_t p=6*static_cast<std::size_t>(e.group);
    const double y[3]={theta[p]*dx[0]+theta[p+1]*dx[1]+theta[p+2]*dx[2],
      theta[p+1]*dx[0]+theta[p+3]*dx[1]+theta[p+4]*dx[2],
      theta[p+2]*dx[0]+theta[p+4]*dx[1]+theta[p+5]*dx[2]};
    result+=0.5*(dx[0]*y[0]+dx[1]*y[1]+dx[2]*y[2]);
  }
  return result;
}

void test_design_and_edge_operator()
{
  constexpr int n=4,d=3*n,p=6;
  Graph graph;
  graph.edges={{0,1,0,{1,0,0}},{1,2,0,{0,-1,0}},{0,3,0,{0,0,0}},{2,3,0,{0,0,1}}};
  std::vector<double> sqrt_atom={1.0,std::sqrt(2.0),std::sqrt(3.0),2.0},sqrt_mass(d),q(d),theta={1.2,-0.31,0.22,0.8,-0.17,1.1};
  for(int a=0;a<3;++a)for(int i=0;i<n;++i){sqrt_mass[a*n+i]=sqrt_atom[i];q[a*n+i]=0.13*(1+i+2*a)*(a%2?-1.0:1.0);}
  std::vector<double> design(static_cast<std::size_t>(d)*p);
  make_design(graph,q,sqrt_mass,n,p,design);
  std::vector<double> predicted_force(d,0.0);
  for(int c=0;c<p;++c)for(int i=0;i<d;++i)predicted_force[i]+=design[static_cast<std::size_t>(c)*d+i]*theta[c];
  std::vector<double> product(d);
  apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,n,product);
  for(int i=0;i<d;++i)check_close(predicted_force[i],-product[i]);

  constexpr double h=2e-6;
  for(int i=0;i<d;++i){auto plus=q,minus=q;plus[i]+=h;minus[i]-=h;
    const double fd=(energy(graph,plus,sqrt_mass,n,theta)-energy(graph,minus,sqrt_mass,n,theta))/(2*h);
    check_close(fd,product[i],2e-8);
  }
}

void test_pap_baseline()
{
  constexpr int n=3,d=3*n;
  const std::vector<double> mass={1.0,2.5,7.0},sqrt_atom={1.0,std::sqrt(2.5),std::sqrt(7.0)};
  std::vector<double> sqrt_mass(d),t(static_cast<std::size_t>(d)*3,0.0),u(static_cast<std::size_t>(d)*3);
  double mass_sum=0.0;for(double m:mass)mass_sum+=m;
  for(int a=0;a<3;++a)for(int i=0;i<n;++i){sqrt_mass[a*n+i]=sqrt_atom[i];t[static_cast<std::size_t>(a)*d+a*n+i]=sqrt_atom[i]/std::sqrt(mass_sum);}
  for(std::size_t i=0;i<u.size();++i)u[i]=std::sin(0.37*(i+1));
  std::vector<double> base(d*d),raw(d*d);
  for(int i=0;i<d;++i)for(int j=i;j<d;++j){const double v=(i==j?2.0:0.07*std::cos(i+2*j));base[static_cast<std::size_t>(i)*d+j]=base[static_cast<std::size_t>(j)*d+i]=v;}
  std::vector<double> contaminated=base;
  for(int i=0;i<d;++i)for(int j=0;j<d;++j){double v=base[static_cast<std::size_t>(i)*d+j];
    for(int a=0;a<3;++a)v+=1e-7*(t[static_cast<std::size_t>(a)*d+i]*u[static_cast<std::size_t>(a)*d+j]+u[static_cast<std::size_t>(a)*d+i]*t[static_cast<std::size_t>(a)*d+j]);
    for(int a=0;a<3;++a)for(int b=0;b<3;++b)v+=t[static_cast<std::size_t>(a)*d+i]*(0.2*(a==b))*t[static_cast<std::size_t>(b)*d+j];
    contaminated[static_cast<std::size_t>(i)*d+j]=v;
  }
  for(int i=0;i<d;++i)for(int j=0;j<d;++j){const double anti=1e-13*(i-j);raw[static_cast<std::size_t>(i)*d+j]=
    sqrt_mass[i]*sqrt_mass[j]*contaminated[static_cast<std::size_t>(i)*d+j]+anti;}
  std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);
  stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(stream,0,d,n,mass,sqrt_mass);
  std::vector<double> k(d*d),translation_projection(d*d),expected(d*d),v(d),out(d);
  for(int i=0;i<d;++i)for(int j=0;j<d;++j)k[static_cast<std::size_t>(i)*d+j]=
    0.5*(raw[static_cast<std::size_t>(i)*d+j]+raw[static_cast<std::size_t>(j)*d+i])/(sqrt_mass[i]*sqrt_mass[j]);
  for(int i=0;i<d;++i)for(int j=0;j<d;++j){double x=(i==j?1.0:0.0);for(int a=0;a<3;++a)x-=t[static_cast<std::size_t>(a)*d+i]*t[static_cast<std::size_t>(a)*d+j];translation_projection[static_cast<std::size_t>(i)*d+j]=x;}
  for(int i=0;i<d;++i)for(int j=0;j<d;++j)for(int a=0;a<d;++a)for(int b=0;b<d;++b)
    expected[static_cast<std::size_t>(i)*d+j]+=translation_projection[static_cast<std::size_t>(i)*d+a]*k[static_cast<std::size_t>(a)*d+b]*translation_projection[static_cast<std::size_t>(b)*d+j];
  for(int i=0;i<d;++i)v[i]=std::cos(0.23*(i+1));baseline.apply(v,out);
  for(int i=0;i<d;++i){double x=0.0;for(int j=0;j<d;++j)x+=expected[static_cast<std::size_t>(i)*d+j]*v[j];check_close(out[i],x,2e-8);}
}

void test_padded_qr_svd_and_cut_qp(cusolverDnHandle_t solver)
{
  constexpr int rows=2,cols=3; // Exercise the production QR's p > d padding branch.
  DeviceQR qr;qr.initialize(rows,cols);
  const std::vector<double> a1={1,0, 0,1, 1,1}; // column-major 2 x 3
  const std::vector<double> a2={1,1, 1,-1, 0,0};
  const std::vector<double> truth={0.7,-1.2,0.4};
  auto rhs=[&](const std::vector<double>& a){std::vector<double> y(rows);for(int i=0;i<rows;++i)for(int j=0;j<cols;++j)y[i]+=a[static_cast<std::size_t>(j)*rows+i]*truth[j];return y;};
  double discarded2=0.0;
  std::vector<double> r,z,r2,z2;qr.compress(a1,rhs(a1),r,z,discarded2);qr.compress(a2,rhs(a2),r2,z2,discarded2);merge_qr(r,z,r2,z2,cols,discarded2);
  const SmallSVD svd=svd_small(solver,r,z,cols);const auto fit=theta_from_eta(svd,svd.eta);
  for(int i=0;i<cols;++i)check_close(fit[i],truth[i],2e-9);
  check_close(discarded2,0.0,2e-9);
  auto noisy_rhs=rhs(a2);noisy_rhs[0]+=0.2;double residual2=0.0;
  qr.compress(a1,rhs(a1),r,z,residual2);qr.compress(a2,noisy_rhs,r2,z2,residual2);merge_qr(r,z,r2,z2,cols,residual2);
  assert(residual2>1e-6);

  SmallSVD identity;identity.p=2;identity.singular={1.0,1.0};identity.eta={0.0,0.0};identity.vt={1.0,0.0,0.0,1.0};
  const std::vector<std::vector<double>> constraints={{1.0,0.0},{-1.0,0.0},{1.0,0.0}};
  identity.eta={2.0,0.0};
  const auto xi=solve_cut_qp(identity,constraints,{0.5,-1.0,0.75});
  check_close(xi[0],1.0,2e-8);check_close(xi[1],0.0,2e-8);

  bool rejected=false;
  try { (void)svd_small(solver,{std::numeric_limits<double>::quiet_NaN(),0.0,0.0,1.0},{0.0,0.0},2); }
  catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
}

void test_read_frame_rejects_invalid_data()
{
  for(const int invalid:{0,1,2,3}){
    std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);
    double step=1.0;std::vector<double> x(3,0.25),f(3,-0.25);
    if(invalid==0)step=std::numeric_limits<double>::quiet_NaN();
    if(invalid==1)x[1]=std::numeric_limits<double>::infinity();
    if(invalid==2)f[2]=std::numeric_limits<double>::quiet_NaN();
    stream.write(reinterpret_cast<const char*>(&step),sizeof(step));
    stream.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(double));
    if(invalid!=3)stream.write(reinterpret_cast<const char*>(f.data()),f.size()*sizeof(double));stream.seekg(0);
    double read_step=0.0;bool rejected=false;
    try { read_frame(stream,x,f,read_step); }
    catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
  }
}

std::vector<std::vector<double>> internal_basis(const int n)
{
  const int d=3*n;std::vector<double> sqrt_atom(n,1.0);std::vector<std::vector<double>> basis;
  for(int i=0;i<d&&basis.size()<static_cast<std::size_t>(d-3);++i){std::vector<double> v(d,0.0);v[i]=1.0;project_translation(v,sqrt_atom,n);
    for(const auto& u:basis){const double dot=std::inner_product(v.begin(),v.end(),u.begin(),0.0);for(int k=0;k<d;++k)v[k]-=dot*u[k];}
    const double norm=std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));if(norm>1e-10){for(double& x:v)x/=norm;basis.push_back(std::move(v));}}
  assert(basis.size()==static_cast<std::size_t>(d-3));return basis;
}

void test_lanczos_finite_internal_space(cusolverDnHandle_t solver)
{
  for(const int n:{2,3}){
    const int d=3*n;const auto basis=internal_basis(n);const int internal=static_cast<int>(basis.size());
    const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);Graph graph;
    for(const double scale:{1.0,1.0e4,1.0e8}){
      std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
      for(int k=0;k<internal;++k)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
        raw[static_cast<std::size_t>(i)*d+j]+=scale*(k+1)*basis[k][i]*basis[k][j];
      std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);
      stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);
      DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);
      std::vector<double> restricted(static_cast<std::size_t>(internal)*internal);
      for(int j=0;j<internal;++j){std::vector<double> product(d);baseline.apply(basis[j],product);
        for(int i=0;i<internal;++i)restricted[static_cast<std::size_t>(i)*internal+j]=
          std::inner_product(basis[i].begin(),basis[i].end(),product.begin(),0.0);}
      const SmallEigen exact=eigen_small(solver,restricted,internal);
      const auto modes=lanczos_low_modes(solver,baseline,graph,{},sqrt_mass,sqrt_atom,n,96,4);
      assert(modes.size()==static_cast<std::size_t>(std::min(4,internal)));
      for(int k=0;k<static_cast<int>(modes.size());++k){
        const auto& v=modes[k].vector;double norm2=0.0;
        for(double x:v)norm2+=x*x;
        check_close(norm2,1.0,2e-8);
        for(int axis=0;axis<3;++axis){double translation=0.0;for(int i=0;i<n;++i)translation+=v[axis*n+i];assert(std::abs(translation)<2e-10);}
        std::vector<double> product(d);baseline.apply(v,product);
        const double rayleigh=std::inner_product(v.begin(),v.end(),product.begin(),0.0);
        check_close(rayleigh,modes[k].value,2e-7);
        check_close(modes[k].value,exact.values[k],2e-7);
      }
      for(int k=0;k<static_cast<int>(modes.size());++k)check_close(exact.values[k],scale*(k+1),2e-8);
    }
  }
}

void test_projected_cg_curvature_witness()
{
  constexpr int n=2,d=6;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);const auto basis=internal_basis(n);const std::vector<double> zero_theta;
  auto initialize=[&](DeviceBaseline& baseline,const std::vector<double>& raw){std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);baseline.initialize(stream,0,d,n,masses,sqrt_mass);};
  std::vector<double> identity(d*d,0.0);for(int i=0;i<d;++i)identity[static_cast<std::size_t>(i)*d+i]=1.0;DeviceBaseline positive;initialize(positive,identity);const CGResult solved=solve_projected_cg(positive,Graph{},basis[0],sqrt_mass,sqrt_atom,zero_theta,n);assert(solved.witness.classification.empty());assert(solved.relative_residual<=1e-8);std::vector<double> action(d);apply_total(positive,Graph{},solved.x,sqrt_mass,sqrt_atom,zero_theta,n,action);for(int i=0;i<d;++i)check_close(action[i],basis[0][i],1e-8);
  std::vector<double> indefinite=identity;for(int i=0;i<d;++i)for(int j=0;j<d;++j)indefinite[static_cast<std::size_t>(i)*d+j]-=2.0*basis[0][i]*basis[0][j];DeviceBaseline negative;initialize(negative,indefinite);const CGResult failed=solve_projected_cg(negative,Graph{},basis[0],sqrt_mass,sqrt_atom,zero_theta,n,3);assert(failed.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION");assert(failed.witness.finite);assert(failed.witness.probe==3&&failed.witness.iteration==0);check_close(failed.witness.p2,1.0,1e-12);check_close(failed.witness.rayleigh,-1.0,1e-10);
  const CGResult soft=solve_projected_cg(positive,Graph{},basis[0],sqrt_mass,sqrt_atom,zero_theta,n,5,1.1);assert(soft.witness.classification=="UNRESOLVED_SOFT_DIRECTION");check_close(soft.witness.rayleigh,1.0,1e-10);
  std::vector<double> zero_matrix(d*d,0.0);DeviceBaseline singular;initialize(singular,zero_matrix);const CGResult zero=solve_projected_cg(singular,Graph{},basis[0],sqrt_mass,sqrt_atom,zero_theta,n,6);assert(zero.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION");check_close(zero.witness.rayleigh,0.0,1e-12);
  Graph edge;edge.edges.push_back({0,1,0,{0,0,0}});const std::vector<double> nonfinite_theta(6,std::numeric_limits<double>::quiet_NaN());const CGResult nonfinite=solve_projected_cg(positive,edge,basis[0],sqrt_mass,sqrt_atom,nonfinite_theta,n,4);assert(nonfinite.witness.classification=="NONFINITE_OPERATOR");
}

void test_lanczos_blindspot_cg_cut_feedback(cusolverDnHandle_t solver)
{
  constexpr int n=40,d=3*n,p=6;const double epsilon=1e-3;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> w(d);
  w[0]=1.0-1.0/n;for(int i=1;i<n;++i)w[i]=-1.0/n;std::vector<double> start(d);for(int i=0;i<d;++i)start[i]=std::sin((i+1)*1.6180339887498948)+std::cos((i+1)*0.7548776662466927);project_translation(start,sqrt_atom,n);double norm=std::sqrt(std::inner_product(start.begin(),start.end(),start.begin(),0.0));for(double& x:start)x/=norm;const double overlap=std::inner_product(w.begin(),w.end(),start.begin(),0.0);double start_x2=0.0;for(int i=0;i<n;++i)start_x2+=start[i]*start[i];for(int i=0;i<n;++i)w[i]-=(overlap/start_x2)*start[i];norm=std::sqrt(std::inner_product(w.begin(),w.end(),w.begin(),0.0));for(double& x:w)x/=norm;assert(std::abs(std::inner_product(w.begin(),w.end(),start.begin(),0.0))<1e-12);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;for(int i=0;i<d;++i)for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]-=2.0*w[i]*w[j];std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);Graph graph;for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)graph.edges.push_back({i,j,0,{0,0,0}});
  const auto modes=lanczos_low_modes(solver,baseline,graph,std::vector<double>(p,0.0),sqrt_mass,sqrt_atom,n,96,4);assert(modes.front().value>0.99&&modes.front().residual<1e-10);std::vector<double> zero(p,0.0);const CGResult cg=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,zero,n,0,epsilon);assert(cg.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION");assert(cg.witness.rayleigh<0.0);
  SmallSVD identity;identity.p=p;identity.singular.assign(p,1.0);identity.eta.assign(p,0.0);identity.vt.assign(p*p,0.0);for(int i=0;i<p;++i)identity.vt[static_cast<std::size_t>(i)*p+i]=1.0;const auto row=spectral_row(graph,w,sqrt_mass,n,p);const double base=std::inner_product(w.begin(),w.end(),cg.witness.base.begin(),0.0);const auto xi=solve_cut_qp(identity,{row},{epsilon-base},epsilon/4.0);const auto theta=theta_from_eta(identity,xi);
  const auto basis=internal_basis(n);const int internal=static_cast<int>(basis.size());std::vector<double> restricted(static_cast<std::size_t>(internal)*internal);for(int j=0;j<internal;++j){std::vector<double> image(d);apply_total(baseline,graph,basis[j],sqrt_mass,sqrt_atom,theta,n,image);for(int i=0;i<internal;++i)restricted[static_cast<std::size_t>(i)*internal+j]=std::inner_product(basis[i].begin(),basis[i].end(),image.begin(),0.0);}const auto exact=eigen_small(solver,restricted,internal);assert(exact.values.front()>=epsilon-2e-10);const CGResult repaired=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,theta,n,0,epsilon);assert(repaired.witness.classification.empty());assert(repaired.relative_residual<=1e-8);
}

void test_probe_covariance_and_ibp(cusolverDnHandle_t solver)
{
  constexpr int n=8,d=3*n,train=21,validation=42,frames=train+validation;
  const double temperature=300.0,kbt=K_B*temperature;
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0),r0(d,0.0);
  std::vector<double> raw(d*d,0.0);for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;
  std::stringstream raw_stream(std::ios::in|std::ios::out|std::ios::binary);raw_stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));raw_stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(raw_stream,0,d,n,masses,sqrt_mass);
  Graph graph;graph.edges={{0,1,0,{0,0,0}}};const std::vector<double> theta={0.1,0.0,0.0,0.1,0.0,0.1};
  const std::vector<int> types={0,1,2,3,0,1,2,3};
  const auto basis=internal_basis(n);const std::string path="rpmd_ja_native_probe_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
  struct RemoveFile{std::string path;~RemoveFile(){std::remove(path.c_str());}} cleanup{path};
  const int r=static_cast<int>(basis.size());std::vector<double> restricted(static_cast<std::size_t>(r)*r);
  for(int j=0;j<r;++j){std::vector<double> action(d);baseline.apply(basis[j],action);std::vector<double> additive(d);apply_additive(graph,basis[j],sqrt_mass,sqrt_atom,theta,n,additive);
    for(int i=0;i<r;++i)restricted[static_cast<std::size_t>(i)*r+j]=std::inner_product(basis[i].begin(),basis[i].end(),action.begin(),0.0)+std::inner_product(basis[i].begin(),basis[i].end(),additive.begin(),0.0);}
  const SmallEigen eig=eigen_small(solver,restricted,r);std::vector<std::vector<double>> modes(r,std::vector<double>(d));
  for(int k=0;k<r;++k){assert(eig.values[k]>0.0);for(int j=0;j<r;++j)for(int i=0;i<d;++i)modes[k][i]+=eig.vectors[static_cast<std::size_t>(j)+static_cast<std::size_t>(k)*r]*basis[j][i];}
  {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);assert(out.good());
    auto frame=[&](double step,const std::vector<double>& x){out.write(reinterpret_cast<const char*>(&step),sizeof(step));out.write(reinterpret_cast<const char*>(x.data()),d*sizeof(double));std::vector<double> add(d),f(d);apply_additive(graph,x,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i)f[i]=-x[i]-add[i];out.write(reinterpret_cast<const char*>(f.data()),d*sizeof(double));};
    std::vector<double> zero(d);double step=0.0;for(int i=0;i<train;++i)frame(step++,zero);
    for(int k=0;k<r;++k){const double amplitude=std::sqrt(r*kbt/eig.values[k]);std::vector<double> x(d);for(int i=0;i<d;++i)x[i]=amplitude*modes[k][i]+0.4*basis[0][i];frame(step++,x);for(int i=0;i<d;++i)x[i]=-amplitude*modes[k][i]+0.4*basis[0][i];frame(step++,x);}
    assert(out.good());
  }
  std::ifstream spool(path,std::ios::binary);assert(spool.good());
  const ResponseCheck result=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{});
  assert(result.probes>=20);assert(result.cg<=1e-8);assert(result.response<=2e-8);assert(result.ibp<=2e-8);assert(result.force_residual<=2e-8);

  spool.close();
  {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);assert(out.good());
    auto frame=[&](double step,const std::vector<double>& x){out.write(reinterpret_cast<const char*>(&step),sizeof(step));out.write(reinterpret_cast<const char*>(x.data()),d*sizeof(double));std::vector<double> add(d),f(d);apply_additive(graph,x,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i)f[i]=-x[i]-add[i];out.write(reinterpret_cast<const char*>(f.data()),d*sizeof(double));};
    std::vector<double> zero(d);double step=0.0;for(int i=0;i<train;++i)frame(step++,zero);
    for(int k=0;k<r;++k){double amplitude=std::sqrt(r*kbt/eig.values[k]);if(k==0)amplitude*=1e-7;std::vector<double> x(d);for(int i=0;i<d;++i)x[i]=amplitude*modes[k][i];frame(step++,x);for(int i=0;i<d;++i)x[i]=-amplitude*modes[k][i];frame(step++,x);}
    assert(out.good());
  }
  spool.open(path,std::ios::binary);assert(spool.good());
  RitzMode soft;soft.value=eig.values[0];soft.vector=modes[0];bool covariance_rejected=false;
  try { (void)validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{soft}); }
  catch(const std::runtime_error& error) { covariance_rejected=std::string(error.what()).find("held-out probe covariance")!=std::string::npos; }
  assert(covariance_rejected);

  spool.close();const std::vector<double> zero_theta(6,0.0);
  {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);assert(out.good());
    auto frame=[&](double step,const std::vector<double>& x){out.write(reinterpret_cast<const char*>(&step),sizeof(step));out.write(reinterpret_cast<const char*>(x.data()),d*sizeof(double));std::vector<double> f(d);for(int i=0;i<d;++i)f[i]=-x[i];out.write(reinterpret_cast<const char*>(f.data()),d*sizeof(double));};
    std::vector<double> zero(d);double step=0.0;for(int i=0;i<train;++i)frame(step++,zero);
    for(const auto& mode:basis){const double amplitude=std::sqrt(r*kbt);std::vector<double> x(d);for(int i=0;i<d;++i)x[i]=amplitude*mode[i];frame(step++,x);for(double& value:x)value=-value;frame(step++,x);}
    assert(out.good());
  }
  spool.open(path,std::ios::binary);assert(spool.good());
  const ResponseCheck zero_result=validate_probes(solver,baseline,graph,zero_theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{});
  assert(zero_result.force_residual<=2e-8);assert(zero_result.response<=2e-8);assert(zero_result.ibp<=2e-8);
}
} // namespace

int main()
{
  test_saved_sample_diagnostic();
  test_design_and_edge_operator();
  test_pap_baseline();
  DeviceQR qr;qr.initialize(2,3);
  test_padded_qr_svd_and_cut_qp(qr.solver);
  test_read_frame_rejects_invalid_data();
  test_lanczos_finite_internal_space(qr.solver);
  test_projected_cg_curvature_witness();
  test_lanczos_blindspot_cg_cut_feedback(qr.solver);
  test_probe_covariance_and_ibp(qr.solver);
}
