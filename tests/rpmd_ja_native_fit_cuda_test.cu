#ifdef NDEBUG
#undef NDEBUG
#endif

#define GPUMD_RPMD_JA_NATIVE_FIT_TEST
#include "../src/measure/rpmd_ja_native_fit.cu"
#include "../src/measure/rpmd_ja_additive.cuh"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <tuple>

static std::vector<double> diagnostic_expected_r0;
static bool diagnostic_should_throw = false;
static std::string covariance_fit_test_fixture_directory;

#ifndef GPUMD_RPMD_JA_NATIVE_FIT_TEST_REAL_PREPARE
Force::Force(void) {}
int Force::get_number_of_potentials() const { return 0; }
Potential& Force::get_potential(const int) { throw std::runtime_error("unexpected potential lookup in native fit regression"); }
void diagnose_rpmd_ja_qnep_reference(double, Atom& atom, Box&, Force&, bool)
{
  std::vector<double> x(diagnostic_expected_r0.size());
  atom.position_per_atom.copy_to_host(x.data());
  for (std::size_t i = 0; i < x.size(); ++i) assert(std::abs(x[i] - diagnostic_expected_r0[i]) < 1e-12);
  if (diagnostic_should_throw) throw std::runtime_error("diagnostic test failure");
}
void generate_rpmd_ja_qnep_raw(const std::string&, double, double, const std::string&, Atom&, Box&, Force&, bool) {}
void prepare_rpmd_ja_qnep_reference(const std::string&, const std::string&, const std::string&,
                                    const RpmdJAModeValidator&, const std::string&,
                                    const std::function<void(const RpmdJAAdditiveData&)>&,
                                    const RpmdJAPrepareResourceObserver&,
                                    const RpmdJAPrepareAllocationObserver&,
                                    const std::function<void(const RpmdJAReference&)>&) {}
RpmdJAPrepareWorkspaceSizes query_rpmd_ja_qnep_prepare_workspace_sizes(int) { return {}; }
RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom&, Box&, Force&) { return {}; }
std::uint64_t rpmd_ja_model_fingerprint(const std::string&) { return 0; }
std::uint64_t rpmd_ja_qnep_config_fingerprint(Force&) { return 0; }
RpmdJAReference read_rpmd_ja_reference(const std::string&) { throw std::runtime_error("unexpected reference readback in native fit regression"); }
#endif

namespace
{
void check_close(const double a,const double b,const double tol=2e-9);
std::vector<double> solve_dense_reference(std::vector<double> matrix,std::vector<double> rhs,const int n);
std::vector<std::vector<double>> internal_basis(const std::vector<double>& sqrt_atom);

void test_covariance_checkpoint_round_trip_and_corruption_rejection()
{
  using namespace rpmd_ja_covariance_checkpoint;
  AcceptedPoint saved;saved.job_id="resume-test-job";saved.identity_sha256=std::string(64,'a');
  saved.branch="B0";saved.family="M0";saved.candidate_index=0;saved.lambda=2.0;saved.epsilon=1e-5;
  saved.rho=0.0;saved.ridge_lambda=2.0;saved.tau=0.2;saved.trust_radius=0.375;
  saved.elapsed_seconds=91.5;saved.max_objective_evaluations=1000;saved.objective_evaluations=23;
  saved.gradient_evaluations=7;saved.total_budget_gradients=7;saved.iterations=6;
  saved.phase_gradients[5]=7;saved.phase_seconds[5]=90.0;
  saved.theta={1.25,-0.5};saved.theta_start={1.0,-0.25};saved.stagnation_best_theta=saved.theta;
  saved.coordinate_scale={2.0,4.0};saved.bfgs_inverse={1.0,0.125,0.125,2.0};
  saved.bfgs_seed_gamma=1.5;saved.bfgs_last_valid_gamma=0.75;saved.bfgs_stage_seeded=true;
  saved.bfgs_seed_source="FIRST_VALID_SECANT";
  const std::string path="rpmd_ja_covariance_checkpoint_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  write_checkpoint(path,saved);
  const AcceptedPoint loaded=read_checkpoint(path,saved.identity_sha256);
  assert(loaded.job_id==saved.job_id&&loaded.branch==saved.branch&&loaded.family==saved.family);
  assert(loaded.iterations==saved.iterations&&loaded.objective_evaluations==saved.objective_evaluations);
  assert(loaded.gradient_evaluations==saved.gradient_evaluations&&loaded.phase_gradients==saved.phase_gradients);
  assert(loaded.elapsed_seconds==saved.elapsed_seconds&&loaded.tau==saved.tau&&loaded.trust_radius==saved.trust_radius);
  assert(loaded.theta==saved.theta&&loaded.theta_start==saved.theta_start&&
    loaded.coordinate_scale==saved.coordinate_scale&&loaded.bfgs_inverse==saved.bfgs_inverse);
  assert(loaded.bfgs_seed_gamma==saved.bfgs_seed_gamma&&loaded.bfgs_last_valid_gamma==saved.bfgs_last_valid_gamma&&
    loaded.bfgs_stage_seeded==saved.bfgs_stage_seeded&&loaded.bfgs_seed_source==saved.bfgs_seed_source);
  bool wrong_identity_rejected=false;try{(void)read_checkpoint(path,std::string(64,'b'));}
  catch(const std::runtime_error&){wrong_identity_rejected=true;}assert(wrong_identity_rejected);
  std::ifstream input(path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(input)),{});input.close();
  assert(bytes.size()>20);bytes[bytes.find("trust_radius")]='X';
  {std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));}
  bool tamper_rejected=false;try{(void)read_checkpoint(path,saved.identity_sha256);}
  catch(const std::runtime_error&){tamper_rejected=true;}assert(tamper_rejected);
  std::remove(path.c_str());
}

std::string write_sample_spool(const std::string& suffix, const int frames, const double interval=2.5,
                               const bool random_internal_samples=false, const double last_frame_shift=0.0)
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
  std::mt19937_64 random(0x5eed1234ULL);std::uniform_real_distribution<double> sample(-1.0,1.0);
  for (int frame = 0; frame < frames; ++frame) {
    const double step = 5.0 + interval * frame;
    double x[6], f[6];
    if(random_internal_samples){for(int axis=0;axis<3;++axis){const double displacement=sample(random),force=sample(random);
        x[2*axis]=5.0+displacement;x[2*axis+1]=5.0-displacement;f[2*axis]=force;f[2*axis+1]=-force;}
      if(frame==frames-1){x[0]+=last_frame_shift;x[1]-=last_frame_shift;f[0]+=last_frame_shift;f[1]-=last_frame_shift;}}
    else for (int i = 0; i < 6; ++i) { x[i] = 0.1 + 0.01 * frame; f[i] = -0.2; }
    out.write(reinterpret_cast<const char*>(&step), sizeof(step));
    out.write(reinterpret_cast<const char*>(x), sizeof(x));
    out.write(reinterpret_cast<const char*>(f), sizeof(f));
  }
  assert(out.good());
  return path;
}

std::string write_covariance_replay_spool(const std::string& suffix,const bool m1=false)
{
  const std::string path="rpmd_ja_covariance_replay_"+std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count())+suffix;
  std::ofstream out(path,std::ios::binary);assert(out.good());
  const char magic[8]={'G','P','J','A','S','M','P','1'};const std::uint32_t version=1,endian=native_endian;
  const std::int32_t n=m1?3:2,beads=4;const double temperature=300.0;
  const double cell[9]={10,0,0,0,10,0,0,0,10};
  const double masses_m0[2]={1.0,2.0},masses_m1[3]={16.0,1.0,16.0};
  const std::int32_t types_m0[2]={0,1},types_m1[3]={0,1,0};
  out.write(magic,sizeof(magic));out.write(reinterpret_cast<const char*>(&version),sizeof(version));
  out.write(reinterpret_cast<const char*>(&endian),sizeof(endian));out.write(reinterpret_cast<const char*>(&n),sizeof(n));
  out.write(reinterpret_cast<const char*>(&beads),sizeof(beads));out.write(reinterpret_cast<const char*>(&temperature),sizeof(temperature));
  out.write(reinterpret_cast<const char*>(cell),sizeof(cell));
  out.write(reinterpret_cast<const char*>(m1?masses_m1:masses_m0),static_cast<std::streamsize>(n*sizeof(double)));
  out.write(reinterpret_cast<const char*>(m1?types_m1:types_m0),static_cast<std::streamsize>(n*sizeof(std::int32_t)));
  for(int frame=0;frame<3;++frame){const double step=frame;const double shift=0.01*frame;
    const double position_m0[6]={5.0+shift,5.5+shift,5.0,5.0,5.0,5.0},force_m0[6]={};
    const double position_m1[9]={3.0+shift,0.1-shift,7.5+shift,0.01*frame,-0.01*frame,0.005*frame,
      0.002*frame,-0.003*frame,0.001*frame},force_m1[9]={};
    out.write(reinterpret_cast<const char*>(&step),sizeof(step));
    out.write(reinterpret_cast<const char*>(m1?position_m1:position_m0),static_cast<std::streamsize>(n*3*sizeof(double)));
    out.write(reinterpret_cast<const char*>(m1?force_m1:force_m0),static_cast<std::streamsize>(n*3*sizeof(double)));}
  assert(out.good());return path;
}

struct RemoveTestFile { std::string path; ~RemoveTestFile() { rpmd_ja_file_io::remove_file(path); } };

std::string read_test_file(const std::string& path)
{
  std::ifstream in(path,std::ios::binary);return std::string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
}

std::vector<double> read_snapshot_vector(const std::string& contents,const std::string& name)
{
  std::istringstream lines(contents);std::string line;
  while(std::getline(lines,line)){std::istringstream fields(line);std::string key;std::size_t count=0;if(!(fields>>key)||key!=name)continue;
    if(!(fields>>count))throw std::runtime_error("invalid CG snapshot vector header");std::vector<double> values(count);
    for(double& value:values)if(!(fields>>value))throw std::runtime_error("truncated CG snapshot vector");return values;}
  throw std::runtime_error("missing CG snapshot vector: "+name);
}

double read_snapshot_scalar(const std::string& contents,const std::string& name)
{
  std::istringstream lines(contents);std::string line;
  while(std::getline(lines,line)){std::istringstream fields(line);std::string key;double value=0.0;if(fields>>key&&key==name&&fields>>value)return value;}
  throw std::runtime_error("missing or invalid CG snapshot scalar: "+name);
}

struct CutQPFixture
{
  SmallSVD svd;
  std::vector<std::vector<double>> rows,a;
  std::vector<double> rhs,b,initial_lambda;
  double primal_tolerance=0.0;
};

std::string covariance_fit_test_fixture_path(const std::string& name)
{
  if(covariance_fit_test_fixture_directory.empty())throw std::runtime_error("native fit CUDA regression requires --fixture-dir");
  const char last=covariance_fit_test_fixture_directory.back();
  return covariance_fit_test_fixture_directory+(last=='/'||last=='\\'?"":"/")+name;
}

void verify_qp_fixture_bytes(const std::string& path,const std::string& bytes)
{
  const FrozenQpFixture* expected=nullptr;
  for(const auto& item:frozen_qp_fixtures){const std::string suffix=item.name;
    if(path.size()>suffix.size()&&path.compare(path.size()-suffix.size(),suffix.size(),suffix)==0&&
       (path[path.size()-suffix.size()-1]=='/'||path[path.size()-suffix.size()-1]=='\\')){expected=&item;break;}}
  if(!expected)throw std::runtime_error("unknown QP regression fixture: "+path);
  rpmd_ja_sha256::Digest sha;
  sha.update(reinterpret_cast<const unsigned char*>(bytes.data()),bytes.size());
  if(sha.finish_hex()!=expected->sha256)throw std::runtime_error("QP regression fixture SHA-256 mismatch: "+path);
}

CutQPFixture read_cut_qp_fixture(const std::string& path)
{
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("cannot open QP regression fixture: "+path);
  const std::string bytes((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>());
  if(file.bad())throw std::runtime_error("cannot read QP regression fixture: "+path);
  verify_qp_fixture_bytes(path,bytes);
  std::istringstream in(bytes);CutQPFixture fixture;std::string line;
  auto read_vector=[](std::istringstream& fields,std::vector<double>& values){std::size_t count=0;if(!(fields>>count))throw std::runtime_error("invalid QP fixture vector header");values.resize(count);for(double& value:values)if(!(fields>>value))throw std::runtime_error("truncated QP fixture vector");};
  while(std::getline(in,line)){std::istringstream fields(line);std::string key;fields>>key;
    if(key=="parameters")fields>>fixture.svd.p;
    else if(key=="primal_tolerance")fields>>fixture.primal_tolerance;
    else if(key=="svd_singular")read_vector(fields,fixture.svd.singular);
    else if(key=="svd_eta")read_vector(fields,fixture.svd.eta);
    else if(key=="initial_lambda")read_vector(fields,fixture.initial_lambda);
    else if(key=="svd_vt"){
      int rows=0,cols=0;fields>>rows>>cols;if(rows!=fixture.svd.p||cols!=fixture.svd.p)throw std::runtime_error("invalid QP fixture SVD dimensions");fixture.svd.vt.resize(static_cast<std::size_t>(rows)*cols);
      for(int i=0;i<rows;++i){if(!std::getline(in,line))throw std::runtime_error("truncated QP fixture SVD vectors");std::istringstream row(line);for(int j=0;j<cols;++j)if(!(row>>fixture.svd.vt[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*rows]))throw std::runtime_error("invalid QP fixture SVD row");}}
    else if(key=="original_rows"||key=="transformed_a"){
      std::size_t count=0;int cols=0;fields>>count>>cols;if(cols!=fixture.svd.p)throw std::runtime_error("invalid QP fixture row width");auto& matrix=key=="original_rows"?fixture.rows:fixture.a;auto& values=key=="original_rows"?fixture.rhs:fixture.b;matrix.assign(count,std::vector<double>(cols));values.resize(count);
      for(std::size_t i=0;i<count;++i){if(!std::getline(in,line))throw std::runtime_error("truncated QP fixture rows");std::istringstream row(line);std::size_t index=0;std::string tag;if(!(row>>index)||index!=i)throw std::runtime_error("invalid QP fixture row index");for(double& value:matrix[i])if(!(row>>value))throw std::runtime_error("invalid QP fixture row data");if(!(row>>tag>>values[i])||tag!=(key=="original_rows"?"rhs":"b"))throw std::runtime_error("invalid QP fixture row right hand side");}}
  }
  if(!in.eof()||fixture.svd.p!=60||fixture.rows.empty()||fixture.a.size()!=fixture.rows.size()||fixture.initial_lambda.size()!=fixture.rows.size()||
     fixture.svd.singular.size()!=static_cast<std::size_t>(fixture.svd.p)||fixture.svd.eta.size()!=static_cast<std::size_t>(fixture.svd.p)||fixture.svd.vt.size()!=static_cast<std::size_t>(fixture.svd.p)*fixture.svd.p||!(fixture.primal_tolerance>0.0))
    throw std::runtime_error("QP regression fixture dimensions or metadata are invalid");
  return fixture;
}

void test_fit_samples_entry_preserves_inputs()
{
  const std::string spool=write_sample_spool(".fit_samples",3,2.0),raw=spool+".qraw",output=spool+".out",kernel=output+".kernel";
  RemoveTestFile clean_spool{spool},clean_raw{raw},clean_kernel{kernel},clean_failure{output+".failure.txt"},clean_lock{output+".fit.lock"};
  {std::ofstream out(raw,std::ios::binary);out<<"external qraw sentinel";assert(out.good());}
  {std::ofstream out(kernel,std::ios::binary);out<<"kernel fixture";assert(out.good());}
  const std::string saved_spool=read_test_file(spool),saved_raw=read_test_file(raw);
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  const std::vector<double> original_position={0.7,0.8,0.9,1.0,1.1,1.2};atom.position_per_atom.copy_from_host(original_position.data());
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;Force force;
  RpmdJANativeFitOptions options;assert(options.max_stability_rounds==160);options.output_path=output;options.kernel_table=kernel;options.raw_input_path=raw;options.cutoff=2.0;options.epsilon=1e-3;options.response_tolerance=0.15;options.fd_step=1e-3;
  atom.cpu_mass[0]=3.0;bool identity_rejected=false;
  try{fit_rpmd_ja_native_reference_from_samples(options,spool,atom,box,force);}catch(const std::runtime_error&){identity_rejected=true;}
  assert(identity_rejected&&read_test_file(spool)==saved_spool&&read_test_file(raw)==saved_raw);
  atom.cpu_mass[0]=1.0;bool early_fit_failure=false;
  try{fit_rpmd_ja_native_reference_from_samples(options,spool,atom,box,force);}catch(const std::runtime_error& e){early_fit_failure=std::string(e.what()).find("held-out frames")!=std::string::npos;}
  assert(early_fit_failure&&read_test_file(spool)==saved_spool&&read_test_file(raw)==saved_raw);
  assert(atom.number_of_beads==0);std::vector<double> after(6);atom.position_per_atom.copy_to_host(after.data());assert(after==original_position);
}

void test_full_spd_failure_keeps_candidate_pack()
{
  const std::string path="rpmd_ja_candidate_pack_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
  {std::ofstream out(path,std::ios::binary);out<<"candidate package";assert(out.good());}
  bool pack_owned=true;
  struct RemoveIfOwned{std::string path;bool& owned;~RemoveIfOwned(){if(owned)std::remove(path.c_str());}} cleanup{path,pack_owned};
  const std::string prepare_error="qNEP translation-complement Hessian is not positive definite (POTRF leading minor 2 of dimension 3, raw: test.qraw)";
  try{throw std::runtime_error(prepare_error);}
  catch(const std::exception& error){assert(preserve_candidate_package_on_full_spd_failure(error.what(),pack_owned));}
  assert(!pack_owned);{std::ifstream candidate(path,std::ios::binary);assert(candidate.good());}
  assert(!preserve_candidate_package_on_full_spd_failure("cuSOLVER Cholesky parameter failure",pack_owned));
  std::remove(path.c_str());
}

void test_response_uncomputed_values_are_explicit()
{
  const ResponseCheck response;
  assert(std::isnan(response.response)&&std::isnan(response.ibp)&&std::isnan(response.cg)&&std::isnan(response.force_residual));
  assert(!response.observed_cov_available&&!response.predicted_cov_available&&!response.cg_residual_available&&!response.response_ibp_available&&!response.force_residual_available);
  std::ostringstream summary;write_response_stats(summary,response);
  assert(summary.str().find("observed_cov_status=NOT_COMPUTED")!=std::string::npos);
  assert(summary.str().find("predicted_cov_status=NOT_COMPUTED")!=std::string::npos);
  assert(summary.str().find("response_uncertainty_status=NOT_COMPUTED")!=std::string::npos);
  assert(summary.str().find("ibp_uncertainty_status=NOT_COMPUTED")!=std::string::npos);
  assert(summary.str().find("response_diagnostic stage=NOT_COMPUTED bootstrap_started=0")!=std::string::npos);
  assert(summary.str().find('\n')==std::string::npos);
  assert(summary.str().find("response_ibp_status=NOT_COMPUTED")!=std::string::npos);
  assert(summary.str().find("force_residual_status=NOT_COMPUTED")!=std::string::npos);
  ResponseCheck with_radii;with_radii.response_band.level_radii={{1,0.25}};with_radii.response_extremes_available=true;
  with_radii.response_ratio_min=0.8;with_radii.response_ratio_max=1.2;std::ostringstream radii_summary;write_response_stats(radii_summary,with_radii);
  const std::string radii_text=radii_summary.str();const auto radii=radii_text.find("response_block_radii=1:0.25");
  const auto ratio=radii_text.find("response_ratio_status=AVAILABLE");assert(radii!=std::string::npos&&ratio>radii);
}

void test_m0_only_resume_replays_only_a_missing_development_response()
{
  ResponseCheck missing;
  assert(development_response_needs_replay(true,true,missing));
  ResponseCheck complete;complete.probes=1;complete.cg_residual_status="CG_PASS";
  complete.cg_residual_available=true;complete.observed_cov_available=true;complete.predicted_cov_available=true;
  complete.response_ibp_available=true;complete.force_residual_available=true;complete.mean_q={0.0};
  complete.observed_matrix={1.0};complete.centered_covariance_matrix={1.0};
  complete.predicted_matrix={1.0};complete.ibp_matrix={0.0};
  assert(response_consumer_statistics_ready(complete));
  assert(!development_response_needs_replay(true,true,complete));
  assert(development_response_needs_replay(true,false,complete));
  assert(development_response_needs_replay(false,true,complete));
}

void test_frozen_response_snapshot_round_trip_and_identity_rejection()
{
  const std::string path="rpmd_ja_frozen_response_snapshot_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  RemoveTestFile cleanup{path};
  ResponseCheck response;response.probes=4;response.total_frames=8;response.validation_frames=4;
  response.statistical_frames=4;response.response=0.1;response.ibp=0.2;response.cg=1.0e-12;
  response.force_residual=0.03;response.response_ratio_min=0.9;response.response_ratio_max=1.1;
  response.observed_cov_min=0.8;response.observed_cov_max=1.2;response.observed_cov_condition=1.5;
  response.predicted_cov_min=0.9;response.predicted_cov_max=1.1;response.predicted_cov_condition=1.3;
  response.baseline_apply_seconds=0.01;response.block_variance_max_relative_delta=0.0;
  response.block_variance_relative_delta={0.0,0.0,0.0,0.0};response.cg_residual_status="CG_PASS";
  response.cg_verification_method="COMPENSATED";response.observed_cov_available=true;
  response.predicted_cov_available=true;response.cg_residual_available=true;
  response.response_ibp_available=true;response.force_residual_available=true;
  response.probe_sources={"LOW_RITZ","FIXED_PHYSICAL_OXYGEN_SKELETON_X","FIXED_RANDOM","TYPE_LOCAL"};
  response.probe_origins={{},{},{},{0,0,1.0}};response.mean_q={0.25,0.1,-0.2,0.3};
  response.observed_matrix.assign(16,0.0);response.centered_covariance_matrix.assign(16,0.0);
  response.predicted_matrix.assign(16,0.0);response.ibp_matrix.assign(16,0.0);response.whitened_matrix.assign(16,0.0);
  for(int index=0;index<4;++index){const std::size_t diagonal=static_cast<std::size_t>(index)*4+index;
    response.observed_matrix[diagonal]=1.0;response.centered_covariance_matrix[diagonal]=0.75;
    response.predicted_matrix[diagonal]=0.95;response.ibp_matrix[diagonal]=0.02;response.whitened_matrix[diagonal]=0.05;}
  response.worst_response_coefficients={1.0,0.0,0.0,0.0};response.worst_ibp_coefficients={1.0,0.0,0.0,0.0};
  response.response_band.estimate=0.1;response.response_band.radius=0.01;
  response.response_band.lower=0.09;response.response_band.upper=0.11;response.response_band.frames=4;
  response.response_band.block_lengths_stable=true;response.response_band.status="RESPONSE_PASS";
  response.response_band.diagnostic_stage="INTERVAL_CHECK";response.response_band.rejection_reason="NONE";
  response.response_band.bootstrap_started=true;response.response_band.level_radii={{1,0.01}};
  response.ibp_band.estimate=0.2;response.ibp_band.radius=0.02;response.ibp_band.lower=0.18;
  response.ibp_band.upper=0.22;response.ibp_band.frames=4;response.ibp_band.block_lengths_stable=true;
  response.ibp_band.status="IBP_INCONCLUSIVE";response.ibp_band.diagnostic_stage="INTERVAL_CHECK";
  response.ibp_band.rejection_reason="NOT_RESOLVED";response.ibp_band.bootstrap_started=true;
  response.ibp_band.level_radii={{1,0.02}};
  const std::vector<double> theta={0.5};const std::string manifest(64,'a'),candidate(64,'b'),input(64,'c');
  const std::string digest=write_blind_response_checkpoint(path,response,theta,manifest,candidate,input,true);
  const std::vector<std::string> expected_sources=response.probe_sources;
  const std::vector<std::string> expected_origins={"-1,-1,NA","-1,-1,NA","-1,-1,NA","0,0,1"};
  const BlindResponseSnapshot restored=read_blind_response_checkpoint(path,manifest,candidate,input,digest,
    expected_sources,expected_origins);
  assert(restored.passed&&restored.theta==theta&&restored.response.mean_q==response.mean_q&&
    restored.response.observed_matrix==response.observed_matrix&&restored.response.centered_covariance_matrix==response.centered_covariance_matrix);
  auto rejects=[&](const std::string& expected_candidate,const std::vector<std::string>& expected_sources){bool rejected=false;
    try{(void)read_blind_response_checkpoint(path,manifest,expected_candidate,input,digest,expected_sources,expected_origins);}
    catch(const std::runtime_error&){rejected=true;}assert(rejected);};
  rejects(std::string(64,'d'),expected_sources);rejects(candidate,{"TYPE_LOCAL"});
  ResponseCheck wrong_dimensions=response;wrong_dimensions.whitened_matrix.clear();bool dimensions_rejected=false;
  try{(void)write_blind_response_checkpoint(path,wrong_dimensions,theta,manifest,candidate,input,true);}
  catch(const std::runtime_error& error){dimensions_rejected=
    std::string(error.what()).find("vector dimensions")!=std::string::npos;}
  assert(dimensions_rejected);
  ResponseCheck invalid_nonlocal=response;invalid_nonlocal.probe_origins[0]={0,0,1.0};
  bool invalid_origin_rejected=false;
  try{(void)write_blind_response_checkpoint(path,invalid_nonlocal,theta,manifest,candidate,input,true);}
  catch(const std::runtime_error& error){invalid_origin_rejected=
    std::string(error.what()).find("source kind")!=std::string::npos;}
  assert(invalid_origin_rejected);
  ResponseCheck missing_type_local=response;missing_type_local.probe_origins[3]={};
  bool missing_type_local_rejected=false;
  try{(void)write_blind_response_checkpoint(path,missing_type_local,theta,manifest,candidate,input,true);}
  catch(const std::runtime_error& error){missing_type_local_rejected=
    std::string(error.what()).find("source kind")!=std::string::npos;}
  assert(missing_type_local_rejected);
  ResponseCheck massless_type_local=response;massless_type_local.probe_origins[3]={0,0,0.0};
  bool massless_type_local_rejected=false;
  try{(void)write_blind_response_checkpoint(path,massless_type_local,theta,manifest,candidate,input,true);}
  catch(const std::runtime_error& error){massless_type_local_rejected=
    std::string(error.what()).find("source kind")!=std::string::npos;}
  assert(massless_type_local_rejected);
  {std::ofstream altered(path,std::ios::binary|std::ios::app);altered<<"tampered\n";}
  bool digest_rejected=false;try{(void)read_blind_response_checkpoint(path,manifest,candidate,input,digest,
      expected_sources,expected_origins);}
  catch(const std::runtime_error& error){digest_rejected=
    std::string(error.what()).find("checksum differs")!=std::string::npos;}
  assert(digest_rejected);
}

void test_publication_io_classification_and_two_rename_recovery()
{
  const std::string output="rpmd_ja_publication_io_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string staged=output+".stage",sidecar=output+".stability",staged_sidecar=staged+".stability";
  RemoveTestFile remove_output{output},remove_stage{staged},remove_sidecar{sidecar},remove_staged_sidecar{staged_sidecar};
  const auto write=[](const std::string& path,const std::string& bytes){
    std::ofstream file(path,std::ios::binary|std::ios::trunc);assert(file.good());file.write(bytes.data(),bytes.size());
    file.close();assert(file.good());};
  const std::string output_bytes="prepared reference",sidecar_bytes="prepared sidecar";
  write(staged,output_bytes);write(staged_sidecar,"foreign sidecar");
  const std::string output_digest=fit_input_content_identity(staged,true).sha256;
  const std::string sidecar_digest=rpmd_ja_covariance_checkpoint::sha256("prepared sidecar");
  bool hard_digest_failure=false;
  try{publish_verified_prepared_artifacts(output,staged,output_digest,sidecar_digest,[](){});}
  catch(const std::runtime_error& error){hard_digest_failure=
    !retryable_publication_io_failure(error.what())&&
    std::string(error.what()).find("foreign or modified")!=std::string::npos;}
  assert(hard_digest_failure);
  write(staged,output_bytes);write(staged_sidecar,sidecar_bytes);
  bool first_rename_interrupted=false;
  try{publish_verified_prepared_artifacts(output,staged,output_digest,sidecar_digest,[](){
      throw std::runtime_error("FEASIBILITY_PUBLISH_RETRYABLE: staged artifact atomic replace failed");});}
  catch(const std::runtime_error& error){first_rename_interrupted=
    retryable_publication_io_failure(error.what())&&fit_input_content_identity(output,true).sha256==output_digest;}
  assert(first_rename_interrupted);
  publish_verified_prepared_artifacts(output,staged,output_digest,sidecar_digest,[](){});
  assert(fit_input_content_identity(output,true).sha256==output_digest&&
    fit_input_content_identity(sidecar,true).sha256==sidecar_digest);
  assert(!retryable_publication_io_failure("FEASIBILITY_PUBLISH_FAIL: final artifact bytes differ from the prepared stage digest"));
  assert(!retryable_publication_io_failure("FEASIBILITY_RESUME_IDENTITY_FAIL: prepared stage artifact disappeared before commit"));
}

void test_regularization_response_delta_uses_full_predicted_matrix()
{
  const double predicted_low=1.0/1.1,predicted_high=1.0/0.9;
  const double error_low=std::abs(1.0/predicted_low-1.0),error_high=std::abs(1.0/predicted_high-1.0);
  const auto one_dimensional=normalized_response_matrix_delta({predicted_low},{predicted_high},1);
  assert(std::abs(error_low-error_high)<1e-14&&one_dimensional.available&&one_dimensional.status=="AVAILABLE");
  assert(one_dimensional.relative>0.1&&std::abs(one_dimensional.relative-2.0/11.0)<1e-14);

  const auto off_diagonal=normalized_response_matrix_delta({2.0,0.25,0.25,1.0},{2.0,-0.25,-0.25,1.0},2);
  assert(off_diagonal.available&&off_diagonal.absolute>0.7&&off_diagonal.relative>0.3);
  const auto zero=normalized_response_matrix_delta({0.0},{0.0},1);
  assert(!zero.available&&zero.status=="NEAR_ZERO_GLOBAL_NORM");
}

void test_probe_moments_centering_and_small_segments()
{
  ProbeMoments moments(1),shifted(1),wrong_force(1);
  const double kbt=1.0;
  for(const double delta:{1.0,-1.0}){
    const std::vector<double> q={delta},f={-delta},q_shifted={delta+17.0},f_wrong={-2.0*delta};
    moments.add(q,f);shifted.add(q_shifted,f);wrong_force.add(q,f_wrong);
  }
  assert(moments.covariance_q()[0]==1.0);
  assert(std::abs(shifted.covariance_q()[0]-moments.covariance_q()[0])<1e-14);
  assert(std::abs(moments.ibp_matrix(kbt)[0])<1e-14);
  assert(std::abs(shifted.ibp_matrix(kbt)[0]-moments.ibp_matrix(kbt)[0])<1e-14);
  assert(std::abs(wrong_force.ibp_matrix(kbt)[0])>0.4);
  assert(std::string(probe_sample_status(2,3))=="INSUFFICIENT_SAMPLES");
}

void test_ibp_diagnostic_status_is_not_a_numerical_gate()
{
  BootstrapBand band;band.estimate=0.8;
  band.status="IBP_PASS";assert(!ibp_statistics_numerically_invalid(band));
  band.status="IBP_FAIL";assert(!ibp_statistics_numerically_invalid(band));
  band.status="IBP_INCONCLUSIVE";assert(!ibp_statistics_numerically_invalid(band));
  band.status="NUMERICAL_FAILURE";assert(ibp_statistics_numerically_invalid(band));
  band.status="IBP_INCONCLUSIVE";band.estimate=std::numeric_limits<double>::quiet_NaN();
  assert(ibp_statistics_numerically_invalid(band));
  band.estimate=1.0;band.diagnostic_stage="INTERVAL_CHECK";band.upper=std::numeric_limits<double>::infinity();
  assert(ibp_statistics_numerically_invalid(band));
}

void test_type_local_source_and_motion_diagnostics()
{
  constexpr int n=40;std::vector<double> sqrt_mass(n,1.0);std::vector<int> types(n);
  for(int i=0;i<n;++i)types[i]=i%4;
  std::vector<std::string> sources;std::vector<ProbeOrigin> origins;
  const auto probes=make_probes({},sqrt_mass,types,n,&sources,&origins);
  assert(probes.size()==sources.size()&&probes.size()==origins.size());
  int type_local_count=0;
  for(std::size_t i=0;i<sources.size();++i)if(sources[i]=="TYPE_LOCAL"){
    ++type_local_count;assert(origins[i].seed_atom_index>=0&&origins[i].seed_atom_index<4);
    assert(origins[i].type_index==types[origins[i].seed_atom_index]&&origins[i].mass==1.0);}
  assert(type_local_count==4);

  SourceAtomMotion motion;motion.atom_index=7;motion.type_index=2;motion.mass=5.0;std::vector<double> reference(24,0.0),position(24,0.0);
  position[7]=1.0;motion.add(position,reference,8,0);position[7]=3.0;motion.add(position,reference,8,0);
  position[7]=7.0;motion.add(position,reference,8,1);
  assert(motion.frames==3&&motion.block_frames[0]==2&&motion.block_frames[1]==1);
  assert(motion.mean_square_displacement()==59.0/3.0&&motion.maximum_adjacent_sampled_frame_displacement==4.0);
  const auto block0=motion.mean_displacement(0),block1=motion.mean_displacement(1);
  assert(block0[0]==2.0&&block0[1]==0.0&&block0[2]==0.0&&block1[0]==7.0);
  const auto local=std::find(sources.begin(),sources.end(),"TYPE_LOCAL");assert(local!=sources.end());
  const std::size_t local_index=static_cast<std::size_t>(std::distance(sources.begin(),local));
  std::ostringstream final_probe_report;write_probe_origin_table(final_probe_report,sources,origins);
  assert(final_probe_report.str().find("16 TYPE_LOCAL 0 0 1")!=std::string::npos);
  FixedProbeStatistics stats;stats.sources={sources[local_index]};stats.origins={origins[local_index]};stats.source_atom_motion.push_back(motion);
  std::ostringstream report;write_probe_origin_diagnostics(report,stats);
  assert(report.str().find("probe_index source seed_atom_index type_index mass")!=std::string::npos);
  assert(report.str().find("SOURCE_ATOM_MOTION")!=std::string::npos);
  assert(report.str().find("maximum_adjacent_sampled_frame_displacement")!=std::string::npos);
  assert(report.str().find("type-local seed vectors are translation-projected and orthogonalized")!=std::string::npos);
}

void test_ibp_noise_channel_selection_and_force_variance()
{
  ProbeMoments direct(1),left(1),right(1),shifted(1);
  const double q_values[]={-1.0,0.0,2.0,3.0},f_values[]={1.0,3.0,5.0,7.0};
  for(int i=0;i<4;++i){direct.add({q_values[i]},{f_values[i]});(i<2?left:right).add({q_values[i]},{f_values[i]});shifted.add({q_values[i]},{f_values[i]+11.0});}
  const ProbeMoments first_half=left,second_half=right;ProbeMoments merged=left;merged.merge(right);
  assert(direct.variance_f().size()==1&&std::abs(direct.variance_f()[0]-5.0)<1e-14);
  assert(std::abs(merged.variance_f()[0]-direct.variance_f()[0])<1e-14);
  assert(std::abs(shifted.variance_f()[0]-direct.variance_f()[0])<1e-13);
  assert(std::abs(direct.ibp_matrix(1.0)[0]-merged.ibp_matrix(1.0)[0])<1e-14);
  ProbeMoments legacy=direct,legacy_first=first_half,legacy_second=second_half;std::fill(legacy.sum_f2.begin(),legacy.sum_f2.end(),0.0);
  std::fill(legacy_first.sum_f2.begin(),legacy_first.sum_f2.end(),0.0);std::fill(legacy_second.sum_f2.begin(),legacy_second.sum_f2.end(),0.0);
  const BootstrapBand direct_band=bootstrap_ibp_band(direct,{first_half,second_half},2,1.0,0.15,91);
  const BootstrapBand legacy_band=bootstrap_ibp_band(legacy,{legacy_first,legacy_second},2,1.0,0.15,91);
  const BootstrapBand merged_band=bootstrap_ibp_band(merged,{first_half,second_half},2,1.0,0.15,91);
  const auto same_or_nan=[](double a,double b){return a==b||(std::isnan(a)&&std::isnan(b));};
  assert(direct_band.estimate==legacy_band.estimate&&same_or_nan(direct_band.radius,legacy_band.radius)&&direct_band.status==legacy_band.status);
  assert(direct_band.estimate==merged_band.estimate&&same_or_nan(direct_band.radius,merged_band.radius)&&direct_band.status==merged_band.status);

  ProbeMoments harmonic(2);const double soft_q=1.0,stiff_q=std::sqrt(1.0/1000.0);
  for(double s0:{-1.0,1.0})for(double s1:{-1.0,1.0})
    harmonic.add({s0,s1*stiff_q},{-s0,-1000.0*s1*stiff_q});
  assert(std::abs(gaussian_variance_proxy(harmonic,1,0,1.0)-1000.0)<1e-10);
  assert(std::abs(harmonic.ibp_matrix(1.0)[2])<1e-14);
  const std::vector<IBPNoiseChannel> selected=select_ibp_noise_channels(harmonic,1.0);
  const auto high_variance=std::find_if(selected.begin(),selected.end(),[](const IBPNoiseChannel& channel){return channel.i==1&&channel.j==0;});
  assert(high_variance!=selected.end()&&high_variance->selection_reason.find("TOP3_GAUSSIAN_VARIANCE_PROXY")!=std::string::npos);
  ProbeMoments tied(3);tied.add({0.0,0.0,0.0},{0.0,0.0,0.0});tied.add({0.0,0.0,0.0},{0.0,0.0,0.0});
  const std::vector<IBPNoiseChannel> tied_channels=select_ibp_noise_channels(tied,1.0);
  assert(tied_channels.size()==6&&tied_channels[0].i==0&&tied_channels[0].j==1&&tied_channels[1].i==0&&tied_channels[1].j==2&&tied_channels[2].i==1&&tied_channels[2].j==0);

  ProbeMoments validation_a(2),validation_b(2);validation_a=harmonic;
  for(double s0:{-2.0,2.0})for(double s1:{-0.1,0.1})validation_b.add({s0,s1},{-0.3*s0,-7.0*s1});
  const FixedChannelNoiseDiagnostic a=diagnose_fixed_ibp_channels(harmonic,{validation_a},4,4,1.0);
  const FixedChannelNoiseDiagnostic b=diagnose_fixed_ibp_channels(harmonic,{validation_b},4,4,1.0);
  assert(a.channels.size()==b.channels.size());
  for(std::size_t i=0;i<a.channels.size();++i)assert(a.channels[i].i==b.channels[i].i&&a.channels[i].j==b.channels[i].j&&a.channels[i].selection_reason==b.channels[i].selection_reason);
  const auto selected_cross=std::find_if(a.channels.begin(),a.channels.end(),[](const IBPNoiseChannel& channel){return channel.i==1&&channel.j==0;});
  assert(selected_cross!=a.channels.end()&&std::abs(selected_cross->validation_entry)<1e-14);
  assert(a.validation_frames_used==4&&a.band.frames==4&&a.validation_frames_omitted==0);
  const auto signed_bounds=signed_channel_interval(-0.2,0.05);
  assert(std::abs(signed_bounds.first+0.25)<1e-15&&std::abs(signed_bounds.second+0.15)<1e-15);

  ProbeMoments invalid_variance(1),valid_validation(1);invalid_variance.frames=2;invalid_variance.sum_f[0]=10.0;invalid_variance.sum_f2[0]=1.0;
  ProbeMoments roundoff_variance(1);roundoff_variance.frames=2;roundoff_variance.sum_f[0]=2.0;roundoff_variance.sum_f2[0]=std::nextafter(2.0,0.0);
  valid_validation.add({-1.0},{1.0});valid_validation.add({1.0},{-1.0});
  assert(std::isnan(invalid_variance.variance_f()[0]));
  assert(roundoff_variance.variance_f()[0]==0.0);
  const FixedChannelNoiseDiagnostic invalid=diagnose_fixed_ibp_channels(invalid_variance,{valid_validation},2,2,1.0);
  assert(invalid.variance_status=="NUMERICAL_FAILURE_TRAIN");
}

void test_validation_noise_blocks_start_at_split_boundary()
{
  constexpr int frames=3101;const std::string spool=write_sample_spool(".noise_blocks",frames,1.0);RemoveTestFile clean_spool{spool};
  std::ifstream in(spool,std::ios::binary);assert(in.good());Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  const SampleHeader header=read_header(in,0,atom,box,0.0,true);std::vector<double> r0(6,0.0);
  read_training_r0(in,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  const std::uint64_t train=2*header.frame_count/3,validation=header.frame_count-train;
  const FixedProbeStatistics stats=collect_fixed_probe_statistics(in,header,header.frame_count,train,r0,true);
  assert(train==2067&&stats.validation_base_block_length==2&&stats.fixed_channel_noise.validation_frames_used==validation&&stats.fixed_channel_noise.validation_frames_omitted==0);
  assert(stats.validation_blocks.size()==validation/2);
  for(const ProbeMoments& block:stats.validation_blocks)assert(block.frames==2);
}

void test_omitted_validation_tail_does_not_change_channel_interval()
{
  const std::string first_path=write_sample_spool(".tail_a",3103,1.0,true,0.0);
  const std::string second_path=write_sample_spool(".tail_b",3103,1.0,true,10000.0);
  RemoveTestFile remove_first{first_path},remove_second{second_path};
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  auto collect=[&](const std::string& path){std::ifstream in(path,std::ios::binary);assert(in.good());const SampleHeader header=read_header(in,0,atom,box,0.0,true);
    std::vector<double> r0(6,0.0);read_training_r0(in,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
    const std::uint64_t train=2*header.frame_count/3;return collect_fixed_probe_statistics(in,header,header.frame_count,train,r0,true);};
  const FixedProbeStatistics first=collect(first_path),second=collect(second_path);
  const FixedChannelNoiseDiagnostic& a=first.fixed_channel_noise;const FixedChannelNoiseDiagnostic& b=second.fixed_channel_noise;
  assert(a.validation_frames_total==1035&&a.validation_frames_used==1034&&a.validation_frames_omitted==1);
  assert(a.status=="COMPUTED_BLOCK_BOOTSTRAP_INTERVAL"&&b.status==a.status);
  assert(a.band.estimate==b.band.estimate&&a.band.status==b.band.status&&a.band.level_radii.size()==b.band.level_radii.size());
  const auto same_or_nan=[](double x,double y){return x==y||(std::isnan(x)&&std::isnan(y));};
  assert(same_or_nan(a.band.radius,b.band.radius));
  for(std::size_t i=0;i<a.band.level_radii.size();++i)assert(a.band.level_radii[i]==b.band.level_radii[i]);
}

void test_multichannel_bootstrap_maps_local_tail_indices()
{
  constexpr int original_count=10;ProbeMoments training(original_count);training.frames=2;
  training.sum_qq[0*original_count+0]=2.0;training.sum_qq[1*original_count+1]=2.0;
  training.sum_qq[2*original_count+2]=1.5;training.sum_qq[9*original_count+9]=0.02;
  training.sum_f2[8]=20.0;training.sum_f2[9]=16.0;training.sum_fq[9*original_count+2]=2.0*std::sqrt(6.0);
  const std::vector<int> expected_probes={0,1,2,8,9};
  std::vector<ProbeMoments> validation_blocks;ProbeMoments validation(original_count);std::mt19937_64 random(0x71b00b5ULL);
  const auto uniform=[&](){return 2.0*(static_cast<double>(random()>>11)/9007199254740992.0)-1.0;};
  for(int block_index=0;block_index<512;++block_index){std::vector<double> q(original_count),f(original_count);for(int probe:expected_probes){q[probe]=uniform();f[probe]=uniform();}
    f[9]=-9.0*q[2]+0.2*uniform();ProbeMoments block(original_count);block.add(q,f);
    for(double& value:q)value=-value;for(double& value:f)value=-value;block.add(q,f);validation.merge(block);validation_blocks.push_back(std::move(block));}
  const FixedChannelNoiseDiagnostic diagnostic=diagnose_fixed_ibp_channels(training,validation_blocks,2,1024,1.0);
  assert(diagnostic.channels.size()==5&&diagnostic.tail_probe_original_indices==expected_probes);
  assert(diagnostic.status=="COMPUTED_BLOCK_BOOTSTRAP_INTERVAL"&&diagnostic.band.block_lengths_stable&&diagnostic.band.frames==1024);
  const std::vector<double> covariance=validation.covariance_fq();double expected_estimate=0.0;
  for(const IBPNoiseChannel& channel:diagnostic.channels){const double entry=covariance[static_cast<std::size_t>(channel.i)*original_count+channel.j]+(channel.i==channel.j?1.0:0.0);
    expected_estimate=std::max(expected_estimate,std::abs(entry));}
  assert(std::abs(diagnostic.band.estimate-expected_estimate)<1e-13&&diagnostic.band.estimate>1.5);
  const auto cross=std::find_if(diagnostic.channels.begin(),diagnostic.channels.end(),[](const IBPNoiseChannel& channel){return channel.i==9&&channel.j==2;});
  assert(cross!=diagnostic.channels.end()&&cross->validation_entry<0.0);
  const auto signed_bounds=signed_channel_interval(cross->validation_entry,diagnostic.band.radius);
  assert(signed_bounds.first<0.0&&signed_bounds.first==cross->validation_entry-diagnostic.band.radius);

  const int local_count=static_cast<int>(expected_probes.size());const ProbeMoments local_validation=select_probe_moments(validation,expected_probes);
  std::vector<ProbeMoments> local_blocks;for(const ProbeMoments& block:validation_blocks)local_blocks.push_back(select_probe_moments(block,expected_probes));
  std::vector<int> local_index(original_count,-1);for(int i=0;i<local_count;++i)local_index[expected_probes[i]]=i;
  const auto metric=[&](const ProbeMoments& moments){const auto cfq=moments.covariance_fq();std::vector<double> entries;
    for(const IBPNoiseChannel& channel:diagnostic.channels){const int i=local_index[channel.i],j=local_index[channel.j];entries.push_back(cfq[static_cast<std::size_t>(i)*local_count+j]+(channel.i==channel.j?1.0:0.0));}return entries;};
  const auto maximum=[](const std::vector<double>& values){double value=0.0;for(double x:values)value=std::max(value,std::abs(x));return value;};
  const auto distance=[maximum](const std::vector<double>& a,const std::vector<double>& b){std::vector<double> delta(a.size());for(std::size_t i=0;i<a.size();++i)delta[i]=a[i]-b[i];return maximum(delta);};
  const std::vector<double> center=metric(local_validation);const auto deviations=resample_percentile_deviations(local_blocks,32,center,500,0xa54ff53a5f1d36f1ULL,metric,distance);
  assert(!deviations.empty()&&std::abs(diagnostic.band.radius-deviations[494])<1e-13);

  // A constant fifth selected probe makes its local (4,4) qq product uninformative.
  std::vector<ProbeMoments> constant_tail_blocks;validation=ProbeMoments(original_count);random.seed(0x71b00b5ULL);
  for(int block_index=0;block_index<512;++block_index){std::vector<double> q(original_count),f(original_count);for(int probe:expected_probes){q[probe]=uniform();f[probe]=uniform();}q[9]=1.0;
    f[9]=-9.0*q[2]+0.2*uniform();ProbeMoments block(original_count);block.add(q,f);
    for(double& value:q)value=-value;for(double& value:f)value=-value;block.add(q,f);validation.merge(block);constant_tail_blocks.push_back(std::move(block));}
  FixedChannelNoiseDiagnostic rejected=diagnose_fixed_ibp_channels(training,constant_tail_blocks,2,1024,1.0);
  assert(rejected.tail_probe_original_indices==expected_probes&&rejected.band.rejection_reason=="ZERO_PRODUCT_VARIANCE");
  assert(rejected.band.rejection_detail.find("product=qq probe_i=4 probe_j=4")!=std::string::npos);
  FixedProbeStatistics report_stats;report_stats.probes.resize(original_count);report_stats.sources.resize(original_count,"FIXED_RANDOM");
  for(int i=0;i<7;++i){report_stats.segments.emplace_back(original_count);report_stats.lags.emplace_back(original_count);}
  report_stats.validation_complete_moments=ProbeMoments(original_count);report_stats.fixed_channel_noise=rejected;
  SampleHeader header;header.frame_count=1024;header.beads=32;header.temperature=300.0;
  const std::string report=format_fixed_probe_report(report_stats,header,1.0,682);
  assert(report.find("fixed_channel_bootstrap_diagnostic stage=TAIL_CHECK")!=std::string::npos);
  assert(report.find("product=qq probe_i=4 probe_j=4")!=std::string::npos);
  assert(report.find("tail_probe_index_space LOCAL_SELECTED_PROBES")!=std::string::npos);
  assert(report.find("tail_probe_original_indices 0 1 2 8 9")!=std::string::npos);
  assert(report.find("tail_probe_scope ALL_FQ_AND_QQ_PRODUCTS_IN_SELECTED_PROBE_SUBSPACE")!=std::string::npos);
}

void test_block_bootstrap_uses_matrix_error_radius()
{
  const std::vector<double> first={1.0,0.0,0.0,0.0},second={0.0,0.0,0.0,1.0};
  assert(nonsymmetric_spectral_norm(first,2)==nonsymmetric_spectral_norm(second,2));
  assert(std::abs(symmetric_matrix_distance(first,second,2)-1.0)<1e-12);

  auto make_blocks=[](const double force_scale){
    std::vector<ProbeMoments> blocks;ProbeMoments full(1);
    std::vector<double> amplitudes(512,1.0);std::fill(amplitudes.begin()+256,amplitudes.end(),std::sqrt(3.0));
    std::mt19937_64 random(7);std::shuffle(amplitudes.begin(),amplitudes.end(),random);
    for(double q:amplitudes){ProbeMoments moment(1);
      moment.add({q},{force_scale*q});moment.add({-q},{-force_scale*q});full.merge(moment);blocks.push_back(std::move(moment));}
    return std::make_pair(std::move(full),std::move(blocks));
  };
  auto correct=make_blocks(-1.0);const BootstrapBand pass=bootstrap_ibp_band(correct.first,correct.second,2,2.0,0.15,7);
  assert(pass.status=="IBP_PASS"&&pass.block_lengths_stable&&std::abs(pass.estimate)<1e-14);
  assert(pass.bootstrap_started&&pass.diagnostic_stage=="INTERVAL_CHECK"&&pass.rejection_reason=="NONE");
  const FixedChannelNoiseDiagnostic fixed_noise=diagnose_fixed_ibp_channels(correct.first,correct.second,2,1024,2.0);
  assert(fixed_noise.channels.size()==1&&fixed_noise.channels[0].i==0&&fixed_noise.channels[0].j==0);
  assert(fixed_noise.status=="COMPUTED_BLOCK_BOOTSTRAP_INTERVAL"&&fixed_noise.band.block_lengths_stable&&fixed_noise.band.frames==1024);
  const auto fixed_interval=signed_channel_interval(fixed_noise.channels[0].validation_entry,fixed_noise.band.radius);
  assert(fixed_interval.first<=fixed_noise.channels[0].validation_entry&&fixed_interval.second>=fixed_noise.channels[0].validation_entry);
  auto biased=make_blocks(-2.0);const BootstrapBand fail=bootstrap_ibp_band(biased.first,biased.second,2,2.0,0.15,7);
  assert(fail.status=="IBP_FAIL"&&fail.lower>0.15);
  assert(fail.bootstrap_started&&fail.rejection_reason=="LOWER_BOUND_EXCEEDS_TOLERANCE");
  biased.second.resize(8);const BootstrapBand insufficient=bootstrap_ibp_band(biased.first,biased.second,2,2.0,0.15,7);
  assert(insufficient.status=="IBP_INCONCLUSIVE"&&!insufficient.block_lengths_stable);
  assert(!insufficient.bootstrap_started&&insufficient.diagnostic_stage=="TAIL_CHECK"&&insufficient.rejection_reason!="NOT_COMPUTED");

  const BootstrapBand response=bootstrap_metric_band(correct.first,biased.second,2,0.15,7,
    [](const ProbeMoments& moments){return moments.covariance_q();},
    [](const std::vector<double>& matrix){return max_abs_symmetric_eigenvalue(matrix,1);},
    [](const std::vector<double>& a,const std::vector<double>& b){return symmetric_matrix_distance(a,b,1);},
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE");
  assert(response.status=="RESPONSE_INCONCLUSIVE");
}

void test_bootstrap_uses_matching_complete_frames_and_circular_blocks()
{
  std::vector<ProbeMoments> blocks;
  for(int i=0;i<5;++i){ProbeMoments block(1);block.add({static_cast<double>(i)},{-static_cast<double>(i)});blocks.push_back(std::move(block));}
  const auto deviations=resample_percentile_deviations(blocks,2,{5.0},32,17,
    [](const ProbeMoments& moments){return std::vector<double>{static_cast<double>(moments.frames)};},
    [](const std::vector<double>& sample,const std::vector<double>& center){return std::abs(sample[0]-center[0]);});
  assert(deviations.size()==32&&std::all_of(deviations.begin(),deviations.end(),[](double value){return value==0.0;}));

  ProbeMoments full(1);std::vector<ProbeMoments> complete_blocks;
  for(int b=0;b<64;++b){ProbeMoments block(1);for(int i=0;i<3;++i)block.add({1.0},{-1.0});full.merge(block);complete_blocks.push_back(std::move(block));}
  full.add({100.0},{-100.0});full.add({101.0},{-101.0});
  const BootstrapBand band=bootstrap_metric_band(full,complete_blocks,3,0.15,23,
    [](const ProbeMoments& moments){return moments.covariance_q();},
    [](const std::vector<double>& matrix){return max_abs_symmetric_eigenvalue(matrix,1);},
    [](const std::vector<double>& a,const std::vector<double>& b){return symmetric_matrix_distance(a,b,1);},
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE");
  assert(band.frames==192&&full.frames==194&&band.estimate==0.0&&band.status=="RESPONSE_INCONCLUSIVE");
  assert(band.rejection_reason=="ZERO_PRODUCT_VARIANCE"&&std::string(bootstrap_block_length_status(band))=="NOT_COMPUTED");
  std::ostringstream diagnostic;write_bootstrap_diagnostic(diagnostic,"IBP_DIAGNOSTIC",band);
  assert(diagnostic.str().find("stage=TAIL_CHECK bootstrap_started=0 configured_replicates=500 completed_levels=0 reason=ZERO_PRODUCT_VARIANCE")!=std::string::npos);
  assert(diagnostic.str().find("product=fq probe_i=0 probe_j=0")!=std::string::npos);

  ProbeMoments tail_base(1),tail_changed(1);std::vector<ProbeMoments> tail_blocks;std::uint32_t state=36;
  for(int b=0;b<512;++b){state=1664525U*state+1013904223U;const double q=2.0*static_cast<double>(state)/4294967296.0-1.0;
    ProbeMoments block(1);for(int i=0;i<4;++i)block.add({q},{-q});tail_base.merge(block);tail_changed.merge(block);tail_blocks.push_back(std::move(block));}
  tail_base.add({0.0},{0.0});tail_changed.add({2000.0},{300.0});
  assert(block_product_tail_covered(tail_blocks,tail_base));
  assert(!block_product_tail_covered(tail_blocks,tail_changed));
  BootstrapBand tail_diagnostic;
  assert(block_product_tail_covered(tail_blocks,tail_changed,&tail_diagnostic)==block_product_tail_covered(tail_blocks,tail_changed));
  assert(tail_diagnostic.rejection_reason!="NOT_COMPUTED"&&tail_diagnostic.rejection_detail.find("base_blocks=512")!=std::string::npos);
  const BootstrapBand base_ibp_band=bootstrap_ibp_band(tail_base,tail_blocks,4,1.0,0.15,41);
  const BootstrapBand changed_ibp_band=bootstrap_ibp_band(tail_changed,tail_blocks,4,1.0,0.15,41);
  assert(base_ibp_band.estimate==changed_ibp_band.estimate&&base_ibp_band.status==changed_ibp_band.status);
  const auto frame_metric=[](const ProbeMoments& moments){return std::vector<double>{static_cast<double>(moments.frames)};};
  const auto frame_norm=[](const std::vector<double>& value){return std::abs(value[0]);};
  const auto frame_distance=[](const std::vector<double>& a,const std::vector<double>& b){return std::abs(a[0]-b[0]);};
  const BootstrapBand base_tail_band=bootstrap_metric_band(tail_base,tail_blocks,4,3000.0,41,frame_metric,frame_norm,frame_distance,
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE");
  const BootstrapBand changed_tail_band=bootstrap_metric_band(tail_changed,tail_blocks,4,3000.0,41,frame_metric,frame_norm,frame_distance,
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE");
  assert(base_tail_band.frames==2048&&changed_tail_band.frames==2048);
  assert(base_tail_band.estimate==changed_tail_band.estimate&&base_tail_band.status=="RESPONSE_PASS"&&changed_tail_band.status==base_tail_band.status);
  assert(base_tail_band.rejection_reason=="NONE"&&base_tail_band.bootstrap_started&&base_tail_band.level_radii.size()>=3);
  const auto equal_or_uncomputed=[](double a,double b){return a==b||(std::isnan(a)&&std::isnan(b));};
  assert(equal_or_uncomputed(base_ibp_band.radius,changed_ibp_band.radius)&&equal_or_uncomputed(base_ibp_band.lower,changed_ibp_band.lower)&&equal_or_uncomputed(base_ibp_band.upper,changed_ibp_band.upper));
}

void test_bootstrap_block_factor_candidates_and_terminal_level()
{
  const std::vector<std::pair<std::size_t,std::vector<int>>> cases={
    {0,{}},{15,{}},{16,{1}},{999,{1,2,4,8,16,32,62}},{1000,{1,2,4,8,16,32,62}},
    {1023,{1,2,4,8,16,32,63}},{1024,{1,2,4,8,16,32,64}}};
  for(const auto& test:cases){const std::vector<int> factors=bootstrap_block_factors(test.first);assert(factors==test.second);
    for(std::size_t i=0;i<factors.size();++i){assert(factors[i]>0&&test.first/static_cast<std::size_t>(factors[i])>=16);if(i)assert(factors[i]>factors[i-1]);}}

  ProbeMoments full(1);std::vector<ProbeMoments> blocks;blocks.reserve(1000);std::uint32_t state=36;
  for(int b=0;b<1000;++b){state=1664525U*state+1013904223U;const double q=2.0*static_cast<double>(state)/4294967296.0-1.0;
    ProbeMoments block(1);block.add({q},{-q});block.add({q},{-q});full.merge(block);blocks.push_back(std::move(block));}
  assert(block_product_tail_covered(blocks,full));
  const double kbt=1.0;const auto ibp_matrix=[=](const ProbeMoments& moments){return moments.ibp_matrix(kbt);};
  const auto ibp_norm=[](const std::vector<double>& matrix){return nonsymmetric_spectral_norm(matrix,1);};
  const auto ibp_distance=[](const std::vector<double>& a,const std::vector<double>& b){return nonsymmetric_matrix_distance(a,b,1);};
  const double expected_estimate=ibp_norm(ibp_matrix(full));
  const BootstrapBand band=bootstrap_ibp_band(full,blocks,2,kbt,0.15,41);
  assert(band.frames==2000&&band.estimate==expected_estimate&&band.bootstrap_started&&band.level_radii.size()==7);
  assert(band.level_radii[4].first==32&&band.level_radii[5].first==64&&band.level_radii[6].first==124);

  const auto frame_metric=[](const ProbeMoments& moments){return std::vector<double>{static_cast<double>(moments.frames)};};
  const auto frame_distance=[](const std::vector<double>& a,const std::vector<double>& b){return std::abs(a[0]-b[0]);};
  const std::vector<double> frame_deviations=resample_percentile_deviations(blocks,62,{2000.0},32,41,frame_metric,frame_distance);
  assert(frame_deviations.size()==32&&std::all_of(frame_deviations.begin(),frame_deviations.end(),[](double value){return value==0.0;}));

  const std::vector<double> center=ibp_matrix(full);
  const std::vector<double> first=resample_percentile_deviations(blocks,62,center,500,41,ibp_matrix,ibp_distance);
  const std::vector<double> second=resample_percentile_deviations(blocks,62,center,500,41,ibp_matrix,ibp_distance);
  const std::size_t q99=static_cast<std::size_t>(std::ceil(0.99*500))-1;
  assert(first.size()==500&&first==second&&first[q99]==band.level_radii.back().second);
}

void test_block_product_tail_uses_segment_centering()
{
  constexpr int frames=4096,block_length=4;const double rho=0.98;std::mt19937_64 random(0x9b05688c2b3e6c1fULL);std::normal_distribution<double> normal;
  ProbeMoments full(1),shifted_full(1),pending(1),shifted_pending(1);std::vector<ProbeMoments> blocks,shifted_blocks;
  double q=normal(random);int count=0;
  for(int frame=0;frame<frames;++frame){if(frame)q=rho*q+std::sqrt(1.0-rho*rho)*normal(random);
    full.add({q},{-q});shifted_full.add({q+17.0},{-q});pending.add({q},{-q});shifted_pending.add({q+17.0},{-q});
    if(++count==block_length){blocks.push_back(std::move(pending));shifted_blocks.push_back(std::move(shifted_pending));pending=ProbeMoments(1);shifted_pending=ProbeMoments(1);count=0;}}
  const bool original_tail=block_product_tail_covered(blocks,full),shifted_tail=block_product_tail_covered(shifted_blocks,shifted_full);
  assert(original_tail&&shifted_tail);
  const BootstrapBand original=bootstrap_ibp_band(full,blocks,block_length,1.0,0.15,31);
  const BootstrapBand shifted=bootstrap_ibp_band(shifted_full,shifted_blocks,block_length,1.0,0.15,31);
  assert(original.frames==frames&&shifted.frames==frames);
  assert(std::abs(original.estimate-shifted.estimate)<1e-12&&original.status==shifted.status);
}

void test_bootstrap_nonfinite_is_numerical_failure()
{
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
  assert(!std::isfinite(max_abs_symmetric_eigenvalue({nan},1)));
  assert(!std::isfinite(nonsymmetric_spectral_norm({inf},1)));
  assert(!std::isfinite(nonsymmetric_spectral_norm({1.0e308},1)));
  ProbeMoments overflow(1);overflow.add({1.0e200},{1.0e200});
  const BootstrapBand band=bootstrap_ibp_band(overflow,{overflow},1,1.0,0.15,9);
  assert(band.status=="NUMERICAL_FAILURE"&&!std::isfinite(band.estimate));
  assert(band.diagnostic_stage=="INPUT_CHECK"&&band.rejection_reason=="NONFINITE_INPUT_MOMENTS"&&!band.bootstrap_started);
}

void test_nonfinite_tail_product_is_numerical_failure()
{
  ProbeMoments full(1);std::vector<ProbeMoments> blocks;blocks.reserve(64);
  for(int b=0;b<64;++b){const double q=b%4<2?1.0e100:-1.0e100,f=b%2==0?1.0e100:-1.0e100;ProbeMoments block(1);block.add({q},{f});full.merge(block);blocks.push_back(std::move(block));}
  const BootstrapBand band=bootstrap_ibp_band(full,blocks,1,1.0,0.15,9);
  assert(std::isfinite(band.estimate)&&std::abs(band.estimate-1.0)<1e-14);
  assert(band.status=="NUMERICAL_FAILURE"&&band.rejection_reason=="NONFINITE_PRODUCT_STATISTICS");
  assert(band.diagnostic_stage=="TAIL_CHECK"&&!band.bootstrap_started);
}

void test_response_bootstrap_ignores_force_position_tail()
{
  ProbeMoments baseline(1),changed(1);std::vector<ProbeMoments> baseline_blocks,changed_blocks;std::uint32_t state=36;
  for(int b=0;b<512;++b){state=1664525U*state+1013904223U;const double q=2.0*static_cast<double>(state)/4294967296.0-1.0;
    const double f0=-q,f1=(((b/128)%2)?1.0:-1.0)*q;ProbeMoments a(1),c(1);a.add({q},{f0});c.add({q},{f1});
    baseline.merge(a);changed.merge(c);baseline_blocks.push_back(std::move(a));changed_blocks.push_back(std::move(c));}
  assert(block_product_tail_covered(baseline_blocks,baseline,nullptr,false,true));
  assert(block_product_tail_covered(changed_blocks,changed,nullptr,false,true));
  assert(!block_product_tail_covered(changed_blocks,changed,nullptr,true,true));
  const auto metric=[](const ProbeMoments& moments){return moments.covariance_q();};
  const auto norm=[](const std::vector<double>& value){return std::abs(value[0]);};
  const auto distance=[](const std::vector<double>& a,const std::vector<double>& b){return std::abs(a[0]-b[0]);};
  const BootstrapBand a=bootstrap_metric_band(baseline,baseline_blocks,1,10.0,29,metric,norm,distance,
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE",false,true);
  const BootstrapBand b=bootstrap_metric_band(changed,changed_blocks,1,10.0,29,metric,norm,distance,
    "RESPONSE_PASS","RESPONSE_FAIL","RESPONSE_INCONCLUSIVE",false,true);
  assert(a.bootstrap_started&&b.bootstrap_started&&a.status==b.status&&a.rejection_reason==b.rejection_reason);
  assert(a.estimate==b.estimate&&a.level_radii==b.level_radii);
  const BootstrapBand ibp=bootstrap_ibp_band(changed,changed_blocks,1,1.0,10.0,29);
  assert(ibp.status=="IBP_INCONCLUSIVE"&&ibp.diagnostic_stage=="TAIL_CHECK");
}

void test_bootstrap_rejection_diagnostics()
{
  const BootstrapBand empty=bootstrap_ibp_band(ProbeMoments(1),{},1,1.0,0.15,1);
  assert(empty.status=="IBP_INCONCLUSIVE"&&empty.diagnostic_stage=="INPUT_CHECK"&&empty.rejection_reason=="EMPTY_BLOCKS"&&!empty.bootstrap_started);
  ProbeMoments full(20);std::vector<ProbeMoments> blocks;std::uint32_t state=36;
  for(int b=0;b<128;++b){state=1664525U*state+1013904223U;const double q=2.0*static_cast<double>(state)/4294967296.0-1.0;
    ProbeMoments block(20);block.add(std::vector<double>(20,q),std::vector<double>(20,-q));full.merge(block);blocks.push_back(std::move(block));}
  const BootstrapBand noise=bootstrap_ibp_band(full,blocks,1,1.0,0.15,1);
  assert(noise.status=="IBP_INCONCLUSIVE"&&noise.diagnostic_stage=="TAIL_CHECK"&&noise.rejection_reason=="ACF_NOISE_TOO_LARGE");
  assert(!noise.bootstrap_started&&noise.level_radii.empty()&&noise.rejection_detail.find("product=fq probe_i=0 probe_j=0")!=std::string::npos);
  assert(block_product_tail_covered(blocks,full)==false);
}

void test_slow_product_correlation_is_inconclusive()
{
  constexpr int frames=4096,block_length=4;std::normal_distribution<double> normal;
  for(const double rho:{0.99,0.999}){
    std::mt19937_64 random(0x1f83d9abfb41bd6bULL+static_cast<std::uint64_t>(rho*1000));
    ProbeMoments full(1),pending(1);std::vector<ProbeMoments> blocks;blocks.reserve(frames/block_length);
    double q=normal(random);int count=0;
    for(int frame=0;frame<frames;++frame){if(frame)q=rho*q+std::sqrt(1.0-rho*rho)*normal(random);
      full.add({q},{-q});pending.add({q},{-q});
      if(++count==block_length){blocks.push_back(std::move(pending));pending=ProbeMoments(1);count=0;}}
    const BootstrapBand band=bootstrap_ibp_band(full,blocks,block_length,1.0,0.15,static_cast<std::uint64_t>(rho*1000));
    assert(band.status=="IBP_INCONCLUSIVE"&&!band.block_lengths_stable);
    assert(band.diagnostic_stage=="TAIL_CHECK"&&!band.bootstrap_started&&band.level_radii.empty());
    assert(band.rejection_reason=="TAIL_UNRESOLVED"||band.rejection_reason=="BLOCK_TOO_SHORT"||band.rejection_reason=="TOO_FEW_EFFECTIVE_BLOCKS");
    assert(band.rejection_detail.find("estimated_iat_blocks=")!=std::string::npos&&band.rejection_detail.find("effective_blocks=")!=std::string::npos);
  }
}

void test_block_bootstrap_harmonic_coverage()
{
  constexpr int frames=512,block_length=1,replicates=80,probes=2;const double rho=0.25;
  std::mt19937_64 random(0x510e527fade682d1ULL);std::normal_distribution<double> normal;int stable=0,covered=0;
  for(int trial=0;trial<replicates;++trial){ProbeMoments full(probes),pending(probes);std::vector<ProbeMoments> blocks;blocks.reserve(frames/block_length);
    std::array<double,probes> q{};for(double& value:q)value=normal(random);
    for(int frame=0;frame<frames;++frame){if(frame)for(double& value:q)value=rho*value+std::sqrt(1.0-rho*rho)*normal(random);
      const std::vector<double> qv(q.begin(),q.end()),fv={-q[0],-q[1]};full.add(qv,fv);pending.add(qv,fv);
      if((frame+1)%block_length==0){blocks.push_back(std::move(pending));pending=ProbeMoments(probes);}}
    const BootstrapBand band=bootstrap_ibp_band(full,blocks,block_length,1.0,0.15,static_cast<std::uint64_t>(trial+1));
    if(band.block_lengths_stable){++stable;if(band.lower<=0.0&&band.upper>=0.0)++covered;}
  }
  constexpr double declared_coverage=0.95,z_score=2.0;
  assert(stable>=72);
  assert(static_cast<double>(covered)/stable>=declared_coverage-z_score*std::sqrt(declared_coverage*(1.0-declared_coverage)/stable));
}

void test_check_samples_is_read_only()
{
  const std::string spool=write_sample_spool(".check",3),report=spool+".txt";
  RemoveTestFile remove_spool{spool};RemoveTestFile remove_report{report};const std::string before=read_test_file(spool);
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  const std::vector<double> position={0.7,0.8,0.9,1.0,1.1,1.2};atom.position_per_atom.copy_from_host(position.data());
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  check_rpmd_ja_native_fit_samples(spool,report,atom,box);
  std::vector<double> after(6);atom.position_per_atom.copy_to_host(after.data());
  const std::string text=read_test_file(report);
  assert(before==read_test_file(spool)&&after==position&&atom.number_of_beads==0);
  assert(text.find("INSUFFICIENT_SAMPLES")!=std::string::npos);
  assert(text.find("IBP_DIAGNOSTIC stage=TAIL_CHECK bootstrap_started=0")!=std::string::npos);
  assert(text.find("block_length_stability NOT_COMPUTED")!=std::string::npos);
  assert(text.find("cannot replace final response validation")!=std::string::npos);
  assert(text.find("lag_autocorrelation")!=std::string::npos&&text.find("IBP_residual")!=std::string::npos);
  assert(text.find("IBP_GAUSSIAN_VARIANCE_PROXY")!=std::string::npos&&text.find("IBP_FIXED_CHANNEL_VALIDATION")!=std::string::npos);
  assert(text.find("selection_source TRAIN_ONLY")!=std::string::npos&&text.find("ADJACENT_SEGMENT_INTERNAL_CHECK")!=std::string::npos);
}

void test_fixed_r0_validation_mean_projection_drift_report()
{
  FixedProbeStatistics stats;stats.probes.resize(2);stats.sources={"FIXED_RANDOM","TYPE_LOCAL"};
  for(int i=0;i<7;++i){stats.segments.emplace_back(2);stats.lags.emplace_back(2);}
  stats.segments[1].add({1.0,-2.0},{0.0,0.0});stats.segments[1].add({3.0,4.0},{0.0,0.0});
  stats.segments[2].add({5.0,-4.0},{0.0,0.0});stats.segments[2].add({7.0,2.0},{0.0,0.0});
  stats.segments[0].merge(stats.segments[1]);stats.segments[0].merge(stats.segments[2]);
  stats.validation_complete_moments=stats.segments[2];
  SampleHeader header;header.frame_count=4;header.temperature=300.0;
  const std::string report=format_fixed_probe_report(stats,header,1.0,2);
  assert(report.find("validation_centering NONE; status DIAGNOSTIC_ONLY; no acceptance threshold")!=std::string::npos);
  assert(report.find("\n0 FIXED_RANDOM 2 2 2 6 4\n")!=std::string::npos);
  assert(report.find("\n1 TYPE_LOCAL 2 2 1 -1 -2\n")!=std::string::npos);
}

void test_sample_spool_minimum_frame_boundary()
{
  Atom atom;atom.number_of_atoms=2;atom.number_of_beads=4;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;
  for(const int frames:{1,2}){const std::string path=write_sample_spool(".min_frames_"+std::to_string(frames),frames);RemoveTestFile cleanup{path};
    std::ifstream in(path,std::ios::binary);bool rejected=false;try{(void)read_header(in,frames,atom,box,300.0);}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
  const std::string path=write_sample_spool(".min_frames_3",3);RemoveTestFile cleanup{path};std::ifstream in(path,std::ios::binary);
  assert(read_header(in,3,atom,box,300.0).frame_count==3);
}

void test_fixed_probe_collection_checks_reference_branches()
{
  const std::string path=write_sample_spool(".branch",3,1.0,true,6.0);RemoveTestFile cleanup{path};
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};Box box{};
  box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  std::ifstream in(path,std::ios::binary);const SampleHeader header=read_header(in,0,atom,box,0.0,true);
  std::vector<double> r0(6,0.0);read_training_r0(in,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  bool rejected=false;try{(void)collect_fixed_probe_statistics(in,header,header.frame_count,2,r0,false,0.15,&box);}
  catch(const std::runtime_error& error){rejected=std::string(error.what()).find("branch limit 0.45")!=std::string::npos;}
  assert(rejected);
}

std::vector<double> snapshot_matrix(const std::string& text,const std::string& name,const int n)
{
  const std::string marker=name+"\n";const auto start=text.find(marker);assert(start!=std::string::npos);
  std::istringstream input(text.substr(start+marker.size()));std::vector<double> values(static_cast<std::size_t>(n)*n);
  for(double& value:values)if(!(input>>value))throw std::runtime_error("truncated response snapshot matrix");return values;
}

double snapshot_scalar(const std::string& text,const std::string& name)
{
  const std::string marker=name+" ";const auto start=text.find(marker);assert(start!=std::string::npos);
  std::istringstream input(text.substr(start+marker.size()));double value=0.0;if(!(input>>value))throw std::runtime_error("invalid response snapshot scalar");return value;
}

std::vector<double> snapshot_vector(const std::string& text,const std::string& name,const int n)
{
  const std::string marker=name+"\n";const auto start=text.find(marker);assert(start!=std::string::npos);
  std::istringstream input(text.substr(start+marker.size()));std::vector<double> values(n);
  for(int i=0;i<n;++i){int index=-1;if(!(input>>index>>values[i])||index!=i)throw std::runtime_error("invalid response snapshot vector");}return values;
}

void test_response_snapshot_preserves_recomputable_matrices()
{
  const std::string path="rpmd_ja_response_state_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  ResponseCheck response;response.probes=2;response.total_frames=12;response.validation_frames=8;response.response=0.5;response.ibp=std::sqrt(0.5*(0.14+std::sqrt(0.016)));response.probe_sources={"FIXED_RANDOM","TYPE_LOCAL"};
  response.observed_matrix={5.2,0.4,0.4,1.3};response.predicted_matrix={4.0,0.0,0.0,1.0};response.ibp_matrix={0.1,0.2,0.0,0.3};response.whitened_matrix={0.3,0.2,0.2,0.3};
  const double inverse_sqrt_two=1.0/std::sqrt(2.0);const std::vector<double> whitened_vectors={inverse_sqrt_two,inverse_sqrt_two,0.0,0.0};
  response.worst_response_coefficients=response_probe_basis_coefficients({0.5,0.0,0.0,1.0},whitened_vectors,2,0);response.worst_ibp_coefficients={0.0,1.0};
  SmallEigen eigen;eigen.n=2;eigen.values={0.1,0.5};eigen.vectors={inverse_sqrt_two,-inverse_sqrt_two,inverse_sqrt_two,inverse_sqrt_two};
  set_response_extremes(response,eigen,{0.5,0.0,0.0,1.0},{{1.0,0.0},{0.0,1.0}},true);
  response.ibp_band.diagnostic_stage="TAIL_CHECK";response.ibp_band.rejection_reason="BLOCK_TOO_SHORT";response.ibp_band.rejection_detail="product=fq probe_i=0 probe_j=1";
  write_response_state(path,response,{1.0,2.0},300.0,0.15);const std::string state=read_test_file(path);
  assert(state.find("IBP_DIAGNOSTIC stage=TAIL_CHECK bootstrap_started=0 configured_replicates=500 completed_levels=0 reason=BLOCK_TOO_SHORT product=fq probe_i=0 probe_j=1")!=std::string::npos);
  const auto ibp=snapshot_matrix(state,"IBP_residual_nonsymmetric",2),white=snapshot_matrix(state,"whitened_response_difference",2),observed=snapshot_matrix(state,"observed_covariance",2),predicted=snapshot_matrix(state,"predicted_covariance",2);
  const double e00=ibp[0]*ibp[0]+ibp[2]*ibp[2],e01=ibp[0]*ibp[1]+ibp[2]*ibp[3],e11=ibp[1]*ibp[1]+ibp[3]*ibp[3];
  const double lambda_max=0.5*(e00+e11+std::hypot(e00-e11,2.0*e01));
  const double response_lambda=0.5*(white[0]+white[3]+std::hypot(white[0]-white[3],2.0*white[1]));
  const double response_lambda_min=0.5*(white[0]+white[3]-std::hypot(white[0]-white[3],2.0*white[1]));
  assert(std::abs(std::max(std::abs(response_lambda),std::abs(response_lambda_min))-snapshot_scalar(state,"response_error"))<1e-14);
  assert(std::abs(std::sqrt(lambda_max)-snapshot_scalar(state,"ibp_error"))<1e-14);
  const auto coefficients=snapshot_vector(state,"worst_response_probe_basis_coefficients",2);double numerator=0.0,denominator=0.0,direct_numerator=0.0,direct_denominator=0.0;
  const std::vector<double> difference={observed[0]-predicted[0],observed[1]-predicted[1],observed[2]-predicted[2],observed[3]-predicted[3]},direct={inverse_sqrt_two,inverse_sqrt_two};
  for(int i=0;i<2;++i)for(int j=0;j<2;++j){numerator+=coefficients[i]*difference[2*i+j]*coefficients[j];denominator+=coefficients[i]*predicted[2*i+j]*coefficients[j];direct_numerator+=direct[i]*difference[2*i+j]*direct[j];direct_denominator+=direct[i]*predicted[2*i+j]*direct[j];}
  assert(std::abs(numerator/denominator-snapshot_scalar(state,"response_error"))<1e-14);
  assert(std::abs(direct_numerator/direct_denominator-snapshot_scalar(state,"response_error"))>1e-3);
  assert(state.find("FIXED_RANDOM")!=std::string::npos&&state.find("TYPE_LOCAL")!=std::string::npos);
}

void test_response_extremes_map_to_internal_coordinates()
{
  const double root_three=std::sqrt(3.0),root_two=std::sqrt(2.0);
  const std::vector<std::vector<double>> probes={
    {root_two/root_three,-1.0/root_three,0.0,0.0,0.0,0.0},
    {0.0,0.0,root_two/root_three,-1.0/root_three,0.0,0.0}};
  ResponseCheck response;response.probes=2;response.response=2.0;
  response.observed_matrix={3.2,0.0,0.0,3.0};response.predicted_matrix={4.0,0.0,0.0,1.0};response.worst_response_coefficients={0.0,1.0};
  response.whitened_matrix={-0.2,0.0,0.0,2.0};response.ibp_matrix={0.0,0.0,0.0,0.0};response.worst_ibp_coefficients={1.0,0.0};
  SmallEigen eigen;eigen.n=2;eigen.values={-0.2,2.0};eigen.vectors={1.0,0.0,0.0,1.0};
  set_response_extremes(response,eigen,{0.5,0.0,0.0,1.0},probes,true);
  check_close(response.response_ratio_min,0.8,1e-15);check_close(response.response_ratio_max,3.0,1e-15);
  assert(response.has_underprediction&&response.has_overprediction);
  assert(response.worst_underpredicted_direction.size()==6&&response.worst_overpredicted_direction.size()==6);
  const std::string path="rpmd_ja_response_extreme_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  write_response_state(path,response,{},300.0,0.15,0.15);const std::string state=read_test_file(path);
  check_close(snapshot_scalar(state,"response_ratio_min"),0.8,1e-15);check_close(snapshot_scalar(state,"response_ratio_max"),3.0,1e-15);
  check_close(snapshot_scalar(state,"worst_underpredicted_response_ratio"),3.0,1e-15);
  check_close(snapshot_scalar(state,"worst_overpredicted_response_ratio"),0.8,1e-15);
  const auto under_coefficients=snapshot_vector(state,"worst_underpredicted_probe_basis_coefficients",2);
  const auto over_coefficients=snapshot_vector(state,"worst_overpredicted_probe_basis_coefficients",2);
  const auto under_direction=snapshot_vector(state,"worst_underpredicted_direction_mass_weighted_internal_xyz_soa",6);
  const auto over_direction=snapshot_vector(state,"worst_overpredicted_direction_mass_weighted_internal_xyz_soa",6);
  double under_observed=0.0,under_predicted=0.0,over_observed=0.0,over_predicted=0.0;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j){under_observed+=under_coefficients[i]*response.observed_matrix[2*i+j]*under_coefficients[j];under_predicted+=under_coefficients[i]*response.predicted_matrix[2*i+j]*under_coefficients[j];
    over_observed+=over_coefficients[i]*response.observed_matrix[2*i+j]*over_coefficients[j];over_predicted+=over_coefficients[i]*response.predicted_matrix[2*i+j]*over_coefficients[j];}
  check_close(under_observed/under_predicted,3.0,1e-15);check_close(over_observed/over_predicted,0.8,1e-15);
  for(int i=0;i<6;++i){check_close(under_direction[i],probes[1][i],1e-15);check_close(over_direction[i],probes[0][i],1e-15);}
  check_close(under_direction[2]+root_two*under_direction[3],0.0,1e-15);
  check_close(over_direction[0]+root_two*over_direction[1],0.0,1e-15);
}

void test_covariance_training_statistics_include_symmetric_cross_terms()
{
  Graph graph;graph.type_pairs={{0,1}};Edge edge;edge.i=0;edge.j=1;edge.group=0;graph.edges.push_back(edge);
  const std::vector<double> r0={0,0,0,0,0,0},position={0,1,0,2,0,-3};std::vector<double> sums(6,0.0);
  add_covariance_frame_statistics(graph,position,r0,2,sums);
  const std::vector<double> expected={1,4,-6,4,-12,9};
  for(std::size_t i=0;i<expected.size();++i)check_close(sums[i],expected[i],1e-15);
  const std::vector<double> sqrt_mass={1,2,1,2,1,2},sqrt_atom={1,2};std::vector<double> q(6),action(6),theta(6,0.0);
  for(int axis=0;axis<3;++axis)for(int atom=0;atom<2;++atom)q[axis*2+atom]=sqrt_mass[axis*2+atom]*position[axis*2+atom];
  project_translation(q,sqrt_atom,2);
  for(int parameter=0;parameter<6;++parameter){std::fill(theta.begin(),theta.end(),0.0);theta[parameter]=1.0;
    apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,2,action);const double direct=std::inner_product(q.begin(),q.end(),action.begin(),0.0);
    check_close(direct,expected[parameter],1e-14);}
}

void test_m1_gpu_chain_matches_cpu_and_dense_reference()
{
  constexpr int n=3,d=3*n,base=6,raw=4,added=2,p=base+added;
  const std::vector<double> masses={1.0,4.0,9.0};std::vector<double> sqrt_atom(n),sqrt_mass(d);
  for(int atom=0;atom<n;++atom){sqrt_atom[atom]=std::sqrt(masses[atom]);for(int axis=0;axis<3;++axis)sqrt_mass[axis*n+atom]=sqrt_atom[atom];}
  Graph graph;graph.type_pairs={{0,0}};graph.edges={{0,1,0,{0,0,0}},{1,2,0,{0,0,0}}};
  graph.base_parameter_count=base;graph.parameter_count=p;graph.raw_local_parameter_count=raw;
  graph.local_base_mix={0.12,-0.03,0.04,0.02,0.0,0.0,-0.02,0.0,0.05,0.0,0.0,-0.04};
  graph.local_transform={1.0,0.2,-0.3,0.8,0.4,-0.1,0.2,1.0};
  rpmd_ja_local_basis::BridgeInput bridge{};bridge.oxygen1={0,{0,0,0}};bridge.hydrogen={1,{0,0,0}};
  bridge.oxygen2={2,{1,-1,0}};bridge.axis={0.4,0.5,0.7681145747868608};bridge.a0=0.31;
  bridge.s_parameter=0;bridge.ell_parameter=1;bridge.cross_parameter=2;bridge.transverse_parameter=3;
  graph.local_terms=rpmd_ja_local_basis::make_bridge_terms(bridge);graph.parameter_basis_norms.assign(p,1.0);

  std::vector<double> raw_hessian(static_cast<std::size_t>(d)*d,0.0);const double mass_sum=std::accumulate(masses.begin(),masses.end(),0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)for(int j=0;j<n;++j){
    const double pi=(i==j?1.0:0.0)-sqrt_atom[i]*sqrt_atom[j]/mass_sum;
    raw_hessian[static_cast<std::size_t>(axis*n+i)*d+axis*n+j]=sqrt_atom[i]*sqrt_atom[j]*pi;}
  std::stringstream raw_stream(std::ios::in|std::ios::out|std::ios::binary);
  raw_stream.write(reinterpret_cast<const char*>(raw_hessian.data()),raw_hessian.size()*sizeof(double));raw_stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(raw_stream,0,d,n,masses,sqrt_mass);
  std::vector<double> baseline_host(static_cast<std::size_t>(d)*d);
  assert(cudaMemcpy(baseline_host.data(),baseline.k,baseline_host.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> theta={0.12,0.01,-0.02,0.15,0.025,0.18,0.04,-0.035};

  std::vector<rpmd_ja_local_basis::Term> input_terms;
  for(const Edge& edge:graph.edges){const rpmd_ja_local_basis::Site first{edge.i,{0,0,0}};
    const rpmd_ja_local_basis::Site second{edge.j,{edge.image[0],edge.image[1],edge.image[2]}};
    const auto terms=rpmd_ja_local_basis::make_symmetric_edge_terms(first,second,6*edge.group);
    input_terms.insert(input_terms.end(),terms.begin(),terms.end());}
  for(rpmd_ja_local_basis::Term term:graph.local_terms){term.parameter+=base;input_terms.push_back(term);}
  const EffectiveM1Parameters effective=effective_m1_parameters(graph,theta);
  std::vector<double> input_theta(static_cast<std::size_t>(base+raw));
  std::copy(effective.base.begin(),effective.base.end(),input_theta.begin());
  std::copy(effective.raw.begin(),effective.raw.end(),input_theta.begin()+base);
  const auto blocks=rpmd_ja_local_basis::hessian_blocks(input_terms,input_theta.data(),base+raw,n,masses.data());
  std::vector<double> expected_matrix=baseline_host;
  for(const auto& block:blocks)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
    expected_matrix[static_cast<std::size_t>(a*n+block.row.atom)*d+b*n+block.column.atom]+=block.value[3*a+b];
  system.assemble(theta);std::vector<double> assembled(expected_matrix.size());
  assert(cudaMemcpy(assembled.data(),system.full,assembled.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  for(std::size_t i=0;i<assembled.size();++i)check_close(assembled[i],expected_matrix[i],2e-11);

  std::vector<double> vectors(static_cast<std::size_t>(d)*2);
  for(std::size_t i=0;i<vectors.size();++i)vectors[i]=std::sin(0.31*(i+1))-0.2*std::cos(0.17*(i+3));
  double *device_vectors=nullptr,*device_actions=nullptr;
  assert(cudaMalloc(reinterpret_cast<void**>(&device_vectors),vectors.size()*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&device_actions),vectors.size()*sizeof(double))==cudaSuccess);
  assert(cudaMemcpy(device_vectors,vectors.data(),vectors.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  system.apply_parameter_operator(p-1,device_vectors,device_actions,2);
  std::vector<double> gpu_actions(vectors.size());assert(cudaMemcpy(gpu_actions.data(),device_actions,gpu_actions.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  std::vector<double> unit_theta(p,0.0);unit_theta[p-1]=1.0;
  for(int column=0;column<2;++column){std::vector<double> input(d),expected(d),actual(d);
    std::copy(vectors.begin()+column*d,vectors.begin()+(column+1)*d,input.begin());
    apply_additive(graph,input,sqrt_mass,sqrt_atom,unit_theta,n,expected);
    std::copy(gpu_actions.begin()+column*d,gpu_actions.begin()+(column+1)*d,actual.begin());
    for(int i=0;i<d;++i)check_close(actual[i],expected[i],2e-11);}
  cudaFree(device_vectors);cudaFree(device_actions);
  DeviceLanczosWorkspace lanczos;lanczos.initialize(graph,sqrt_mass,sqrt_atom,1,theta.size());lanczos.set_theta(theta);
  assert(cudaMemcpy(lanczos.v,vectors.data(),static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  lanczos.additive_action(lanczos.v);std::vector<double> lanczos_action(d),cpu_action(d),lanczos_input(vectors.begin(),vectors.begin()+d);
  assert(cudaMemcpy(lanczos_action.data(),lanczos.add,static_cast<std::size_t>(d)*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  apply_additive(graph,lanczos_input,sqrt_mass,sqrt_atom,theta,n,cpu_action);
  for(int i=0;i<d;++i)check_close(lanczos_action[i],cpu_action[i],2e-10);

  const std::vector<double> position={0.1,0.7,-0.2,0.3,-0.1,0.6,0.2,0.8,-0.4},r0(d,0.0);
  std::vector<double> statistics(p,0.0),block_statistics(p,0.0);
  add_covariance_frame_statistics(graph,position,r0,n,statistics,nullptr,&block_statistics);
  for(int output=0;output<p;++output){std::vector<double> basis_theta(p,0.0);basis_theta[output]=1.0;
    const EffectiveM1Parameters basis_effective=effective_m1_parameters(graph,basis_theta);
    std::vector<double> mapped(static_cast<std::size_t>(base+raw));
    std::copy(basis_effective.base.begin(),basis_effective.base.end(),mapped.begin());
    std::copy(basis_effective.raw.begin(),basis_effective.raw.end(),mapped.begin()+base);
    const auto value=rpmd_ja_local_basis::evaluate(input_terms,mapped.data(),base+raw,position.data(),n);
    check_close(statistics[output],2.0*value.energy,2e-11);
    check_close(block_statistics[output],statistics[output],2e-14);}

  Graph m0=graph;m0.parameter_count=base;m0.raw_local_parameter_count=0;m0.local_terms.clear();
  m0.local_transform.clear();m0.local_base_mix.clear();m0.parameter_basis_norms.clear();
  DeviceCovarianceSystem m0_system(baseline,m0,sqrt_mass,masses);
  std::vector<double> zero_extra=theta;zero_extra[base]=zero_extra[base+1]=0.0;
  system.assemble(zero_extra);std::vector<double> m1_zero(assembled.size());
  assert(cudaMemcpy(m1_zero.data(),system.full,m1_zero.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  const std::vector<double> m0_theta(theta.begin(),theta.begin()+base);m0_system.assemble(m0_theta);
  std::vector<double> m0_matrix(assembled.size());assert(cudaMemcpy(m0_matrix.data(),m0_system.full,m0_matrix.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  for(std::size_t i=0;i<m0_matrix.size();++i)check_close(m1_zero[i],m0_matrix[i],2e-12);
  std::vector<double> m1_action(d),m0_action(d);std::vector<double> q(vectors.begin(),vectors.begin()+d);
  apply_additive(graph,q,sqrt_mass,sqrt_atom,zero_extra,n,m1_action);apply_additive(m0,q,sqrt_mass,sqrt_atom,m0_theta,n,m0_action);
  for(int i=0;i<d;++i)check_close(m1_action[i],m0_action[i],2e-14);

  constexpr double epsilon=0.05;double logdet=0.0,shifted_logdet=0.0;std::vector<double> trace,shifted_trace;
  assert(system.evaluate_factors(theta,epsilon,logdet,shifted_logdet,&trace,&shifted_trace,false,true));
  std::vector<double> full_matrix(expected_matrix.size());
  assert(cudaMemcpy(full_matrix.data(),system.full,full_matrix.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  const auto basis=internal_basis(sqrt_atom);const int r=static_cast<int>(basis.size());
  std::vector<double> restricted(static_cast<std::size_t>(r)*r,0.0);
  for(int column=0;column<r;++column){std::vector<double> action(d,0.0);
    for(int i=0;i<d;++i)for(int j=0;j<d;++j)action[i]+=full_matrix[static_cast<std::size_t>(i)*d+j]*basis[column][j];
    for(int row=0;row<r;++row)restricted[static_cast<std::size_t>(row)*r+column]=compensated_dot(basis[row],action);}
  auto dense_inverse=[&](const double shift){std::vector<double> matrix=restricted;
    for(int i=0;i<r;++i)matrix[static_cast<std::size_t>(i)*r+i]-=shift;std::vector<double> inverse(static_cast<std::size_t>(d)*d,0.0);
    for(int column=0;column<r;++column){std::vector<double> unit(static_cast<std::size_t>(r),0.0);unit[column]=1.0;
      const auto solved=solve_dense_reference(matrix,unit,r);
      for(int row=0;row<r;++row)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
        inverse[static_cast<std::size_t>(i)*d+j]+=basis[row][i]*solved[row]*basis[column][j];}
    return inverse;};
  const auto inverse=dense_inverse(0.0),shifted_inverse=dense_inverse(epsilon);
  for(int output=0;output<p;++output){std::vector<double> basis_theta(p,0.0);basis_theta[output]=1.0;
    const EffectiveM1Parameters basis_effective=effective_m1_parameters(graph,basis_theta);
    std::vector<double> mapped(static_cast<std::size_t>(base+raw));
    std::copy(basis_effective.base.begin(),basis_effective.base.end(),mapped.begin());
    std::copy(basis_effective.raw.begin(),basis_effective.raw.end(),mapped.begin()+base);
    const auto basis_blocks=rpmd_ja_local_basis::hessian_blocks(input_terms,mapped.data(),base+raw,n,masses.data());
    double expected_trace=0.0,expected_shifted_trace=0.0;
    for(const auto& block:basis_blocks)for(int a=0;a<3;++a)for(int b=0;b<3;++b){
      const int row=a*n+block.row.atom,column=b*n+block.column.atom;const double value=block.value[3*a+b];
      expected_trace+=value*inverse[static_cast<std::size_t>(column)*d+row];
      expected_shifted_trace+=value*shifted_inverse[static_cast<std::size_t>(column)*d+row];}
    check_close(trace[output],expected_trace,3e-7);check_close(shifted_trace[output],expected_shifted_trace,3e-7);}
}

void test_m1_bridge_images_share_hydrogen_anchored_branch()
{
  constexpr int n=3;Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;
  box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  const std::vector<double> r0={3.0,0.1,7.5,0.0,0.0,0.0,0.0,0.0,0.0};
  const std::vector<int> types={0,1,0};const std::vector<std::string> symbols={"O","H","O"};
  const std::vector<double> masses={16.0,1.0,16.0};constexpr double cutoff=5.2;
  Graph graph=make_graph(r0,types,box,cutoff);
  const auto& metadata=build_m1_local_basis(graph,r0,types,symbols,masses,box,cutoff,1);
  assert(metadata.hydrogen_count==1&&metadata.bridge_candidate_count==1&&metadata.accepted_bridge_count==1);
  assert(metadata.frozen_bridges.size()==1);
  const Graph::FrozenBridge& bridge=metadata.frozen_bridges.front();
  assert(bridge.accepted&&!bridge.ambiguous&&bridge.oxygen_candidates.size()==2);
  assert(bridge.oxygen1.atom==2&&bridge.oxygen1.image[0]==-1);
  assert(bridge.hydrogen.atom==1&&bridge.hydrogen.image[0]==0);
  assert(bridge.oxygen2.atom==0&&bridge.oxygen2.image[0]==0);
  int independent_image[3];const auto independent_oo=mic_displacement(r0,n,0,2,box,independent_image);
  const double independent_distance=std::sqrt(independent_oo[0]*independent_oo[0]+independent_oo[1]*independent_oo[1]+independent_oo[2]*independent_oo[2]);
  check_close(bridge.oo,5.5,1e-14);check_close(independent_distance,4.5,1e-14);
  assert(bridge.oo>independent_distance+0.9);

  Graph low_support=make_graph(r0,types,box,cutoff);
  const auto& low_metadata=build_m1_local_basis(low_support,r0,types,symbols,masses,box,cutoff,3);
  assert(low_metadata.minimum_support_count==3);
  assert(low_metadata.added_parameter_count==0&&low_support.parameter_count==low_support.base_parameter_count);
  assert(low_support.local_terms.empty());
  assert(low_metadata.support_policy.find("DROP_BEFORE_OPERATOR_RESIDUALIZATION")!=std::string::npos);
}

void test_fixed_physical_response_probes_are_frozen_and_ranked()
{
  constexpr int n=5;Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=12.0;
  box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=1.0/12.0;
  const std::vector<double> r0={1.0,2.0,3.0,2.0,7.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0};
  const std::vector<int> types={0,1,0,2,3};
  const std::vector<std::string> symbols={"O","H","O","Na","Cl"};
  const std::vector<double> sqrt_mass={4.0,1.0,4.0,std::sqrt(22.989769),std::sqrt(35.45)};
  constexpr double cutoff=4.0;
  const auto channels=make_fixed_physical_probes(r0,types,symbols,sqrt_mass,box,cutoff);
  const auto repeated=make_fixed_physical_probes(r0,types,symbols,sqrt_mass,box,cutoff);
  assert(channels.size()==7&&repeated.size()==channels.size());
  bool found_oho_near=false,found_oxygen_x=false,found_salt_z=false;
  for(std::size_t i=0;i<channels.size();++i){
    assert(channels[i].source==repeated[i].source&&channels[i].vector==repeated[i].vector);
    found_oho_near=found_oho_near||channels[i].source=="FIXED_PHYSICAL_OHO_NEAR_SALT";
    found_oxygen_x=found_oxygen_x||channels[i].source=="FIXED_PHYSICAL_OXYGEN_SKELETON_X";
    found_salt_z=found_salt_z||channels[i].source=="FIXED_PHYSICAL_SALT_POLARITY_Z";}
  assert(found_oho_near&&found_oxygen_x&&found_salt_z);
  std::vector<std::string> first_sources,second_sources;
  std::vector<ProbeOrigin> first_origins,second_origins;
  const auto first=make_probes({},sqrt_mass,types,n,&first_sources,&first_origins,&channels);
  const auto second=make_probes({},sqrt_mass,types,n,&second_sources,&second_origins,&channels);
  assert(first_sources==second_sources&&first.size()==second.size());
  assert(first_sources.size()>=channels.size());
  for(std::size_t i=0;i<channels.size();++i)assert(first_sources[i]==channels[i].source);
  for(std::size_t i=0;i<first.size();++i)assert(first[i]==second[i]);
}

struct IndependentAdditiveOperators
{
  std::vector<double> k;
  std::array<std::vector<double>,3> h;
};

IndependentAdditiveOperators independently_assemble_additive_operators(const Graph& graph,
  const std::vector<double>& theta,const std::vector<double>& positions,const Box& box,const int n)
{
  const int d=3*n;const EffectiveM1Parameters effective=effective_m1_parameters(graph,theta);
  IndependentAdditiveOperators result;result.k.assign(static_cast<std::size_t>(d)*d,0.0);
  for(auto& matrix:result.h)matrix.assign(static_cast<std::size_t>(d)*d,0.0);
  auto coefficient=[](const rpmd_ja_local_basis::Term& term,const int site,const int axis){
    const auto& value=term.coefficient[site];return axis==0?value.x:axis==1?value.y:value.z;};
  auto symmetric_component=[](const int first,const int second){const int low=std::min(first,second),high=std::max(first,second);
    return low==0?(high==0?0:(high==1?1:2)):(low==1?(high==1?3:4):5);};
  for(int center=0;center<n;++center)for(const SiteEdge& edge:graph.sites[center]){
    if(edge.edge_index<0||edge.group<0)continue;double geometry[3]={};
    for(int alpha=0;alpha<3;++alpha){geometry[alpha]=positions[alpha*n+edge.j]-positions[alpha*n+center];
      for(int cell_axis=0;cell_axis<3;++cell_axis)geometry[alpha]+=box.cpu_h[3*alpha+cell_axis]*edge.image[cell_axis];}
    for(int mu=0;mu<3;++mu)for(int axis=0;axis<3;++axis){
      const double block=0.5*effective.base[6*edge.group+symmetric_component(mu,axis)];
      const int row[2]={axis*n+edge.j,axis*n+center},column[2]={mu*n+edge.j,mu*n+center};
      const double sign[2]={1.0,-1.0};
      for(int first=0;first<2;++first)for(int second=0;second<2;++second)
        result.k[static_cast<std::size_t>(row[first])*d+column[second]]+=sign[first]*block*sign[second];
      for(int alpha=0;alpha<3;++alpha){const int velocity=mu*n+edge.j;
        result.h[alpha][static_cast<std::size_t>(velocity)*d+axis*n+edge.j]-=geometry[alpha]*block;
        result.h[alpha][static_cast<std::size_t>(velocity)*d+axis*n+center]+=geometry[alpha]*block;}
    }
  }
  for(const auto& term:graph.local_terms){
    const double parameter=effective.raw[static_cast<std::size_t>(term.parameter)]*term.weight;
    for(int row=0;row<term.atom_count;++row)for(int column=0;column<term.atom_count;++column)
      for(int a=0;a<3;++a)for(int b=0;b<3;++b)
        result.k[static_cast<std::size_t>(a*n+term.site[row].atom)*d+b*n+term.site[column].atom]+=
          parameter*coefficient(term,row,a)*coefficient(term,column,b);
    for(int center=0;center<term.atom_count;++center){const double scale=parameter*term.energy_share[center];
      for(int anchor=0;anchor<term.atom_count;++anchor)if(anchor!=center){
        const int velocity_atom=term.site[anchor].atom;double geometry[3]={};
        for(int alpha=0;alpha<3;++alpha){geometry[alpha]=positions[alpha*n+velocity_atom]-positions[alpha*n+term.site[center].atom];
          for(int cell_axis=0;cell_axis<3;++cell_axis)
            geometry[alpha]+=box.cpu_h[3*alpha+cell_axis]*(term.site[anchor].image[cell_axis]-term.site[center].image[cell_axis]);}
        for(int mu=0;mu<3;++mu)for(int displacement=0;displacement<term.atom_count;++displacement)if(displacement!=center)
          for(int alpha=0;alpha<3;++alpha)for(int axis=0;axis<3;++axis){
            const double value=-geometry[alpha]*scale*coefficient(term,anchor,mu)*coefficient(term,displacement,axis);
            const int velocity=mu*n+velocity_atom;
            result.h[alpha][static_cast<std::size_t>(velocity)*d+axis*n+term.site[displacement].atom]+=value;
            result.h[alpha][static_cast<std::size_t>(velocity)*d+axis*n+term.site[center].atom]-=value;
          }
      }
    }
  }
  return result;
}

void test_native_m1_writer_reader_cross_neighbor_images_and_transport()
{
  constexpr int n=3,d=3*n;Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;
  box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  const std::vector<double> r0={3.0,0.1,7.5,0.0,0.0,0.0,0.0,0.0,0.0};
  const std::vector<int> types={0,1,0};const std::vector<std::string> symbols={"O","H","O"};
  const std::vector<double> masses={16.0,1.0,16.0};constexpr double cutoff=5.2;
  Graph third=make_graph(r0,types,box,cutoff);
  Graph m0_graph=make_graph(r0,types,box,cutoff);
  const NativeAdditivePackageEstimate m0_support_estimate=estimate_native_additive_package_size(m0_graph,n);
  std::uint64_t m0_formula=native_additive_package_header_bytes;
  for(const auto& site:m0_graph.sites){
    const std::uint64_t z=site.size(),q=3*z;std::uint64_t base_slots=0;
    for(const SiteEdge& edge:site)if(edge.edge_index>=0)++base_slots;
    const std::uint64_t support_entries=std::min<std::uint64_t>(q*q,9*base_slots);
    const std::uint64_t support_blocks=std::min<std::uint64_t>(z*z,base_slots);
    const std::uint64_t dense_bytes=8*q*q,scalar_bytes=8+16*support_entries;
    const std::uint64_t block_bytes=8+80*support_blocks;
    m0_formula+=sizeof(std::int32_t)+16*z+q*sizeof(double)+sizeof(std::uint32_t)+
      std::min(dense_bytes,std::min(scalar_bytes,block_bytes));
  }
  assert(m0_support_estimate.bytes==m0_formula);
  const auto& metadata=build_m1_local_basis(third,r0,types,symbols,masses,box,cutoff,1);
  assert(metadata.accepted_bridge_count==1&&metadata.added_parameter_count>0);
  const int p=third.parameter_count,base=third.base_parameter_count;
  std::vector<double> theta(static_cast<std::size_t>(p),0.0);
  for(int column=base;column<p;++column)theta[column]=0.13+0.07*(column-base);
  std::vector<std::array<double,3>> edge_linear(third.edges.size());
  for(const auto& term:third.local_terms){double share_sum=0.0;for(int site=0;site<term.atom_count;++site)share_sum+=term.energy_share[site];
    assert(std::abs(share_sum-1.0)<1e-12);}
  Graph half=third;
  for(auto& term:half.local_terms)if(term.atom_count==3){
    term.energy_share[0]=0.5;term.energy_share[1]=0.25;term.energy_share[2]=0.25;}
  RpmdJAReference reference;reference.number_of_atoms=n;reference.temperature=300.0;
  reference.masses=masses;reference.positions=r0;for(int i=0;i<9;++i)reference.cell[i]=box.cpu_h[i];
  const std::vector<double> raw_gradient(d,0.0);
  struct RoundTrip{RpmdJAAdditiveData package;std::vector<NativeAdditiveSiteBlock> blocks;};
  auto round_trip=[&](const Graph& graph,const char* suffix,const std::uint32_t version,const std::uint32_t response_status=1U){
    const std::string path=std::string("rpmd_ja_native_m1_writer_")+suffix+"_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
    std::vector<NativeAdditiveSiteBlock> blocks;std::uint64_t estimated_bytes=0;
    {
      std::ofstream out(path,std::ios::binary);assert(out.good());NativeAdditivePackageHeader header;
      header.version=version;header.number_of_atoms=n;header.beads=4;header.temperature=300.0;header.epsilon=0.05;
      header.response=version==7?(response_status==0?0.01:response_status==3?std::numeric_limits<double>::quiet_NaN():0.05):0.01;
      header.response_tolerance=0.02;header.training_frames=1;header.validation_frames=1;
      if(version==7){header.use_flags=3;header.response_status=response_status;}
      const NativeAdditivePackageEstimate prefit_estimate=estimate_native_additive_package_size(graph,n,version);
      estimated_bytes=write_native_additive_package(out,header,graph,theta,edge_linear,n,prefit_estimate.bytes);assert(out.good());
    }
    const NativeAdditivePackageEstimate prefit_estimate=estimate_native_additive_package_size(graph,n,version);
    assert(prefit_estimate.bytes>=estimated_bytes&&prefit_estimate.maximum_site_neighbors>0);
    std::ifstream sized(path,std::ios::binary|std::ios::ate);assert(sized.good());
    assert(static_cast<std::uint64_t>(sized.tellg())==estimated_bytes);sized.close();
    blocks=build_native_additive_site_blocks(graph,theta,edge_linear,n);
    {
      std::ifstream header(path,std::ios::binary);assert(header.good());char magic[8]{};header.read(magic,sizeof(magic));
      const std::uint32_t written_version=rpmd_ja_additive_read<std::uint32_t>(header);
      assert(written_version==version);header.seekg(static_cast<std::streamoff>(native_additive_package_header_bytes+
        (version>=7?native_additive_graph_reference_metadata_bytes:0)));
      const std::int32_t first_site_count=rpmd_ja_additive_read<std::int32_t>(header);
      assert(first_site_count==static_cast<std::int32_t>(blocks.front().edges.size()));
    }
    const RpmdJAAdditiveData package=read_rpmd_ja_additive(path,"test-raw.qraw",reference,4,raw_gradient);
    assert(package.internal_mass_com==(version==6||version==7));
    assert(package.diagnostic==(version==7));
    if(version==7){const char* expected_status=response_status==0?"PASS":response_status==1?"FAIL":response_status==2?"INCONCLUSIVE":"NOT_MEASURED";
      assert(package.numerical_status=="PASS"&&package.response_status==expected_status);}
    assert(std::remove(path.c_str())==0);return RoundTrip{package,std::move(blocks)};
  };
  const RoundTrip one_third=round_trip(third,"third",5);const RoundTrip one_half=round_trip(half,"half",5);
  const RoundTrip internal_com=round_trip(third,"internal_com",6);assert(internal_com.package.internal_mass_com);
  const RoundTrip diagnostic_com=round_trip(third,"diagnostic_com",7,1);
  assert(diagnostic_com.package.internal_mass_com&&diagnostic_com.package.diagnostic&&
    diagnostic_com.package.numerical_status=="PASS"&&diagnostic_com.package.response_status=="FAIL");
  assert(round_trip(third,"diagnostic_pass",7,0).package.response_status=="PASS");
  assert(round_trip(third,"diagnostic_inconclusive",7,2).package.response_status=="INCONCLUSIVE");
  assert(round_trip(third,"diagnostic_not_measured",7,3).package.response_status=="NOT_MEASURED");
  auto dense=[](const RpmdJASparseRows& sparse){
    const int dimension=static_cast<int>(sparse.offsets.size())-1;std::vector<double> result(static_cast<std::size_t>(dimension)*dimension,0.0);
    for(int row=0;row<dimension;++row)for(std::size_t index=sparse.offsets[row];index<sparse.offsets[row+1];++index)
      result[static_cast<std::size_t>(row)*dimension+sparse.columns[index]]=sparse.values[index];return result;};
  const IndependentAdditiveOperators expected=independently_assemble_additive_operators(third,theta,r0,box,n);
  const IndependentAdditiveOperators expected_half=independently_assemble_additive_operators(half,theta,r0,box,n);
  const std::string state_path="rpmd_ja_m1_mapping_replay_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  RemoveTestFile state_cleanup{state_path};
  CovarianceFitResult saved_fit;saved_fit.theta=theta;saved_fit.theta_start=theta;
  saved_fit.theta_baseline.assign(theta.size(),0.0);saved_fit.coordinate_scale.assign(theta.size(),1.0);
  saved_fit.statistics.assign(theta.size(),0.0);saved_fit.barrier_tau=1.0;
  RpmdJANativeFitOptions saved_options;saved_options.cutoff=cutoff;saved_options.epsilon=0.01;
  saved_options.fd_step=1e-4;saved_options.sample_interval=1;
  SampleHeader saved_header{};saved_header.temperature=300.0;saved_header.beads=4;
  {
    Graph to_save=third;QrawIdentity identity{};
    write_covariance_state(state_path,saved_fit,0.0,"test.samples","test.qraw",saved_options,
      saved_header,identity,to_save,6,4,0.01);
    to_save=Graph{};
  }
  std::ifstream state_file(state_path);const auto saved_fields=read_covariance_state_fields(state_file);
  assert(saved_fields.at("state_format_version")=="4"&&saved_fields.at("parameter_family")=="M1");
  Graph rebuilt=make_graph(r0,types,box,cutoff);
  build_m1_local_basis(rebuilt,r0,types,symbols,masses,box,cutoff,1);
  assert(std::stoull(saved_fields.at("M1_basis_fingerprint_fnv64"))==covariance_m1_basis_fingerprint(rebuilt));
  std::istringstream saved_theta(saved_fields.at("theta"));std::size_t saved_count=0;saved_theta>>saved_count;
  assert(saved_count==theta.size());std::vector<double> replay_theta(saved_count);
  for(double& value:replay_theta)assert(static_cast<bool>(saved_theta>>value));
  const auto replay=independently_assemble_additive_operators(rebuilt,replay_theta,r0,box,n);
  for(std::size_t i=0;i<expected.k.size();++i)check_close(replay.k[i],expected.k[i],1e-12);
  for(int axis=0;axis<3;++axis)for(std::size_t i=0;i<expected.h[axis].size();++i)
    check_close(replay.h[axis][i],expected.h[axis][i],1e-12);
  Graph changed_share=rebuilt;changed_share.local_terms.front().energy_share[0]+=0.01;
  assert(covariance_m1_basis_fingerprint(changed_share)!=covariance_m1_basis_fingerprint(rebuilt));
  Graph changed_image=rebuilt;changed_image.local_terms.front().site[0].image[0]+=1;
  assert(covariance_m1_basis_fingerprint(changed_image)!=covariance_m1_basis_fingerprint(rebuilt));
  Graph changed_transform=rebuilt;changed_transform.local_transform[0]+=0.01;
  assert(covariance_m1_basis_fingerprint(changed_transform)!=covariance_m1_basis_fingerprint(rebuilt));
  const auto k_third=dense(one_third.package.k),k_half=dense(one_half.package.k);
  for(std::size_t i=0;i<expected.k.size();++i){check_close(k_third[i],expected.k[i],3e-10);check_close(k_half[i],expected_half.k[i],3e-10);}
  for(int axis=0;axis<3;++axis)for(int row=0;row<d;++row){double sum=0.0;
    for(int atom=0;atom<n;++atom)sum+=k_third[static_cast<std::size_t>(row)*d+axis*n+atom];check_close(sum,0.0,3e-10);}
  bool cross_neighbor=false;
  for(int a=0;a<3;++a)cross_neighbor|=std::abs(k_third[static_cast<std::size_t>(a*n+0)*d+a*n+2])>1e-10;
  assert(cross_neighbor);
  bool image_edge=false,off_diagonal_site_block=false;
  for(const auto& block:one_third.blocks){
    for(const SiteEdge& edge:block.edges)image_edge|=edge.image[0]!=0||edge.image[1]!=0||edge.image[2]!=0;
    const int q=static_cast<int>(block.edges.size())*3;
    for(std::size_t left=0;left<block.edges.size();++left)for(std::size_t right=0;right<block.edges.size();++right)
      if(left!=right&&block.edges[left].j!=block.edges[right].j)
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)
          off_diagonal_site_block|=std::abs(block.quadratic[(3*left+a)*q+3*right+b])>1e-10;
  }
  assert(image_edge&&off_diagonal_site_block);
  bool nonsymmetric=false,gauge_h_changed=false;
  for(int alpha=0;alpha<3;++alpha){const auto h_third=dense(one_third.package.h[alpha]);const auto h_half=dense(one_half.package.h[alpha]);
    for(std::size_t i=0;i<expected.h[alpha].size();++i)check_close(h_third[i],expected.h[alpha][i],3e-10);
    for(int row=0;row<d;++row)for(int column=0;column<d;++column){
      nonsymmetric|=std::abs(h_third[static_cast<std::size_t>(row)*d+column]-h_third[static_cast<std::size_t>(column)*d+row])>1e-10;
      gauge_h_changed|=std::abs(h_third[static_cast<std::size_t>(row)*d+column]-h_half[static_cast<std::size_t>(row)*d+column])>1e-10;}}
  assert(nonsymmetric&&gauge_h_changed);
}

std::vector<std::uint32_t> read_native_package_quadratic_modes(const std::string& path,const int n)
{
  std::ifstream in(path,std::ios::binary);assert(in.good());char magic[8]{};in.read(magic,sizeof(magic));
  const auto version=rpmd_ja_additive_read<std::uint32_t>(in);(void)rpmd_ja_additive_read<std::uint32_t>(in);
  (void)rpmd_ja_additive_read<std::int32_t>(in);(void)rpmd_ja_additive_read<std::int32_t>(in);
  (void)rpmd_ja_additive_read<std::uint64_t>(in);for(int i=0;i<4;++i)(void)rpmd_ja_additive_read<double>(in);
  (void)rpmd_ja_additive_read<std::uint64_t>(in);(void)rpmd_ja_additive_read<std::uint64_t>(in);
  if(version>=7)for(int i=0;i<3;++i)(void)rpmd_ja_additive_read<std::uint32_t>(in);
  assert(static_cast<std::uint64_t>(in.tellg())==native_additive_package_header_bytes+
    (version>=7?native_additive_graph_reference_metadata_bytes:0));
  std::vector<std::uint32_t> modes;for(int center=0;center<n;++center){
    const int z=rpmd_ja_additive_read<std::int32_t>(in),q=3*z;assert(z>=0);
    in.seekg(static_cast<std::streamoff>(16*z+sizeof(double)*q),std::ios::cur);
    const std::uint32_t mode=rpmd_ja_additive_read<std::uint32_t>(in);modes.push_back(mode);
    if(mode==0)in.seekg(static_cast<std::streamoff>(sizeof(double)*q*q),std::ios::cur);
    else{const std::uint64_t count=rpmd_ja_additive_read<std::uint64_t>(in);
      const std::uint64_t bytes_per_item=mode==1?2*sizeof(std::int32_t)+sizeof(double):2*sizeof(std::int32_t)+9*sizeof(double);
      assert(mode==1||mode==2);in.seekg(static_cast<std::streamoff>(count*bytes_per_item),std::ios::cur);}
    assert(in.good());}
  return modes;
}

void test_native_additive_package_accounting_modes_and_bounds()
{
  const auto serialize=[&](const std::string& name,const Graph& graph,const std::vector<double>& theta,
    const std::vector<double>& positions,const std::vector<double>& masses,const int n){
    const std::string path="rpmd_ja_package_accounting_"+name+"_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";RemoveTestFile cleanup{path};
    const std::vector<std::array<double,3>> edge_linear(graph.edges.size());NativeAdditivePackageHeader header;
    header.version=7;header.number_of_atoms=n;header.beads=4;header.source_fingerprint=0;
    header.temperature=300.0;header.epsilon=1e-8;header.response=0.05;header.response_tolerance=0.02;
    header.training_frames=1;header.validation_frames=1;header.use_flags=3;header.response_status=1;
    const NativeAdditivePackageEstimate estimate=estimate_native_additive_package_size(graph,n,header.version);
    std::uint64_t written=0;{std::ofstream out(path,std::ios::binary);assert(out.good());
      written=write_native_additive_package(out,header,graph,theta,edge_linear,n,estimate.bytes);out.close();assert(out.good());}
    std::ifstream sized(path,std::ios::binary|std::ios::ate);assert(sized.good());
    const auto actual=static_cast<std::uint64_t>(sized.tellg());assert(written==actual&&actual<=estimate.bytes);
    RpmdJAReference reference;reference.number_of_atoms=n;reference.temperature=300.0;reference.masses=masses;reference.positions=positions;
    reference.cell[0]=reference.cell[4]=reference.cell[8]=10.0;const std::vector<double> gradient(static_cast<std::size_t>(3*n),0.0);
    const RpmdJAAdditiveData parsed=read_rpmd_ja_additive(path,"test-raw.qraw",reference,4,gradient);
    assert(parsed.diagnostic&&parsed.internal_mass_com&&parsed.response_status=="FAIL");
    return std::make_tuple(actual,read_native_package_quadratic_modes(path,n),estimate.bytes);};

  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  const std::vector<double> positions2={2.0,4.0,2.0,4.0,2.0,4.0};const std::vector<int> types2={0,0};
  const Graph dense_graph=make_graph(positions2,types2,box,4.0);std::vector<double> dense_theta(6,0.1);
  const auto dense=serialize("dense",dense_graph,dense_theta,positions2,{1.0,1.0},2);
  assert(std::get<0>(dense)==std::get<2>(dense)&&std::get<1>(dense)==std::vector<std::uint32_t>({0,0}));

  const std::vector<double> positions3={2.0,4.0,6.0,2.0,4.0,6.0,2.0,4.0,6.0};const std::vector<int> types3={0,0,0};
  const Graph chain=make_graph(positions3,types3,box,3.6);assert(chain.sites[1].size()==2);
  const auto zeros=serialize("zero",chain,std::vector<double>(6,0.0),positions3,{1.0,1.0,1.0},3);
  assert(std::get<0>(zeros)<=std::get<2>(zeros)&&std::get<1>(zeros)[1]==1);
  const auto signed_zeros=serialize("signed_zero",chain,std::vector<double>(6,-0.0),positions3,{1.0,1.0,1.0},3);
  assert(std::get<0>(signed_zeros)==std::get<2>(signed_zeros)&&std::get<1>(signed_zeros)[1]==2);
  std::vector<double> sparse_theta(6,0.0);sparse_theta[0]=0.2;
  const auto scalar=serialize("scalar",chain,sparse_theta,positions3,{1.0,1.0,1.0},3);
  assert(std::get<0>(scalar)<=std::get<2>(scalar)&&std::get<1>(scalar)[1]==1);
  const auto blocks=serialize("blocks",chain,std::vector<double>(6,0.1),positions3,{1.0,1.0,1.0},3);
  assert(std::get<0>(blocks)==std::get<2>(blocks)&&std::get<1>(blocks)[1]==2);

  std::ostringstream rejected;NativeAdditivePackageHeader header;header.version=7;header.number_of_atoms=2;header.beads=4;
  header.temperature=300.0;header.epsilon=1e-8;header.response=0.05;header.response_tolerance=0.02;
  header.training_frames=1;header.validation_frames=1;header.use_flags=3;header.response_status=1;
  const auto exact=estimate_native_additive_package_size(dense_graph,2,7);bool bad_bound_rejected=false;
  try{(void)write_native_additive_package(rejected,header,dense_graph,dense_theta,
      std::vector<std::array<double,3>>(dense_graph.edges.size()),2,exact.bytes-1);}
  catch(const std::runtime_error& error){bad_bound_rejected=std::string(error.what()).find("conservative support bound")!=std::string::npos;}
  assert(bad_bound_rejected);
}

Graph make_two_atom_covariance_graph()
{
  Graph graph;graph.type_pairs={{0,1}};Edge edge;edge.i=0;edge.j=1;edge.group=0;graph.edges.push_back(edge);return graph;
}

void test_covariance_statistics_read_only_training_frames()
{
  SampleHeader header{};header.n=2;header.frames=0;Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> r0(6,0.0);std::stringstream spool(std::ios::in|std::ios::out|std::ios::binary);
  for(int frame=0;frame<2;++frame){const double step=static_cast<double>(frame);const double position[6]={0.0,1.0+frame,0.0,0.0,0.0,0.0};
    const double force[6]={};spool.write(reinterpret_cast<const char*>(&step),sizeof(step));
    spool.write(reinterpret_cast<const char*>(position),sizeof(position));spool.write(reinterpret_cast<const char*>(force),sizeof(force));}
  std::vector<std::vector<double>> blocks;std::vector<std::uint64_t> block_frames;
  const std::vector<double> statistics=collect_covariance_fit_statistics(spool,header,3,2,graph,r0,300.0,nullptr,&blocks,&block_frames);
  assert(statistics.size()==6);check_close(statistics[0],5.0/(K_B*300.0*2.0),1e-12);
  assert(blocks.size()==2&&block_frames==std::vector<std::uint64_t>({1,1}));
  check_close(blocks[0][0],1.0/(K_B*300.0),1e-12);check_close(blocks[1][0],4.0/(K_B*300.0),1e-12);
}

void test_blind_fixed_r0_admission_reuses_training_channels_and_block_plan()
{
  SampleHeader header{};header.n=2;header.masses={1.0,1.0};header.frames=0;
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> r0(6,0.0);
  const std::vector<double> sqrt_mass(6,1.0);
  const std::vector<FixedPhysicalProbe> probes={{"frozen_probe",{1.0,-1.0,0.0,0.0,0.0,0.0}}};
  auto samples=[&](const double shift){std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);
    for(int frame=0;frame<8;++frame){const double step=frame;const double x=1.0+0.05*(frame%3);
      const double position[6]={shift,shift+x,0.0,0.0,0.0,0.0};const double force[6]={};
      stream.write(reinterpret_cast<const char*>(&step),sizeof(step));
      stream.write(reinterpret_cast<const char*>(position),sizeof(position));
      stream.write(reinterpret_cast<const char*>(force),sizeof(force));}
    return stream;};
  std::stringstream training=samples(0.0),blind_pass=samples(0.0),blind_shifted=samples(10.0);
  const auto training_channels=collect_fixed_r0_admission_channels(training,header,8,graph,r0,300.0,
    probes,sqrt_mass,1ULL<<20);
  const auto matching_channels=collect_fixed_r0_admission_channels(blind_pass,header,8,graph,r0,300.0,
    probes,sqrt_mass,1ULL<<20);
  const auto shifted_channels=collect_fixed_r0_admission_channels(blind_shifted,header,8,graph,r0,300.0,
    probes,sqrt_mass,1ULL<<20);
  assert(training_channels.names==matching_channels.names&&training_channels.names==shifted_channels.names);
  rpmd_ja_statistics_policy::BlockPlan frozen_plan;frozen_plan.status="PASS";frozen_plan.block_length=1;
  frozen_plan.adjacent_block_length=2;frozen_plan.tau_frames=1.0;
  const auto config=rpmd_ja_statistics_policy::frozen_config();
  const std::vector<std::string> no_expected_zero_mean_channels;
  const auto matching=rpmd_ja_statistics_policy::assess_fixed_r0_means(training_channels,matching_channels,
    frozen_plan,config,&no_expected_zero_mean_channels);
  assert(matching.status=="PASS"&&frozen_plan.block_length==1);
  const auto shifted=rpmd_ja_statistics_policy::assess_fixed_r0_means(training_channels,shifted_channels,
    frozen_plan,config,&no_expected_zero_mean_channels);
  assert(shifted.status!="PASS"&&frozen_plan.block_length==1);
}

void initialize_two_atom_covariance_baseline(DeviceBaseline& baseline,const bool identity_internal)
{
  baseline.d=6;assert(cublasCreate(&baseline.blas)==CUBLAS_STATUS_SUCCESS);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),36*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.q),6*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.out),6*sizeof(double))==cudaSuccess);
  std::vector<double> matrix(36,0.0);
  const double t[2]={1.0/std::sqrt(5.0),2.0/std::sqrt(5.0)};
  if(identity_internal)for(int axis=0;axis<3;++axis)for(int i=0;i<2;++i)for(int j=0;j<2;++j){const int row=axis*2+i,column=axis*2+j;
    matrix[static_cast<std::size_t>(row)*6+column]=(i==j?1.0:0.0)-t[i]*t[j];}
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
}

void initialize_identity_covariance_baseline(DeviceBaseline& baseline,const int n)
{
  baseline.d=3*n;assert(cublasCreate(&baseline.blas)==CUBLAS_STATUS_SUCCESS);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),static_cast<std::size_t>(baseline.d)*baseline.d*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.q),static_cast<std::size_t>(baseline.d)*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.out),static_cast<std::size_t>(baseline.d)*sizeof(double))==cudaSuccess);
  std::vector<double> matrix(static_cast<std::size_t>(baseline.d)*baseline.d,0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)for(int j=0;j<n;++j){
    const int row=axis*n+i,column=axis*n+j;
    matrix[static_cast<std::size_t>(row)*baseline.d+column]=(i==j?1.0:0.0)-1.0/n;}
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
}

void test_nonconverged_a0_baseline_restore_rebuilds_factors()
{
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::uint64_t evaluations=0;
  const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,1e-6,evaluations);
  CovarianceFitResult fit;fit.converged=false;fit.status="MAX_ITERATIONS";fit.covariance_baseline=reference;
  fit.objective_data=1.25;fit.objective_prior=0.5;fit.objective_total=1.75;
  rpmd_ja_device::CovarianceFitWorkspace workspace;
  rebuild_restored_a0_baseline(fit,baseline,graph,sqrt_mass,masses,&workspace,1e-6,nullptr,nullptr);
  assert(!fit.converged&&fit.covariance_baseline.logdet==reference.logdet&&
    fit.covariance_baseline.shifted_logdet==reference.shifted_logdet);
  assert(fit.objective_data==1.25&&fit.objective_prior==0.5&&fit.objective_total==1.75);
}

void test_covariance_development_statistics_keep_fixed_training_center()
{
  SampleHeader header{};header.n=2;header.temperature=300.0;header.masses={1.0,4.0};
  header.types={0,1};header.frames=0;const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> r0(6,0.0),sqrt_atom={1.0,2.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.batch_q),static_cast<std::size_t>(6)*DeviceBaseline::batch_capacity*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.batch_out),static_cast<std::size_t>(6)*DeviceBaseline::batch_capacity*sizeof(double))==cudaSuccess);
  std::stringstream spool(std::ios::in|std::ios::out|std::ios::binary);
  for(int frame=0;frame<10;++frame){const double step=static_cast<double>(frame);
    const double position[6]={0.0,static_cast<double>(frame+1),0.0,0.0,0.0,0.0};const double force[6]={};
    spool.write(reinterpret_cast<const char*>(&step),sizeof(step));
    spool.write(reinterpret_cast<const char*>(position),sizeof(position));spool.write(reinterpret_cast<const char*>(force),sizeof(force));}
  std::vector<std::vector<double>> blocks;std::vector<std::uint64_t> block_frames;
  std::vector<double> development;double development_baseline=0.0;
  std::vector<std::vector<double>> development_blocks;std::vector<double> development_baselines;
  std::vector<std::uint64_t> development_frames;
  const std::vector<double> training=collect_covariance_fit_statistics(spool,header,10,6,graph,r0,300.0,
    nullptr,&blocks,&block_frames,nullptr,nullptr,&baseline,&sqrt_mass,&sqrt_atom,&development,&development_baseline,
    &development_blocks,&development_baselines,&development_frames);
  check_close(training[0],91.0/(K_B*300.0*6.0),1e-12);
  check_close(development[0],294.0/(K_B*300.0*4.0),1e-12);
  check_close(development_baseline,235.2/(K_B*300.0*4.0),2e-12);
  assert(block_frames.size()==6&&blocks.size()==6);
  assert(development_frames==std::vector<std::uint64_t>({1,1,1,1})&&development_blocks.size()==4);
  const double development_squares[]={49.0,64.0,81.0,100.0};
  double development_block_mean=0.0,development_baseline_block_mean=0.0;
  for(std::size_t block=0;block<development_blocks.size();++block){
    check_close(development_blocks[block][0],development_squares[block]/(K_B*300.0),1e-12);
    check_close(development_baselines[block],0.8*development_squares[block]/(K_B*300.0),2e-12);
    development_block_mean+=development_blocks[block][0]/development_blocks.size();
    development_baseline_block_mean+=development_baselines[block]/development_baselines.size();}
  check_close(development_block_mean,development[0],1e-12);
  check_close(development_baseline_block_mean,development_baseline,1e-12);
  baseline.release();
}

void test_covariance_probe_session_reuses_factors_and_rejects_stale_assembly()
{
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0},sqrt_atom={1.0,2.0};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  constexpr double small_eigenvalue=1e-5,epsilon=1e-7;std::vector<double> theta(6,0.0);theta[0]=(-1.0+small_eigenvalue)/1.25;
  CovarianceProbeSolveSession session(system,sqrt_atom);const std::uint64_t factors_before=system.factorizations;
  session.factor(theta,epsilon,true,true);assert(system.factorizations-factors_before==2);
  std::vector<std::vector<double>> physical_rhs(6,std::vector<double>(6,0.0));
  for(int axis=0;axis<3;++axis)for(int atom=0;atom<2;++atom)physical_rhs[2*axis+atom][axis*2+atom]=1.0;
  for(const bool shifted:{false,true}){const CovarianceProbeBatch physical=session.solve(physical_rhs,shifted);assert(physical.status=="PASS");
    for(std::size_t column=0;column<physical_rhs.size();++column){std::vector<double> expected=physical_rhs[column],action(6),base(6),add(6),residual(6);
      project_translation(expected,sqrt_atom,2);baseline.apply(physical.solution[column],base);
      apply_additive(graph,physical.solution[column],sqrt_mass,sqrt_atom,theta,2,add);
      for(int i=0;i<6;++i){action[i]=base[i]+add[i]-(shifted?epsilon*physical.solution[column][i]:0.0);residual[i]=action[i]-expected[i];}
      project_translation(residual,sqrt_atom,2);const double relative=std::sqrt(compensated_dot(residual,residual))/
        std::max(std::sqrt(compensated_dot(expected,expected)),1e-300);assert(std::isfinite(relative)&&relative<=2e-7);}}
  std::vector<std::vector<double>> rhs(33,std::vector<double>(6,0.0));for(std::size_t i=0;i<rhs.size();++i)rhs[i][(i%2)*2]=1.0;
  const CovarianceProbeBatch solved=session.solve(rhs,false);assert(solved.status=="PASS"&&solved.columns.size()==33);
  assert(system.factorizations-factors_before==2);
  check_close(solved.columns[0].quadratic_post,0.8/small_eigenvalue,2e-6);
  check_close(solved.columns[1].quadratic_post,0.8,2e-8);
  for(const CovarianceProbeColumn& column:solved.columns){assert(column.factor_residual_post<=1e-7&&column.original_residual_post<=1e-7);
    assert(column.refinement_steps>=0&&column.refinement_steps<=2);assert(std::isfinite(column.quadratic_pre)&&std::isfinite(column.quadratic_post));
    if(column.refinement_steps>0)assert(column.original_residual_post<=column.original_residual_pre*(1.0+1e-10));}
  for(const std::vector<double>& solution:solved.solution)for(int axis=0;axis<3;++axis)
    assert(std::abs(sqrt_atom[0]*solution[axis*2]+sqrt_atom[1]*solution[axis*2+1])<=1e-8);
  const CovarianceProbeModes modes=covariance_bounded_inverse_modes(system.solver,session,baseline,graph,theta,
    sqrt_mass,sqrt_atom,2,epsilon,4,12);
  assert(modes.status=="RITZ_RESIDUAL_RESOLVED"&&modes.modes.size()==3&&modes.iterations>=1&&modes.iterations<=12);
  assert(system.factorizations-factors_before==2);
  check_close(modes.modes[0].actual_rayleigh,small_eigenvalue,1e-8);
  check_close(modes.modes[1].actual_rayleigh,1.0,1e-8);check_close(modes.modes[2].actual_rayleigh,1.0,1e-8);
  for(const RitzMode& mode:modes.modes){assert(mode.steps_used==modes.iterations&&mode.steps_used<=12);
    for(int axis=0;axis<3;++axis)assert(std::abs(sqrt_atom[0]*mode.vector[axis*2]+sqrt_atom[1]*mode.vector[axis*2+1])<=1e-10);}
  system.assemble(theta);
  bool stale_rejected=false;try{session.solve(rhs,false);}catch(const std::logic_error&){stale_rejected=true;}
  assert(stale_rejected);
}

void test_covariance_probe_refinement_improves_each_original_residual_and_rejects_large_error()
{
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0},sqrt_atom={1.0,2.0};
  const std::vector<double> theta(6,0.0);std::vector<std::vector<double>> rhs(1,std::vector<double>(6,0.0));rhs[0][0]=1.0;
  for(const double perturbation:{1e-8,1e-4}){DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);CovarianceProbeSolveSession session(system,sqrt_atom);session.factor(theta,1e-10,true,false);
    std::vector<double> factor(static_cast<std::size_t>(system.r)*system.r);
    assert(cudaMemcpy(factor.data(),system.compact,factor.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
    factor[0]+=perturbation;assert(cudaMemcpy(system.compact,factor.data(),factor.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
    const CovarianceProbeBatch solved=session.solve(rhs,false);assert(solved.columns.size()==1);
    if(perturbation<1e-6){assert(solved.status=="PASS");assert(solved.columns[0].refinement_steps>0&&solved.columns[0].refinement_steps<=2);
      assert(solved.columns[0].original_residual_pre>1e-12);
      assert(solved.columns[0].original_residual_post<solved.columns[0].original_residual_pre);
    }else{assert(solved.status=="RESIDUAL_FAIL");assert(solved.columns[0].factor_residual_post>1e-7);}}
}

void test_covariance_operator_gram_includes_shared_atom_overlap(cusolverDnHandle_t solver)
{
  Graph graph;graph.type_pairs={{0,1},{0,2}};graph.edges={{0,1,0,{0,0,0}},{0,2,1,{0,0,0}}};
  const CovarianceOperatorGram gram=covariance_operator_gram(solver,graph,{1.0,1.0,1.0});
  assert(gram.status=="PASS"&&gram.groups==2&&gram.rank==2&&gram.eigenvalues.size()==2);
  check_close(gram.normalized[0],1.0,1e-14);check_close(gram.normalized[3],1.0,1e-14);
  check_close(gram.normalized[1],0.25,1e-14);check_close(gram.normalized[2],0.25,1e-14);
  check_close(gram.eigenvalues[0],0.75,1e-12);check_close(gram.eigenvalues[1],1.25,1e-12);
}

void test_covariance_local_group_hessian_matches_tiny_objective_fd(cusolverDnHandle_t solver)
{
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> masses={1.0,4.0};
  const std::vector<double> sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0},sqrt_atom={1.0,2.0};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);const std::vector<double> theta={0.2,0.01,-0.02,0.25,0.03,0.3};
  constexpr double epsilon=0.01,rho=0.0,tau=0.2,ridge_lambda=0.35;const CovarianceLocalGroupBasis local=covariance_local_group_basis(graph,sqrt_mass,2,0,32);
  assert(local.status=="READY"&&local.dimension==6);
  CovarianceProbeSolveSession session(system,sqrt_atom);session.factor(theta,epsilon,true,true);
  std::vector<std::vector<double>> rhs(6,std::vector<double>(6,0.0));
  for(int column=0;column<6;++column)rhs[column][static_cast<std::size_t>(column)]=1.0;
  const CovarianceProbeBatch solved_d=session.solve(rhs,false),solved_s=session.solve(rhs,true);
  double asym_d=0.0,asym_s=0.0;const auto inverse_d=covariance_local_inverse(solved_d,local,2,asym_d);
  const auto inverse_s=covariance_local_inverse(solved_s,local,2,asym_s);assert(asym_d<=1e-8&&asym_s<=1e-8);
  const auto trace_d=covariance_local_trace_products(inverse_d,local),trace_s=covariance_local_trace_products(inverse_s,local);
  const auto internal=internal_basis(sqrt_atom);const int d=6,r=static_cast<int>(internal.size());std::vector<double> restricted(static_cast<std::size_t>(r)*r,0.0);
  for(int column=0;column<r;++column){std::vector<double> raw(d),add(d),action(d);baseline.apply(internal[column],raw);apply_additive(graph,internal[column],sqrt_mass,sqrt_atom,theta,2,add);
    for(int i=0;i<d;++i)action[i]=raw[i]+add[i];for(int row=0;row<r;++row)restricted[static_cast<std::size_t>(row)*r+column]=compensated_dot(internal[row],action);}
  auto full_inverse=[&](const double shift){std::vector<double> matrix=restricted;for(int i=0;i<r;++i)matrix[static_cast<std::size_t>(i)*r+i]-=shift;
    std::vector<double> inverse(static_cast<std::size_t>(d)*d,0.0);
    for(int column=0;column<r;++column){std::vector<double> unit(static_cast<std::size_t>(r),0.0);unit[column]=1.0;
      const std::vector<double> solved=solve_dense_reference(matrix,unit,r);
      for(int coefficient=0;coefficient<r;++coefficient)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
        inverse[static_cast<std::size_t>(i)*d+j]+=internal[coefficient][i]*solved[coefficient]*internal[column][j];}
    return inverse;};
  auto full_trace=[&](const std::vector<double>& inverse,const int a,const int b){std::vector<double> xa(static_cast<std::size_t>(d)*d,0.0),xb(xa.size(),0.0);
    for(int i=0;i<d;++i)for(int j=0;j<d;++j)for(int k=0;k<d;++k){xa[static_cast<std::size_t>(i)*d+j]+=inverse[static_cast<std::size_t>(i)*d+k]*local.basis[a][static_cast<std::size_t>(k)*d+j];
        xb[static_cast<std::size_t>(i)*d+j]+=inverse[static_cast<std::size_t>(i)*d+k]*local.basis[b][static_cast<std::size_t>(k)*d+j];}
    NeumaierAccumulator sum;for(int i=0;i<d;++i)for(int j=0;j<d;++j)sum.add_product(xa[static_cast<std::size_t>(i)*d+j],xb[static_cast<std::size_t>(j)*d+i]);return sum.value();};
  for(const auto& op:{std::make_pair(0.0,&trace_d),std::make_pair(epsilon,&trace_s)}){const std::vector<double> inverse=full_inverse(op.first);
    for(int a=0;a<6;++a)for(int b=0;b<6;++b)check_close((*op.second)[static_cast<std::size_t>(a)*6+b],full_trace(inverse,a,b),2e-7);}
  const std::vector<double> scale={0.5,1.25,1.5,0.75,2.0,0.6};
  const CovarianceLocalHessianTerms hessian=covariance_local_group_hessian(trace_d,trace_s,rho,tau,ridge_lambda,scale,0);
  for(int a=0;a<6;++a)for(int b=0;b<6;++b)
    check_close(hessian.ridge[static_cast<std::size_t>(a)*6+b],a==b?ridge_lambda:0.0,1e-15);
  const double analytic=hessian.total[0];
  std::vector<double> extended_scale(12);for(std::size_t i=0;i<extended_scale.size();++i)extended_scale[i]=0.25+0.125*i;
  const CovarianceLocalHessianTerms scaled_hessian=covariance_local_group_hessian(trace_d,trace_s,rho,tau,ridge_lambda,extended_scale,6);
  for(int a=0;a<6;++a)for(int b=0;b<6;++b){const std::size_t at=static_cast<std::size_t>(a)*6+b;
    const double expected=(0.5*(1.0+rho)*trace_d[at]+tau*trace_s[at])*extended_scale[6+a]*extended_scale[6+b]+
      (a==b?ridge_lambda:0.0);
    check_close(scaled_hessian.total[at],expected,2e-15);}
  CovarianceBaseline reference;reference.theta=theta;std::vector<double> shifted_trace;
  assert(system.evaluate_factors(theta,epsilon,reference.logdet,reference.shifted_logdet,&reference.trace,&shifted_trace,false,true));
  const std::vector<double> statistics={0.2,-0.03,0.04,0.25,0.01,0.3};std::uint64_t evaluations=0;
  const CovarianceObjectiveEvaluation center=evaluate_covariance_fit_objective(system,theta,reference,statistics,rho,epsilon,tau,evaluations,false,
    false,true,nullptr,nullptr,&scale,ridge_lambda);
  constexpr double h=1e-4;double values[2];
  for(int side=0;side<2;++side){std::vector<double> trial=theta;trial[0]+=(side==0?1.0:-1.0)*h*scale[0];
    const CovarianceObjectiveEvaluation value=evaluate_covariance_fit_objective(system,trial,reference,statistics,rho,epsilon,tau,evaluations,false,
      false,true,nullptr,nullptr,&scale,ridge_lambda);
    assert(value.feasible);values[side]=value.objective;}
  const double finite_difference=(values[0]-2.0*center.objective+values[1])/(h*h);
  check_close(analytic,finite_difference,3e-5);
  assert(saved_d_mode_aggregate_status(true,true)=="INCONCLUSIVE_RITZ_RESIDUAL");
  assert(saved_d_mode_aggregate_status(false,true)=="FAILED");
}

void test_covariance_force_residual_uses_training_frames_only()
{
  SampleHeader header{};header.n=2;header.frames=0;const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0},sqrt_atom={1.0,2.0};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);std::stringstream spool(std::ios::in|std::ios::out|std::ios::binary);
  for(int frame=0;frame<2;++frame){const double step=static_cast<double>(frame),position[6]={0.0,0.0,0.0,0.0,0.0,0.0},force[6]={};
    spool.write(reinterpret_cast<const char*>(&step),sizeof(step));spool.write(reinterpret_cast<const char*>(position),sizeof(position));spool.write(reinterpret_cast<const char*>(force),sizeof(force));}
  spool.seekg(0);const std::vector<double> theta(6,0.0);
  const auto residuals=compare_covariance_training_force_residuals(spool,header,3,2,graph,std::vector<double>(6,0.0),baseline,sqrt_mass,sqrt_atom,theta,theta);
  assert(std::isfinite(residuals.first)&&std::isfinite(residuals.second));
}

void test_covariance_block_newton_host_arithmetic()
{
  std::array<double,36> diagonal_hessian{};std::array<double,6> gradient{};
  const double diagonal[6]={1e-8,1e-4,1.0,1e2,1e4,1e8};
  for(int i=0;i<6;++i){diagonal_hessian[static_cast<std::size_t>(i)*6+i]=diagonal[i];gradient[i]=2.0*diagonal[i];}
  const CovarianceBlockNewtonResult equilibrated=covariance_local_newton_direction(diagonal_hessian,gradient);
  assert(equilibrated.status=="PASS"&&equilibrated.damping==0.0&&equilibrated.relative_residual<=1e-8);
  for(const double value:equilibrated.direction)check_close(value,-2.0,1e-12);
  const CovarianceBlockNewtonResult generalized_six=covariance_joint_newton_direction(
    std::vector<double>(diagonal_hessian.begin(),diagonal_hessian.end()),std::vector<double>(gradient.begin(),gradient.end()));
  assert(generalized_six.status==equilibrated.status&&generalized_six.direction==equilibrated.direction&&generalized_six.damping==equilibrated.damping);
  bool nonfinite_threw=false;gradient[0]=std::numeric_limits<double>::infinity();
  try{(void)covariance_local_newton_direction(diagonal_hessian,gradient);}catch(const std::runtime_error&){nonfinite_threw=true;}
  assert(nonfinite_threw);gradient[0]=2.0*diagonal[0];
  std::array<double,36> asymmetric_hessian{};for(int i=0;i<6;++i)asymmetric_hessian[static_cast<std::size_t>(i)*6+i]=1.0;
  asymmetric_hessian[1]=1e-4;assert(covariance_local_newton_direction(asymmetric_hessian,gradient).status=="HESSIAN_NOT_SYMMETRIC");
  std::array<double,36> nonpositive_hessian{};nonpositive_hessian[0]=0.0;
  assert(covariance_local_newton_direction(nonpositive_hessian,gradient).status=="NONPOSITIVE_DIAGONAL");

  std::array<double,36> damped_hessian{};for(int i=0;i<6;++i)damped_hessian[static_cast<std::size_t>(i)*6+i]=1.0;
  damped_hessian[1]=damped_hessian[6]=1.0001;gradient={1.0,0.5,0.0,0.0,0.0,0.0};
  const CovarianceBlockNewtonResult damped=covariance_local_newton_direction(damped_hessian,gradient);
  assert(damped.status=="PASS"&&damped.damping>0.0&&damped.damping<=1e-2&&damped.relative_residual<=1e-8&&damped.slope<0.0);

  std::vector<double> full_gradient(12,0.0),bfgs_direction(12,0.0);full_gradient[0]=1.0;full_gradient[6]=1.0;
  bfgs_direction[0]=-5.0;bfgs_direction[6]=4.0;CovarianceBlockNewtonResult block;block.status="PASS";block.direction[0]=-2.0;
  const CovarianceDirectionChoice pure_block=covariance_choose_block_direction(full_gradient,bfgs_direction,0,block);
  assert(pure_block.method=="BLOCK"&&pure_block.reason=="HYBRID_NON_DESCENT_BLOCK_DESCENT"&&pure_block.selected_slope==-2.0);
  std::vector<double> hybrid_gradient(6,0.0),hybrid_bfgs(6,0.0);hybrid_gradient[0]=1.0;hybrid_bfgs[0]=-1.0;
  const CovarianceDirectionChoice hybrid=covariance_choose_block_direction(hybrid_gradient,hybrid_bfgs,0,block);
  assert(hybrid.method=="HYBRID"&&hybrid.reason=="HYBRID_DESCENT"&&hybrid.selected_slope<0.0);
  block.status="LOCAL_SOLVE_RESIDUAL_FAIL";const CovarianceDirectionChoice fallback=covariance_choose_block_direction(full_gradient,bfgs_direction,0,block);
  assert(fallback.method=="BFGS"&&fallback.direction==bfgs_direction&&fallback.reason=="LOCAL_SOLVE_RESIDUAL_FAIL");
  const std::vector<double> theta={0.7316913311810651},scale={0.000145},z={theta[0]/scale[0]},preflight_gradient={1.0},preflight_bfgs={-1.0};
  CovarianceDirectionChoice unrepresentable;unrepresentable.method="BLOCK";unrepresentable.direction={-1e-20};unrepresentable.selected_slope=-1e-20;
  const CovarianceDirectionChoice preflight=covariance_preflight_block_direction(theta,z,scale,preflight_gradient,preflight_bfgs,-1.0,unrepresentable);
  assert(preflight.method=="BFGS"&&preflight.direction==preflight_bfgs&&preflight.selected_slope==-1.0&&
    preflight.reason=="BLOCK_STEP_UNREPRESENTABLE_BFGS_FALLBACK");

  Graph graph;graph.type_pairs={{0,1}};for(int atom=1;atom<=32;++atom){Edge edge;edge.i=0;edge.j=atom;edge.group=0;graph.edges.push_back(edge);}
  const CovarianceLocalGroupBasis capped=covariance_local_group_basis(graph,std::vector<double>(3*33,1.0),33,0,32);
  assert(capped.status=="NOT_EVALUATED_SUPPORT_CAP"&&capped.support_count==33);

  const Graph tiny=make_two_atom_covariance_graph();const CovarianceLocalGroupBasis local=covariance_local_group_basis(tiny,{1.0,2.0,1.0,2.0,1.0,2.0},2,0,32);
  CovarianceProbeBatch asymmetric;asymmetric.status="PASS";asymmetric.solution.assign(6,std::vector<double>(6,0.0));
  for(int i=0;i<6;++i)asymmetric.solution[i][i]=1.0;asymmetric.solution[1][0]=2.0;
  double relative_asymmetry=0.0;bool controlled=false;
  try{(void)covariance_local_inverse(asymmetric,local,2,relative_asymmetry);}
  catch(const CovarianceLocalInverseAsymmetryError&){controlled=true;}
  assert(controlled&&std::isfinite(relative_asymmetry)&&relative_asymmetry>1e-8);
}

void test_covariance_joint_block_includes_cross_group_hessian()
{
  constexpr int n=8,d=9;
  Graph graph;graph.type_pairs={{0,0},{0,1},{1,1}};
  graph.edges={{1,4,0,{0,0,0}},{0,7,1,{0,0,0}},{4,7,2,{0,0,0}}};
  std::vector<double> sqrt_mass(3*n,1.0);const double atom_sqrt[8]={1.0,2.0,1.5,2.0,3.0,1.0,2.5,4.0};
  for(int axis=0;axis<3;++axis)for(int atom=0;atom<n;++atom)sqrt_mass[axis*n+atom]=atom_sqrt[atom];
  std::vector<double> gradient(18,0.0);gradient[0]=0.6;gradient[6]=0.1;gradient[12]=1.0;
  const CovarianceJointGroupTopology topology=covariance_joint_group_topology(graph);
  const CovarianceJointGroupSelection selection=covariance_select_joint_groups(topology,gradient,32,2);
  assert(selection.status=="READY"&&selection.groups==std::vector<int>({2,0})&&selection.group_supports==std::vector<std::size_t>({2,2}));
  assert(selection.dominant_group==2&&selection.dominant_group_support==2&&topology.atoms[2]==std::vector<int>({4,7}));
  assert(selection.support_count==3&&selection.atoms==std::vector<int>({1,4,7}));
  const CovarianceJointGroupBasis basis=covariance_joint_group_basis(graph,topology,sqrt_mass,n,selection.groups,selection.atoms,32);
  assert(basis.status=="READY"&&basis.atoms==std::vector<int>({1,4,7})&&basis.dimension==9&&basis.basis.size()==12);
  check_close(basis.basis[0][static_cast<std::size_t>(1)*d+1],1.0/9.0,1e-15);
  check_close(basis.basis[0][static_cast<std::size_t>(1)*d+2],-1.0/12.0,1e-15);
  check_close(basis.basis[6][0],1.0/4.0,1e-15);
  check_close(basis.basis[6][1],-1.0/6.0,1e-15);
  check_close(basis.basis[7][4],-1.0/6.0,1e-15);
  check_close(basis.basis[7][static_cast<std::size_t>(4)*d+0],-1.0/6.0,1e-15);

  auto inverse=[&](const double diagonal){std::vector<double> result(static_cast<std::size_t>(d)*d,0.0);
    for(int i=0;i<d;++i)for(int j=0;j<d;++j)result[static_cast<std::size_t>(i)*d+j]=i==j?diagonal+0.03*i:0.002/(1+std::abs(i-j));return result;};
  const std::vector<double> inverse_d=inverse(1.0),inverse_s=inverse(1.4);
  auto direct_trace=[&](const std::vector<double>& inv,const int a,const int b){std::vector<double> left(static_cast<std::size_t>(d)*d,0.0),right(left.size(),0.0);
    for(int i=0;i<d;++i)for(int k=0;k<d;++k)for(int j=0;j<d;++j){
      left[static_cast<std::size_t>(i)*d+j]+=inv[static_cast<std::size_t>(i)*d+k]*basis.basis[a][static_cast<std::size_t>(k)*d+j];
      right[static_cast<std::size_t>(i)*d+j]+=inv[static_cast<std::size_t>(i)*d+k]*basis.basis[b][static_cast<std::size_t>(k)*d+j];}
    NeumaierAccumulator sum;for(int i=0;i<d;++i)for(int j=0;j<d;++j)sum.add_product(left[static_cast<std::size_t>(i)*d+j],right[static_cast<std::size_t>(j)*d+i]);return sum.value();};
  const std::vector<double> trace_d=covariance_joint_trace_products(inverse_d,basis),trace_s=covariance_joint_trace_products(inverse_s,basis);
  for(int a=0;a<12;++a)for(int b=0;b<12;++b){const std::size_t at=static_cast<std::size_t>(a)*12+b;
    check_close(trace_d[at],direct_trace(inverse_d,a,b),2e-13);check_close(trace_s[at],direct_trace(inverse_s,a,b),2e-13);}
  std::vector<double> scale(18);for(std::size_t i=0;i<scale.size();++i)scale[i]=0.4+0.07*i;
  constexpr double ridge_lambda=0.4;
  const CovarianceJointHessianTerms hessian=covariance_joint_group_hessian(trace_d,trace_s,0.0,0.3,ridge_lambda,scale,selection.groups);
  const std::size_t cross=6;assert(std::abs(hessian.total[cross])>1e-5);
  check_close(hessian.total[cross],(0.5*trace_d[cross]+0.3*trace_s[cross])*scale[12]*scale[0],2e-14);
  check_close(hessian.ridge[0],ridge_lambda,1e-15);
  check_close(hessian.ridge[cross],0.0,1e-15);
  std::vector<double> selected_gradient(12);for(int component=0;component<6;++component){
    selected_gradient[component]=gradient[12+component];selected_gradient[6+component]=gradient[component];}
  const CovarianceBlockNewtonResult joint=covariance_joint_newton_direction(hessian.total,selected_gradient);
  assert(joint.status=="PASS"&&joint.direction.size()==12);
  std::vector<double> rhs(12);for(int i=0;i<12;++i)rhs[i]=-selected_gradient[i];
  const std::vector<double> dense=solve_dense_reference(hessian.total,rhs,12);
  for(int i=0;i<12;++i)check_close(joint.direction[i],dense[i],2e-10);
  const std::vector<double> independent_first=solve_dense_reference(
    {hessian.total[0],hessian.total[1],hessian.total[2],hessian.total[3],hessian.total[4],hessian.total[5],
     hessian.total[12],hessian.total[13],hessian.total[14],hessian.total[15],hessian.total[16],hessian.total[17],
     hessian.total[24],hessian.total[25],hessian.total[26],hessian.total[27],hessian.total[28],hessian.total[29],
     hessian.total[36],hessian.total[37],hessian.total[38],hessian.total[39],hessian.total[40],hessian.total[41],
     hessian.total[48],hessian.total[49],hessian.total[50],hessian.total[51],hessian.total[52],hessian.total[53],
     hessian.total[60],hessian.total[61],hessian.total[62],hessian.total[63],hessian.total[64],hessian.total[65]},
    std::vector<double>(rhs.begin(),rhs.begin()+6),6);
  assert(std::abs(independent_first[0]-joint.direction[0])>1e-8);
  const CovarianceDirectionChoice global=covariance_choose_joint_block_direction(gradient,std::vector<double>(18,-1.0),selection.groups,joint);
  assert(global.method=="JOINT_BLOCK"&&global.direction.size()==18);
  for(int component=0;component<6;++component){check_close(global.direction[12+component],joint.direction[component],1e-15);
    check_close(global.direction[component],joint.direction[6+component],1e-15);assert(global.direction[6+component]==0.0);}

  Graph capped; capped.type_pairs={{0,0},{0,1},{1,1}};
  for(int atom=1;atom<=19;++atom)capped.edges.push_back({0,atom,0,{0,0,0}});
  for(int atom=21;atom<=39;++atom)capped.edges.push_back({20,atom,1,{0,0,0}});
  capped.edges.push_back({0,40,2,{0,0,0}});
  std::vector<double> ranked_gradient(18,0.0);ranked_gradient[0]=9.0;ranked_gradient[6]=8.0;ranked_gradient[12]=7.0;
  const CovarianceJointGroupTopology capped_topology=covariance_joint_group_topology(capped);
  const CovarianceJointGroupSelection selected=covariance_select_joint_groups(capped_topology,ranked_gradient,32,3);
  assert(selected.status=="READY"&&selected.groups==std::vector<int>({0,2})&&selected.support_count==21);
  Graph empty_non_dominant;empty_non_dominant.type_pairs={{0,0},{0,1},{1,1}};
  empty_non_dominant.edges={{0,1,0,{0,0,0}},{2,3,2,{0,0,0}}};
  const CovarianceJointGroupTopology empty_non_dominant_topology=covariance_joint_group_topology(empty_non_dominant);
  const CovarianceJointGroupSelection skip_empty=covariance_select_joint_groups(empty_non_dominant_topology,{9,0,0,0,0,0,8,0,0,0,0,0,7,0,0,0,0,0},32,3);
  assert(skip_empty.status=="READY"&&skip_empty.groups==std::vector<int>({0,2}));
  const CovarianceJointGroupSelection empty_dominant=covariance_select_joint_groups(empty_non_dominant_topology,{9,0,0,0,0,0,10,0,0,0,0,0,7,0,0,0,0,0},32,3);
  assert(empty_dominant.status=="DOMINANT_EMPTY_GROUP"&&empty_dominant.groups.empty()&&empty_dominant.dominant_group==1);
  Graph dominant_too_large;dominant_too_large.type_pairs={{0,0}};
  for(int atom=1;atom<=32;++atom)dominant_too_large.edges.push_back({0,atom,0,{0,0,0}});
  const CovarianceJointGroupTopology large_topology=covariance_joint_group_topology(dominant_too_large);
  const CovarianceJointGroupSelection rejected=covariance_select_joint_groups(large_topology,{1.0,0,0,0,0,0},32,3);
  assert(rejected.status=="DOMINANT_SUPPORT_CAP"&&rejected.groups.empty()&&rejected.dominant_group==0&&rejected.dominant_group_support==33);
  CovarianceBlockNewtonResult capped_direction;capped_direction.status=rejected.status;
  const CovarianceDirectionChoice capped_fallback=covariance_choose_joint_block_direction({1.0,0,0,0,0,0},{-1.0,0,0,0,0,0},rejected.groups,capped_direction);
  assert(capped_fallback.method=="BFGS"&&capped_fallback.reason=="DOMINANT_SUPPORT_CAP");
  Graph invalid_atom;invalid_atom.type_pairs={{0,0}};invalid_atom.edges={{0,8,0,{0,0,0}}};
  const CovarianceJointGroupTopology invalid_atom_topology=covariance_joint_group_topology(invalid_atom);
  const CovarianceJointGroupSelection invalid_atom_selection=covariance_select_joint_groups(invalid_atom_topology,{1.0,0,0,0,0,0},32,3);
  bool invalid_atom_rejected=false;try{(void)covariance_joint_group_basis(invalid_atom,invalid_atom_topology,std::vector<double>(24,1.0),8,
      invalid_atom_selection.groups,invalid_atom_selection.atoms,32);}catch(const std::invalid_argument&){invalid_atom_rejected=true;}
  assert(invalid_atom_rejected);
  bool dimensions_rejected=false;try{(void)covariance_select_joint_groups(topology,std::vector<double>(17,0.0),32,3);}
    catch(const std::invalid_argument&){dimensions_rejected=true;}assert(dimensions_rejected);
  bool nonfinite_rejected=false;try{std::vector<double> invalid_gradient(18,0.0);invalid_gradient[0]=std::numeric_limits<double>::quiet_NaN();
      (void)covariance_select_joint_groups(topology,invalid_gradient,32,3);}catch(const std::runtime_error&){nonfinite_rejected=true;}
  assert(nonfinite_rejected);
}

void test_covariance_group_bfgs_independent_history()
{
  constexpr int groups=2,parameters=6*groups;
  std::vector<double> hessian(static_cast<std::size_t>(parameters)*parameters,0.0);
  for(int group=0;group<groups;++group)for(int component=0;component<6;++component){
    const int index=6*group+component;hessian[static_cast<std::size_t>(index)*parameters+index]=(group?1e6:1.0)*(component+1);}
  hessian[6]=0.2*std::sqrt(hessian[0]*hessian[static_cast<std::size_t>(6)*parameters+6]);
  hessian[static_cast<std::size_t>(6)*parameters]=hessian[6];
  std::vector<double> point(parameters);for(int i=0;i<parameters;++i)point[i]=0.1*(i+1);
  auto multiply=[&](const std::vector<double>& vector){std::vector<double> result(parameters,0.0);
    for(int i=0;i<parameters;++i)for(int j=0;j<parameters;++j)result[i]+=hessian[static_cast<std::size_t>(i)*parameters+j]*vector[j];return result;};
  const std::vector<double> gradient=multiply(point);std::array<CovarianceBFGSHistory,groups> history;
  for(CovarianceBFGSHistory& item:history)covariance_reset_bfgs_history(item,6,true);
  const CovarianceDirectionChoice first=covariance_choose_group_bfgs_direction(gradient,0,history[0]);
  assert(first.method=="GROUP_BFGS"&&first.selected_slope<0.0&&first.direction.size()==parameters);
  for(int i=6;i<parameters;++i)assert(first.direction[i]==0.0);
  for(int i=0;i<6;++i)check_close(first.direction[i],-gradient[i],1e-14);
  std::vector<double> step(parameters);for(int i=0;i<parameters;++i)step[i]=0.2*first.direction[i];
  const std::vector<double> change=multiply(step);
  const auto first_update=covariance_update_group_bfgs_history(history[0],0,step,change);
  assert(first_update==CovarianceGroupBFGSUpdateStatus::Seeded&&history[0].stage_seeded&&history[0].seed_source=="FIRST_VALID_SECANT");
  assert(history[1].seed_gamma==1.0&&!history[1].stage_seeded);
  const double group0_gamma=covariance_dot(std::vector<double>(step.begin(),step.begin()+6),std::vector<double>(change.begin(),change.begin()+6)) /
    covariance_dot(std::vector<double>(change.begin(),change.begin()+6),std::vector<double>(change.begin(),change.begin()+6));
  check_close(history[0].seed_gamma,group0_gamma,1e-13);
  const CovarianceDirectionChoice second=covariance_choose_group_bfgs_direction(gradient,1,history[1]);
  assert(second.method=="GROUP_BFGS"&&second.selected_slope<0.0);
  for(int i=0;i<6;++i)assert(second.direction[i]==0.0);
  std::vector<double> second_step(parameters);for(int i=0;i<parameters;++i)second_step[i]=0.2*second.direction[i];
  const std::vector<double> second_change=multiply(second_step);
  assert(covariance_update_group_bfgs_history(history[1],1,second_step,second_change)==CovarianceGroupBFGSUpdateStatus::Seeded);
  assert(history[1].seed_gamma!=history[0].seed_gamma);

  const CovarianceBFGSHistory committed=history[0];CovarianceBFGSHistory rejected_pending=committed;
  std::vector<double> mixed_step=step;mixed_step[6]=second_step[6];const std::vector<double> mixed_change=multiply(mixed_step);
  assert(covariance_update_group_bfgs_history(rejected_pending,0,mixed_step,mixed_change)==CovarianceGroupBFGSUpdateStatus::SkippedMixedStep);
  assert(rejected_pending.inverse_hessian==committed.inverse_hessian&&rejected_pending.seed_gamma==committed.seed_gamma&&
    rejected_pending.seed_source==committed.seed_source&&rejected_pending.stage_seeded==committed.stage_seeded);
  CovarianceBFGSHistory overflow_history;covariance_reset_bfgs_history(overflow_history,6,true);bool overflow_rejected=false;
  std::vector<double> huge_step(12,0.0),huge_change(12,0.0);huge_step[0]=huge_change[0]=1e200;
  try{(void)covariance_update_group_bfgs_history(overflow_history,0,huge_step,huge_change);}catch(const std::runtime_error&){overflow_rejected=true;}
  assert(overflow_rejected&&!overflow_history.stage_seeded&&overflow_history.seed_gamma==1.0);
  CovarianceBFGSHistory budget_pending=committed;
  std::vector<double> pure_step(12,0.0),pure_change(12,0.0);for(int i=0;i<6;++i){pure_step[i]=0.1*step[i];pure_change[i]=0.1*change[i];}
  assert(covariance_update_group_bfgs_history(budget_pending,0,pure_step,pure_change)==CovarianceGroupBFGSUpdateStatus::Updated);
  assert(committed.inverse_hessian==history[0].inverse_hessian&&committed.seed_gamma==history[0].seed_gamma);
  covariance_reset_bfgs_history(history[0],6,true);
  assert(!history[0].stage_seeded&&std::isnan(history[0].last_valid_gamma)&&history[0].seed_gamma==1.0&&history[0].seed_source=="UNIT_IDENTITY");

  std::vector<double> coupled_hessian(static_cast<std::size_t>(parameters)*parameters,0.0);
  std::array<double,groups*6> curvature{};
  for(int group=0;group<groups;++group)for(int component=0;component<6;++component)
    curvature[6*group+component]=(group?1e10:1e4)*(1.0+0.1*component);
  for(int i=0;i<parameters;++i){coupled_hessian[static_cast<std::size_t>(i)*parameters+i]=curvature[i];
    for(int j=i+1;j<parameters;++j){const bool same_group=i/6==j/6;
      const double normalized=(same_group&&j/6==i/6&&j-i==1)?0.03:i/6!=j/6&&i%6==j%6?0.2:0.0;
      if(normalized!=0.0){const double value=normalized*std::sqrt(curvature[i]*curvature[j]);
        coupled_hessian[static_cast<std::size_t>(i)*parameters+j]=coupled_hessian[static_cast<std::size_t>(j)*parameters+i]=value;}}}
  for(int i=0;i<parameters;++i){double normalized_off_diagonal_sum=0.0;
    for(int j=0;j<parameters;++j)if(i!=j)
      normalized_off_diagonal_sum+=std::abs(coupled_hessian[static_cast<std::size_t>(i)*parameters+j])/std::sqrt(curvature[i]*curvature[j]);
    assert(normalized_off_diagonal_sum<1.0);}
  auto coupled_gradient=[&](const std::vector<double>& vector){std::vector<double> result(parameters,0.0);
    for(int i=0;i<parameters;++i)for(int j=0;j<parameters;++j)result[i]+=coupled_hessian[static_cast<std::size_t>(i)*parameters+j]*vector[j];return result;};
  auto quadratic=[&](const std::vector<double>& vector){const std::vector<double> product=coupled_gradient(vector);
    return 0.5*covariance_dot(vector,product);};
  std::vector<double> point_coupled(parameters);for(int i=0;i<parameters;++i)point_coupled[i]=1.0/std::sqrt(curvature[i]);
  std::array<CovarianceBFGSHistory,groups> coupled_history;for(auto& item:coupled_history)covariance_reset_bfgs_history(item,6,true);
  const std::vector<double> initial_gradient=coupled_gradient(point_coupled);const double global_scale=covariance_norm_inf(initial_gradient);
  std::vector<double> coupled_gradient_now=initial_gradient;double objective=quadratic(point_coupled);int accepted_steps=0;
  for(;accepted_steps<200&&covariance_norm_inf(coupled_gradient_now)/global_scale>1e-6;++accepted_steps){
    std::array<double,groups> group_norm{};for(int group=0;group<groups;++group)for(int component=0;component<6;++component)
      group_norm[group]=std::max(group_norm[group],std::abs(coupled_gradient_now[6*group+component]));
    const int active=group_norm[1]>group_norm[0]?1:0;
    CovarianceDirectionChoice direction=covariance_choose_group_bfgs_direction(coupled_gradient_now,active,coupled_history[active]);
    if(!(direction.selected_slope<0.0)){std::vector<double> local_gradient(6),local_direction;
      std::copy_n(coupled_gradient_now.begin()+6*active,6,local_gradient.begin());
      covariance_reset_non_descent_direction(coupled_history[active],local_gradient,local_direction);
      direction=covariance_choose_group_bfgs_direction(coupled_gradient_now,active,coupled_history[active]);}
    assert(direction.selected_slope<0.0);double alpha=std::min(1.0,1.0/covariance_norm_inf(direction.direction));
    bool accepted=false;std::vector<double> next_point,next_gradient,actual_step;double next_objective=0.0;
    for(int trial=0;trial<64;++trial){next_point=point_coupled;actual_step.resize(parameters);
      for(int i=0;i<parameters;++i){next_point[i]+=alpha*direction.direction[i];actual_step[i]=next_point[i]-point_coupled[i];}
      next_gradient=coupled_gradient(next_point);next_objective=quadratic(next_point);
      const double candidate_slope=covariance_dot(next_gradient,direction.direction);
      if(next_objective<=objective+1e-4*alpha*direction.selected_slope&&candidate_slope<=0.0){accepted=true;break;}alpha*=0.5;}
    assert(accepted&&next_objective<objective);std::vector<double> full_change(parameters);
    for(int i=0;i<parameters;++i)full_change[i]=next_gradient[i]-coupled_gradient_now[i];
    const auto update=covariance_update_group_bfgs_history(coupled_history[active],active,actual_step,full_change);
    assert(update==CovarianceGroupBFGSUpdateStatus::Updated||update==CovarianceGroupBFGSUpdateStatus::Seeded||
      update==CovarianceGroupBFGSUpdateStatus::Reset);
    point_coupled=std::move(next_point);coupled_gradient_now=std::move(next_gradient);objective=next_objective;
  }
  assert(accepted_steps<200&&covariance_norm_inf(coupled_gradient_now)/global_scale<=1e-6);
}

void test_covariance_joint_ray_spectrum_and_proposal(cusolverDnHandle_t solver)
{
  constexpr int n=8,d=9;Graph graph;graph.type_pairs={{0,0},{0,1},{1,1}};
  graph.edges={{1,4,0,{0,0,0}},{0,2,1,{0,0,0}},{4,7,2,{0,0,0}}};
  const double root_mass[8]={1.0,2.0,1.5,2.0,3.0,1.0,2.5,4.0};std::vector<double> sqrt_mass(3*n);
  for(int axis=0;axis<3;++axis)for(int atom=0;atom<n;++atom)sqrt_mass[axis*n+atom]=root_mass[atom];
  const CovarianceJointGroupTopology topology=covariance_joint_group_topology(graph);
  const std::vector<int> groups={2,0};const std::vector<int> atoms={1,4,7};
  const CovarianceJointGroupBasis basis=covariance_joint_group_basis(graph,topology,sqrt_mass,n,groups,atoms,32);
  assert(basis.status=="READY"&&basis.dimension==d&&groups[0]!=groups[1]);
  const std::vector<double> direction={0.04,0.01,-0.02,0.03,0.015,-0.01,-0.03,0.02,0.01,0.025,-0.015,0.02};
  std::vector<double> scale(18),statistics(18,0.0),baseline_trace(18,0.0),update(static_cast<std::size_t>(d)*d,0.0);
  for(std::size_t parameter=0;parameter<scale.size();++parameter)scale[parameter]=0.7+0.03*parameter;
  for(std::size_t local=0;local<direction.size();++local){const std::size_t parameter=6*static_cast<std::size_t>(groups[local/6])+local%6;
    statistics[parameter]=-100.0*direction[local];for(std::size_t entry=0;entry<update.size();++entry)
      update[entry]+=scale[parameter]*direction[local]*basis.basis[local][entry];}
  std::vector<double> projection(static_cast<std::size_t>(d)*d,0.0);double norm2=0.0;
  for(const int atom:atoms)norm2+=root_mass[atom]*root_mass[atom];
  for(int axis=0;axis<3;++axis)for(int i=0;i<3;++i)for(int j=0;j<3;++j){const int row=axis*3+i,column=axis*3+j;
    projection[static_cast<std::size_t>(row)*d+column]=(i==j?1.0:0.0)-root_mass[atoms[i]]*root_mass[atoms[j]]/norm2;}
  std::vector<double> inverse_s(projection.size());for(std::size_t i=0;i<projection.size();++i)inverse_s[i]=1.3*projection[i];
  const CovarianceJointRaySpectrum spectrum_d=covariance_joint_ray_spectrum(solver,projection,update,d);
  const CovarianceJointRaySpectrum spectrum_s=covariance_joint_ray_spectrum(solver,inverse_s,update,d);
  assert(spectrum_d.status=="PASS"&&spectrum_s.status=="PASS"&&spectrum_d.eigen_evaluations==2&&spectrum_s.eigen_evaluations==2);
  assert(spectrum_d.inverse_min_eigenvalue>=-1e-10&&spectrum_s.inverse_min_eigenvalue>=-1e-10);
  assert(spectrum_d.eigen_residual<1e-8&&spectrum_s.eigen_residual<1e-8);
  auto multiply=[&](const std::vector<double>& left,const std::vector<double>& right){std::vector<double> product(static_cast<std::size_t>(d)*d,0.0);
    for(int i=0;i<d;++i)for(int j=0;j<d;++j)for(int k=0;k<d;++k)
      product[static_cast<std::size_t>(i)*d+j]+=left[static_cast<std::size_t>(i)*d+k]*right[static_cast<std::size_t>(k)*d+j];return product;};
  std::vector<double> matrix_d=multiply(multiply(projection,update),projection),matrix_s(matrix_d.size());
  for(std::size_t i=0;i<matrix_s.size();++i)matrix_s[i]=1.3*matrix_d[i];
  auto dense_logdet_trace=[&](const std::vector<double>& matrix,const double alpha){std::vector<double> shifted=matrix;
    for(int i=0;i<d;++i)for(int j=0;j<d;++j)shifted[static_cast<std::size_t>(i)*d+j]*=alpha;
    for(int i=0;i<d;++i)shifted[static_cast<std::size_t>(i)*d+i]+=1.0;
    std::vector<double> lower(static_cast<std::size_t>(d)*d,0.0);double logdet=0.0;
    for(int i=0;i<d;++i)for(int j=0;j<=i;++j){double value=shifted[static_cast<std::size_t>(i)*d+j];
      for(int k=0;k<j;++k)value-=lower[static_cast<std::size_t>(i)*d+k]*lower[static_cast<std::size_t>(j)*d+k];
      if(i==j){assert(value>0.0);lower[static_cast<std::size_t>(i)*d+j]=std::sqrt(value);logdet+=2.0*std::log(lower[static_cast<std::size_t>(i)*d+j]);}
      else lower[static_cast<std::size_t>(i)*d+j]=value/lower[static_cast<std::size_t>(j)*d+j];}
    std::vector<double> inverse(static_cast<std::size_t>(d)*d,0.0);
    for(int column=0;column<d;++column){std::vector<double> unit(d,0.0);unit[column]=1.0;
      const std::vector<double> solved=solve_dense_reference(shifted,unit,d);
      for(int row=0;row<d;++row)inverse[static_cast<std::size_t>(row)*d+column]=solved[row];}
    NeumaierAccumulator trace;for(int i=0;i<d;++i)for(int j=0;j<d;++j)
      trace.add_product(inverse[static_cast<std::size_t>(i)*d+j],matrix[static_cast<std::size_t>(j)*d+i]);
    return std::pair<double,double>{logdet,trace.value()};};
  constexpr double rho=0.4,tau=0.3;NeumaierAccumulator linear_sum;
  for(std::size_t local=0;local<direction.size();++local){const std::size_t parameter=6*static_cast<std::size_t>(groups[local/6])+local%6;
    linear_sum.add_product(0.5*(statistics[parameter]+rho*baseline_trace[parameter]),scale[parameter]*direction[local]);}
  const double linear=linear_sum.value(),a=0.5*(1.0+rho);
  const double slope_zero=linear-a*dense_logdet_trace(matrix_d,0.0).second-tau*dense_logdet_trace(matrix_s,0.0).second;
  for(const double alpha:{0.001,0.01,0.05}){double ray_delta=0.0,ray_slope=0.0;
    assert(covariance_joint_ray_value(spectrum_d.eigenvalues,spectrum_s.eigenvalues,linear,rho,tau,alpha,ray_delta,ray_slope));
    const auto dense_d=dense_logdet_trace(matrix_d,alpha),dense_s=dense_logdet_trace(matrix_s,alpha);
    const double dense_delta=alpha*linear-a*dense_d.first-tau*dense_s.first;
    const double dense_slope=linear-a*dense_d.second-tau*dense_s.second;
    check_close(ray_delta,dense_delta,2e-10);check_close(ray_slope,dense_slope,2e-10);}
  CovarianceBlockNewtonResult block;block.status="PASS";block.ray_status="PASS";block.direction=direction;
  block.ray_d=spectrum_d;block.ray_s=spectrum_s;block.ray_eigen_evaluations=4;block.ray_eigen_seconds=0.001;
  const CovarianceJointRayProposal proposal=covariance_joint_ray_proposal(block,groups,statistics,baseline_trace,scale,rho,tau,slope_zero,1.0);
  assert(proposal.usable&&proposal.status=="PROPOSED"&&proposal.slope_relative_error<1e-6&&proposal.cheap_evaluations<=64);
  assert(proposal.proposed_alpha>0.0&&proposal.proposed_alpha<=4.0&&proposal.ray_slope<proposal.ray_slope_target);
  {
    CovarianceBlockNewtonResult boundary;boundary.status="PASS";boundary.ray_status="PASS";boundary.direction.assign(6,0.0);boundary.direction[0]=1.0;
    boundary.ray_d.status=boundary.ray_s.status="PASS";boundary.ray_d.eigenvalues={-2.0};boundary.ray_s.eigenvalues={-1.0};
    boundary.ray_d.ray_min_eigenvalue=-2.0;boundary.ray_s.ray_min_eigenvalue=-1.0;
    const std::vector<int> selected={0};const std::vector<double> scale_one(6,1.0),base(6,0.0),stats={-10.0,0,0,0,0,0};
    const double exact_slope=-5.0-0.55*(-2.0)-0.1*(-1.0);
    const CovarianceJointRayProposal inside=covariance_joint_ray_proposal(boundary,selected,stats,base,scale_one,0.1,0.1,exact_slope,1.0);
    assert(inside.usable&&inside.feasible_boundary==0.5&&inside.model_upper==0.45&&inside.proposed_alpha<inside.feasible_boundary);
    auto rotated_matrix=[](const double low,const double high){return std::vector<double>{0.5*(low+high),0.5*(low-high),0.5*(low-high),0.5*(low+high)};};
    auto cholesky2=[](const std::vector<double>& matrix,const double alpha){double a=1.0+alpha*matrix[0],b=alpha*matrix[1],c=alpha*matrix[3];
      if(!(a>0.0)||!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c))return false;
      const double l=std::sqrt(a),pivot=c-b*b/a;return std::isfinite(l)&&std::isfinite(pivot)&&pivot>0.0;};
    const auto d_inside=rotated_matrix(-2.0,0.1),s_inside=rotated_matrix(-1.0,0.2);
    assert(cholesky2(d_inside,inside.proposed_alpha)&&cholesky2(s_inside,inside.proposed_alpha));
    assert(!cholesky2(d_inside,0.55));
    boundary.ray_d.eigenvalues={-0.1};boundary.ray_s.eigenvalues={0.1};boundary.ray_d.ray_min_eigenvalue=-0.1;
    const double outside_slope=-5.0-0.55*(-0.1)-0.1*0.1;
    const CovarianceJointRayProposal outside=covariance_joint_ray_proposal(boundary,selected,stats,base,scale_one,0.1,0.1,outside_slope,1.0);
    assert(outside.usable&&outside.feasible_boundary==10.0&&outside.model_upper==4.0&&outside.proposed_alpha==4.0);
    const auto d_outside=rotated_matrix(-0.1,0.2),s_outside=rotated_matrix(0.1,0.3);
    assert(cholesky2(d_outside,outside.proposed_alpha)&&cholesky2(s_outside,outside.proposed_alpha));
    assert(!cholesky2(d_outside,11.0));
    boundary.ray_d.eigenvalues={0.2};boundary.ray_s.eigenvalues={0.1};boundary.ray_d.ray_min_eigenvalue=0.2;
    const CovarianceJointRayProposal unbounded=covariance_joint_ray_proposal(boundary,selected,stats,base,scale_one,0.1,0.1,-5.0-0.55*0.2-0.1*0.1,1.0);
    assert(unbounded.usable&&std::isinf(unbounded.feasible_boundary)&&unbounded.proposed_alpha==4.0);
    const CovarianceJointRayProposal mismatch=covariance_joint_ray_proposal(boundary,selected,stats,base,scale_one,0.1,0.1,-1.0,1.0);
    assert(!mismatch.usable&&mismatch.status=="FALLBACK_MODEL_SLOPE_MISMATCH");
    const std::vector<double> underflow_stats={-2e-308,0,0,0,0,0};boundary.ray_d.eigenvalues={0.0};boundary.ray_s.eigenvalues={0.0};
    const CovarianceJointRayProposal no_progress=covariance_joint_ray_proposal(boundary,selected,underflow_stats,base,scale_one,0.1,0.1,-1e-308,1e-320);
    assert(!no_progress.usable&&no_progress.status=="FALLBACK_NO_MODEL_PROGRESS");
  }
  {
    SmallEigen damaged;damaged.n=2;damaged.values={1.0,2.0};damaged.vectors={0.0,1.0,1.0,0.0};
    assert(covariance_symmetric_eigen_residual({1.0,0.0,0.0,2.0},damaged,2)>1e-8);
    const auto roundoff_negative=covariance_joint_ray_spectrum(solver,{1.0,0.0,0.0,-1e-12},{1.0,0.0,0.0,1.0},2);
    assert(roundoff_negative.status=="PASS");
    const auto material_negative=covariance_joint_ray_spectrum(solver,{1.0,0.0,0.0,-1e-5},{1.0,0.0,0.0,1.0},2);
    assert(material_negative.status=="INVERSE_NOT_PSD");
    const auto asymmetry=covariance_joint_ray_spectrum(solver,{1.0,1e-4,0.0,1.0},{1.0,0.0,0.0,1.0},2);
    assert(asymmetry.status=="INVERSE_ASYMMETRY");
    bool nonfinite_failed=false;try{(void)covariance_joint_ray_spectrum(solver,{1.0,0.0,0.0,std::numeric_limits<double>::infinity()},{1.0,0.0,0.0,1.0},2);}
      catch(const std::runtime_error&){nonfinite_failed=true;}assert(nonfinite_failed);
    bool dimensions_failed=false;try{(void)covariance_joint_ray_spectrum(solver,{1.0},{1.0},2);}
      catch(const std::invalid_argument&){dimensions_failed=true;}assert(dimensions_failed);
  }
}

void test_covariance_joint_block_commit_attribution_is_transactional()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  {
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    assert(fit.joint_block_enabled&&fit.last_joint_attribution.available&&fit.last_joint_attribution.model_status=="COMPUTED");
    assert(fit.bfgs_recent_pairs==1&&trace.str().find("bfgs_history_mode=RECENT_REPLAY")!=std::string::npos);
    assert(fit.joint_ray_attempt.iteration==0&&fit.joint_ray_attempt.line_status=="STEP_COMMITTED");
    assert(fit.joint_ray_attempt.usable&&fit.joint_ray_attempt.accepted&&fit.joint_ray_committed.usable);
    assert(fit.block_newton_factorizations==0&&fit.block_newton_rhs_columns==0);
    assert(fit.joint_block_factorizations>0&&fit.joint_block_rhs_columns>0);
    assert(trace.str().find("COVARIANCE_JOINT_COMMIT_ATTRIBUTION iteration=0")!=std::string::npos);
    const CovarianceJointStepAttribution& attribution=fit.last_joint_attribution;
    assert(attribution.actual_step.size()==6&&attribution.predicted_gradient.size()==6&&attribution.model_hessian.size()==36);
    std::vector<double> delta(6),model_action(6);
    for(int i=0;i<6;++i){delta[i]=attribution.observed_gradient[i]-attribution.gradient_before[i];
      for(int j=0;j<6;++j)model_action[i]+=attribution.model_hessian[static_cast<std::size_t>(i)*6+j]*attribution.actual_step[j];}
    double expected_denominator=std::max(covariance_norm_inf(delta),covariance_norm_inf(model_action));
    expected_denominator=std::max(expected_denominator,1e-300);
    for(int i=0;i<6;++i)check_close(attribution.predicted_gradient[i],attribution.gradient_before[i]+model_action[i],1e-12);
    for(int i=0;i<6;++i)delta[i]-=model_action[i];
    check_close(attribution.model_discrepancy_denominator,expected_denominator,1e-12);
    check_close(attribution.normalized_model_discrepancy,covariance_norm_inf(delta)/expected_denominator,1e-12);
  }
  {
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=1;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    const std::vector<double> rejected_theta=covariance_fit_test_hooks.last_residual_rejected_theta;covariance_fit_test_hooks={};
    const std::string log=trace.str();const std::size_t rejected=log.find("GRADIENT_RESIDUAL_REJECT iteration=0 ");
    const std::size_t committed=log.find("STEP_COMMITTED iteration=0"),attribution_pos=log.find("COVARIANCE_JOINT_COMMIT_ATTRIBUTION iteration=0");
    assert(fit.gradient_residual_reject_trials==1&&fit.iterations==1&&fit.last_valid.theta==fit.theta);
    assert(fit.bfgs_recent_pairs==1);
    assert(fit.last_joint_attribution.available&&fit.last_joint_attribution.iteration==0&&fit.last_joint_attribution.tau==1.0);
    assert(fit.block_newton_factorizations==0&&fit.block_newton_rhs_columns==0);
    assert(fit.joint_block_factorizations>0&&fit.joint_block_rhs_columns>0);
    assert(!rejected_theta.empty()&&rejected_theta!=fit.theta&&rejected<committed&&committed<attribution_pos);
    assert(log.find("STEP_COMMITTED iteration=0 tau=1 alpha=")!=std::string::npos&&!fit.failed_trial.available);
    assert(fit.joint_ray_attempt.usable&&fit.joint_ray_attempt.trials>=2&&fit.joint_ray_attempt.line_status=="STEP_COMMITTED");
    check_close(fit.joint_ray_attempt.actual_accepted_alpha,fit.joint_ray_attempt.proposed_alpha*
      std::ldexp(1.0,1-fit.joint_ray_attempt.trials),1e-15);
    check_close(fit.joint_ray_committed.actual_accepted_alpha,fit.joint_ray_attempt.actual_accepted_alpha,1e-15);
    assert(log.find("COVARIANCE_JOINT_COMMIT_ATTRIBUTION",attribution_pos+1)==std::string::npos);
    const std::string path="rpmd_ja_native_joint_attribution_state_test_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
    RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=1e-6;options.fd_step=1e-4;options.sample_interval=1;options.covariance_joint_block=true;
    SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};
    write_covariance_state(path,fit,0.1,"test.samples","test.qraw",options,header,identity,graph,4,2);
    std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
    assert(fields.at("joint_block_enabled")=="1"&&fields.at("last_joint_attribution_available")=="1");
    assert(fields.at("last_joint_attribution_iteration")=="0"&&fields.at("last_joint_attribution_tau")=="1");
    assert(fields.at("last_joint_attribution_model_status")=="COMPUTED");
    assert(fields.at("last_joint_ray_attempt_status")=="PROPOSED"&&fields.at("last_joint_ray_attempt_line_status")=="STEP_COMMITTED");
    assert(fields.at("last_joint_ray_attempt_used")=="1"&&fields.at("last_joint_ray_committed_used")=="1");
    check_close(std::stod(fields.at("last_joint_ray_attempt_actual_accepted_alpha")),fit.joint_ray_attempt.actual_accepted_alpha,1e-15);
  }
  {
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=64;
    covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=16;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    covariance_fit_test_hooks={};
    assert(fit.status=="EVALUATION_BUDGET_EXHAUSTED"&&fit.iterations==0&&fit.last_valid.theta==fit.theta);
    assert(fit.joint_ray_attempt.iteration==0&&fit.joint_ray_attempt.line_status=="EVALUATION_BUDGET_EXHAUSTED");
    assert(!fit.joint_ray_attempt.accepted&&fit.joint_ray_attempt.actual_accepted_alpha!=fit.joint_ray_attempt.actual_accepted_alpha);
    assert(fit.joint_ray_committed.status=="NOT_APPLICABLE");
    assert(fit.bfgs_recent_pairs==0);
    assert(!fit.last_joint_attribution.available&&trace.str().find("COVARIANCE_JOINT_COMMIT_ATTRIBUTION")==std::string::npos);
  }
}

void test_covariance_joint_large_group_bfgs_commit_and_default()
{
  constexpr int n=36;Graph graph;graph.type_pairs={{0,0},{0,1}};
  for(int atom=0;atom<33;++atom)graph.edges.push_back({atom,atom+1,0,{0,0,0}});
  graph.edges.push_back({34,35,1,{0,0,0}});
  const std::vector<double> masses(n,1.0),sqrt_mass(3*n,1.0);
  std::vector<double> statistics(12,0.0);for(int component=0;component<6;++component){statistics[component]=10.0;statistics[6+component]=0.01;}
  {
    DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=1;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    covariance_fit_test_hooks={};const std::string log=trace.str();
    assert(fit.joint_block_enabled&&fit.iterations==1&&fit.direction_method=="GROUP_BFGS");
    assert(fit.direction_group==0&&fit.direction_group_support==34&&fit.last_joint_attribution.available);
    assert(fit.last_joint_attribution.groups==std::vector<int>{0}&&fit.last_joint_attribution.group_supports==std::vector<std::size_t>{34});
    assert(fit.last_joint_attribution.parameter_count==6&&fit.last_joint_attribution.support_count==34);
    assert(fit.last_joint_attribution.model_status=="APPROXIMATE_MODEL_UNAVAILABLE");
    assert(fit.dominant_cap_rescue_next&&fit.dominant_cap_rescue_attempts==0&&fit.dominant_cap_rescue_commits==0);
    assert(fit.joint_ray_attempt.status=="NOT_USED_GROUP_BFGS"&&fit.joint_ray_attempt.line_status=="STEP_COMMITTED");
    assert(fit.joint_block_factorizations==0&&fit.joint_block_rhs_columns==0);
    assert(fit.bfgs_recent_pairs==1);
    assert(log.find("bfgs_history_mode=RECENT_REPLAY")!=std::string::npos);
    assert(fit.group_bfgs_committed_updates[0]==1&&fit.group_bfgs_update_status[0]=="SEEDED");
    assert(fit.group_bfgs_source[0]=="FIRST_VALID_SECANT"&&fit.group_bfgs_committed_updates[1]==0);
    assert(fit.gradient_residual_reject_trials==1&&log.find("selection_status=DOMINANT_SUPPORT_CAP")!=std::string::npos);
    assert(log.find("method=GROUP_BFGS")!=std::string::npos&&log.find("COVARIANCE_GROUP_BFGS_COMMIT")!=std::string::npos);
    const std::string path="rpmd_ja_native_group_bfgs_state_test_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
    RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=1e-6;options.fd_step=1e-4;options.sample_interval=1;options.covariance_joint_block=true;
    SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};
    write_covariance_state(path,fit,0.1,"test.samples","test.qraw",options,header,identity,graph,n,2);
    std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
    assert(fields.at("direction_method")=="GROUP_BFGS"&&fields.at("direction_group_support")=="34");
    assert(fields.at("last_joint_attribution_parameter_count")=="6");
    assert(fields.at("dominant_cap_rescue_next")=="1"&&fields.at("dominant_cap_rescue_attempts")=="0");
    assert(fields.at("group_bfgs_direction_source")=="UNIT_IDENTITY"&&fields.at("group_bfgs_group_0_update_status")=="SEEDED");
    assert(fields.at("bfgs_recent_pairs")=="1");
    assert(fields.at("last_joint_ray_attempt_status")=="NOT_USED_GROUP_BFGS");
    assert(fields.at("inverse_bfgs_spectrum_snapshot")=="DIRECTION_BEFORE_STEP"&&fields.at("inverse_bfgs_spectrum_iteration")=="0");
    assert(fields.at("run_trace_residual_original_any_checked")=="0"&&fields.at("run_trace_residual_worst_original_matrix_kind")=="NOT_CHECKED");
    assert(std::isnan(std::stod(fields.at("run_trace_residual_worst_original_relative_residual"))));
    assert(std::stod(fields.at("inverse_bfgs_spectrum_total_seconds"))>=std::stod(fields.at("inverse_bfgs_spectrum_seconds")));
  }
  {
    DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=10;
    covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=2;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    covariance_fit_test_hooks={};const std::string log=trace.str();
    assert(fit.direction_method=="GROUP_BFGS"&&fit.direction_group==0&&fit.direction_group_support==34);
    assert(fit.iterations==0&&!fit.converged&&!fit.last_joint_attribution.available&&fit.bfgs_recent_pairs==0);
    assert(fit.status=="EVALUATION_BUDGET_EXHAUSTED"&&fit.objective_evaluations>0);
    assert(fit.gradient_residual_reject_trials>0&&fit.failed_trial.failure_stage=="GRADIENT_RESIDUAL_REJECT");
    assert(!fit.dominant_cap_rescue_next&&fit.dominant_cap_rescue_attempts==0&&fit.dominant_cap_rescue_commits==0);
    assert(fit.theta==fit.theta_start&&fit.last_valid.available&&fit.last_valid.theta==fit.theta_start);
    assert(fit.group_bfgs_direction_gamma==1.0&&fit.group_bfgs_direction_source=="UNIT_IDENTITY");
    assert(fit.group_bfgs_committed_updates[0]==0&&fit.group_bfgs_update_status[0]=="NOT_UPDATED");
    assert(fit.group_bfgs_gamma[0]==1.0&&fit.group_bfgs_source[0]=="UNIT_IDENTITY");
    assert(log.find("selection_status=DOMINANT_SUPPORT_CAP")!=std::string::npos&&log.find("method=GROUP_BFGS")!=std::string::npos);
    assert(log.find("COVARIANCE_GROUP_BFGS_COMMIT")==std::string::npos&&log.find("REFERENCE_WRITTEN")==std::string::npos);
  }
  {
    DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,2,&trace,false,true);
    const std::string log=trace.str();
    assert(fit.iterations==2&&fit.dominant_cap_rescue_attempts==1&&fit.dominant_cap_rescue_commits==1);
    assert(fit.bfgs_recent_pairs==2);
    assert(log.find("bfgs_history_mode=RECENT_REPLAY")!=std::string::npos);
    assert(fit.last_joint_attribution.available&&fit.last_joint_attribution.method=="BFGS");
    assert(fit.last_joint_attribution.reason=="GROUP_GLOBAL_RESCUE"&&fit.last_joint_attribution.parameter_count>6);
    assert(std::find(fit.last_joint_attribution.groups.begin(),fit.last_joint_attribution.groups.end(),1)!=fit.last_joint_attribution.groups.end());
    assert(fit.group_bfgs_committed_updates[0]==1&&fit.group_bfgs_committed_updates[1]==0);
    assert(!fit.dominant_cap_rescue_next&&log.find("method=BFGS reason=GROUP_GLOBAL_RESCUE")!=std::string::npos);
  }
  {
    DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    covariance_fit_test_hooks={};const CovarianceFitResult single_step=solve_covariance_reference(system,statistics,1e-6,0.1,1,nullptr,false,true);
    DeviceBaseline retry_baseline;initialize_identity_covariance_baseline(retry_baseline,n);
    DeviceCovarianceSystem retry_system(retry_baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=64;
    covariance_fit_test_hooks.gradient_residual_failure_iteration=1;
    covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=32;
    const CovarianceFitResult rejected_rescue=solve_covariance_reference(retry_system,statistics,1e-6,0.1,2,&trace,false,true);
    covariance_fit_test_hooks={};const std::string log=trace.str();
    assert(single_step.iterations==1&&rejected_rescue.iterations==1);
    assert(single_step.bfgs_recent_pairs==1&&rejected_rescue.bfgs_recent_pairs==single_step.bfgs_recent_pairs);
    assert(rejected_rescue.theta==single_step.theta&&rejected_rescue.last_valid.theta==single_step.last_valid.theta);
    assert(rejected_rescue.gradient_residual_reject_trials>0);
    assert(rejected_rescue.dominant_cap_rescue_attempts==1&&rejected_rescue.dominant_cap_rescue_commits==0);
    assert(rejected_rescue.dominant_cap_rescue_next&&rejected_rescue.group_bfgs_committed_updates[0]==1);
    assert(rejected_rescue.group_bfgs_source[0]==single_step.group_bfgs_source[0]);
    assert(rejected_rescue.group_bfgs_gamma[0]==single_step.group_bfgs_gamma[0]);
    assert(rejected_rescue.group_bfgs_update_status[0]==single_step.group_bfgs_update_status[0]);
    assert(log.find("method=BFGS reason=GROUP_GLOBAL_RESCUE")!=std::string::npos);
    assert(log.find("LINE_SEARCH_FAILURE iter=1")!=std::string::npos);
  }
  {
    DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,false);
    assert(!fit.joint_block_enabled&&fit.direction_method=="BFGS"&&fit.joint_block_factorizations==0&&fit.joint_block_rhs_columns==0);
    assert(trace.str().find("method=GROUP_BFGS")==std::string::npos);
  }
}

void test_covariance_baseline_brackets_both_signs_and_zero()
{
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};const Graph graph=make_two_atom_covariance_graph();
  const auto numeric_pass=[](const CovarianceBaseline& reference,const std::uint64_t evaluations){
    assert(std::isfinite(reference.phi)&&std::isfinite(reference.phi_relative)&&
      std::isfinite(reference.phi_solve_residual_heuristic)&&std::isfinite(reference.root_relative_tolerance));
    assert(reference.minimum_shifted_cholesky_diagonal>0.0&&reference.epsilon>0.0);
    assert(evaluations<=200);
    assert(!reference.epsilon_constraint_active);
    assert(reference.stationarity_status=="FREE_ROOT"||reference.stationarity_status=="SAVED_BASELINE_FREE_ROOT_VERIFIED");
    assert(reference.phi_relative<=reference.root_relative_tolerance);
    assert(reference.root_bracket_width>=0.0&&reference.root_iterations<=64);
  };
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};std::uint64_t evaluations=0;
    std::uint64_t optimizer_evaluations=0;
    const CovarianceBaseline optimizer_reference=build_graph_covariance_baseline(system,statistics,0.01,optimizer_evaluations);
    check_close(optimizer_reference.mu,2.533333333333333,2e-7);assert(!optimizer_reference.epsilon_constraint_active);
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,0.01,evaluations);
    check_close(reference.mu,2.533333333333333,2e-7);assert(!reference.epsilon_constraint_active);numeric_pass(reference,evaluations);
    GraphReferenceBaselineCandidate saved;saved.mu=reference.mu;saved.theta=reference.theta;
    std::uint64_t candidate_evaluations=0;
    const CovarianceBaseline reused=build_graph_reference_baseline(system,statistics,0.01,candidate_evaluations,
      nullptr,nullptr,nullptr,&saved);
    check_close(reused.mu,reference.mu,1e-12);assert(candidate_evaluations==1);
    assert(reused.stationarity_status=="SAVED_BASELINE_FREE_ROOT_VERIFIED");
    int injected_residual=0;
    system.before_factor=[&](){if(injected_residual++==0){CovarianceTraceDiagnostics diagnostic;
      diagnostic.matrix_kind="D";diagnostic.status="RESIDUAL_FAIL";throw CovarianceSolveResidualError(diagnostic);}};
    candidate_evaluations=0;
    const CovarianceBaseline retried=build_graph_reference_baseline(system,statistics,0.01,candidate_evaluations,
      nullptr,nullptr,nullptr,&saved);
    system.before_factor={};
    assert(retried.stationarity_status=="FREE_ROOT"&&candidate_evaluations>1);
    system.before_factor=[](){throw std::runtime_error("injected graph candidate hard failure");};
    bool hard_failure_propagated=false;
    try{(void)build_graph_reference_baseline(system,statistics,0.01,candidate_evaluations,nullptr,nullptr,nullptr,&saved);}
    catch(const std::runtime_error& error){hard_failure_propagated=std::string(error.what())=="injected graph candidate hard failure";}
    system.before_factor={};assert(hard_failure_propagated);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={2.5,0.0,0.0,2.5,0.0,2.5};std::uint64_t evaluations=0;
    std::uint64_t optimizer_evaluations=0;
    const CovarianceBaseline optimizer_reference=build_graph_covariance_baseline(system,statistics,0.01,optimizer_evaluations);
    check_close(optimizer_reference.mu,-0.4,2e-7);assert(!optimizer_reference.epsilon_constraint_active);
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,0.01,evaluations);
    check_close(reference.mu,-0.4,2e-7);assert(!reference.epsilon_constraint_active);numeric_pass(reference,evaluations);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::uint64_t evaluations=0;
    std::uint64_t optimizer_evaluations=0;
    const CovarianceBaseline optimizer_reference=build_graph_covariance_baseline(system,statistics,1e-6,optimizer_evaluations);
    check_close(optimizer_reference.mu,0.0,2e-7);assert(!optimizer_reference.epsilon_constraint_active);
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,1e-6,evaluations);
    check_close(reference.mu,0.0,2e-7);assert(!reference.epsilon_constraint_active);numeric_pass(reference,evaluations);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={1000.0,0.0,0.0,1000.0,0.0,1000.0};std::uint64_t evaluations=0;
    std::uint64_t optimizer_evaluations=0;
    const CovarianceBaseline optimizer_reference=build_graph_covariance_baseline(system,statistics,0.01,optimizer_evaluations);
    assert(optimizer_reference.epsilon_constraint_active&&optimizer_reference.mu>=optimizer_reference.boundary_mu);
    GraphReferenceSearchDiagnostics search;
    covariance_fit_test_hooks={};
    bool unconfirmed=false;
    try{(void)build_graph_reference_baseline(system,statistics,0.01,evaluations,nullptr,nullptr,nullptr,nullptr,&search);}
    catch(const std::runtime_error& error){unconfirmed=std::string(error.what()).find("BASELINE_NUMERICAL_UNCONFIRMED")!=std::string::npos;}
    const int valid_evaluations=covariance_fit_test_hooks.graph_reference_stationarity_valid_evaluations;covariance_fit_test_hooks={};
    assert(unconfirmed&&search.stage=="POSITIVE_SIGN_UNCONFIRMED"&&valid_evaluations>0);
  }
  {
    constexpr double epsilon=1.0e-8,expected_mu=1.3e-8;
    const std::vector<double> unit_masses={1.0,1.0},unit_sqrt_mass={1.0,1.0,1.0,1.0,1.0,1.0};
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,false);
    DeviceCovarianceSystem system(baseline,graph,unit_sqrt_mass,unit_masses);
    const double slope=3.0/expected_mu;
    const std::vector<double> statistics={slope/3.0,0.0,0.0,slope/3.0,0.0,slope/3.0};
    std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,epsilon,evaluations);
    check_close(reference.mu,expected_mu,1e-7);
    assert(reference.phi_relative<=1e-7&&reference.root_iterations<=64&&
      reference.minimum_shifted_cholesky_diagonal>0.0&&!reference.epsilon_constraint_active);
  }
  {
    constexpr double epsilon=1.0e-8,expected_mu=(epsilon/2.0)*(1.0+5.0e-5);
    const std::vector<double> unit_masses={1.0,1.0},unit_sqrt_mass={1.0,1.0,1.0,1.0,1.0,1.0};
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,false);
    DeviceCovarianceSystem system(baseline,graph,unit_sqrt_mass,unit_masses);
    const double slope=3.0/expected_mu;
    const std::vector<double> statistics={slope/3.0,0.0,0.0,slope/3.0,0.0,slope/3.0};
    std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,epsilon,evaluations);
    assert(std::abs(reference.boundary_mu-epsilon/2.0)<1e-16);
    assert(std::abs(reference.mu-expected_mu)<=1e-15);
    assert(reference.mu>reference.boundary_mu&&reference.mu-reference.boundary_mu<1e-12);
    assert(!reference.epsilon_constraint_active);
    assert(reference.phi_relative<=1e-7&&reference.minimum_shifted_cholesky_diagonal>0.0);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};std::uint64_t evaluations=0;
    GraphReferenceSearchDiagnostics search;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.trace_solution_failure_column=0;
    covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
    covariance_fit_test_hooks.trace_solution_failure_count_remaining=20;
    const CovarianceBaseline reference=build_graph_reference_baseline(system,statistics,0.01,evaluations,
      nullptr,nullptr,nullptr,nullptr,&search);
    const int injected_failures=20-covariance_fit_test_hooks.trace_solution_failure_count_remaining;
    covariance_fit_test_hooks={};
    numeric_pass(reference,evaluations);
    assert(injected_failures==20&&search.residual_failures==20);
    assert(search.last_residual_stage=="INTERIOR_SIGN_BRACKET"&&std::isfinite(search.last_residual_mu));
    assert(search.last_residual_d_trace.matrix_kind=="D"&&search.last_residual_d_trace.status=="RESIDUAL_FAIL");
    assert(search.last_residual_d_trace.factor_solve_relative_residual>1e-7);
  }
  {
    const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};
    DeviceBaseline probe_baseline;initialize_two_atom_covariance_baseline(probe_baseline,true);
    DeviceCovarianceSystem probe_system(probe_baseline,graph,sqrt_mass,masses);std::uint64_t probe_evaluations=0;
    (void)build_graph_reference_baseline(probe_system,statistics,0.01,probe_evaluations);
    assert(probe_system.trace_solve_calls>0);
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::uint64_t evaluations=0;
    GraphReferenceSearchDiagnostics search;covariance_fit_test_hooks={};
    covariance_fit_test_hooks.trace_solution_failure_column=0;covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
    covariance_fit_test_hooks.trace_solution_failure_skip=static_cast<int>(probe_system.trace_solve_calls-1);
    covariance_fit_test_hooks.trace_solution_failure_count_remaining=1;
    bool final_residual_failure=false;
    try{(void)build_graph_reference_baseline(system,statistics,0.01,evaluations,nullptr,nullptr,nullptr,nullptr,&search);}
    catch(const CovarianceSolveResidualError&){final_residual_failure=true;}
    covariance_fit_test_hooks={};
    assert(final_residual_failure&&search.stage=="FINAL_STRICT_CERTIFICATE");
    assert(search.last_residual_stage=="FINAL_STRICT_CERTIFICATE"&&search.last_residual_mu==search.mu&&
      search.residual_failures==1);
    assert(search.last_residual_d_trace.matrix_kind=="D"&&search.last_residual_d_trace.status=="RESIDUAL_FAIL"&&
      search.last_residual_d_trace.factor_solve_relative_residual>1e-7);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};std::uint64_t evaluations=0;
    GraphReferenceSearchDiagnostics search;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.trace_solution_failure_column=0;
    covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
    bool unconfirmed=false;
    try{(void)build_graph_reference_baseline(system,statistics,0.01,evaluations,nullptr,nullptr,nullptr,nullptr,&search);}
    catch(const std::runtime_error& error){unconfirmed=std::string(error.what()).find("BASELINE_NUMERICAL_UNCONFIRMED")!=std::string::npos;}
    CovarianceFitResult failure;capture_graph_reference_fit_diagnostics(failure,system,search);
    covariance_fit_test_hooks={};
    assert(unconfirmed&&search.stage=="POSITIVE_SIGN_UNCONFIRMED"&&search.residual_failures==64&&evaluations<=200);
    assert(search.last_residual_stage=="INTERIOR_SIGN_BRACKET"&&search.last_residual_mu==search.mu);
    assert(failure.factorizations>0&&failure.solve_rhs_columns>0&&failure.trace_solve_calls>0);
    assert(failure.failed_d_trace.matrix_kind=="D"&&failure.failed_d_trace.status=="RESIDUAL_FAIL");
    assert(failure.failed_d_trace.factor_solve_relative_residual>1e-7);
    assert(failure.run_trace_residual_summary.failed_logical_batches>0&&
      failure.run_trace_residual_summary.worst_factor_matrix_kind=="D");
  }
}

void test_a0_soft_modes_use_covariance_baseline_point(cusolverDnHandle_t solver)
{
  CovarianceFitResult a0;
  a0.theta={0.1,0.0,0.0,10.0,0.0,10.0};
  a0.covariance_baseline.theta={10.0,0.0,0.0,10.0,0.0,0.1};
  assert(a0.theta!=a0.covariance_baseline.theta);
  const std::vector<double>& mode_theta=covariance_a0_soft_mode_theta(a0);
  assert(&mode_theta==&a0.covariance_baseline.theta&&mode_theta!=a0.theta);

  constexpr int n=2,d=3*n;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;
  std::stringstream raw_stream(std::ios::in|std::ios::out|std::ios::binary);
  raw_stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));raw_stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(raw_stream,0,d,n,masses,sqrt_mass);
  Graph graph;graph.edges.push_back({0,1,0,{0,0,0}});
  const auto solved_modes=lanczos_low_modes(solver,baseline,graph,a0.theta,sqrt_mass,sqrt_atom,n,8,4,0.0);
  const auto baseline_modes=lanczos_low_modes(solver,baseline,graph,mode_theta,sqrt_mass,sqrt_atom,n,8,4,0.0);
  assert(!solved_modes.empty()&&!baseline_modes.empty());
  const auto reliable=[](const RitzMode& mode){return std::isfinite(mode.residual)&&
    mode.residual<=1.0e-6*std::max(std::abs(mode.actual_rayleigh),1.0e-12);};
  assert(reliable(solved_modes.front())&&reliable(baseline_modes.front()));
  assert(std::abs(covariance_dot(solved_modes.front().vector,baseline_modes.front().vector))<1.0e-8);
  std::vector<std::string> sources;const std::vector<int> types={0,1};
  const auto common=make_common_probe_set({baseline_modes.front()},sqrt_atom,types,n,&sources);
  assert(!common.empty()&&!sources.empty()&&sources.front()=="LOW_RITZ");
  assert(std::abs(covariance_dot(common.front(),baseline_modes.front().vector))>1.0-1.0e-10);
  assert(std::abs(covariance_dot(common.front(),solved_modes.front().vector))<1.0e-8);
}

void test_covariance_optimizer_updates_from_zero_baseline_and_recovers_harmonic_solution()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  system.run_trace_residual_summary.failed_logical_batches=17;system.run_trace_residual_summary.worst_factor_relative_residual=123.0;
  system.run_trace_residual_summary.worst_factor_matrix_kind="STALE";
  system.run_trace_residual_summary.original_any_checked=true;system.run_trace_residual_summary.original_batches_checked=11;
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::ostringstream trace;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,300,&trace);
  assert(fit.converged&&fit.status=="COVARIANCE_OPTIMIZER_CONVERGED"&&fit.iterations>0);
  assert(std::abs(fit.baseline_mu)<2e-7&&fit.barrier_center_condition_pass&&fit.barrier_gap_estimate<=1e-6);
  assert(fit.initialization_status=="SKIPPED_BASELINE_INTERIOR"&&fit.initialization_evaluations==1);
  assert(fit.initialization_gradient_calls==1&&fit.full_gradient_evaluations==fit.initialization_gradient_calls+fit.line_search_gradient_calls);
  assert(fit.theta_start==fit.theta_baseline&&fit.start_graph_delta==0.0);
  assert(trace.str().find("COVARIANCE_INITIALIZATION status SKIPPED_BASELINE_INTERIOR")!=std::string::npos);
  assert(trace.str().find("status=ARMIJO_ACCEPTED")!=std::string::npos&&trace.str().find("STEP_COMMITTED iteration=0")!=std::string::npos);
  const std::string trace_text=trace.str();const std::size_t first_trial=trace_text.find("LINE_SEARCH iter=0 ");
  const std::size_t first_step=trace_text.find("step_inf_norm=",first_trial);
  assert(first_trial!=std::string::npos&&first_step!=std::string::npos);
  std::istringstream first_step_value(trace_text.substr(first_step+std::string("step_inf_norm=").size()));
  double scaled_step_inf_norm=2.0;assert(first_step_value>>scaled_step_inf_norm&&scaled_step_inf_norm<=1.0);
  assert(fit.line_search_evaluations<=fit.line_search_trials);
  assert(fit.shifted_not_spd_trials+fit.armijo_reject_trials<=fit.line_search_evaluations);
  const double expected[6]={(1.1/0.35-1.0)/1.25,0.0,0.0,0.0,0.0,(1.1/1.85-1.0)/1.25};
  for(int i=0;i<6;++i)check_close(fit.theta[i],expected[i],2e-4);
  assert(std::isfinite(fit.raw_gradient_inf_norm)&&fit.raw_gradient_inf_norm<1e-5);
  std::uint64_t evaluations=0;const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,1e-6,evaluations);
  const CovarianceObjectiveEvaluation start=evaluate_covariance_fit_objective(system,fit.theta_start,reference,statistics,0.1,1e-6,1.0,evaluations);
  assert(start.feasible);check_close(start.objective,fit.start_objective,1e-12);
  check_close(covariance_norm_inf(start.gradient),fit.start_raw_gradient_inf_norm,1e-12);
  assert(fit.start_gradient_solve_status=="PASS"&&std::isfinite(fit.start_scaled_kkt_residual));
  check_close(fit.start_scaled_kkt_residual,covariance_norm_inf(covariance_scaled_gradient(start.gradient,fit.coordinate_scale))/fit.kkt_gradient_denominator,1e-12);
  check_close(fit.start_gradient_data_scaled_inf_norm,covariance_norm_inf(covariance_scaled_gradient(start.gradient_data,fit.coordinate_scale)),1e-12);
  check_close(fit.start_gradient_prior_scaled_inf_norm,covariance_norm_inf(covariance_scaled_gradient(start.gradient_prior,fit.coordinate_scale)),1e-12);
  check_close(fit.start_gradient_barrier_scaled_inf_norm,covariance_norm_inf(covariance_scaled_gradient(start.gradient_barrier,fit.coordinate_scale)),1e-12);
  assert(fit.inverse_bfgs_spectrum.status!="NOT_EVALUATED"&&fit.inverse_bfgs_spectrum.snapshot_stage=="DIRECTION_BEFORE_STEP"&&
    fit.inverse_bfgs_spectrum.iteration>=0&&fit.inverse_bfgs_spectrum.iteration<fit.iterations);
  assert(std::isfinite(fit.inverse_bfgs_spectrum.seconds)&&
    fit.inverse_bfgs_spectrum_seconds>=fit.inverse_bfgs_spectrum.seconds);
  assert(fit.run_trace_residual_summary.factor_batches_checked>0&&fit.run_trace_residual_summary.failed_logical_batches==0);
  assert(fit.run_trace_residual_summary.original_batches_checked==0&&!fit.run_trace_residual_summary.original_any_checked);
  assert(fit.run_trace_residual_summary.worst_factor_matrix_kind!="STALE"&&fit.run_trace_residual_summary.worst_factor_relative_residual<1.0);
  const std::size_t first_spectrum=trace_text.find("COVARIANCE_BFGS_SPECTRUM snapshot=DIRECTION_BEFORE_STEP iteration=0");
  assert(first_spectrum!=std::string::npos);
  const std::string first_spectrum_line=trace_text.substr(first_spectrum,trace_text.find('\n',first_spectrum)-first_spectrum);
  assert(first_spectrum_line.find("status=SPD_VERIFIED")!=std::string::npos&&first_spectrum_line.find("condition_number=1")!=std::string::npos);
  assert(trace_text.find("inverse_bfgs_spectrum_total_seconds=")!=std::string::npos);
  assert(trace_text.find("bfgs_secant_cosine=")!=std::string::npos);
}

void test_covariance_trust_region_staged_barrier_harmonic_solve()
{
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  std::ostringstream trace;covariance_fit_test_hooks={};
  covariance_fit_test_hooks.trust_region_forced_rejections_remaining=1;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,300,&trace,false,false,true,true,
    {},{},0.0,16);
  const int full_directional_checks=covariance_fit_test_hooks.directional_curvature_reference_checks;
  const int step_directional_checks=covariance_fit_test_hooks.directional_curvature_step_checks;
  const int forced_rejections_remaining=covariance_fit_test_hooks.trust_region_forced_rejections_remaining;
  covariance_fit_test_hooks={};const std::string log=trace.str();
  assert(fit.converged&&fit.status=="COVARIANCE_OPTIMIZER_CONVERGED"&&fit.trust_region_enabled&&fit.staged_barrier_enabled);
  assert(fit.scaled_kkt_residual<=1e-6&&fit.barrier_center_condition_pass&&fit.barrier_gap_estimate<=1e-6);
  assert(log.find("BFGS_STAGE_RESET tau=0.2")!=std::string::npos&&log.find("BFGS_STAGE_RESET tau=0.04")!=std::string::npos);
  int full_summaries=0,step_summaries=0,tau_stage_summaries=0,step_summaries_with_one_direction=0;
  std::istringstream lines(log);std::string line;
  while(std::getline(lines,line)){if(line.find("COVARIANCE_CURVATURE_DIRECTION_SUMMARY ")==std::string::npos)continue;
    if(line.find("trigger=SUGGESTED_STEP")!=std::string::npos){++step_summaries;
      if(line.find("checks=1")!=std::string::npos)++step_summaries_with_one_direction;}
    else{++full_summaries;if(line.find("trigger=TAU_STAGE")!=std::string::npos)++tau_stage_summaries;}}
  const auto count_text=[&](const std::string& needle){int count=0;std::size_t position=0;
    while((position=log.find(needle,position))!=std::string::npos){++count;position+=needle.size();}return count;};
  assert(full_directional_checks==full_summaries&&step_directional_checks==step_summaries&&step_summaries>0);
  assert(step_summaries==static_cast<int>(fit.line_search_trials));
  assert(count_text("COVARIANCE_CURVATURE_SUGGESTED_STEP iteration=")==static_cast<int>(fit.line_search_trials));
  assert(forced_rejections_remaining==0);
  assert(step_summaries_with_one_direction==step_summaries);
  assert(tau_stage_summaries==count_text("BFGS_STAGE_RESET tau="));
  const auto field_value=[](const std::string& text,const std::string& name){
    const std::string key=name+"=";const std::size_t begin=text.find(key);
    if(begin==std::string::npos)throw std::runtime_error("trust-region trace is missing field "+name);
    const std::size_t value_begin=begin+key.size(),end=text.find(' ',value_begin);
    return std::stod(text.substr(value_begin,end-value_begin));};
  const std::size_t forced_rejection=log.find("COVARIANCE_TEST_FORCED_TRUST_REGION_REJECTION ");
  assert(forced_rejection!=std::string::npos);
  const std::string forced_line=log.substr(forced_rejection,log.find('\n',forced_rejection)-forced_rejection);
  const int injected_iteration=static_cast<int>(field_value(forced_line,"iteration"));
  const int injected_retry=static_cast<int>(field_value(forced_line,"retry"));
  const std::string rejected_prefix="TRUST_REGION_TRIAL iteration="+std::to_string(injected_iteration)+
    " retry="+std::to_string(injected_retry);
  const std::size_t rejected_trial=log.find(rejected_prefix,forced_rejection);
  const std::string retry_prefix="COVARIANCE_CURVATURE_SUGGESTED_STEP iteration="+std::to_string(injected_iteration)+
    " retry="+std::to_string(injected_retry+1);
  const std::size_t retry_direction=log.find(retry_prefix,rejected_trial);
  assert(rejected_trial!=std::string::npos&&retry_direction!=std::string::npos);
  const std::string rejected_line=log.substr(rejected_trial,log.find('\n',rejected_trial)-rejected_trial);
  const std::string retry_line=log.substr(retry_direction,log.find('\n',retry_direction)-retry_direction);
  assert(rejected_line.find("status=RATIO_REJECT")!=std::string::npos);
  assert(field_value(rejected_line,"next_radius")<field_value(rejected_line,"radius"));
  assert(field_value(retry_line,"radius")==field_value(rejected_line,"next_radius"));
  std::vector<std::string> accepted_taus;std::istringstream accepted_lines(log);
  while(std::getline(accepted_lines,line))if(line.find("TRUST_REGION_STEP_ACCEPT ")==0){
    const std::size_t tau_at=line.find(" tau=");if(tau_at!=std::string::npos){const std::size_t value_at=tau_at+5;
      const std::size_t end=line.find(' ',value_at);accepted_taus.push_back(line.substr(value_at,end-value_at));}}
  bool two_accepts_same_tau=false;
  for(std::size_t i=0;i<accepted_taus.size();++i)for(std::size_t j=0;j<i;++j)
    if(accepted_taus[i]==accepted_taus[j])two_accepts_same_tau=true;
  assert(two_accepts_same_tau);

  std::uint64_t evaluations=0;const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,1e-6,evaluations);
  std::vector<double> expected=fit.theta_baseline;
  for(std::size_t i=0;i<expected.size();++i){
    if(reference.trace[i]!=0.0){
      const double weighted_statistic=statistics[i]+0.1*reference.trace[i];
      const double target_ratio=1.1*reference.trace[i]/weighted_statistic;
      expected[i]+=(target_ratio-1.0)/reference.trace[i];
    }else assert(statistics[i]==0.0);
  }
  for(std::size_t i=0;i<expected.size();++i)check_close(fit.theta[i],expected[i],2e-4);
  assert(fit.last_valid.available&&fit.last_valid.theta==fit.theta&&fit.last_valid.tau==fit.barrier_tau);
  const CovarianceObjectiveEvaluation final=evaluate_covariance_fit_objective(system,fit.theta,reference,statistics,0.1,1e-6,
    fit.barrier_tau,evaluations);
  assert(final.feasible);check_close(final.data,fit.objective_data,1e-12);check_close(final.prior,fit.objective_prior,1e-12);
  check_close(final.barrier,fit.objective_barrier,1e-12);check_close(final.objective,fit.objective_total,1e-12);
  check_close(final.objective,fit.last_valid.objective,1e-12);
  const double final_kkt=covariance_norm_inf(covariance_scaled_gradient(final.gradient,fit.coordinate_scale))/fit.kkt_gradient_denominator;
  check_close(final_kkt,fit.scaled_kkt_residual,1e-12);

  DeviceBaseline bounded_baseline;initialize_two_atom_covariance_baseline(bounded_baseline,true);
  DeviceCovarianceSystem bounded_system(bounded_baseline,graph,sqrt_mass,masses);std::ostringstream bounded_trace;
  covariance_fit_test_hooks={};covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=2;
  const CovarianceFitResult bounded=solve_covariance_reference(bounded_system,statistics,1e-6,0.1,300,&bounded_trace,false,false,true,true);
  covariance_fit_test_hooks={};
  assert(!bounded.converged&&bounded.status=="EVALUATION_BUDGET_EXHAUSTED"&&bounded.trust_region_enabled&&bounded.staged_barrier_enabled);
  assert(bounded.last_valid.available);
}

void test_native_covariance_mid_a0_resume_preserves_original_start_and_stagnation_window()
{
  constexpr double epsilon=0.01,ridge=0.7;
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  DeviceBaseline device;initialize_two_atom_covariance_baseline(device,true);
  DeviceCovarianceSystem system(device,graph,sqrt_mass,masses);std::uint64_t evaluations=0;
  const CovarianceBaseline baseline=build_graph_covariance_baseline(system,statistics,epsilon,evaluations);
  const std::vector<double> scales=system.coordinate_scales(baseline.theta,epsilon);
  const std::vector<double> original_start=baseline.theta;
  rpmd_ja_feasibility::Budget budget;rpmd_ja_feasibility::Limits limits;
  const double now=covariance_budget_now_seconds();
  assert(rpmd_ja_feasibility::begin(budget,limits,now)==rpmd_ja_feasibility::BudgetResult::OK);
  assert(rpmd_ja_feasibility::enter_phase(budget,limits,rpmd_ja_feasibility::Phase::A0,now)==
    rpmd_ja_feasibility::BudgetResult::OK);
  rpmd_ja_covariance_checkpoint::AcceptedPoint metadata;metadata.job_id="native-mid-a0-resume";
  metadata.identity_sha256=std::string(64,'a');metadata.branch="A0";metadata.family="M0";
  metadata.candidate_index=0;metadata.selection_state="NOT_FROZEN";metadata.epsilon=epsilon;
  metadata.ridge_lambda=ridge;metadata.tau=1.0;metadata.trust_radius=1.0;
  const std::string path="rpmd_ja_mid_a0_resume_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".checkpoint";
  rpmd_ja_covariance_checkpoint::AcceptedPoint saved;
  const auto sink=[&](const rpmd_ja_covariance_checkpoint::AcceptedPoint& point){saved=point;};
  covariance_fit_test_hooks={};covariance_fit_test_hooks.checkpoint_interruptions_remaining=1;
  const CovarianceFitResult interrupted=solve_covariance_reference(system,statistics,epsilon,0.0,1,nullptr,
    false,false,true,false,{}, {},ridge,16,&baseline,&scales,&original_start,1.0,true,&budget,&limits,nullptr,&metadata,sink);
  covariance_fit_test_hooks={};
  assert(saved.iterations>=1&&saved.theta_start==original_start&&saved.theta.size()==original_start.size());
  assert(saved.theta!=saved.theta_start);
  assert(interrupted.reason.find("TEST_CHECKPOINT_INTERRUPT")!=std::string::npos);
  rpmd_ja_covariance_checkpoint::write_checkpoint(path,saved);
  const auto restored=rpmd_ja_covariance_checkpoint::read_checkpoint(path,metadata.identity_sha256);
  rpmd_ja_covariance_checkpoint::AcceptedPoint first_resumed;
  bool captured_first_resume_checkpoint=false;
  const auto resume_sink=[&](const rpmd_ja_covariance_checkpoint::AcceptedPoint& point){
    if(!captured_first_resume_checkpoint){first_resumed=point;captured_first_resume_checkpoint=true;}};
  const CovarianceFitResult resumed=solve_covariance_reference(system,statistics,epsilon,0.0,2,nullptr,
    false,false,true,false,{}, {},ridge,16,&baseline,&scales,&original_start,1.0,true,&budget,&limits,&restored,&metadata,resume_sink);
  std::remove(path.c_str());
  assert(captured_first_resume_checkpoint&&resumed.theta_start==original_start&&first_resumed.theta_start==original_start);
  assert(first_resumed.theta==saved.theta&&first_resumed.stagnation_gradient_attempts==saved.stagnation_gradient_attempts&&
    first_resumed.stagnation_low_progress_windows==saved.stagnation_low_progress_windows&&
    first_resumed.stagnation_progress_attempts_in_window==saved.stagnation_progress_attempts_in_window&&
    first_resumed.same_point_curvature_rejections==saved.same_point_curvature_rejections);
  assert(resumed.start_metrics_point=="RESTORED_ACCEPTED_POINT");
  std::uint64_t independent_evaluations=0;
  const CovarianceObjectiveEvaluation independent=evaluate_covariance_fit_objective(system,first_resumed.theta,baseline,statistics,
    0.0,epsilon,1.0,independent_evaluations,true,false,true,nullptr,nullptr,&scales,ridge);
  check_close(resumed.start_objective,independent.objective,1e-12);
  check_close(resumed.start_raw_gradient_inf_norm,covariance_norm_inf(independent.gradient),1e-12);
  DeviceBaseline continuous_device;initialize_two_atom_covariance_baseline(continuous_device,true);
  DeviceCovarianceSystem continuous_system(continuous_device,graph,sqrt_mass,masses);
  rpmd_ja_feasibility::Budget continuous_budget;
  const double continuous_now=covariance_budget_now_seconds();
  assert(rpmd_ja_feasibility::begin(continuous_budget,limits,continuous_now)==rpmd_ja_feasibility::BudgetResult::OK);
  assert(rpmd_ja_feasibility::enter_phase(continuous_budget,limits,rpmd_ja_feasibility::Phase::A0,continuous_now)==
    rpmd_ja_feasibility::BudgetResult::OK);
  const CovarianceFitResult continuous=solve_covariance_reference(continuous_system,statistics,epsilon,0.0,2,nullptr,
    false,false,true,false,{}, {},ridge,16,&baseline,&scales,&original_start,1.0,true,&continuous_budget,&limits);
  assert(resumed.theta.size()==continuous.theta.size());
  for(std::size_t parameter=0;parameter<resumed.theta.size();++parameter)
    check_close(resumed.theta[parameter],continuous.theta[parameter],1e-9);
}

void test_covariance_inverse_bfgs_spectrum_statuses()
{
  cusolverDnHandle_t solver=nullptr;assert(cusolverDnCreate(&solver)==CUSOLVER_STATUS_SUCCESS);
  const auto spd=covariance_inverse_bfgs_spectrum(solver,{1e-80,0.0,0.0,0.0,2e-80,0.0,0.0,0.0,4e-80},3);
  assert(spd.status=="SPD_VERIFIED"&&spd.matrix_scale==4e-80&&spd.condition_number>=3.999999999&&spd.condition_number<=4.000000001);
  assert(spd.min_eigenvalue>0.0&&spd.max_eigenvalue>=spd.min_eigenvalue&&spd.relative_asymmetry==0.0);
  assert(std::isfinite(spd.eigen_residual)&&spd.eigen_residual<=1e-8&&std::isfinite(spd.seconds));

  const auto indefinite=covariance_inverse_bfgs_spectrum(solver,{1.0,0.0,0.0,0.0,-2.0,0.0,0.0,0.0,3.0},3);
  assert(indefinite.status=="INDEFINITE"&&indefinite.min_eigenvalue<0.0&&std::isnan(indefinite.condition_number));
  const auto singular=covariance_inverse_bfgs_spectrum(solver,{1.0,0.0,0.0,0.0,1e-17,0.0,0.0,0.0,2.0},3);
  assert(singular.status=="UNRESOLVED_NEAR_SINGULAR"&&std::isnan(singular.condition_number));

  const auto asymmetric=covariance_inverse_bfgs_spectrum(solver,{2.0,0.01,0.0,0.0,3.0,0.0,0.0,0.0,4.0},3);
  assert(asymmetric.status=="UNVERIFIED_ASYMMETRY"&&asymmetric.relative_asymmetry>1e-8);
  std::vector<double> over_cap(61*61,0.0);
  assert(covariance_inverse_bfgs_spectrum(solver,over_cap,61).status=="NOT_EVALUATED_PARAMETER_COUNT");
  over_cap[0]=std::numeric_limits<double>::quiet_NaN();bool nonfinite_failed=false;
  try{(void)covariance_inverse_bfgs_spectrum(solver,over_cap,61);}catch(const std::runtime_error&){nonfinite_failed=true;}
  assert(nonfinite_failed);
  assert(cusolverDnDestroy(solver)==CUSOLVER_STATUS_SUCCESS);
}

void test_covariance_trust_region_subproblem_and_barrier_cache()
{
  CovarianceTrustRegionModel interior_model;interior_model.positive_definite=true;interior_model.status="SPD_VERIFIED";
  interior_model.eigenvalues={1e-10};interior_model.eigenvectors={1.0};
  const std::vector<double> interior_gradient={1.0};
  const CovarianceTrustRegionStep interior=covariance_trust_region_subproblem(interior_model,interior_gradient,1.0);
  assert(interior.lambda==0.0&&!interior.touches_boundary);check_close(interior.norm,1e-10,1e-20);
  const double retry_radius=covariance_trust_region_reduced_radius(1.0,interior.norm);
  assert(retry_radius<interior.norm);
  const CovarianceTrustRegionStep resolved=covariance_trust_region_subproblem(interior_model,interior_gradient,retry_radius);
  assert(resolved.lambda>0.0&&resolved.touches_boundary&&resolved.norm<=retry_radius);

  const std::vector<double> rounded_theta={1.0},rounded_z={10000.0},rounded_scale={1e-4},rounded_direction={1.9e-12};
  std::vector<double> rounded_next_z,rounded_step,rounded_next_theta;
  assert(covariance_build_scaled_trial(rounded_theta,rounded_z,rounded_scale,rounded_direction,1.0,
    rounded_next_z,rounded_step,rounded_next_theta));
  assert(rounded_step[0]==(rounded_next_theta[0]-rounded_theta[0])/rounded_scale[0]);
  constexpr double rounded_radius=1.9e-12;
  assert(rounded_step[0]>rounded_radius&&rounded_next_z[0]==rounded_next_theta[0]/rounded_scale[0]);
  assert(covariance_trust_region_reduced_radius(rounded_radius,rounded_step[0])<rounded_radius);

  auto objective_evaluation=[](const std::vector<double>& theta,const std::vector<double>& diagonal,
      const std::vector<double>& shifted_diagonal){CovarianceObjectiveEvaluation value;value.feasible=true;
    value.cached_theta=theta;value.cached_epsilon=1e-6;value.cholesky_diagonal=diagonal;
    value.shifted_cholesky_diagonal=shifted_diagonal;return value;};
  const double next_one=std::nextafter(1.0,2.0);
  CovarianceObjectiveEvaluation objective_current=objective_evaluation({0.0},{1e300,1.0},{1e300,1.0});
  CovarianceObjectiveEvaluation objective_candidate=objective_evaluation({0.0},{1e300,next_one},{1e300,1.0});
  objective_current.logdet=2.0*std::log(1e300);objective_candidate.logdet=objective_current.logdet;
  objective_current.objective=objective_candidate.objective=-1e300;
  const CovarianceObjectiveChange small_change=covariance_trust_region_objective_change(
    objective_current,objective_candidate,{0.0},{0.0},{0.0},{0.0},0.0,0.0);
  assert(objective_current.logdet==objective_candidate.logdet&&small_change.value<0.0);
  assert(covariance_objective_change_resolved(small_change.value,small_change.resolution));
  check_close(covariance_paired_log_ratio(1e300,2e300),std::log(2.0),1e-15);
  check_close(covariance_paired_log_ratio(1e-300,1e300),std::log(1e300)-std::log(1e-300),1e-12);
  check_close(covariance_paired_log_ratio(1e300,1e-300),std::log(1e-300)-std::log(1e300),1e-12);
  const double subnormal_candidate=std::sqrt(1e-323),subnormal_ratio=subnormal_candidate/1e154;
  assert(subnormal_ratio>0.0&&subnormal_ratio<std::numeric_limits<double>::min());
  const double subnormal_oracle=-726.4715666914936;
  check_close(covariance_paired_log_ratio(1e154,subnormal_candidate),subnormal_oracle,1e-13);
  CovarianceObjectiveEvaluation subnormal_current=objective_evaluation({0.0},{1e154},{1.0});
  CovarianceObjectiveEvaluation subnormal_next=objective_evaluation({0.0},{subnormal_candidate},{1.0});
  const CovarianceObjectiveChange subnormal_change=covariance_trust_region_objective_change(
    subnormal_current,subnormal_next,{0.0},{0.0},{0.0},{0.0},0.0,0.0);
  assert(std::abs(subnormal_change.value+subnormal_oracle)<=subnormal_change.resolution);
  assert(covariance_objective_change_resolved(subnormal_change.value,subnormal_change.resolution));
  const CovarianceTrustRegionTrialDecision subnormal_uphill=covariance_trust_region_trial_decision(
    1.0,std::numeric_limits<double>::epsilon(),-subnormal_change.value,subnormal_change.resolution);
  assert(subnormal_uphill.disposition==CovarianceTrustRegionTrialDisposition::PoorRatio);
  auto rejects_stale_difference=[&](const CovarianceObjectiveEvaluation& candidate){
    try{(void)covariance_trust_region_objective_change(objective_current,candidate,{0.0},{0.0},{0.0},{0.0},0.0,0.0);}
    catch(const std::invalid_argument&){return true;}return false;};
  CovarianceObjectiveEvaluation stale_theta=objective_candidate;stale_theta.cached_theta={1.0};
  CovarianceObjectiveEvaluation stale_epsilon=objective_candidate;stale_epsilon.cached_epsilon=2e-6;
  CovarianceObjectiveEvaluation missing_diagonal=objective_candidate;missing_diagonal.cholesky_diagonal.clear();
  assert(rejects_stale_difference(stale_theta)&&rejects_stale_difference(stale_epsilon)&&rejects_stale_difference(missing_diagonal));

  constexpr std::size_t background_dimension=17685;constexpr double epsilon=1e-6;
  const double p=(1.0-4.54e-6)-1.0,a=0.55+1.0/(1.0-epsilon)+2e-6;
  const double lambda=std::exp(100000.0/static_cast<double>(background_dimension-1));
  const double d_background=std::sqrt(lambda),shifted_background=std::sqrt(lambda-epsilon);
  std::vector<double> d0(background_dimension,d_background),d1=d0,s0(background_dimension,shifted_background),s1=s0;
  d0.back()=1.0;d1.back()=std::sqrt(1.0+p);s0.back()=std::sqrt(1.0-epsilon);s1.back()=std::sqrt(1.0-epsilon+p);
  CovarianceObjectiveEvaluation cancellation_current=objective_evaluation({0.0},d0,s0);
  CovarianceObjectiveEvaluation cancellation_candidate=objective_evaluation({p},d1,s1);
  cancellation_current.logdet=100000.0;
  cancellation_candidate.logdet=cancellation_current.logdet+std::log1p(p);
  cancellation_current.shifted_logdet=100000.0+(background_dimension-1)*std::log1p(-epsilon/lambda)+std::log1p(-epsilon);
  cancellation_candidate.shifted_logdet=cancellation_current.shifted_logdet+std::log1p(p/(1.0-epsilon));
  cancellation_current.objective=cancellation_candidate.objective=-100000.0;
  const CovarianceObjectiveChange cancellation_change=covariance_trust_region_objective_change(
    cancellation_current,cancellation_candidate,{0.0},{p},{2.0*a},{0.0},0.1,1.0);
  const double naive_change=a*p-0.55*(cancellation_candidate.logdet-cancellation_current.logdet)-
    (cancellation_candidate.shifted_logdet-cancellation_current.shifted_logdet);
  const double analytic_change=a*p-0.55*std::log1p(p)-std::log1p(p/(1.0-epsilon));
  assert(naive_change<0.0&&analytic_change>0.0&&cancellation_change.value>0.0);
  check_close(cancellation_change.value,analytic_change,2e-15);
  assert(covariance_objective_change_resolved(cancellation_change.value,cancellation_change.resolution));

  CovarianceTrustRegionModel toy_model;toy_model.positive_definite=true;toy_model.status="SPD_VERIFIED";
  toy_model.eigenvalues={1.0};toy_model.eigenvectors={1.0};const std::vector<double> toy_gradient={1.0};
  const CovarianceTrustRegionStep full_step=covariance_trust_region_subproblem(toy_model,toy_gradient,1.0);
  const double full_x=full_step.step[0],full_actual=1.0-(1.0+full_x+full_x*full_x);
  const CovarianceTrustRegionTrialDecision full_decision=covariance_trust_region_trial_decision(
    full_step.predicted_reduction,covariance_trust_region_prediction_resolution(toy_model,toy_gradient,full_step.step),
    full_actual,covariance_local_difference_resolution(std::abs(full_x)+full_x*full_x,2));
  assert(full_x==-1.0&&full_step.predicted_reduction==0.5&&full_actual==0.0);
  assert(full_decision.disposition==CovarianceTrustRegionTrialDisposition::UnresolvedActual);
  const double unresolved_actual_retry_radius=covariance_trust_region_reduced_radius(1.0,full_step.norm);
  assert(unresolved_actual_retry_radius==0.5);
  const CovarianceTrustRegionStep retry_step=covariance_trust_region_subproblem(toy_model,toy_gradient,unresolved_actual_retry_radius);
  const double retry_x=retry_step.step[0],retry_actual=1.0-(1.0+retry_x+retry_x*retry_x);
  const CovarianceTrustRegionTrialDecision retry_decision=covariance_trust_region_trial_decision(
    retry_step.predicted_reduction,covariance_trust_region_prediction_resolution(toy_model,toy_gradient,retry_step.step),
    retry_actual,covariance_local_difference_resolution(std::abs(retry_x)+retry_x*retry_x,2));
  assert(retry_x==-0.5&&retry_actual==0.25&&retry_decision.disposition==CovarianceTrustRegionTrialDisposition::Accept);
  check_close(retry_decision.ratio,2.0/3.0,1e-15);
  const CovarianceTrustRegionTrialDecision stagnant=covariance_trust_region_trial_decision(1e-300,1e-299,0.0,1e-320);
  assert(stagnant.disposition==CovarianceTrustRegionTrialDisposition::ModelStagnation);

  CovarianceTrustRegionModel model;model.positive_definite=true;model.status="SPD_VERIFIED";
  model.eigenvalues={4.0,1.0};model.eigenvectors={1.0,0.0,0.0,1.0};
  const std::vector<double> gradient={1.0,1.0};
  const CovarianceTrustRegionStep half=covariance_trust_region_subproblem(model,gradient,0.5);
  assert(half.lambda>0.0&&half.touches_boundary);check_close(half.norm,0.5,1e-12);
  check_close(half.predicted_reduction,covariance_trust_region_predicted_reduction(model,gradient,half.step),1e-15);
  const CovarianceTrustRegionStep quarter=covariance_trust_region_subproblem(model,gradient,0.25);
  check_close(quarter.norm,0.25,1e-12);assert(quarter.lambda>half.lambda);
  assert(std::abs(2.0*quarter.step[0]-half.step[0])+std::abs(2.0*quarter.step[1]-half.step[1])>1e-3);
  const CovarianceTrustRegionStep unconstrained=covariance_trust_region_subproblem(model,gradient,10.0);
  assert(unconstrained.lambda==0.0);check_close(unconstrained.step[0],-4.0,1e-15);check_close(unconstrained.step[1],-1.0,1e-15);
  check_close(unconstrained.predicted_reduction,2.5,1e-15);

  CovarianceObjectiveEvaluation current;current.feasible=true;current.cached_theta={1.0};current.cached_epsilon=0.01;
  current.data=2.0;current.prior=3.0;current.shifted_logdet=0.5;current.trace={5.0};current.shifted_trace={4.0};
  CovarianceBaseline baseline;baseline.theta={1.0};baseline.trace={2.0};baseline.shifted_logdet=0.3;
  covariance_recompose_objective_for_tau(current,{1.0},baseline,{10.0},0.5,0.01,0.1);
  check_close(current.barrier,-0.02,1e-15);check_close(current.objective,4.98,1e-15);check_close(current.gradient[0],1.35,1e-15);
  covariance_recompose_objective_for_tau(current,{1.0},baseline,{10.0},0.5,0.01,0.2);
  check_close(current.gradient[0],0.95,1e-15);
  bool stale_cache_rejected=false;try{covariance_recompose_objective_for_tau(current,{1.0+1e-12},baseline,{10.0},0.5,0.01,0.2);}
  catch(const std::runtime_error&){stale_cache_rejected=true;}assert(stale_cache_rejected);

  current.gradient={1.0};current.feasible=true;
  assert(!covariance_barrier_stage_pass(current,5e-3,1.0,1e-8,false));
  assert(covariance_barrier_stage_pass(current,5e-3,1.0,1e-8,true));
  assert(covariance_barrier_stage_pass(current,5e-7,1e-12,1e-8,true));
  current.feasible=false;assert(!covariance_barrier_stage_pass(current,0.0,1.0,1e-8,true));
}

void test_covariance_direct_hessian_preconditioner(cusolverDnHandle_t solver)
{
  const CovarianceTrustRegionModel diagonal=covariance_direct_hessian_trust_region_model(solver,{2.0,0.0,0.0,8.0},2);
  assert(diagonal.positive_definite&&diagonal.clipped_eigenvalues==0);
  const CovarianceTrustRegionStep exact=covariance_trust_region_subproblem(diagonal,{2.0,8.0},10.0);
  assert(exact.lambda==0.0);check_close(exact.step[0],-1.0,1e-14);check_close(exact.step[1],-1.0,1e-14);

  constexpr double ridge=0.01,dominant=1e18;
  const std::vector<double> near_rank_one={dominant+ridge,dominant,dominant,dominant+ridge};
  const CovarianceTrustRegionModel regularized=covariance_direct_hessian_trust_region_model(solver,near_rank_one,2);
  assert(regularized.positive_definite&&regularized.clipped_eigenvalues==1&&regularized.spectral_floor>0.0);
  assert(regularized.status=="MODEL_REGULARIZED_SOFT_EIGENVALUES");
  const CovarianceTrustRegionStep step=covariance_trust_region_subproblem(regularized,{1.0,-1.0},1.0);
  assert(std::isfinite(step.norm)&&step.norm<=1.0&&std::all_of(step.step.begin(),step.step.end(),[](double value){return std::isfinite(value);}));

  CovarianceTrustRegionModel cancellation;cancellation.positive_definite=true;cancellation.eigenvalues={1.0,1.0};
  cancellation.eigenvectors={1.0,0.0,0.0,1.0};cancellation.coordinate_transform={1.0,1.0};
  cancellation.hessian_matrix={1e16,1e16,1e16,1e16};
  const double cancellation_resolution=covariance_trust_region_prediction_resolution(cancellation,{0.0,0.0},{1.0,-1.0});
  assert(cancellation_resolution>100.0);
}

void test_covariance_centered_ridge_objective_and_tau_cache()
{
  constexpr double epsilon=0.01,ridge=0.7;DeviceBaseline device;initialize_two_atom_covariance_baseline(device,true);
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceCovarianceSystem system(device,graph,sqrt_mass,masses);const std::vector<double> statistics={1,0,0,1,0,1};
  std::uint64_t evaluations=0;const CovarianceBaseline baseline=build_graph_covariance_baseline(system,statistics,epsilon,evaluations);
  const std::vector<double> scale=system.coordinate_scales(baseline.theta,epsilon);
  CovarianceObjectiveEvaluation at_center=evaluate_covariance_fit_objective(system,baseline.theta,baseline,statistics,0.0,
    epsilon,1.0,evaluations,true,false,true,nullptr,nullptr,&scale,ridge);
  assert(at_center.feasible&&at_center.ridge==0.0&&std::all_of(at_center.cached_ridge_coordinates.begin(),
    at_center.cached_ridge_coordinates.end(),[](double value){return value==0.0;}));
  assert(std::all_of(at_center.gradient_ridge.begin(),at_center.gradient_ridge.end(),[](double value){return value==0.0;}));
  const std::vector<double> direction=graph_laplacian_direction(graph,system.p);std::vector<double> trial=baseline.theta;
  for(std::size_t i=0;i<trial.size();++i)trial[i]+=1e-3*scale[i]*direction[i];
  CovarianceObjectiveEvaluation value=evaluate_covariance_fit_objective(system,trial,baseline,statistics,0.0,
    epsilon,1.0,evaluations,true,false,true,nullptr,nullptr,&scale,ridge);
  assert(value.feasible);double z2=0.0;
  for(std::size_t i=0;i<trial.size();++i){const double z=(trial[i]-baseline.theta[i])/scale[i];z2+=z*z;
    check_close(value.gradient_ridge[i],ridge*(trial[i]-baseline.theta[i])/(scale[i]*scale[i]),1e-12);}
  check_close(value.ridge,0.5*ridge*z2,1e-12);
  const double ridge_before=value.ridge;covariance_recompose_objective_for_tau(value,trial,baseline,statistics,0.0,epsilon,0.25,scale,ridge);
  check_close(value.ridge,ridge_before,1e-14);check_close(value.objective,value.data+value.prior+value.ridge+value.barrier,1e-13);
  const CovarianceObjectiveEvaluation fresh=evaluate_covariance_fit_objective(system,trial,baseline,statistics,0.0,
    epsilon,0.25,evaluations,true,false,true,nullptr,nullptr,&scale,ridge);
  check_close(value.objective,fresh.objective,1e-12);check_close(value.ridge,fresh.ridge,1e-14);
  for(std::size_t i=0;i<trial.size();++i)check_close(value.gradient[i],fresh.gradient[i],1e-12);
}

void test_covariance_original_residual_gates_convergence()
{
  constexpr double epsilon=0.01;const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceBaseline device;initialize_two_atom_covariance_baseline(device,true);DeviceCovarianceSystem system(device,graph,sqrt_mass,masses);
  CovarianceBaseline baseline;baseline.theta.assign(6,0.0);baseline.epsilon=epsilon;baseline.mu=0.0;
  std::vector<double> shifted_trace;assert(system.evaluate_factors(baseline.theta,epsilon,baseline.logdet,baseline.shifted_logdet,
    &baseline.trace,&shifted_trace));
  const std::vector<double> scale=system.coordinate_scales(baseline.theta,epsilon);std::vector<double> statistics(6);
  for(std::size_t i=0;i<statistics.size();++i)statistics[i]=baseline.trace[i]+2.0*shifted_trace[i];
  std::ostringstream trace;const CovarianceFitResult fit=solve_covariance_reference(system,statistics,epsilon,0.0,1,&trace,
    false,false,false,false,{}, {},0.7,0,&baseline,&scale,&baseline.theta,1.0,true,nullptr,nullptr);
  assert(fit.converged&&fit.status=="COVARIANCE_FIXED_TAU_CONVERGED");
  assert(fit.original_residual_initial_pass&&fit.original_residual_final_pass);
  assert(fit.original_residual_initial_status=="PASS"&&fit.original_residual_final_status=="PASS");
  assert(fit.original_residual_gradient_attempts==2&&fit.full_gradient_evaluations==2);
  assert(fit.run_trace_residual_summary.original_any_checked&&fit.run_trace_residual_summary.original_batches_checked>0);
  assert(trace.str().find("trace_residual_original_any_checked=1")!=std::string::npos);

  const std::string path="rpmd_ja_native_original_residual_state_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=epsilon;options.fd_step=1e-4;options.sample_interval=1;
  SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};
  write_covariance_state(path,fit,0.0,"test.samples","test.qraw",options,header,identity,graph,2,2);
  std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
  assert(fields.at("original_residual_initial_status")=="PASS"&&fields.at("original_residual_final_status")=="PASS");
  assert(fields.at("original_residual_initial_pass")=="1"&&fields.at("original_residual_final_pass")=="1");
  assert(fields.at("original_residual_gradient_attempts")=="2");
  assert(fields.at("run_trace_residual_original_any_checked")=="1");

  rpmd_ja_feasibility::Budget limited_budget;rpmd_ja_feasibility::Limits limited_limits;limited_limits.max_exact_gradients=1;
  const double budget_start=covariance_budget_now_seconds();
  assert(rpmd_ja_feasibility::begin(limited_budget,limited_limits,budget_start)==rpmd_ja_feasibility::BudgetResult::OK);
  assert(rpmd_ja_feasibility::enter_phase(limited_budget,limited_limits,rpmd_ja_feasibility::Phase::A0,budget_start)==
    rpmd_ja_feasibility::BudgetResult::OK);
  const CovarianceFitResult budget_limited=solve_covariance_reference(system,statistics,epsilon,0.0,1,nullptr,
    false,false,false,false,{}, {},0.7,0,&baseline,&scale,&baseline.theta,1.0,true,&limited_budget,&limited_limits);
  assert(!budget_limited.converged&&budget_limited.status=="FEASIBILITY_BUDGET_GRADIENT_LIMIT");
  assert(budget_limited.original_residual_initial_pass&&!budget_limited.original_residual_final_pass);
  assert(budget_limited.original_residual_final_status=="BUDGET_LIMIT");
  assert(budget_limited.original_residual_gradient_attempts==1&&limited_budget.exact_gradients==1);
}

void test_feasibility_same_file_path_aliases()
{
  const std::string path="rpmd_ja_same_file_identity_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".tmp";
  const std::string copy=path+".copy";RemoveTestFile clean_path{path},clean_copy{copy};
  {std::ofstream out(path,std::ios::binary);assert(out.good());out<<"same file identity";assert(out.good());}
  {std::ifstream in(path,std::ios::binary);std::ofstream out(copy,std::ios::binary);assert(in.good()&&out.good());out<<in.rdbuf();assert(out.good());}
  assert(feasibility_same_file_path(path,"./"+path));
  assert(!feasibility_same_file_path(path,copy));
  const std::string utf8_path="rpmd_ja_\xE4\xB8\xAD\xE6\x96\x87_file_identity_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".tmp";
  RemoveTestFile clean_utf8{utf8_path};
  {rpmd_ja_file_io::ofstream out(utf8_path,std::ios::binary);assert(out.good());out<<"UTF-8 path identity";assert(out.good());}
  assert(feasibility_same_file_path(utf8_path,"./"+utf8_path));
}

void test_covariance_prior_calibration_ignores_m1_tail()
{
  const double nan=std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> statistics={1,0,0,1,0,1,-7,nan},baseline={1,0,0,1,0,1,nan,-9};
  const std::vector<std::vector<double>> blocks={{1,0,0,1,0,1,-5,nan}};
  const std::vector<std::uint64_t> frames={3};
  std::ostringstream contaminated,clean;
  write_covariance_prior_calibration(contaminated,statistics,baseline,blocks,frames,6,0.0);
  write_covariance_prior_calibration(clean,{1,0,0,1,0,1,0,0},{1,0,0,1,0,1,0,0},{{1,0,0,1,0,1,0,0}},frames,6,0.0);
  assert(contaminated.str()==clean.str());
  assert(contaminated.str().find("base_parameter_count=6 total_parameter_count=8 groups=1")!=std::string::npos);
  assert(contaminated.str().find("parameter_count=2 status=NOT_APPLICABLE")!=std::string::npos);

  bool m0_block_mismatch_rejected=false;
  try{std::ostringstream output;write_covariance_prior_calibration(output,{1,0,0,1,0,1},{1,0,0,1,0,1},{{1,0,0,1,0}},frames,6,0.0);}
  catch(const std::invalid_argument&){m0_block_mismatch_rejected=true;}
  assert(m0_block_mismatch_rejected);
}

void test_budget_observation_classifies_hard_clock_errors()
{
  using rpmd_ja_feasibility::BudgetResult;
  covariance_require_expected_budget_observation(BudgetResult::OK,"test");
  covariance_require_expected_budget_observation(BudgetResult::FAMILY_WALL_LIMIT,"test");
  for(const BudgetResult result:{BudgetResult::INVALID_CLOCK,BudgetResult::CLOCK_BACKWARDS,
      BudgetResult::NOT_STARTED,BudgetResult::INVALID_LIMITS}){
    bool rejected=false;try{covariance_require_expected_budget_observation(result,"test");}
    catch(const std::runtime_error&){rejected=true;}assert(rejected);
  }
}

void test_covariance_solve_write_diagnose_replay_without_ridge()
{
  const std::string spool_path=write_covariance_replay_spool(".samples"),raw_path=spool_path+".qraw";
  const std::string state_path=spool_path+".covariance_state",report_path=spool_path+".diagnostic";
  RemoveTestFile clean_spool{spool_path},clean_raw{raw_path},clean_state{state_path},clean_report{report_path};
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;Force force;
  std::ifstream spool(spool_path,std::ios::binary);assert(spool.good());
  const SampleHeader header=read_header(spool,0,atom,box,0.0,true,true);std::vector<double> r0(6);
  const double interval=read_training_r0(spool,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  const int n=header.n,d=3*n;Graph graph=make_graph(r0,header.types,box,5.0);
  double raw_cell[18];std::copy(box.cpu_h,box.cpu_h+18,raw_cell);const int pbc[3]={1,1,1};
  const std::int32_t raw_types[2]={0,1};const double raw_masses[2]={1.0,2.0};
  const char qraw_magic[8]={'G','P','J','Q','R','A','W','\0'};
  const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::uint32_t raw_version=3;const std::uint64_t model=0,config=0;const std::int32_t charge=1,kspace=1;
  const double fd_step=1e-4,mesh=0.5,energy=0.0;std::vector<double> site_energy(n,0.0),raw_force(d,0.0),virial(9*n,0.0);
  std::vector<double> v(static_cast<std::size_t>(d)*n,0.0),c(static_cast<std::size_t>(3)*d*d,0.0),k(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)k[static_cast<std::size_t>(i)*d+i]=1.0;
  double raw_stats[18]={};raw_stats[16]=fd_step;raw_stats[17]=3.0;
  {std::ofstream raw(raw_path,std::ios::binary);assert(raw.good());raw.write(qraw_magic,sizeof(qraw_magic));
    write(raw,raw_version);write(raw,native_endian);write(raw,std::int32_t(n));write(raw,std::int32_t(d));
    write(raw,header.temperature);write(raw,fd_step);write(raw,model);write(raw,config);write(raw,charge);write(raw,kspace);write(raw,mesh);
    raw.write(layout,sizeof(layout));raw.write(reinterpret_cast<const char*>(raw_cell),sizeof(raw_cell));raw.write(reinterpret_cast<const char*>(pbc),sizeof(pbc));
    raw.write(reinterpret_cast<const char*>(raw_types),sizeof(raw_types));raw.write(reinterpret_cast<const char*>(raw_masses),sizeof(raw_masses));
    raw.write(reinterpret_cast<const char*>(r0.data()),static_cast<std::streamsize>(r0.size()*sizeof(double)));
    write(raw,energy);raw.write(reinterpret_cast<const char*>(site_energy.data()),static_cast<std::streamsize>(site_energy.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(raw_force.data()),static_cast<std::streamsize>(raw_force.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(virial.data()),static_cast<std::streamsize>(virial.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(c.data()),static_cast<std::streamsize>(c.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(k.data()),static_cast<std::streamsize>(k.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(c.data()),static_cast<std::streamsize>(c.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(raw_stats),sizeof(raw_stats));assert(raw.good());}
  RpmdJANativeFitOptions options;options.temperature=header.temperature;options.cutoff=5.0;options.epsilon=0.01;
  options.fd_step=fd_step;options.sample_interval=static_cast<int>(interval);options.covariance_ridge_lambda=0.0;
  covariance_fit_test_allow_qraw_fixture_identity=true;
  struct ResetFixtureIdentity{~ResetFixtureIdentity(){covariance_fit_test_allow_qraw_fixture_identity=false;}} reset_fixture_identity;
  std::ifstream raw(raw_path,std::ios::binary);const QrawIdentity identity=validate_qraw_identity(raw,options,header,r0,box,force);
  std::vector<double> sqrt_mass={1.0,std::sqrt(2.0),1.0,std::sqrt(2.0),1.0,std::sqrt(2.0)};
  DeviceBaseline baseline_device;baseline_device.initialize(raw,identity.k_offset,d,n,header.masses,sqrt_mass);raw.close();
  DeviceCovarianceSystem system(baseline_device,graph,sqrt_mass,header.masses);
  std::vector<double> edge_statistics;const std::vector<double> statistics=collect_covariance_fit_statistics(
    spool,header,header.frame_count,2,graph,r0,header.temperature,&edge_statistics);
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,options.epsilon,0.1,1,nullptr);
  write_covariance_state(state_path,fit,0.1,spool_path,raw_path,options,header,identity,graph,header.frame_count,2);
  {std::ifstream saved(state_path);const auto fields=read_covariance_state_fields(saved);
    assert(fields.at("target_objective")=="COVARIANCE_SHRINKAGE"&&fields.at("ridge_coordinates")=="0");}
  diagnose_rpmd_ja_native_covariance(state_path,report_path,options.cutoff,options.epsilon,fit.barrier_tau,
    options.fd_step,false,atom,box,force);
  const std::string report=read_test_file(report_path);
  assert(report.find("state_replay_status PASS")!=std::string::npos);
  const std::string wrong_state_path=state_path+".wrong",wrong_report_path=report_path+".wrong";
  RemoveTestFile clean_wrong_state{wrong_state_path},clean_wrong_report{wrong_report_path};
  std::string wrong_state=read_test_file(state_path);const std::string objective="target_objective COVARIANCE_SHRINKAGE";
  const std::size_t objective_at=wrong_state.find(objective);assert(objective_at!=std::string::npos);
  wrong_state.replace(objective_at,objective.size(),"target_objective FROZEN_CENTER_RIDGE");
  {std::ofstream out(wrong_state_path);out<<wrong_state;assert(out.good());}
  std::string mismatch_error;try{diagnose_rpmd_ja_native_covariance(wrong_state_path,wrong_report_path,options.cutoff,
      options.epsilon,fit.barrier_tau,options.fd_step,false,atom,box,force);}
    catch(const std::runtime_error& error){mismatch_error=error.what();}
  assert(mismatch_error.find("target_objective disagrees with rho/ridge_lambda")!=std::string::npos);

  assert(fit.converged);
  const std::string package_path=state_path+".a0_reuse.package";RemoveTestFile clean_package{package_path};
  const std::string manifest_bytes="RPJAFEAS9\ntraining_sha256 \""+std::string(64,'a')+"\"\n";
  const std::map<std::string,std::string> manifest_fields={{"training_sha256",std::string(64,'a')}};
  const std::string source_machine(64,'e'),source_job="native-a0-export-test",state_bytes=read_test_file(state_path);
  const auto record=rpmd_ja_a0_reuse::make_export_record(manifest_bytes,manifest_fields,state_bytes,
    source_machine,source_job,fit.optimizer_seconds,fit.optimizer_seconds,true,true,false);
  rpmd_ja_a0_reuse::write_package_atomic(package_path,manifest_bytes,state_bytes,record);
  const auto exported=rpmd_ja_a0_reuse::parse_package(read_test_file(package_path));
  verify_imported_a0_parameter_map(exported.state_bytes,graph);
  auto restore_imported=[&](const std::string& bytes){std::istringstream state(bytes);
    return restore_covariance_fit_state(state,"native exported A0 package",statistics,
      std::vector<std::vector<double>>{},std::vector<std::uint64_t>{},options.epsilon,0.1,0.0,"M0",spool_path,raw_path);};
  auto imported=restore_imported(exported.state_bytes);
  rebuild_restored_a0_baseline(imported,baseline_device,graph,sqrt_mass,header.masses,nullptr,options.epsilon,nullptr,nullptr,nullptr,true);
  verify_imported_a0_fit(imported,baseline_device,graph,sqrt_mass,header.masses,statistics,0.1,options.epsilon,
    nullptr,nullptr,nullptr,nullptr);

  auto replace_vector_first=[](std::string bytes,const std::string& key,const double delta){
    const std::string prefix=key+" ";const std::size_t begin=bytes.find(prefix);assert(begin!=std::string::npos);
    const std::size_t end=bytes.find('\n',begin);assert(end!=std::string::npos);
    std::istringstream row(bytes.substr(begin+prefix.size(),end-begin-prefix.size()));std::size_t count=0;
    assert(static_cast<bool>(row>>count)&&count>0);std::vector<double> values(count);
    for(double& value:values)assert(static_cast<bool>(row>>value));values[0]+=delta;
    std::ostringstream replacement;replacement<<key<<' '<<count<<std::setprecision(17);
    for(const double value:values)replacement<<' '<<value;
    bytes.replace(begin,end-begin,replacement.str());return bytes;};
  auto replace_scalar=[](std::string bytes,const std::string& key,const std::string& value){
    const std::string prefix="\n"+key+" ";const std::size_t begin=bytes.find(prefix);assert(begin!=std::string::npos);
    const std::size_t end=bytes.find('\n',begin+1);assert(end!=std::string::npos);
    bytes.replace(begin+1,end-begin-1,key+" "+value);return bytes;};
  auto reject_rewrapped=[&](const std::string& bad_state,const char* expected){
    const auto bad_record=rpmd_ja_a0_reuse::make_export_record(manifest_bytes,manifest_fields,bad_state,source_machine,
      source_job,fit.optimizer_seconds,fit.optimizer_seconds,true,true,false);
    const auto bad_package=rpmd_ja_a0_reuse::parse_package(rpmd_ja_a0_reuse::serialize_package(manifest_bytes,bad_state,bad_record));
    bool rejected=false;try{auto bad=restore_imported(bad_package.state_bytes);
      rebuild_restored_a0_baseline(bad,baseline_device,graph,sqrt_mass,header.masses,nullptr,options.epsilon,nullptr,nullptr,nullptr,true);
      verify_imported_a0_fit(bad,baseline_device,graph,sqrt_mass,header.masses,statistics,0.1,options.epsilon,
        nullptr,nullptr,nullptr,nullptr);}
    catch(const std::runtime_error& error){rejected=std::string(error.what()).find(expected)!=std::string::npos;}
    assert(rejected);};
  reject_rewrapped(replace_vector_first(exported.state_bytes,"coordinate_scale",0.25),"coordinate scale");
  std::string bad_parameter=replace_vector_first(exported.state_bytes,"theta",0.25);
  bad_parameter=replace_vector_first(std::move(bad_parameter),"last_valid_theta",0.25);
  reject_rewrapped(bad_parameter,"do not replay");
  reject_rewrapped(replace_scalar(exported.state_bytes,"objective_data","999"),"objective components");
  reject_rewrapped(replace_scalar(exported.state_bytes,"target_objective","FROZEN_CENTER_RIDGE"),"target_objective");
}

void test_m1_covariance_calibration_uses_only_base_prefix_in_replay()
{
  const std::string spool_path=write_covariance_replay_spool(".m1.samples",true),raw_path=spool_path+".qraw";
  const std::string state_path=spool_path+".covariance_state",report_path=spool_path+".diagnostic";
  RemoveTestFile clean_spool{spool_path},clean_raw{raw_path},clean_state{state_path},clean_report{report_path};
  Atom atom;atom.number_of_atoms=3;atom.cpu_mass={16.0,1.0,16.0};atom.cpu_type={0,1,0};
  atom.cpu_atom_symbol={"O","H","O"};atom.number_of_beads=0;atom.position_per_atom.resize(9);
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  assert(box.pbc_x==1&&box.pbc_y==1&&box.pbc_z==1);Force force;std::ifstream spool(spool_path,std::ios::binary);assert(spool.good());
  const SampleHeader header=read_header(spool,0,atom,box,0.0,true,true);std::vector<double> r0(9);
  const double interval=read_training_r0(spool,header,header.frame_count,std::numeric_limits<double>::quiet_NaN(),r0);
  const int n=header.n,d=3*n;Graph graph=make_graph(r0,header.types,box,5.2);
  const auto& metadata=build_m1_local_basis(graph,r0,header.types,atom.cpu_atom_symbol,header.masses,box,5.2,1);
  assert(metadata.accepted_bridge_count==1&&graph.parameter_count%6!=0&&
    graph.base_parameter_count%6==0&&graph.base_parameter_count<graph.parameter_count);
  double raw_cell[18];std::copy(box.cpu_h,box.cpu_h+18,raw_cell);const int pbc[3]={1,1,1};
  std::vector<std::int32_t> raw_types(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i)raw_types[i]=header.types[i];
  const std::vector<double> raw_masses=header.masses;const char qraw_magic[8]={'G','P','J','Q','R','A','W','\0'};
  const char layout[]="xyz_soa;derivative_input_rows_output_columns";const std::uint32_t raw_version=3;
  const std::uint64_t model=0,config=0;const std::int32_t charge=1,kspace=1;
  const double fd_step=1e-4,mesh=0.5,energy=0.0;
  std::vector<double> site_energy(static_cast<std::size_t>(n),0.0),raw_force(static_cast<std::size_t>(d),0.0);
  std::vector<double> virial(static_cast<std::size_t>(9)*n,0.0),v(static_cast<std::size_t>(d)*n,0.0);
  std::vector<double> c(static_cast<std::size_t>(3)*d*d,0.0),k(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)k[static_cast<std::size_t>(i)*d+i]=1.0;
  double raw_stats[18]={};raw_stats[16]=fd_step;raw_stats[17]=3.0;
  {std::ofstream raw(raw_path,std::ios::binary);assert(raw.good());raw.write(qraw_magic,sizeof(qraw_magic));
    write(raw,raw_version);write(raw,native_endian);write(raw,std::int32_t(n));write(raw,std::int32_t(d));
    write(raw,header.temperature);write(raw,fd_step);write(raw,model);write(raw,config);write(raw,charge);write(raw,kspace);write(raw,mesh);
    raw.write(layout,sizeof(layout));raw.write(reinterpret_cast<const char*>(raw_cell),sizeof(raw_cell));raw.write(reinterpret_cast<const char*>(pbc),sizeof(pbc));
    raw.write(reinterpret_cast<const char*>(raw_types.data()),static_cast<std::streamsize>(raw_types.size()*sizeof(raw_types[0])));
    raw.write(reinterpret_cast<const char*>(raw_masses.data()),static_cast<std::streamsize>(raw_masses.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(r0.data()),static_cast<std::streamsize>(r0.size()*sizeof(double)));
    write(raw,energy);raw.write(reinterpret_cast<const char*>(site_energy.data()),static_cast<std::streamsize>(site_energy.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(raw_force.data()),static_cast<std::streamsize>(raw_force.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(virial.data()),static_cast<std::streamsize>(virial.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(c.data()),static_cast<std::streamsize>(c.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(k.data()),static_cast<std::streamsize>(k.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(c.data()),static_cast<std::streamsize>(c.size()*sizeof(double)));
    raw.write(reinterpret_cast<const char*>(raw_stats),sizeof(raw_stats));assert(raw.good());}
  RpmdJANativeFitOptions options;options.temperature=header.temperature;options.cutoff=5.2;options.epsilon=0.01;
  options.fd_step=fd_step;options.sample_interval=static_cast<int>(interval);options.covariance_ridge_lambda=0.3;
  covariance_fit_test_allow_qraw_fixture_identity=true;
  struct ResetFixtureIdentity{~ResetFixtureIdentity(){covariance_fit_test_allow_qraw_fixture_identity=false;}} reset_fixture_identity;
  std::ifstream raw(raw_path,std::ios::binary);const QrawIdentity identity=validate_qraw_identity(raw,options,header,r0,box,force);
  std::vector<double> sqrt_mass(static_cast<std::size_t>(d));
  for(int axis=0;axis<3;++axis)for(int atom_index=0;atom_index<n;++atom_index)
    sqrt_mass[static_cast<std::size_t>(axis)*n+atom_index]=std::sqrt(header.masses[atom_index]);
  DeviceBaseline baseline_device;baseline_device.initialize(raw,identity.k_offset,d,n,header.masses,sqrt_mass);raw.close();
  DeviceCovarianceSystem system(baseline_device,graph,sqrt_mass,header.masses);
  std::vector<double> edge_statistics;std::vector<std::vector<double>> training_block_statistics;
  std::vector<std::uint64_t> training_block_frames;
  const std::vector<double> statistics=collect_covariance_fit_statistics(spool,header,header.frame_count,2,graph,r0,
    header.temperature,&edge_statistics,&training_block_statistics,&training_block_frames);
  std::vector<double> theta_start(static_cast<std::size_t>(graph.parameter_count),0.0);
  for(int parameter=graph.base_parameter_count;parameter<graph.parameter_count;++parameter)
    theta_start[parameter]=0.01*(parameter-graph.base_parameter_count+1);
  std::ostringstream solve_trace;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,options.epsilon,0.0,1,&solve_trace,
    false,false,false,false,training_block_statistics,training_block_frames,options.covariance_ridge_lambda,0,
    nullptr,nullptr,&theta_start,1.0,true);
  const std::string added_status="COVARIANCE_PRIOR_M1_ADDED_DIRECTIONS parameter_count="+
    std::to_string(graph.parameter_count-graph.base_parameter_count)+" status=NOT_APPLICABLE";
  assert(solve_trace.str().find(added_status)!=std::string::npos);
  assert(solve_trace.str().find("selection=USER_SUPPLIED_RHO")!=std::string::npos);
  assert(solve_trace.str().find("base_parameter_count="+std::to_string(graph.base_parameter_count)+
    " total_parameter_count="+std::to_string(graph.parameter_count)+" groups="+
    std::to_string(graph.base_parameter_count/6))!=std::string::npos);
  write_covariance_state(state_path,fit,0.0,spool_path,raw_path,options,header,identity,graph,header.frame_count,2);
  {std::ifstream saved(state_path);const auto fields=read_covariance_state_fields(saved);
    assert(fields.at("parameter_family")=="M1"&&std::stoull(fields.at("parameter_count"))==graph.parameter_count);
    assert(fields.at("target_objective")=="FROZEN_CENTER_RIDGE"&&std::stod(fields.at("ridge_lambda"))==options.covariance_ridge_lambda);
    std::istringstream theta_row(fields.at("last_valid_theta"));std::size_t count=0;assert(theta_row>>count);
    std::vector<double> saved_theta(count);for(double& value:saved_theta)assert(theta_row>>value);
    assert(count==static_cast<std::size_t>(graph.parameter_count));
    assert(std::any_of(saved_theta.begin()+graph.base_parameter_count,saved_theta.end(),[](double value){return value!=0.0;}));}
  diagnose_rpmd_ja_native_covariance(state_path,report_path,options.cutoff,options.epsilon,fit.barrier_tau,
    options.fd_step,false,atom,box,force);
  const std::string report=read_test_file(report_path);
  assert(report.find("state_replay_status PASS")!=std::string::npos);
  assert(report.find(added_status)!=std::string::npos);
  assert(report.find("base_parameter_count="+std::to_string(graph.base_parameter_count)+
    " total_parameter_count="+std::to_string(graph.parameter_count)+" groups="+
    std::to_string(graph.base_parameter_count/6))!=std::string::npos);

  const std::string cg_witness=state_path+".cg",cg_report=state_path+".cg_report";
  RemoveTestFile clean_cg_witness{cg_witness},clean_cg_report{cg_report};
  CGWitness witness;witness.classification="TRUE_RESIDUAL_FAILURE";witness.probe=2;witness.iteration=3;
  witness.true_residual_computed=true;witness.finite=true;witness.relative_residual=1.0;
  witness.rhs_input.resize(static_cast<std::size_t>(d));witness.solution.resize(static_cast<std::size_t>(d));
  for(int i=0;i<d;++i){witness.rhs_input[i]=0.03*(i+1);witness.solution[i]=0.02*(i+1);}
  std::vector<double> sqrt_atom(static_cast<std::size_t>(n));
  for(int i=0;i<n;++i)sqrt_atom[i]=std::sqrt(header.masses[i]);
  witness.rhs=witness.rhs_input;project_translation(witness.rhs,sqrt_atom,n);
  project_translation(witness.solution,sqrt_atom,n);
  std::vector<double> witness_theta(static_cast<std::size_t>(graph.parameter_count),0.0);
  witness_theta.back()=0.2;
  RpmdJANativeFitOptions cg_options=options;cg_options.internal_mass_com=false;
  const CGReplayContext cg_replay{cg_options,header.frame_count,identity.model,identity.config,identity.model,
    identity.config,identity.charge,identity.kspace,identity.mesh,identity.active_mesh};
  write_cg_witness(cg_witness,witness,witness_theta,raw_path,spool_path,cg_replay,&graph);
  const CGWitnessSnapshot saved_witness=read_cg_witness(cg_witness,d,0);
  assert(saved_witness.witness_format_version==2&&saved_witness.parameter_family=="M1");
  assert(saved_witness.parameter_count==graph.parameter_count&&saved_witness.m1_minimum_support_count==1);
  const std::string cg_sidecar=saved_witness.m1_basis_sidecar;RemoveTestFile clean_cg_sidecar{cg_sidecar};
  replay_rpmd_ja_native_cg(cg_witness,cg_report,false,atom,box,force);
  const std::string cg_report_text=read_test_file(cg_report);
  assert(cg_report_text.find("parameter_family M1")!=std::string::npos);
  assert(cg_report_text.find("M1_basis_replay_status PASS")!=std::string::npos);
  assert(cg_report_text.find("fast_true_relative_residual ")!=std::string::npos&&
    cg_report_text.find("compensated_true_relative_residual ")!=std::string::npos);
  std::string basis_snapshot=read_test_file(cg_sidecar);assert(!basis_snapshot.empty());
  basis_snapshot[0]=basis_snapshot[0]=='#'?'!':'#';
  {std::ofstream out(cg_sidecar,std::ios::binary|std::ios::trunc);out.write(basis_snapshot.data(),basis_snapshot.size());assert(out.good());}
  const std::string tampered_report=cg_report+".tampered";RemoveTestFile clean_tampered_report{tampered_report};
  bool tamper_rejected=false;try{replay_rpmd_ja_native_cg(cg_witness,tampered_report,false,atom,box,force);}
  catch(const std::runtime_error& error){tamper_rejected=std::string(error.what()).find("M1 basis snapshot")!=std::string::npos;}
  std::ifstream tampered_output(tampered_report,std::ios::binary);
  assert(tamper_rejected&&!tampered_output.good());
}

void test_covariance_analytic_curvature_whitened_gram()
{
  constexpr int n=2,d=6,r=3,p=6,probes=16;const double epsilon=0.01;
  CovarianceAnalyticCurvatureCache component_cache;component_cache.valid=true;
  component_cache.gram_d={2.0,1.0,1.0,4.0};component_cache.gram_q={1.0,-0.5,-0.5,3.0};
  const CovarianceCurvatureComponents known_components=covariance_directional_curvature_components(
    component_cache,{0.6,0.8},0.2,0.5,0.3);
  check_close(known_components.data,2.544,1e-14);check_close(known_components.barrier,0.9,1e-14);
  check_close(known_components.ridge,0.3,1e-14);check_close(known_components.total,3.744,1e-14);
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  const Graph graph=make_two_atom_covariance_graph();
  std::vector<double> householder;
  {DeviceCovarianceSystem geometry(baseline,graph,sqrt_mass,masses);householder=geometry.householder_host;}
  std::vector<int> compact_full;for(int full=0;full<d;++full)if(full!=0&&full!=n&&full!=2*n)compact_full.push_back(full);
  assert(static_cast<int>(compact_full.size())==r);
  std::vector<double> q(static_cast<std::size_t>(d)*r,0.0);
  for(int column=0;column<r;++column){const int full=compact_full[column],axis=full/n;
    for(int row=0;row<d;++row)q[static_cast<std::size_t>(column)*d+row]=
      (row==full?1.0:0.0)-2.0*householder[static_cast<std::size_t>(axis)*d+row]*
      householder[static_cast<std::size_t>(axis)*d+full];}

  const std::vector<double> d0={0.015,0.002,0.0,0.002,0.04,0.005,0.0,0.005,0.08};
  std::vector<double> full_d(static_cast<std::size_t>(d)*d,0.0);
  for(int row=0;row<d;++row)for(int column=0;column<d;++column)for(int a=0;a<r;++a)for(int b=0;b<r;++b)
    full_d[static_cast<std::size_t>(row)*d+column]+=q[static_cast<std::size_t>(a)*d+row]*
      d0[static_cast<std::size_t>(a)*r+b]*q[static_cast<std::size_t>(b)*d+column];
  assert(cudaMemcpy(baseline.k,full_d.data(),full_d.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> theta(p,0.0),scale={0.7,0.8,0.9,1.1,1.2,1.3};
  CovarianceBaseline reference;reference.theta=theta;reference.epsilon=epsilon;
  double logdet=0.0,shifted_logdet=0.0;assert(system.evaluate_factors(theta,epsilon,logdet,shifted_logdet,&reference.trace));
  reference.logdet=logdet;reference.shifted_logdet=shifted_logdet;

  std::vector<double> omega(static_cast<std::size_t>(r)*128);
  for(int row=0;row<r;++row)for(int column=0;column<128;++column){
    const unsigned parity=static_cast<unsigned>(row&column);int bits=0;for(unsigned x=parity;x;x>>=1)bits+=static_cast<int>(x&1U);
    omega[static_cast<std::size_t>(column)*r+row]=(bits&1)?-1.0:1.0;}
  const int aa[p]={0,0,0,1,1,2},bb[p]={0,1,2,1,2,2};std::vector<std::vector<double>> a_compact(p,std::vector<double>(r*r,0.0));
  const double signs[2]={-1.0,1.0};const int atoms[2]={0,1};
  for(int parameter=0;parameter<p;++parameter){std::vector<double> ambient(static_cast<std::size_t>(d)*d,0.0);
    for(int u=0;u<2;++u)for(int v=0;v<2;++v){const double coefficient=signs[u]*signs[v]/
        (sqrt_mass[aa[parameter]*n+atoms[u]]*sqrt_mass[bb[parameter]*n+atoms[v]]);
      const int row=aa[parameter]*n+atoms[u],column=bb[parameter]*n+atoms[v];
      ambient[static_cast<std::size_t>(row)*d+column]+=coefficient;
      if(aa[parameter]!=bb[parameter])ambient[static_cast<std::size_t>(column)*d+row]+=coefficient;}
    for(int i=0;i<r;++i)for(int j=0;j<r;++j)for(int row=0;row<d;++row)for(int column=0;column<d;++column)
      a_compact[parameter][static_cast<std::size_t>(i)*r+j]+=q[static_cast<std::size_t>(i)*d+row]*
        ambient[static_cast<std::size_t>(row)*d+column]*q[static_cast<std::size_t>(j)*d+column];}
  auto multiply=[](const std::vector<double>& left,const std::vector<double>& right,const int size){
    std::vector<double> out(static_cast<std::size_t>(size)*size,0.0);
    for(int i=0;i<size;++i)for(int j=0;j<size;++j)for(int k=0;k<size;++k)
      out[static_cast<std::size_t>(i)*size+j]+=left[static_cast<std::size_t>(i)*size+k]*right[static_cast<std::size_t>(k)*size+j];return out;};
  auto inverse=[](const std::vector<double>& matrix,const int size){std::vector<double> a(static_cast<std::size_t>(size)*(2*size),0.0);
    for(int i=0;i<size;++i){for(int j=0;j<size;++j)a[static_cast<std::size_t>(i)*(2*size)+j]=matrix[static_cast<std::size_t>(i)*size+j];
      a[static_cast<std::size_t>(i)*(2*size)+size+i]=1.0;}
    for(int k=0;k<size;++k){const double pivot=a[static_cast<std::size_t>(k)*(2*size)+k];assert(pivot>0.0);
      for(int j=0;j<2*size;++j)a[static_cast<std::size_t>(k)*(2*size)+j]/=pivot;
      for(int i=0;i<size;++i)if(i!=k){const double factor=a[static_cast<std::size_t>(i)*(2*size)+k];
        for(int j=0;j<2*size;++j)a[static_cast<std::size_t>(i)*(2*size)+j]-=factor*a[static_cast<std::size_t>(k)*(2*size)+j];}}
    std::vector<double> out(static_cast<std::size_t>(size)*size);for(int i=0;i<size;++i)for(int j=0;j<size;++j)
      out[static_cast<std::size_t>(i)*size+j]=a[static_cast<std::size_t>(i)*(2*size)+size+j];return out;};
  auto gram=[](const std::vector<double>& inv,const std::vector<double>& left,const std::vector<double>& right,const int size){
    auto product=[](const std::vector<double>& a,const std::vector<double>& b,const int m){std::vector<double> c(static_cast<std::size_t>(m)*m,0.0);
      for(int i=0;i<m;++i)for(int j=0;j<m;++j)for(int k=0;k<m;++k)c[static_cast<std::size_t>(i)*m+j]+=a[static_cast<std::size_t>(i)*m+k]*b[static_cast<std::size_t>(k)*m+j];return c;};
    const auto x=product(product(product(inv,left,size),inv,size),right,size);double trace=0.0;
    for(int i=0;i<size;++i)trace+=x[static_cast<std::size_t>(i)*size+i];return trace;};
  const std::vector<double> shifted_d={d0[0]-epsilon,d0[1],d0[2],d0[3],d0[4]-epsilon,d0[5],d0[6],d0[7],d0[8]-epsilon};
  const auto inv_d=inverse(d0,r),inv_q=inverse(shifted_d,r);
  auto whitened_operators=[&](const std::vector<double>& covariance){
    std::vector<double> lower(static_cast<std::size_t>(r)*r,0.0);
    for(int row=0;row<r;++row)for(int column=0;column<=row;++column){
      double value=covariance[static_cast<std::size_t>(row)*r+column];
      for(int k=0;k<column;++k)value-=lower[static_cast<std::size_t>(row)*r+k]*lower[static_cast<std::size_t>(column)*r+k];
      lower[static_cast<std::size_t>(row)*r+column]=row==column?std::sqrt(value):
        value/lower[static_cast<std::size_t>(column)*r+column];}
    const std::vector<double> inv_lower=inverse(lower,r);std::vector<double> inv_lower_t(inv_lower.size());
    for(int row=0;row<r;++row)for(int column=0;column<r;++column)
      inv_lower_t[static_cast<std::size_t>(row)*r+column]=inv_lower[static_cast<std::size_t>(column)*r+row];
    std::vector<std::vector<double>> operators(p);
    for(int parameter=0;parameter<p;++parameter){std::vector<double> scaled=a_compact[parameter];
      for(double& value:scaled)value*=scale[parameter];
      operators[parameter]=multiply(multiply(inv_lower,scaled,r),inv_lower_t,r);}
    return operators;};
  auto projected_gram=[&](const std::vector<std::vector<double>>& operators,const std::vector<double>& basis,
      const std::vector<double>& residual,const int rank,const int columns){
    std::vector<double> expected(static_cast<std::size_t>(p)*p,0.0);
    for(int i=0;i<p;++i)for(int j=0;j<p;++j){double value=0.0;
      for(int k=0;k<rank;++k)for(int row=0;row<r;++row){double bi=0.0,bj=0.0;
        for(int column=0;column<r;++column){bi+=operators[i][static_cast<std::size_t>(row)*r+column]*
            basis[static_cast<std::size_t>(k)*r+column];
          bj+=operators[j][static_cast<std::size_t>(row)*r+column]*basis[static_cast<std::size_t>(k)*r+column];}
        value+=bi*bj;}
      for(int probe=0;probe<columns;++probe)for(int row=0;row<r;++row){double bi=0.0,bj=0.0;
        for(int column=0;column<r;++column){bi+=operators[i][static_cast<std::size_t>(row)*r+column]*
            residual[static_cast<std::size_t>(probe)*r+column];
          bj+=operators[j][static_cast<std::size_t>(row)*r+column]*residual[static_cast<std::size_t>(probe)*r+column];}
        value+=bi*bj/static_cast<double>(columns);}
      expected[static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*p]=value;}
    return expected;};
  const std::vector<double> omega16(omega.begin(),omega.begin()+static_cast<std::ptrdiff_t>(r)*probes);
  const std::vector<double> direction={0.25,-1.0,0.75,-0.5,1.25,-0.875};
  const auto operators_d=whitened_operators(d0),operators_q=whitened_operators(shifted_d);
  std::uint64_t combined_rhs=0;CovarianceAnalyticCurvatureWorkspace action_workspace(system,probes);
  auto check_combined_action=[&](double* factor,const std::vector<std::vector<double>>& operators){
    (void)action_workspace.gram_for_factor(factor,omega16,probes,scale,combined_rhs,nullptr,nullptr,false);
    const std::vector<double> combined=action_workspace.combine_parameter_actions(direction,probes);
    for(int probe=0;probe<probes;++probe)for(int row=0;row<r;++row){double expected=0.0;
      for(int parameter=0;parameter<p;++parameter){double action=0.0;
        for(int column=0;column<r;++column)action+=operators[parameter][static_cast<std::size_t>(row)*r+column]*
          omega16[static_cast<std::size_t>(probe)*r+column];
        expected+=direction[parameter]*action;}
      check_close(combined[static_cast<std::size_t>(probe)*r+row],expected,2e-10);}};
  check_combined_action(system.compact,operators_d);
  check_combined_action(system.shifted,operators_q);
  const std::vector<double> partial_basis={1.0,0.0,0.0};
  const std::vector<double> partial_residual=covariance_project_curvature_probes(omega16,partial_basis,r,probes);
  for(int probe=0;probe<probes;++probe)
    check_close(partial_residual[static_cast<std::size_t>(probe)*r],0.0,1e-14);
  const auto expected_partial_d=projected_gram(operators_d,partial_basis,partial_residual,1,probes);
  const auto expected_partial_q=projected_gram(operators_q,partial_basis,partial_residual,1,probes);
  std::uint64_t oracle_rhs=0;
  const CovarianceAnalyticCurvatureCache partial_oracle=covariance_build_analytic_curvature_cache(
    system,theta,epsilon,scale,probes,oracle_rhs,nullptr,nullptr,&omega16,covariance_primary_curvature_seed,&partial_basis);
  assert(partial_oracle.valid&&partial_oracle.curvature_subspace_rank==1&&
    partial_oracle.curvature_subspace==partial_basis);
  for(std::size_t i=0;i<expected_partial_d.size();++i){
    check_close(partial_oracle.gram_d[i],expected_partial_d[i],2e-10);
    check_close(partial_oracle.gram_q[i],expected_partial_q[i],2e-10);}
  const std::vector<double> full_basis={1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0};
  const CovarianceAnalyticCurvatureCache full_oracle=covariance_build_analytic_curvature_cache(
    system,theta,epsilon,scale,probes,oracle_rhs,nullptr,nullptr,&omega16,covariance_primary_curvature_seed,&full_basis);
  assert(full_oracle.curvature_subspace_rank==r);
  for(int i=0;i<p;++i)for(int j=0;j<p;++j){
    const std::size_t index=static_cast<std::size_t>(i)+static_cast<std::size_t>(j)*p;
    check_close(full_oracle.gram_d[index],scale[i]*scale[j]*gram(inv_d,a_compact[i],a_compact[j],r),2e-10);
    check_close(full_oracle.gram_q[index],scale[i]*scale[j]*gram(inv_q,a_compact[i],a_compact[j],r),2e-10);}
  const auto d_a=multiply(d0,a_compact[1],r),a_d=multiply(a_compact[1],d0,r);double commutator=0.0;
  for(std::size_t i=0;i<d_a.size();++i)commutator=std::hypot(commutator,d_a[i]-a_d[i]);
  assert(commutator>1e-4);
  CovarianceIndependentCurvatureDiagnostic nested_diagnostic;std::uint64_t rhs_columns=0;
  const CovarianceAnalyticCurvatureCache cache=covariance_build_stable_analytic_curvature_cache(
    system,theta,epsilon,scale,probes,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,rhs_columns,
    nullptr,nullptr,&omega,nullptr,&nested_diagnostic);
  assert(cache.valid&&cache.probes==32&&cache.levels_built==2&&cache.stability_status=="STABLE_10_PERCENT_ENGINEERING_ONLY");
  assert(cache.curvature_subspace_rank>0&&cache.curvature_subspace_rank<=std::min(8,r)&&
    cache.curvature_subspace.size()==static_cast<std::size_t>(r)*cache.curvature_subspace_rank&&
    nested_diagnostic.curvature_subspace_rank==cache.curvature_subspace_rank&&
    nested_diagnostic.curvature_pilot_seed==covariance_pilot_curvature_seed);
  assert(nested_diagnostic.frozen_at_probes==16&&nested_diagnostic.directions.size()==3);
  assert(nested_diagnostic.levels.size()==2&&nested_diagnostic.levels[0].probes==16&&
    nested_diagnostic.levels[1].probes==32&&nested_diagnostic.levels[0].primary.size()==3&&
    nested_diagnostic.levels[1].primary.size()==3);
  const std::uint64_t expected_curvature_columns=static_cast<std::uint64_t>(1+p)*
    (2*std::min(4,r)+2ULL*(16+cache.curvature_subspace_rank)+2ULL*(32+cache.curvature_subspace_rank));
  assert(cache.rhs_columns==rhs_columns&&rhs_columns==expected_curvature_columns);
  assert(cache.relative_matrix_change<=0.1&&cache.relative_step_change<=0.1);
  const CovarianceAnalyticCurvatureCache upgraded=covariance_build_stable_analytic_curvature_cache(
    system,theta,epsilon,scale,64,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},0.5,rhs_columns,nullptr,nullptr,&omega,&cache);
  assert(upgraded.valid&&upgraded.probes==64&&upgraded.levels_built==1);
  assert(upgraded.curvature_subspace==cache.curvature_subspace);
  assert(upgraded.stability_status=="STABLE_10_PERCENT_ENGINEERING_ONLY"&&upgraded.relative_matrix_change<=0.1&&upgraded.relative_step_change<=0.1);
  assert(upgraded.rhs_columns==rhs_columns&&rhs_columns-cache.rhs_columns==
    2ULL*(1+p)*(64+cache.curvature_subspace_rank));
  std::vector<double> next_theta=theta;next_theta[0]=1e-4;const std::uint64_t before_inherited=rhs_columns;
  const CovarianceAnalyticCurvatureCache inherited=covariance_build_analytic_curvature_cache(
    system,next_theta,epsilon,scale,upgraded.probes,rhs_columns);
  assert(inherited.valid&&inherited.probes==64&&inherited.levels_built==1);
  assert(rhs_columns-before_inherited==2ULL*(1+p)*64);
  constexpr double rho=0.1,tau=1.0,ridge=0.017;
  const std::vector<double> actual=covariance_analytic_curvature_hessian(cache,p,rho,tau,ridge);
  std::vector<double> expected(static_cast<std::size_t>(p)*p);
  for(int i=0;i<p;++i)for(int j=0;j<p;++j){const double gd=gram(inv_d,a_compact[i],a_compact[j],r);
    const double gq=gram(inv_q,a_compact[i],a_compact[j],r);
    expected[static_cast<std::size_t>(i)*p+j]=scale[i]*scale[j]*(0.5*(1.0+rho)*gd+tau*gq)+(i==j?ridge:0.0);}
  for(std::size_t i=0;i<expected.size();++i)check_close(actual[i],expected[i],2e-8);

  std::uint64_t independent_rhs=0;
  CovarianceIndependentCurvatureDiagnostic frozen_diagnostic;
  const CovarianceAnalyticCurvatureCache independent_primary=covariance_build_stable_analytic_curvature_cache(
    system,theta,epsilon,scale,16,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,independent_rhs,
    nullptr,nullptr,nullptr,nullptr,&frozen_diagnostic);
  assert(independent_primary.valid);
  const auto frozen_directions=frozen_diagnostic.directions;assert(frozen_diagnostic.frozen_at_probes==16);
  const auto primary_omega=covariance_curvature_probes(r,independent_primary.probes);
  const auto independent_omega=covariance_curvature_probes(r,independent_primary.probes,nullptr,
    covariance_independent_curvature_seed);
  assert(primary_omega!=independent_omega);
  const std::vector<double> identity_hessian(static_cast<std::size_t>(p)*p,0.0);
  std::vector<double> perturbed_hessian=identity_hessian;
  for(int i=0;i<p;++i)perturbed_hessian[static_cast<std::size_t>(i)*p+i]=1.0;
  perturbed_hessian[0]+=1.0;
  const double injected_error=covariance_directional_hessian_relative_error(identity_hessian,perturbed_hessian,
    std::vector<double>{1,0,0,0,0,0});
  assert(injected_error>covariance_independent_directional_tolerance);
  const CovarianceIndependentCurvatureCheck independent_check=covariance_verify_independent_curvature_batch(system,
    independent_primary,epsilon,scale,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,independent_rhs,
    nullptr,nullptr,nullptr,&frozen_diagnostic);
  assert(independent_check.valid&&independent_check.status=="PASS_INDEPENDENT_OMEGA_DIRECTIONAL");
  assert(independent_check.primary.curvature_subspace==independent_check.independent.curvature_subspace&&
    independent_check.independent.curvature_subspace==independent_primary.curvature_subspace);
  assert(independent_check.probes>=independent_primary.probes&&independent_check.probes<=128);
  assert(independent_check.directions_checked==3&&independent_check.independent_rhs_columns>0);
  assert(independent_check.independent_batches>0&&independent_rhs>=independent_check.independent_rhs_columns);
  assert(independent_check.max_relative_directional_error<=covariance_independent_directional_tolerance);
  std::uint64_t plain_rhs=0;
  const CovarianceIndependentCurvatureCheck plain_check=covariance_verify_independent_curvature_batch(system,
    independent_primary,epsilon,scale,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,plain_rhs);
  assert(plain_check.valid==independent_check.valid&&plain_check.status==independent_check.status&&
    plain_check.probes==independent_check.probes&&plain_check.directions_checked==independent_check.directions_checked&&
    plain_check.max_relative_directional_error==independent_check.max_relative_directional_error&&
    plain_check.primary.gram_d==independent_check.primary.gram_d&&plain_check.primary.gram_q==independent_check.primary.gram_q&&
    plain_check.independent.gram_d==independent_check.independent.gram_d&&
    plain_check.independent.gram_q==independent_check.independent.gram_q&&
    plain_check.independent.curvature_subspace==independent_check.independent.curvature_subspace);
  assert(frozen_diagnostic.directions.size()==frozen_directions.size());
  for(std::size_t i=0;i<frozen_directions.size();++i){assert(frozen_diagnostic.directions[i].first==frozen_directions[i].first);
    assert(frozen_diagnostic.directions[i].second==frozen_directions[i].second);}
  bool evaluated_independent_level=false;
  for(const auto& level:frozen_diagnostic.levels)if(level.probes==independent_check.probes){
    evaluated_independent_level=level.primary_status=="EVALUATED"&&level.independent_status=="EVALUATED"&&
      level.primary.size()==frozen_directions.size()&&level.independent.size()==frozen_directions.size();}
  assert(evaluated_independent_level);
  std::ostringstream diagnostic_log;
  covariance_write_independent_curvature_diagnostic(diagnostic_log,frozen_diagnostic,0.1,1.0,0.017,"TEST");
  assert(diagnostic_log.str().find("COVARIANCE_CURVATURE_GATE_DIRECTION")!=std::string::npos&&
    diagnostic_log.str().find("role=ORIGINAL_GATE")!=std::string::npos&&
    diagnostic_log.str().find("name=PROPOSED_TR_STEP")!=std::string::npos&&
    diagnostic_log.str().find("curvature_pilot_seed="+std::to_string(covariance_pilot_curvature_seed))!=std::string::npos);
  CovarianceIndependentCurvatureDiagnostic failed_diagnostic;std::uint64_t failed_rhs=0;
  const CovarianceAnalyticCurvatureCache failed_primary=covariance_build_stable_analytic_curvature_cache(
    system,theta,epsilon,scale,16,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,failed_rhs,
    nullptr,nullptr,nullptr,nullptr,&failed_diagnostic);
  const std::vector<double> failed_hessian=covariance_analytic_curvature_hessian(failed_primary,p,0.1,1.0,0.017);
  const CovarianceTrustRegionModel failed_model=covariance_direct_hessian_trust_region_model(system.solver,failed_hessian,p);
  assert(failed_model.positive_definite);
  const auto failed_gate_directions=covariance_independent_gate_directions(failed_model,{1,2,3,4,5,6},1.0);
  double failed_hessian_norm=0.0;for(const double value:failed_hessian)failed_hessian_norm=std::hypot(failed_hessian_norm,value);
  const double failure_bias=10.0*(failed_hessian_norm+1.0)/(0.5*(1.0+0.1));
  CovarianceAnalyticCurvatureCache biased_primary=failed_primary;
  for(int i=0;i<p;++i)biased_primary.gram_d[static_cast<std::size_t>(i)+static_cast<std::size_t>(i)*p]+=failure_bias;
  const std::vector<double> biased_hessian=covariance_analytic_curvature_hessian(biased_primary,p,0.1,1.0,0.017);
  for(const auto& named:failed_gate_directions)
    assert(covariance_directional_hessian_relative_error(failed_hessian,biased_hessian,named.second)>
      covariance_independent_directional_tolerance);
  covariance_fit_test_hooks={};covariance_fit_test_hooks.independent_curvature_bias_remaining=100;
  covariance_fit_test_hooks.independent_curvature_diagonal_bias=failure_bias;
  const CovarianceIndependentCurvatureCheck failed_independent=covariance_verify_independent_curvature_batch(system,
    failed_primary,epsilon,scale,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,failed_rhs,
    nullptr,nullptr,nullptr,&failed_diagnostic);
  covariance_fit_test_hooks={};
  assert(!failed_independent.valid&&failed_independent.status=="CURVATURE_UNRESOLVED_INDEPENDENT_BATCH"&&
    failed_independent.probes==128&&failed_independent.max_relative_directional_error>covariance_independent_directional_tolerance);
  assert(failed_diagnostic.frozen_at_probes==16&&failed_diagnostic.directions.size()==3);
  for(int expected_probes:{16,32,64,128}){bool found=false;for(const auto& level:failed_diagnostic.levels)
      if(level.probes==expected_probes){found=true;assert(level.primary_status=="EVALUATED"&&level.independent_status=="EVALUATED");}
    assert(found);}
  covariance_fit_test_hooks.independent_curvature_hard_failures_remaining=1;
  bool hard_error_propagated=false;
  try{(void)covariance_verify_independent_curvature_batch(system,independent_check.primary,
      epsilon,scale,0.1,1.0,0.017,std::vector<double>{1,2,3,4,5,6},1.0,independent_rhs);}
  catch(const std::runtime_error& error){hard_error_propagated=std::string(error.what()).find("CUDA_ERROR:")==0;}
  assert(hard_error_propagated&&covariance_fit_test_hooks.independent_curvature_hard_failures_remaining==0);
  CovarianceAnalyticCurvatureCache same_point=independent_check.independent;
  assert(covariance_same_curvature_point(independent_check.primary,same_point));
  const std::vector<double> at_tau_one=covariance_analytic_curvature_hessian(same_point,p,0.1,1.0,0.017);
  const std::vector<double> at_tau_point_two=covariance_analytic_curvature_hessian(same_point,p,0.1,0.2,0.017);
  bool changed_with_tau=false;for(std::size_t i=0;i<at_tau_one.size();++i)
    changed_with_tau|=std::abs(at_tau_one[i]-at_tau_point_two[i])>1e-12;
  assert(changed_with_tau);
  const auto reused_tau=covariance_verify_independent_curvature_batch(system,independent_check.primary,
    epsilon,scale,0.1,0.2,0.017,std::vector<double>{1,2,3,4,5,6},1.0,independent_rhs,
    nullptr,nullptr,&independent_check.independent);
  assert(reused_tau.valid&&reused_tau.independent_batches==0&&
    reused_tau.primary.curvature_subspace==independent_check.primary.curvature_subspace);
  same_point.theta[0]+=1e-4;assert(!covariance_same_curvature_point(independent_check.primary,same_point));
  same_point=independent_check.independent;same_point.coordinate_scale[0]*=2.0;
  assert(!covariance_same_curvature_point(independent_check.primary,same_point));
  same_point=independent_check.independent;same_point.epsilon+=1e-4;
  assert(!covariance_same_curvature_point(independent_check.primary,same_point));
  same_point=independent_check.independent;++same_point.graph_assembly_generation;
  assert(!covariance_same_curvature_point(independent_check.primary,same_point));
  same_point=independent_check.independent;same_point.curvature_subspace[0]+=1e-3;
  assert(!covariance_same_curvature_point(independent_check.primary,same_point));
}

void test_covariance_fit_workspace_grows_reuses_and_releases()
{
  rpmd_ja_device::CovarianceFitWorkspace workspace;
  const std::uint64_t first_growth=workspace.curvature_growth_transient_bytes(8,4,32,6,16);
  assert(first_growth>0);
  workspace.reserve_curvature(8,4,32,6,16);
  assert(workspace.curvature_allocation_count==7&&workspace.curvature_capacity_bytes()>0);
  const std::uint64_t first_peak=workspace.curvature_peak_bytes;
  assert(workspace.curvature_growth_transient_bytes(8,4,32,6,16)==0);
  workspace.reserve_curvature(8,4,32,6,16);
  assert(workspace.curvature_allocation_count==7&&workspace.curvature_peak_bytes==first_peak);
  const std::uint64_t larger_growth=workspace.curvature_growth_transient_bytes(16,8,64,12,16);
  assert(larger_growth>0);
  workspace.reserve_curvature(16,8,64,12,16);
  assert(workspace.curvature_allocation_count==12&&workspace.curvature_peak_bytes>first_peak);
  const std::uint64_t larger_peak=workspace.curvature_peak_bytes;
  workspace.release_curvature();
  assert(workspace.curvature_capacity_bytes()==0&&workspace.curvature_peak_bytes==larger_peak);
  workspace.reserve_curvature(8,4,32,6,16);
  assert(workspace.curvature_allocation_count==19&&workspace.curvature_peak_bytes==larger_peak);
}

void test_covariance_fit_workspace_is_reused_by_sequential_system_curvature_builds()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  rpmd_ja_device::CovarianceFitWorkspace workspace;
  const std::vector<double> theta0(6,0.0),scale(6,1.0);
  std::vector<double> theta1=theta0;theta1[0]=1.0e-3;
  CovarianceAnalyticCurvatureCache first,second;
  std::uint64_t rhs_columns=0;
  {
    DeviceCovarianceSystem system0(baseline,graph,sqrt_mass,masses,&workspace);
    first=covariance_build_analytic_curvature_cache(system0,theta0,0.01,scale,16,rhs_columns);
    assert(first.valid&&workspace.curvature_allocation_count==7);
    const std::uint64_t first_peak=workspace.curvature_peak_bytes;
    DeviceCovarianceSystem system1(baseline,graph,sqrt_mass,masses,&workspace);
    second=covariance_build_analytic_curvature_cache(system1,theta1,0.01,scale,16,rhs_columns);
    assert(second.valid&&workspace.curvature_allocation_count==7&&workspace.curvature_peak_bytes==first_peak);
    assert(!covariance_same_curvature_point(first,second));
    double gram_difference=0.0;
    for(std::size_t i=0;i<first.gram_d.size();++i)gram_difference+=std::abs(first.gram_d[i]-second.gram_d[i]);
    assert(gram_difference>1.0e-10);
  }
  assert(workspace.curvature_capacity_bytes()>0);
  workspace.release_curvature();
  assert(workspace.curvature_capacity_bytes()==0);
}

void test_covariance_device_workspace_reuses_buffers_static_graph_and_same_point_factors()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  const Graph m0=make_two_atom_covariance_graph();Graph m1=m0;
  m1.base_parameter_count=6;m1.parameter_count=7;m1.raw_local_parameter_count=1;
  m1.local_base_mix.assign(6,0.0);m1.local_transform={1.0};m1.parameter_basis_norms.assign(7,1.0);
  rpmd_ja_local_basis::EdgeInput local_edge{};local_edge.first={0,{0,0,0}};local_edge.second={1,{0,0,0}};
  local_edge.axis={1.0,0.0,0.0};local_edge.longitudinal_parameter=0;local_edge.transverse_parameter=0;
  m1.local_terms=rpmd_ja_local_basis::make_edge_terms(local_edge);
  for(auto& term:m1.local_terms)term.parameter=0;
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> theta0(6,0.0),theta1(7,0.0);std::vector<double> theta_point=theta0;theta_point[0]=1.0e-3;
  DeviceCovarianceSystemWorkspace shared;
  auto evaluate=[&](const Graph& graph,const std::vector<double>& theta,const double epsilon,
                    DeviceCovarianceSystemWorkspace& workspace){
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses,nullptr,&workspace);
    double logdet=0.0,shifted=0.0;
    if(!system.evaluate_factors(theta,epsilon,logdet,shifted,nullptr))
      throw std::runtime_error("workspace reuse regression produced a non-SPD matrix");
    return std::array<double,2>{logdet,shifted};};
  const std::vector<std::tuple<const Graph*,const std::vector<double>*,double>> cases={
    {&m0,&theta0,0.01},{&m0,&theta0,0.01},{&m0,&theta0,0.02},{&m0,&theta_point,0.02},
    {&m1,&theta1,0.02},{&m0,&theta0,0.01},{&m0,&theta0,0.01}};
  std::vector<std::array<double,2>> shared_values;shared_values.reserve(cases.size());
  for(std::size_t i=0;i<cases.size();++i){const auto& item=cases[i];
    const std::uint64_t factorizations_before=shared.factorization_count;
    const std::uint64_t uploads_before=shared.static_upload_count;
    shared_values.push_back(evaluate(*std::get<0>(item),*std::get<1>(item),std::get<2>(item),shared));
    if(i==1){assert(shared.factorization_count==factorizations_before&&shared.static_upload_count==uploads_before);}
    if(i==2||i==3||i==4||i==5)assert(shared.factorization_count>factorizations_before);
    if(i==4||i==5)assert(shared.static_upload_count==uploads_before+1);
    if(i==6)assert(shared.factorization_count==factorizations_before&&shared.static_upload_count==uploads_before);
  }
  assert(shared.static_upload_count==3&&shared.allocation_count>0);
  std::uint64_t independent_allocations=0,independent_uploads=0;
  for(std::size_t i=0;i<cases.size();++i){DeviceCovarianceSystemWorkspace independent;
    const auto& item=cases[i];const auto values=evaluate(*std::get<0>(item),*std::get<1>(item),std::get<2>(item),independent);
    check_close(shared_values[i][0],values[0],1e-12);check_close(shared_values[i][1],values[1],1e-12);
    independent_allocations+=independent.allocation_count;independent_uploads+=independent.static_upload_count;}
  assert(shared.allocation_count<independent_allocations&&shared.static_upload_count<independent_uploads);

  Graph same_address=m1;std::vector<double> local_theta(7,0.0);local_theta[6]=1.0e-4;
  const std::uint64_t uploads_before_graph_mutation=shared.static_upload_count;
  const std::uint64_t factors_before_graph_mutation=shared.factorization_count;
  (void)evaluate(same_address,local_theta,0.02,shared);
  same_address.local_transform[0]=0.75;
  const auto changed_graph_value=evaluate(same_address,local_theta,0.02,shared);
  assert(shared.static_upload_count==uploads_before_graph_mutation+1&&shared.factorization_count>factors_before_graph_mutation);
  {DeviceCovarianceSystemWorkspace independent;
    const auto expected=evaluate(same_address,local_theta,0.02,independent);
    check_close(changed_graph_value[0],expected[0],1e-12);check_close(changed_graph_value[1],expected[1],1e-12);}

  DeviceBaseline changed_baseline;initialize_two_atom_covariance_baseline(changed_baseline,true);
  std::vector<double> changed_matrix(36,0.0);const double t[2]={1.0/std::sqrt(5.0),2.0/std::sqrt(5.0)};
  for(int axis=0;axis<3;++axis)for(int i=0;i<2;++i)for(int j=0;j<2;++j){
    const int row=axis*2+i,column=axis*2+j;
    changed_matrix[static_cast<std::size_t>(row)*6+column]=(i==j?1.0:0.0)-t[i]*t[j];}
  changed_matrix[0]+=0.05;
  assert(cudaMemcpy(changed_baseline.k,changed_matrix.data(),changed_matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  auto evaluate_changed_baseline=[&](DeviceBaseline& selected,DeviceCovarianceSystemWorkspace& workspace){
    DeviceCovarianceSystem system(selected,same_address,sqrt_mass,masses,nullptr,&workspace);
    double logdet=0.0,shifted=0.0;if(!system.evaluate_factors(local_theta,0.02,logdet,shifted,nullptr))
      throw std::runtime_error("baseline identity regression produced a non-SPD matrix");
    return std::array<double,2>{logdet,shifted};};
  const std::uint64_t uploads_before_baseline_change=shared.static_upload_count;
  const std::uint64_t factors_before_baseline_change=shared.factorization_count;
  const auto changed_baseline_value=evaluate_changed_baseline(changed_baseline,shared);
  assert(shared.static_upload_count==uploads_before_baseline_change&&shared.factorization_count>factors_before_baseline_change);
  {DeviceCovarianceSystemWorkspace independent;
    const auto expected=evaluate_changed_baseline(changed_baseline,independent);
    check_close(changed_baseline_value[0],expected[0],1e-12);check_close(changed_baseline_value[1],expected[1],1e-12);}
  const std::uint64_t factors_before_baseline_hit=shared.factorization_count;
  (void)evaluate_changed_baseline(changed_baseline,shared);
  assert(shared.factorization_count==factors_before_baseline_hit);
}

void test_covariance_factor_identity_recovery()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const double epsilon=0.01;std::vector<double> accepted(6,0.0),candidate=accepted;
  double logdet=0.0,shifted_logdet=0.0;
  assert(system.evaluate_factors(accepted,epsilon,logdet,shifted_logdet,nullptr));
  assert(system.factors_valid&&system.factor_theta==accepted&&system.factor_epsilon==epsilon);
  const std::uint64_t accepted_generation=system.factor_assembly_generation;

  candidate[0]=1e-4;
  assert(system.evaluate_factors(candidate,epsilon,logdet,shifted_logdet,nullptr));
  assert(system.factors_valid&&system.factor_theta==candidate);
  const std::uint64_t candidate_generation=system.factor_assembly_generation;
  const std::uint64_t after_candidate_factors=system.factorizations;
  assert(system.ensure_factors(accepted,epsilon));
  assert(system.factor_theta==accepted&&system.factor_assembly_generation>candidate_generation&&
    system.factor_assembly_generation>accepted_generation);
  assert(system.factorizations==after_candidate_factors+2);

  const std::uint64_t before_tau_reuse=system.factorizations;
  assert(system.ensure_factors(accepted,epsilon));
  assert(system.factorizations==before_tau_reuse);

  assert(!system.evaluate_factors(candidate,1.0,logdet,shifted_logdet,nullptr));
  assert(!system.factors_valid&&system.factor_theta.empty());
  assert(system.ensure_factors(accepted,epsilon));
  assert(system.factors_valid&&system.factor_theta==accepted&&system.factor_epsilon==epsilon);
}

void test_covariance_optimizer_boundary_start_preserves_prior_center_and_reports_status()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);const std::vector<double> statistics={1000.0,0.0,0.0,1000.0,0.0,1000.0};
  std::uint64_t baseline_evaluations=0;const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,0.01,baseline_evaluations);
  assert(reference.epsilon_constraint_active);
  std::ostringstream trace;const CovarianceFitResult fit=solve_covariance_reference(system,statistics,0.01,0.1,1,&trace);
  assert(fit.epsilon_constraint_status=="ACTIVE"&&fit.initialization_evaluations<=16);
  assert(fit.initialization_status=="IMPROVED"||fit.initialization_status=="IMPROVED_EVALUATION_LIMIT");
  assert(fit.start_graph_delta>0.0&&fit.start_objective<0.0&&std::isfinite(fit.start_raw_gradient_inf_norm));
  assert(fit.theta_baseline==reference.theta&&fit.theta_start.size()==fit.theta_baseline.size());
  const std::vector<double> direction=graph_laplacian_direction(graph);
  for(std::size_t i=0;i<direction.size();++i)check_close(fit.theta_start[i],fit.theta_baseline[i]+fit.start_graph_delta*direction[i],1e-12);
  assert(trace.str().find("COVARIANCE_INITIALIZATION_TRIAL")!=std::string::npos);
  assert(trace.str().find("phase=SHRINK")!=std::string::npos&&trace.str().find("phase=EXPAND")!=std::string::npos);
  assert(trace.str().find("theta_start ")!=std::string::npos);
  std::uint64_t start_evaluations=0;
  const CovarianceObjectiveEvaluation start=evaluate_covariance_fit_objective(system,fit.theta_start,reference,statistics,0.1,0.01,1.0,start_evaluations);
  assert(start.feasible);check_close(start.objective,fit.start_objective,1e-12);
  check_close(covariance_norm_inf(start.gradient),fit.start_raw_gradient_inf_norm,1e-12);

  std::vector<double> shifted_unstable_theta=fit.theta_baseline;
  for(std::size_t i=0;i<shifted_unstable_theta.size();++i)shifted_unstable_theta[i]-=1000.0*direction[i];
  std::uint64_t evaluations=0;
  const CovarianceObjectiveEvaluation shifted=evaluate_covariance_fit_objective(system,shifted_unstable_theta,reference,statistics,0.1,0.01,1.0,evaluations,false);
  assert(!shifted.feasible&&shifted.status==CovarianceObjectiveStatus::ShiftedNotSpd);
}

void test_covariance_initialization_shrinks_before_expanding()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> direction=graph_laplacian_direction(graph),zero_statistics(6,0.0);
  const std::vector<double> seed_statistics={1e6,0.0,0.0,1e6,0.0,1e6};std::uint64_t baseline_evaluations=0;
  CovarianceBaseline reference=build_graph_covariance_baseline(system,seed_statistics,0.01,baseline_evaluations);assert(reference.epsilon_constraint_active);
  const std::vector<double> scale=system.coordinate_scales(reference.theta,0.01);
  double delta_scale=std::numeric_limits<double>::infinity();for(std::size_t i=0;i<direction.size();++i)if(direction[i]>0.0)delta_scale=std::min(delta_scale,scale[i]/direction[i]);
  double initial_delta=0.01*delta_scale;std::uint64_t objective_evaluations=0;
  std::vector<double> coarse_theta(reference.theta.size()),half_theta(reference.theta.size());
  for(std::size_t i=0;i<direction.size();++i){coarse_theta[i]=reference.theta[i]+initial_delta*direction[i];half_theta[i]=reference.theta[i]+0.5*initial_delta*direction[i];}
  const CovarianceObjectiveEvaluation coarse_zero=evaluate_covariance_fit_objective(system,coarse_theta,reference,zero_statistics,0.1,0.01,1.0,objective_evaluations,false);
  const CovarianceObjectiveEvaluation half_zero=evaluate_covariance_fit_objective(system,half_theta,reference,zero_statistics,0.1,0.01,1.0,objective_evaluations,false);
  assert(coarse_zero.feasible&&half_zero.feasible);
  const double lower_diagonal=-coarse_zero.objective/(1.5*initial_delta),upper_diagonal=-half_zero.objective/(0.75*initial_delta);
  assert(lower_diagonal>0.0&&lower_diagonal<upper_diagonal);
  const double diagonal=0.5*(lower_diagonal+upper_diagonal);const std::vector<double> statistics={diagonal,0.0,0.0,diagonal,0.0,diagonal};
  baseline_evaluations=0;reference=build_graph_covariance_baseline(system,statistics,0.01,baseline_evaluations);assert(reference.epsilon_constraint_active);
  const std::vector<double> final_scale=system.coordinate_scales(reference.theta,0.01);
  delta_scale=std::numeric_limits<double>::infinity();for(std::size_t i=0;i<direction.size();++i)if(direction[i]>0.0)delta_scale=std::min(delta_scale,final_scale[i]/direction[i]);
  initial_delta=0.01*delta_scale;objective_evaluations=0;
  const CovarianceObjectiveEvaluation base_value=evaluate_covariance_fit_objective(system,reference.theta,reference,statistics,0.1,0.01,1.0,objective_evaluations,false);
  for(std::size_t i=0;i<direction.size();++i){coarse_theta[i]=reference.theta[i]+initial_delta*direction[i];half_theta[i]=reference.theta[i]+0.5*initial_delta*direction[i];}
  const CovarianceObjectiveEvaluation coarse=evaluate_covariance_fit_objective(system,coarse_theta,reference,statistics,0.1,0.01,1.0,objective_evaluations,false);
  const CovarianceObjectiveEvaluation half=evaluate_covariance_fit_objective(system,half_theta,reference,statistics,0.1,0.01,1.0,objective_evaluations,false);
  assert(base_value.feasible&&coarse.feasible&&half.feasible&&coarse.objective>base_value.objective&&half.objective<base_value.objective);
  std::ostringstream trace;const CovarianceFitResult fit=solve_covariance_reference(system,statistics,0.01,0.1,1,&trace);
  assert(fit.epsilon_constraint_status=="ACTIVE"&&fit.start_graph_delta>0.0);
  check_close(fit.start_graph_delta,0.5*initial_delta,1e-10);
  assert(fit.initialization_status=="IMPROVED"&&fit.initialization_evaluations<=16);
  assert(fit.theta_baseline==reference.theta&&fit.theta_start.size()==fit.theta_baseline.size());
  for(std::size_t i=0;i<direction.size();++i)check_close(fit.theta_start[i],fit.theta_baseline[i]+fit.start_graph_delta*direction[i],1e-12);
  assert(trace.str().find("phase=SHRINK")!=std::string::npos&&trace.str().find("phase=EXPAND")!=std::string::npos);
  std::uint64_t evaluations=0;const CovarianceObjectiveEvaluation selected=evaluate_covariance_fit_objective(system,fit.theta_start,reference,statistics,0.1,0.01,1.0,evaluations);
  assert(selected.feasible);check_close(selected.objective,fit.start_objective,1e-12);
  check_close(covariance_norm_inf(selected.gradient),fit.start_raw_gradient_inf_norm,1e-12);
}

void test_covariance_block_newton_optimizer_direction_trace()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::ostringstream trace;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,10,&trace,true);
  const std::string log=trace.str();
  assert(fit.block_newton_enabled&&fit.iterations>0&&fit.block_newton_factorizations>=2);
  assert(fit.block_newton_rhs_columns>0&&fit.direction_group==0&&fit.direction_group_support==2);
  assert(log.find("COVARIANCE_DIRECTION_MODE block_newton=ENABLED")!=std::string::npos);
  assert(log.find("COVARIANCE_DIRECTION iteration=0")!=std::string::npos);
  assert(log.find("method=HYBRID")!=std::string::npos||log.find("method=BLOCK")!=std::string::npos);
  assert(log.find("extra_factorizations=2")!=std::string::npos&&log.find("extra_rhs_columns=")!=std::string::npos);
  assert(log.find("dominant_scaled_gradient_parameter=")!=std::string::npos&&log.find("scaled_kkt_residual=")!=std::string::npos);
}

void test_covariance_trust_region_roundoff_boundary_rebuild()
{
  const std::vector<double> theta={1.0},z={10000.0},scale={1e-4},direction={4e-12};
  constexpr double radius=4e-12;
  std::vector<double> next_z,actual_step,next_theta;bool repaired=false;
  std::vector<double> initial_z,initial_step,initial_theta;
  assert(covariance_build_scaled_trial(theta,z,scale,direction,1.0,initial_z,initial_step,initial_theta));
  assert(initial_step[0]>radius);
  assert(covariance_build_trust_region_trial(theta,z,scale,direction,4e-12,radius,
    next_z,actual_step,next_theta,repaired));
  assert(repaired&&radius==4e-12&&actual_step[0]>0.0&&actual_step[0]<=radius);
  assert(actual_step[0]==(next_theta[0]-theta[0])/scale[0]);
  assert(next_z[0]==next_theta[0]/scale[0]);

  const std::vector<double> unequal_theta={1.0,2.0},unequal_z={10000.0,10000.0};
  const std::vector<double> unequal_scale={1e-4,2e-4},unequal_direction={4e-12,0.8e-12};
  assert(covariance_build_scaled_trial(unequal_theta,unequal_z,unequal_scale,unequal_direction,1.0,
    initial_z,initial_step,initial_theta));
  assert(initial_step[1]==0.0&&std::hypot(initial_step[0],initial_step[1])>4.3e-12);
  repaired=false;
  assert(covariance_build_trust_region_trial(unequal_theta,unequal_z,unequal_scale,
    unequal_direction,std::hypot(4e-12,0.8e-12),4.3e-12,next_z,actual_step,next_theta,repaired));
  assert(repaired&&actual_step[0]>0.0&&std::hypot(actual_step[0],actual_step[1])<=4.3e-12);
  assert(actual_step[0]==(next_theta[0]-unequal_theta[0])/unequal_scale[0]);
  assert(actual_step[1]==(next_theta[1]-unequal_theta[1])/unequal_scale[1]);

  repaired=false;
  assert(covariance_build_trust_region_trial(theta,z,scale,{1.9e-12},1.9e-12,1.9e-12,
    next_z,actual_step,next_theta,repaired));
  assert(!repaired&&actual_step[0]>1.9e-12);

  repaired=false;
  assert(covariance_build_trust_region_trial(theta,z,scale,{5e-12},5e-12,4e-12,
    next_z,actual_step,next_theta,repaired));
  assert(!repaired&&actual_step[0]>4e-12);

  bool invalid_scale_rejected=false;
  try{(void)covariance_build_trust_region_trial(theta,z,{0.0},direction,4e-12,radius,
    next_z,actual_step,next_theta,repaired);}catch(const std::invalid_argument&){invalid_scale_rejected=true;}
  assert(invalid_scale_rejected);
  bool nonfinite_rejected=false;
  try{(void)covariance_build_trust_region_trial(theta,z,scale,
    {std::numeric_limits<double>::quiet_NaN()},4e-12,radius,next_z,actual_step,next_theta,repaired);}
  catch(const std::invalid_argument&){nonfinite_rejected=true;}
  assert(nonfinite_rejected);

  const double q=std::nextafter(1.0,2.0)-1.0;
  const std::vector<double> tie_theta={1.0,1.0},tie_z={1.0,1.0},tie_scale={1.0,1.0};
  const std::vector<double> tie_direction={0.6*q,0.9*q};const double tie_radius=std::hypot(tie_direction[0],tie_direction[1]);
  repaired=false;
  assert(covariance_build_trust_region_trial(tie_theta,tie_z,tie_scale,tie_direction,tie_radius,tie_radius,
    next_z,actual_step,next_theta,repaired));
  assert(repaired&&actual_step[0]==0.0&&actual_step[1]==q&&std::hypot(actual_step[0],actual_step[1])<=tie_radius);
}

void test_covariance_optimizer_trace_parses_legacy_and_trust_region_steps()
{
  std::istringstream trace(
    "STEP_COMMITTED iteration=1 tau=0.1 alpha=0.5 predicted_first_order_decrease=2 actual_objective_decrease=1 bfgs_update=UPDATED\r\n"
    "TRUST_REGION_TRIAL iteration=2 retry=0 status=RATIO_REJECT\r\n"
    "TRUST_REGION_STEP_ACCEPT iteration=2 tau=0.1 lambda=1 radius=2 next_radius=2 step_norm=2 predicted=3 actual=2 ratio=0.66 scaled_kkt_residual=0.01 full_gradient_evaluations=2 bfgs_update=SEEDED\r\n"
    "TRUST_REGION_STEP_REJECT iteration=3 retry=1 status=POOR_RATIO\r\n"
    "TRUST_REGION_STEP_ACCEPT iteration=3 tau=0.1 lambda=2 radius=1 next_radius=1 step_norm=1 predicted=NOT_STORED actual=NOT_STORED ratio=NOT_STORED bfgs_update=RESET\r\n"
    "BFGS_CURVATURE_SEED tau=0.1 gamma=1 source=START\r\nBFGS_RESET iteration=4 reason=TEST\r\n");
  const CovarianceOptimizerTraceSummary summary=covariance_parse_optimizer_trace(trace);
  assert(summary.accepted_steps()==3&&summary.legacy_steps==1&&summary.trust_region_steps==2);
  assert(summary.bfgs_seed_events==1&&summary.bfgs_reset_events==1&&summary.tail.size()==3);
  assert(summary.tail[0].kind=="STEP_COMMITTED"&&summary.tail[1].kind=="TRUST_REGION_STEP_ACCEPT");
  assert(!summary.tail[1].fields.count("alpha")&&!summary.tail[1].fields.count("predicted_first_order_decrease"));
  assert(summary.tail[1].fields.at("predicted")=="3"&&summary.tail[1].fields.at("actual")=="2");
  assert(summary.tail[2].fields.at("predicted")=="NOT_STORED"&&summary.tail[2].fields.at("actual")=="NOT_STORED");

  std::ostringstream many_steps;
  for(int iteration=0;iteration<40;++iteration)
    many_steps<<"TRUST_REGION_STEP_ACCEPT iteration="<<iteration<<" predicted="<<iteration<<"\r\n";
  std::istringstream bounded_input(many_steps.str());
  const CovarianceOptimizerTraceSummary bounded=covariance_parse_optimizer_trace(bounded_input);
  assert(bounded.trust_region_steps==40&&bounded.tail.size()==32);
  assert(bounded.tail.front().fields.at("iteration")=="8"&&bounded.tail.back().fields.at("iteration")=="39");
}

void test_covariance_trust_region_residual_rejection_is_transactional()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
  covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=1;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,false,true,false);
  covariance_fit_test_hooks={};const std::string log=trace.str();
  assert(fit.trust_region_enabled&&!fit.block_newton_enabled&&!fit.joint_block_enabled&&fit.iterations==1);
  assert(fit.trust_region_model_decompositions==1&&fit.trust_region_gradient_residual_rejects==1);
  assert(fit.trust_region_radius_resolves>=1&&fit.gradient_residual_reject_trials==1);
  assert(fit.full_gradient_evaluations==fit.initialization_gradient_calls+fit.line_search_gradient_calls);
  const std::size_t rejected=log.find("GRADIENT_RESIDUAL_REJECT iteration=0");
  const std::size_t accepted=log.find("TRUST_REGION_STEP_ACCEPT iteration=0");
  assert(rejected!=std::string::npos&&accepted!=std::string::npos&&rejected<accepted);
  assert(log.find("TRUST_REGION_STEP_ACCEPT iteration=0",accepted+1)==std::string::npos);
  assert(log.find("bfgs_update=",accepted)!=std::string::npos&&fit.bfgs_recent_pairs<=1);
  assert(fit.last_valid.available&&fit.last_valid.theta==fit.theta&&!fit.failed_trial.available);
  const std::string path="rpmd_ja_native_trust_region_state_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=1e-6;options.fd_step=1e-4;options.sample_interval=1;
  options.covariance_trust_region=true;options.covariance_staged_barrier=false;
  SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};
  CovarianceFitResult state_fit=fit;state_fit.training_block_statistics={{1,2,3,4,5,6}};state_fit.training_block_frames={2};
  write_covariance_state(path,state_fit,0.1,"test.samples","test.qraw",options,header,identity,graph,4,2);
  std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
  assert(fields.at("optimizer_step_mode")=="TRUST_REGION_OPT_IN"&&fields.at("trust_region_enabled")=="1");
  assert(fields.at("staged_barrier_enabled")=="0"&&fields.at("barrier_stage_policy")=="STRICT_KKT_EACH_STAGE");
  assert(fields.at("trust_region_model_decompositions")=="1"&&fields.at("trust_region_gradient_residual_rejects")=="1");
  assert(fields.at("training_block_count")=="1"&&fields.at("training_block_frames_0")=="2");
  assert(read_snapshot_vector(read_test_file(path),"training_block_statistics_0")==std::vector<double>({1,2,3,4,5,6}));
}

void test_covariance_initialization_reports_no_improvement_at_budget()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> statistics={1e12,0.0,0.0,1e12,0.0,1e12};std::ostringstream trace;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,0.01,0.1,1,&trace);
  assert(fit.epsilon_constraint_status=="ACTIVE"&&fit.initialization_status=="NO_IMPROVEMENT_INITIALIZATION_EVALUATION_LIMIT");
  assert(fit.initialization_evaluations==16&&fit.start_graph_delta==0.0&&fit.theta_start==fit.theta_baseline);
  assert(trace.str().find("phase=SHRINK")!=std::string::npos&&trace.str().find("phase=EXPAND")==std::string::npos);
  assert(trace.str().find("stop_reason INITIALIZATION_EVALUATION_LIMIT")!=std::string::npos);
}

void test_covariance_directional_curvature_requires_stable_resolved_match()
{
  const std::vector<double> duplicate_actions={1.0,0.0,0.0,2.0,0.0,0.0,0.0,1.0,0.0,0.0,2.0,0.0};
  const std::vector<double> rank_deficient_basis=covariance_orthonormalize_curvature_actions(duplicate_actions,3,4);
  assert(rank_deficient_basis.size()==6);
  const std::vector<double> zero_basis=covariance_orthonormalize_curvature_actions(std::vector<double>(9,0.0),3,3);
  assert(zero_basis.empty());
  bool nonfinite_pilot_rejected=false;
  try{(void)covariance_orthonormalize_curvature_actions({1.0,std::numeric_limits<double>::quiet_NaN()},2,1);}
  catch(const std::invalid_argument&){nonfinite_pilot_rejected=true;}
  assert(nonfinite_pilot_rejected);
  const std::vector<double> complement=covariance_project_curvature_probes(
    {1.0,2.0,3.0,4.0,5.0,6.0},rank_deficient_basis,3,2);
  for(int column=0;column<2;++column)for(int basis_column=0;basis_column<2;++basis_column){
    double dot=0.0;for(int row=0;row<3;++row)
      dot+=rank_deficient_basis[static_cast<std::size_t>(basis_column)*3+row]*
        complement[static_cast<std::size_t>(column)*3+row];
    check_close(dot,0.0,1e-13);}
  const double binade=std::ldexp(1.0,53);
  assert(!covariance_directional_displacement_representable({binade},{1.0},{1.0},1.0));
  assert(covariance_directional_displacement_representable({binade},{1.0},{1.0},2.0));
  assert(!covariance_directional_displacement_representable({binade},{3000.0},{1.0},1.0e-3));
  assert(covariance_directional_displacement_representable({binade},{3000.0},{1.0},2.0e-3));
  const auto pass=covariance_compare_directional_curvature(2.0,2.1,2.025,2.00625,1e-10,1e-10,1e-10);
  assert(pass.resolved&&pass.status=="PASS"&&pass.middle==2.025&&pass.fine==2.00625&&
    pass.relative_denominator==2.1&&
    pass.total_absolute_tolerance==pass.relative_tolerance*pass.relative_denominator+pass.absolute_tolerance&&
    pass.absolute_error<=pass.total_absolute_tolerance&&pass.step_relative_error<0.01);
  const auto unstable=covariance_compare_directional_curvature(1.5,1.0,1.5,2.0,1e-10,1e-10,1e-10);
  assert(!unstable.resolved&&unstable.status=="STEP_HALVING_UNSTABLE");
  const auto unresolved=covariance_compare_directional_curvature(1.0,1e-12,2e-12,3e-12,1e-8,1e-8,1e-8);
  assert(!unresolved.resolved&&unresolved.status=="DIFFERENCE_BELOW_RESOLUTION");
  const auto mismatch=covariance_compare_directional_curvature(4.0,2.1,2.025,2.00625,1e-10,1e-10,1e-10);
  assert(!mismatch.resolved&&mismatch.status=="GRAM_FINITE_DIFFERENCE_MISMATCH");
  const auto shared_fd_primary=covariance_compare_directional_curvature(2.0,2.1,2.025,2.00625,1e-10,1e-10,1e-10);
  const auto shared_fd_independent=covariance_compare_directional_curvature(4.0,2.1,2.025,2.00625,1e-10,1e-10,1e-10);
  const auto shared_fd_both_mismatch=covariance_compare_directional_curvature(5.0,2.1,2.025,2.00625,1e-10,1e-10,1e-10);
  const auto shared_fd_unresolved=covariance_compare_directional_curvature(2.0,1e-12,2e-12,3e-12,1e-8,1e-8,1e-8);
  assert(shared_fd_primary.resolved&&!shared_fd_independent.resolved&&!shared_fd_both_mismatch.resolved);
  assert(shared_fd_primary.extrapolated==shared_fd_independent.extrapolated&&
    shared_fd_independent.extrapolated==shared_fd_both_mismatch.extrapolated);
  assert(shared_fd_unresolved.status=="DIFFERENCE_BELOW_RESOLUTION");
  CovarianceDirectionalCurvatureSummary empty_fd_classification;
  assert(covariance_failure_fd_status(empty_fd_classification,3)=="FD_ONLY_NO_COMPLETED_DIRECTIONS");
  CovarianceDirectionalCurvatureSummary both_fd_matches;both_fd_matches.status="DIAGNOSTIC_COMPLETED";
  both_fd_matches.completed_directions=3;both_fd_matches.primary_matches=3;both_fd_matches.comparison_matches=3;
  assert(covariance_failure_fd_status(both_fd_matches,3)=="FD_ONLY_BOTH_MATCH");
  CovarianceDirectionalCurvatureSummary partial_fd_classification;partial_fd_classification.completed_directions=1;
  partial_fd_classification.primary_matches=1;partial_fd_classification.comparison_matches=1;
  assert(covariance_failure_fd_status(partial_fd_classification,3)=="FD_ONLY_PARTIAL_BOTH_MATCH");
  CovarianceDirectionalCurvatureSummary fd_classification;fd_classification.completed_directions=3;fd_classification.primary_matches=3;
  fd_classification.comparison_matches=2;fd_classification.comparison_checks=3;
  assert(covariance_failure_fd_status(fd_classification,3)=="FD_ONLY_PRIMARY_MATCH_INDEPENDENT_MISMATCH");
  fd_classification.primary_matches=1;fd_classification.comparison_matches=1;
  assert(covariance_failure_fd_status(fd_classification,3)=="FD_ONLY_BOTH_MISMATCH");
  fd_classification.unresolved_checks=1;
  assert(covariance_failure_fd_status(fd_classification,3)=="FD_ONLY_UNRESOLVED_DIFFERENCE_BELOW_RESOLUTION");
  const double q=0.001/0.9,analytic=1.0/(q*q);
  const auto log_barrier_second=[&](const double h){return -((std::log(q+h)-2.0*std::log(q))+std::log(q-h))/(h*h);};
  const auto boundary_unstable=covariance_compare_directional_curvature(analytic,log_barrier_second(1.0e-3),
    log_barrier_second(5.0e-4),log_barrier_second(2.5e-4),0.0,0.0,0.0);
  const auto boundary_stable=covariance_compare_directional_curvature(analytic,log_barrier_second(5.0e-4),
    log_barrier_second(2.5e-4),log_barrier_second(1.25e-4),0.0,0.0,0.0);
  assert(boundary_unstable.status=="STEP_HALVING_UNSTABLE");
  assert(boundary_stable.resolved&&boundary_stable.status=="PASS");
}

void test_covariance_independent_failure_diagnostic_never_commits()
{
  const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  auto assert_injected_gate_errors_exceed_tolerance=[](const std::string& log){
    std::size_t position=0;int checked=0;
    while((position=log.find("COVARIANCE_CURVATURE_GATE_DIRECTION",position))!=std::string::npos){
      const std::size_t end=log.find('\n',position);const std::string line=log.substr(position,end-position);
      const std::size_t error_position=line.find("relative_error=");assert(error_position!=std::string::npos);
      std::istringstream error_stream(line.substr(error_position+std::string("relative_error=").size()));
      double error=0.0;assert(error_stream>>error&&error>covariance_independent_directional_tolerance);
      ++checked;position=end==std::string::npos?log.size():end+1;}
    assert(checked>=3);};
  for(const std::uint64_t fd_limit:{0ULL,64ULL}){
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.independent_curvature_bias_remaining=100;
    covariance_fit_test_hooks.independent_curvature_diagonal_bias=1.0e50;
    covariance_fit_test_hooks.failure_fd_objective_limit=fd_limit;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,
      false,false,true,false,{}, {},0.0,16);
    const int failure_fd_runs=covariance_fit_test_hooks.failure_fd_runs;covariance_fit_test_hooks={};
    assert(failure_fd_runs==1&&fit.trust_region_enabled&&fit.status=="CURVATURE_UNRESOLVED"&&fit.iterations==0);
    assert(fit.last_valid.available&&fit.last_valid.theta==fit.theta);
    assert(fit.analytic_curvature_direction_status.rfind("DIAGNOSTIC_ONLY_",0)==0&&
      fit.analytic_curvature_direction_seconds>0.0);
    const std::string log=trace.str();const std::size_t summary_at=log.find("COVARIANCE_CURVATURE_FAILURE_FD_SUMMARY");
    assert(summary_at!=std::string::npos&&log.find("diagnostic_only=1",summary_at)!=std::string::npos&&
      log.find("acceptance_gate=NONE",summary_at)!=std::string::npos);
    const std::string summary_line=log.substr(summary_at,log.find('\n',summary_at)-summary_at);
    if(fd_limit==0){
      assert(summary_line.find("status=FD_ONLY_DIAGNOSTIC_OBJECTIVE_LIMIT")!=std::string::npos&&
        summary_line.find("objective_evaluations=0")!=std::string::npos&&summary_line.find("objective_ceiling=0")!=std::string::npos);
      assert(fit.analytic_curvature_direction_checks==0&&fit.analytic_curvature_direction_objectives==0&&
        fit.analytic_curvature_direction_status=="DIAGNOSTIC_ONLY_FD_ONLY_DIAGNOSTIC_OBJECTIVE_LIMIT");
    }else{
      assert(summary_line.find("status=FD_ONLY_DIAGNOSTIC_OBJECTIVE_LIMIT")==std::string::npos&&
        summary_line.find("objective_evaluations=0")==std::string::npos&&summary_line.find("directions_completed=0")==std::string::npos);
      assert(fit.analytic_curvature_direction_checks>0&&fit.analytic_curvature_direction_objectives>0&&
        fit.analytic_curvature_direction_objectives<=64&&fit.analytic_curvature_direction_status!="PASS");
      assert(log.find("relative_denominator=")!=std::string::npos&&
        log.find("resolution_tolerance=")!=std::string::npos&&
        log.find("allowed_absolute_error=")!=std::string::npos);
    }
    assert(log.find("COVARIANCE_CURVATURE_NESTED_LEVEL")!=std::string::npos&&
      log.find("COVARIANCE_CURVATURE_GATE_DIRECTION")!=std::string::npos&&
      log.find("name=PROPOSED_TR_STEP")!=std::string::npos&&log.find("denominator=")!=std::string::npos);
    assert_injected_gate_errors_exceed_tolerance(log);
    assert(log.find("TRUST_REGION_STEP_ACCEPT ")==std::string::npos);
    const std::uint64_t factors_before=system.factorizations;
    assert(system.ensure_factors(fit.theta,1e-6)&&system.factorizations==factors_before);
  }
}

void test_covariance_gradient_residual_reject_retries_before_commit()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={-1.0};CovarianceFitResult fit;
  fit.theta=theta;fit.iterations=0;fit.last_valid.available=true;fit.last_valid.theta=theta;fit.last_valid.objective=0.0;
  CovarianceBFGSHistory bfgs;covariance_reset_bfgs_history(bfgs,theta.size(),true);const std::vector<double> initial_hessian=bfgs.inverse_hessian;
  std::ostringstream trace;int objective_calls=0,gradient_calls=0;
  auto evaluate=[&](const std::vector<double>& trial,std::uint64_t& evaluations){++objective_calls;++evaluations;
    if(objective_calls==2)assert(trace.str().find("status=GRADIENT_RESIDUAL_REJECT")!=std::string::npos);
    CovarianceObjectiveEvaluation value;value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;
    value.objective=trial[0]+0.5*trial[0]*trial[0];return value;};
  auto complete=[&](CovarianceObjectiveEvaluation& value){++gradient_calls;
    if(gradient_calls==1){value.gradient={123.0};value.gradient_data={123.0};CovarianceTraceDiagnostics diagnostic;
      diagnostic.matrix_kind="D";diagnostic.status="RESIDUAL_FAIL";diagnostic.first_failure_batch_start=32;diagnostic.first_failure_columns=32;
      diagnostic.worst_factor_solve_relative_residual=2e-7;diagnostic.original_residual_checked=true;diagnostic.worst_original_solve_relative_residual=3e-7;
      throw CovarianceSolveResidualError(diagnostic);}
    value.gradient={0.5};value.gradient_data={0.5};value.gradient_prior={0.0};value.gradient_barrier={0.0};};
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,100,fit,0,0.5,evaluate,complete,&trace);
  assert(line.accepted&&!line.step_underflow&&line.trials==2&&line.alpha==0.5&&line.step.size()==1&&line.step[0]==-0.5);
  assert(objective_calls==2&&gradient_calls==2&&fit.objective_evaluations==2&&fit.line_search_trials==2&&fit.line_search_evaluations==2);
  assert(fit.gradient_residual_reject_trials==1&&line.gradient_residual_rejected);
  assert(fit.theta==theta&&fit.iterations==0&&fit.last_valid.theta==theta&&bfgs.inverse_hessian==initial_hessian);
  assert(line.failed_d_trace.status=="RESIDUAL_FAIL"&&line.failed_d_trace.first_failure_batch_start==32);
  const std::string trace_text=trace.str();const std::size_t rejected=trace_text.find("GRADIENT_RESIDUAL_REJECT iteration=0 trial=1 alpha=1 tau=0.5 matrix_kind=D first_failure_batch_start=32");
  assert(rejected!=std::string::npos&&trace_text.find("worst_factor_residual=2e-07",rejected)!=std::string::npos);
  assert(trace_text.find("worst_original_residual=3e-07",rejected)!=std::string::npos&&trace_text.find("STEP_COMMITTED")==std::string::npos);
}

void test_covariance_directional_overshoot_rejects_before_commit()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={1.0};CovarianceFitResult fit;
  fit.theta=theta;fit.iterations=0;fit.last_valid.available=true;fit.last_valid.theta=theta;fit.last_valid.objective=0.0;
  CovarianceBFGSHistory bfgs;covariance_reset_bfgs_history(bfgs,theta.size(),true);const std::vector<double> initial_hessian=bfgs.inverse_hessian;
  std::ostringstream trace;int objective_calls=0,gradient_calls=0;double evaluated_x=0.0;
  auto evaluate=[&](const std::vector<double>& trial,std::uint64_t& evaluations){++objective_calls;++evaluations;evaluated_x=trial[0];CovarianceObjectiveEvaluation value;
    value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-trial[0]+0.75*trial[0]*trial[0];return value;};
  auto complete=[&](CovarianceObjectiveEvaluation& value){++gradient_calls;value.gradient={-1.0+1.5*evaluated_x};};
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,100,fit,0,0.5,evaluate,complete,&trace);
  assert(line.accepted&&!line.step_underflow&&line.trials==2&&line.alpha==0.5);
  assert(objective_calls==2&&gradient_calls==2&&fit.objective_evaluations==2&&fit.line_search_trials==2&&fit.line_search_evaluations==2);
  assert(fit.directional_overshoot_reject_trials==1&&fit.theta==theta&&fit.iterations==0&&fit.last_valid.theta==theta);
  assert(bfgs.inverse_hessian==initial_hessian);
  const std::string trace_text=trace.str();assert(trace_text.find("alpha=1 tau=0.5 current_objective=0 candidate_objective=-0.25 current_directional_slope=-1 candidate_directional_slope=0.5 status=DIRECTIONAL_OVERSHOOT_REJECT")!=std::string::npos);
  assert(trace_text.find("reason=POSITIVE_SLOPE_NOT_ALLOWED absolute_slope_ratio=0.5")!=std::string::npos);
  assert(trace_text.find("source=HALF_FALLBACK proposed_alpha=0.5 shrink_ratio=0.5")!=std::string::npos);
  assert(trace_text.find("alpha=0.5 tau=0.5 current_objective=0 candidate_objective=-0.3125 current_directional_slope=-1 candidate_directional_slope=-0.25 status=DIRECTIONAL_ACCEPTED")!=std::string::npos);

  CovarianceFitResult budget;budget.theta=theta;budget.iterations=0;budget.last_valid.available=true;budget.last_valid.theta=theta;
  constexpr double eps=0.005,delta=0.01;
  const CovarianceBacktrackingResult rejected=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,3,budget,0,0.5,
    [&](const std::vector<double>& trial,std::uint64_t& evaluations){++evaluations;evaluated_x=trial[0];CovarianceObjectiveEvaluation value;
      value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;
      value.objective=eps*evaluated_x-(1.0+eps)*delta*(1.0-std::exp(-evaluated_x/delta));return value;},
    [&](CovarianceObjectiveEvaluation& value){value.gradient={eps-(1.0+eps)*std::exp(-evaluated_x/delta)};});
  assert(!rejected.accepted&&rejected.evaluation_budget_exhausted&&rejected.trials==3);
  assert(budget.directional_overshoot_reject_trials==3&&budget.failed_trial.available);
  assert(budget.failed_trial.failure_stage=="DIRECTIONAL_OVERSHOOT_REJECT"&&budget.failed_trial.alpha==0.25);
  assert(budget.theta==theta&&budget.iterations==0&&budget.last_valid.theta==theta);

  const std::string path="rpmd_ja_native_covariance_overshoot_state_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=1e-6;options.fd_step=1e-4;options.sample_interval=1;
  SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};Graph graph=make_two_atom_covariance_graph();
  write_covariance_state(path,budget,0.1,"test.samples","test.qraw",options,header,identity,graph,4,2);
  std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
  assert(fields.at("directional_overshoot_reject_trials")=="3");

  for(const bool wrong_dimension:{false,true}){CovarianceFitResult fatal;bool propagated=false;
    try{covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fatal,0,0.5,
        [](const std::vector<double>&,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
          value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-0.5;return value;},
        [wrong_dimension](CovarianceObjectiveEvaluation& value){value.gradient=wrong_dimension?std::vector<double>{1.0,2.0}:
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()};},nullptr,true,false);
    }catch(const std::exception&){propagated=true;}
    assert(propagated&&fatal.directional_overshoot_reject_trials==0&&fatal.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");
    assert(fatal.iterations==0&&!fatal.last_valid.available&&fatal.theta.empty());
  }
}

void test_covariance_joint_bfgs_rejects_positive_slope_and_uses_secant_shrink()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={1.0};
  for(const auto& item:std::vector<std::pair<double,double>>{{1.8,0.5}}){
    CovarianceFitResult fit;std::ostringstream trace;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.5*item.first*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){value.gradient={-1.0+item.first*evaluated_x};},&trace,false,false,true);
    const std::string log=trace.str();
    assert(line.accepted&&line.trials==2&&line.alpha==item.second&&fit.directional_overshoot_reject_trials==1);
    assert(log.find("status=DIRECTIONAL_OVERSHOOT_REJECT reason=POSITIVE_SLOPE_NOT_ALLOWED")!=std::string::npos);
    std::ostringstream expected_proposal;expected_proposal<<"source=DIRECTIONAL_SECANT proposed_alpha="<<item.second<<" shrink_ratio="<<item.second;
    assert(log.find(expected_proposal.str())!=std::string::npos);
    assert(log.find("status=DIRECTIONAL_ACCEPTED reason=NONPOSITIVE_SLOPE")!=std::string::npos);
  }
}

void test_covariance_joint_logdet_target_rejects_positive_slopes()
{
  const std::vector<double> theta={0.0,0.0},z=theta,scale={1.0,1.0},direction={1.0,0.001};
  auto terms=[](const std::vector<double>& point){
    const double a=0.01+point[0]+point[1],b=0.1*point[1],c=1.0+point[0]+100.0*point[1];
    const double da=a+0.1,dc=c+0.1,det_s=a*c-b*b,det_d=da*dc-b*b;
    return std::array<double,7>{a,b,c,det_s,det_d,da,dc};
  };
  auto objective=[&](const std::vector<double>& point){const auto m=terms(point);
    if(!(m[0]>0.0&&m[5]>0.0&&m[3]>0.0&&m[4]>0.0))return std::numeric_limits<double>::infinity();
    return 10.0*point[0]+20.0*point[1]-0.55*std::log(m[4])-std::log(m[3]);};
  auto gradient=[&](const std::vector<double>& point){const auto m=terms(point);
    const double tr_d0=(m[5]+m[6])/m[4],tr_s0=(m[0]+m[2])/m[3];
    const double tr_d1=(m[6]+100.0*m[5]-0.2*m[1])/m[4],tr_s1=(m[2]+100.0*m[0]-0.2*m[1])/m[3];
    return std::vector<double>{10.0-0.55*tr_d0-tr_s0,20.0-0.55*tr_d1-tr_s1};
  };
  const double slope=covariance_dot(gradient(theta),direction),h=1e-6;
  std::vector<double> plus=theta,minus=theta;for(std::size_t i=0;i<plus.size();++i){plus[i]=h*direction[i];minus[i]=-h*direction[i];}
  assert(std::abs((objective(plus)-objective(minus))/(2.0*h)-slope)<1e-8*std::abs(slope));
  check_close(slope,-96.735,1e-12);

  CovarianceFitResult fit;std::ostringstream trace;std::vector<double> evaluated;
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,objective(theta),slope,1.0,1.0,16,fit,0,0.5,
    [&](const std::vector<double>& point,std::uint64_t& evaluations){++evaluations;evaluated=point;CovarianceObjectiveEvaluation value;
      value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=objective(point);return value;},
    [&](CovarianceObjectiveEvaluation& value){value.gradient=gradient(evaluated);},&trace,true,false,true);
  assert(line.accepted&&line.trials==4&&line.alpha==0.125&&fit.directional_overshoot_reject_trials==2);
  const std::string log=trace.str();
  std::istringstream lines(log);std::string line_text;std::vector<double> candidate_slopes;
  while(std::getline(lines,line_text)){const std::string key="candidate_directional_slope=";const std::size_t position=line_text.find(key);
    if(position!=std::string::npos)candidate_slopes.push_back(std::stod(line_text.substr(position+key.size())));}
  assert(candidate_slopes.size()==3&&candidate_slopes[0]>6.0&&candidate_slopes[0]<6.2&&candidate_slopes[1]>3.2&&candidate_slopes[1]<3.5&&
    candidate_slopes[2]<0.0&&candidate_slopes[2]>-1.3);
  assert(log.find("source=DIRECTIONAL_SECANT proposed_alpha=0.25 shrink_ratio=0.5")!=std::string::npos);
  assert(log.find("source=DIRECTIONAL_SECANT proposed_alpha=0.125 shrink_ratio=0.5")!=std::string::npos);
}

void test_covariance_gradient_residual_budget_keeps_last_valid_and_reject()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={-1.0};CovarianceFitResult fit;
  fit.theta=theta;fit.iterations=0;fit.last_valid.available=true;fit.last_valid.theta=theta;fit.last_valid.objective=0.0;
  CovarianceBFGSHistory bfgs;covariance_reset_bfgs_history(bfgs,theta.size(),true);const std::vector<double> initial_hessian=bfgs.inverse_hessian;
  std::ostringstream trace;int objective_calls=0;
  auto evaluate=[&](const std::vector<double>& trial,std::uint64_t& evaluations){++objective_calls;++evaluations;CovarianceObjectiveEvaluation value;
    value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=trial[0]+0.5*trial[0]*trial[0];return value;};
  auto complete=[&](CovarianceObjectiveEvaluation& value){value.gradient={456.0};value.gradient_prior={456.0};CovarianceTraceDiagnostics diagnostic;
    diagnostic.matrix_kind="D_MINUS_EPSILON_I";diagnostic.status="RESIDUAL_FAIL";diagnostic.first_failure_batch_start=16*objective_calls;
    diagnostic.first_failure_columns=16;diagnostic.worst_factor_solve_relative_residual=4e-7;throw CovarianceSolveResidualError(diagnostic);};
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,3,fit,0,0.5,evaluate,complete,&trace,true,false);
  assert(!line.accepted&&!line.step_underflow&&line.evaluation_budget_exhausted&&line.trials==3);
  assert(objective_calls==3&&fit.objective_evaluations==3&&fit.gradient_residual_reject_trials==3);
  assert(!fit.converged&&fit.theta==theta&&fit.iterations==0&&fit.last_valid.theta==theta&&bfgs.inverse_hessian==initial_hessian);
  assert(fit.failed_trial.available&&fit.failed_trial.failure_stage=="GRADIENT_RESIDUAL_REJECT"&&fit.failed_trial.alpha==0.25);
  assert(fit.failed_shifted_trace.first_failure_batch_start==48&&fit.failed_shifted_trace.matrix_kind=="D_MINUS_EPSILON_I");
  assert(line.next.gradient.empty()&&line.next.gradient_prior.empty());

  CovarianceFitResult cuda_failure;std::uint64_t before=0;
  bool propagated=false;
  try{covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,cuda_failure,0,0.5,
      [&](const std::vector<double>&,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;value.feasible=true;
        value.status=CovarianceObjectiveStatus::Feasible;value.objective=-0.5;return value;},
      [&](CovarianceObjectiveEvaluation&){throw std::runtime_error("simulated CUDA solve failure");});}
  catch(const std::runtime_error& error){propagated=std::string(error.what())=="simulated CUDA solve failure";}
  before=cuda_failure.gradient_residual_reject_trials;
  assert(propagated&&before==0&&cuda_failure.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");

  CovarianceFitResult nonfinite_failure;propagated=false;
  try{covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,nonfinite_failure,0,0.5,
      [&](const std::vector<double>&,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;value.feasible=true;
        value.status=CovarianceObjectiveStatus::Feasible;value.objective=-0.5;return value;},
      [&](CovarianceObjectiveEvaluation& value){value.gradient={std::numeric_limits<double>::quiet_NaN()};
        throw std::runtime_error("non-finite covariance gradient");});}
  catch(const std::runtime_error& error){propagated=std::string(error.what())=="non-finite covariance gradient";}
  assert(propagated&&nonfinite_failure.gradient_residual_reject_trials==0&&nonfinite_failure.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");
}

void test_covariance_armijo_bounded_expansion_preserves_last_valid_candidate()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={1.0};
  auto initialize=[](){CovarianceFitResult fit;fit.theta={0.0};fit.iterations=0;fit.last_valid.available=true;
    fit.last_valid.theta={0.0};fit.last_valid.objective=0.0;return fit;};
  {
    CovarianceFitResult fit=initialize();std::ostringstream trace;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [](CovarianceObjectiveEvaluation& value){value.gradient={-0.5};},&trace);
    assert(line.accepted&&line.trials==1&&line.alpha==1.0);
    assert(fit.objective_evaluations==1&&fit.theta==theta&&fit.last_valid.theta==theta);
  }
  {
    CovarianceFitResult fit=initialize();std::ostringstream trace;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){value.gradient={-1.0+0.1*evaluated_x};},&trace);
    assert(line.accepted&&line.alpha==4.0&&line.trials==3&&fit.objective_evaluations==3);
    check_close(line.next_theta[0],4.0,1e-15);check_close(line.next.objective,-3.2,1e-15);
    assert(trace.str().find("COVARIANCE_STEP_EXPAND iteration=0 from_alpha=1")!=std::string::npos);
    assert(trace.str().find("COVARIANCE_STEP_EXPAND iteration=0 from_alpha=2")!=std::string::npos);
    assert(fit.theta==theta&&fit.last_valid.theta==theta);
  }
  {
    CovarianceFitResult fit=initialize();std::ostringstream trace;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){value.gradient={-1.0+0.1*evaluated_x};},&trace,false);
    assert(line.accepted&&line.alpha==1.0&&line.trials==1&&fit.objective_evaluations==1);
    assert(trace.str().find("COVARIANCE_STEP_EXPAND")==std::string::npos);
  }
  {
    CovarianceFitResult fit=initialize();int calls=0;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,1,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++calls;value.gradient={-1.0+0.1*evaluated_x};});
    assert(line.accepted&&!line.evaluation_budget_exhausted&&line.alpha==1.0&&line.trials==1&&calls==1);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0,gradient_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++gradients;gradient_x=evaluated_x;value.gradient={-1.0+0.1*evaluated_x};},nullptr,true,true);
    assert(line.accepted&&line.alpha==4.0&&line.trials==3&&fit.line_search_evaluations==3&&fit.objective_evaluations==3);
    assert(gradients==1&&gradient_x==line.next_theta[0]&&line.next.objective==-3.2);
    assert(fit.line_search_gradient_calls==1&&fit.full_gradient_evaluations==fit.line_search_gradient_calls&&
      fit.line_search_objective_seconds>=0.0&&fit.line_search_gradient_seconds>=0.0);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0,gradient_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=x[0]<1.5;value.status=value.feasible?CovarianceObjectiveStatus::Feasible:CovarianceObjectiveStatus::ShiftedNotSpd;
        value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++gradients;gradient_x=evaluated_x;value.gradient={-1.0+0.1*evaluated_x};},nullptr,true,true);
    assert(line.accepted&&line.alpha==1.0&&line.trials==3&&fit.objective_evaluations==3);
    assert(gradients==1&&gradient_x==line.next_theta[0]);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0,gradient_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.4*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++gradients;gradient_x=evaluated_x;value.gradient={-1.0+0.8*evaluated_x};},nullptr,true,true);
    assert(line.accepted&&line.alpha==1.0&&line.trials==3&&fit.objective_evaluations==3);
    assert(gradients==1&&gradient_x==line.next_theta[0]);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0,gradient_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,1,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++gradients;value.gradient={-1.0+0.1*evaluated_x};},nullptr,true,true);
    assert(line.accepted&&line.alpha==1.0&&line.trials==1&&fit.objective_evaluations==1&&gradients==1);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0,gradient_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){if(++gradients==1){CovarianceTraceDiagnostics error;error.matrix_kind="D_MINUS_EPSILON_I";
          error.status="RESIDUAL_FAIL";error.worst_factor_solve_relative_residual=2e-7;throw CovarianceSolveResidualError(error);}
        gradient_x=evaluated_x;value.gradient={-1.0+0.1*evaluated_x};},nullptr,true,true);
    assert(line.accepted&&line.gradient_residual_rejected&&line.alpha==2.0&&line.trials==4);
    assert(fit.gradient_residual_reject_trials==1&&gradients==2&&fit.line_search_gradient_calls==2&&gradient_x==line.next_theta[0]);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,2,fit,0,0.5,
      [](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
        value.feasible=x[0]<1.5;value.status=value.feasible?CovarianceObjectiveStatus::Feasible:CovarianceObjectiveStatus::ShiftedNotSpd;
        value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation&){++gradients;},nullptr,true,true,true);
    assert(!line.accepted&&line.evaluation_budget_exhausted&&line.trials==2&&fit.objective_evaluations==2&&gradients==0);
    assert(fit.theta==theta&&fit.last_valid.theta==theta);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;bool propagated=false;
    try{(void)covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
        [](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
          value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
        [&](CovarianceObjectiveEvaluation& value){++gradients;value.gradient={std::numeric_limits<double>::quiet_NaN()};},nullptr,true,true);}
    catch(const std::runtime_error& error){propagated=std::string(error.what()).find("non-finite")!=std::string::npos;}
    assert(propagated&&gradients==1&&fit.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=x[0]<1.5;value.status=value.feasible?CovarianceObjectiveStatus::Feasible:CovarianceObjectiveStatus::ShiftedNotSpd;
        value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++gradients;value.gradient={-1.0+0.1*evaluated_x};});
    assert(line.accepted&&line.alpha==1.0&&line.trials==2&&fit.shifted_not_spd_trials==1&&gradients==1);
    check_close(line.next_theta[0],1.0,1e-15);
  }
  {
    CovarianceFitResult fit=initialize();int gradients=0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
      [](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){if(++gradients==1){value.gradient={-0.9};return;}CovarianceTraceDiagnostics error;
        error.matrix_kind="D_MINUS_EPSILON_I";error.status="RESIDUAL_FAIL";error.worst_factor_solve_relative_residual=2e-7;
        throw CovarianceSolveResidualError(error);});
    assert(line.accepted&&line.gradient_residual_rejected&&line.alpha==1.0&&line.trials==2);
    assert(fit.gradient_residual_reject_trials==1&&fit.failed_trial.failure_stage=="GRADIENT_RESIDUAL_REJECT");
    check_close(line.next_theta[0],1.0,1e-15);
  }
  for(const bool nonfinite:{false,true}){
    CovarianceFitResult fit=initialize();int gradients=0;bool propagated=false;
    try{covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,10,fit,0,0.5,
        [](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
          value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
        [&](CovarianceObjectiveEvaluation& value){if(++gradients==1){value.gradient={-0.9};return;}
          if(nonfinite)value.gradient={std::numeric_limits<double>::quiet_NaN()};else throw std::runtime_error("simulated CUDA failure");});}
    catch(const std::runtime_error& error){propagated=nonfinite?std::string(error.what()).find("non-finite")!=std::string::npos:
      std::string(error.what())=="simulated CUDA failure";}
    assert(propagated&&gradients==2&&fit.objective_evaluations==2&&fit.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");
  }
}

void test_covariance_bfgs_first_secant_seeds_and_updates_inverse_hessian()
{
  CovarianceBFGSHistory history;covariance_reset_bfgs_history(history,2,true);
  const std::vector<double> step={1.0,0.0},gradient_change={2.0,1.0};
  assert(covariance_update_inverse_hessian(history,step,gradient_change)==CovarianceBFGSUpdateStatus::Seeded);
  check_close(history.seed_gamma,0.4,1e-15);assert(history.seed_source=="FIRST_VALID_SECANT"&&history.stage_seeded);
  check_close(history.inverse_hessian[0],0.6,1e-15);check_close(history.inverse_hessian[1],-0.2,1e-15);
  check_close(history.inverse_hessian[2],-0.2,1e-15);check_close(history.inverse_hessian[3],0.4,1e-15);
  const std::vector<double> gradient={1.0,2.0};std::vector<double> direction(2,0.0);
  for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)direction[i]-=history.inverse_hessian[2*i+j]*gradient[j];
  assert(covariance_dot(gradient,direction)<0.0);check_close(covariance_dot(gradient,direction),-1.4,1e-15);
  assert(covariance_update_inverse_hessian(history,{1.0,0.0},{1.0,0.0})==CovarianceBFGSUpdateStatus::Updated);
  assert(history.seed_gamma==0.4&&history.seed_source=="FIRST_VALID_SECANT");check_close(history.last_valid_gamma,1.0,1e-15);
  covariance_reset_bfgs_history(history,2,false);assert(history.seed_source=="RECENT_VALID_GAMMA_RESET");
  check_close(history.seed_gamma,1.0,1e-15);check_close(history.inverse_hessian[0],1.0,1e-15);check_close(history.inverse_hessian[3],1.0,1e-15);
  covariance_reset_bfgs_history(history,2,true);assert(!history.stage_seeded&&history.seed_source=="UNIT_IDENTITY"&&history.seed_gamma==1.0);
}

void test_covariance_non_descent_reset_uses_restored_gamma()
{
  CovarianceBFGSHistory history;covariance_reset_bfgs_history(history,2,true);
  history.recent_secants.push_back({{1.0,0.0},{2.0,0.0}});
  history.stage_seeded=true;history.last_valid_gamma=0.25;history.inverse_hessian={-1.0,0.0,0.0,-1.0};
  const std::vector<double> gradient={4.0,8.0};std::vector<double> direction(2);
  for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)direction[i]-=history.inverse_hessian[2*i+j]*gradient[j];
  assert(!(covariance_dot(gradient,direction)<0.0));
  const double slope=covariance_reset_non_descent_direction(history,gradient,direction);
  assert(history.seed_gamma==0.25&&history.seed_source=="RECENT_VALID_GAMMA_RESET");
  assert(history.recent_secants.empty());
  assert(direction[0]==-1.0&&direction[1]==-2.0&&slope==-20.0&&covariance_norm_inf(direction)==2.0);
  const double alpha=std::min(1.0,1.0/covariance_norm_inf(direction));assert(alpha==0.5);
}

void test_covariance_recent_bfgs_replays_latest_gamma_and_bounds_history()
{
  CovarianceBFGSHistory history;covariance_reset_bfgs_history(history,2,true);
  const std::vector<double> first_step={1.0,0.0},first_change={2.0,0.0};
  const std::vector<double> latest_step={2.0,0.0},latest_change={1.0,0.0};
  assert(covariance_update_recent_inverse_hessian(history,first_step,first_change)==CovarianceBFGSUpdateStatus::Seeded);
  assert(covariance_update_recent_inverse_hessian(history,latest_step,latest_change)==CovarianceBFGSUpdateStatus::Updated);
  assert(history.recent_secants.size()==2&&history.seed_source=="RECENT_VALID_GAMMA_REPLAY");
  check_close(history.seed_gamma,2.0,1e-15);check_close(history.inverse_hessian[3],2.0,1e-15);

  const double coupled_hessian[2][2]={{4.0,1.0},{1.0,3.0}};
  const std::vector<std::vector<double>> committed_steps={{1.0,0.0},{1.0,1.0},{0.0,1.0},{1.0,-1.0}};
  const std::vector<std::pair<std::vector<double>,std::vector<double>>> coupled_secants=[&](){
    std::vector<std::pair<std::vector<double>,std::vector<double>>> secants;
    for(const std::vector<double>& step:committed_steps){std::vector<double> change(2);
      for(int i=0;i<2;++i)for(int j=0;j<2;++j)change[i]+=coupled_hessian[i][j]*step[j];
      secants.emplace_back(step,std::move(change));}
    return secants;}();
  covariance_reset_bfgs_history(history,2,true);
  int secant_index=0;
  for(const auto& secant:coupled_secants)
    assert(covariance_update_recent_inverse_hessian(history,secant.first,secant.second)==
      (secant_index++==0?CovarianceBFGSUpdateStatus::Seeded:CovarianceBFGSUpdateStatus::Updated));
  const double gamma=5.0/13.0;double reference[2][2]={{gamma,0.0},{0.0,gamma}};
  for(const auto& secant:coupled_secants){const auto& s=secant.first;const auto& y=secant.second;
    const double sy=s[0]*y[0]+s[1]*y[1],rho=1.0/sy;
    const double v[2][2]={{1.0-rho*s[0]*y[0],-rho*s[0]*y[1]},
      {-rho*s[1]*y[0],1.0-rho*s[1]*y[1]}};double vh[2][2]={{0.0,0.0},{0.0,0.0}},next[2][2];
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)for(int k=0;k<2;++k)vh[i][j]+=v[i][k]*reference[k][j];
    for(int i=0;i<2;++i)for(int j=0;j<2;++j){next[i][j]=rho*s[i]*s[j];
      for(int k=0;k<2;++k)next[i][j]+=vh[i][k]*v[j][k];}
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)reference[i][j]=next[i][j];}
  for(int i=0;i<2;++i)for(int j=0;j<2;++j)
    check_close(history.inverse_hessian[static_cast<std::size_t>(2*i+j)],reference[i][j],1e-14);
  assert(history.inverse_hessian[0]>0.0&&
    history.inverse_hessian[0]*history.inverse_hessian[3]-history.inverse_hessian[1]*history.inverse_hessian[2]>0.0);
  check_close(history.inverse_hessian[1],history.inverse_hessian[2],1e-14);
  check_close(3.0*history.inverse_hessian[0]-2.0*history.inverse_hessian[1],1.0,1e-14);
  check_close(3.0*history.inverse_hessian[2]-2.0*history.inverse_hessian[3],-1.0,1e-14);
  check_close(history.seed_gamma,gamma,1e-15);

  covariance_reset_bfgs_history(history,2,true);
  for(int i=0;i<10;++i)assert(covariance_update_recent_inverse_hessian(history,{1.0,0.0},{1.0+0.1*i,0.0})==
    (i==0?CovarianceBFGSUpdateStatus::Seeded:CovarianceBFGSUpdateStatus::Updated));
  assert(history.recent_secants.size()==covariance_bfgs_recent_history_limit);
  check_close(history.recent_secants.front().gradient_change[0],1.2,1e-15);
  check_close(history.recent_secants.back().gradient_change[0],1.9,1e-15);
  const double valid_gamma=history.last_valid_gamma;
  assert(covariance_update_recent_inverse_hessian(history,{1.0,0.0},{-1.0,0.0})==CovarianceBFGSUpdateStatus::Reset);
  assert(history.recent_secants.empty()&&history.seed_source=="RECENT_VALID_GAMMA_RESET");
  check_close(history.seed_gamma,valid_gamma,1e-15);

  covariance_update_recent_inverse_hessian(history,{1.0,0.0},{1.0,0.0});
  covariance_reset_bfgs_history(history,2,true);
  assert(history.recent_secants.empty()&&!history.stage_seeded&&std::isnan(history.last_valid_gamma));

  covariance_update_recent_inverse_hessian(history,{1.0,0.0},{1.0,0.0});
  const std::size_t before=history.recent_secants.size();bool nonfinite_rejected=false;
  try{covariance_update_recent_inverse_hessian(history,{std::numeric_limits<double>::quiet_NaN(),0.0},{1.0,0.0});}
  catch(const std::runtime_error& error){nonfinite_rejected=std::string(error.what()).find("non-finite")!=std::string::npos;}
  assert(nonfinite_rejected&&history.recent_secants.size()==before);
  history.inverse_hessian[0]=std::numeric_limits<double>::quiet_NaN();nonfinite_rejected=false;
  try{covariance_update_recent_inverse_hessian(history,{1.0,0.0},{1.0,0.0});}
  catch(const std::runtime_error& error){nonfinite_rejected=std::string(error.what()).find("non-finite")!=std::string::npos;}
  assert(nonfinite_rejected&&history.recent_secants.size()==before);

  covariance_reset_bfgs_history(history,2,true);
  assert(covariance_update_recent_inverse_hessian(history,{1e100,0.0},{1e100,0.0})==CovarianceBFGSUpdateStatus::Seeded);
  check_close(history.seed_gamma,1.0,1e-15);assert(std::all_of(history.inverse_hessian.begin(),history.inverse_hessian.end(),[](double value){return std::isfinite(value);}));
}

void test_covariance_objective_only_overshoot_uses_directional_secant()
{
  const std::vector<double> theta={0.0},z={0.0},scale={1.0},direction={1.0};
  CovarianceFitResult fit;fit.theta=theta;double evaluated_x=0.0;int objective_calls=0,gradient_calls=0;
  constexpr double curvature=2.0/7.0;std::ostringstream trace;
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,20,fit,0,0.5,
    [&](const std::vector<double>& x,std::uint64_t& evaluations){++objective_calls;++evaluations;evaluated_x=x[0];
      CovarianceObjectiveEvaluation value;value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;
      value.objective=-x[0]+0.5*curvature*x[0]*x[0];return value;},
    [&](CovarianceObjectiveEvaluation& value){++gradient_calls;value.gradient={-1.0+curvature*evaluated_x};},&trace,true,true);
  assert(line.accepted&&line.trials==4);check_close(line.alpha,3.5,1e-14);check_close(line.next_theta[0],3.5,1e-14);
  assert(objective_calls==4&&gradient_calls==2&&fit.objective_evaluations==4&&fit.line_search_evaluations==4);
  assert(fit.line_search_gradient_calls==2&&trace.str().find("source=DIRECTIONAL_SECANT proposed_alpha=3.5 shrink_ratio=0.875")!=std::string::npos);

  for(const bool nonfinite:{false,true}){CovarianceFitResult failed;int gradients=0;bool propagated=false;
    try{(void)covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,20,failed,0,0.5,
        [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;CovarianceObjectiveEvaluation value;
          value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.5*curvature*x[0]*x[0];evaluated_x=x[0];return value;},
        [&](CovarianceObjectiveEvaluation& value){++gradients;if(nonfinite)value.gradient={std::numeric_limits<double>::quiet_NaN()};
          else throw std::runtime_error("simulated CUDA failure");},nullptr,true,true);}
    catch(const std::runtime_error& error){propagated=nonfinite?std::string(error.what()).find("non-finite")!=std::string::npos:
      std::string(error.what())=="simulated CUDA failure";}
    assert(propagated&&gradients==1&&failed.objective_evaluations==3&&failed.line_search_gradient_calls==1);
    assert(failed.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");}

  CovarianceFitResult fallback;int fallback_gradients=0;double fallback_x=0.0;std::ostringstream fallback_trace;
  const double largest=std::numeric_limits<double>::max();
  const CovarianceBacktrackingResult fallback_line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-largest,1.0,1.0,10,fallback,0,0.5,
    [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;fallback_x=x[0];CovarianceObjectiveEvaluation value;
      value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-2e304*x[0];return value;},
    [&](CovarianceObjectiveEvaluation& value){++fallback_gradients;value.gradient={fallback_x>3.0?largest:-largest};},&fallback_trace,true,true);
  assert(fallback_line.accepted&&fallback_line.trials==4&&fallback_line.alpha==2.0);
  assert(fallback_gradients==2&&fallback.objective_evaluations==4&&fallback.line_search_gradient_calls==2);
  assert(fallback_trace.str().find("source=HALF_FALLBACK proposed_alpha=2")!=std::string::npos);
}

void test_covariance_solver_residual_retry_commits_once()
{
  for(const bool block_newton:{false,true}){
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
    const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=1;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,block_newton);
    const std::vector<double> rejected_theta=covariance_fit_test_hooks.last_residual_rejected_theta;covariance_fit_test_hooks={};
    const std::string log=trace.str();const std::size_t rejected=log.find("GRADIENT_RESIDUAL_REJECT iteration=0 ");
    const std::size_t committed=log.find("STEP_COMMITTED iteration=0");
    assert(rejected!=std::string::npos&&committed!=std::string::npos&&rejected<committed);
    assert(fit.gradient_residual_reject_trials==1&&fit.iterations==1&&fit.theta==fit.last_valid.theta);
    assert(!rejected_theta.empty()&&rejected_theta!=fit.theta);
    assert(!fit.failed_trial.available&&fit.failed_trial.failure_stage=="NONE");
    assert(fit.block_newton_enabled==block_newton);
    if(block_newton)assert(log.find("COVARIANCE_DIRECTION iteration=0")!=std::string::npos&&fit.block_newton_factorizations==2);
  }
}

void test_covariance_solver_residual_budget_preserves_state_points()
{
  for(const bool block_newton:{false,true}){
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
    const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=64;
    covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=16;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,block_newton);
    const std::vector<double> rejected_theta=covariance_fit_test_hooks.last_residual_rejected_theta;covariance_fit_test_hooks={};
    assert(fit.status=="EVALUATION_BUDGET_EXHAUSTED"&&!fit.converged&&fit.iterations==0);
    assert(fit.gradient_residual_reject_trials>0&&fit.last_valid.available&&fit.failed_trial.available);
    assert(fit.failed_trial.failure_stage=="GRADIENT_RESIDUAL_REJECT"&&fit.last_valid.theta==fit.theta);
    assert(fit.failed_trial.theta!=fit.last_valid.theta&&rejected_theta==fit.failed_trial.theta);
    assert(fit.block_newton_enabled==block_newton);
    if(block_newton)assert(trace.str().find("COVARIANCE_DIRECTION iteration=0")!=std::string::npos&&fit.block_newton_factorizations==2);

    const std::string path="rpmd_ja_native_covariance_reject_state_test_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
    RpmdJANativeFitOptions options;options.cutoff=5.0;options.epsilon=1e-6;options.fd_step=1e-4;options.sample_interval=1;options.covariance_block_newton=block_newton;
    SampleHeader header{};header.temperature=300.0;header.beads=4;QrawIdentity identity{};
    write_covariance_state(path,fit,0.1,"test.samples","test.qraw",options,header,identity,graph,4,2);
    std::ifstream state_file(path);const auto fields=read_covariance_state_fields(state_file);
    assert(fields.at("status")=="EVALUATION_BUDGET_EXHAUSTED");
    assert(fields.at("block_newton_enabled")== (block_newton?"1":"0"));
    assert(fields.at("optimizer_step_mode")== (block_newton?"BLOCK_NEWTON_OPT_IN":"BFGS_DEFAULT"));
    assert(fields.at("last_valid_available")=="1"&&fields.at("failed_trial_available")=="1");
    assert(read_snapshot_vector(read_test_file(path),"last_valid_theta")==fit.last_valid.theta);
    assert(read_snapshot_vector(read_test_file(path),"failed_trial_theta")==fit.failed_trial.theta);
    assert(fit.failed_trial.theta!=fit.last_valid.theta);
  }
}

void test_covariance_fixed_coordinate_fd_uses_selected_coordinate_scale()
{
  const std::vector<double> theta={0.437,46.718},scale={1.0,1.0},z=theta;std::ostringstream report;
  const double expected_step=std::cbrt(std::numeric_limits<double>::epsilon())*std::max(1.0,std::abs(z[0]));
  const CovarianceFixedCoordinateFDResult result=covariance_fixed_coordinate_fd(theta,z,scale,0,-1.0,0.0,
    [](const std::vector<double>& trial){CovarianceObjectiveEvaluation value;value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;
      const double delta=trial[0]-0.437;value.objective=-delta+0.5*delta*delta;return value;},&report);
  assert(result.initial_step==expected_step&&result.initial_step<std::cbrt(std::numeric_limits<double>::epsilon())*std::abs(z[1]));
  assert(result.evaluations>0&&result.evaluations<=16&&result.status=="CONSISTENT"&&result.one_percent_check=="PASS");
  assert(report.str().find("directional_fd_probe side=POSITIVE")!=std::string::npos);
}

void test_covariance_fixed_coordinate_fd_switches_feasible_side_with_shared_budget()
{
  const std::vector<double> theta={0.0},scale={1.0},z={0.0};const double initial_step=std::cbrt(std::numeric_limits<double>::epsilon());
  std::vector<double> evaluated_coordinates;std::ostringstream report;
  const CovarianceFixedCoordinateFDResult result=covariance_fixed_coordinate_fd(theta,z,scale,0,-1.0,0.0,
    [&](const std::vector<double>& trial){CovarianceObjectiveEvaluation value;const double delta=trial[0];evaluated_coordinates.push_back(delta);
      if(delta>0.0&&delta<initial_step/3.0){value.status=CovarianceObjectiveStatus::ShiftedNotSpd;return value;}
      value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-delta;return value;},&report);
  assert(result.evaluations==6&&result.evaluations<=16&&evaluated_coordinates.size()==result.evaluations);
  assert(evaluated_coordinates[0]>0.0&&evaluated_coordinates[1]>0.0&&evaluated_coordinates[2]>0.0);
  check_close(evaluated_coordinates[2],initial_step/4.0,1e-15);check_close(evaluated_coordinates[3],-initial_step/4.0,1e-15);
  assert(result.side==-1&&result.stable_halvings==2&&result.status=="CONSISTENT"&&result.one_percent_check=="PASS");
  const std::string trace=report.str();const std::size_t switched=trace.find("directional_fd_probe side=NEGATIVE");
  assert(switched!=std::string::npos&&trace.substr(switched,trace.find('\n',switched)-switched).find("stable_halvings=0 status=FEASIBLE")!=std::string::npos);
}

void test_covariance_near_boundary_fd_requires_stable_steps()
{
  constexpr double epsilon=0.01,gap=5e-10,tau=1.0;
  DeviceBaseline baseline;baseline.d=6;assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),36*sizeof(double))==cudaSuccess);
  const double t[2]={1.0/std::sqrt(5.0),2.0/std::sqrt(5.0)};std::vector<double> matrix(36,0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<2;++i)for(int j=0;j<2;++j){const int row=axis*2+i,column=axis*2+j;
    matrix[static_cast<std::size_t>(row)*6+column]=(epsilon+gap)*((i==j?1.0:0.0)-t[i]*t[j]);}
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  const Graph graph=make_two_atom_covariance_graph();const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);const std::vector<double> theta(6,0.0),statistics(6,0.0);
  CovarianceBaseline reference;reference.theta=theta;
  assert(system.evaluate_factors(theta,epsilon,reference.logdet,reference.shifted_logdet,&reference.trace));
  std::uint64_t evaluations=0;const CovarianceObjectiveEvaluation current=evaluate_covariance_fit_objective(system,theta,reference,statistics,0.0,epsilon,tau,evaluations);
  assert(current.feasible);const std::vector<double> scale=system.coordinate_scales(theta,epsilon),g=covariance_scaled_gradient(current.gradient,scale);
  std::size_t index=0;for(std::size_t i=1;i<g.size();++i)if(std::abs(g[i])>std::abs(g[index]))index=i;
  const double slope=g[index];assert(std::isfinite(slope));std::vector<double> z(theta.size());for(std::size_t i=0;i<theta.size();++i)z[i]=theta[i]/scale[i];
  std::ostringstream report;const std::uint64_t before=evaluations;
  const CovarianceFixedCoordinateFDResult result=covariance_fixed_coordinate_fd(theta,z,scale,index,slope,current.objective,
    [&](const std::vector<double>& trial){return evaluate_covariance_fit_objective(system,trial,reference,statistics,0.0,epsilon,tau,evaluations,false);},&report);
  assert(result.initial_step==std::cbrt(std::numeric_limits<double>::epsilon())*std::max(1.0,std::abs(z[index])));
  assert(result.evaluations==evaluations-before&&result.evaluations>8&&result.evaluations<=16);
  assert(result.stable_halvings>=2&&result.status=="CONSISTENT"&&result.relative_error<=0.01&&result.one_percent_check=="PASS");
  assert(report.str().find("directional_fd_probe side=")!=std::string::npos);

  std::ostringstream curvature_trace;std::string curvature_failure;std::vector<double> empty_gram(36,0.0),zero_gradient(6,0.0);
  const std::uint64_t curvature_evaluations_before=evaluations;
  try{(void)covariance_verify_directional_curvature(system,current,reference,statistics,0.0,epsilon,tau,scale,0.0,
      empty_gram,zero_gradient,nullptr,evaluations,curvature_evaluations_before+8,nullptr,nullptr,&curvature_trace,"NEAR_BOUNDARY_TEST");}
  catch(const std::runtime_error& error){curvature_failure=error.what();}
  assert(curvature_failure.find("EVALUATION_BUDGET_EXHAUSTED")!=std::string::npos);
  assert(evaluations-curvature_evaluations_before==8);
  assert(curvature_trace.str().find("status=STEP_BOUNDARY")!=std::string::npos);
  assert(curvature_trace.str().find("shrink=1")!=std::string::npos);
  assert(curvature_trace.str().find("status=STEP_HALVING_UNSTABLE_RETRY")==std::string::npos);
  assert(system.factors_valid&&system.factor_theta==theta&&system.factor_epsilon==epsilon);

  std::vector<double> diagnostic_scale(6,1.0);diagnostic_scale[0]=std::numeric_limits<double>::denorm_min();
  const std::vector<std::pair<std::string,std::vector<double>>> diagnostic_directions={
    {"UNREPRESENTABLE",{1,0,0,0,0,0}},{"LATER_DIRECTION",{0,1,0,0,0,0}}};
  CovarianceDirectionalCurvatureSummary partial_summary;std::ostringstream diagnostic_trace;
  const std::uint64_t diagnostic_objectives_before=evaluations;
  const CovarianceDirectionalCurvatureSummary partial=covariance_verify_directional_curvature(system,current,reference,statistics,
    0.0,epsilon,tau,diagnostic_scale,0.0,empty_gram,zero_gradient,nullptr,evaluations,diagnostic_objectives_before+64,
    nullptr,nullptr,&diagnostic_trace,"FAILURE_FD_CONTINUE",true,&empty_gram,true,&diagnostic_directions,&partial_summary);
  assert(partial.direction_statuses.size()==2&&partial.direction_statuses[0].second=="UNRESOLVED_REPRESENTABILITY"&&
    partial.direction_statuses[1].first=="LATER_DIRECTION"&&partial.completed_directions==1&&partial.objectives>0);
  assert(partial.objectives==evaluations-diagnostic_objectives_before&&
    covariance_failure_fd_status(partial,2)=="FD_ONLY_UNRESOLVED_REPRESENTABILITY");
  assert(system.factors_valid&&system.factor_theta==current.cached_theta&&system.factor_epsilon==epsilon);

  const std::vector<std::pair<std::string,std::vector<double>>> malformed_directions={{"SHORT",{1.0}}};
  const std::uint64_t malformed_objectives_before=evaluations;bool malformed_direction_rejected=false;
  try{(void)covariance_verify_directional_curvature(system,current,reference,statistics,0.0,epsilon,tau,scale,0.0,
      empty_gram,zero_gradient,nullptr,evaluations,malformed_objectives_before+64,nullptr,nullptr,nullptr,
      "MALFORMED_FROZEN_DIRECTION",true,&empty_gram,true,&malformed_directions);}
  catch(const std::invalid_argument&){malformed_direction_rejected=true;}
  assert(malformed_direction_rejected&&evaluations==malformed_objectives_before);

  const std::vector<std::pair<std::string,std::vector<double>>> budget_direction={{"BUDGET",{0,1,0,0,0,0}}};
  covariance_fit_test_hooks={};covariance_fit_test_hooks.failure_fd_budget_errors_remaining=1;
  covariance_fit_test_hooks.failure_fd_budget_error_status="FEASIBILITY_BUDGET_INVALID_CLOCK";
  bool hard_budget_propagated=false;
  try{(void)covariance_verify_directional_curvature(system,current,reference,statistics,0.0,epsilon,tau,scale,0.0,
      empty_gram,zero_gradient,nullptr,evaluations,evaluations+64,nullptr,nullptr,nullptr,
      "HARD_BUDGET_PROPAGATION",true,&empty_gram,true,&budget_direction);}
  catch(const CovarianceFeasibilityBudgetError& error){hard_budget_propagated=std::string(error.what())=="FEASIBILITY_BUDGET_INVALID_CLOCK";}
  covariance_fit_test_hooks={};
  assert(hard_budget_propagated&&system.factors_valid&&system.factor_theta==current.cached_theta&&system.factor_epsilon==epsilon);

  const std::vector<std::pair<std::string,std::vector<double>>> budget_stopped_directions={
    {"BUDGET_STOP",{1,0,0,0,0,0}},{"NOT_RUN",{0,1,0,0,0,0}}};
  covariance_fit_test_hooks.failure_fd_budget_errors_remaining=1;
  covariance_fit_test_hooks.failure_fd_budget_error_status="FEASIBILITY_BUDGET_WALL_LIMIT";
  std::ostringstream budget_trace;const CovarianceDirectionalCurvatureSummary budget_stop=
    covariance_verify_directional_curvature(system,current,reference,statistics,0.0,epsilon,tau,scale,0.0,
      empty_gram,zero_gradient,nullptr,evaluations,evaluations+64,nullptr,nullptr,&budget_trace,
      "EXPECTED_BUDGET_STOP",true,&empty_gram,true,&budget_stopped_directions);
  covariance_fit_test_hooks={};
  assert(budget_stop.status=="FEASIBILITY_BUDGET_WALL_LIMIT"&&budget_stop.objectives==0&&
    budget_stop.direction_statuses.size()==2&&budget_stop.direction_statuses[0].second=="FEASIBILITY_BUDGET_STOP"&&
    budget_stop.direction_statuses[1].second=="NOT_EVALUATED_BUDGET");
  assert(budget_trace.str().find("name=NOT_RUN status=NOT_EVALUATED_BUDGET")!=std::string::npos&&
    system.factors_valid&&system.factor_theta==current.cached_theta&&system.factor_epsilon==epsilon);
}

void test_covariance_fixed_coordinate_fd_one_percent_is_separate_from_classification()
{
  const std::vector<double> theta={0.437},scale={1.0},z=theta;
  const CovarianceFixedCoordinateFDResult result=covariance_fixed_coordinate_fd(theta,z,scale,0,-1.03,0.0,
    [](const std::vector<double>& trial){CovarianceObjectiveEvaluation value;value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;
      const double delta=trial[0]-0.437;value.objective=-delta+0.5*delta*delta;return value;});
  assert(result.status=="CONSISTENT"&&result.relative_error>0.01&&result.relative_error<0.05&&result.one_percent_check=="FAIL");
}

void test_covariance_stable_directional_fd_consistency_and_mismatch()
{
  const double x=1.0,objective=0.5*x*x,analytic_slope=x;double h=0.1,previous=0.0,finite_difference=0.0;
  int stable_halvings=0;bool have_previous=false,stable=false;
  for(int probe=0;probe<8;++probe){const double candidate_objective=0.5*(x+h)*(x+h);
    finite_difference=(candidate_objective-objective)/h;
    double relative_change=0.0;if(have_previous&&covariance_directional_fd_stabilized(previous,finite_difference,stable_halvings,relative_change)){stable=true;break;}
    previous=finite_difference;have_previous=true;h*=0.5;
  }
  assert(stable&&stable_halvings>=2);
  double relative_error=0.0;
  assert(std::string(covariance_directional_fd_status(analytic_slope,finite_difference,stable,relative_error))=="CONSISTENT");
  assert(relative_error<0.01);
  assert(std::string(covariance_directional_fd_status(3.0,finite_difference,stable,relative_error))=="MISMATCH");
}

void test_covariance_inconclusive_diagnostic_preserves_failure_status()
{
  const char* failures[]={"LINE_SEARCH_FAILED","STEP_UNDERFLOW","EVALUATION_BUDGET_EXHAUSTED"};
  const char* line_search_failures[]={"MAX_BACKTRACKS","STEP_UNDERFLOW","EVALUATION_BUDGET_EXHAUSTED"};
  const char* diagnostics[]={"INCONCLUSIVE_UNSTABLE_FD","INCONCLUSIVE_STEP","INCONCLUSIVE_BUDGET"};
  for(std::size_t i=0;i<3;++i){CovarianceFitResult result;result.status=failures[i];result.line_search_failure_status=line_search_failures[i];result.reason="original failure";
    apply_covariance_directional_diagnostic(result,diagnostics[i]);
    assert(result.status==failures[i]&&result.line_search_failure_status==line_search_failures[i]&&result.reason=="original failure");}
  CovarianceFitResult mismatch;mismatch.status="LINE_SEARCH_FAILED";mismatch.line_search_failure_status="MAX_BACKTRACKS";
  apply_covariance_directional_diagnostic(mismatch,"MISMATCH");
  assert(mismatch.status=="GRADIENT_CONSISTENCY_SUSPECT"&&mismatch.line_search_failure_status=="MAX_BACKTRACKS");
}

void test_covariance_scaled_trial_round_trip_underflow()
{
  const std::vector<double> theta={0.7316913311810651},scale={0.000145},direction={1.0};
  const std::vector<double> z={theta[0]/scale[0]};const double small_step=std::numeric_limits<double>::epsilon()*std::abs(z[0])/4.0;
  assert(z[0]+small_step==z[0]&&scale[0]*z[0]!=theta[0]);
  std::vector<double> next_z,step,next_theta;std::uint64_t objective_evaluations=0;
  if(covariance_build_scaled_trial(theta,z,scale,direction,small_step,next_z,step,next_theta))++objective_evaluations;
  assert(objective_evaluations==0&&next_z==z&&step[0]==0.0);
}

void test_covariance_optimizer_rank_deficiency_budget_and_33_dimension_barrier()
{
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
    const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> rank_deficient={0.0,0.0,0.0,0.0,0.0,1.25};
    const CovarianceFitResult fit=solve_covariance_reference(system,rank_deficient,0.01,0.7,300);
    assert(fit.converged&&std::all_of(fit.theta.begin(),fit.theta.end(),[](double value){return std::isfinite(value);}));
    const std::vector<double> anisotropic={0.3125,0.0,0.0,1.25,0.0,2.1875};
    const CovarianceFitResult budget=solve_covariance_reference(system,anisotropic,1e-6,0.1,1);
    assert(!budget.converged&&budget.status=="ITERATION_BUDGET_EXHAUSTED");
  }
  {
    constexpr int n=12,d=3*n;std::vector<double> masses(n,1.0),sqrt_mass(d,1.0);DeviceBaseline baseline;baseline.d=d;
    assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),static_cast<std::size_t>(d)*d*sizeof(double))==cudaSuccess);
    assert(cudaMemset(baseline.k,0,static_cast<std::size_t>(d)*d*sizeof(double))==cudaSuccess);
    Graph graph;graph.type_pairs={{0,0}};for(int i=0;i<n-1;++i){Edge edge;edge.i=i;edge.j=i+1;edge.group=0;graph.edges.push_back(edge);}
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::vector<double> theta(6,0.0);theta[0]=theta[3]=theta[5]=1.0;
    double logdet=0.0,shifted_logdet=0.0;std::vector<double> statistics;
    assert(system.evaluate_factors(theta,1e-3,logdet,shifted_logdet,&statistics));
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-3,0.4,300);
    assert(system.r==33&&fit.converged&&fit.barrier_center_condition_pass&&fit.barrier_gap_estimate<=1e-6);
  }
}

void test_covariance_objective_analytic_gradient_matches_finite_difference()
{
  DeviceBaseline baseline;baseline.d=6;assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),36*sizeof(double))==cudaSuccess);
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const double t[2]={1.0/std::sqrt(5.0),2.0/std::sqrt(5.0)};std::vector<double> matrix(36,0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<2;++i)for(int j=0;j<2;++j){const int row=axis*2+i,column=axis*2+j;
    matrix[static_cast<std::size_t>(row)*6+column]=(i==j?1.0:0.0)-t[i]*t[j];}
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  Graph graph;graph.type_pairs={{0,1}};Edge edge;edge.i=0;edge.j=1;edge.group=0;graph.edges.push_back(edge);
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};
  std::uint64_t evaluations=0;const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,0.01,evaluations);
  constexpr double rho=0.7,tau=1e-4;const CovarianceObjectiveEvaluation analytic=evaluate_covariance_fit_objective(system,reference.theta,reference,statistics,rho,0.01,tau,evaluations);
  assert(analytic.feasible&&analytic.gradient.size()==6);
  for(std::size_t i=0;i<reference.theta.size();++i){const double step=1e-5*std::max(1.0,std::abs(reference.theta[i]));
    std::vector<double> plus=reference.theta,minus=reference.theta;plus[i]+=step;minus[i]-=step;
    const auto fplus=evaluate_covariance_fit_objective(system,plus,reference,statistics,rho,0.01,tau,evaluations,false);
    const auto fminus=evaluate_covariance_fit_objective(system,minus,reference,statistics,rho,0.01,tau,evaluations,false);
    assert(fplus.feasible&&fminus.feasible);const double numerical=(fplus.objective-fminus.objective)/(2.0*step);
    check_close(analytic.gradient[i],numerical,2e-5);
    check_close(analytic.gradient_data[i],(fplus.data-fminus.data)/(2.0*step),2e-5);
    check_close(analytic.gradient_prior[i],(fplus.prior-fminus.prior)/(2.0*step),2e-5);
    check_close(analytic.gradient_barrier[i],(fplus.barrier-fminus.barrier)/(2.0*step),2e-5);}
  std::vector<double> hessian(36);
  for(int column=0;column<6;++column){const double step=2e-5*std::max(1.0,std::abs(reference.theta[column]));
    std::vector<double> plus=reference.theta,minus=reference.theta;plus[column]+=step;minus[column]-=step;
    const auto gplus=evaluate_covariance_fit_objective(system,plus,reference,statistics,rho,0.01,tau,evaluations);
    const auto gminus=evaluate_covariance_fit_objective(system,minus,reference,statistics,rho,0.01,tau,evaluations);
    assert(gplus.feasible&&gminus.feasible);for(int row=0;row<6;++row)hessian[row*6+column]=(gplus.gradient[row]-gminus.gradient[row])/(2.0*step);}
  for(int row=0;row<6;++row)for(int column=0;column<row;++column)check_close(hessian[row*6+column],hessian[column*6+row],2e-4);
}

void test_covariance_gradient_decomposition_preserves_total()
{
  const std::vector<double> statistics={1.2,-0.4,0.8},baseline_trace={0.3,0.2,-0.1};
  const std::vector<double> trace={0.7,-0.2,0.5},shifted_trace={0.1,0.4,-0.3};
  constexpr double rho=0.1,tau=0.7;CovarianceObjectiveEvaluation value;
  covariance_decompose_gradient(value,statistics,baseline_trace,trace,shifted_trace,rho,tau);
  for(std::size_t i=0;i<statistics.size();++i){
    const double data=0.5*(statistics[i]-trace[i]),prior=0.5*rho*(baseline_trace[i]-trace[i]),barrier=-tau*shifted_trace[i];
    const double original=0.5*(statistics[i]+rho*baseline_trace[i])-0.5*(1.0+rho)*trace[i]-tau*shifted_trace[i];
    check_close(value.gradient_data[i],data,1e-15);check_close(value.gradient_prior[i],prior,1e-15);
    check_close(value.gradient_barrier[i],barrier,1e-15);check_close(value.gradient[i],original,1e-15);
    check_close(value.gradient_data[i]+value.gradient_prior[i]+value.gradient_barrier[i],value.gradient[i],2e-15);
  }
}

void test_covariance_edge_attribution_reconstructs_and_ranks()
{
  Graph graph;graph.type_pairs={{0,1},{0,2}};
  graph.edges={{0,1,0,{0,0,0}},{1,2,0,{0,0,0}},{2,3,1,{1,0,0}}};
  std::vector<double> edge_statistics(18,0.0),baseline_edge_trace(18,0.0),saved_edge_trace(18,0.0),shifted_edge_trace(18,0.0);
  edge_statistics[0]=7.0;edge_statistics[1]=10.0;edge_statistics[6]=3.0;edge_statistics[12]=1.0;
  edge_statistics[13]=200.0;baseline_edge_trace[13]=0.4;saved_edge_trace[13]=0.2;shifted_edge_trace[13]=0.1;
  baseline_edge_trace[0]=2.0;baseline_edge_trace[6]=2.0;
  saved_edge_trace[0]=1.0;saved_edge_trace[6]=5.0;
  shifted_edge_trace[0]=0.2;shifted_edge_trace[6]=0.4;
  const std::vector<double> statistics={10,10,0,0,0,0, 1,200,0,0,0,0};
  const std::vector<double> baseline_trace={4,0,0,0,0,0, 0,0.4,0,0,0,0};
  const std::vector<double> saved_trace={6,0,0,0,0,0, 0,0.2,0,0,0,0};
  const std::vector<double> shifted_trace={0.6,0,0,0,0,0, 0,0.1,0,0,0,0};
  constexpr double rho=0.5,tau=0.25;CovarianceObjectiveEvaluation value;value.feasible=true;
  covariance_decompose_gradient(value,statistics,baseline_trace,saved_trace,shifted_trace,rho,tau);
  std::vector<double> scale(12,1.0);scale[0]=3.0;scale[1]=0.5;scale[7]=0.001;
  const CovarianceEdgeAttribution attribution=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,rho,tau,true);
  assert(attribution.status=="PASS"&&attribution.dominant_parameter==0);
  assert((attribution.ranked_edges==std::vector<std::size_t>{0,1}));
  check_close(attribution.edge_data[0],3.0,1e-15);check_close(attribution.edge_prior[0],0.25,1e-15);
  check_close(attribution.edge_barrier[0],-0.05,1e-15);check_close(attribution.edge_total[0],3.2,1e-15);
  check_close(attribution.edge_total[6],-1.85,1e-15);check_close(attribution.edge_total[12],0.5,1e-15);
  check_close(attribution.edge_total[1],5.0,1e-15);check_close(attribution.edge_total[13],99.925,1e-15);
  check_close(attribution.atom_total[0],1.6,1e-15);check_close(attribution.atom_total[1],0.675,1e-15);
  check_close(attribution.atom_total[2],-0.925,1e-15);check_close(attribution.atom_total[3],0.0,1e-15);
  check_close(attribution.atom_absolute_total[1],2.525,1e-15);
  scale[1]=30.0;
  const CovarianceEdgeAttribution xy_group_zero=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,rho,tau,true);
  assert(xy_group_zero.status=="PASS"&&xy_group_zero.dominant_parameter==1);
  assert((xy_group_zero.ranked_edges==std::vector<std::size_t>{0,1}));
  assert(xy_group_zero.dominant_group_edge_count==2);
  check_close(xy_group_zero.sum_abs_scaled_edge_contribution,150.0,1e-12);
  assert(xy_group_zero.atom_total[3]==0.0);
  scale[1]=0.001;scale[7]=1.0;
  const CovarianceEdgeAttribution other_group=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,rho,tau,true);
  assert(other_group.status=="PASS"&&other_group.dominant_parameter==7);
  assert((other_group.ranked_edges==std::vector<std::size_t>{2}));
  assert(other_group.dominant_group_edge_count==1);
  check_close(other_group.sum_abs_scaled_edge_contribution,99.925,1e-12);
  check_close(other_group.top_20_abs_fraction,1.0,1e-15);
  assert(other_group.atom_total[0]==0.0&&other_group.atom_total[1]==0.0);

  std::vector<double> ridge_theta(12,0.0),ridge_center(12,0.0);ridge_theta[0]=0.8;
  covariance_decompose_gradient(value,statistics,baseline_trace,saved_trace,shifted_trace,0.0,tau,
    &ridge_theta,&ridge_center,&scale,0.5);
  const CovarianceEdgeAttribution ridge_attribution=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,0.0,tau,true);
  assert(ridge_attribution.status=="PASS"&&ridge_attribution.parameter_ridge_gradient.size()==12);
  check_close(ridge_attribution.parameter_ridge_gradient[0],0.8*0.5/(3.0*3.0),1e-15);
  check_close(ridge_attribution.edge_total[0]+ridge_attribution.edge_total[6]+ridge_attribution.parameter_ridge_gradient[0],
    value.gradient[0],1e-13);

  const CovarianceEdgeAttribution invalid_residuals=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,rho,tau,false);
  assert(invalid_residuals.status=="TRACE_RESIDUAL_FAIL"&&invalid_residuals.ranked_edges.empty());
  saved_edge_trace[0]+=1.0;
  const CovarianceEdgeAttribution mismatch=covariance_edge_gradient_attribution(
    graph,4,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,scale,value,rho,tau,true);
  assert(mismatch.status=="RECONSTRUCTION_FAIL"&&mismatch.ranked_edges.empty());
}

void test_covariance_edge_attribution_reports_diffuse_top20_fractions()
{
  constexpr int edge_count=21,atom_count=edge_count+1;Graph graph;graph.type_pairs={{0,0}};
  std::vector<double> edge_statistics(6*edge_count,0.0),baseline_edge_trace(6*edge_count,0.0);
  std::vector<double> saved_edge_trace(6*edge_count,0.0),shifted_edge_trace(6*edge_count,0.0);
  for(int edge=0;edge<edge_count;++edge){graph.edges.push_back({edge,edge+1,0,{0,0,0}});edge_statistics[6*edge]=2.0;}
  const std::vector<double> statistics={2.0*edge_count,0,0,0,0,0};
  const std::vector<double> trace(6,0.0),shifted_trace(6,0.0);
  constexpr double rho=0.5,tau=0.25;CovarianceObjectiveEvaluation value;value.feasible=true;
  covariance_decompose_gradient(value,statistics,trace,trace,shifted_trace,rho,tau);
  const CovarianceEdgeAttribution attribution=covariance_edge_gradient_attribution(
    graph,atom_count,edge_statistics,baseline_edge_trace,saved_edge_trace,shifted_edge_trace,
    std::vector<double>(6,1.0),value,rho,tau,true);
  assert(attribution.status=="PASS"&&attribution.dominant_group_edge_count==edge_count);
  assert(attribution.ranked_edges.size()==20&&attribution.ranked_atoms.size()==20);
  check_close(attribution.sum_abs_scaled_edge_contribution,edge_count,1e-14);
  check_close(attribution.top_20_abs_fraction,20.0/edge_count,1e-14);
  check_close(attribution.sum_abs_atom_contribution,edge_count,1e-14);
  check_close(attribution.top_20_atom_abs_fraction,20.0/edge_count,1e-14);
}

void test_covariance_edge_capture_reconstructs_group_trace_and_statistics()
{
  constexpr int n=3,d=3*n;Graph graph;graph.type_pairs={{0,0}};
  graph.edges={{0,1,0,{0,0,0}},{0,2,0,{0,0,0}},{1,2,0,{0,0,0}}};
  const std::vector<double> r0(d,0.0),frame0={0.0,1.0,2.0, 0.0,1.0,-1.0, 1.0,0.0,2.0};
  const std::vector<double> frame1={0.0,0.5,1.5, 0.0,-0.5,-1.5, 0.25,0.75,1.25};
  SampleHeader header{};header.n=n;header.frames=0;std::stringstream spool(std::ios::in|std::ios::out|std::ios::binary);
  int frame_index=0;for(const auto& position:{frame0,frame1}){const double step=frame_index++,force[9]={};
    spool.write(reinterpret_cast<const char*>(&step),sizeof(step));spool.write(reinterpret_cast<const char*>(position.data()),d*sizeof(double));
    spool.write(reinterpret_cast<const char*>(force),sizeof(force));}
  std::vector<double> edge_statistics;const std::vector<double> statistics=collect_covariance_fit_statistics(
    spool,header,2,2,graph,r0,300.0,&edge_statistics);
  for(int component=0;component<6;++component){NeumaierAccumulator sum;for(std::size_t edge=0;edge<graph.edges.size();++edge)sum.add(edge_statistics[6*edge+component]);
    check_close(sum.value(),statistics[component],2e-9);}

  DeviceBaseline baseline;baseline.d=d;assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),static_cast<std::size_t>(d)*d*sizeof(double))==cudaSuccess);
  std::vector<double> matrix(static_cast<std::size_t>(d)*d,0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)for(int j=0;j<n;++j){const int row=axis*n+i,column=axis*n+j;
    matrix[static_cast<std::size_t>(row)*d+column]=(i==j?1.0:0.0)-1.0/n;}
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  const std::vector<double> masses(n,1.0),sqrt_mass(d,1.0),theta={0.2,0.0,0.0,0.3,0.0,0.4};
  DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);double logdet=0.0,shifted_logdet=0.0;
  std::vector<double> trace,shifted_trace,edge_trace,edge_shifted_trace;
  assert(system.evaluate_factors(theta,0.05,logdet,shifted_logdet,&trace,&shifted_trace,false,true,&edge_trace,&edge_shifted_trace));
  assert(edge_trace.size()==6*graph.edges.size()&&edge_shifted_trace.size()==edge_trace.size());
  for(int component=0;component<6;++component){NeumaierAccumulator sum;for(std::size_t edge=0;edge<graph.edges.size();++edge)sum.add(edge_trace[6*edge+component]);
    check_close(sum.value(),trace[component],2e-12);}
  for(int component=0;component<6;++component){NeumaierAccumulator sum;for(std::size_t edge=0;edge<graph.edges.size();++edge)sum.add(edge_shifted_trace[6*edge+component]);
    check_close(sum.value(),shifted_trace[component],2e-12);}
}

void test_m1_statistics_compensate_projection_and_operator_mix()
{
  constexpr int n=9,d=3*n;Graph graph;graph.type_pairs={{0,0}};graph.base_parameter_count=6;
  graph.raw_local_parameter_count=1;graph.parameter_count=7;graph.local_base_mix.assign(6,0.0);
  graph.local_base_mix[1]=graph.local_base_mix[2]=graph.local_base_mix[4]=1.0;graph.local_transform={1.0};
  graph.edges={{0,1,0,{0,0,0}},{2,3,0,{0,0,0}},{4,5,0,{0,0,0}}};
  const std::vector<double> r0(d,0.0);std::vector<double> position(d,0.0);
  position[1]=1.0e8;position[n+1]=5.0e7;
  position[3]=1.0;position[2*n+3]=0.5;
  position[n+5]=1.0e8;position[2*n+5]=-5.0e7;
  position[6]=1.0e16;position[7]=1.0;position[8]=5.0e15;
  const rpmd_ja_local_basis::Site sites[3]={{6,{0,0,0}},{7,{0,0,0}},{8,{0,0,0}}};
  const rpmd_ja_local_basis::Vec3 coefficients[3]={{1.0,0.0,0.0},{1.0,0.0,0.0},{-2.0,0.0,0.0}};
  const double shares[3]={1.0/3.0,1.0/3.0,1.0/3.0};
  for(const double weight:{1.0e16,1.0,-1.0e16}){
    graph.local_terms.push_back(rpmd_ja_local_basis::detail::make_term(sites,coefficients,3,0,weight,shares));}
  std::vector<double> statistics(static_cast<std::size_t>(graph.parameter_count),0.0);
  add_covariance_frame_statistics(graph,position,r0,n,statistics);
  check_close(statistics[6],2.0,0.0);
}

void test_fixed_internal_probe_statistics_detect_zero_com_internal_shift()
{
  const int n=2;const std::vector<double> r0(6,0.0),masses={1.0,1.0},sqrt_mass(6,1.0);
  const double inverse_sqrt_two=1.0/std::sqrt(2.0);const std::vector<double> sqrt_atom={inverse_sqrt_two,inverse_sqrt_two};
  FixedPhysicalProbe probe;probe.source="INTERNAL_X";probe.vector={inverse_sqrt_two,-inverse_sqrt_two,0,0,0,0};
  std::vector<FixedPhysicalProbe> probes={probe};
  rpmd_ja_statistics_policy::TrainingChannels channels;
  channels.names={"fixed_r0_second_moment_x","fixed_r0_second_moment_y","fixed_r0_second_moment_z","fixed_internal_projection:INTERNAL_X"};
  channels.values.resize(channels.names.size());
  std::vector<double> first={1.0,-1.0,0,0,0,0},second={2.0,-2.0,0,0,0,0};
  append_fixed_r0_physical_frame(channels,first,r0,masses,sqrt_mass,probes,n);
  append_fixed_r0_physical_frame(channels,second,r0,masses,sqrt_mass,probes,n);
  check_close(channels.values[3][0],std::sqrt(2.0),1e-14);
  check_close(channels.values[3][1],2.0*std::sqrt(2.0),1e-14);
  check_close(channels.values[0][0],1.0,0.0);check_close(channels.values[0][1],4.0,0.0);
  check_close((first[0]+first[1])/2.0,(second[0]+second[1])/2.0,0.0);
  const auto orthonormal=make_internal_statistics_probes(probes,sqrt_atom,n);
  assert(orthonormal.size()==1);
  for(int axis=0;axis<3;++axis){double translation_component=0.0;
    for(int atom=0;atom<n;++atom)translation_component+=orthonormal[0].vector[axis*n+atom]*sqrt_atom[atom];
    assert(std::abs(translation_component)<1e-14);}
}

void test_admission_statistics_storage_peak_boundary()
{
  const std::uint64_t estimate=estimate_admission_statistics_storage_bytes(600000,400000,92,6,8,1000);
  assert(estimate==943301728ULL);
  assert(estimate>512ULL*1024*1024);
  assert(estimate_admission_statistics_storage_bytes(std::numeric_limits<std::uint64_t>::max(),
    400000,92,6,8,1000)==std::numeric_limits<std::uint64_t>::max());
  constexpr std::uint64_t budget=512ULL*1024*1024;
  const std::uint64_t channel_only=(330000ULL+340000ULL)*100ULL*sizeof(double);
  assert(channel_only<budget);
  assert(estimate_fixed_r0_drift_storage_bytes(330000,340000,100,100)>budget);
  assert(estimate_fixed_r0_drift_storage_bytes(std::numeric_limits<std::uint64_t>::max(),1,100,100)==
    std::numeric_limits<std::uint64_t>::max());
}

struct CpuCovarianceTraceReference{std::vector<double> trace,edge_trace;};

CpuCovarianceTraceReference covariance_trace_cpu_reference(const Graph& graph,const std::vector<double>& sqrt_mass,
  const std::vector<double>& theta,const double epsilon,const int omitted_begin=-1,const int omitted_end=-1)
{
  const int n=static_cast<int>(sqrt_mass.size()/3),d=3*n;const auto basis=internal_basis(std::vector<double>(n,1.0));const int r=static_cast<int>(basis.size());
  std::vector<double> additive(static_cast<std::size_t>(d)*d,0.0);const double signs[2]={-1.0,1.0};
  for(const Edge& edge:graph.edges){const int atoms[2]={edge.i,edge.j};
    for(int u=0;u<2;++u)for(int v=0;v<2;++v)for(int a=0;a<3;++a)for(int b=0;b<3;++b){
      const int row=a*n+atoms[u],column=b*n+atoms[v];const std::size_t parameter=6*static_cast<std::size_t>(edge.group)+edge_parameter_index(a,b);
      additive[static_cast<std::size_t>(row)*d+column]+=signs[u]*signs[v]*theta[parameter]/(sqrt_mass[row]*sqrt_mass[column]);}}
  std::vector<double> additive_basis(static_cast<std::size_t>(d)*r,0.0);
  for(int row=0;row<d;++row)for(int column=0;column<r;++column)for(int inner=0;inner<d;++inner)
    additive_basis[static_cast<std::size_t>(row)*r+column]+=additive[static_cast<std::size_t>(row)*d+inner]*basis[column][inner];
  std::vector<double> restricted(static_cast<std::size_t>(r)*r,0.0);
  for(int row=0;row<r;++row)for(int column=0;column<r;++column){double value=row==column?1.0-epsilon:0.0;
    for(int inner=0;inner<d;++inner)value+=basis[row][inner]*additive_basis[static_cast<std::size_t>(inner)*r+column];
    restricted[static_cast<std::size_t>(row)*r+column]=value;}
  std::vector<double> inverse(static_cast<std::size_t>(r)*r,0.0);
  for(int column=0;column<r;++column){std::vector<double> unit(r,0.0);unit[column]=1.0;const auto solved=solve_dense_reference(restricted,unit,r);
    for(int row=0;row<r;++row)inverse[static_cast<std::size_t>(row)*r+column]=solved[row];}
  std::vector<double> left(static_cast<std::size_t>(d)*r,0.0),full_inverse(static_cast<std::size_t>(d)*d,0.0);
  for(int row=0;row<d;++row)for(int column=0;column<r;++column)for(int inner=0;inner<r;++inner)
    left[static_cast<std::size_t>(row)*r+column]+=basis[inner][row]*inverse[static_cast<std::size_t>(inner)*r+column];
  for(int row=0;row<d;++row)for(int column=0;column<d;++column)for(int inner=0;inner<r;++inner)
    full_inverse[static_cast<std::size_t>(row)*d+column]+=left[static_cast<std::size_t>(row)*r+inner]*basis[inner][column];
  CpuCovarianceTraceReference out;out.trace.assign(6*graph.type_pairs.size(),0.0);out.edge_trace.assign(6*graph.edges.size(),0.0);
  for(std::size_t edge_index=0;edge_index<graph.edges.size();++edge_index){const Edge& edge=graph.edges[edge_index];double edge_left[9]={},edge_right[9]={};
    for(int source=0;source<d;++source){if(source>=omitted_begin&&source<omitted_end)continue;const int atom=source%n,beta=source/n;
      if(atom!=edge.i&&atom!=edge.j)continue;
      for(int alpha=0;alpha<3;++alpha){const double ci=full_inverse[static_cast<std::size_t>(alpha*n+edge.i)*d+source];
        const double cj=full_inverse[static_cast<std::size_t>(alpha*n+edge.j)*d+source];const std::size_t at=3*alpha+beta;
        if(atom==edge.i)edge_left[at]+=ci/(sqrt_mass[edge.i]*sqrt_mass[edge.i])-cj/(sqrt_mass[edge.i]*sqrt_mass[edge.j]);
        else edge_right[at]+=cj/(sqrt_mass[edge.j]*sqrt_mass[edge.j])-ci/(sqrt_mass[edge.j]*sqrt_mass[edge.i]);}}
    double block[9];for(int a=0;a<3;++a)for(int b=0;b<3;++b){const std::size_t ab=3*a+b,ba=3*b+a;
      block[ab]=0.5*((edge_left[ab]+edge_right[ab])+(edge_left[ba]+edge_right[ba]));}
    const std::size_t offset=6*edge_index;out.edge_trace[offset]=block[0];out.edge_trace[offset+1]=2.0*block[1];
    out.edge_trace[offset+2]=2.0*block[2];out.edge_trace[offset+3]=block[4];out.edge_trace[offset+4]=2.0*block[5];out.edge_trace[offset+5]=block[8];
    for(int component=0;component<6;++component)out.trace[6*static_cast<std::size_t>(edge.group)+component]+=out.edge_trace[offset+component];}
  return out;
}

void test_covariance_trace_wide_solve_matches_cpu_and_filters_failed_tail()
{
  constexpr int n=47,d=3*n;constexpr double epsilon=0.025;assert(DeviceCovarianceSystem::solve_batch==32);
  assert(DeviceCovarianceSystem::trace_solve_batch==128&&DeviceCovarianceSystem::trace_residual_batch==32);
  const std::vector<double> masses(n,1.0),sqrt_mass(d,1.0);Graph graph;graph.type_pairs={{0,0}};
  for(int atom=0;atom<n-1;++atom)graph.edges.push_back({atom,atom+1,0,{0,0,0}});
  const std::vector<double> theta={0.12,0.009,-0.006,0.15,0.008,0.18};
  const CpuCovarianceTraceReference expected_d=covariance_trace_cpu_reference(graph,sqrt_mass,theta,0.0);
  const CpuCovarianceTraceReference expected_shifted=covariance_trace_cpu_reference(graph,sqrt_mass,theta,epsilon);
  DeviceBaseline baseline;initialize_identity_covariance_baseline(baseline,n);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  double logdet=0.0,shifted_logdet=0.0;std::vector<double> trace,shifted_trace,edge_trace,edge_shifted_trace;
  assert(system.evaluate_factors(theta,epsilon,logdet,shifted_logdet,&trace,&shifted_trace,false,true,&edge_trace,&edge_shifted_trace));
  assert(system.trace_solve_calls==4&&system.solve_rhs_columns==2*static_cast<std::uint64_t>(d));
  for(const CovarianceTraceDiagnostics* diagnostic:{&system.last_d_trace,&system.last_shifted_trace}){
    assert(diagnostic->status=="PASS"&&!diagnostic->original_residual_checked);
    assert(diagnostic->rhs_columns_checked==static_cast<std::uint64_t>(d)&&diagnostic->total_rhs_columns==d);
    assert(diagnostic->rhs_batch_start==128&&diagnostic->rhs_columns==d-128);}
  for(std::size_t i=0;i<trace.size();++i){check_close(trace[i],expected_d.trace[i],2e-8);check_close(shifted_trace[i],expected_shifted.trace[i],2e-8);}
  for(std::size_t i=0;i<edge_trace.size();++i){check_close(edge_trace[i],expected_d.edge_trace[i],2e-8);check_close(edge_shifted_trace[i],expected_shifted.edge_trace[i],2e-8);}

  system.assemble(theta);assert(system.factor(system.compact,logdet));
  (void)system.trace_derivatives(system.compact,"D",0.0,true,true);
  assert(system.last_d_trace.status=="PASS"&&system.last_d_trace.original_residual_checked);

  system.assemble(theta);assert(system.factor(system.compact,logdet));
  covariance_fit_test_hooks.trace_solution_failure_column=130;covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
  std::vector<double> partial_edges;const std::vector<double> partial=system.trace_derivatives(system.compact,"D",0.0,true,false,&partial_edges);
  covariance_fit_test_hooks.trace_solution_failure_column=-1;
  const CpuCovarianceTraceReference expected_without_tail=covariance_trace_cpu_reference(graph,sqrt_mass,theta,0.0,128,d);
  assert(system.last_d_trace.status=="RESIDUAL_FAIL"&&system.last_d_trace.first_failure_batch_start==128&&system.last_d_trace.first_failure_columns==d-128);
  assert(system.last_d_trace.rhs_columns_checked==static_cast<std::uint64_t>(d));
  for(std::size_t i=0;i<partial.size();++i)check_close(partial[i],expected_without_tail.trace[i],2e-8);
  for(std::size_t i=0;i<partial_edges.size();++i)check_close(partial_edges[i],expected_without_tail.edge_trace[i],2e-8);

  system.assemble(theta);assert(system.factor(system.compact,logdet));
  covariance_fit_test_hooks.trace_solution_failure_column=40;covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
  CovarianceObjectiveEvaluation value;value.gradient.clear();value.feasible=true;value.cached_theta=theta;value.cached_epsilon=0.1;
  CovarianceBaseline reference;reference.trace=expected_d.trace;
  const std::uint64_t rhs_before_strict=system.solve_rhs_columns,calls_before_strict=system.trace_solve_calls;
  bool strict_failed=false;try{complete_covariance_fit_gradient_from_factors(system,value,reference,expected_d.trace,0.5,0.1,theta,1.0);}
  catch(const CovarianceSolveResidualError& error){strict_failed=true;
    assert(error.d_trace.status=="RESIDUAL_FAIL"&&error.d_trace.first_failure_batch_start==32&&error.d_trace.first_failure_columns==32);
    assert(error.d_trace.rhs_columns_checked==64&&system.solve_rhs_columns-rhs_before_strict==128&&system.trace_solve_calls-calls_before_strict==1);}
  covariance_fit_test_hooks.trace_solution_failure_column=-1;
  assert(strict_failed&&value.gradient.empty());
}

void test_covariance_gemm_original_residual_matches_cpu()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const int map[3]={1,3,5};const double internal[9]={4.0,1.0,-0.25,1.0,3.0,0.5,-0.25,0.5,2.0};
  std::vector<double> matrix(36,0.0),solution={0.25,-0.5,0.75,-0.2,0.4,0.1},rhs={1.1,-0.7,2.0,-0.3,0.8,-1.2};
  constexpr double epsilon=0.125;
  for(int row=0;row<3;++row)for(int column=0;column<3;++column)
    matrix[static_cast<std::size_t>(map[row])*6+map[column]]=internal[row*3+column];
  assert(cudaMemcpy(system.full,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  assert(cudaMemcpy(system.rhs,solution.data(),solution.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  assert(cudaMemcpy(system.rhs_check,rhs.data(),rhs.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  covariance_expand_solutions<<<1,32>>>(system.rhs,system.ambient,system.full_to_compact,system.d,system.r,2);
  assert(cudaGetLastError()==cudaSuccess&&cudaDeviceSynchronize()==cudaSuccess);
  system.original_solve_residual(system.rhs,system.rhs_check,2,epsilon,system.rhs_temp);
  std::vector<double> actual(6),expected(6);assert(cudaMemcpy(actual.data(),system.rhs_temp,actual.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  for(int column=0;column<2;++column)for(int row=0;row<3;++row){double value=-epsilon*solution[column*3+row]-rhs[column*3+row];
    for(int inner=0;inner<3;++inner)value+=internal[row*3+inner]*solution[column*3+inner];expected[column*3+row]=value;}
  for(std::size_t i=0;i<actual.size();++i)check_close(actual[i],expected[i],2e-14);
}

void test_covariance_strict_residual_failure_updates_overall_worst()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> zero_theta(6,0.0);system.assemble(zero_theta);double logdet=0.0,shifted_logdet=0.0;assert(system.factor(system.compact,logdet));
  std::vector<double> matrix(36);assert(cudaMemcpy(matrix.data(),system.full,matrix.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  matrix[1*6+1]+=0.5;assert(cudaMemcpy(system.full,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  (void)system.trace_derivatives(system.compact,"D",0.0,false,true);
  assert(system.last_d_trace.status=="PASS"&&!system.last_d_trace.original_residual_checked);
  bool failed=false;try{(void)system.trace_derivatives(system.compact,"D",0.0,true,true);}
  catch(const CovarianceSolveResidualError& error){failed=error.d_trace.status=="RESIDUAL_FAIL"&&error.d_trace.original_residual_checked;}
  assert(failed&&system.last_d_trace.status=="RESIDUAL_FAIL");
  assert(system.last_d_trace.worst_original_solve_relative_residual>1e-7);
  check_close(system.worst_solve_relative_residual,system.last_d_trace.worst_original_solve_relative_residual,1e-15);
  const double original_run_worst=system.run_trace_residual_summary.worst_original_relative_residual;
  assert(system.run_trace_residual_summary.original_any_checked&&system.run_trace_residual_summary.original_batches_checked==1);
  assert(system.run_trace_residual_summary.failed_logical_batches==1&&system.run_trace_residual_summary.first_failure_matrix_kind=="D");
  assert(system.run_trace_residual_summary.first_failure_channel=="ORIGINAL"&&system.run_trace_residual_summary.first_failure_batch_start==0);

  system.assemble(zero_theta);assert(system.factor(system.compact,logdet));
  covariance_fit_test_hooks.trace_solution_failure_column=0;covariance_fit_test_hooks.trace_solution_failure_delta=1e-4;
  assert(system.evaluate_factors(zero_theta,1e-6,logdet,shifted_logdet,nullptr,nullptr));
  bool shifted_failed=false;try{(void)system.trace_derivatives(system.shifted,"D_MINUS_EPSILON_I",1e-6,false,true);}
  catch(const CovarianceSolveResidualError&){shifted_failed=true;}
  covariance_fit_test_hooks.trace_solution_failure_column=-1;
  assert(shifted_failed&&system.run_trace_residual_summary.failed_logical_batches==2);
  assert(system.run_trace_residual_summary.worst_factor_matrix_kind=="D_MINUS_EPSILON_I");
  assert(system.run_trace_residual_summary.worst_factor_batch_start==0&&system.run_trace_residual_summary.worst_factor_relative_residual>1e-7);

  std::vector<double> trace,shifted_trace;
  assert(system.evaluate_factors(zero_theta,1e-6,logdet,shifted_logdet,&trace,&shifted_trace,false,true));
  CovarianceBaseline reference;reference.trace=trace;CovarianceObjectiveEvaluation candidate;candidate.feasible=true;
  candidate.cached_theta=zero_theta;candidate.cached_epsilon=1e-6;
  complete_covariance_fit_gradient_from_factors(system,candidate,reference,trace,0.5,1e-6,zero_theta,0.1);
  assert(system.worst_solve_relative_residual<1e-7);
  assert(system.run_trace_residual_summary.failed_logical_batches==2&&
    system.run_trace_residual_summary.first_failure_matrix_kind=="D"&&system.run_trace_residual_summary.first_failure_channel=="ORIGINAL");
  check_close(system.run_trace_residual_summary.worst_original_relative_residual,original_run_worst,1e-15);
}

void test_covariance_saved_gradient_does_not_require_baseline_shifted_solve()
{
  CovarianceTraceDiagnostics baseline_d,baseline_shifted,saved_d,saved_shifted;
  baseline_d.status=saved_d.status=saved_shifted.status="PASS";
  baseline_d.rhs_columns_checked=saved_d.rhs_columns_checked=saved_shifted.rhs_columns_checked=6;
  baseline_d.total_rhs_columns=saved_d.total_rhs_columns=saved_shifted.total_rhs_columns=6;
  baseline_shifted.status="RESIDUAL_FAIL";
  assert(!covariance_trace_diagnostics_pass(baseline_shifted,6));
  assert(covariance_saved_gradient_diagnostics_pass(baseline_d,saved_d,saved_shifted,6));
}

void test_covariance_state_parser_accepts_lf_and_crlf()
{
  const std::string lf="last_valid_available 1\ncom_convention internal_mass_com_pullback_v1\nlast_valid_tau 5.65e-11\noptimizer_step_mode TRUST_REGION_OPT_IN\nstaged_barrier_enabled 1\n";
  std::string crlf;for(const char character:lf){if(character=='\n')crlf.push_back('\r');crlf.push_back(character);}
  std::istringstream lf_input(lf),crlf_input(crlf);
  const auto lf_fields=read_covariance_state_fields(lf_input),crlf_fields=read_covariance_state_fields(crlf_input);
  assert(lf_fields.at("last_valid_available")=="1"&&crlf_fields.at("last_valid_available")=="1");
  assert(lf_fields.at("com_convention")=="internal_mass_com_pullback_v1"&&
    crlf_fields.at("com_convention")=="internal_mass_com_pullback_v1");
  assert(lf_fields.at("optimizer_step_mode")=="TRUST_REGION_OPT_IN"&&crlf_fields.at("optimizer_step_mode")=="TRUST_REGION_OPT_IN");
  assert(lf_fields.at("staged_barrier_enabled")=="1"&&crlf_fields.at("staged_barrier_enabled")=="1");
  const double lf_tau=std::stod(lf_fields.at("last_valid_tau")),crlf_tau=std::stod(crlf_fields.at("last_valid_tau"));
  check_close(lf_tau,crlf_tau,0.0);assert(covariance_state_tau_matches(lf_tau,crlf_tau));
}

void test_feasibility_manifest_parser_accepts_lf_and_crlf()
{
  const std::string stem="rpmd_ja_feasibility_manifest_line_ending_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string lf_path=stem+"_lf.txt",crlf_path=stem+"_crlf.txt";
  RemoveTestFile clean_lf{lf_path},clean_crlf{crlf_path};
  {std::ofstream out(lf_path,std::ios::binary);assert(out.good());out<<"RPJAFEAS9\ntraining_source_id \"train\"\nrule \"frozen\"\n";assert(out.good());}
  {std::ofstream out(crlf_path,std::ios::binary);assert(out.good());out<<"RPJAFEAS9\r\ntraining_source_id \"train\"\r\nrule \"frozen\"\r\n";assert(out.good());}
  const auto lf=read_feasibility_manifest(lf_path),crlf=read_feasibility_manifest(crlf_path);
  assert(lf==crlf&&crlf.at("training_source_id")=="train"&&crlf.at("rule")=="frozen");
}

void test_feasibility_manifest_freezes_independent_curvature_contract()
{
  std::map<std::string,std::string> fields={{"curvature_primary_seed","8d12e5b7c3a964f1"},
    {"curvature_independent_seed","4f1bbcdc6762c9a5"},{"curvature_probe_cap","128"},
    {"curvature_directional_tolerance","0.1"}};
  const auto contract=feasibility_manifest_curvature_contract(fields);
  assert(contract.primary_seed==covariance_primary_curvature_seed&&
    contract.independent_seed==covariance_independent_curvature_seed&&
    contract.probe_cap==CovarianceAnalyticCurvatureWorkspace::max_probes&&
    contract.directional_tolerance==covariance_independent_directional_tolerance);
  fields["curvature_probe_cap"]="64";bool rejected=false;
  try{(void)feasibility_manifest_curvature_contract(fields);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
}

void test_feasibility_manifest_freezes_candidate_cost_contract()
{
  std::map<std::string,std::string> fields={{"candidate_cost_stop_rule","MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1"},
    {"candidate_cost_safety_factor","1.5"}};
  const auto contract=feasibility_manifest_candidate_cost_contract(fields);
  assert(contract.rule=="MAX_COMPLETE_OR_PARTIAL_LOWER_BOUND_SECONDS_V1"&&contract.safety_factor==1.5);
  fields["candidate_cost_safety_factor"]="0.9";bool rejected=false;
  try{(void)feasibility_manifest_candidate_cost_contract(fields);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
}

void test_feasibility_manifest_freezes_experiment_contract()
{
  std::map<std::string,std::string> fields={{"lambda_factors","0.01,0.0001,0.000001"},
    {"wall_seconds","28800"},{"max_exact_gradients","240"},{"validation_reserve_fraction","0.25"},
    {"max_a_branch_gradients","8"},{"max_fit_family_gradients","112"},{"a_wall_fraction","0.1"},
    {"family_wall_split_rule","EQUAL_REMAINING_FIT_TIME_V1"},
    {"family_gradient_split_rule","EQUAL_REMAINING_EXACT_GRADIENTS_V1"},
    {"development_max_blocks","8"},{"development_min_blocks","4"},
    {"development_block_rule","CONTIGUOUS_EQUAL_FRAME_COUNTS_V1"}};
  const auto contract=feasibility_manifest_experiment_contract(fields);
  assert(contract.lambda_factors==std::vector<double>({1e-2,1e-4,1e-6}));
  assert(contract.max_development_blocks==8&&contract.min_development_blocks==4);
  fields["max_exact_gradients"]="241";bool rejected=false;
  try{(void)feasibility_manifest_experiment_contract(fields);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
}

void test_covariance_state_tau_comparison_is_relative_for_small_values()
{
  constexpr double saved_tau=5.65e-11;
  assert(covariance_state_tau_matches(saved_tau,saved_tau));
  assert(!covariance_state_tau_matches(saved_tau,1e-12));
}

void test_covariance_failed_trial_does_not_replace_last_valid_point()
{
  CovarianceFitResult result;result.last_valid.available=true;result.last_valid.theta={1.0,2.0};result.last_valid.objective=3.0;
  CovarianceObjectiveEvaluation trial;trial.data=4.0;trial.prior=5.0;trial.barrier=-1.0;trial.objective=8.0;
  record_covariance_failed_trial(result.failed_trial,{9.0,10.0},19,0.125,1.0,trial);
  assert(result.failed_trial.available&&result.failed_trial.iteration==19&&result.failed_trial.failure_stage=="GRADIENT_VALIDATION_PENDING");
  assert((result.failed_trial.theta==std::vector<double>{9.0,10.0})&&result.last_valid.available);
  assert((result.last_valid.theta==std::vector<double>{1.0,2.0})&&result.last_valid.objective==3.0);
}

void test_covariance_prior_mixing_formula()
{
  check_close(covariance_prior_mixed_variance_ratio(0.1,4.0),1.4/1.1,1e-15);
  check_close(covariance_prior_rho_upper_bound(4.0,0.1),0.1/2.9,1e-15);
  assert(std::isinf(covariance_prior_rho_upper_bound(1.05,0.1)));
}

void test_soft_mode_training_variance_uses_same_r0_second_moments()
{
  const std::array<double,5> projections{{2.0,2.0,3.0,4.0,4.0}};SoftModeTrainingVariance support;
  for(std::size_t frame=0;frame<projections.size();++frame)support.add(projections[frame],frame,projections.size(),4);
  assert((support.block_frames==std::array<std::uint64_t,4>{{1,1,1,2}}));
  check_close(support.pooled.value()/projections.size(),9.8,1e-15);
  check_close(support.blocks[0].value()/support.block_frames[0],4.0,1e-15);
  check_close(support.blocks[1].value()/support.block_frames[1],4.0,1e-15);
  check_close(support.blocks[2].value()/support.block_frames[2],9.0,1e-15);
  check_close(support.blocks[3].value()/support.block_frames[3],16.0,1e-15);
  check_close((16.0-4.0)/(support.pooled.value()/projections.size()),12.0/9.8,1e-15);
}

void test_saved_d_mode_participation_projection_and_status()
{
  RitzMode mode;mode.value=2.5;mode.actual_rayleigh=2.5;mode.residual=1.0e-9;
  mode.vector={0.6,0.8,0.0,0.0,0.0,0.0};
  const auto participation=mass_weighted_mode_participation(mode.vector,{4,9});
  assert(participation.status=="PASS");
  check_close(participation.atom_fractions[0],0.36,1e-15);
  check_close(participation.atom_fractions[1],0.64,1e-15);
  assert(participation.type_fractions.size()==2);
  assert(participation.type_fractions[0].first==4&&participation.type_fractions[1].first==9);
  check_close(participation.type_fractions[0].second,0.36,1e-15);
  check_close(participation.type_fractions[1].second,0.64,1e-15);
  Edge edge{};edge.i=0;edge.j=1;
  const std::vector<double> sqrt_atom{1.0,2.0},r0{0.0,2.0,0.0,0.0,0.0,0.0};Box box{};
  box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;
  const SavedModePairProjection pair=saved_mode_pair_projection(mode.vector,sqrt_atom,r0,box,edge);
  check_close(pair.relative[0],-0.2,1e-15);check_close(pair.relative[1],0.0,1e-15);check_close(pair.relative[2],0.0,1e-15);
  check_close(pair.norm,0.2,1e-15);check_close(pair.longitudinal,-0.2,1e-15);
  assert(saved_d_mode_localization_status(mode,0.05)=="PASS");
  assert(saved_d_mode_numerical_status(mode,participation,0.05,true)=="PASS");
  assert(saved_d_mode_numerical_status(mode,participation,0.05,false)=="FAILED");
  assert(saved_d_mode_aggregate_status(true,false)=="PASS");
  assert(saved_d_mode_aggregate_status(true,true)=="INCONCLUSIVE_RITZ_RESIDUAL");
  assert(saved_d_mode_aggregate_status(false,true)=="FAILED");
  mode.residual=1.0e6;
  assert(saved_d_mode_localization_status(mode,0.05)=="INCONCLUSIVE");
  assert(saved_d_mode_numerical_status(mode,participation,0.05,true)=="INCONCLUSIVE_RITZ_RESIDUAL");
  assert(saved_d_mode_aggregate_status(true,true)=="INCONCLUSIVE_RITZ_RESIDUAL");
  mode.actual_rayleigh=std::numeric_limits<double>::quiet_NaN();
  assert(saved_d_mode_numerical_status(mode,participation,0.05,true)=="FAILED");
}

void test_compressed_training_force_residual_distinguishes_fit_cost()
{
  const std::vector<double> r={1.0},z={2.0};constexpr double discarded2=64.0,force2=100.0;
  const double unconstrained=compressed_training_force_residual(r,z,discarded2,force2,{2.0});
  const double constrained=compressed_training_force_residual(r,z,discarded2,force2,{3.0});
  check_close(unconstrained,0.8,1e-15);
  check_close(constrained,std::sqrt(65.0)/10.0,1e-15);
  assert(constrained>unconstrained);
  check_close(constrained*constrained-unconstrained*unconstrained,0.01,1e-15);
}

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

void check_close(const double a, const double b, const double tol)
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
  const auto direct=spectral_row(graph,q,sqrt_mass,n,p);std::vector<double> design_based(p,0.0);
  for(int c=0;c<p;++c)for(int i=0;i<d;++i)design_based[c]-=q[i]*design[static_cast<std::size_t>(c)*d+i];
  for(int c=0;c<p;++c)check_close(direct[c],design_based[c],2e-12);
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

void test_baseline_batch_matches_gemv()
{
  constexpr int n=3,d=3*n,columns=32;std::vector<double> masses={1.0,2.5,7.0},sqrt_mass(d);
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)sqrt_mass[axis*n+i]=std::sqrt(masses[i]);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=2.0;
  std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);
  std::vector<double> input(static_cast<std::size_t>(d)*columns),batch_output(input.size()),single(d),single_output(d);
  for(int column=0;column<columns;++column)for(int i=0;i<d;++i)input[static_cast<std::size_t>(column)*d+i]=std::sin(0.17*(i+1)*(column+1));
  baseline.apply_batch(input,columns,batch_output);
  for(int column=0;column<columns;++column){std::copy_n(input.begin()+static_cast<std::size_t>(column)*d,d,single.begin());baseline.apply(single,single_output);
    for(int i=0;i<d;++i)check_close(batch_output[static_cast<std::size_t>(column)*d+i],single_output[i],2e-10);}
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
  const auto qp=solve_cut_qp(solver,identity,constraints,{0.5,-1.0,0.75});
  const auto& xi=qp.xi;
  check_close(xi[0],1.0,2e-8);check_close(xi[1],0.0,2e-8);

  identity.eta={0.0,0.0};
  const double delta=1e-4;
  const std::vector<std::vector<double>> near_parallel={{1.0,0.0},{1.0,delta}};
  const auto slow_qp=solve_cut_qp(solver,identity,near_parallel,{2.0,2.0+0.25*delta*delta});
  assert(slow_qp.certificate.accepted);
  assert(slow_qp.last_multiplier_change>1e-11);

  const std::vector<std::vector<double>> one_constraint={{1.0,0.0}};
  const auto non_complementary=check_cut_qp_kkt(identity,one_constraint,{0.0},one_constraint,{0.0},{1.0},{1.0,0.0},1e-8);
  assert(!non_complementary.accepted);
  assert(non_complementary.max_primal_violation==0.0);
  assert(non_complementary.max_complementarity>1e-7);
  identity.eta={0.0,0.0};
  const auto located_residuals=check_cut_qp_kkt(identity,{{1.0,0.0},{0.0,1.0}},{0.0,0.0},
    {{1.0,0.0},{0.0,1.0}},{0.0,0.0},{1.0,2.0},{1.2,1.3},1e-8);
  assert(located_residuals.finite&&located_residuals.multipliers_nonnegative);
  assert(located_residuals.worst_complementarity_index==1&&located_residuals.worst_complementarity_lambda==2.0);
  check_close(located_residuals.worst_complementarity_primal_slack,1.3,1e-15);
  check_close(located_residuals.worst_complementarity_dual_slack,1.3,1e-15);
  check_close(located_residuals.worst_complementarity_value,2.6,1e-15);
  assert(located_residuals.worst_stationarity_index==1);
  check_close(located_residuals.worst_stationarity_residual,-0.7,1e-15);
  check_close(located_residuals.max_lambda,2.0,0.0);
  assert(located_residuals.max_lambda_index==1);
  const auto diagnostic_scale_overflow=check_cut_qp_kkt(identity,{{1.0e308,-1.0e308}},{0.0},{{1.0,1.0}},{2.0},{1.0},{1.0,1.0},1e-8);
  assert(diagnostic_scale_overflow.accepted&&diagnostic_scale_overflow.finite&&!diagnostic_scale_overflow.diagnostic_finite);
  std::ostringstream overflow_fields;write_qp_certificate_fields(overflow_fields,diagnostic_scale_overflow,1e-8);
  assert(overflow_fields.str().find("finite=1 diagnostic_finite=0")!=std::string::npos);
  assert(overflow_fields.str().find("accepted=1")!=std::string::npos);
  SmallSVD scalar_identity;scalar_identity.p=1;scalar_identity.singular={1.0};scalar_identity.eta={0.0};scalar_identity.vt={1.0};
  const double two_to_54=18014398509481984.0;
  const auto compensated_stationarity=check_cut_qp_kkt(scalar_identity,{{1.0},{1.0}},{two_to_54,two_to_54},
    {{1.0},{1.0}},{two_to_54,two_to_54},{1.0,two_to_54},{two_to_54},1e-8);
  assert(compensated_stationarity.worst_stationarity_residual==-1.0&&compensated_stationarity.max_stationarity==1.0);
  const double product_offset=std::ldexp(1.0,-27),product_left=1.0+product_offset,product_right=1.0-product_offset;
  const double rounded_product=product_left*product_right;
  assert(rounded_product==1.0);
  const double product_roundoff_residual=compensated_constraint_residual({product_left},{product_right},1.0);
  assert(product_roundoff_residual==std::ldexp(1.0,-54));

  const std::vector<std::vector<double>> unequal_tolerance_rows={{1.0,0.0},{0.0,1.0}};
  const std::vector<double> unequal_tolerance_rhs={100000.0+1.048e-9,1.0+1e-12};
  const auto unequal_tolerance=check_cut_qp_kkt(identity,unequal_tolerance_rows,unequal_tolerance_rhs,
    unequal_tolerance_rows,unequal_tolerance_rhs,{100000.0,1.0},{100000.0,1.0},1e-8);
  assert(unequal_tolerance.minimum_slack_constraint_index==0);
  assert(unequal_tolerance.worst_primal_excess_index==1);
  assert(unequal_tolerance.max_primal_excess>8.8e-13&&unequal_tolerance.max_primal_excess<9.0e-13);
  assert(unequal_tolerance.minimum_slack_allowed_error==1e-8&&unequal_tolerance.allowed_error[1]<2e-13);
  check_close(unequal_tolerance.worst_primal_allowed_error,unequal_tolerance.allowed_error[1],0.0);
  assert(!unequal_tolerance.primal_constraints_pass);

  const auto first_qp=solve_cut_qp(solver,identity,{{1.0,0.0}},{0.25});
  const std::vector<std::vector<double>> extended={{1.0,0.0},{0.0,1.0},{1.0,1.0}};
  const std::vector<double> extended_rhs={0.25,0.75,1.2};
  const auto warm_qp=solve_cut_qp(solver,identity,extended,extended_rhs,1e-8,first_qp.lambda);
  const auto cold_qp=solve_cut_qp(solver,identity,extended,extended_rhs);
  assert(warm_qp.certificate.accepted==cold_qp.certificate.accepted);
  assert(warm_qp.certificate.accepted);
  check_close(std::inner_product(warm_qp.xi.begin(),warm_qp.xi.end(),warm_qp.xi.begin(),0.0),
    std::inner_product(cold_qp.xi.begin(),cold_qp.xi.end(),cold_qp.xi.begin(),0.0),2e-8);

  const auto previous_qp=solve_cut_qp(solver,identity,{{1.0,0.0},{0.0,1.0}},{1.0,1.0});
  const std::string qp_state_path="rpmd_ja_qp_state_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  bool infeasible_rejected=false;
  try { (void)solve_cut_qp(solver,identity,{{1.0,0.0},{0.0,1.0},{-1.0,0.0},{0.0,-1.0}},{1.0,1.0,1.0,1.0},1e-8,previous_qp.lambda,qp_state_path,7); }
  catch(const std::runtime_error&) { infeasible_rejected=true; }
  assert(infeasible_rejected);
  std::ifstream qp_state_file(qp_state_path);
  const std::string qp_state((std::istreambuf_iterator<char>(qp_state_file)),std::istreambuf_iterator<char>());
  assert(qp_state.find("outer_index 7")!=std::string::npos);
  assert(qp_state.find("iterations 200000")!=std::string::npos);
  assert(qp_state.find("method coordinate")!=std::string::npos&&qp_state.find("polish_attempts 3")!=std::string::npos);
  assert(qp_state.find("active_constraints ")!=std::string::npos&&qp_state.find("polish_updates ")!=std::string::npos&&qp_state.find("polish_failure_reason ACTIVE_")!=std::string::npos);
  assert(qp_state.find("primal_tolerance 1e-08")!=std::string::npos);
  assert(qp_state.find("svd_singular")!=std::string::npos&&qp_state.find("svd_vt")!=std::string::npos&&qp_state.find("svd_eta")!=std::string::npos);
  assert(qp_state.find("original_rows")!=std::string::npos&&qp_state.find("transformed_a")!=std::string::npos&&qp_state.find("gram")!=std::string::npos);
  assert(qp_state.find("initial_lambda")!=std::string::npos&&qp_state.find("lambda")!=std::string::npos&&qp_state.find("xi")!=std::string::npos&&qp_state.find("theta")!=std::string::npos);
  assert(qp_state.find("minimum_slack_constraint_index")!=std::string::npos&&qp_state.find("minimum_primal_slack")!=std::string::npos&&qp_state.find("minimum_slack_allowed_error")!=std::string::npos);
  assert(qp_state.find("worst_primal_excess_index")!=std::string::npos&&qp_state.find("max_primal_excess")!=std::string::npos&&qp_state.find("primal_constraints_pass 0")!=std::string::npos);
  assert(qp_state.find("last_multiplier_change")!=std::string::npos&&qp_state.find("max_complementarity")!=std::string::npos&&qp_state.find("max_stationarity")!=std::string::npos);

  auto diagnostic_field=[](const std::string& diagnostic,const std::string& key){const std::string marker=key+"=";const std::size_t begin=diagnostic.find(marker);assert(begin!=std::string::npos);const std::size_t value_begin=begin+marker.size();const std::size_t end=diagnostic.find(' ',value_begin);return diagnostic.substr(value_begin,end==std::string::npos?std::string::npos:end-value_begin);};
  std::string coordinate_diagnostic;std::vector<std::string> polish_diagnostics;std::istringstream diagnostic_lines(qp_state);std::string diagnostic_line;
  while(std::getline(diagnostic_lines,diagnostic_line)){
    if(diagnostic_line.rfind("QP_COORDINATE_DIAGNOSTIC ",0)==0){assert(coordinate_diagnostic.empty());coordinate_diagnostic=diagnostic_line;}
    if(diagnostic_line.rfind("QP_POLISH_DIAGNOSTIC ",0)==0)polish_diagnostics.push_back(diagnostic_line);
  }
  assert(!coordinate_diagnostic.empty()&&polish_diagnostics.size()==3);
  assert(diagnostic_field(coordinate_diagnostic,"iteration")=="200000");
  assert(diagnostic_field(coordinate_diagnostic,"certificate_status")=="COMPUTED");
  bool saw_not_computed=false;
  const int polish_iterations[3]={64,4096,200000};
  for(int i=0;i<3;++i){const std::string& diagnostic=polish_diagnostics[i];assert(diagnostic_field(diagnostic,"iteration")==std::to_string(polish_iterations[i]));
    assert(diagnostic.find("failure_reason=ACTIVE_")!=std::string::npos);
    for(const char* key:{"primal_pass","multipliers_pass","complementarity_pass","stationarity_pass","finite"})assert(diagnostic.find(std::string(key)+"=")!=std::string::npos);
    const std::string status=diagnostic_field(diagnostic,"certificate_status");
    if(status=="NOT_COMPUTED"){
      saw_not_computed=true;assert(diagnostic_field(diagnostic,"primal_pass")=="NOT_COMPUTED");
      assert(diagnostic.find("max_primal_excess=")==std::string::npos&&diagnostic.find("max_complementarity=")==std::string::npos&&diagnostic.find("max_stationarity=")==std::string::npos);
    }else{
      assert(status=="COMPUTED");
      assert(diagnostic.find("worst_complementarity_primal_slack=")!=std::string::npos&&diagnostic.find("worst_stationarity_component=")!=std::string::npos);
    }
  }
  assert(saw_not_computed);

  std::istringstream qp_state_stream(qp_state);std::string line;
  std::vector<std::vector<double>> saved_rows,saved_a;std::vector<double> saved_rhs,saved_b,saved_initial_lambda,saved_lambda,saved_xi,saved_theta,saved_eta;
  std::vector<double> saved_primal,saved_allowed,saved_excess,saved_dual,saved_state_lambda,saved_complementarity;std::size_t saved_constraints=0,saved_parameters=0;
  double summary_max_primal_excess=0.0,summary_max_complementarity=0.0;std::size_t summary_worst_excess_index=0;int summary_primal_pass=-1;
  auto parse_vector_line=[](std::istringstream& fields,std::vector<double>& values){std::size_t count=0;fields>>count;values.resize(count);for(double& value:values)fields>>value;};
  while(std::getline(qp_state_stream,line)){
    std::istringstream fields(line);std::string key;fields>>key;
    if(key=="max_primal_excess")fields>>summary_max_primal_excess;
    else if(key=="worst_primal_excess_index")fields>>summary_worst_excess_index;
    else if(key=="primal_constraints_pass")fields>>summary_primal_pass;
    else if(key=="max_complementarity")fields>>summary_max_complementarity;
    else if(key=="initial_lambda")parse_vector_line(fields,saved_initial_lambda);
    else if(key=="lambda")parse_vector_line(fields,saved_lambda);
    else if(key=="xi")parse_vector_line(fields,saved_xi);
    else if(key=="theta")parse_vector_line(fields,saved_theta);
    else if(key=="svd_eta")parse_vector_line(fields,saved_eta);
    else if(key=="original_rows"){
      fields>>saved_constraints>>saved_parameters;saved_rows.assign(saved_constraints,std::vector<double>(saved_parameters));saved_rhs.resize(saved_constraints);
      for(std::size_t i=0;i<saved_constraints;++i){assert(std::getline(qp_state_stream,line));std::istringstream row_fields(line);std::size_t index=0;std::string rhs_tag;row_fields>>index;assert(index==i);for(double& value:saved_rows[i])row_fields>>value;row_fields>>rhs_tag>>saved_rhs[i];assert(rhs_tag=="rhs");}
    }else if(key=="transformed_a"){
      std::size_t count=0,columns=0;fields>>count>>columns;assert(count==saved_constraints&&columns==saved_parameters);saved_a.assign(count,std::vector<double>(columns));saved_b.resize(count);
      for(std::size_t i=0;i<count;++i){assert(std::getline(qp_state_stream,line));std::istringstream row_fields(line);std::size_t index=0;std::string b_tag;row_fields>>index;assert(index==i);for(double& value:saved_a[i])row_fields>>value;row_fields>>b_tag>>saved_b[i];assert(b_tag=="b");}
    }else if(key=="constraint_state"){
      for(std::size_t i=0;i<saved_constraints;++i){assert(std::getline(qp_state_stream,line));std::istringstream row_fields(line);std::size_t index=0;double primal=0.0,allowed=0.0,excess=0.0,dual=0.0,multiplier=0.0,complementarity=0.0;row_fields>>index>>primal>>allowed>>excess>>dual>>multiplier>>complementarity;assert(index==i);saved_primal.push_back(primal);saved_allowed.push_back(allowed);saved_excess.push_back(excess);saved_dual.push_back(dual);saved_state_lambda.push_back(multiplier);saved_complementarity.push_back(complementarity);}
    }
  }
  assert(saved_parameters==2&&saved_constraints==4&&saved_initial_lambda.size()==4);
  check_close(saved_initial_lambda[0],previous_qp.lambda[0],0.0);check_close(saved_initial_lambda[1],previous_qp.lambda[1],0.0);
  check_close(saved_initial_lambda[2],0.0,0.0);check_close(saved_initial_lambda[3],0.0,0.0);
  assert(saved_lambda.size()==4&&saved_xi.size()==2&&saved_theta.size()==2&&saved_eta.size()==2&&saved_a.size()==4&&saved_primal.size()==4&&saved_excess.size()==4&&saved_dual.size()==4&&saved_complementarity.size()==4&&saved_state_lambda.size()==4);
  assert(summary_primal_pass==0);
  auto check_replayed_value=[](double actual,double saved){assert(std::abs(actual-saved)<=2e-12*std::max({1.0,std::abs(actual),std::abs(saved)}));};
  double replayed_max_excess=0.0,replayed_max_complementarity=0.0;std::size_t replayed_worst_excess_index=0;
  for(std::size_t i=0;i<saved_constraints;++i){
    double primal=-saved_rhs[i],dual=-saved_b[i];
    for(std::size_t k=0;k<saved_parameters;++k){primal+=saved_rows[i][k]*saved_theta[k];dual+=saved_a[i][k]*(saved_xi[k]-saved_eta[k]);}
    check_replayed_value(primal,saved_primal[i]);check_replayed_value(std::max(0.0,-primal-saved_allowed[i]),saved_excess[i]);check_replayed_value(dual,saved_dual[i]);
    check_replayed_value(saved_lambda[i],saved_state_lambda[i]);check_replayed_value(saved_lambda[i]*dual,saved_complementarity[i]);
    if(saved_excess[i]>replayed_max_excess){replayed_max_excess=saved_excess[i];replayed_worst_excess_index=i;}
    replayed_max_complementarity=std::max(replayed_max_complementarity,std::abs(saved_lambda[i]*dual));
  }
  check_replayed_value(replayed_max_excess,summary_max_primal_excess);check_replayed_value(replayed_max_complementarity,summary_max_complementarity);
  assert(replayed_worst_excess_index==summary_worst_excess_index);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"max_primal_excess")),replayed_max_excess);
  assert(std::stoull(diagnostic_field(coordinate_diagnostic,"worst_primal_constraint"))==replayed_worst_excess_index);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_primal_allowed_error")),saved_allowed[replayed_worst_excess_index]);
  std::size_t replayed_worst_complementarity_index=0,replayed_worst_stationarity_index=0,replayed_max_lambda_index=0;double replayed_worst_complementarity=0.0,replayed_worst_stationarity=0.0,replayed_worst_stationarity_residual=0.0,replayed_max_lambda=0.0;
  for(std::size_t i=0;i<saved_constraints;++i){const double magnitude=std::abs(saved_lambda[i]*saved_dual[i]);if(magnitude>replayed_worst_complementarity){replayed_worst_complementarity=magnitude;replayed_worst_complementarity_index=i;}}
  for(std::size_t k=0;k<saved_parameters;++k){double stationarity=saved_xi[k]-saved_eta[k];for(std::size_t i=0;i<saved_constraints;++i)stationarity-=saved_a[i][k]*saved_lambda[i];if(std::abs(stationarity)>replayed_worst_stationarity){replayed_worst_stationarity=std::abs(stationarity);replayed_worst_stationarity_index=k;replayed_worst_stationarity_residual=stationarity;}}
  for(std::size_t i=0;i<saved_constraints;++i)if(saved_lambda[i]>replayed_max_lambda){replayed_max_lambda=saved_lambda[i];replayed_max_lambda_index=i;}
  assert(std::stoull(diagnostic_field(coordinate_diagnostic,"worst_complementarity_constraint"))==replayed_worst_complementarity_index);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_complementarity_lambda")),saved_lambda[replayed_worst_complementarity_index]);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_complementarity_primal_slack")),saved_primal[replayed_worst_complementarity_index]);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_complementarity_dual_slack")),saved_dual[replayed_worst_complementarity_index]);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_complementarity_value")),saved_complementarity[replayed_worst_complementarity_index]);
  assert(std::stoull(diagnostic_field(coordinate_diagnostic,"worst_stationarity_component"))==replayed_worst_stationarity_index);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"worst_stationarity_residual")),replayed_worst_stationarity_residual);
  check_replayed_value(std::stod(diagnostic_field(coordinate_diagnostic,"max_lambda")),replayed_max_lambda);
  assert(std::stoull(diagnostic_field(coordinate_diagnostic,"max_lambda_index"))==replayed_max_lambda_index);
  qp_state_file.close();assert(std::remove(qp_state_path.c_str())==0);

  bool rejected=false;
  try { (void)svd_small(solver,{std::numeric_limits<double>::quiet_NaN(),0.0,0.0,1.0},{0.0,0.0},2); }
  catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
}

void test_active_set_qp_snapshot_and_rejection(cusolverDnHandle_t solver)
{
  const std::string fixture_path=covariance_fit_test_fixture_path("ja_reference_resample.bin.qp_state.txt");
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);
  assert(fixture.rows.size()==233);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  assert(replay.method==std::string("active_set_polish"));
  assert(((replay.iterations==64&&replay.polish_attempts==1)||(replay.iterations==4096&&replay.polish_attempts==2)||
    (replay.iterations==200000&&replay.polish_attempts==3))&&replay.polish_updates>0);
  assert(replay.polish_diagnostics.size()==static_cast<std::size_t>(replay.polish_attempts));
  assert(replay.polish_diagnostics.back().certificate_computed);
  const auto& polish_certificate=replay.polish_diagnostics.back().certificate;
  assert(polish_certificate.accepted==replay.certificate.accepted&&polish_certificate.worst_complementarity_index==replay.certificate.worst_complementarity_index&&
    polish_certificate.worst_stationarity_index==replay.certificate.worst_stationarity_index&&polish_certificate.max_lambda_index==replay.certificate.max_lambda_index);
  check_close(polish_certificate.max_complementarity,replay.certificate.max_complementarity,0.0);
  check_close(polish_certificate.worst_complementarity_value,replay.certificate.worst_complementarity_value,0.0);
  check_close(polish_certificate.worst_stationarity_residual,replay.certificate.worst_stationarity_residual,0.0);
  assert(replay.certificate.accepted&&replay.certificate.primal_constraints_pass&&replay.certificate.finite);
  assert(replay.certificate.primal_slack.size()==233&&replay.lambda.size()==233&&replay.certificate.max_primal_excess==0.0);
  assert(replay.certificate.max_complementarity<=qp_complementarity_tolerance&&replay.certificate.max_stationarity<=qp_stationarity_tolerance);

  SmallSVD identity;identity.p=2;identity.singular={1.0,1.0};identity.eta={0.0,0.0};identity.vt={1.0,0.0,0.0,1.0};
  const std::vector<std::vector<double>> rows={{1.0,0.0},{0.0,1.0}};const std::vector<double> rhs={1.0,-0.7};
  const ActiveSetPolishResult negative=polish_cut_qp(solver,identity,rows,rhs,rows,rhs,{1.0,1.0},1e-8);
  const double rounded_alpha=0.1/(0.1-(-0.7));assert(0.1+rounded_alpha*(-0.7-0.1)<0.0);
  const ActiveSetPolishResult rounded_exit=polish_cut_qp(solver,identity,rows,rhs,rows,rhs,{1.0,0.1},1e-8);
  assert(negative.accepted&&negative.lambda[0]>=0.0&&negative.lambda[1]==0.0&&negative.updates>0);
  assert(rounded_exit.accepted&&rounded_exit.lambda[0]>=0.0&&rounded_exit.lambda[1]==0.0&&rounded_exit.updates>0);
  assert(negative.certificate_computed&&rounded_exit.certificate_computed);
  assert(negative.certificate.primal_slack.size()==rows.size()&&negative.certificate.primal_slack[1]>0.0);
  assert(rounded_exit.certificate.primal_slack.size()==rows.size()&&rounded_exit.certificate.primal_slack[1]>0.0);

  const std::vector<std::vector<double>> dependent={{1.0,0.0},{2.0,0.0}};const std::vector<double> dependent_rhs={1.0,2.0};
  const ActiveSetPolishResult rank_failure=polish_cut_qp(solver,identity,dependent,dependent_rhs,dependent,dependent_rhs,{1.0,1.0},1e-8);
  assert(rank_failure.accepted&&rank_failure.certificate_computed&&rank_failure.certificate.primal_slack.size()==2&&rank_failure.certificate.accepted);
  std::vector<double> equality_y,equality_lambda;std::string equality_failure;
  assert(!solve_active_equalities(solver,dependent,dependent_rhs,{0,1},identity.p,identity.eta,equality_y,equality_lambda,equality_failure));
  assert(equality_failure=="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT");
  const std::vector<std::vector<double>> too_many_active={{1.0,0.0},{0.0,1.0},{1.0,1.0}};const std::vector<double> too_many_rhs={1.0,1.0,2.0};
  const ActiveSetPolishResult dimension_rank_failure=polish_cut_qp(solver,identity,too_many_active,too_many_rhs,too_many_active,too_many_rhs,{1.0,1.0,1.0},1e-8);
  assert(dimension_rank_failure.accepted&&dimension_rank_failure.certificate.primal_slack.size()==3&&dimension_rank_failure.certificate.accepted);
  assert(dimension_rank_failure.certificate_computed);
  const ActiveSetPolishResult early_exit=polish_cut_qp(solver,identity,rows,rhs,rows,rhs,{-1.0,0.0},1e-8);
  assert(!early_exit.accepted&&!early_exit.certificate_computed&&early_exit.failure_reason=="ACTIVE_INITIAL_MULTIPLIER_INVALID");
}

void test_active_set_preserves_svd_primal(cusolverDnHandle_t solver)
{
  const std::string fixture_path=covariance_fit_test_fixture_path("ja_reference_replay_v1.bin.qp_state.txt");
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);assert(fixture.rows.size()==20);
  std::vector<double> seed(fixture.rows.size(),0.0);for(const int row:{0,1,2,3,4,5,12})seed[row]=1.0;
  const ActiveSetPolishResult polished=polish_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,seed,fixture.primal_tolerance);
  assert(polished.accepted&&polished.certificate.accepted&&polished.active_constraints==7);
  assert(polished.certificate.max_primal_excess==0.0&&polished.certificate.max_complementarity<=qp_complementarity_tolerance&&polished.certificate.max_stationarity<=qp_stationarity_tolerance);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  assert(replay.certificate.accepted&&replay.certificate.primal_constraints_pass&&replay.certificate.finite);
  assert(replay.certificate.max_complementarity<=qp_complementarity_tolerance&&replay.certificate.max_stationarity<=qp_stationarity_tolerance);
  const QPCertificate rechecked=check_cut_qp_kkt(fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,replay.lambda,replay.xi,fixture.primal_tolerance);
  assert(rechecked.accepted);
}

void test_active_set_refines_kkt_residual(cusolverDnHandle_t solver)
{
  const std::string fixture_path=covariance_fit_test_fixture_path("ja_reference_replay_v2.bin.qp_state.txt");
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);assert(fixture.rows.size()==28);
  std::vector<double> seed(fixture.rows.size(),0.0);for(const int row:{0,1,2,3,4,5,6,20,22,25,27})seed[row]=1.0;
  const ActiveSetPolishResult polished=polish_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,seed,fixture.primal_tolerance);
  assert(polished.accepted&&polished.certificate.accepted&&polished.active_constraints==11);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  assert(replay.certificate.accepted&&replay.certificate.primal_constraints_pass&&replay.certificate.finite);
  const QPCertificate rechecked=check_cut_qp_kkt(fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,replay.lambda,replay.xi,fixture.primal_tolerance);
  assert(rechecked.accepted&&rechecked.max_primal_excess==0.0);
  assert(rechecked.max_complementarity<=qp_complementarity_tolerance&&rechecked.max_stationarity<=qp_stationarity_tolerance);
}

void test_active_set_rank_deficient_warm_start(cusolverDnHandle_t solver)
{
  const std::string fixture_path=covariance_fit_test_fixture_path("ja_reference_replay_v4.bin.qp_state.txt");
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);assert(fixture.rows.size()==42);
  std::vector<double> seed(fixture.rows.size(),0.0);for(const int row:{0,1,2,3,5,6,20,22,24,27,28,30,35,38,39})seed[row]=1.0;
  const ActiveSetPolishResult polished=polish_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,seed,fixture.primal_tolerance);
  assert(polished.accepted&&polished.certificate.accepted&&polished.certificate.primal_slack.size()==42);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  const QPCertificate rechecked=check_cut_qp_kkt(fixture.svd,fixture.rows,fixture.rhs,fixture.a,fixture.b,replay.lambda,replay.xi,fixture.primal_tolerance);
  assert(replay.certificate.accepted&&rechecked.accepted&&rechecked.primal_slack.size()==42);
}

void test_outer128_stationarity_joint_polish(cusolverDnHandle_t solver)
{
  const std::string fixture_path=covariance_fit_test_fixture_path("ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt");
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);
  assert(fixture.svd.p==60&&fixture.rows.size()==766&&fixture.a.size()==766&&fixture.rhs.size()==766&&fixture.b.size()==766&&fixture.initial_lambda.size()==766);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  assert(replay.method==std::string("active_set_polish")&&replay.polish_attempts>0&&replay.polish_attempts<=3);
  assert(replay.certificate.accepted&&replay.certificate.finite&&replay.certificate.primal_constraints_pass);
  assert(replay.certificate.primal_slack.size()==766&&replay.lambda.size()==766);
  assert(replay.certificate.max_primal_excess==0.0&&replay.certificate.max_complementarity<=qp_complementarity_tolerance);
  assert(replay.certificate.max_stationarity<=qp_stationarity_tolerance);
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

std::vector<std::vector<double>> internal_basis(const std::vector<double>& sqrt_atom)
{
  const int n=static_cast<int>(sqrt_atom.size()),d=3*n;std::vector<std::vector<double>> basis;
  for(int i=0;i<d&&basis.size()<static_cast<std::size_t>(d-3);++i){std::vector<double> v(d,0.0);v[i]=1.0;project_translation(v,sqrt_atom,n);
    for(const auto& u:basis){const double dot=std::inner_product(v.begin(),v.end(),u.begin(),0.0);for(int k=0;k<d;++k)v[k]-=dot*u[k];}
    const double norm=std::sqrt(std::inner_product(v.begin(),v.end(),v.begin(),0.0));if(norm>1e-10){for(double& x:v)x/=norm;basis.push_back(std::move(v));}}
  assert(basis.size()==static_cast<std::size_t>(d-3));return basis;
}

std::vector<std::vector<double>> internal_basis(const int n){return internal_basis(std::vector<double>(n,1.0));}

void test_lanczos_finite_internal_space(cusolverDnHandle_t solver)
{
  for(const int n:{2,3}){
    const int d=3*n;std::vector<double> masses(n),sqrt_atom(n),sqrt_mass(d);for(int i=0;i<n;++i){masses[i]=i+1.0;sqrt_atom[i]=std::sqrt(masses[i]);}for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)sqrt_mass[axis*n+i]=sqrt_atom[i];
    const auto basis=internal_basis(sqrt_atom);const int internal=static_cast<int>(basis.size());Graph graph;graph.edges.push_back({0,1,0,{0,0,0}});
    const std::vector<double> theta={0.2,0.03,-0.02,0.25,0.04,0.3};
    for(const double scale:{1.0,1.0e4,1.0e8}){
      std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
      for(int k=0;k<internal;++k)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
        raw[static_cast<std::size_t>(i)*d+j]+=scale*(k+1)*basis[k][i]*basis[k][j]*sqrt_mass[i]*sqrt_mass[j];
      std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);
      stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);
      DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);
      std::vector<double> restricted(static_cast<std::size_t>(internal)*internal);
      for(int j=0;j<internal;++j){std::vector<double> product(d),add(d);baseline.apply(basis[j],product);apply_additive(graph,basis[j],sqrt_mass,sqrt_atom,theta,n,add);
        for(int q=0;q<d;++q)product[q]+=add[q];
        for(int i=0;i<internal;++i)restricted[static_cast<std::size_t>(i)*internal+j]=
          std::inner_product(basis[i].begin(),basis[i].end(),product.begin(),0.0);}
      const SmallEigen exact=eigen_small(solver,restricted,internal);
      const auto modes=lanczos_low_modes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,n,96,4,0.0);
      assert(modes.size()==static_cast<std::size_t>(std::min(4,internal)));
      for(int k=0;k<static_cast<int>(modes.size());++k){
        const auto& v=modes[k].vector;double norm2=0.0;
        for(double x:v)norm2+=x*x;
        check_close(norm2,1.0,2e-8);
        for(int axis=0;axis<3;++axis){double translation=0.0;for(int i=0;i<n;++i)translation+=sqrt_atom[i]*v[axis*n+i];assert(std::abs(translation)<2e-10);}
        for(int prior=0;prior<k;++prior)check_close(std::inner_product(v.begin(),v.end(),modes[prior].vector.begin(),0.0),0.0,2e-7);
        std::vector<double> product(d),add(d);baseline.apply(v,product);const double base_rayleigh=std::inner_product(v.begin(),v.end(),product.begin(),0.0);apply_additive(graph,v,sqrt_mass,sqrt_atom,theta,n,add);
        double residual2=0.0;for(int i=0;i<d;++i){product[i]+=add[i];const double residual=product[i]-modes[k].actual_rayleigh*v[i];residual2+=residual*residual;}
        check_close(modes[k].residual,std::sqrt(residual2),2e-7);
        const double rayleigh=std::inner_product(v.begin(),v.end(),product.begin(),0.0);
        check_close(modes[k].base_rayleigh,base_rayleigh,2e-7);
        check_close(modes[k].additive_rayleigh,rayleigh-base_rayleigh,2e-7);
        check_close(rayleigh,modes[k].value,2e-7);
        check_close(modes[k].value,exact.values[k],2e-7);
      }
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
  assert(nonfinite.witness.base.size()==d&&nonfinite.witness.add.size()==d&&nonfinite.witness.action.size()==d);
}

void test_production_true_residual_recovery()
{
  constexpr int n=2,d=3*n;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0),theta;
  const auto basis=internal_basis(n);std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;
  auto initialize=[&](DeviceBaseline& baseline,const std::vector<double>& matrix){
    std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(matrix.data()),matrix.size()*sizeof(double));stream.seekg(0);
    baseline.initialize(stream,0,d,n,masses,sqrt_mass);};
  auto make_failure=[&](const std::vector<double>& x){CGResult failed;failed.x=x;set_true_residual_failure(failed,7,12,basis[0],basis[0],1e-8,0.1,{});return failed;};
  std::vector<double> saved_x=basis[0];for(double& value:saved_x)value*=0.9;
  DeviceBaseline positive;initialize(positive,raw);CGResult recovered=make_failure(saved_x);CGProbeRecoveryDiagnostic recovered_diagnostic;
  assert(recover_cg_true_residual_failure(positive,Graph{},basis[0],recovered,sqrt_mass,sqrt_atom,theta,n,recovered_diagnostic,3));
  assert(recovered_diagnostic.recovered&&recovered_diagnostic.initial_cg_status=="TRUE_RESIDUAL_FAILURE");
  assert(recovered_diagnostic.verification_method=="COMPENSATED"&&recovered_diagnostic.initial_compensated_residual>1e-8);
  assert(recovered_diagnostic.final_compensated_residual<=1e-8&&recovered_diagnostic.refinement_rounds==1);
  assert(recovered.witness.classification=="TRUE_RESIDUAL_FAILURE"&&recovered_diagnostic.original_witness.solution==saved_x);
  for(int i=0;i<d;++i)check_close(recovered.x[i],basis[0][i],1e-8);

  std::fill(raw.begin(),raw.end(),0.0);DeviceBaseline singular;initialize(singular,raw);CGResult rejected=make_failure(saved_x);CGProbeRecoveryDiagnostic rejected_diagnostic;
  assert(!recover_cg_true_residual_failure(singular,Graph{},basis[0],rejected,sqrt_mass,sqrt_atom,theta,n,rejected_diagnostic));
  assert(!rejected_diagnostic.recovered&&rejected.x==saved_x&&rejected_diagnostic.original_witness.solution==saved_x);
  assert(rejected_diagnostic.refinement_status=="CORRECTION_NONPOSITIVE_OPERATOR_DIRECTION");
  assert(rejected.witness.classification=="TRUE_RESIDUAL_FAILURE"&&rejected_diagnostic.final_compensated_residual>1e-8);

  ResponseCheck response;response.cg_residual_status="CG_PASS";response.cg_verification_method="COMPENSATED";response.cg= recovered_diagnostic.final_compensated_residual;
  response.cg_probe_recoveries.push_back(recovered_diagnostic);std::ostringstream stats;write_response_stats(stats,response);
  assert(stats.str().find("cg_verification_method=COMPENSATED")!=std::string::npos);
  assert(stats.str().find("initial_cg_status=TRUE_RESIDUAL_FAILURE")!=std::string::npos);
  assert(stats.str().find("final_compensated_residual=")!=std::string::npos);
  RpmdJANativeFitOptions options;options.output_path="recovery_test";options.cutoff=4.0;options.temperature=300.0;options.fd_step=1e-3;options.sample_interval=10;options.epsilon=0.0;options.internal_mass_com=true;
  const CGReplayContext replay{options,30,11,12,11,12,1,1,0.9,0.9};
  const std::string base="rpmd_ja_recovered_witness_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".cg_witness.txt";
  const std::string expected_path=base+".outer_3.probe_7.txt";RemoveTestFile cleanup{expected_path};
  std::vector<std::string> saved_paths;write_recovered_cg_witnesses(response,base,theta,"qraw_fixture","samples_fixture",replay,saved_paths);
  assert(saved_paths.size()==1&&saved_paths.front()==expected_path);
  assert(response.cg_probe_recoveries.front().witness_snapshot_path==expected_path);
  const CGWitnessSnapshot recovered_snapshot=read_cg_witness(expected_path,d,0);
  assert(recovered_snapshot.classification=="TRUE_RESIDUAL_FAILURE"&&recovered_snapshot.probe==7);
}

void test_bounded_cg_replay_refinement()
{
  constexpr int n=2,d=3*n;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);
  const auto basis=internal_basis(n);std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;
  std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);std::vector<double> saved_x=basis[0];for(double& value:saved_x)value*=0.9;
  const std::vector<double> theta;const CGResidualComparison initial=compare_cg_true_residuals(baseline,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n);
  std::vector<double> matrix_before(raw.size());assert(cudaMemcpy(matrix_before.data(),baseline.k,raw.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  const CGReplayRefinement refined=refine_cg_replay_solution(baseline,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n,0,initial);
  assert(refined.status=="COMPENSATED_RESIDUAL_TARGET_REACHED"&&refined.rounds.size()==1);
  assert(refined.residuals.compensated_relative<initial.compensated_relative&&refined.residuals.compensated_relative<=1e-8);
  assert(refined.residuals.fast_relative<=1e-8&&refined.dense_matrix_actions<=4096);
  for(int i=0;i<d;++i)check_close(refined.solution[i],basis[0][i],1e-8);
  std::vector<double> matrix_after(raw.size());assert(cudaMemcpy(matrix_after.data(),baseline.k,raw.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  assert(matrix_after==matrix_before&&theta.empty());

  std::fill(raw.begin(),raw.end(),0.0);std::stringstream singular_stream(std::ios::in|std::ios::out|std::ios::binary);
  singular_stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));singular_stream.seekg(0);
  DeviceBaseline singular;singular.initialize(singular_stream,0,d,n,masses,sqrt_mass);
  const CGResidualComparison singular_initial=compare_cg_true_residuals(singular,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n);
  const CGReplayRefinement rejected=refine_cg_replay_solution(singular,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n,0,singular_initial);
  assert(rejected.status=="CORRECTION_NONPOSITIVE_OPERATOR_DIRECTION"&&rejected.rounds.size()==1);
  assert(rejected.rounds[0].status=="NONPOSITIVE_OPERATOR_DIRECTION"&&rejected.solution==saved_x);

  CGResidualComparison adverse_initial=initial;
  for(double& value:adverse_initial.compensated)value=-value;
  const CGReplayRefinement no_decrease=refine_cg_replay_solution(baseline,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n,0,adverse_initial);
  assert(no_decrease.status=="NO_RESIDUAL_DECREASE"&&no_decrease.rounds.size()==1);
  assert(no_decrease.solution==saved_x&&no_decrease.residuals.compensated==adverse_initial.compensated);

  std::vector<double> multi_mode_raw(static_cast<std::size_t>(d)*d,0.0);
  const double eigenvalues[]={1.0,100.0,10000.0};
  for(int mode=0;mode<3;++mode)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
    multi_mode_raw[static_cast<std::size_t>(i)*d+j]+=eigenvalues[mode]*basis[mode][i]*basis[mode][j];
  std::stringstream multi_mode_stream(std::ios::in|std::ios::out|std::ios::binary);
  multi_mode_stream.write(reinterpret_cast<const char*>(multi_mode_raw.data()),multi_mode_raw.size()*sizeof(double));multi_mode_stream.seekg(0);
  DeviceBaseline multi_mode;multi_mode.initialize(multi_mode_stream,0,d,n,masses,sqrt_mass);
  std::vector<double> multi_mode_rhs(d,0.0),zero_solution(d,0.0);
  for(int mode=0;mode<3;++mode)for(int i=0;i<d;++i)multi_mode_rhs[i]+=basis[mode][i];
  const CGResidualComparison multi_mode_initial=compare_cg_true_residuals(multi_mode,Graph{},multi_mode_rhs,zero_solution,sqrt_mass,sqrt_atom,theta,n);
  const CGReplayRefinement budget_limited=refine_cg_replay_solution(multi_mode,Graph{},multi_mode_rhs,zero_solution,sqrt_mass,sqrt_atom,theta,n,0,multi_mode_initial,1,7);
  assert(budget_limited.status=="MATRIX_ACTION_LIMIT"&&budget_limited.rounds.size()==1);
  assert(budget_limited.rounds[0].status=="MATRIX_ACTION_LIMIT"&&budget_limited.rounds[0].solver_counts_available);
  assert(budget_limited.rounds[0].iterations==2&&budget_limited.rounds[0].dense_matrix_actions==2);
  assert(budget_limited.dense_matrix_actions==2&&budget_limited.dense_matrix_actions<=7);
  assert(budget_limited.solution==zero_solution);
  std::ostringstream budget_report;write_cg_replay_refinement_round(budget_report,1,budget_limited.rounds[0]);
  assert(budget_report.str().find("iterations 2")!=std::string::npos);

  DeviceBaseline unavailable;unavailable.d=d;
  const CGReplayRefinement exception=refine_cg_replay_solution(unavailable,Graph{},basis[0],saved_x,sqrt_mass,sqrt_atom,theta,n,0,initial);
  assert(exception.status=="CORRECTION_CG_EXCEPTION"&&exception.rounds.size()==1);
  assert(!exception.rounds[0].solver_counts_available&&exception.solution==saved_x);
  std::ostringstream exception_report;write_cg_replay_refinement_round(exception_report,1,exception.rounds[0]);
  assert(exception_report.str().find("iterations NOT_AVAILABLE")!=std::string::npos);
  assert(exception_report.str().find("restarts NOT_AVAILABLE")!=std::string::npos);
}

void test_compensated_m1_local_action_recovers_cancellation()
{
  Graph graph;graph.type_pairs={{0,1}};graph.base_parameter_count=6;graph.parameter_count=7;graph.raw_local_parameter_count=2;
  graph.local_base_mix.assign(6,0.0);graph.local_transform={1.0,1.0};
  rpmd_ja_local_basis::EdgeInput edge{};edge.first={0,{0,0,0}};edge.second={1,{1,-2,0}};
  edge.axis={1.0,0.0,0.0};edge.longitudinal_parameter=0;edge.transverse_parameter=1;
  const auto terms=rpmd_ja_local_basis::make_edge_terms(edge);assert(!terms.empty());
  auto large=terms.front(),small=terms.front(),cancel=terms.front();
  large.parameter=0;large.weight=1.0e16;small.parameter=0;small.weight=1.0;cancel.parameter=1;cancel.weight=-1.0e16;
  graph.local_terms={large,small,cancel};
  const std::vector<double> q={0.0,1.0,0.0,0.0,0.0,0.0};
  const std::vector<double> sqrt_mass(6,1.0),sqrt_atom(2,1.0),theta={0,0,0,0,0,0,1.0};
  std::vector<double> fast(6),high(6),low(6);
  apply_additive(graph,q,sqrt_mass,sqrt_atom,theta,2,fast);
  apply_additive_compensated(graph,q,sqrt_mass,theta,2,high,low);
  assert(fast[0]==0.0&&fast[1]==0.0);
  check_close(high[0]+low[0],-1.0,1e-14);check_close(high[1]+low[1],1.0,1e-14);
  for(int axis=1;axis<3;++axis)for(int atom=0;atom<2;++atom)
    assert(high[axis*2+atom]==0.0&&low[axis*2+atom]==0.0);
}

void test_compensated_cg_true_residual_action()
{
  constexpr int n=3,d=3*n;DeviceBaseline baseline;baseline.d=d;
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.k),static_cast<std::size_t>(d)*d*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.q),d*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.out),d*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.compensated_high),d*sizeof(double))==cudaSuccess);
  assert(cudaMalloc(reinterpret_cast<void**>(&baseline.compensated_low),d*sizeof(double))==cudaSuccess);
  assert(cublasCreate(&baseline.blas)==CUBLAS_STATUS_SUCCESS);
  std::vector<double> matrix(static_cast<std::size_t>(d)*d,0.0);
  for(int i=1;i<d;++i)for(int j=0;j<d;++j)matrix[static_cast<std::size_t>(i)*d+j]=0.013*(i+1)-0.021*(j+1);
  matrix[0]=1.0e16;matrix[1]=1.0;matrix[2]=-1.0e16;
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  std::vector<double> x(d,1.0),high(d),low(d),fast(d);baseline.apply_compensated(x,high,low);baseline.apply(x,fast);
  for(int i=0;i<d;++i){NeumaierAccumulator expected;for(int j=0;j<d;++j)expected.add_product(matrix[static_cast<std::size_t>(i)*d+j],x[j]);
    check_close(high[i]+low[i],expected.value(),1e-12*std::max(1.0,std::abs(expected.value())));}
  check_close(high[0]+low[0],1.0,0.0);
  const std::vector<double> masses={1.0,2.0,3.0};std::vector<double> sqrt_atom(n),sqrt_mass(d);
  for(int i=0;i<n;++i)sqrt_atom[i]=std::sqrt(masses[i]);for(int a=0;a<3;++a)for(int i=0;i<n;++i)sqrt_mass[a*n+i]=sqrt_atom[i];
  Graph graph;graph.edges.push_back({0,1,0,{0,0,0}});graph.edges.push_back({1,2,0,{0,0,0}});
  const std::vector<double> theta={2.0,0.3,-0.2,1.7,0.4,2.3},rhs={0.2,-0.1,0.4,0.3,0.5,-0.2,0.1,-0.4,0.6};
  const CGResidualComparison comparison=compare_cg_true_residuals(baseline,graph,rhs,x,sqrt_mass,sqrt_atom,theta,n);
  assert(std::isfinite(comparison.fast_relative)&&std::isfinite(comparison.compensated_relative)&&std::isfinite(comparison.difference_norm));
  for(int axis=0;axis<3;++axis){double translation=0.0;for(int i=0;i<n;++i)translation+=sqrt_atom[i]*comparison.compensated[axis*n+i];assert(std::abs(translation)<1e-10);}
  const CGResidualComparison without_graph=compare_cg_true_residuals(baseline,Graph{},rhs,x,sqrt_mass,sqrt_atom,theta,n);
  double graph_difference=0.0;for(int i=0;i<d;++i)graph_difference=std::hypot(graph_difference,comparison.compensated[i]-without_graph.compensated[i]);
  assert(graph_difference>1e-8);

  std::fill(matrix.begin(),matrix.end(),0.0);matrix[static_cast<std::size_t>(n)*d+n]=1.0e16;
  matrix[static_cast<std::size_t>(n)*d+n+1]=1.0;matrix[static_cast<std::size_t>(n)*d+n+2]=-1.0e16;
  assert(cudaMemcpy(baseline.k,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  const double qi=3000000.0000000005,qj=7000000.000000001;
  const std::vector<double> precise_x={qi,qj,0.0,1.0,1.0,1.0,0.0,0.0,0.0};
  const std::vector<double> precise_sqrt_atom={3.0,7.0,2.0};
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)sqrt_mass[axis*n+i]=precise_sqrt_atom[i];
  Graph precise_graph;precise_graph.edges.push_back({0,1,0,{0,0,0}});
  const std::vector<double> precise_theta={80.0,0.0,0.0,0.0,0.0,0.0};
  std::vector<double> precise_rhs(d,0.0);precise_rhs[n]=std::nextafter(1.0,2.0);
  assert(qj/sqrt_mass[1]-qi/sqrt_mass[0]==0.0);
  const CGResidualComparison precise=compare_cg_true_residuals(baseline,precise_graph,precise_rhs,precise_x,
    sqrt_mass,precise_sqrt_atom,precise_theta,n);
  // Independent 90-digit reference, rounded to double for these exact binary inputs.
  const double expected[]={-5.9131592039077998e-10,2.5342110873890574e-10,0.0,
    1.8981232356494612e-16,-7.5208656506865438e-17,-2.1488187573390126e-17,0.0,0.0,0.0};
  const double tolerances[]={5e-24,5e-24,5e-24,5e-30,5e-30,5e-30,5e-30,5e-30,5e-30};
  for(int i=0;i<d;++i)assert(std::abs(precise.compensated[i]-expected[i])<tolerances[i]);
}

void test_replay_rejects_existing_report_before_reading_inputs()
{
  const std::string report="rpmd_ja_cg_replay_existing_report_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  RemoveTestFile cleanup{report};{std::ofstream out(report,std::ios::binary);out<<"preserve this report";assert(out.good());}
  Atom atom;Box box{};Force force;bool rejected_early=false;
  try{replay_rpmd_ja_native_cg("missing_witness_for_early_report_check",report,false,atom,box,force);}
  catch(const std::runtime_error& error){rejected_early=std::string(error.what()).find("report already exists")!=std::string::npos;}
  assert(rejected_early&&read_test_file(report)=="preserve this report");
}

void test_cg_witness_vector_line_endings()
{
  constexpr int d=6;CGWitness witness;witness.classification="TRUE_RESIDUAL_FAILURE";
  witness.probe=0;witness.iteration=1;witness.true_residual_computed=true;
  witness.rhs_input={1.0,-1.0,2.0,-2.0,3.0,-3.0};witness.rhs=witness.rhs_input;witness.solution.assign(d,0.0);
  const std::vector<double> theta={1.0,-2.0,0.0,3.0,-4.0,5.0};
  RpmdJANativeFitOptions options;options.cutoff=5.0;options.temperature=300.0;options.fd_step=1e-3;
  options.sample_interval=100;options.epsilon=1e-6;options.internal_mass_com=true;
  const CGReplayContext replay{options,30,11,12,11,12,1,1,0.9,0.9};
  const std::string path="rpmd_ja_cg_vector_parser_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  RemoveTestFile cleanup{path};write_cg_witness(path,witness,theta,"qraw_fixture","samples_fixture",replay);
  const std::string original=read_test_file(path);
  for(const bool trailing_space:{false,true}){
    std::istringstream lines(original);std::string line;std::ofstream out(path);
    while(std::getline(lines,line))out<<line<<(trailing_space?" \t\r\n":"\n");out.close();assert(out);
    for(const int parameters:{0,d}){const auto parsed=read_cg_witness(path,d,parameters);
      assert(parsed.theta==theta&&parsed.rhs_input==witness.rhs_input&&parsed.rhs_projected==witness.rhs&&parsed.solution==witness.solution);}
  }
  for(const char* key:{"theta","rhs_input_xyz_soa","rhs_projected_xyz_soa","solution_x_xyz_soa"}){
    for(const char* invalid:{"6 1 2 3 4 5","6 1 2 3 4 5 6 extra","6 1 2 3 4 5 nan","5 1 2 3 4 5"}){
      std::istringstream lines(original);std::string line;std::ofstream out(path);
      while(std::getline(lines,line)){if(line.compare(0,std::strlen(key)+1,std::string(key)+" ")==0)out<<key<<' '<<invalid<<'\n';else out<<line<<'\n';}
      out.close();assert(out);bool rejected=false;try{(void)read_cg_witness(path,d,d);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
    }
  }
  {std::istringstream lines(original);std::string line;std::ofstream out(path);
    while(std::getline(lines,line))if(line.rfind("witness_format_version ",0)!=0&&
      line.rfind("parameter_family ",0)!=0&&line.rfind("parameter_count ",0)!=0)out<<line<<'\n';
    out.close();assert(out);}
  const CGWitnessSnapshot legacy=read_cg_witness(path,d,d);
  assert(legacy.witness_format_version==1&&legacy.parameter_family=="M0"&&legacy.theta==theta);
}

void test_cg_failure_snapshot_round_trip_replay()
{
  const double offset=std::ldexp(1.0,-27);
  assert(compensated_dot({1.0+offset,1.0},{1.0-offset,-1.0})==-std::ldexp(1.0,-54));
  assert(std::fma(1.0+offset,1.0-offset,-1.0)==-std::ldexp(1.0,-54));
  constexpr int n=2,d=3*n;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);const auto basis=internal_basis(n);
  std::vector<double> input_rhs=basis.front();input_rhs[0]+=0.25;input_rhs[1]+=0.25;std::vector<double> projected_rhs=input_rhs;project_translation(projected_rhs,sqrt_atom,n);
  // A fixed near-singular SPD block makes the recursive and true residuals diverge reproducibly.
  const double off_diagonal=1.0-std::ldexp(1.0,-40);std::vector<double> ill_conditioned(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)ill_conditioned[static_cast<std::size_t>(i)*d+i]=1.0;
  for(int i=0;i<2;++i)for(int j=0;j<2;++j){const double value=(i==j?1.0:off_diagonal)-(i==j?1.0:0.0);
    for(int row=0;row<d;++row)for(int column=0;column<d;++column)ill_conditioned[static_cast<std::size_t>(row)*d+column]+=value*basis[i][row]*basis[j][column];}
  std::stringstream ill_input(std::ios::in|std::ios::out|std::ios::binary);ill_input.write(reinterpret_cast<const char*>(ill_conditioned.data()),ill_conditioned.size()*sizeof(double));ill_input.seekg(0);
  DeviceBaseline ill_baseline;ill_baseline.initialize(ill_input,0,d,n,masses,sqrt_mass);
  std::vector<double> zero_theta;for(int i=0;i<d;++i)input_rhs[i]+=0.9*basis[1][i];projected_rhs=input_rhs;project_translation(projected_rhs,sqrt_atom,n);
  constexpr double failure_epsilon=0.0;
  const CGResult failed=solve_projected_cg(ill_baseline,Graph{},input_rhs,sqrt_mass,sqrt_atom,zero_theta,n,7,failure_epsilon);
  assert(failed.witness.classification=="TRUE_RESIDUAL_FAILURE"&&failed.witness.finite&&failed.witness.probe==7);
  assert(failed.residual_restarts==2&&failed.witness.residual_restarts==2&&failed.witness.residual_checks.size()==3);
  for(int i=0;i<3;++i){const auto& check=failed.witness.residual_checks[i];assert(check.restarts==i&&check.recursive_relative<=1e-8&&check.true_relative>1e-8);
    if(i>0)assert(check.iteration>failed.witness.residual_checks[i-1].iteration);}
  assert(failed.witness.iteration==failed.witness.residual_checks.back().iteration);
  check_close(failed.witness.recursive_relative_residual,failed.witness.residual_checks.back().recursive_relative,0.0);
  check_close(failed.witness.relative_residual,failed.witness.residual_checks.back().true_relative,0.0);
  RpmdJANativeFitOptions options;options.output_path="test_candidate";options.cutoff=4.0;options.temperature=300.0;options.fd_step=1e-3;options.sample_interval=10;options.epsilon=failure_epsilon;options.internal_mass_com=true;
  const CGReplayContext replay{options,30,11,12,11,12,1,1,0.9,0.9};
  const std::string path="rpmd_ja_cg_replay_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";RemoveTestFile cleanup{path};
  const std::vector<double> theta=zero_theta;write_cg_witness(path,failed.witness,theta,"qraw_fixture","samples_fixture",replay);
  const std::string snapshot=read_test_file(path);assert(snapshot.find("classification TRUE_RESIDUAL_FAILURE")!=std::string::npos);
  assert(snapshot.find("curvature_diagnostic_status NOT_COMPUTED")!=std::string::npos);
  assert(snapshot.find("true_residual_status COMPUTED")!=std::string::npos);
  check_close(read_snapshot_scalar(snapshot,"true_relative_residual"),failed.witness.relative_residual,0.0);
  const double saved_epsilon=read_snapshot_scalar(snapshot,"epsilon");check_close(saved_epsilon,failure_epsilon,0.0);
  assert(snapshot.find("residual_restarts 2")!=std::string::npos&&snapshot.find("residual_checks 3")!=std::string::npos);
  for(const auto& check:failed.witness.residual_checks){std::ostringstream expected;expected<<"residual_check "<<check.iteration<<' '<<check.restarts<<' ';assert(snapshot.find(expected.str())!=std::string::npos);}
  assert(snapshot.find("qraw_model_fingerprint 11")!=std::string::npos&&snapshot.find("active_config_fingerprint 12")!=std::string::npos);
  const auto saved_rhs=read_snapshot_vector(snapshot,"rhs_input_xyz_soa"),saved_projected_rhs=read_snapshot_vector(snapshot,"rhs_projected_xyz_soa"),saved_theta=read_snapshot_vector(snapshot,"theta"),saved_x=read_snapshot_vector(snapshot,"solution_x_xyz_soa");
  assert(saved_rhs==input_rhs&&saved_projected_rhs==projected_rhs&&saved_rhs!=saved_projected_rhs&&saved_theta==theta&&saved_x==failed.x);
  const auto parsed_snapshot=read_cg_witness(path,d,0);assert(parsed_snapshot.classification=="TRUE_RESIDUAL_FAILURE"&&parsed_snapshot.qraw_path=="qraw_fixture");
  assert(parsed_snapshot.rhs_input==saved_rhs&&parsed_snapshot.rhs_projected==saved_projected_rhs&&parsed_snapshot.solution==saved_x);
  const auto parsed_paths=read_cg_witness_paths(path);assert(parsed_paths.first=="qraw_fixture"&&parsed_paths.second=="samples_fixture");
  const CGResult replayed=solve_projected_cg(ill_baseline,Graph{},saved_rhs,sqrt_mass,sqrt_atom,saved_theta,n,7,saved_epsilon);
  assert(replayed.witness.classification=="TRUE_RESIDUAL_FAILURE"&&replayed.residual_restarts==2&&replayed.witness.residual_checks.size()==3);
  for(int i=0;i<3;++i){check_close(replayed.witness.residual_checks[i].recursive_relative,failed.witness.residual_checks[i].recursive_relative,1e-10);
    check_close(replayed.witness.residual_checks[i].true_relative,failed.witness.residual_checks[i].true_relative,1e-10);}
  for(int i=0;i<d;++i)check_close(replayed.x[i],saved_x[i],1e-10);
  constexpr double early_epsilon=2.0;
  const CGResult early=solve_projected_cg(ill_baseline,Graph{},input_rhs,sqrt_mass,sqrt_atom,zero_theta,n,8,early_epsilon);assert(!early.witness.true_residual_computed);
  RpmdJANativeFitOptions early_options=options;early_options.epsilon=early_epsilon;const CGReplayContext early_replay{early_options,30,11,12,11,12,1,1,0.9,0.9};
  const std::string early_path=path+".early";RemoveTestFile early_cleanup{early_path};write_cg_witness(early_path,early.witness,zero_theta,"qraw_fixture","samples_fixture",early_replay);
  const std::string early_snapshot=read_test_file(early_path);assert(early_snapshot.find("recursive_relative_residual ")!=std::string::npos);
  check_close(read_snapshot_scalar(early_snapshot,"epsilon"),early_epsilon,0.0);
  assert(early_snapshot.find("true_residual_status NOT_COMPUTED")!=std::string::npos&&early_snapshot.find("true_relative_residual NOT_COMPUTED")!=std::string::npos);
  ResponseCheck early_response;early_response.cg=early.relative_residual;early_response.cg_residual_status=early.witness.classification;early_response.witness=early.witness;
  std::ostringstream early_trace;write_response_stats(early_trace,early_response);assert(early_trace.str().find("cg_true_residual_status=NOT_COMPUTED")!=std::string::npos);
  assert(early_trace.str().find("cg_recursive_relative_residual=NOT_RECORDED")==std::string::npos);
}

void test_lanczos_blindspot_cg_cut_feedback(cusolverDnHandle_t solver)
{
  constexpr int n=40,d=3*n,p=6;const double epsilon=1e-3;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> w(d);
  w[0]=1.0-1.0/n;for(int i=1;i<n;++i)w[i]=-1.0/n;std::vector<double> start(d);for(int i=0;i<d;++i)start[i]=std::sin((i+1)*1.6180339887498948)+std::cos((i+1)*0.7548776662466927);project_translation(start,sqrt_atom,n);double norm=std::sqrt(std::inner_product(start.begin(),start.end(),start.begin(),0.0));for(double& x:start)x/=norm;const double overlap=std::inner_product(w.begin(),w.end(),start.begin(),0.0);double start_x2=0.0;for(int i=0;i<n;++i)start_x2+=start[i]*start[i];for(int i=0;i<n;++i)w[i]-=(overlap/start_x2)*start[i];norm=std::sqrt(std::inner_product(w.begin(),w.end(),w.begin(),0.0));for(double& x:w)x/=norm;assert(std::abs(std::inner_product(w.begin(),w.end(),start.begin(),0.0))<1e-12);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;for(int i=0;i<d;++i)for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]-=2.0*w[i]*w[j];std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);Graph graph;for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)graph.edges.push_back({i,j,0,{0,0,0}});
  const auto modes=lanczos_low_modes(solver,baseline,graph,std::vector<double>(p,0.0),sqrt_mass,sqrt_atom,n,96,4,epsilon);assert(modes.front().value>0.99&&modes.front().residual<1e-10);std::vector<double> zero(p,0.0);const CGResult cg=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,zero,n,0,epsilon);assert(cg.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION");assert(cg.witness.rayleigh<0.0);const auto witness_modes=lanczos_low_modes(solver,baseline,graph,std::vector<double>(p,0.0),sqrt_mass,sqrt_atom,n,16,4,epsilon,cg.witness.direction);assert(witness_modes.front().value<-0.99&&witness_modes.front().residual<1e-10);
  SmallSVD identity;identity.p=p;identity.singular.assign(p,1.0);identity.eta.assign(p,0.0);identity.vt.assign(p*p,0.0);for(int i=0;i<p;++i)identity.vt[static_cast<std::size_t>(i)*p+i]=1.0;const auto row=spectral_row(graph,w,sqrt_mass,n,p);const double base=std::inner_product(w.begin(),w.end(),cg.witness.base.begin(),0.0);const auto qp=solve_cut_qp(solver,identity,{row},{epsilon-base},epsilon/4.0);const auto theta=theta_from_eta(identity,qp.xi);
  const auto basis=internal_basis(n);const int internal=static_cast<int>(basis.size());std::vector<double> restricted(static_cast<std::size_t>(internal)*internal);for(int j=0;j<internal;++j){std::vector<double> image(d);apply_total(baseline,graph,basis[j],sqrt_mass,sqrt_atom,theta,n,image);for(int i=0;i<internal;++i)restricted[static_cast<std::size_t>(i)*internal+j]=std::inner_product(basis[i].begin(),basis[i].end(),image.begin(),0.0);}const auto exact=eigen_small(solver,restricted,internal);assert(exact.values.front()>=epsilon-2e-10);const CGResult repaired=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,theta,n,0,epsilon);assert(repaired.witness.classification.empty());assert(repaired.relative_residual<=1e-8);
}

void test_lanczos_independent_seed_escapes_invariant_subspace(cusolverDnHandle_t solver)
{
  constexpr int n=2,d=6;const auto basis=internal_basis(n);const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> raw(d*d,0.0);
  const double eigenvalues[3]={1.0,-2.0,3.0};for(int k=0;k<3;++k)for(int i=0;i<d;++i)for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]+=eigenvalues[k]*basis[k][i]*basis[k][j];
  std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);Graph graph;
  DeviceLanczosWorkspace workspace;workspace.initialize(graph,sqrt_mass,sqrt_atom,8,0);
  const auto warm=lanczos_low_modes(solver,baseline,graph,{},sqrt_mass,sqrt_atom,n,8,3,0.0,basis[0],&workspace);const auto independent=lanczos_low_modes(solver,baseline,graph,{},sqrt_mass,sqrt_atom,n,8,3,0.0,{},&workspace);
  assert(warm.size()==1&&warm.front().residual<1e-12);check_close(warm.front().value,1.0,1e-12);assert(independent.front().value<-1.99&&independent.front().residual<1e-10);
}

struct RotatingCutResult{int qp_rounds=0;double minimum_curvature=0.0,curvature_tolerance=0.0;bool passed=false;};

RotatingCutResult run_rotating_cut_case(cusolverDnHandle_t solver,const double epsilon,const double cut_target)
{
  const std::vector<double> base={0.9499966591319635,-0.07971285149661844,-0.07971285149661844,1.8255837324986859};
  const std::vector<std::vector<double>> parameter_matrices={{1.5279445230841366,0.11974557011184332,0.11974557011184332,1.0465547385810312},
    {-0.029712482020309544,-0.7990636043681327,-0.7990636043681327,0.08472964953127277}};
  SmallSVD identity;identity.p=2;identity.singular={1.0,1.0};identity.eta={0.0,0.0};identity.vt={1.0,0.0,0.0,1.0};
  auto quadratic=[](const std::vector<double>& matrix,const std::vector<double>& v){return v[0]*(matrix[0]*v[0]+matrix[1]*v[1])+v[1]*(matrix[2]*v[0]+matrix[3]*v[1]);};
  std::vector<std::vector<double>> rows;std::vector<double> rhs,lambda,theta(2,0.0);RotatingCutResult result;
  for(int round=0;round<12;++round){std::vector<double> matrix=base;for(int parameter=0;parameter<2;++parameter)for(int i=0;i<4;++i)matrix[i]+=theta[parameter]*parameter_matrices[parameter][i];
    const SmallEigen eig=eigen_small(solver,matrix,2,1);const std::vector<double> v={eig.vectors[0],eig.vectors[1]};const double rayleigh=quadratic(matrix,v),base_rayleigh=quadratic(base,v);
    const std::vector<double> row={quadratic(parameter_matrices[0],v),quadratic(parameter_matrices[1],v)};double scale=0.0;
    const double curvature_tolerance=curvature_rounding_tolerance(epsilon,base_rayleigh,row,theta,scale);
    result.minimum_curvature=rayleigh;result.curvature_tolerance=curvature_tolerance;
    if(rayleigh>=epsilon-curvature_tolerance){result.qp_rounds=round;result.passed=true;return result;}
    rows.push_back(row);rhs.push_back(cut_target-base_rayleigh);const QPSolution qp=solve_cut_qp(solver,identity,rows,rhs,epsilon/4.0,lambda);
    theta=qp.certificate.theta;lambda=qp.lambda;
  }
  result.qp_rounds=12;return result;
}

void test_stability_cut_guard_reduces_rotating_mode_rounds(cusolverDnHandle_t solver)
{
  const double epsilon=1.0,cut_guard=epsilon*stability_cut_guard_fraction,cut_target=epsilon+cut_guard;
  assert(cut_guard>0.0&&cut_target>epsilon);
  const RotatingCutResult unguarded=run_rotating_cut_case(solver,epsilon,epsilon);
  const RotatingCutResult guarded=run_rotating_cut_case(solver,epsilon,cut_target);
  assert(unguarded.passed&&guarded.passed&&guarded.qp_rounds<unguarded.qp_rounds);
  assert(unguarded.minimum_curvature>=epsilon-unguarded.curvature_tolerance);
  assert(guarded.minimum_curvature>=epsilon-guarded.curvature_tolerance);
}

void test_lanczos_staged_early_exit(cusolverDnHandle_t solver)
{
  constexpr int n=12,d=3*n,steps=33;const auto basis=internal_basis(n);assert(basis.size()==static_cast<std::size_t>(steps));
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> initial(d,0.0),unstable_matrix(static_cast<std::size_t>(d)*d,0.0),positive_matrix(static_cast<std::size_t>(d)*d,0.0);
  for(int k=0;k<steps;++k){const double unstable_value=k==0?-0.5:0.5+0.1*k,positive_value=0.5+0.1*k;
    for(int i=0;i<d;++i){initial[i]+=basis[k][i];for(int j=0;j<d;++j){const double tile=basis[k][i]*basis[k][j];unstable_matrix[static_cast<std::size_t>(i)*d+j]+=unstable_value*tile;positive_matrix[static_cast<std::size_t>(i)*d+j]+=positive_value*tile;}}}
  auto initialize=[&](DeviceBaseline& baseline,const std::vector<double>& matrix){std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(matrix.data()),matrix.size()*sizeof(double));input.seekg(0);baseline.initialize(input,0,d,n,masses,sqrt_mass);};
  const Graph graph;const double epsilon=0.1;DeviceBaseline unstable;initialize(unstable,unstable_matrix);
  const auto early=lanczos_low_modes(solver,unstable,graph,{},sqrt_mass,sqrt_atom,n,steps,4,epsilon,initial);
  assert(early.front().early_exit&&early.front().steps_used==32&&early.front().dense_matrix_actions==early.front().steps_used+static_cast<int>(early.size()));
  for(const auto& mode:early)assert(mode.dense_matrix_actions==early.front().dense_matrix_actions);
  assert(early.front().actual_rayleigh<epsilon-early.front().curvature_tolerance&&early.front().residual>0.0);
  assert(early.front().curvature_scale>=1.0&&early.front().curvature_tolerance>0.0);
  DeviceBaseline positive;initialize(positive,positive_matrix);
  const auto complete=lanczos_low_modes(solver,positive,graph,{},sqrt_mass,sqrt_atom,n,steps,4,epsilon,initial);
  assert(!complete.front().early_exit&&complete.front().steps_used==steps);
  assert(complete.front().dense_matrix_actions==complete.front().steps_used+2*static_cast<int>(complete.size()));
  for(const auto& mode:complete)assert(mode.dense_matrix_actions==complete.front().dense_matrix_actions);
}

void test_lanczos_continues_across_checkpoints(cusolverDnHandle_t solver)
{
  constexpr int n=87,d=3*n,steps=d-3,wanted=4;const auto basis=internal_basis(n);assert(basis.size()==static_cast<std::size_t>(steps));
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> initial(d,0.0),matrix(static_cast<std::size_t>(d)*d,0.0);
  for(int k=0;k<steps;++k){const double eigenvalue=0.5+0.01*k;for(int i=0;i<d;++i){initial[i]+=basis[k][i];for(int j=0;j<d;++j)matrix[static_cast<std::size_t>(i)*d+j]+=eigenvalue*basis[k][i]*basis[k][j];}}
  std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(matrix.data()),matrix.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);
  const auto modes=lanczos_low_modes(solver,baseline,Graph{}, {},sqrt_mass,sqrt_atom,n,steps,wanted,0.1,initial);
  const int expected_actions=steps+5*wanted;
  assert(modes.size()==wanted&&!modes.front().early_exit&&modes.front().steps_used==steps);
  for(const auto& mode:modes){assert(mode.steps_used==steps&&mode.dense_matrix_actions==expected_actions);assert(mode.actual_rayleigh>0.1&&mode.residual<1e-8);}
}

void test_lanczos_iteration_budget_interrupts_one_overrun(cusolverDnHandle_t solver)
{
  constexpr int n=8,d=3*n;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
  for(int coordinate=0;coordinate<d;++coordinate)raw[static_cast<std::size_t>(coordinate)*d+coordinate]=1.0;
  std::stringstream raw_stream(std::ios::in|std::ios::out|std::ios::binary);
  raw_stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));raw_stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(raw_stream,0,d,n,masses,sqrt_mass);
  rpmd_ja_feasibility::Limits limits;limits.wall_seconds=100.0;limits.validation_reserve_fraction=0.25;
  limits.max_fit_family_wall_seconds=32.5;
  rpmd_ja_feasibility::Budget budget;
  assert(rpmd_ja_feasibility::begin(budget,limits,0.0)==rpmd_ja_feasibility::BudgetResult::OK);
  assert(rpmd_ja_feasibility::enter_phase(budget,limits,rpmd_ja_feasibility::Phase::A1,10.0)==
    rpmd_ja_feasibility::BudgetResult::OK);
  assert(rpmd_ja_feasibility::enter_phase(budget,limits,rpmd_ja_feasibility::Phase::B,10.0)==
    rpmd_ja_feasibility::BudgetResult::OK);
  int clock_checks=0;double maximum_step_seconds=0.0;bool stopped_on_family_wall=false;
  const std::function<void(int,double)> injected_clock=[&](const int,double step_seconds){
    maximum_step_seconds=std::max(maximum_step_seconds,step_seconds);
    const double now=clock_checks++==0?42.0:43.25;
    const auto result=rpmd_ja_feasibility::observe_time(budget,limits,now);
    if(result!=rpmd_ja_feasibility::BudgetResult::OK){
      stopped_on_family_wall=result==rpmd_ja_feasibility::BudgetResult::FAMILY_WALL_LIMIT;
      throw std::runtime_error("FEASIBILITY_BUDGET_FAMILY_WALL_LIMIT");}
  };
  bool stopped=false;
  try{(void)lanczos_low_modes(solver,baseline,Graph{}, {},sqrt_mass,sqrt_atom,n,8,1,0.0,{},nullptr,injected_clock);}
  catch(const std::runtime_error& error){stopped=std::string(error.what())=="FEASIBILITY_BUDGET_FAMILY_WALL_LIMIT";}
  assert(stopped&&stopped_on_family_wall&&clock_checks==2&&maximum_step_seconds>=0.0);
  const double overrun=budget.phase_seconds[rpmd_ja_feasibility::phase_index(rpmd_ja_feasibility::Phase::B)]-
    limits.max_fit_family_wall_seconds;
  assert(std::abs(overrun-0.75)<1.0e-12);
  assert(rpmd_ja_feasibility::enter_phase(budget,limits,rpmd_ja_feasibility::Phase::C,43.25)==
    rpmd_ja_feasibility::BudgetResult::OK);
  const double c_available=limits.wall_seconds-budget.elapsed_seconds-
    limits.wall_seconds*limits.validation_reserve_fraction;
  assert(std::abs(c_available-(limits.max_fit_family_wall_seconds-overrun))<1.0e-12);
}

void test_lanczos_curvature_rounding_boundary(cusolverDnHandle_t solver)
{
  constexpr int n=12,d=3*n,steps=3*n-3;const double epsilon=1.0,stiffness=1.0e8;const auto basis=internal_basis(n);
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);Graph graph;
  for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)graph.edges.push_back({i,j,0,{0,0,0}});
  std::vector<double> theta(6,0.0);theta[0]=theta[3]=theta[5]=-stiffness;const auto row=spectral_row(graph,basis.front(),sqrt_mass,n,6);double reference_scale=0.0;
  const double tolerance=curvature_rounding_tolerance(epsilon,stiffness*n+epsilon,row,theta,reference_scale);
  assert(tolerance>0.0&&reference_scale>1.0);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0),initial(d,0.0);
  for(int axis=0;axis<3;++axis)for(int i=0;i<n;++i)for(int j=0;j<n;++j)raw[static_cast<std::size_t>(axis*n+i)*d+axis*n+j]+=stiffness*(i==j?n-1:-1);
  for(int k=0;k<steps;++k){const double eigenvalue=epsilon-tolerance*(0.25+0.5*static_cast<double>(k)/(steps-1));
    for(int i=0;i<d;++i){initial[i]+=basis[k][i];for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]+=eigenvalue*basis[k][i]*basis[k][j];}}
  std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);
  const auto modes=lanczos_low_modes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,n,steps,4,epsilon,initial);
  assert(!modes.front().early_exit&&modes.front().steps_used==steps);
  assert(modes.front().actual_rayleigh<epsilon&&modes.front().actual_rayleigh>=epsilon-modes.front().curvature_tolerance);
}

void test_lanczos_deep_search_does_not_early_cut(cusolverDnHandle_t solver)
{
  constexpr int n=66,d=3*n,internal=d-3;const double epsilon=0.1;const auto basis=internal_basis(n);
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> initial(d,0.0),matrix(static_cast<std::size_t>(d)*d,0.0);
  for(int k=0;k<internal;++k){const double eigenvalue=k==0?-0.5:0.5+0.1*k;
    for(int i=0;i<d;++i){initial[i]+=basis[k][i];for(int j=0;j<d;++j)matrix[static_cast<std::size_t>(i)*d+j]+=eigenvalue*basis[k][i]*basis[k][j];}}
  std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(matrix.data()),matrix.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);
  const auto coarse=lanczos_low_modes(solver,baseline,Graph{}, {},sqrt_mass,sqrt_atom,n,96,4,epsilon,initial);
  assert(coarse.front().early_exit&&coarse.front().steps_used==32&&coarse.front().actual_rayleigh<epsilon-coarse.front().curvature_tolerance);
  const auto deep=lanczos_low_modes(solver,baseline,Graph{}, {},sqrt_mass,sqrt_atom,n,192,4,epsilon,initial);
  assert(!deep.front().early_exit&&deep.front().steps_used==192&&deep.front().actual_rayleigh<epsilon-deep.front().curvature_tolerance);
  assert(deep.front().dense_matrix_actions==192+4*4);
}

std::vector<double> solve_dense_reference(std::vector<double> matrix,std::vector<double> rhs,const int n)
{
  for(int k=0;k<n;++k){int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(matrix[static_cast<std::size_t>(i)*n+k])>
      std::abs(matrix[static_cast<std::size_t>(pivot)*n+k]))pivot=i;
    assert(std::abs(matrix[static_cast<std::size_t>(pivot)*n+k])>1e-14);
    if(pivot!=k){for(int j=k;j<n;++j)std::swap(matrix[static_cast<std::size_t>(k)*n+j],matrix[static_cast<std::size_t>(pivot)*n+j]);std::swap(rhs[k],rhs[pivot]);}
    for(int i=k+1;i<n;++i){const double factor=matrix[static_cast<std::size_t>(i)*n+k]/matrix[static_cast<std::size_t>(k)*n+k];
      for(int j=k+1;j<n;++j)matrix[static_cast<std::size_t>(i)*n+j]-=factor*matrix[static_cast<std::size_t>(k)*n+j];rhs[i]-=factor*rhs[k];}}
  for(int i=n-1;i>=0;--i){for(int j=i+1;j<n;++j)rhs[i]-=matrix[static_cast<std::size_t>(i)*n+j]*rhs[j];rhs[i]/=matrix[static_cast<std::size_t>(i)*n+i];}
  return rhs;
}

void test_probe_covariance_and_ibp(cusolverDnHandle_t solver)
{
  constexpr int n=8,d=3*n,train=21,validation=33,frames=train+validation;
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
    for(int t=0;t<validation;++t){std::vector<double> x(d);for(int k=0;k<r;++k){const int frequency=k/2+1;const double phase=2.0*std::acos(-1.0)*frequency*t/validation;const double value=std::sqrt(2.0*kbt/eig.values[k])*(k%2?std::sin(phase):std::cos(phase));for(int i=0;i<d;++i)x[i]+=value*modes[k][i];}frame(step++,x);}
    assert(out.good());
  }
  std::ifstream spool(path,std::ios::binary);assert(spool.good());
  const ResponseCheck result=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{});
  assert(result.probes>=20);assert(result.validation_frames==validation&&result.statistical_frames==33);assert(result.observed_cov_available&&result.predicted_cov_available&&result.cg_residual_available&&result.response_ibp_available&&result.force_residual_available);assert(result.observed_cov_min>0.0&&result.observed_cov_max>=result.observed_cov_min&&result.observed_cov_condition>=1.0);assert(result.predicted_cov_min>0.0&&result.predicted_cov_max>=result.predicted_cov_min&&result.predicted_cov_condition>=1.0);assert(result.block_frames[0]+result.block_frames[1]+result.block_frames[2]+result.block_frames[3]==validation);assert(std::isfinite(result.block_variance_max_relative_delta));assert(result.cg<=1e-8);assert(result.response<=2e-8);assert(result.ibp<=2e-8);assert(result.force_residual<=2e-8);

  int budget_observations=0;bool returned_after_budget_stop=false,budget_stop_observed=false;
  const std::function<void()> stop_during_cg=[&](){
    if(++budget_observations==3)throw std::runtime_error("injected mid-CG feasibility budget stop");};
  std::vector<std::vector<double>> budget_probes(1,std::vector<double>(d));
  for(int i=0;i<d;++i)budget_probes[0][i]=static_cast<double>(i+1);
  std::vector<std::string> budget_probe_sources={"BUDGET_TEST"};
  std::vector<ProbeOrigin> budget_probe_origins(1);
  try{(void)validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,train,1,
      temperature,{},0.0,nullptr,0.15,0.15,-1,{},false,nullptr,false,-1,
      &budget_probes,&budget_probe_sources,&budget_probe_origins,stop_during_cg);
    returned_after_budget_stop=true;
  }catch(const std::runtime_error& error){budget_stop_observed=std::string(error.what())=="injected mid-CG feasibility budget stop";}
  assert(budget_stop_observed&&!returned_after_budget_stop&&budget_observations==3);

  const std::vector<RitzMode> common_soft_modes={modes.front()};
  std::vector<std::string> frozen_common_sources;std::vector<ProbeOrigin> frozen_common_origins;
  const auto frozen_common_probes=make_common_probe_set(common_soft_modes,sqrt_atom,types,n,
    &frozen_common_sources,&frozen_common_origins,nullptr,validation-1);
  const ResponseCheck development_common=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,
    types,n,r0,spool,0,frames,train,1,temperature,common_soft_modes,0.0,nullptr,0.15,0.15,-1,{},false,nullptr,true,
    validation-1,&frozen_common_probes,&frozen_common_sources,&frozen_common_origins);
  const std::streamoff blind_frame_start=static_cast<std::streamoff>(train)*
    static_cast<std::streamoff>(sizeof(double)+2*d*sizeof(double));
  const ResponseCheck blind_common=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,
    types,n,r0,spool,blind_frame_start,validation,0,1,temperature,common_soft_modes,0.0,nullptr,0.15,0.15,-1,{},false,nullptr,true,
    validation-1,&frozen_common_probes,&frozen_common_sources,&frozen_common_origins);
  assert(frozen_common_sources==development_common.probe_sources);
  assert(frozen_common_sources==blind_common.probe_sources);
  assert(frozen_common_probes.size()==static_cast<std::size_t>(development_common.probes));
  assert(frozen_common_probes.size()==static_cast<std::size_t>(blind_common.probes));

  const auto fail_first_probe=[](const int probe,const std::vector<double>&){
    CGResult cg;cg.relative_residual=1.0;cg.witness.classification="CG_SOLVER_BUDGET_EXHAUSTED";cg.witness.probe=probe;return cg;};
  const ResponseCheck incomplete=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,
    train,1,temperature,common_soft_modes,0.0,nullptr,0.15,0.15,-1,fail_first_probe,true,&common_physical_probes,true,
    validation-1,&frozen_common_probes,&frozen_common_sources,&frozen_common_origins);
  assert(incomplete.probes==static_cast<int>(frozen_common_probes.size()));
  assert(!incomplete.observed_cov_available&&!incomplete.predicted_cov_available&&!incomplete.response_ibp_available);
  assert(incomplete.mean_q.empty()&&incomplete.observed_matrix.empty()&&incomplete.centered_covariance_matrix.empty());
  for(const char* consumer:{"A0_DEVELOPMENT","BLIND"}){
    std::ostringstream diagnostic;write_response_stats(diagnostic,incomplete);
    assert(!write_response_consumer_statistics(diagnostic,consumer,incomplete));
    assert(diagnostic.str().find(std::string(consumer)+"_statistics_status NOT_COMPUTED")!=std::string::npos);
    assert(diagnostic.str().find("fixed_R0_mean_norm=NOT_COMPUTED")!=std::string::npos);
  }

  const auto response_probes=make_probes({},sqrt_atom,types,n);const int m=static_cast<int>(response_probes.size());assert(m==result.probes);
  std::vector<double> reference_full(static_cast<std::size_t>(d)*d,0.0);
  for(int i=0;i<d;++i)reference_full[static_cast<std::size_t>(i)*d+i]=1.0;
  const double edge_block[3][3]={{theta[0],theta[1],theta[2]},{theta[1],theta[3],theta[4]},{theta[2],theta[4],theta[5]}};
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){const int ai=a*n,aj=ai+1,bi=b*n,bj=bi+1;const double value=edge_block[a][b];
    reference_full[static_cast<std::size_t>(ai)*d+bi]+=value;reference_full[static_cast<std::size_t>(ai)*d+bj]-=value;
    reference_full[static_cast<std::size_t>(aj)*d+bi]-=value;reference_full[static_cast<std::size_t>(aj)*d+bj]+=value;}
  std::vector<double> reference_restricted(static_cast<std::size_t>(r)*r,0.0);
  for(int k=0;k<r;++k)for(int l=0;l<r;++l)for(int i=0;i<d;++i)for(int j=0;j<d;++j)
    reference_restricted[static_cast<std::size_t>(k)*r+l]+=basis[k][i]*reference_full[static_cast<std::size_t>(i)*d+j]*basis[l][j];
  for(std::size_t i=0;i<restricted.size();++i)check_close(reference_restricted[i],restricted[i],2e-12);
  std::vector<double> expected_predicted(static_cast<std::size_t>(m)*m);
  for(int j=0;j<m;++j){std::vector<double> rhs_coefficients(r);
    for(int k=0;k<r;++k)rhs_coefficients[k]=std::inner_product(basis[k].begin(),basis[k].end(),response_probes[j].begin(),0.0);
    const std::vector<double> solution_coefficients=solve_dense_reference(reference_restricted,rhs_coefficients,r);std::vector<double> solution(d,0.0);
    for(int k=0;k<r;++k)for(int i=0;i<d;++i)solution[i]+=basis[k][i]*solution_coefficients[k];
    for(int i=0;i<m;++i)expected_predicted[static_cast<std::size_t>(i)*m+j]=kbt*std::inner_product(response_probes[i].begin(),response_probes[i].end(),solution.begin(),0.0);}
  std::vector<double> injected_original_solution;bool recovery_injected=false;
  const auto solve_probe=[&](const int probe,const std::vector<double>& rhs){
    CGResult cg=solve_projected_cg(baseline,graph,rhs,sqrt_mass,sqrt_atom,theta,n,probe,0.0);
    if(probe==0){for(double& value:cg.x)value*=1.0-1e-6;injected_original_solution=cg.x;const auto comparison=compare_cg_true_residuals(baseline,graph,rhs,cg.x,sqrt_mass,sqrt_atom,theta,n);
      std::vector<double> projected_rhs=rhs;project_translation(projected_rhs,sqrt_atom,n);
      set_true_residual_failure(cg,probe,cg.iterations,rhs,projected_rhs,5e-9,comparison.fast_relative,{});recovery_injected=true;}
    return cg;};
  const ResponseCheck recovered_result=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{},0.0,nullptr,0.15,0.15,42,solve_probe);
  assert(recovery_injected&&recovered_result.cg_residual_status=="CG_PASS"&&recovered_result.cg_verification_method=="MIXED");
  assert(recovered_result.cg_probe_recoveries.size()==1&&recovered_result.cg_probe_recoveries[0].recovered);
  assert(recovered_result.observed_cov_available&&recovered_result.predicted_cov_available&&recovered_result.response_ibp_available&&recovered_result.force_residual_available);
  assert(recovered_result.cg<=1e-8&&recovered_result.cg_probe_recoveries[0].original_witness.solution==injected_original_solution);
  assert(recovered_result.predicted_matrix.size()==expected_predicted.size());
  for(std::size_t i=0;i<expected_predicted.size();++i)check_close(recovered_result.predicted_matrix[i],expected_predicted[i],2e-8);
  std::ostringstream recovery_stats;write_response_stats(recovery_stats,recovered_result);assert(recovery_stats.str().find("cg_verification_method=MIXED")!=std::string::npos);
  RpmdJANativeFitOptions replay_options;replay_options.output_path="recovery_integration_test";replay_options.cutoff=4.0;replay_options.temperature=temperature;replay_options.fd_step=1e-3;replay_options.sample_interval=1;replay_options.epsilon=0.0;replay_options.internal_mass_com=true;
  const CGReplayContext replay_context{replay_options,frames,11,12,11,12,1,1,0.9,0.9};
  const std::string recovery_base="rpmd_ja_response_recovered_witness_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".cg_witness.txt";
  const std::string recovery_path=recovery_base+".outer_42.probe_0.txt";RemoveTestFile remove_recovery{recovery_path};std::vector<std::string> saved_recovery_paths;
  ResponseCheck mutable_recovered_result=recovered_result;
  write_recovered_cg_witnesses(mutable_recovered_result,recovery_base,theta,"qraw_fixture","samples_fixture",replay_context,saved_recovery_paths);
  assert(saved_recovery_paths.size()==1&&saved_recovery_paths[0]==recovery_path);
  const CGWitnessSnapshot saved_recovery=read_cg_witness(recovery_path,d,static_cast<int>(theta.size()));
  assert(saved_recovery.classification=="TRUE_RESIDUAL_FAILURE"&&saved_recovery.solution==injected_original_solution);

  spool.close();
  {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);assert(out.good());
    auto frame=[&](double step,const std::vector<double>& x){out.write(reinterpret_cast<const char*>(&step),sizeof(step));out.write(reinterpret_cast<const char*>(x.data()),d*sizeof(double));std::vector<double> add(d),f(d);apply_additive(graph,x,sqrt_mass,sqrt_atom,theta,n,add);for(int i=0;i<d;++i)f[i]=-x[i]-add[i];out.write(reinterpret_cast<const char*>(f.data()),d*sizeof(double));};
    std::vector<double> zero(d);double step=0.0;for(int i=0;i<train;++i)frame(step++,zero);
    for(int i=0;i<validation;++i)frame(step++,zero);
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
    for(int t=0;t<validation;++t){std::vector<double> x(d);for(int k=0;k<r;++k){const int frequency=k/2+1;const double phase=2.0*std::acos(-1.0)*frequency*t/validation;const double value=std::sqrt(2.0*kbt)*(k%2?std::sin(phase):std::cos(phase));for(int i=0;i<d;++i)x[i]+=value*basis[k][i];}frame(step++,x);}
    assert(out.good());
  }
  spool.open(path,std::ios::binary);assert(spool.good());
  const ResponseCheck zero_result=validate_probes(solver,baseline,graph,zero_theta,sqrt_mass,sqrt_atom,types,n,r0,spool,0,frames,21,1,temperature,{});
  assert(zero_result.force_residual<=2e-8);assert(zero_result.response<=2e-8);assert(zero_result.ibp<=2e-8);
}

void test_probe_selection_skips_duplicate_low_modes()
{
  const int n=40,d=3*n;std::vector<double> sqrt_atom(n,1.0);std::vector<int> types(n);
  for(int i=0;i<n;++i)types[i]=i%4;
  const std::vector<RitzMode> none;const auto base=make_probes(none,sqrt_atom,types,n);assert(base.size()>=16);
  std::vector<RitzMode> candidates(6);for(auto& mode:candidates)mode.vector=base.front();
  for(int k=0;k<8;++k){RitzMode mode;mode.vector.assign(d,0.0);mode.vector[50+k]=1.0;candidates.push_back(std::move(mode));}
  const auto selected=make_probes(candidates,sqrt_atom,types,n);assert(selected.size()==base.size()+4);
  for(std::size_t i=0;i<selected.size();++i)for(std::size_t j=0;j<i;++j)assert(std::abs(std::inner_product(selected[i].begin(),selected[i].end(),selected[j].begin(),0.0))<1e-10);
}

void test_common_low_ritz_modes_get_probe_rank_before_fixed_channels()
{
  constexpr int n=2,d=3*n;const std::vector<double> sqrt_atom(n,1.0);const std::vector<int> types={0,1};
  std::vector<RitzMode> modes(3);for(int axis=0;axis<3;++axis){modes[axis].vector.assign(d,0.0);
    modes[axis].vector[axis*n]=1.0;modes[axis].vector[axis*n+1]=-1.0;}
  FixedPhysicalProbe fixed;fixed.source="FIXED_PHYSICAL_TEST";fixed.vector.assign(d,0.0);fixed.vector[0]=1.0;
  const std::vector<FixedPhysicalProbe> fixed_channels={fixed};
  std::vector<std::string> frozen_sources,development_sources,blind_sources;
  const auto frozen=make_common_probe_set(modes,sqrt_atom,types,n,&frozen_sources,nullptr,&fixed_channels);
  const auto development=make_common_probe_set(modes,sqrt_atom,types,n,&development_sources,nullptr,&fixed_channels);
  const auto blind=make_common_probe_set(modes,sqrt_atom,types,n,&blind_sources,nullptr,&fixed_channels);
  assert(frozen.size()==3&&frozen_sources.size()==3);
  assert(std::count(frozen_sources.begin(),frozen_sources.end(),"LOW_RITZ")==3);
  assert(frozen_sources==development_sources&&frozen_sources==blind_sources);
  assert(frozen.size()==development.size()&&frozen.size()==blind.size());
  for(std::size_t probe=0;probe<frozen.size();++probe){
    assert(frozen[probe].size()==development[probe].size()&&frozen[probe].size()==blind[probe].size());
    for(std::size_t coordinate=0;coordinate<frozen[probe].size();++coordinate){
      assert(frozen[probe][coordinate]==development[probe][coordinate]);
      assert(frozen[probe][coordinate]==blind[probe][coordinate]);
    }
  }
  std::vector<std::string> fixed_sources;
  (void)make_probes({},sqrt_atom,types,n,&fixed_sources,nullptr,&fixed_channels);
  const std::size_t selected_fixed_rank=frozen.size()-3;
  assert(fixed_sources.size()>selected_fixed_rank);
  assert(rpmd_ja_feasibility::common_soft_probe_capacity(3,4,28)==3);
}

void test_common_probe_rank_is_capped_for_nearly_full_fixed_basis(cusolverDnHandle_t solver)
{
  constexpr int n=8,d=3*n,validation_frames=2048,maximum_probe_rank=20;
  const double temperature=300.0,kbt=K_B*temperature;
  const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0),r0(d,0.0);
  std::vector<int> types(n);for(int atom=0;atom<n;++atom)types[atom]=atom%4;
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);
  for(int coordinate=0;coordinate<d;++coordinate)raw[static_cast<std::size_t>(coordinate)*d+coordinate]=1.0;
  std::stringstream raw_stream(std::ios::in|std::ios::out|std::ios::binary);
  raw_stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));raw_stream.seekg(0);
  DeviceBaseline baseline;baseline.initialize(raw_stream,0,d,n,masses,sqrt_mass);
  std::vector<std::string> fixed_only_sources,uncapped_sources;
  std::vector<ProbeOrigin> fixed_only_origins,uncapped_origins;
  const auto fixed_only=make_probes({},sqrt_atom,types,n,&fixed_only_sources,&fixed_only_origins);
  assert(fixed_only.size()==20&&fixed_only_sources.size()==20&&fixed_only_origins.size()==20);
  RitzMode soft;soft.vector=internal_basis(n).front();const std::vector<RitzMode> soft_modes={soft};
  FixedPhysicalProbe physical;physical.source="FIXED_PHYSICAL_TEST";physical.vector=internal_basis(n).back();
  const std::vector<FixedPhysicalProbe> fixed_physical={physical};
  const auto uncapped=make_common_probe_set(soft_modes,sqrt_atom,types,n,&uncapped_sources,&uncapped_origins,
    &fixed_physical);
  assert(uncapped.size()==21&&uncapped_sources.size()==21&&uncapped_origins.size()==21);
  std::vector<std::string> frozen_sources;std::vector<ProbeOrigin> frozen_origins;
  const auto frozen=make_common_probe_set(soft_modes,sqrt_atom,types,n,&frozen_sources,&frozen_origins,&fixed_physical,
    maximum_probe_rank);
  assert(frozen.size()==maximum_probe_rank&&frozen_sources.size()==frozen.size()&&frozen_origins.size()==frozen.size());
  const auto count_source=[](const std::vector<std::string>& sources,const char* value){
    return std::count(sources.begin(),sources.end(),value);};
  const auto count_type_origins=[](const std::vector<ProbeOrigin>& origins){
    return std::count_if(origins.begin(),origins.end(),[](const ProbeOrigin& origin){return origin.seed_atom_index>=0;});};
  assert(count_source(fixed_only_sources,"LOW_RITZ")==0&&count_source(uncapped_sources,"LOW_RITZ")==1&&
    count_source(frozen_sources,"LOW_RITZ")==1);
  assert(count_source(fixed_only_sources,"FIXED_RANDOM")==16&&
    count_source(uncapped_sources,"FIXED_RANDOM")==16&&count_source(frozen_sources,"FIXED_RANDOM")==16);
  assert(count_source(uncapped_sources,"FIXED_PHYSICAL_TEST")==1&&count_source(frozen_sources,"FIXED_PHYSICAL_TEST")==1);
  assert(count_source(fixed_only_sources,"TYPE_LOCAL")==4&&count_source(uncapped_sources,"TYPE_LOCAL")==3&&
    count_source(frozen_sources,"TYPE_LOCAL")==2);
  assert(count_type_origins(fixed_only_origins)==4&&count_type_origins(uncapped_origins)==3&&
    count_type_origins(frozen_origins)==2);
  for(std::size_t index=0;index<frozen_sources.size();++index)
    assert(probe_origin_valid(frozen_sources[index],frozen_origins[index]));
  const std::string path="rpmd_ja_nearly_full_common_probe_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
  struct RemoveFile{std::string path;~RemoveFile(){std::remove(path.c_str());}} cleanup{path};
  {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);assert(out.good());
    for(int frame=0;frame<validation_frames;++frame){
      const double step=static_cast<double>(frame);std::vector<double> x(d,0.0),force(d,0.0);
      for(int probe=0;probe<maximum_probe_rank;++probe){const int frequency=probe/2+1;
        const double phase=2.0*std::acos(-1.0)*frequency*(frame%32)/32.0;
        const double value=std::sqrt(2.0*kbt)*(probe%2?std::sin(phase):std::cos(phase));
        for(int coordinate=0;coordinate<d;++coordinate)x[coordinate]+=value*frozen[probe][coordinate];}
      for(int coordinate=0;coordinate<d;++coordinate)force[coordinate]=-x[coordinate];
      out.write(reinterpret_cast<const char*>(&step),sizeof(step));
      out.write(reinterpret_cast<const char*>(x.data()),d*sizeof(double));
      out.write(reinterpret_cast<const char*>(force.data()),d*sizeof(double));
    }
    assert(out.good());
  }
  std::ifstream spool(path,std::ios::binary);assert(spool.good());
  Graph graph;const std::vector<double> theta;
  const ResponseCheck development=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,
    spool,0,validation_frames,0,1,temperature,soft_modes,0.0,nullptr,0.15,0.15,-1,{},false,nullptr,true,
    maximum_probe_rank,&frozen,&frozen_sources,&frozen_origins);
  const ResponseCheck blind=validate_probes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,types,n,r0,
    spool,0,validation_frames,0,1,temperature,soft_modes,0.0,nullptr,0.15,0.15,-1,{},false,nullptr,true,
    maximum_probe_rank,&frozen,&frozen_sources,&frozen_origins);
  assert(development.validation_frames==validation_frames&&blind.validation_frames==validation_frames);
  assert(development.probes==maximum_probe_rank&&blind.probes==maximum_probe_rank);
  assert(development.probe_sources==frozen_sources&&blind.probe_sources==frozen_sources);
  assert(blind.probe_origins.size()==frozen_origins.size());
  for(std::size_t index=0;index<frozen_origins.size();++index)
    assert(encode_probe_origin(blind.probe_origins[index])==encode_probe_origin(frozen_origins[index]));
  assert(response_consumer_statistics_ready(blind)&&blind.response_band.status=="RESPONSE_PASS");
  const std::string snapshot_path="rpmd_ja_scored_common_probe_snapshot_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".txt";
  RemoveTestFile snapshot_cleanup{snapshot_path};
  const std::vector<double> snapshot_theta={0.0};const std::string manifest(64,'a'),candidate(64,'b'),input(64,'c');
  const std::string snapshot_digest=write_blind_response_checkpoint(snapshot_path,blind,snapshot_theta,manifest,candidate,input,true);
  std::vector<std::string> expected_origins;for(const ProbeOrigin& origin:frozen_origins)
    expected_origins.push_back(encode_probe_origin(origin));
  const BlindResponseSnapshot restored=read_blind_response_checkpoint(snapshot_path,manifest,candidate,input,
    snapshot_digest,frozen_sources,expected_origins);
  assert(restored.passed&&restored.response.probe_sources==frozen_sources&&
    restored.response.mean_q==blind.mean_q&&restored.response.observed_matrix==blind.observed_matrix&&
    restored.response.centered_covariance_matrix==blind.centered_covariance_matrix);
}

void test_frozen_qp_fixture_consumer_identity()
{
  const std::string path=covariance_fit_test_fixture_path(frozen_qp_fixtures[0].name);
  const std::string bytes=read_test_file(path);
  verify_qp_fixture_bytes(path,bytes);
  std::string changed=bytes;assert(!changed.empty());changed[0]^=1;
  bool rejected=false;try{verify_qp_fixture_bytes(path,changed);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  rejected=false;try{(void)read_cut_qp_fixture(covariance_fit_test_fixture_path("missing/")+frozen_qp_fixtures[0].name);}
  catch(const std::runtime_error&){rejected=true;}assert(rejected);
  std::map<std::string,std::string> receipt;std::string canonical;
  for(unsigned i=0;i<5;++i){const std::string prefix="cuda_regression_fixture_"+std::to_string(i+1);
    receipt[prefix+"_name"]=frozen_qp_fixtures[i].name;receipt[prefix+"_sha256"]=frozen_qp_fixtures[i].sha256;
    canonical+=frozen_qp_fixtures[i].name;canonical.push_back('\0');canonical+=frozen_qp_fixtures[i].sha256;canonical.push_back('\n');}
  rpmd_ja_sha256::Digest sha;sha.update(reinterpret_cast<const unsigned char*>(canonical.data()),canonical.size());
  receipt["cuda_regression_fixture_set_sha256"]=sha.finish_hex();verify_frozen_qp_fixture_receipt(receipt);
  receipt["cuda_regression_fixture_1_sha256"]=std::string(64,'0');
  rejected=false;try{verify_frozen_qp_fixture_receipt(receipt);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  receipt["cuda_regression_fixture_1_sha256"]=frozen_qp_fixtures[0].sha256;
  receipt["cuda_regression_fixture_set_sha256"]=std::string(64,'0');
  rejected=false;try{verify_frozen_qp_fixture_receipt(receipt);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
}

void test_feasibility_solver_stop_classification()
{
  assert(covariance_expected_candidate_stop("FEASIBILITY_BUDGET_WALL_LIMIT"));
  assert(covariance_expected_candidate_stop("CURVATURE_UNRESOLVED_INDEPENDENT_BATCH"));
  assert(covariance_expected_candidate_stop("CURVATURE_UNRESOLVED: DIFFERENCE_BELOW_RESOLUTION direction=FIXED_RANDOM"));
  assert(covariance_expected_candidate_stop("BASELINE_SPD_FAIL"));
  assert(!covariance_expected_candidate_stop("NUMERICAL_SOLVE_FAIL"));
  assert(!covariance_expected_candidate_stop("CURVATURE_UNRESOLVED_INDEPENDENT_BATCH_ERROR"));
  assert(!covariance_expected_candidate_stop("CURVATURE_UNRESOLVED: GRAM_FINITE_DIFFERENCE_MISMATCH direction=FIXED_RANDOM"));
  assert(!covariance_expected_candidate_stop("CURVATURE_UNRESOLVED: NONFINITE_DIFFERENCE direction=FIXED_RANDOM"));
  assert(!covariance_expected_candidate_stop("CUDA_ERROR"));
  assert(!covariance_expected_candidate_stop("FEASIBILITY_BUDGET_INVALID_CLOCK"));
  bool invalid_budget_rethrown=false;
  try{try{throw CovarianceFeasibilityBudgetError("FEASIBILITY_BUDGET_INVALID_CLOCK");}
    catch(const CovarianceFeasibilityBudgetError& error){(void)covariance_independent_budget_stop_or_throw(error);}}
  catch(const CovarianceFeasibilityBudgetError& error){invalid_budget_rethrown=std::string(error.what())=="FEASIBILITY_BUDGET_INVALID_CLOCK";}
  assert(invalid_budget_rethrown);
  bool unknown_result_rejected=false;
  try{(void)covariance_independent_failure_status_or_throw("NOT_STARTED");}
  catch(const std::runtime_error& error){unknown_result_rejected=std::string(error.what()).find("IMPLEMENTATION_FAIL:")==0;}
  assert(unknown_result_rejected);
  assert(covariance_independent_failure_status_or_throw("CURVATURE_UNRESOLVED_INDEPENDENT_BATCH")==
    "CURVATURE_UNRESOLVED_INDEPENDENT_BATCH");
}

void test_cgroup_host_memory_limit()
{
  std::istringstream maximum("100"),current("60");
  assert(rpmd_ja_file_io::cgroup_memory_remaining_bytes(maximum,current)==40);
  std::istringstream full_maximum("100"),full_current("120");
  assert(rpmd_ja_file_io::cgroup_memory_remaining_bytes(full_maximum,full_current)==0);
  std::istringstream unlimited("max"),unlimited_current("60");
  assert(rpmd_ja_file_io::cgroup_memory_remaining_bytes(unlimited,unlimited_current)==
    std::numeric_limits<std::uint64_t>::max());
}

void test_published_outputs_cleaned_after_readback_failure()
{
  const std::string suffix=std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string output="rpmd_ja_cleanup_test_"+suffix,sidecar=output+".stability";
  const std::string output_tmp=output+".tmp",sidecar_tmp=sidecar+".tmp",pack=output+".pack";
  auto write_file=[](const std::string& path){std::ofstream out(path,std::ios::binary);out<<"owned";out.close();
    if(!out)throw std::runtime_error("could not create cleanup ownership fixture");};
  bool injected=false;
  try{
    NativeFitScratchOwnership ownership;ownership.output=output;ownership.sidecar=sidecar;
    ownership.output_tmp=output_tmp;ownership.sidecar_tmp=sidecar_tmp;ownership.pack=pack;
    write_file(output);write_file(sidecar);write_file(output_tmp);write_file(sidecar_tmp);write_file(pack);
    ownership.pack_owned=true;
    take_published_prepare_ownership(ownership);
    covariance_fit_test_hooks.fail_after_prepare_publication=true;
    try{maybe_fail_after_prepare_publication_for_test();}
    catch(const std::runtime_error& error){injected=std::string(error.what())=="TEST_PREPARE_POST_PUBLICATION_FAILURE";throw;}
  }catch(const std::runtime_error&){}
  if(!injected)throw std::runtime_error("post-publication prepare failure injection did not reach the cleanup handler");
  for(const std::string& path:{output,sidecar,output_tmp,sidecar_tmp,pack}){
    std::ifstream input(path,std::ios::binary);if(input.good())throw std::runtime_error("cleanup left an owned publication artifact");}
}

void test_covariance_prior_ridge_contract()
{
  assert(rpmd_ja_covariance_prior_ridge_valid(0.1,0.0));
  assert(rpmd_ja_covariance_prior_ridge_valid(0.0,0.017685));
  assert(!rpmd_ja_covariance_prior_ridge_valid(0.0,0.0));
  assert(!rpmd_ja_covariance_prior_ridge_valid(-0.1,0.017685));
  assert(!rpmd_ja_covariance_prior_ridge_valid(std::numeric_limits<double>::quiet_NaN(),0.017685));
  assert(!rpmd_ja_covariance_prior_ridge_valid(0.0,std::numeric_limits<double>::infinity()));
  assert(!rpmd_ja_covariance_prior_ridge_valid(0.1,-0.01));
}
} // namespace

#ifndef GPUMD_RPMD_JA_NATIVE_FIT_TEST_LIBRARY
int main(int argc,char** argv)
{
  if(argc!=3||std::string(argv[1])!="--fixture-dir"||argv[2][0]=='\0'){
    std::fprintf(stderr,"usage: rpmd_ja_native_fit_cuda_test --fixture-dir <tests/data directory>\n");return 2;}
  covariance_fit_test_fixture_directory=argv[2];
  test_covariance_prior_ridge_contract();
  test_frozen_qp_fixture_consumer_identity();
  test_feasibility_solver_stop_classification();
  test_cgroup_host_memory_limit();
  test_published_outputs_cleaned_after_readback_failure();
  test_nonconverged_a0_baseline_restore_rebuilds_factors();
  test_cg_witness_vector_line_endings();
  test_fit_samples_entry_preserves_inputs();
  test_full_spd_failure_keeps_candidate_pack();
  test_response_uncomputed_values_are_explicit();
  test_m0_only_resume_replays_only_a_missing_development_response();
  test_frozen_response_snapshot_round_trip_and_identity_rejection();
  test_publication_io_classification_and_two_rename_recovery();
  test_regularization_response_delta_uses_full_predicted_matrix();
  test_probe_moments_centering_and_small_segments();
  test_ibp_diagnostic_status_is_not_a_numerical_gate();
  test_type_local_source_and_motion_diagnostics();
  test_ibp_noise_channel_selection_and_force_variance();
  test_validation_noise_blocks_start_at_split_boundary();
  test_omitted_validation_tail_does_not_change_channel_interval();
  test_multichannel_bootstrap_maps_local_tail_indices();
  test_block_bootstrap_uses_matrix_error_radius();
  test_bootstrap_uses_matching_complete_frames_and_circular_blocks();
  test_bootstrap_block_factor_candidates_and_terminal_level();
  test_block_product_tail_uses_segment_centering();
  test_bootstrap_nonfinite_is_numerical_failure();
  test_nonfinite_tail_product_is_numerical_failure();
  test_response_bootstrap_ignores_force_position_tail();
  test_bootstrap_rejection_diagnostics();
  test_slow_product_correlation_is_inconclusive();
  test_block_bootstrap_harmonic_coverage();
  test_check_samples_is_read_only();
  test_fixed_r0_validation_mean_projection_drift_report();
  test_sample_spool_minimum_frame_boundary();
  test_fixed_probe_collection_checks_reference_branches();
  test_response_snapshot_preserves_recomputable_matrices();
  test_response_extremes_map_to_internal_coordinates();
  test_covariance_training_statistics_include_symmetric_cross_terms();
  test_covariance_statistics_read_only_training_frames();
  test_blind_fixed_r0_admission_reuses_training_channels_and_block_plan();
  test_covariance_development_statistics_keep_fixed_training_center();
  test_covariance_probe_session_reuses_factors_and_rejects_stale_assembly();
  test_covariance_probe_refinement_improves_each_original_residual_and_rejects_large_error();
  test_covariance_force_residual_uses_training_frames_only();
  test_covariance_baseline_brackets_both_signs_and_zero();
  test_covariance_optimizer_updates_from_zero_baseline_and_recovers_harmonic_solution();
  test_covariance_trust_region_staged_barrier_harmonic_solve();
  test_native_covariance_mid_a0_resume_preserves_original_start_and_stagnation_window();
  test_covariance_inverse_bfgs_spectrum_statuses();
  test_covariance_block_newton_optimizer_direction_trace();
  test_covariance_optimizer_boundary_start_preserves_prior_center_and_reports_status();
  test_covariance_initialization_shrinks_before_expanding();
  test_covariance_initialization_reports_no_improvement_at_budget();
  test_covariance_gradient_residual_reject_retries_before_commit();
  test_covariance_directional_overshoot_rejects_before_commit();
  test_covariance_joint_bfgs_rejects_positive_slope_and_uses_secant_shrink();
  test_covariance_joint_logdet_target_rejects_positive_slopes();
  test_covariance_gradient_residual_budget_keeps_last_valid_and_reject();
  test_covariance_armijo_bounded_expansion_preserves_last_valid_candidate();
  test_covariance_bfgs_first_secant_seeds_and_updates_inverse_hessian();
  test_covariance_non_descent_reset_uses_restored_gamma();
  test_covariance_recent_bfgs_replays_latest_gamma_and_bounds_history();
  test_covariance_objective_only_overshoot_uses_directional_secant();
  test_covariance_solver_residual_retry_commits_once();
  test_covariance_solver_residual_budget_preserves_state_points();
  test_covariance_fixed_coordinate_fd_uses_selected_coordinate_scale();
  test_covariance_fixed_coordinate_fd_switches_feasible_side_with_shared_budget();
  test_covariance_near_boundary_fd_requires_stable_steps();
  test_covariance_fixed_coordinate_fd_one_percent_is_separate_from_classification();
  test_covariance_stable_directional_fd_consistency_and_mismatch();
  test_covariance_inconclusive_diagnostic_preserves_failure_status();
  test_covariance_scaled_trial_round_trip_underflow();
  test_covariance_optimizer_rank_deficiency_budget_and_33_dimension_barrier();
  test_covariance_objective_analytic_gradient_matches_finite_difference();
  test_covariance_trust_region_subproblem_and_barrier_cache();
  test_covariance_trust_region_roundoff_boundary_rebuild();
  test_covariance_optimizer_trace_parses_legacy_and_trust_region_steps();
  test_covariance_gradient_decomposition_preserves_total();
  test_covariance_edge_attribution_reconstructs_and_ranks();
  test_covariance_edge_attribution_reports_diffuse_top20_fractions();
  test_covariance_edge_capture_reconstructs_group_trace_and_statistics();
  test_m1_statistics_compensate_projection_and_operator_mix();
  test_fixed_internal_probe_statistics_detect_zero_com_internal_shift();
  test_admission_statistics_storage_peak_boundary();
  test_m1_gpu_chain_matches_cpu_and_dense_reference();
  test_m1_bridge_images_share_hydrogen_anchored_branch();
  test_fixed_physical_response_probes_are_frozen_and_ranked();
  test_native_m1_writer_reader_cross_neighbor_images_and_transport();
  test_native_additive_package_accounting_modes_and_bounds();
  test_covariance_trace_wide_solve_matches_cpu_and_filters_failed_tail();
  test_covariance_gemm_original_residual_matches_cpu();
  test_covariance_strict_residual_failure_updates_overall_worst();
  test_covariance_saved_gradient_does_not_require_baseline_shifted_solve();
  test_covariance_state_parser_accepts_lf_and_crlf();
  test_covariance_checkpoint_round_trip_and_corruption_rejection();
  test_feasibility_manifest_parser_accepts_lf_and_crlf();
  test_feasibility_manifest_freezes_independent_curvature_contract();
  test_feasibility_manifest_freezes_candidate_cost_contract();
  test_feasibility_manifest_freezes_experiment_contract();
  test_covariance_state_tau_comparison_is_relative_for_small_values();
  test_covariance_failed_trial_does_not_replace_last_valid_point();
  test_covariance_prior_mixing_formula();
  test_soft_mode_training_variance_uses_same_r0_second_moments();
  test_saved_d_mode_participation_projection_and_status();
  test_compressed_training_force_residual_distinguishes_fit_cost();
  test_saved_sample_diagnostic();
  test_design_and_edge_operator();
  test_pap_baseline();
  test_baseline_batch_matches_gemv();
  DeviceQR qr;qr.initialize(2,3);
  test_a0_soft_modes_use_covariance_baseline_point(qr.solver);
  test_covariance_operator_gram_includes_shared_atom_overlap(qr.solver);
  test_covariance_direct_hessian_preconditioner(qr.solver);
  test_covariance_centered_ridge_objective_and_tau_cache();
  test_covariance_original_residual_gates_convergence();
  test_feasibility_same_file_path_aliases();
  test_covariance_solve_write_diagnose_replay_without_ridge();
  test_covariance_prior_calibration_ignores_m1_tail();
  test_budget_observation_classifies_hard_clock_errors();
  test_m1_covariance_calibration_uses_only_base_prefix_in_replay();
  test_covariance_factor_identity_recovery();
  test_covariance_directional_curvature_requires_stable_resolved_match();
  test_covariance_independent_failure_diagnostic_never_commits();
  test_covariance_analytic_curvature_whitened_gram();
  test_covariance_fit_workspace_grows_reuses_and_releases();
  test_covariance_fit_workspace_is_reused_by_sequential_system_curvature_builds();
  test_covariance_device_workspace_reuses_buffers_static_graph_and_same_point_factors();
  test_covariance_local_group_hessian_matches_tiny_objective_fd(qr.solver);
  test_covariance_block_newton_host_arithmetic();
  test_covariance_trust_region_residual_rejection_is_transactional();
  test_covariance_joint_block_includes_cross_group_hessian();
  test_covariance_group_bfgs_independent_history();
  test_covariance_joint_ray_spectrum_and_proposal(qr.solver);
  test_covariance_joint_block_commit_attribution_is_transactional();
  test_covariance_joint_large_group_bfgs_commit_and_default();
  test_padded_qr_svd_and_cut_qp(qr.solver);
  test_active_set_qp_snapshot_and_rejection(qr.solver);
  test_active_set_preserves_svd_primal(qr.solver);
  test_active_set_refines_kkt_residual(qr.solver);
  test_active_set_rank_deficient_warm_start(qr.solver);
  test_outer128_stationarity_joint_polish(qr.solver);
  test_read_frame_rejects_invalid_data();
  test_stability_cut_guard_reduces_rotating_mode_rounds(qr.solver);
  test_lanczos_staged_early_exit(qr.solver);
  test_lanczos_continues_across_checkpoints(qr.solver);
  test_lanczos_iteration_budget_interrupts_one_overrun(qr.solver);
  test_lanczos_curvature_rounding_boundary(qr.solver);
  test_lanczos_deep_search_does_not_early_cut(qr.solver);
  test_lanczos_finite_internal_space(qr.solver);
  test_projected_cg_curvature_witness();
  test_production_true_residual_recovery();
  test_bounded_cg_replay_refinement();
  test_compensated_m1_local_action_recovers_cancellation();
  test_compensated_cg_true_residual_action();
  test_replay_rejects_existing_report_before_reading_inputs();
  test_cg_failure_snapshot_round_trip_replay();
  test_lanczos_blindspot_cg_cut_feedback(qr.solver);
  test_lanczos_independent_seed_escapes_invariant_subspace(qr.solver);
  test_probe_selection_skips_duplicate_low_modes();
  test_common_low_ritz_modes_get_probe_rank_before_fixed_channels();
  test_common_probe_rank_is_capped_for_nearly_full_fixed_basis(qr.solver);
  test_probe_covariance_and_ibp(qr.solver);
}
#endif
