#ifdef NDEBUG
#error "This test requires assertions enabled"
#endif

#include "../src/measure/rpmd_ja_native_fit.cu"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <random>
#include <sstream>

static std::vector<double> diagnostic_expected_r0;
static bool diagnostic_should_throw = false;

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
                                    const RpmdJAModeValidator&, const std::string&) {}
RpmdJAModeValidator make_rpmd_ja_qnep_mode_validator(Atom&, Box&, Force&) { return {}; }
std::uint64_t rpmd_ja_model_fingerprint(const std::string&) { return 0; }
std::uint64_t rpmd_ja_qnep_config_fingerprint(Force&) { return 0; }

namespace
{
std::string write_sample_spool(const std::string& suffix, const int frames, const double interval=2.5)
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
    const double step = 5.0 + interval * frame;
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

std::string read_test_file(const std::string& path)
{
  std::ifstream in(path,std::ios::binary);return std::string(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
}

struct CutQPFixture
{
  SmallSVD svd;
  std::vector<std::vector<double>> rows,a;
  std::vector<double> rhs,b,initial_lambda;
  double primal_tolerance=0.0;
};

CutQPFixture read_cut_qp_fixture(const std::string& path)
{
  std::ifstream in(path);if(!in)throw std::runtime_error("cannot open QP regression fixture: "+path);CutQPFixture fixture;std::string line;
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
  if(!in.eof()||fixture.svd.p!=60||fixture.rows.size()!=233||fixture.a.size()!=fixture.rows.size()||fixture.initial_lambda.size()!=fixture.rows.size()||
     fixture.svd.singular.size()!=static_cast<std::size_t>(fixture.svd.p)||fixture.svd.eta.size()!=static_cast<std::size_t>(fixture.svd.p)||fixture.svd.vt.size()!=static_cast<std::size_t>(fixture.svd.p)*fixture.svd.p||!(fixture.primal_tolerance>0.0))
    throw std::runtime_error("QP regression fixture dimensions or metadata are invalid");
  return fixture;
}

void test_fit_samples_entry_preserves_inputs()
{
  const std::string spool=write_sample_spool(".fit_samples",3,2.0),raw=spool+".qraw",output=spool+".out";
  RemoveTestFile clean_spool{spool},clean_raw{raw},clean_failure{output+".failure.txt"},clean_lock{output+".fit.lock"};
  {std::ofstream out(raw,std::ios::binary);out<<"external qraw sentinel";assert(out.good());}
  const std::string saved_spool=read_test_file(spool),saved_raw=read_test_file(raw);
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass={1.0,2.0};atom.cpu_type={0,1};atom.number_of_beads=0;atom.position_per_atom.resize(6);
  const std::vector<double> original_position={0.7,0.8,0.9,1.0,1.1,1.2};atom.position_per_atom.copy_from_host(original_position.data());
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;Force force;
  RpmdJANativeFitOptions options;assert(options.max_stability_rounds==160);options.output_path=output;options.kernel_table="unused-before-validation";options.raw_input_path=raw;options.cutoff=2.0;options.epsilon=1e-3;options.response_tolerance=0.15;options.fd_step=1e-3;
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

  const std::vector<std::vector<double>> unequal_tolerance_rows={{1.0,0.0},{0.0,1.0}};
  const std::vector<double> unequal_tolerance_rhs={100000.0+1.048e-9,1.0+1e-12};
  const auto unequal_tolerance=check_cut_qp_kkt(identity,unequal_tolerance_rows,unequal_tolerance_rhs,
    unequal_tolerance_rows,unequal_tolerance_rhs,{100000.0,1.0},{100000.0,1.0},1e-8);
  assert(unequal_tolerance.minimum_slack_constraint_index==0);
  assert(unequal_tolerance.worst_primal_excess_index==1);
  assert(unequal_tolerance.max_primal_excess>8.8e-13&&unequal_tolerance.max_primal_excess<9.0e-13);
  assert(unequal_tolerance.minimum_slack_allowed_error==1e-8&&unequal_tolerance.allowed_error[1]<2e-13);
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
  qp_state_file.close();assert(std::remove(qp_state_path.c_str())==0);

  bool rejected=false;
  try { (void)svd_small(solver,{std::numeric_limits<double>::quiet_NaN(),0.0,0.0,1.0},{0.0,0.0},2); }
  catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
}

void test_active_set_qp_snapshot_and_rejection(cusolverDnHandle_t solver)
{
  std::string fixture_path="tests/data/ja_reference_resample.bin.qp_state.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
  const CutQPFixture fixture=read_cut_qp_fixture(fixture_path);
  const QPSolution replay=solve_cut_qp(solver,fixture.svd,fixture.rows,fixture.rhs,fixture.primal_tolerance,fixture.initial_lambda);
  assert(replay.method==std::string("active_set_polish"));
  assert(((replay.iterations==64&&replay.polish_attempts==1)||(replay.iterations==4096&&replay.polish_attempts==2)||
    (replay.iterations==200000&&replay.polish_attempts==3))&&replay.polish_updates>0);
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
  assert(negative.certificate.primal_slack.size()==rows.size()&&negative.certificate.primal_slack[1]>0.0);
  assert(rounded_exit.certificate.primal_slack.size()==rows.size()&&rounded_exit.certificate.primal_slack[1]>0.0);

  const std::vector<std::vector<double>> dependent={{1.0,0.0},{2.0,0.0}};const std::vector<double> dependent_rhs={1.0,2.0};
  const ActiveSetPolishResult rank_failure=polish_cut_qp(solver,identity,dependent,dependent_rhs,dependent,dependent_rhs,{1.0,1.0},1e-8);
  assert(!rank_failure.accepted&&rank_failure.failure_reason=="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT");
  const std::vector<std::vector<double>> too_many_active={{1.0,0.0},{0.0,1.0},{1.0,1.0}};const std::vector<double> too_many_rhs={1.0,1.0,2.0};
  const ActiveSetPolishResult dimension_rank_failure=polish_cut_qp(solver,identity,too_many_active,too_many_rhs,too_many_active,too_many_rhs,{1.0,1.0,1.0},1e-8);
  assert(!dimension_rank_failure.accepted&&dimension_rank_failure.failure_reason=="ACTIVE_MATRIX_NUMERICAL_RANK_DEFICIENT");
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
      const auto modes=lanczos_low_modes(solver,baseline,graph,theta,sqrt_mass,sqrt_atom,n,96,4);
      assert(modes.size()==static_cast<std::size_t>(std::min(4,internal)));
      for(int k=0;k<static_cast<int>(modes.size());++k){
        const auto& v=modes[k].vector;double norm2=0.0;
        for(double x:v)norm2+=x*x;
        check_close(norm2,1.0,2e-8);
        for(int axis=0;axis<3;++axis){double translation=0.0;for(int i=0;i<n;++i)translation+=sqrt_atom[i]*v[axis*n+i];assert(std::abs(translation)<2e-10);}
        for(int prior=0;prior<k;++prior)check_close(std::inner_product(v.begin(),v.end(),modes[prior].vector.begin(),0.0),0.0,2e-7);
        std::vector<double> product(d),add(d);baseline.apply(v,product);const double base_rayleigh=std::inner_product(v.begin(),v.end(),product.begin(),0.0);apply_additive(graph,v,sqrt_mass,sqrt_atom,theta,n,add);
        double residual2=0.0;for(int i=0;i<d;++i){product[i]+=add[i];const double residual=product[i]-modes[k].value*v[i];residual2+=residual*residual;}
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
}

void test_lanczos_blindspot_cg_cut_feedback(cusolverDnHandle_t solver)
{
  constexpr int n=40,d=3*n,p=6;const double epsilon=1e-3;const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> w(d);
  w[0]=1.0-1.0/n;for(int i=1;i<n;++i)w[i]=-1.0/n;std::vector<double> start(d);for(int i=0;i<d;++i)start[i]=std::sin((i+1)*1.6180339887498948)+std::cos((i+1)*0.7548776662466927);project_translation(start,sqrt_atom,n);double norm=std::sqrt(std::inner_product(start.begin(),start.end(),start.begin(),0.0));for(double& x:start)x/=norm;const double overlap=std::inner_product(w.begin(),w.end(),start.begin(),0.0);double start_x2=0.0;for(int i=0;i<n;++i)start_x2+=start[i]*start[i];for(int i=0;i<n;++i)w[i]-=(overlap/start_x2)*start[i];norm=std::sqrt(std::inner_product(w.begin(),w.end(),w.begin(),0.0));for(double& x:w)x/=norm;assert(std::abs(std::inner_product(w.begin(),w.end(),start.begin(),0.0))<1e-12);
  std::vector<double> raw(static_cast<std::size_t>(d)*d,0.0);for(int i=0;i<d;++i)raw[static_cast<std::size_t>(i)*d+i]=1.0;for(int i=0;i<d;++i)for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]-=2.0*w[i]*w[j];std::stringstream input(std::ios::in|std::ios::out|std::ios::binary);input.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));input.seekg(0);DeviceBaseline baseline;baseline.initialize(input,0,d,n,masses,sqrt_mass);Graph graph;for(int i=0;i<n;++i)for(int j=i+1;j<n;++j)graph.edges.push_back({i,j,0,{0,0,0}});
  const auto modes=lanczos_low_modes(solver,baseline,graph,std::vector<double>(p,0.0),sqrt_mass,sqrt_atom,n,96,4);assert(modes.front().value>0.99&&modes.front().residual<1e-10);std::vector<double> zero(p,0.0);const CGResult cg=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,zero,n,0,epsilon);assert(cg.witness.classification=="NONPOSITIVE_OPERATOR_DIRECTION");assert(cg.witness.rayleigh<0.0);const auto witness_modes=lanczos_low_modes(solver,baseline,graph,std::vector<double>(p,0.0),sqrt_mass,sqrt_atom,n,16,4,cg.witness.direction);assert(witness_modes.front().value<-0.99&&witness_modes.front().residual<1e-10);
  SmallSVD identity;identity.p=p;identity.singular.assign(p,1.0);identity.eta.assign(p,0.0);identity.vt.assign(p*p,0.0);for(int i=0;i<p;++i)identity.vt[static_cast<std::size_t>(i)*p+i]=1.0;const auto row=spectral_row(graph,w,sqrt_mass,n,p);const double base=std::inner_product(w.begin(),w.end(),cg.witness.base.begin(),0.0);const auto qp=solve_cut_qp(solver,identity,{row},{epsilon-base},epsilon/4.0);const auto theta=theta_from_eta(identity,qp.xi);
  const auto basis=internal_basis(n);const int internal=static_cast<int>(basis.size());std::vector<double> restricted(static_cast<std::size_t>(internal)*internal);for(int j=0;j<internal;++j){std::vector<double> image(d);apply_total(baseline,graph,basis[j],sqrt_mass,sqrt_atom,theta,n,image);for(int i=0;i<internal;++i)restricted[static_cast<std::size_t>(i)*internal+j]=std::inner_product(basis[i].begin(),basis[i].end(),image.begin(),0.0);}const auto exact=eigen_small(solver,restricted,internal);assert(exact.values.front()>=epsilon-2e-10);const CGResult repaired=solve_projected_cg(baseline,graph,w,sqrt_mass,sqrt_atom,theta,n,0,epsilon);assert(repaired.witness.classification.empty());assert(repaired.relative_residual<=1e-8);
}

void test_lanczos_independent_seed_escapes_invariant_subspace(cusolverDnHandle_t solver)
{
  constexpr int n=2,d=6;const auto basis=internal_basis(n);const std::vector<double> masses(n,1.0),sqrt_atom(n,1.0),sqrt_mass(d,1.0);std::vector<double> raw(d*d,0.0);
  const double eigenvalues[3]={1.0,-2.0,3.0};for(int k=0;k<3;++k)for(int i=0;i<d;++i)for(int j=0;j<d;++j)raw[static_cast<std::size_t>(i)*d+j]+=eigenvalues[k]*basis[k][i]*basis[k][j];
  std::stringstream stream(std::ios::in|std::ios::out|std::ios::binary);stream.write(reinterpret_cast<const char*>(raw.data()),raw.size()*sizeof(double));stream.seekg(0);DeviceBaseline baseline;baseline.initialize(stream,0,d,n,masses,sqrt_mass);Graph graph;
  DeviceLanczosWorkspace workspace;workspace.initialize(graph,sqrt_mass,sqrt_atom,8,0);
  const auto warm=lanczos_low_modes(solver,baseline,graph,{},sqrt_mass,sqrt_atom,n,8,3,basis[0],&workspace);const auto independent=lanczos_low_modes(solver,baseline,graph,{},sqrt_mass,sqrt_atom,n,8,3,{},&workspace);
  assert(warm.size()==1&&warm.front().residual<1e-12);check_close(warm.front().value,1.0,1e-12);assert(independent.front().value<-1.99&&independent.front().residual<1e-10);
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
  assert(result.probes>=20);assert(result.validation_frames==validation);assert(result.observed_cov_available&&result.predicted_cov_available&&result.cg_residual_available&&result.response_ibp_available&&result.force_residual_available);assert(result.observed_cov_min>0.0&&result.observed_cov_max>=result.observed_cov_min&&result.observed_cov_condition>=1.0);assert(result.predicted_cov_min>0.0&&result.predicted_cov_max>=result.predicted_cov_min&&result.predicted_cov_condition>=1.0);assert(result.block_frames[0]+result.block_frames[1]+result.block_frames[2]+result.block_frames[3]==validation);assert(std::isfinite(result.block_variance_max_relative_delta));assert(result.cg<=1e-8);assert(result.response<=2e-8);assert(result.ibp<=2e-8);assert(result.force_residual<=2e-8);

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
} // namespace

int main()
{
  test_fit_samples_entry_preserves_inputs();
  test_full_spd_failure_keeps_candidate_pack();
  test_response_uncomputed_values_are_explicit();
  test_probe_moments_centering_and_small_segments();
  test_block_bootstrap_uses_matrix_error_radius();
  test_bootstrap_uses_matching_complete_frames_and_circular_blocks();
  test_block_product_tail_uses_segment_centering();
  test_bootstrap_nonfinite_is_numerical_failure();
  test_bootstrap_rejection_diagnostics();
  test_slow_product_correlation_is_inconclusive();
  test_block_bootstrap_harmonic_coverage();
  test_check_samples_is_read_only();
  test_response_snapshot_preserves_recomputable_matrices();
  test_saved_sample_diagnostic();
  test_design_and_edge_operator();
  test_pap_baseline();
  DeviceQR qr;qr.initialize(2,3);
  test_padded_qr_svd_and_cut_qp(qr.solver);
  test_active_set_qp_snapshot_and_rejection(qr.solver);
  test_read_frame_rejects_invalid_data();
  test_lanczos_finite_internal_space(qr.solver);
  test_projected_cg_curvature_witness();
  test_lanczos_blindspot_cg_cut_feedback(qr.solver);
  test_lanczos_independent_seed_escapes_invariant_subspace(qr.solver);
  test_probe_selection_skips_duplicate_low_modes();
  test_probe_covariance_and_ibp(qr.solver);
}
