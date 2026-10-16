#ifdef NDEBUG
#error "This test requires assertions enabled"
#endif

#define GPUMD_RPMD_JA_NATIVE_FIT_TEST
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
void check_close(const double a,const double b,const double tol=2e-9);
std::vector<double> solve_dense_reference(std::vector<double> matrix,std::vector<double> rhs,const int n);
std::vector<std::vector<double>> internal_basis(const std::vector<double>& sqrt_atom);

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

struct RemoveTestFile { std::string path; ~RemoveTestFile() { std::remove(path.c_str()); } };

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
  if(!in.eof()||fixture.svd.p!=60||fixture.rows.empty()||fixture.a.size()!=fixture.rows.size()||fixture.initial_lambda.size()!=fixture.rows.size()||
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
  ResponseCheck with_radii;with_radii.response_band.level_radii={{1,0.25}};with_radii.response_extremes_available=true;
  with_radii.response_ratio_min=0.8;with_radii.response_ratio_max=1.2;std::ostringstream radii_summary;write_response_stats(radii_summary,with_radii);
  const std::string radii_text=radii_summary.str();const auto radii=radii_text.find("response_block_radii=1:0.25");
  const auto ratio=radii_text.find("response_ratio_status=AVAILABLE");assert(radii!=std::string::npos&&ratio>radii);
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
  const std::vector<double> statistics=collect_covariance_fit_statistics(spool,header,3,2,graph,r0,300.0);
  assert(statistics.size()==6);check_close(statistics[0],5.0/(K_B*300.0*2.0),1e-12);
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
  constexpr double epsilon=0.01,rho=0.7,tau=0.2;const CovarianceLocalGroupBasis local=covariance_local_group_basis(graph,sqrt_mass,2,0,32);
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
  const CovarianceLocalHessianTerms hessian=covariance_local_group_hessian(trace_d,trace_s,rho,tau,std::vector<double>(6,1.0),0);
  const double analytic=hessian.total[0];
  std::vector<double> extended_scale(12);for(std::size_t i=0;i<extended_scale.size();++i)extended_scale[i]=0.25+0.125*i;
  const CovarianceLocalHessianTerms scaled_hessian=covariance_local_group_hessian(trace_d,trace_s,rho,tau,extended_scale,6);
  for(int a=0;a<6;++a)for(int b=0;b<6;++b){const std::size_t at=static_cast<std::size_t>(a)*6+b;
    const double expected=(0.5*(1.0+rho)*trace_d[at]+tau*trace_s[at])*extended_scale[6+a]*extended_scale[6+b];
    check_close(scaled_hessian.total[at],expected,2e-15);}
  CovarianceBaseline reference;reference.theta=theta;std::vector<double> shifted_trace;
  assert(system.evaluate_factors(theta,epsilon,reference.logdet,reference.shifted_logdet,&reference.trace,&shifted_trace,false,true));
  const std::vector<double> statistics={0.2,-0.03,0.04,0.25,0.01,0.3};std::uint64_t evaluations=0;
  const CovarianceObjectiveEvaluation center=evaluate_covariance_fit_objective(system,theta,reference,statistics,rho,epsilon,tau,evaluations,false);
  constexpr double h=1e-4;double values[2];
  for(int side=0;side<2;++side){std::vector<double> trial=theta;trial[0]+=(side==0?1.0:-1.0)*h;
    const CovarianceObjectiveEvaluation value=evaluate_covariance_fit_objective(system,trial,reference,statistics,rho,epsilon,tau,evaluations,false);
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
  const CovarianceJointHessianTerms hessian=covariance_joint_group_hessian(trace_d,trace_s,0.6,0.3,scale,selection.groups);
  const std::size_t cross=6;assert(std::abs(hessian.total[cross])>1e-5);
  check_close(hessian.total[cross],(0.5*(1.0+0.6)*trace_d[cross]+0.3*trace_s[cross])*scale[12]*scale[0],2e-14);
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

void test_covariance_joint_block_commit_attribution_is_transactional()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};
  {
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    assert(fit.joint_block_enabled&&fit.last_joint_attribution.available&&fit.last_joint_attribution.model_status=="COMPUTED");
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
    assert(fit.last_joint_attribution.available&&fit.last_joint_attribution.iteration==0&&fit.last_joint_attribution.tau==1.0);
    assert(fit.block_newton_factorizations==0&&fit.block_newton_rhs_columns==0);
    assert(fit.joint_block_factorizations>0&&fit.joint_block_rhs_columns>0);
    assert(!rejected_theta.empty()&&rejected_theta!=fit.theta&&rejected<committed&&committed<attribution_pos);
    assert(log.find("STEP_COMMITTED iteration=0 tau=1 alpha=0.5")!=std::string::npos&&!fit.failed_trial.available);
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
  }
  {
    DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);std::ostringstream trace;
    covariance_fit_test_hooks={};covariance_fit_test_hooks.gradient_residual_failures_remaining=64;
    covariance_fit_test_hooks.objective_evaluation_budget_after_baseline=16;
    const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,1,&trace,false,true);
    covariance_fit_test_hooks={};
    assert(fit.status=="EVALUATION_BUDGET_EXHAUSTED"&&fit.iterations==0&&fit.last_valid.theta==fit.theta);
    assert(!fit.last_joint_attribution.available&&trace.str().find("COVARIANCE_JOINT_COMMIT_ATTRIBUTION")==std::string::npos);
  }
}

void test_covariance_baseline_brackets_both_signs_and_zero()
{
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};const Graph graph=make_two_atom_covariance_graph();
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3,0.08,-0.04,0.25,0.06,0.35};std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,0.01,evaluations);
    check_close(reference.mu,2.533333333333333,2e-7);assert(!reference.epsilon_constraint_active);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={2.5,0.0,0.0,2.5,0.0,2.5};std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,0.01,evaluations);
    check_close(reference.mu,-0.4,2e-7);assert(!reference.epsilon_constraint_active);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,1e-6,evaluations);
    check_close(reference.mu,0.0,2e-7);assert(!reference.epsilon_constraint_active);
  }
  {
    DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
    const std::vector<double> statistics={1000.0,0.0,0.0,1000.0,0.0,1000.0};std::uint64_t evaluations=0;
    const CovarianceBaseline reference=build_graph_covariance_baseline(system,statistics,0.01,evaluations);
    assert(reference.epsilon_constraint_active&&reference.mu>=reference.boundary_mu);
  }
}

void test_covariance_optimizer_updates_from_zero_baseline_and_recovers_harmonic_solution()
{
  DeviceBaseline baseline;initialize_two_atom_covariance_baseline(baseline,true);const Graph graph=make_two_atom_covariance_graph();
  const std::vector<double> masses={1.0,4.0},sqrt_mass={1.0,2.0,1.0,2.0,1.0,2.0};DeviceCovarianceSystem system(baseline,graph,sqrt_mass,masses);
  const std::vector<double> statistics={0.3125,0.0,0.0,1.25,0.0,2.1875};std::ostringstream trace;
  const CovarianceFitResult fit=solve_covariance_reference(system,statistics,1e-6,0.1,300,&trace);
  assert(fit.converged&&fit.status=="COVARIANCE_OPTIMIZER_CONVERGED"&&fit.iterations>0);
  assert(std::abs(fit.baseline_mu)<2e-7&&fit.barrier_center_condition_pass&&fit.barrier_gap_estimate<=1e-6);
  assert(fit.initialization_status=="SKIPPED_BASELINE_INTERIOR"&&fit.initialization_evaluations==1);
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
          std::vector<double>{std::numeric_limits<double>::quiet_NaN()};});
    }catch(const std::exception&){propagated=true;}
    assert(propagated&&fatal.directional_overshoot_reject_trials==0&&fatal.failed_trial.failure_stage=="GRADIENT_SOLVE_FAIL");
  }
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
  const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,3,fit,0,0.5,evaluate,complete,&trace);
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
    CovarianceFitResult fit=initialize();int calls=0;double evaluated_x=0.0;
    const CovarianceBacktrackingResult line=covariance_backtrack_armijo(theta,z,scale,direction,0.0,-1.0,1.0,1.0,1,fit,0,0.5,
      [&](const std::vector<double>& x,std::uint64_t& evaluations){++evaluations;evaluated_x=x[0];CovarianceObjectiveEvaluation value;
        value.feasible=true;value.status=CovarianceObjectiveStatus::Feasible;value.objective=-x[0]+0.05*x[0]*x[0];return value;},
      [&](CovarianceObjectiveEvaluation& value){++calls;value.gradient={-1.0+0.1*evaluated_x};});
    assert(line.accepted&&!line.evaluation_budget_exhausted&&line.alpha==1.0&&line.trials==1&&calls==1);
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
  history.stage_seeded=true;history.last_valid_gamma=0.25;history.inverse_hessian={-1.0,0.0,0.0,-1.0};
  const std::vector<double> gradient={4.0,8.0};std::vector<double> direction(2);
  for(std::size_t i=0;i<2;++i)for(std::size_t j=0;j<2;++j)direction[i]-=history.inverse_hessian[2*i+j]*gradient[j];
  assert(!(covariance_dot(gradient,direction)<0.0));
  const double slope=covariance_reset_non_descent_direction(history,gradient,direction);
  assert(history.seed_gamma==0.25&&history.seed_source=="RECENT_VALID_GAMMA_RESET");
  assert(direction[0]==-1.0&&direction[1]==-2.0&&slope==-20.0&&covariance_norm_inf(direction)==2.0);
  const double alpha=std::min(1.0,1.0/covariance_norm_inf(direction));assert(alpha==0.5);
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
  const std::vector<double> zero_theta(6,0.0);system.assemble(zero_theta);double logdet=0.0;assert(system.factor(system.compact,logdet));
  std::vector<double> matrix(36);assert(cudaMemcpy(matrix.data(),system.full,matrix.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess);
  matrix[1*6+1]+=0.5;assert(cudaMemcpy(system.full,matrix.data(),matrix.size()*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
  bool failed=false;try{(void)system.trace_derivatives(system.compact,"D",0.0,true,true);}catch(const std::runtime_error&){failed=true;}
  assert(failed&&system.last_d_trace.status=="RESIDUAL_FAIL");
  assert(system.last_d_trace.worst_original_solve_relative_residual>1e-7);
  check_close(system.worst_solve_relative_residual,system.last_d_trace.worst_original_solve_relative_residual,1e-15);
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
  const std::string lf="last_valid_available 1\ncom_convention internal_mass_com_pullback_v1\nlast_valid_tau 5.65e-11\n";
  std::string crlf;for(const char character:lf){if(character=='\n')crlf.push_back('\r');crlf.push_back(character);}
  std::istringstream lf_input(lf),crlf_input(crlf);
  const auto lf_fields=read_covariance_state_fields(lf_input),crlf_fields=read_covariance_state_fields(crlf_input);
  assert(lf_fields.at("last_valid_available")=="1"&&crlf_fields.at("last_valid_available")=="1");
  assert(lf_fields.at("com_convention")=="internal_mass_com_pullback_v1"&&
    crlf_fields.at("com_convention")=="internal_mass_com_pullback_v1");
  const double lf_tau=std::stod(lf_fields.at("last_valid_tau")),crlf_tau=std::stod(crlf_fields.at("last_valid_tau"));
  check_close(lf_tau,crlf_tau,0.0);assert(covariance_state_tau_matches(lf_tau,crlf_tau));
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
  std::string fixture_path="tests/data/ja_reference_resample.bin.qp_state.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
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
  std::string fixture_path="tests/data/ja_reference_replay_v1.bin.qp_state.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
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
  std::string fixture_path="tests/data/ja_reference_replay_v2.bin.qp_state.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
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
  std::string fixture_path="tests/data/ja_reference_replay_v4.bin.qp_state.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
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
  std::string fixture_path="tests/data/ja_reference_outer128_766cuts_stationarity_v1.qp_fixture.txt";{std::ifstream probe(fixture_path);if(!probe)fixture_path="../"+fixture_path;}
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
} // namespace

int main()
{
  test_cg_witness_vector_line_endings();
  test_fit_samples_entry_preserves_inputs();
  test_full_spd_failure_keeps_candidate_pack();
  test_response_uncomputed_values_are_explicit();
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
  test_covariance_probe_session_reuses_factors_and_rejects_stale_assembly();
  test_covariance_probe_refinement_improves_each_original_residual_and_rejects_large_error();
  test_covariance_force_residual_uses_training_frames_only();
  test_covariance_baseline_brackets_both_signs_and_zero();
  test_covariance_optimizer_updates_from_zero_baseline_and_recovers_harmonic_solution();
  test_covariance_block_newton_optimizer_direction_trace();
  test_covariance_optimizer_boundary_start_preserves_prior_center_and_reports_status();
  test_covariance_initialization_shrinks_before_expanding();
  test_covariance_initialization_reports_no_improvement_at_budget();
  test_covariance_gradient_residual_reject_retries_before_commit();
  test_covariance_directional_overshoot_rejects_before_commit();
  test_covariance_gradient_residual_budget_keeps_last_valid_and_reject();
  test_covariance_armijo_bounded_expansion_preserves_last_valid_candidate();
  test_covariance_bfgs_first_secant_seeds_and_updates_inverse_hessian();
  test_covariance_non_descent_reset_uses_restored_gamma();
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
  test_covariance_gradient_decomposition_preserves_total();
  test_covariance_edge_attribution_reconstructs_and_ranks();
  test_covariance_edge_attribution_reports_diffuse_top20_fractions();
  test_covariance_edge_capture_reconstructs_group_trace_and_statistics();
  test_covariance_gemm_original_residual_matches_cpu();
  test_covariance_strict_residual_failure_updates_overall_worst();
  test_covariance_saved_gradient_does_not_require_baseline_shifted_solve();
  test_covariance_state_parser_accepts_lf_and_crlf();
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
  test_covariance_operator_gram_includes_shared_atom_overlap(qr.solver);
  test_covariance_local_group_hessian_matches_tiny_objective_fd(qr.solver);
  test_covariance_block_newton_host_arithmetic();
  test_covariance_joint_block_includes_cross_group_hessian();
  test_covariance_joint_block_commit_attribution_is_transactional();
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
  test_lanczos_curvature_rounding_boundary(qr.solver);
  test_lanczos_deep_search_does_not_early_cut(qr.solver);
  test_lanczos_finite_internal_space(qr.solver);
  test_projected_cg_curvature_witness();
  test_production_true_residual_recovery();
  test_bounded_cg_replay_refinement();
  test_compensated_cg_true_residual_action();
  test_replay_rejects_existing_report_before_reading_inputs();
  test_cg_failure_snapshot_round_trip_replay();
  test_lanczos_blindspot_cg_cut_feedback(qr.solver);
  test_lanczos_independent_seed_escapes_invariant_subspace(qr.solver);
  test_probe_selection_skips_duplicate_low_modes();
  test_probe_covariance_and_ibp(qr.solver);
}
