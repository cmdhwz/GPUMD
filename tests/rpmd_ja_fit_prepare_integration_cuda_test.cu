#ifdef NDEBUG
#undef NDEBUG
#endif

#define GPUMD_RPMD_JA_NATIVE_FIT_TEST_REAL_PREPARE
#define GPUMD_RPMD_JA_NATIVE_FIT_TEST_LIBRARY
#include "rpmd_ja_native_fit_cuda_test.cu"
#define GPUMD_RPMD_JA_QNEP_PREPARE_TEST_LIBRARY
#include "rpmd_ja_qnep_prepare_cuda_test.cu"
#include "integrate/integrate.cuh"
#include "measure/hac.cuh"
#include "measure/measure.cuh"
#include "model/group.cuh"

namespace
{
struct IntegrationCleanup
{
  std::vector<std::string> paths;
  ~IntegrationCleanup(){for(const std::string& path:paths)clean(path);}
};

void test_measure_rpmd_ja_use_propagation_order_and_reset()
{
  Atom atom;Box box;Force force;Integrate integrate;std::vector<Group> groups;
  auto add_hac=[&](Measure& measure){
    if(!measure.parse_action({"compute_hac","1","1","1"},1,integrate,groups,atom,box,force,"nep.txt"))
      throw std::runtime_error("compute_hac could not be registered in the configuration propagation fixture");
    auto* hac=dynamic_cast<HAC*>(measure.get_actions().back().get());
    if(hac==nullptr)throw std::runtime_error("compute_hac registration did not create a HAC action");
    return hac;
  };
  Measure before;before.set_rpmd_ja(true,"diagnostic-reference.bin",true);HAC* before_hac=add_hac(before);
  if(!before_hac->rpmd_ja_enabled_for_test()||!before_hac->rpmd_ja_diagnostic_for_test()||
     before_hac->rpmd_ja_reference_path_for_test()!="diagnostic-reference.bin")
    throw std::runtime_error("rpmd_ja setup before compute_hac creation was not copied to the new action");
  before.set_rpmd_ja(false);
  if(before_hac->rpmd_ja_enabled_for_test()||before_hac->rpmd_ja_diagnostic_for_test())
    throw std::runtime_error("rpmd_ja off/reset did not clear an existing compute_hac action");

  Measure after;HAC* after_hac=add_hac(after);after.set_rpmd_ja(true,"diagnostic-reference.bin",true);
  if(!after_hac->rpmd_ja_enabled_for_test()||!after_hac->rpmd_ja_diagnostic_for_test()||
     after_hac->rpmd_ja_reference_path_for_test()!="diagnostic-reference.bin")
    throw std::runtime_error("rpmd_ja setup after compute_hac creation was not copied to the existing action");
  after.set_rpmd_ja(false);
  if(after_hac->rpmd_ja_enabled_for_test()||after_hac->rpmd_ja_diagnostic_for_test())
    throw std::runtime_error("multi-run rpmd_ja reset left a diagnostic use flag on compute_hac");
  HAC* next_run=add_hac(after);
  if(next_run->rpmd_ja_enabled_for_test()||next_run->rpmd_ja_diagnostic_for_test())
    throw std::runtime_error("compute_hac in the next run inherited stale diagnostic reference state");
}

void test_reference_use_compatibility_by_backend()
{
  const std::string formal_v3="native_reference_transport;finite_temperature_additive_v1;beads=4;derivative=3";
  const std::string diagnostic_v3=formal_v3+";use=DIAGNOSTIC;response=FAIL;numerical=PASS";
  if(!rpmd_ja_reference_policy_use_matches(0,"",false)||
     !rpmd_ja_reference_policy_use_matches(1,"",false)||
     rpmd_ja_reference_policy_use_matches(0,"",true)||
     rpmd_ja_reference_policy_use_matches(1,"",true)||
     !rpmd_ja_reference_policy_use_matches(2,"native_reference_transport",false)||
     !rpmd_ja_reference_policy_use_matches(2,formal_v3,false)||
     rpmd_ja_reference_policy_use_matches(2,formal_v3,true)||
     !rpmd_ja_reference_policy_use_matches(2,diagnostic_v3,true)||
     rpmd_ja_reference_policy_use_matches(2,diagnostic_v3,false)||
     rpmd_ja_reference_policy_use_matches(2,"",false)||
     rpmd_ja_reference_policy_use_matches(3,"",false))
    throw std::runtime_error("reference use compatibility did not preserve legacy formal backends and restrict diagnostics to qNEP v3");
}

std::string write_graph_reference_samples(const std::string& path,const int frames)
{
  std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot create graph-reference samples");
  const char magic[8]={'G','P','J','A','S','M','P','1'};const std::uint32_t version=1,endian=native_endian;
  const std::int32_t n=2,beads=4;const double temperature=300.0;
  const double cell[9]={10,0,0,0,10,0,0,0,10},masses[2]={1.0,1.07};const std::int32_t types[2]={0,0};
  out.write(magic,sizeof(magic));out.write(reinterpret_cast<const char*>(&version),sizeof(version));
  out.write(reinterpret_cast<const char*>(&endian),sizeof(endian));out.write(reinterpret_cast<const char*>(&n),sizeof(n));
  out.write(reinterpret_cast<const char*>(&beads),sizeof(beads));out.write(reinterpret_cast<const char*>(&temperature),sizeof(temperature));
  out.write(reinterpret_cast<const char*>(cell),sizeof(cell));out.write(reinterpret_cast<const char*>(masses),sizeof(masses));
  out.write(reinterpret_cast<const char*>(types),sizeof(types));
  const double base[6]={2.0,4.0,2.0,4.0,2.0,4.0},amplitude[3]={0.02,0.06,0.12};
  constexpr int training=80;
  for(int frame=0;frame<frames;++frame){const double step=5.0+2.0*frame;
    const double phase=2.0*3.14159265358979323846*(frame%training)/training;
    double position[6],force[6];for(int axis=0;axis<3;++axis){const double displacement=amplitude[axis]*std::sin(phase);
      position[2*axis]=base[2*axis]+displacement;position[2*axis+1]=base[2*axis+1]-displacement;
      force[2*axis]=-0.2*displacement;force[2*axis+1]=0.2*displacement;}
    out.write(reinterpret_cast<const char*>(&step),sizeof(step));
    out.write(reinterpret_cast<const char*>(position),sizeof(position));out.write(reinterpret_cast<const char*>(force),sizeof(force));}
  out.close();if(!out)throw std::runtime_error("failed writing graph-reference samples");return path;
}

void patch_graph_reference_raw_geometry(const std::string& raw)
{
  const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  const std::streamoff header=8+4+4+4+4+8+8+8+8+4+4+8+sizeof(layout);
  const double cell[18]={10,0,0,0,10,0,0,0,10, 0.1,0,0,0,0.1,0,0,0,0.1};
  std::fstream file(raw,std::ios::binary|std::ios::in|std::ios::out);if(!file)throw std::runtime_error("cannot patch graph-reference qraw geometry");
  file.seekp(header);file.write(reinterpret_cast<const char*>(cell),sizeof(cell));
  file.close();if(!file)throw std::runtime_error("failed patching graph-reference qraw cell");
  const std::vector<double> masses={1.0,1.07},positions={2.0,4.0,2.0,4.0,2.0,4.0};
  const std::vector<int> types={0,0};patch_m1_atom_identity(raw,types,masses,positions);
}

void test_graph_reference_samples_to_diagnostic_prepare_and_hard_failure()
{
  const std::string stem="rpmd_ja_graph_reference_e2e_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string samples=write_graph_reference_samples(stem+".samples",120),raw=stem+".qraw",kernel=stem+".kernel";
  const std::string output=stem+".ja",reuse_output=stem+".reuse.ja",bad_state_output=stem+".badstate.ja",
    state=stem+".covariance_state.txt",bad_state=stem+".bad.covariance_state.txt",hard_output=stem+".hard.ja";
  IntegrationCleanup cleanup{{samples,raw,kernel,output,output+".fit.txt",output+".fit_trace.txt",output+".fit.lock",
    output+".additive.tmp",reuse_output,reuse_output+".fit.txt",reuse_output+".fit_trace.txt",reuse_output+".fit.lock",
    reuse_output+".additive.tmp",bad_state_output,bad_state_output+".fit_trace.txt",bad_state_output+".fit.lock",
    bad_state,state,hard_output,hard_output+".failure.txt",hard_output+".fit_trace.txt",hard_output+".fit.lock",
    hard_output+".additive.tmp"}};
  const std::vector<double> masses={1.0,1.07};const std::vector<int> types={0,0};
  make_input(raw,kernel,2,0.5,3,-1.0,0.0,-1,false,&masses,&types);patch_graph_reference_raw_geometry(raw);
  Atom atom;atom.number_of_atoms=2;atom.cpu_mass=masses;atom.cpu_type=types;atom.number_of_beads=0;atom.position_per_atom.resize(6);
  const std::vector<double> initial={2.0,4.0,2.0,4.0,2.0,4.0};atom.position_per_atom.copy_from_host(initial.data());
  Box box{};box.cpu_h[0]=box.cpu_h[4]=box.cpu_h[8]=10.0;box.cpu_h[9]=box.cpu_h[13]=box.cpu_h[17]=0.1;
  box.pbc_x=box.pbc_y=box.pbc_z=1;Force force;
  RpmdJANativeFitOptions options;options.method=RpmdJANativeFitMethod::GaussianCovarianceShrinkage;
  options.graph_reference=true;options.output_path=output;options.kernel_table=kernel;options.raw_input_path=raw;
  options.cutoff=4.0;options.epsilon=1.0e-3;options.response_tolerance=1.0e-6;options.fd_step=0.01;
  covariance_fit_test_allow_qraw_fixture_identity=true;covariance_fit_test_hooks={};
  covariance_fit_test_hooks.skip_graph_reference_mode_validator=true;
  fit_rpmd_ja_native_reference_from_samples(options,samples,atom,box,force);
  covariance_fit_test_hooks={};covariance_fit_test_allow_qraw_fixture_identity=false;
  const RpmdJAReference prepared=read_rpmd_ja_reference(output);
  if(!rpmd_ja_reference_policy_is_diagnostic(prepared.mechanical_policy)||
     rpmd_ja_reference_policy_numerical_status(prepared.mechanical_policy)!="PASS")
    throw std::runtime_error("graph-reference end-to-end fixture did not preserve its diagnostic numerical identity");
  const std::string response=rpmd_ja_reference_policy_response_status(prepared.mechanical_policy);
  if(response!="FAIL"&&response!="INCONCLUSIVE")
    throw std::runtime_error("anisotropic graph-reference fixture did not produce a held-out response failure or inconclusive result");
  if(rpmd_ja_reference_policy_use_matches(prepared.backend,prepared.mechanical_policy,false)||
     !rpmd_ja_reference_policy_use_matches(prepared.backend,prepared.mechanical_policy,true))
    throw std::runtime_error("ordinary rpmd_ja on use accepted a diagnostic prepared reference");
  const std::string trace=read_test_file(output+".fit_trace.txt"),fit=read_test_file(output+".fit.txt");
  if(trace.find("optimizer_60_parameter_run=0")==std::string::npos||
     trace.find("numerical_status NOT_EVALUATED")==std::string::npos||
     trace.find("GRAPH_REFERENCE_NUMERICAL_STATUS PASS certificate=FULL_CERTIFICATE_PASS readback_identity=PASS")==std::string::npos||
     trace.find("GRAPH_REFERENCE_PREPARE_READBACK_PASS") == std::string::npos||
     trace.find("COVARIANCE_OPTIMIZER_CONVERGED")!=std::string::npos||
     fit.find("mechanical_policy "+prepared.mechanical_policy)==std::string::npos)
    throw std::runtime_error("graph-reference trace or fit report is inconsistent with the prepared diagnostic readback");

  double baseline_mu=0.0;std::vector<double> baseline_theta;
  {
    std::istringstream fit_lines(fit);std::string line;
    while(std::getline(fit_lines,line)){
      if(line.rfind("baseline_mu ",0)==0){std::istringstream value(line.substr(12));if(!(value>>baseline_mu))throw std::runtime_error("graph-reference fit report has no scalar baseline_mu");}
      if(line.rfind("baseline_theta ",0)==0){std::istringstream values(line.substr(15));std::size_t count=0;if(!(values>>count))throw std::runtime_error("graph-reference fit report has malformed baseline_theta");
        baseline_theta.resize(count);for(double& value:baseline_theta)if(!(values>>value)||!std::isfinite(value))throw std::runtime_error("graph-reference fit report has truncated baseline_theta");}
    }
  }
  std::ifstream sample_input(samples,std::ios::binary);const SampleHeader saved_header=read_header(sample_input,120,atom,box,300.0,true,true);
  std::vector<double> saved_r0(6,0.0);read_training_r0(sample_input,saved_header,120,2.0,saved_r0);
  RpmdJANativeFitOptions state_options=options;state_options.output_path=output;state_options.temperature=300.0;state_options.sample_interval=2;
  covariance_fit_test_allow_qraw_fixture_identity=true;
  rpmd_ja_file_io::ifstream raw_input(raw,std::ios::binary);QrawIdentity saved_identity=validate_qraw_identity(raw_input,state_options,saved_header,saved_r0,box,force);
  saved_identity.source_fnv64=rpmd_ja_model_fingerprint(raw);
  covariance_fit_test_allow_qraw_fixture_identity=false;
  Graph saved_graph=make_graph(saved_r0,saved_header.types,box,state_options.cutoff);
  CovarianceFitResult old_fit;old_fit.status="MAX_ITERATIONS";old_fit.reason="budget exhausted; baseline candidate only";
  old_fit.statistics.resize(static_cast<std::size_t>(saved_graph.parameter_count));old_fit.baseline_mu=baseline_mu;
  old_fit.epsilon_constraint_status="ACTIVE";old_fit.theta_baseline=baseline_theta;old_fit.theta=baseline_theta;
  write_covariance_state(state,old_fit,0.0,samples,raw,state_options,saved_header,saved_identity,saved_graph,120,80);
  std::string corrupted_state=read_test_file(state);const std::string config_field="qraw_config_fingerprint ";
  const std::size_t config_at=corrupted_state.find(config_field);
  if(config_at==std::string::npos)throw std::runtime_error("saved baseline fixture has no qraw config identity");
  const std::size_t config_end=corrupted_state.find('\n',config_at);
  corrupted_state.replace(config_at,config_end-config_at,config_field+"1");
  {std::ofstream bad_output(bad_state,std::ios::binary|std::ios::trunc);bad_output<<corrupted_state;
    if(!bad_output)throw std::runtime_error("cannot write mismatched baseline-state fixture");}
  options.output_path=bad_state_output;options.baseline_state_path=bad_state;covariance_fit_test_allow_qraw_fixture_identity=true;
  bool wrong_state_rejected=false;
  try{fit_rpmd_ja_native_reference_from_samples(options,samples,atom,box,force);}
  catch(const std::runtime_error& error){wrong_state_rejected=std::string(error.what()).find("GRAPH_REFERENCE_BASELINE_STATE_IDENTITY_FAIL")!=std::string::npos;}
  covariance_fit_test_hooks={};covariance_fit_test_allow_qraw_fixture_identity=false;
  if(!wrong_state_rejected||exists(bad_state_output)||exists(bad_state_output+".stability")||exists(bad_state_output+".additive.tmp"))
    throw std::runtime_error("mismatched saved graph-baseline identity was accepted or published");
  options.output_path=reuse_output;options.baseline_state_path=state;covariance_fit_test_allow_qraw_fixture_identity=true;
  covariance_fit_test_hooks.skip_graph_reference_mode_validator=true;
  fit_rpmd_ja_native_reference_from_samples(options,samples,atom,box,force);
  covariance_fit_test_hooks={};covariance_fit_test_allow_qraw_fixture_identity=false;
  const std::string reuse_trace=read_test_file(reuse_output+".fit_trace.txt");
  if(reuse_trace.find("stationarity_status SAVED_BASELINE_FREE_ROOT_VERIFIED")==std::string::npos||
     reuse_trace.find("optimizer_60_parameter_run=0")==std::string::npos)
    throw std::runtime_error("graph-reference baseline_state candidate was not reused after its fresh numerical check");

  options.output_path=hard_output;covariance_fit_test_hooks={};covariance_fit_test_allow_qraw_fixture_identity=true;
  covariance_fit_test_hooks.skip_graph_reference_mode_validator=true;
  options.baseline_state_path.clear();
  covariance_fit_test_hooks.fail_graph_reference_response_hard=true;
  bool hard_error=false;try{fit_rpmd_ja_native_reference_from_samples(options,samples,atom,box,force);}
  catch(const std::runtime_error& error){hard_error=std::string(error.what()).find("TEST_GRAPH_REFERENCE_RESPONSE_HARD_FAILURE")!=std::string::npos;}
  covariance_fit_test_hooks={};covariance_fit_test_allow_qraw_fixture_identity=false;
  if(!hard_error||exists(hard_output)||exists(hard_output+".stability")||exists(hard_output+".additive.tmp"))
    throw std::runtime_error("hard response failure was swallowed or published a graph-reference package");
}

void test_frozen_fit_input_fingerprints_reject_mutation()
{
  const std::string stem="rpmd_ja_frozen_fit_input_identity_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string raw=stem+".qraw",kernel=stem+".kernel";
  IntegrationCleanup cleanup{{raw,kernel}};
  auto write_text=[](const std::string& path,const char* value){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create frozen input identity fixture");
    out<<value;out.close();if(!out)throw std::runtime_error("failed writing frozen input identity fixture");
  };
  write_text(raw,"raw-start");write_text(kernel,"kernel-start");
  const std::uint64_t raw_identity=rpmd_ja_model_fingerprint(raw),kernel_identity=rpmd_ja_model_fingerprint(kernel);
  verify_frozen_fit_input_fingerprints(raw,kernel,raw_identity,kernel_identity);
  write_text(raw,"raw-changed");bool raw_rejected=false;
  try{verify_frozen_fit_input_fingerprints(raw,kernel,raw_identity,kernel_identity);}
  catch(const std::runtime_error& error){raw_rejected=std::string(error.what()).find("qraw content changed")!=std::string::npos;}
  if(!raw_rejected)throw std::runtime_error("frozen fit input identity accepted modified qraw content");
  write_text(raw,"raw-start");write_text(kernel,"kernel-changed");bool kernel_rejected=false;
  try{verify_frozen_fit_input_fingerprints(raw,kernel,raw_identity,kernel_identity);}
  catch(const std::runtime_error& error){kernel_rejected=std::string(error.what()).find("kernel table content changed")!=std::string::npos;}
  if(!kernel_rejected)throw std::runtime_error("frozen fit input identity accepted modified kernel content");
}

void test_manifest_sha256_rejects_qraw_and_kernel_replacement_at_native_start()
{
  const std::string stem="rpmd_ja_manifest_source_sha256_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const std::string raw=stem+".qraw",kernel=stem+".kernel";
  IntegrationCleanup cleanup{{raw,kernel}};
  auto write_text=[](const std::string& path,const char* value){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create manifest source identity fixture");
    out<<value;out.close();if(!out)throw std::runtime_error("failed writing manifest source identity fixture");
  };
  write_text(raw,"abc");write_text(kernel,"kernel-start");
  const FitInputContentIdentity raw_start=fit_input_content_identity(raw);
  const FitInputContentIdentity kernel_start=fit_input_content_identity(kernel);
  if(raw_start.sha256!="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"||
     raw_start.fnv64!=rpmd_ja_model_fingerprint(raw))
    throw std::runtime_error("native SHA-256/FNV scan did not match known qraw identity vectors");
  const std::map<std::string,std::string> manifest={{"raw_sha256",raw_start.sha256},
    {"kernel_sha256",kernel_start.sha256}};
  verify_feasibility_source_content_hashes(manifest,kernel_start,raw_start);

  write_text(raw,"replacement-qraw");bool raw_rejected=false;
  try{verify_feasibility_source_content_hashes(manifest,kernel_start,fit_input_content_identity(raw));}
  catch(const std::runtime_error& error){raw_rejected=std::string(error.what()).find("qraw content differs")!=std::string::npos;}
  if(!raw_rejected)throw std::runtime_error("native manifest identity accepted qraw replaced after wrapper verification");

  write_text(raw,"abc");write_text(kernel,"replacement-kernel");bool kernel_rejected=false;
  try{verify_feasibility_source_content_hashes(manifest,fit_input_content_identity(kernel),raw_start);}
  catch(const std::runtime_error& error){kernel_rejected=std::string(error.what()).find("kernel content differs")!=std::string::npos;}
  if(!kernel_rejected)throw std::runtime_error("native manifest identity accepted kernel replaced after wrapper verification");
}

void patch_m1_atom_identity(const std::string& raw,const std::vector<int>& types,const std::vector<double>& masses,
  const std::vector<double>& positions)
{
  const int n=static_cast<int>(types.size()),d=3*n;const char layout[]="xyz_soa;derivative_input_rows_output_columns";
  if(masses.size()!=types.size()||positions.size()!=static_cast<std::size_t>(d))
    throw std::invalid_argument("M1 integration raw atom fixture dimensions do not match");
  const std::streamoff header=8+4+4+4+4+8+8+8+8+4+4+8+sizeof(layout);
  const std::streamoff types_at=header+18*8+12,masses_at=types_at+n*4,positions_at=masses_at+n*8;
  std::fstream file(raw,std::ios::binary|std::ios::in|std::ios::out);
  if(!file)throw std::runtime_error("cannot open M1 integration raw fixture");
  file.seekp(types_at);file.write(reinterpret_cast<const char*>(types.data()),n*sizeof(int));
  file.seekp(masses_at);file.write(reinterpret_cast<const char*>(masses.data()),n*sizeof(double));
  file.seekp(positions_at);file.write(reinterpret_cast<const char*>(positions.data()),d*sizeof(double));
  file.close();if(!file)throw std::runtime_error("failed patching M1 integration raw atom identity");
}

double additive_sparse_value(const RpmdJASparseRows& matrix,const int row,const int column)
{
  for(std::size_t index=matrix.offsets[static_cast<std::size_t>(row)];index<matrix.offsets[static_cast<std::size_t>(row)+1];++index)
    if(matrix.columns[index]==column)return matrix.values[index];
  return 0.0;
}

void erase_additive_sparse_entry(RpmdJASparseRows& matrix,const int d,const int row,const int column)
{
  const auto begin=matrix.columns.begin()+static_cast<std::ptrdiff_t>(matrix.offsets[row]);
  const auto end=matrix.columns.begin()+static_cast<std::ptrdiff_t>(matrix.offsets[row+1]);
  const auto found=std::lower_bound(begin,end,column);
  if(found==end||*found!=column)throw std::runtime_error("small-entry fixture could not find the sparse entry to erase");
  const std::size_t index=static_cast<std::size_t>(found-matrix.columns.begin());
  matrix.columns.erase(found);matrix.values.erase(matrix.values.begin()+static_cast<std::ptrdiff_t>(index));
  for(int offset=row+1;offset<=d;++offset)--matrix.offsets[offset];
}

void test_native_additive_entry_verifier_rejects_small_scale_K_H()
{
  constexpr int n=2,d=3*n;Graph graph;graph.sites.resize(n);graph.parameter_count=1;graph.raw_local_parameter_count=1;
  graph.local_transform={1.0};
  const rpmd_ja_local_basis::Site sites[2]={{0,{0,0,0}},{1,{0,0,0}}};
  const rpmd_ja_local_basis::Vec3 coefficients[2]={{1.0,0.0,0.0},{-1.0,0.0,0.0}};
  const double shares[2]={1.0,5.0e-13};
  graph.local_terms.push_back(rpmd_ja_local_basis::detail::make_term(sites,coefficients,2,0,1.0,shares));
  const std::vector<double> theta={1.0e-9},positions={0.0,1.0,0.0,0.0,0.0,0.0};Box box{};
  const IndependentAdditiveOperators expected=independently_assemble_additive_operators(graph,theta,positions,box,n);
  auto compress_dense=[&](const std::vector<double>& dense){
    std::vector<RpmdJAAdditiveEntry> entries;
    for(int row=0;row<d;++row)for(int column=0;column<d;++column){const double value=dense[static_cast<std::size_t>(row)*d+column];
      if(value!=0.0)entries.push_back({row,column,value});}
    return rpmd_ja_additive_compress(d,entries);
  };
  RpmdJAAdditiveData package;package.k=compress_dense(expected.k);
  const double share_sum=shares[0]+shares[1];for(double& value:package.k.values)value*=share_sum;
  for(int alpha=0;alpha<3;++alpha)package.h[alpha]=compress_dense(expected.h[alpha]);
  const int k_row=0,k_column=0,h_row=1,h_column=1;
  const double small_k=additive_sparse_value(package.k,k_row,k_column);
  const double small_h=additive_sparse_value(package.h[0],h_row,h_column);
  if(!(small_k>0.0&&small_h<0.0)||std::abs(small_k-1.0e-9)>1.0e-21||std::abs(small_h+1.0e-9)>1.0e-21)
    throw std::runtime_error("small-entry fixture must contain K=1e-9 and H0=-1e-9");
  const auto small_entry_error_over_limit=verify_native_additive_package_entries(package,graph,theta,positions,box,n);
  if(!std::all_of(small_entry_error_over_limit.begin(),small_entry_error_over_limit.end(),
       [](double error){return std::isfinite(error)&&error<=1.0;})||!(small_entry_error_over_limit[0]>0.0))
    throw std::runtime_error("small-entry positive control has invalid error-over-limit metrics");

  auto require_missing_entry_rejected=[&](RpmdJAAdditiveData corrupted,const bool is_k,const int row,const int column){
    RpmdJASparseRows& matrix=is_k?corrupted.k:corrupted.h[0];
    erase_additive_sparse_entry(matrix,d,row,column);
    const double reference=is_k?1.0e-9:-1.0e-9;
    const double legacy_scaled_error=std::abs(reference)/std::max({1.0,std::abs(reference),0.0});
    if(additive_sparse_value(matrix,row,column)!=0.0||legacy_scaled_error>2.0e-9)
      throw std::runtime_error("small-entry fixture no longer reproduces the legacy max(1,...) false acceptance");
    const char* label=is_k?"K":"H0";bool rejected=false;
    try{(void)verify_native_additive_package_entries(corrupted,graph,theta,positions,box,n);}
    catch(const std::runtime_error& error){
      const std::string expected_message=std::string(label)+" sparse entry differs from the independent local-energy formula at row "+
        std::to_string(row)+", column "+std::to_string(column);
      if(std::string(error.what()).find(expected_message)==std::string::npos)
        throw std::runtime_error(std::string("small-entry verifier rejected ")+label+" for the wrong reason: "+error.what());
      rejected=true;
    }
    if(!rejected)throw std::runtime_error(std::string("small-entry verifier accepted missing 1e-9 ")+label+" entry");
  };
  require_missing_entry_rejected(package,true,k_row,k_column);
  require_missing_entry_rejected(package,false,h_row,h_column);

  Graph cancellation_graph=graph;cancellation_graph.local_terms.clear();const double cancellation_shares[2]={0.5,0.5};
  cancellation_graph.local_terms.push_back(rpmd_ja_local_basis::detail::make_term(sites,coefficients,2,0,1000.0,cancellation_shares));
  cancellation_graph.local_terms.push_back(rpmd_ja_local_basis::detail::make_term(sites,coefficients,2,0,-999.999999999,cancellation_shares));
  const std::vector<double> cancellation_theta={1.0};
  const IndependentAdditiveOperators cancellation_expected=
    independently_assemble_additive_operators(cancellation_graph,cancellation_theta,positions,box,n);
  RpmdJAAdditiveData cancellation_package;cancellation_package.k=compress_dense(cancellation_expected.k);
  for(int alpha=0;alpha<3;++alpha)cancellation_package.h[alpha]=compress_dense(cancellation_expected.h[alpha]);
  const double cancelled_k=additive_sparse_value(cancellation_package.k,0,0);
  const double cancelled_h=additive_sparse_value(cancellation_package.h[0],1,1);
  if(!(cancelled_k>0.0&&cancelled_h<0.0)||std::abs(cancelled_k-1.0e-9)>2.0e-14||
     std::abs(cancelled_h+0.5e-9)>2.0e-14)
    throw std::runtime_error("cancelled fixture must retain soft K and H0 entries after signed-term cancellation");
  const auto cancellation_error_over_limit=verify_native_additive_package_entries(
    cancellation_package,cancellation_graph,cancellation_theta,positions,box,n);
  if(!std::all_of(cancellation_error_over_limit.begin(),cancellation_error_over_limit.end(),
       [](double error){return std::isfinite(error)&&error<=1.0;}))
    throw std::runtime_error("cancelled positive control has invalid error-over-limit metrics");
  auto require_cancelled_missing_rejected=[&](RpmdJAAdditiveData corrupted,const bool is_k,const int row,const int column){
    RpmdJASparseRows& matrix=is_k?corrupted.k:corrupted.h[0];const char* label=is_k?"K":"H0";
    const double reference=is_k?cancelled_k:cancelled_h;
    erase_additive_sparse_entry(matrix,d,row,column);
    const double legacy_scaled_error=std::abs(reference)/std::max({1.0,std::abs(reference),0.0});
    if(additive_sparse_value(matrix,row,column)!=0.0||legacy_scaled_error>2.0e-9)
      throw std::runtime_error("cancelled fixture no longer reproduces legacy acceptance of an omitted soft entry");
    bool rejected=false;try{(void)verify_native_additive_package_entries(corrupted,cancellation_graph,cancellation_theta,positions,box,n);}
    catch(const std::runtime_error& error){
      const std::string expected_message=std::string(label)+" sparse entry differs from the independent local-energy formula at row "+
        std::to_string(row)+", column "+std::to_string(column);
      if(std::string(error.what()).find(expected_message)==std::string::npos)
        throw std::runtime_error(std::string("cancelled verifier rejected ")+label+" for the wrong reason: "+error.what());
      rejected=true;
    }
    if(!rejected)throw std::runtime_error(std::string("cancelled verifier accepted missing ")+label+" residual entry");
  };
  require_cancelled_missing_rejected(cancellation_package,true,0,0);
  require_cancelled_missing_rejected(cancellation_package,false,1,1);
}

RpmdJASparseRows transpose_additive_sparse(const RpmdJASparseRows& matrix,const int d)
{
  std::vector<RpmdJAAdditiveEntry> entries;entries.reserve(matrix.values.size());
  for(int row=0;row<d;++row)for(std::size_t index=matrix.offsets[row];index<matrix.offsets[row+1];++index)
    entries.push_back({matrix.columns[index],row,matrix.values[index]});
  return rpmd_ja_additive_compress(d,entries);
}

std::vector<double> additive_sparse_action(const RpmdJASparseRows& matrix,const std::vector<double>& vector,const int d)
{
  std::vector<double> result(static_cast<std::size_t>(d));
  for(int row=0;row<d;++row)for(std::size_t index=matrix.offsets[row];index<matrix.offsets[row+1];++index)
    result[row]+=matrix.values[index]*vector[matrix.columns[index]];
  return result;
}

void run_actual_m1_writer_prepare_readback(const std::string& base)
{
  constexpr int n=3,d=9;const std::string raw=base+".qraw",kernel=base+".kernel",pack=base+".gpjaadd";
  const std::string baseline_path=base+".baseline.ja",output_path=base+".m1_com.ja";
  const std::string entry_failure_path=base+".entry_check_failure.ja",readback_failure_path=base+".readback_failure.ja";
  IntegrationCleanup cleanup{{raw,kernel,pack,baseline_path,output_path,entry_failure_path,readback_failure_path}};
  const std::vector<int> types={0,1,0};const std::vector<double> masses={16.0,1.0,16.0};
  const std::vector<double> positions={9.8,0.1,0.4,0.0,0.1,0.0,0.0,0.2,0.0};
  make_input(raw,kernel,n,0.05,3,-1.0,0.0,-1,false,&masses,&types);
  patch_additive_source_geometry(raw,n);patch_m1_atom_identity(raw,types,masses,positions);

  Box box{};const double cell_and_inverse[18]={10.0,0.4,0.0,0.0,11.0,0.3,0.0,0.0,12.0,
    0.1,-0.4/110.0,0.4*0.3/1320.0,0.0,1.0/11.0,-0.3/132.0,0.0,0.0,1.0/12.0};
  std::copy(cell_and_inverse,cell_and_inverse+18,box.cpu_h);
  Graph graph=make_graph(positions,types,box,5.2);const std::vector<std::string> symbols={"O","H","O"};
  const auto& metadata=build_m1_local_basis(graph,positions,types,symbols,masses,box,5.2,1);
  if(metadata.accepted_bridge_count!=1||metadata.added_parameter_count==0||graph.local_terms.empty())
    throw std::runtime_error("actual M1 writer fixture did not retain the periodic O-H-O bridge");
  std::vector<double> theta(static_cast<std::size_t>(graph.parameter_count),0.0);
  for(int parameter=0;parameter<graph.base_parameter_count;++parameter)theta[parameter]=0.05;
  for(int parameter=graph.base_parameter_count;parameter<graph.parameter_count;++parameter)
    theta[parameter]=0.13+0.07*(parameter-graph.base_parameter_count);
  std::vector<std::array<double,3>> edge_linear(graph.edges.size());
  const IndependentAdditiveOperators expected=independently_assemble_additive_operators(graph,theta,positions,box,n);

  prepare_rpmd_ja_qnep_reference(raw,baseline_path,kernel);
  const RpmdJAReference baseline=read_rpmd_ja_reference(baseline_path);
  std::uint64_t estimated_package_bytes=0;
  {
    std::ofstream out(pack,std::ios::binary);if(!out)throw std::runtime_error("cannot create actual M1 integration package");
    out.write(additive_magic,8);write(out,std::uint32_t(6));write(out,native_endian);write(out,std::int32_t(n));write(out,std::int32_t(8));
    write(out,rpmd_ja_model_fingerprint(raw));write(out,300.0);write(out,1.0e-8);write(out,0.0);write(out,0.02);
    write(out,std::uint64_t(1));write(out,std::uint64_t(1));
    estimated_package_bytes=write_native_additive_site_blocks(out,graph,theta,edge_linear,n);
    out.close();if(!out)throw std::runtime_error("failed closing actual M1 integration package");
  }
  std::ifstream sized(pack,std::ios::binary|std::ios::ate);
  if(!sized||static_cast<std::uint64_t>(sized.tellg())!=estimated_package_bytes)
    throw std::runtime_error("actual M1 version-6 package byte count differs from writer estimate");
  sized.close();
  const RpmdJAAdditiveData additive=read_rpmd_ja_additive(pack,raw,baseline,8,std::vector<double>(d,0.0));
  if(!additive.internal_mass_com||additive.k.values.empty()||
     std::abs(additive_sparse_value(additive.k,0,2))+std::abs(additive_sparse_value(additive.k,2,0))<=1.0e-10)
    throw std::runtime_error("actual M1 writer package lost its version-6 COM identity or O-O cross-neighbor K");
  std::array<double,4> package_entry_error_over_limits{};bool package_entries_checked=false;
  RpmdJAPrepareResourcePlan prepare_resource_plan;bool prepare_resource_plan_seen=false;
  std::vector<std::string> prepare_allocation_requests;
  const RpmdJAPrepareWorkspaceSizes queried_prepare_workspace=query_rpmd_ja_qnep_prepare_workspace_sizes(n);
  prepare_rpmd_ja_qnep_reference(raw,output_path,kernel,{},pack,[&](const RpmdJAAdditiveData& parsed){
    package_entry_error_over_limits=verify_native_additive_package_entries(parsed,graph,theta,positions,box,n);
    package_entries_checked=true;
  },[&](const RpmdJAPrepareResourcePlan& plan){prepare_resource_plan=plan;prepare_resource_plan_seen=true;},
  [&](const char* name,const std::uint64_t bytes){
    if(bytes==0)throw std::runtime_error("actual prepare allocation observer received a zero-byte request");
    if(std::string(name)=="allocate scalar POTRF query matrix"&&bytes!=sizeof(double))
      throw std::runtime_error("prepare workspace query allocated more than its scalar device placeholder");
    prepare_allocation_requests.emplace_back(name);
  });
  if(!package_entries_checked||!std::all_of(package_entry_error_over_limits.begin(),package_entry_error_over_limits.end(),
    [](double error){return std::isfinite(error)&&error<=1.0;}))
    throw std::runtime_error("actual M1 prepare did not run its independent sparse-entry check");
  if(!prepare_resource_plan_seen||prepare_resource_plan.potrf_workspace_bytes==0||
     prepare_resource_plan.svd_workspace_bytes==0||
     prepare_resource_plan.potrf_workspace_bytes!=queried_prepare_workspace.potrf_workspace_bytes||
     prepare_resource_plan.svd_workspace_bytes!=queried_prepare_workspace.svd_workspace_bytes||
     prepare_resource_plan.svd_peak_bytes>prepare_resource_plan.planned_peak_gpu_bytes||
     std::find(prepare_allocation_requests.begin(),prepare_allocation_requests.end(),"allocate SVD workspace")==prepare_allocation_requests.end())
    throw std::runtime_error("actual M1 prepare disagreed with the no-large-allocation cuSOLVER workspace query");

  auto require_entry_check_rejects=[&](const RpmdJAAdditiveData& corrupted,const char* label){
    bool rejected=false;try{verify_native_additive_package_entries(corrupted,graph,theta,positions,box,n);}
    catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error(std::string("independent sparse-entry check accepted corrupted ")+label);
  };
  RpmdJAAdditiveData bad_k=additive;bad_k.k.values.front()+=0.125;
  require_entry_check_rejects(bad_k,"K entry");
  RpmdJAAdditiveData transposed_h=additive;transposed_h.h[0]=transpose_additive_sparse(additive.h[0],d);
  require_entry_check_rejects(transposed_h,"transposed nonsymmetric H");

  const auto local_term_it=std::find_if(graph.local_terms.begin(),graph.local_terms.end(),
    [](const rpmd_ja_local_basis::Term& term){return term.atom_count==3;});
  if(local_term_it==graph.local_terms.end())throw std::runtime_error("M1 orthogonal-perturbation fixture has no three-site local block");
  const auto& local_term=*local_term_it;
  const int center_atom=local_term.site[1].atom,local_row=local_term.site[0].atom,local_column=local_term.site[2].atom;
  if(local_row==center_atom||local_column==center_atom||local_row==local_column)
    throw std::runtime_error("M1 orthogonal-perturbation fixture has repeated local atoms");
  std::vector<double> old_probe(d);for(int index=0;index<d;++index)
    old_probe[index]=std::sin(0.731*(index+1))+std::cos(0.173*(index+3));
  const int q0=local_row,q1=local_column,qc=center_atom;
  const double relative0=old_probe[q0]-old_probe[qc],relative1=old_probe[q1]-old_probe[qc];
  const double u0=relative1,u1=-relative0,uc=-(u0+u1);
  if(std::abs(u0)+std::abs(u1)<=1.0e-12||std::abs(u0*relative0+u1*relative1)>1.0e-12)
    throw std::runtime_error("M1 fixed-probe-orthogonal local perturbation direction is degenerate");
  const std::vector<double> old_action=additive_sparse_action(additive.k,old_probe,d);
  for(int row=0;row<d;++row){double independent_action=0.0;
    for(int column=0;column<d;++column)independent_action+=expected.k[static_cast<std::size_t>(row)*d+column]*old_probe[column];
    if(std::abs(old_action[row]-independent_action)>2.0e-10*std::max({1.0,std::abs(old_action[row]),std::abs(independent_action)}))
      throw std::runtime_error("actual M1 package does not match the former single-probe action baseline");}
  RpmdJAAdditiveData orthogonal_local_perturbation=additive;
  std::vector<RpmdJAAdditiveEntry> perturbed_entries;perturbed_entries.reserve(additive.k.values.size()+9);
  for(int row=0;row<d;++row)for(std::size_t index=additive.k.offsets[row];index<additive.k.offsets[row+1];++index)
    perturbed_entries.push_back({row,additive.k.columns[index],additive.k.values[index]});
  constexpr double perturbation=0.25;
  const int support[3]={q0,q1,qc};const double local_vector[3]={u0,u1,uc};
  for(int row=0;row<3;++row)for(int column=0;column<3;++column)
    perturbed_entries.push_back({support[row],support[column],perturbation*local_vector[row]*local_vector[column]});
  orthogonal_local_perturbation.k=rpmd_ja_additive_compress(d,perturbed_entries);
  const std::vector<double> changed_action=additive_sparse_action(orthogonal_local_perturbation.k,old_probe,d);
  double action_difference=0.0,action_scale=1.0;
  for(int row=0;row<d;++row){action_difference=std::max(action_difference,std::abs(changed_action[row]-old_action[row]));
    action_scale=std::max({action_scale,std::abs(changed_action[row]),std::abs(old_action[row])});}
  if(action_difference>2.0e-12*action_scale)
    throw std::runtime_error("symmetric local K perturbation is not orthogonal to the former fixed probe");
  require_entry_check_rejects(orthogonal_local_perturbation,"fixed-probe-orthogonal local K perturbation");

  bool entry_check_failure_rejected=false;
  try{prepare_rpmd_ja_qnep_reference(raw,entry_failure_path,kernel,{},pack,
      [](const RpmdJAAdditiveData&){throw std::runtime_error("injected independent package-entry callback failure");});}
  catch(const std::runtime_error& error){entry_check_failure_rejected=
    std::string(error.what())=="injected independent package-entry callback failure";}
  auto exists=[](const std::string& path){std::ifstream file(path,std::ios::binary);return file.good();};
  if(!entry_check_failure_rejected||exists(entry_failure_path)||exists(entry_failure_path+".stability")||
     exists(entry_failure_path+".tmp")||exists(entry_failure_path+".stability.tmp")||
     exists(entry_failure_path+".failure.txt"))
    throw std::runtime_error("failed package-entry validation left a publishable reference artifact");

  bool corrupted_after_publish=false;std::string readback_failure;
  rpmd_ja_qnep_prepare_test::after_publish_before_readback=[&](const std::string& published_path){
    if(published_path!=readback_failure_path)throw std::runtime_error("post-publish readback hook received an unexpected path");
    if(!exists(published_path)||!exists(published_path+".stability"))
      throw std::runtime_error("post-publish readback hook ran before output and sidecar were published");
    std::ofstream corrupt(published_path,std::ios::binary|std::ios::trunc);
    if(!corrupt)throw std::runtime_error("cannot corrupt the published reference for readback failure injection");
    corrupt.write("BADMAGIC",8);corrupt.close();
    if(!corrupt)throw std::runtime_error("failed writing the published reference readback failure fixture");
    corrupted_after_publish=true;
  };
  try{prepare_rpmd_ja_qnep_reference(raw,readback_failure_path,kernel,{},pack);}
  catch(const std::runtime_error& error){readback_failure=error.what();}
  rpmd_ja_qnep_prepare_test::after_publish_before_readback={};
  if(!corrupted_after_publish||readback_failure.find("unsupported RPMD-JA reference magic")!=0||
     exists(readback_failure_path)||exists(readback_failure_path+".stability")||
     exists(readback_failure_path+".tmp")||exists(readback_failure_path+".stability.tmp")||
     exists(readback_failure_path+".failure.txt"))
    throw std::runtime_error("post-publication readback failure did not clean the prepared reference artifacts");

  const RpmdJAReference result=read_rpmd_ja_reference(output_path);
  if(result.backend!=2||result.number_of_atoms!=n||!result.stability_checked||result.block_dynamical.tiles.empty()||
     result.block_site_transpose[0].tiles.empty()||result.mechanical_policy!=
       "native_reference_transport;internal_mass_com_pullback_v1;finite_temperature_additive_v1;beads=8;derivative=3")
    throw std::runtime_error("actual M1 version-6 package did not pass prepare/readback policy checks");
  for(int row=0;row<d;++row)for(int column=0;column<d;++column){
    const double expected_package=expected.k[static_cast<std::size_t>(row)*d+column];
    const double actual_package=additive_sparse_value(additive.k,row,column);
    if(std::abs(actual_package-expected_package)>3.0e-10*std::max(1.0,std::abs(expected_package)))
      throw std::runtime_error("actual M1 package K disagrees with the independent local-energy Hessian");
    const double expected_d=tile_value(baseline.block_dynamical,row,column)+expected_package/
      std::sqrt(masses[row%n]*masses[column%n]);
    const double actual_d=tile_value(result.block_dynamical,row,column);
    if(std::abs(actual_d-expected_d)>5.0e-8*std::max(1.0,std::abs(expected_d)))
      throw std::runtime_error("actual M1 prepare/readback D matrix disagrees with the independent local-energy Hessian");
  }
  for(int alpha=0;alpha<3;++alpha){
    std::vector<double> expected_stored(static_cast<std::size_t>(d)*d),omitted_additive(static_cast<std::size_t>(d)*d),
      transposed_additive(static_cast<std::size_t>(d)*d);
    double additive_norm=0.0,asymmetry=0.0;
    for(int row=0;row<d;++row)for(int column=0;column<d;++column){
      const double baseline_value=tile_value(baseline.block_site_transpose[alpha],row,column);
      const double mass=std::sqrt(masses[row%n]*masses[column%n]);
      const double correct=expected.h[alpha][static_cast<std::size_t>(row)*d+column]/mass;
      const double wrong_transpose=expected.h[alpha][static_cast<std::size_t>(column)*d+row]/mass;
      expected_stored[static_cast<std::size_t>(row)*d+column]=baseline_value+correct;
      omitted_additive[static_cast<std::size_t>(row)*d+column]=baseline_value;
      transposed_additive[static_cast<std::size_t>(row)*d+column]=baseline_value+wrong_transpose;
      additive_norm=std::hypot(additive_norm,correct);
      asymmetry=std::max(asymmetry,std::abs(correct-wrong_transpose));
      const double package_value=additive_sparse_value(additive.h[alpha],row,column);
      if(std::abs(package_value-expected.h[alpha][static_cast<std::size_t>(row)*d+column])>
         3.0e-10*std::max(1.0,std::abs(package_value)))
        throw std::runtime_error("actual M1 package H disagrees with the independent local-energy transport formula");
    }
    if(additive_norm<=1.0e-7||asymmetry<=1.0e-7)
      throw std::runtime_error("actual M1 integration fixture did not produce a resolvable asymmetric H contribution");
    auto matches=[&](const std::vector<double>& expected_matrix){
      for(int row=0;row<d;++row)for(int column=0;column<d;++column){
        const double actual=tile_value(result.block_site_transpose[alpha],row,column);
        const double value=expected_matrix[static_cast<std::size_t>(row)*d+column];
        if(std::abs(actual-value)>5.0e-8*std::max(1.0,std::abs(value)))return false;
      }
      return true;
    };
    if(!matches(expected_stored))throw std::runtime_error("actual M1 prepare/readback H matrix disagrees with the independent local-energy transport formula");
    if(matches(omitted_additive))throw std::runtime_error("actual M1 prepare/readback H check failed to detect an omitted additive H");
    if(matches(transposed_additive))throw std::runtime_error("actual M1 prepare/readback H check failed to detect transposed additive H");
  }
  if(!has_compressed_h(result))throw std::runtime_error("actual M1 prepare/readback did not retain compressed H tiles");
  const std::string owned_failure_path=base+".owned_failure.ja";
  const std::string owned_failure_sidecar=owned_failure_path+".stability";
  const std::string owned_failure_tmp=owned_failure_path+".tmp";
  const std::string owned_failure_sidecar_tmp=owned_failure_sidecar+".tmp";
  cleanup.paths.insert(cleanup.paths.end(),{owned_failure_path,owned_failure_sidecar,owned_failure_tmp,owned_failure_sidecar_tmp});
  auto path_exists=[](const std::string& path){std::ifstream file(path,std::ios::binary);return file.good();};
  bool post_publish_failure_injected=false;
  {
    NativeFitScratchOwnership ownership;ownership.raw=raw;ownership.pack=pack;
    ownership.output=owned_failure_path;ownership.sidecar=owned_failure_sidecar;
    ownership.output_tmp=owned_failure_tmp;ownership.sidecar_tmp=owned_failure_sidecar_tmp;
    ownership.raw_owned=true;ownership.pack_owned=true;
    covariance_fit_test_hooks.fail_after_prepare_publication=true;
    try{
      prepare_and_take_published_ownership(ownership,[&](){
        prepare_rpmd_ja_qnep_reference(raw,owned_failure_path,kernel,{},pack);
      },[&](){
        if(!path_exists(owned_failure_path)||!path_exists(owned_failure_sidecar))
          throw std::runtime_error("production ownership callback ran before prepare published both files");
        for(const std::string& temporary:{owned_failure_tmp,owned_failure_sidecar_tmp}){
          std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
          if(!out)throw std::runtime_error("cannot create post-prepare temporary failure fixture");
          out<<"interrupted atomic write";
        }
        maybe_fail_after_prepare_publication_for_test();
      });
    }catch(const std::runtime_error& error){
      post_publish_failure_injected=std::string(error.what())=="TEST_PREPARE_POST_PUBLICATION_FAILURE";
    }
    covariance_fit_test_hooks={};
  }
  if(!post_publish_failure_injected||path_exists(raw)||path_exists(pack)||path_exists(owned_failure_path)||
     path_exists(owned_failure_sidecar)||path_exists(owned_failure_tmp)||path_exists(owned_failure_sidecar_tmp))
    throw std::runtime_error("real prepare post-publication journal failure left owned reference, sidecar, temporary, or additive artifacts");

  const std::string transaction_job="integration-publish-"+base;
  const std::string transaction_output=base+".transaction.ja";
  const std::string transaction_stage=transaction_output+".stage."+
    rpmd_ja_covariance_checkpoint::sha256(transaction_job).substr(0,16);
  const std::string foreign_output=base+".foreign.ja",foreign_stage=foreign_output+".stage."+
    rpmd_ja_covariance_checkpoint::sha256(transaction_job).substr(0,16);
  cleanup.paths.insert(cleanup.paths.end(),{transaction_output,transaction_output+".stability",transaction_stage,
    transaction_stage+".stability",transaction_stage+".tmp",transaction_stage+".stability.tmp",
    foreign_output,foreign_output+".stability",foreign_stage,foreign_stage+".stability"});
  prepare_rpmd_ja_qnep_reference(raw,transaction_stage,kernel,{},pack);
  const std::string output_digest=fit_input_content_identity(transaction_stage,true).sha256;
  const std::string sidecar_digest=fit_input_content_identity(transaction_stage+".stability",true).sha256;
  bool two_rename_interruption=false;
  covariance_fit_test_hooks.fail_between_reference_and_sidecar_renames=true;
  try{publish_verified_prepared_artifacts(transaction_output,transaction_stage,output_digest,sidecar_digest,[&](){
      maybe_fail_between_publication_renames_for_test();});}
  catch(const std::runtime_error& error){two_rename_interruption=
    std::string(error.what())=="TEST_PUBLICATION_INTERRUPTED_BETWEEN_RENAMES";}
  covariance_fit_test_hooks.fail_between_reference_and_sidecar_renames=false;
  if(!two_rename_interruption||fit_input_content_identity(transaction_output,true).sha256!=output_digest||
     fit_input_content_identity(transaction_stage+".stability",true).sha256!=sidecar_digest)
    throw std::runtime_error("publication interruption did not preserve the exact first-final/second-stage transaction state");
  publish_verified_prepared_artifacts(transaction_output,transaction_stage,output_digest,sidecar_digest,[](){});
  if(fit_input_content_identity(transaction_output,true).sha256!=output_digest||
     fit_input_content_identity(transaction_output+".stability",true).sha256!=sidecar_digest||
     path_exists(transaction_stage)||path_exists(transaction_stage+".stability"))
    throw std::runtime_error("publication recovery did not finish both digest-bound renames");

  {std::ifstream input(transaction_output,std::ios::binary);std::ofstream output(foreign_stage,std::ios::binary|std::ios::trunc);
    output<<input.rdbuf();if(!input.eof()&&!input)throw std::runtime_error("cannot copy the prepared reference negative fixture");}
  {std::ifstream input(transaction_output+".stability",std::ios::binary);std::ofstream output(foreign_stage+".stability",std::ios::binary|std::ios::trunc);
    output<<input.rdbuf();if(!input.eof()&&!input)throw std::runtime_error("cannot copy the prepared certificate negative fixture");}
  {std::ofstream foreign(foreign_output,std::ios::binary|std::ios::trunc);foreign<<"foreign final";}
  bool foreign_final_rejected=false;
  try{publish_verified_prepared_artifacts(foreign_output,foreign_stage,output_digest,sidecar_digest,[](){});}
  catch(const std::runtime_error& error){foreign_final_rejected=
    std::string(error.what()).find("foreign or modified final/staged")!=std::string::npos;}
  std::ifstream foreign(foreign_output,std::ios::binary);std::string foreign_bytes;foreign>>foreign_bytes;
  if(!foreign_final_rejected||foreign_bytes!="foreign")
    throw std::runtime_error("publication recovery accepted or overwrote a foreign final reference");
}
} // namespace

int main(int argc,char** argv)
{
  try{
    if(argc>1&&std::string(argv[1])!="--integration-smoke")throw std::invalid_argument("usage: rpmd_ja_fit_prepare_integration_cuda_test [--integration-smoke]");
    const std::string base="rpmd_ja_actual_m1_prepare_"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    test_feasibility_manifest_parser_accepts_lf_and_crlf();
    test_frozen_fit_input_fingerprints_reject_mutation();
    test_manifest_sha256_rejects_qraw_and_kernel_replacement_at_native_start();
    test_native_additive_entry_verifier_rejects_small_scale_K_H();
    test_measure_rpmd_ja_use_propagation_order_and_reset();
    test_reference_use_compatibility_by_backend();
    test_graph_reference_samples_to_diagnostic_prepare_and_hard_failure();
    run_actual_m1_writer_prepare_readback(base);run_additive(base,1);run_additive(base+"_internal_com",2);
    std::printf("ACTUAL_M1_V2_PREPARE_READBACK_PASS\n");return 0;
  }catch(const std::exception& error){
    std::fprintf(stderr,"actual M1 prepare integration failed: %s\n",error.what());return 1;
  }
}
