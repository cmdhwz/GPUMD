#ifdef NDEBUG
#undef NDEBUG
#endif
#include "../src/measure/rpmd_ja_additive.cuh"
#include <cassert>
#include <chrono>
#include <cstdio>

std::uint64_t rpmd_ja_model_fingerprint(const std::string&) { return 0; }

template <typename T> static void put(std::ostream& out, const T& value)
{
  out.write(reinterpret_cast<const char*>(&value), sizeof(value));
  assert(out.good());
}

static void compare(const std::vector<double>& a, const std::vector<double>& b)
{
  assert(a.size()==b.size());
  for(std::size_t i=0;i<a.size();++i)
    assert(std::abs(a[i]-b[i])<=2.0e-14*std::max({1.0,std::abs(a[i]),std::abs(b[i])}));
}

struct Edge { int j; int image[3]; };

int main()
{
  std::uint64_t cumulative_site_elements=0;
  for(int site=0;site<5896;++site)
    cumulative_site_elements+=rpmd_ja_additive_validate_site_dimension(133,site);
  assert(cumulative_site_elements>rpmd_ja_additive_max_expanded_elements);
  bool per_site_limit_rejected=false;
  try { (void)rpmd_ja_additive_validate_site_dimension(667,5896); }
  catch(const std::runtime_error& error) {
    per_site_limit_rejected=std::string(error.what()).find("2001x2001")!=std::string::npos;
  }
  assert(per_site_limit_rejected);
  constexpr int n=3,d=3*n,z=2,q=3*z;
  const std::vector<std::vector<Edge>> graph={
    {{1,{1,0,0}},{2,{0,-1,0}}},
    {{0,{-1,0,0}},{2,{0,0,0}}},
    {{1,{0,0,0}},{0,{0,1,0}}}};
  const double positions[d]={0.2,1.1,-0.3, 0.4,-0.2,1.7, 0.1,0.8,-0.6};
  const double cell[9]={2.0,0.1,0.0, 0.0,1.7,0.2, 0.1,0.0,2.3};
  const double ell[q]={0.3,-0.2,0.7,-0.1,0.5,0.4};
  double b[q*q]={};
  for(int x=0;x<q;++x)b[x*q+x]=1.0+0.2*x;
  b[0*q+4]=b[4*q+0]=0.35; b[1*q+5]=b[5*q+1]=-0.27; b[2*q+3]=b[3*q+2]=0.19;
  b[2*q+1]=-0.0;

  RpmdJAReference ref;ref.number_of_atoms=n;ref.temperature=300.0;
  ref.masses={1.0,2.0,5.0};
  ref.positions.assign(positions,positions+d);std::copy(cell,cell+9,ref.cell);
  std::vector<double> dense_k(d*d),dense_h[3],gradient(d);
  for(auto& h:dense_h)h.assign(d*d,0.0);
  for(int center=0;center<n;++center){
    const auto& edges=graph[center];const int qi=3*static_cast<int>(edges.size());
    std::vector<double> local_b(static_cast<std::size_t>(qi)*qi,0.0),local_ell(qi,0.0);
    if(center==0){local_b.assign(b,b+q*q);local_ell.assign(ell,ell+q);}
    std::vector<double> geometry(3*edges.size());
    for(std::size_t e=0;e<edges.size();++e)for(int axis=0;axis<3;++axis){
      double g=positions[axis*n+edges[e].j]-positions[axis*n+center];
      for(int c=0;c<3;++c)g+=cell[3*axis+c]*edges[e].image[c];
      geometry[3*e+axis]=g;
    }
    for(int x=0;x<qi;++x){const int axis=x%3,j=edges[x/3].j;gradient[axis*n+j]+=local_ell[x];gradient[axis*n+center]-=local_ell[x];}
    for(int x=0;x<qi;++x)for(int y=0;y<qi;++y){
      const int ax=x%3,ay=y%3;const auto& ex=edges[x/3];const auto& ey=edges[y/3];
      const int ix[2]={ax*n+ex.j,ax*n+center},iy[2]={ay*n+ey.j,ay*n+center};
      const double sign[2]={1.0,-1.0},value=local_b[static_cast<std::size_t>(x)*qi+y];
      for(int u=0;u<2;++u)for(int v=0;v<2;++v)dense_k[ix[u]*d+iy[v]]+=sign[u]*value*sign[v];
    }
    for(std::size_t e=0;e<edges.size();++e)for(int mu=0;mu<3;++mu){
      const int velocity=mu*n+edges[e].j;
      for(int y=0;y<qi;++y){const int axis=y%3;const auto& edge=edges[y/3];const int displacement[2]={axis*n+edge.j,axis*n+center};
        const double sign[2]={1.0,-1.0},value=local_b[(3*e+mu)*qi+y];
        for(int alpha=0;alpha<3;++alpha)for(int v=0;v<2;++v)
          dense_h[alpha][velocity*d+displacement[v]]-=geometry[3*e+alpha]*value*sign[v];
      }
    }
  }
  std::vector<double> raw_gradient(d);for(int i=0;i<d;++i)raw_gradient[i]=-gradient[i];
  const std::string package="rpmd_ja_additive_sparse_test_"+
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".bin";
  struct RemoveFile { std::string path; ~RemoveFile(){std::remove(path.c_str());} } cleanup{package};
  const std::string truncated_package=package+".truncated";RemoveFile truncated_cleanup{truncated_package};
  auto write_sparse_package=[&](const std::uint32_t version){
    std::ofstream out(package,std::ios::binary|std::ios::trunc);assert(out.good());
    out.write("GPJAADD1",8);put(out,version);put(out,std::uint32_t(0x01020304));
    put(out,std::int32_t(n));put(out,std::int32_t(2));put(out,std::uint64_t(0));
    put(out,300.0);put(out,0.05);put(out,0.0);put(out,0.1);put(out,std::uint64_t(1));put(out,std::uint64_t(1));
    for(int center=0;center<n;++center){
      const auto& edges=graph[center];const int qi=3*static_cast<int>(edges.size());
      put(out,std::int32_t(edges.size()));
      for(const auto& edge:edges){put(out,std::int32_t(edge.j));for(int a=0;a<3;++a)put(out,std::int32_t(edge.image[a]));}
      for(int x=0;x<qi;++x)put(out,center==0?ell[x]:0.0);
      std::vector<RpmdJAAdditiveEntry> entries;
      if(center==0)for(int row=0;row<qi;++row)for(int column=0;column<qi;++column){
        const double value=b[static_cast<std::size_t>(row)*qi+column];
        if(value!=0.0||std::signbit(value))entries.push_back({row,column,value});}
      put(out,static_cast<std::uint64_t>(entries.size()));
      for(const auto& entry:entries){put(out,std::int32_t(entry.row));put(out,std::int32_t(entry.column));put(out,entry.value);}
    }
    assert(out.good());
  };
  auto write_mode_package=[&](const std::uint32_t version,const std::uint32_t mode){
    std::ofstream out(package,std::ios::binary|std::ios::trunc);assert(out.good());
    out.write("GPJAADD1",8);put(out,version);put(out,std::uint32_t(0x01020304));
    put(out,std::int32_t(n));put(out,std::int32_t(2));put(out,std::uint64_t(0));
    put(out,300.0);put(out,0.05);put(out,0.0);put(out,0.1);put(out,std::uint64_t(1));put(out,std::uint64_t(1));
    for(int center=0;center<n;++center){
      const auto& edges=graph[center];const int qi=3*static_cast<int>(edges.size());
      put(out,std::int32_t(edges.size()));
      for(const auto& edge:edges){put(out,std::int32_t(edge.j));for(int a=0;a<3;++a)put(out,std::int32_t(edge.image[a]));}
      for(int x=0;x<qi;++x)put(out,center==0?ell[x]:0.0);
      put(out,mode);
      if(mode==0){for(int x=0;x<qi*qi;++x)put(out,center==0?b[x]:0.0);}
      else if(mode==1){
        std::vector<RpmdJAAdditiveEntry> entries;
        if(center==0)for(int row=0;row<qi;++row)for(int column=0;column<qi;++column){
          const double value=b[static_cast<std::size_t>(row)*qi+column];
          if(value!=0.0||std::signbit(value))entries.push_back({row,column,value});}
        put(out,static_cast<std::uint64_t>(entries.size()));
        for(const auto& entry:entries){put(out,std::int32_t(entry.row));put(out,std::int32_t(entry.column));put(out,entry.value);}
      }else{
        std::vector<std::array<int,2>> blocks;
        if(center==0)for(int row=0;row<static_cast<int>(edges.size());++row)
          for(int column=0;column<static_cast<int>(edges.size());++column){bool active=false;
            for(int a=0;a<3;++a)for(int c=0;c<3;++c){const double value=b[static_cast<std::size_t>(3*row+a)*qi+3*column+c];
              active=active||value!=0.0||std::signbit(value);}
            if(active)blocks.push_back({row,column});}
        put(out,static_cast<std::uint64_t>(blocks.size()));
        for(const auto& index:blocks){put(out,std::int32_t(index[0]));put(out,std::int32_t(index[1]));
          for(int a=0;a<3;++a)for(int c=0;c<3;++c)put(out,b[static_cast<std::size_t>(3*index[0]+a)*qi+3*index[1]+c]);}
      }
    }
    assert(out.good());
  };
  {
    std::ofstream out(package,std::ios::binary|std::ios::trunc);assert(out.good());
    out.write("GPJAADD1",8);put(out,std::uint32_t(1));put(out,std::uint32_t(0x01020304));
    put(out,std::int32_t(n));put(out,std::int32_t(2));put(out,std::uint64_t(0));
    put(out,300.0);put(out,0.05);put(out,0.0);put(out,0.1);put(out,std::uint64_t(1));put(out,std::uint64_t(1));
    for(int center=0;center<n;++center){
      const auto& edges=graph[center];const int qi=3*static_cast<int>(edges.size());
      put(out,std::int32_t(edges.size()));
      for(const auto& edge:edges){put(out,std::int32_t(edge.j));for(int a=0;a<3;++a)put(out,std::int32_t(edge.image[a]));}
      for(int x=0;x<qi;++x)put(out,center==0?ell[x]:0.0);
      for(int x=0;x<qi*qi;++x)put(out,center==0?b[x]:0.0);
    }
  }
  const auto actual=read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient);
  auto verify_operators=[&](const RpmdJAAdditiveData& data){
    std::vector<double> dense_k_actual(d*d);
    for(int row=0;row<d;++row)for(std::size_t p=data.k.offsets[row];p<data.k.offsets[row+1];++p)
      dense_k_actual[static_cast<std::size_t>(row)*d+data.k.columns[p]]=data.k.values[p];
    compare(dense_k_actual,dense_k);
    for(int alpha=0;alpha<3;++alpha){
      std::vector<double> dense_h_actual(d*d);
      for(int row=0;row<d;++row)for(std::size_t p=data.h[alpha].offsets[row];p<data.h[alpha].offsets[row+1];++p)
        dense_h_actual[static_cast<std::size_t>(row)*d+data.h[alpha].columns[p]]=data.h[alpha].values[p];
      compare(dense_h_actual,dense_h[alpha]);
    }
    compare(data.linear_gradient,gradient);
  };
  verify_operators(actual);
  assert(dense_h[0][1*d+5]!=dense_h[0][5*d+1]);

  std::vector<double> internal_raw_gradient=raw_gradient;
  for(int axis=0;axis<3;++axis){
    const double net=3.0+axis;
    for(int i=0;i<n;++i)internal_raw_gradient[axis*n+i]+=ref.masses[i]/10.0*net;
  }
  {
    std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);
    out.seekp(8);put(out,std::uint32_t(2));
  }
  const auto internal=read_rpmd_ja_additive(package,"stub.raw",ref,-1,internal_raw_gradient);
  assert(internal.internal_mass_com);
  compare(internal.linear_gradient,gradient);
  {
    std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);
    out.seekp(8);put(out,std::uint32_t(1));
  }
  bool legacy_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,internal_raw_gradient); }
  catch(const std::runtime_error&) { legacy_rejected=true; }
  assert(legacy_rejected);
  write_sparse_package(3);
  const auto sparse=read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient);
  assert(!sparse.internal_mass_com);verify_operators(sparse);
  bool non_com_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,internal_raw_gradient); }
  catch(const std::runtime_error&) { non_com_rejected=true; }
  assert(non_com_rejected);
  {
    std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);
    out.seekp(static_cast<std::streamoff>(80+4+16*z+8*q+8));put(out,std::int32_t(q));
  }
  bool bad_index_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient); }
  catch(const std::runtime_error&) { bad_index_rejected=true; }
  assert(bad_index_rejected);
  write_sparse_package(4);
  const auto internal_sparse=read_rpmd_ja_additive(package,"stub.raw",ref,-1,internal_raw_gradient);
  assert(internal_sparse.internal_mass_com);verify_operators(internal_sparse);
  bool version_rejected=false;
  write_sparse_package(3);
  {std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);out.seekp(8);put(out,std::uint32_t(7));}
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient); }
  catch(const std::runtime_error&) { version_rejected=true; }
  assert(version_rejected);
  write_sparse_package(4);
  {
    std::ifstream in(package,std::ios::binary|std::ios::ate);assert(in.good());
    const std::streamoff size=in.tellg();assert(size>1);std::vector<char> bytes(static_cast<std::size_t>(size-1));
    in.seekg(0);in.read(bytes.data(),static_cast<std::streamsize>(bytes.size()));assert(in.good());
    std::ofstream out(truncated_package,std::ios::binary|std::ios::trunc);
    out.write(bytes.data(),static_cast<std::streamsize>(bytes.size()));assert(out.good());
  }
  bool truncated_rejected=false;
  try { (void)read_rpmd_ja_additive(truncated_package,"stub.raw",ref,-1,internal_raw_gradient); }
  catch(const std::runtime_error&) { truncated_rejected=true; }
  assert(truncated_rejected);
  for(const std::uint32_t mode:{0U,1U,2U}){
    write_mode_package(5,mode);
    const auto encoded=read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient);
    assert(!encoded.internal_mass_com);verify_operators(encoded);
  }
  write_mode_package(6,2);
  const auto encoded_com=read_rpmd_ja_additive(package,"stub.raw",ref,-1,internal_raw_gradient);
  assert(encoded_com.internal_mass_com);verify_operators(encoded_com);
  write_mode_package(5,0);
  {
    std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);
    out.seekp(static_cast<std::streamoff>(80+4+16*z+8*q));put(out,std::uint32_t(3));
  }
  bool bad_mode_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient); }
  catch(const std::runtime_error&) { bad_mode_rejected=true; }
  assert(bad_mode_rejected);
  write_mode_package(5,2);
  {
    std::fstream out(package,std::ios::binary|std::ios::in|std::ios::out);
    out.seekp(static_cast<std::streamoff>(80+4+16*z+8*q+4+8));put(out,std::int32_t(z));
  }
  bool bad_block_index_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient); }
  catch(const std::runtime_error&) { bad_block_index_rejected=true; }
  assert(bad_block_index_rejected);
  {
    std::ofstream out(package,std::ios::binary|std::ios::trunc);assert(out.good());
    out.write("GPJAADD1",8);put(out,std::uint32_t(5));put(out,std::uint32_t(0x01020304));
    put(out,std::int32_t(1000));put(out,std::int32_t(2));put(out,std::uint64_t(0));
    put(out,300.0);put(out,0.05);put(out,0.0);put(out,0.1);put(out,std::uint64_t(1));put(out,std::uint64_t(1));
    put(out,std::int32_t(3000));for(int site=1;site<1000;++site)put(out,std::int32_t(0));
  }
  RpmdJAReference large_ref;large_ref.number_of_atoms=1000;large_ref.temperature=300.0;
  bool oversized_rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",large_ref,-1,std::vector<double>(3000,0.0)); }
  catch(const std::runtime_error& error) { oversized_rejected=std::string(error.what()).find("9000x9000")!=std::string::npos; }
  assert(oversized_rejected);
  write_sparse_package(4);

  ref.cell[0]=std::numeric_limits<double>::infinity();
  bool rejected=false;
  try { (void)read_rpmd_ja_additive(package,"stub.raw",ref,-1,raw_gradient); }
  catch(const std::runtime_error&) { rejected=true; }
  assert(rejected);
}
