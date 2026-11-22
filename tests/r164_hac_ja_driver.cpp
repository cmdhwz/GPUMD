#include "r164_hac_ja.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc,char** argv) {
  if (argc!=4) { std::fprintf(stderr,"usage: r164_hac_ja_driver N Nc output_interval\n"); return 2; }
  int n=std::atoi(argv[1]), nc=std::atoi(argv[2]), interval=std::atoi(argv[3]);
  std::vector<double> v(n*9); std::vector<unsigned char> m(n);
  for (int i=0;i<n;++i) {
    m[i]=(i%7!=1 && i%11!=3);
    for (int j=0;j<9;++j) v[i*9+j]=std::sin(i*(j+1)*0.031)+i*0.0001*(j+1);
  }
  auto start=std::chrono::steady_clock::now();
  r164_hac_ja::write(".",v,m,9,nc,interval,1.0,2.0,1);
  std::printf("%.6f\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
}
