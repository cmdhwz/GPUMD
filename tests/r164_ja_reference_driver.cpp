#include "../src/measure/r164_ja_reference.h"
#include "../src/measure/r164_r27.h"
#include <cstring>
#include <iostream>
int main(int argc, char** argv) {
  if (argc<2) return 2;
  try {
    const auto x=read_r164_ja_reference(argv[1]);
    if (argc>=3) {
      if (argc==4 && x.file_sha256!=argv[3]) return 4;
      for (int a=0;a<3;++a) {
        if (x.row[a]!=r164_r27::row[a]) return 3;
        for (int d=0;d<3;++d) {
          if (std::memcmp(&x.reference[a][d],&r164_r27::reference[a][d],sizeof(double))) return 3;
          for (int k=0;k<2;++k) if (std::memcmp(&x.basis[a][d][k],&r164_r27::basis[a][d][k],sizeof(double))) return 3;
        }
      }
      for (int d=0;d<3;++d) for (int k=0;k<2;++k) for (int j=0;j<2;++j)
        if (std::memcmp(&x.D[d][k][j],&r164_r27::D[d][k][j],sizeof(double))) return 3;
    }
  } catch (const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
