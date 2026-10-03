# RPMD-JA reference math CPU check

From the repository root, compile and run the standalone C++14 check with:

```sh
g++ -std=c++14 -O2 -Isrc tests/rpmd_ja_reference_math_test.cpp -o rpmd_ja_reference_math_test
./rpmd_ja_reference_math_test
```

It executes the production scalar math header and checks it against an independent direct Fock-weight oracle at moderate frequencies, the zero-axis limit for `u=10, v=1e-14` in both orders, the complete large-`b` limit, a nonsymmetric row-major matrix, a finite-difference site-flow derivative, and translation-complement orthogonality.
