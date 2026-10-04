Run the production-math CPU oracle with:

```sh
g++ -std=c++14 -O2 -Isrc tests/rpmd_ja_sparse_measurement_test.cpp -o rpmd_ja_sparse_measurement_test
./rpmd_ja_sparse_measurement_test
```

The oracle checks the Chebyshev recurrence against explicit `T2(Z)=2Z^2-I` and `T3(Z)=4Z^3-3Z`, then independently expands the P and Q bilinear sums for three nonsymmetric directional operators and distinct displacement/velocity vectors.

Run the CUDA fixture against the production workspace with:

```sh
nvcc -std=c++14 -Isrc tests/rpmd_ja_sparse_cuda_test.cu src/measure/rpmd_ja_sparse.cu src/measure/rpmd_ja_qnep_cached.cu src/utilities/error.cu -lcublas -lcusolver -o rpmd_ja_sparse_cuda_test
./rpmd_ja_sparse_cuda_test
```

The CUDA fixture calls `RpmdJASparseWorkspace::initialize` and `compute_correction` directly. It covers the pure-NEP CSR path and fixed-reference qNEP cached modal path, including asymmetric `B^T`, a positive soft mode, unequal-mass translation projection, zero-rank kernels, common translation and velocity invariance, full-fractional multi-cell wrapping in a triclinic cell, and latched non-finite device errors. The CUDA command remains unrun until a CUDA compiler and GPU are available.
