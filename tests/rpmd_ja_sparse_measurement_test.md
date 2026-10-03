Run the production-math CPU oracle with:

```sh
g++ -std=c++14 -O2 -Isrc tests/rpmd_ja_sparse_measurement_test.cpp -o rpmd_ja_sparse_measurement_test
./rpmd_ja_sparse_measurement_test
```

The oracle checks the Chebyshev recurrence against explicit `T2(Z)=2Z^2-I` and `T3(Z)=4Z^3-3Z`, then independently expands the P and Q bilinear sums for three nonsymmetric directional operators and distinct displacement/velocity vectors.

Run the CUDA fixture against the production workspace with:

```sh
nvcc -std=c++17 -Isrc tests/rpmd_ja_sparse_cuda_test.cu src/measure/rpmd_ja_sparse.cu src/utilities/error.cu -o rpmd_ja_sparse_cuda_test
./rpmd_ja_sparse_cuda_test
```

The CUDA fixture calls `RpmdJASparseWorkspace::initialize` and `compute_correction` directly. It covers asymmetric `B^T`, signed modal weights, three directions including a zero direction, unequal-mass translation projection, full-fractional multi-cell wrapping in a triclinic cell, and the latched non-finite device error. The CUDA command remains unrun until a CUDA compiler and GPU are available.
