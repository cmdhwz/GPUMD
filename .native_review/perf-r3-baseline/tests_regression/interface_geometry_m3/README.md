# M3 native verification fixtures

`interface_shc_math_test.cpp` is the host-only assertion test for directed and inverse pair signs, periodic seams, nonperiodic cuts, the reference MIC branch, compact selection/bounds, bilateral correlations, grid-resolution rejection, and analytic versus numeric trapezoid integration. Run it with `c++ -std=c++17 -O0 -g interface_shc_math_test.cpp -o interface_shc_math_test`.

`interface_sampling_time_test.cpp` calls the production fixed-step/time helpers for one million 1 fs steps, large run starts at 2000 ps and 10000 ps, a rejected 0.5 fs change, and full blocks with sample intervals 1 and 3. `interface_shc_stats_test.cpp` calls the production block-range and Welford helpers; it checks the 100/10 two-block mean of 55, first-block `nan` error, second-block error of 45, and cumulative/current ranges. Both are built and run by CTest.

The SHC action now stores sampled plane flux in a reusable GPU buffer sized for one block and copies sampled rows directly into the preallocated host cache only at block boundaries. The GPU atom history remains bounded by the selected plane atoms and `Nc`; the host cache grows with all sampled flux rows and complete-block correlations. The action prints both allocations before the run. It writes flux, correlation, and spectrum data during `post_run`, so a process interruption before then can lose the buffered results. A tail block keeps flux rows but contributes no spectrum. `interface_shc_stats_test.cpp` exercises the production cache-span helpers with two full blocks and a tail, the GPU fill index limit, and sticky first-error/reset behavior. The empty-plane fixture verifier checks zero plane flux, correlation, and spectrum output when no atoms are selected.

`nep_plane_flux_consistency.cu` directly constructs the repository's `NEP` class using `tests_pytest/fixtures/models/nep_C.txt`. It sets `N1=0, N2=N`, builds a distorted diamond cell with 21.402 A periodic x/y and nonperiodic z, arms the production `NEPPlaneFluxView`, and compares site energies, forces, and virials with the view disabled. It finite-differences the actual NEP site-energy partition for atoms on both sides of an internal plane at two displacement sizes, then checks the partition energy-rate identity against the captured plane flux. This is the real NEP consistency test; it does not copy the NEP derivative formula into the test.

The standalone CMake target links the existing NEP, neighbor, potential, DFTD3, box, and required utility translation units without linking the GPUMD main. Its default CUDA architecture is sm_60 because the production NEP path uses double-precision `atomicAdd`; override it for a modern GPU when building locally. With CUDA Toolkit and an available GPU, run it using:

```
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The build command includes the CUDA consistency executable and all three host-only tests. The architecture override is optional; when omitted, the standalone target uses the sm_60 default. For a single-configuration generator, `--config Release` may be omitted or selected with `-DCMAKE_BUILD_TYPE=Release` at configure time.

The `run.in` fixture exercises the registered Action on a deterministically initialized, slightly distorted 6x6x6 diamond-carbon input for eight steps, producing two complete four-sample blocks. `verify_fixture.cpp` checks nonzero finite flux/correlation, finite two-block errors, the version-3 SHC schema, cumulative and current-block time ranges/sample counts, and consistency of the exported-grid trapezoid with the analytic partial-range integral. Run the GPUMD executable from this directory, then compile and run the verifier with a host C++17 compiler.

`empty_plane/run.in` is a separate small orthogonal-box input with its nonperiodic plane in vacuum, outside the selected-atom cutoff. Its verifier checks that the zero-selected-atom path emits finite zeros without launching zero-sized GPU work. `unsupported_ensemble/run.in` is an expected-failure input for the `ti_spring` rejection guard.

These fixtures are software-path checks, not material-property validation. In the current environment no CUDA/C++ compiler or GPUMD executable is available, so none of the C++/CUDA assertions, build target, or runtime inputs have been run.
