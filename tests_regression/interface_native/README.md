# Unified host-only native tests

This CMake project builds the M0 geometry helpers, M1 profile math, M2 VACF normalization, M3 SHC/time/statistics assertions, and the M4 local atom pack helper with C++14. It has no CUDA language, GPU, Python, or added dependency requirement. The M4 runtime output verifier is built as a utility target, not registered as a no-input CTest.

Configure and run from the repository root:

```text
cmake -S tests_regression/interface_native -B build/interface_native
cmake --build build/interface_native --config Release
ctest --test-dir build/interface_native -C Release --output-on-failure
```

For a single-configuration generator, set `-DCMAKE_BUILD_TYPE=Release` during configure and omit `-C Release` from CTest. `tests_regression/interface_geometry_m3/CMakeLists.txt` remains the separate CUDA NEP consistency entry point.

The local atom dump runtime fixture is under `tests_regression/interface_atoms/runtime/{baseline,local_only}`. Run each `run.in` in its fixture directory with a GPU-enabled GPUMD build, then invoke the built `interface_atoms_runtime_verifier` from `runtime` with both output paths. The local-only fixture also checks a later run's unwrapped image history. Exact commands are in `tests_regression/interface_atoms/README.md`.
