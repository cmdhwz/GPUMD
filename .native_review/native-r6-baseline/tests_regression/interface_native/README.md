# Unified host-only native tests

This CMake project builds the M0 geometry helpers, M1 profile math, M2 VACF normalization, M3 SHC/time/statistics assertions, and the M4 local atom pack helper with C++14. It has no CUDA language, GPU, Python, or added dependency requirement. The M4 runtime output verifier is built as a utility target, not registered as a no-input CTest.

Configure and run from the repository root:

```text
cmake -S tests_regression/interface_native -B build/interface_native
cmake --build build/interface_native --config Release
ctest --test-dir build/interface_native -C Release --output-on-failure
```

For a single-configuration generator, set `-DCMAKE_BUILD_TYPE=Release` during configure and omit `-C Release` from CTest. `tests_regression/interface_geometry_m3/CMakeLists.txt` remains the separate CUDA NEP consistency entry point.

The local atom dump runtime fixture is under `tests_regression/interface_atoms/runtime`. Run its `run.in` in a fresh directory with a GPU-enabled GPUMD build, then execute the built `interface_atoms_runtime_verifier` from that directory to compare the local rows against the same-run full-system `dump_xyz` baseline. The fixture instructions are in `tests_regression/interface_atoms/README.md`.
