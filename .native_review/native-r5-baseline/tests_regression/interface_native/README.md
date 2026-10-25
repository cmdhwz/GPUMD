# Unified host-only native tests

This CMake project builds the M0 geometry helpers, M1 profile math, M2 VACF normalization, and M3 SHC/time/statistics assertions with C++14. It has no CUDA language, GPU, Python, or added dependency requirement.

Configure and run from the repository root:

```text
cmake -S tests_regression/interface_native -B build/interface_native
cmake --build build/interface_native --config Release
ctest --test-dir build/interface_native -C Release --output-on-failure
```

For a single-configuration generator, set `-DCMAKE_BUILD_TYPE=Release` during configure and omit `-C Release` from CTest. `tests_regression/interface_geometry_m3/CMakeLists.txt` remains the separate CUDA NEP consistency entry point.
