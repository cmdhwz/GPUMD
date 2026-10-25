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

The native interface fixture preparer and verifier are host C++ targets and do not link GPUMD. To register actual GPUMD runtime tests, configure with `-DGPUMD_EXECUTABLE=<absolute-path-to-gpumd>`; a nonempty path that does not name a file is a configure error. CTest registers the short NVE/profile/SHC check, explicit-empty-SHC rejection, a geometry split check with distinct physical/sampling planes, and a legacy unsorted-interface fallback check. The 20,000-step `heat_lan` sign/current check is separately opt-in with `-DGPUMD_RUN_LONG_INTERFACE_TESTS=ON`. All fixture files and outputs are generated below fixed mode directories under the CMake build runtime directory, and the carbon NEP model is read from the repository fixture without being copied.

The NVE verifier checks the fixed reference-bin energy change against trapezoidal integration of `Q_left-Q_right`, verifies M1 interface-crossing labels against an empty physical-interface list despite two nonempty bulk sampling planes, and independently compares same-block mean flux, C0, and a full-Nyquist trapezoid recomputed from the exported frequency rows. The separate geometry split test puts one physical plane inside bin 1 and a sampling plane inside bin 4; the profile must flag only bin 1 while SHC reports the sampling coordinate. The legacy test omits `sampling_planes`, provides physical coordinates in `7,3` order, and verifies sorted fallback in metadata, M1 labels, and SHC plane IDs.

The opt-in heat fixture uses nonperiodic z and transverse periodicity, 10,000 warm-up steps, then a 10,000-step candidate production window (reported steps 10001-20000) split into two halves. It computes `P_in_source=-s_source`, `P_in_sink=-s_sink`, and `Q_bath=(P_in_source-P_in_sink)/2`. Both half windows check each signal against its block variability, paired source/sink and plane/bath residuals, half-to-half bath/plane stability, and blockwise total-energy slope. A residual whose paired block-SE exceeds its explicit budget fails as statistically insufficient; adjacent blocks are not assumed independent confidence intervals. Tolerances remain uncalibrated because the fixture has not run. These all-carbon fixtures test Action/geometry plumbing, not a heterogeneous-material model or material transport acceptance.
