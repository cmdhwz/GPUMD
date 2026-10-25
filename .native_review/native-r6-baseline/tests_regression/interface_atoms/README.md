# Native local atom dump fixture

`runtime/run.in` writes a full-system, double-precision `dump_xyz` baseline and the new delayed local-window dump in the same three-step NVE run. The selected geometry window contains global atom IDs 0 and 1. After running GPUMD from the `runtime` directory, run the C++ verifier from that directory; it compares the positions, unwrapped positions, velocities (including A/fs to A/ps conversion), forces, site energies, IDs, steps, and times for both complete-block frames and the residual tail against `baseline.xyz`.

The verifier target is built by `tests_regression/interface_native/CMakeLists.txt`, but is not a no-input CTest. The host pack-helper assertions are registered with CTest. Example host-test commands from the repository root:

```powershell
cmake -S tests_regression/interface_native -B build/interface-native
cmake --build build/interface-native --config Release
ctest --test-dir build/interface-native -C Release --output-on-failure
```

Then, after a separate GPU-enabled GPUMD build, run `gpumd` in the fixture directory and invoke the built `interface_atoms_runtime_verifier` there. Use a fresh fixture directory for each run because native geometry outputs and `local_atoms.out` refuse existing files, while `baseline.xyz` appends.
