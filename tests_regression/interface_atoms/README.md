# Native local atom dump fixture

`runtime/baseline/run.in` and `runtime/local_only/run.in` start from identical three-step NVE inputs. The baseline folder uses a full-system, double-precision `dump_xyz`; the local-only folder's first run uses only `dump_interface_atoms` for global atom IDs 0 and 1. Its second run enables standard `dump_xyz ... unwrapped_position` after the atoms have crossed a periodic seam, checking that late unwrapped-position enablement preserves image history. Run GPUMD separately from both folders, then invoke the C++ verifier from `runtime` with paths to `baseline/baseline.xyz`, `local_only/local_atoms.out`, and `local_only/late.xyz`. It compares positions, velocities (including A/fs to A/ps conversion), forces, site energies, IDs, steps, and times for the complete-block and tail frames, and checks the second-run unwrapped coordinate.

The verifier target is built by `tests_regression/interface_native/CMakeLists.txt`, but is not a no-input CTest. The host pack-helper assertions are registered with CTest. Example host-test commands from the repository root:

```powershell
cmake -S tests_regression/interface_native -B build/interface-native
cmake --build build/interface-native --config Release
ctest --test-dir build/interface-native -C Release --output-on-failure
```

Then, after a separate GPU-enabled GPUMD build, run `gpumd` in each fixture directory and invoke the built verifier from `runtime`:

```powershell
Push-Location baseline; gpumd; Pop-Location
Push-Location local_only; gpumd; Pop-Location
& ..\..\..\build\interface-native\Release\interface_atoms_runtime_verifier.exe baseline\baseline.xyz local_only\local_atoms.out local_only\late.xyz
```

For a single-configuration CMake generator, omit `Release` from the verifier path. Use fresh copies for reruns because native geometry outputs and `<prefix>_atoms.out` refuse existing files, while `dump_xyz` appends.
