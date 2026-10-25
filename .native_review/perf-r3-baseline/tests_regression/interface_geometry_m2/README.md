# M0/M1/M2 native fixtures

`native_math_test.cpp` checks PBC wrapping, bin/material/fixed labels, multiple interface planes, empty mappings, per-lag origins and ring reset, plus a pure-sinusoid peak and the raw-C0 full-Nyquist integral. From this directory, build it with a C++17 compiler:

```text
c++ -std=c++17 -O0 -g native_math_test.cpp -o native_math_test
native_math_test
```

Run GPUMD from this directory with `run.in`, then build and run `verify_fixture.cpp` with `c++ -std=c++17 -O0 -g verify_fixture.cpp -o verify_fixture`. It checks the generated geometry map (including a bath/non-bath material mixture), the 19-column shared profile with a full and residual frame, repeated DOS actions for a range/type selection, VACF/VDOS row shapes, and `nan` output for an empty bin. The `zero_direction/` fixture separately checks a populated, exactly stationary y/z channel: its raw VACF, VDOS, and integrals are zero while only normalized VACF is `nan`. Run the `bad_bath_groups/` and `region_heat/` inputs separately; both are valid GPUMD heat-Langevin configurations that should be rejected by `interface_geometry` because the group IDs do not match or the thermostat uses fractional regions. Run the `nonorthogonal/` and `deform/` fixtures separately; each should exit with a geometry rejection. The standalone `tests_regression/interface_geometry_only/` fixture exercises real `InterfaceGeometry::initialize` without a consumer.

Run GPUMD from `zero_direction/` with its `run.in`, then compile and run `verify_zero_direction.cpp` there with `c++ -std=c++17 -O0 -g verify_zero_direction.cpp -o verify_zero_direction`. The heat-bath rejection fixtures should exit during pre-run initialization with the corresponding mismatch or region-mode diagnostic.

These fixtures have not been executed in this environment. A CUDA GPUMD build and a host C++ compiler are needed for the runtime/action checks.
