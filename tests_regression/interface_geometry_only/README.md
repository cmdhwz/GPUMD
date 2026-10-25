# Geometry-only runtime fixture

Build the reader with `c++ -std=c++17 -O0 -g verify_geometry_only.cpp -o verify_geometry_only`, then run `gpumd` in this directory with `run.in` and execute the reader. This exercises the real `interface_geometry -> Measure::pre_run -> InterfaceGeometry::initialize` path without an M1/M2 consumer and checks all three output files.

The fixture has not been executed here. GPU runtime checks need a CUDA GPUMD binary.
