cmake_minimum_required(VERSION 3.12)
project(GPUMDInterfaceNativeTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
get_filename_component(GPUMD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

function(add_interface_host_test target source)
  add_executable(${target} "${GPUMD_ROOT}/${source}")
  add_test(NAME ${target} COMMAND ${target})
endfunction()

add_interface_host_test(
  interface_geometry_m0_test
  tests_regression/interface_geometry_m2/native_math_test.cpp)
add_interface_host_test(
  interface_profile_m1_test
  tests_regression/interface_profile_m1/native_profile_math_test.cpp)
add_interface_host_test(
  interface_dos_m2_test
  tests_regression/interface_geometry_m2/interface_dos_math_test.cpp)
add_interface_host_test(
  interface_shc_m3_math_test
  tests_regression/interface_geometry_m3/interface_shc_math_test.cpp)
add_interface_host_test(
  interface_sampling_time_test
  tests_regression/interface_geometry_m3/interface_sampling_time_test.cpp)
add_interface_host_test(
  interface_shc_stats_test
  tests_regression/interface_geometry_m3/interface_shc_stats_test.cpp)
add_interface_host_test(
  interface_atoms_pack_test
  tests_regression/interface_atoms/interface_atoms_pack_test.cpp)

# This verifier consumes files produced by the native GPUMD runtime fixture and is
# intentionally built, but not registered as a no-input CTest.
add_executable(
  interface_atoms_runtime_verifier
  "${GPUMD_ROOT}/tests_regression/interface_atoms/runtime/verify_fixture.cpp")
