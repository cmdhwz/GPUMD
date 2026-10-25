cmake_minimum_required(VERSION 3.12)
project(GPUMDInterfaceNativeTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GPUMD_EXECUTABLE "" CACHE FILEPATH "External GPU-enabled GPUMD executable for interface runtime fixtures")
option(GPUMD_RUN_LONG_INTERFACE_TESTS "Register the multi-thousand-step heat_lan acceptance fixture" OFF)

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

add_executable(
  interface_native_fixture_prepare
  "${GPUMD_ROOT}/tests_regression/interface_native/interface_fixture_prepare.cpp")
add_executable(
  interface_native_fixture_verify
  "${GPUMD_ROOT}/tests_regression/interface_native/interface_fixture_verify.cpp")

if(NOT "${GPUMD_EXECUTABLE}" STREQUAL "")
  if(NOT EXISTS "${GPUMD_EXECUTABLE}" OR IS_DIRECTORY "${GPUMD_EXECUTABLE}")
    message(FATAL_ERROR "GPUMD_EXECUTABLE must name an existing GPUMD executable: ${GPUMD_EXECUTABLE}")
  endif()
  set(INTERFACE_CARBON_MODEL "${GPUMD_ROOT}/tests_pytest/fixtures/models/nep_C.txt")
  set(INTERFACE_RUNTIME_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/run_gpumd_fixture.cmake")
  foreach(mode IN ITEMS nve empty_shc geometry_split geometry_legacy)
    add_test(NAME interface_runtime_${mode}
      COMMAND ${CMAKE_COMMAND}
        "-DGPUMD_EXECUTABLE=${GPUMD_EXECUTABLE}"
        "-DPREPARE=$<TARGET_FILE:interface_native_fixture_prepare>"
        "-DVERIFY=$<TARGET_FILE:interface_native_fixture_verify>"
        "-DCARBON_MODEL=${INTERFACE_CARBON_MODEL}"
        "-DBUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime/interface_${mode}"
        "-DMODE=${mode}"
        -P "${INTERFACE_RUNTIME_SCRIPT}")
  endforeach()
  if(GPUMD_RUN_LONG_INTERFACE_TESTS)
    add_test(NAME interface_runtime_heat_lan
      COMMAND ${CMAKE_COMMAND}
        "-DGPUMD_EXECUTABLE=${GPUMD_EXECUTABLE}"
        "-DPREPARE=$<TARGET_FILE:interface_native_fixture_prepare>"
        "-DVERIFY=$<TARGET_FILE:interface_native_fixture_verify>"
        "-DCARBON_MODEL=${INTERFACE_CARBON_MODEL}"
        "-DBUILD_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/runtime/interface_heat_lan"
        "-DMODE=heat_lan"
        -P "${INTERFACE_RUNTIME_SCRIPT}")
    set_tests_properties(interface_runtime_heat_lan PROPERTIES LABELS "long;gpu;interface")
  else()
    message(STATUS "Long heat_lan runtime acceptance test is opt-in: set GPUMD_RUN_LONG_INTERFACE_TESTS=ON")
  endif()
else()
  message(STATUS "Interface GPUMD runtime tests NOT registered; set GPUMD_EXECUTABLE to an external GPU-enabled executable")
endif()
