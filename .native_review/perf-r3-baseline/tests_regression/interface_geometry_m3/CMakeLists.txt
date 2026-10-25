cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 CACHE STRING "CUDA architecture for the NEP plane-flux test")
endif()

project(NEPPlaneFluxConsistency LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

get_filename_component(GPUMD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_executable(nep_plane_flux_consistency
  nep_plane_flux_consistency.cu
  "${GPUMD_ROOT}/src/force/nep.cu"
  "${GPUMD_ROOT}/src/force/neighbor.cu"
  "${GPUMD_ROOT}/src/force/potential.cu"
  "${GPUMD_ROOT}/src/force/dftd3.cu"
  "${GPUMD_ROOT}/src/model/box.cu"
  "${GPUMD_ROOT}/src/utilities/compact_nep.cu"
  "${GPUMD_ROOT}/src/utilities/error.cu"
  "${GPUMD_ROOT}/src/utilities/read_file.cu"
  "${GPUMD_ROOT}/src/utilities/run_input.cu")
target_include_directories(nep_plane_flux_consistency PRIVATE "${GPUMD_ROOT}/src")
target_link_libraries(nep_plane_flux_consistency PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)

enable_testing()
add_executable(interface_shc_math_test interface_shc_math_test.cpp)
add_test(NAME interface_shc_math_test COMMAND interface_shc_math_test)
add_executable(interface_sampling_time_test interface_sampling_time_test.cpp)
add_test(NAME interface_sampling_time_test COMMAND interface_sampling_time_test)
add_executable(interface_shc_stats_test interface_shc_stats_test.cpp)
add_test(NAME interface_shc_stats_test COMMAND interface_shc_stats_test)

add_test(NAME nep_plane_flux_consistency
  COMMAND nep_plane_flux_consistency "${GPUMD_ROOT}/tests_pytest/fixtures/models/nep_C.txt")
set_tests_properties(nep_plane_flux_consistency PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
