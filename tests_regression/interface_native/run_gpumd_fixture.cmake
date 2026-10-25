foreach(required IN ITEMS GPUMD_EXECUTABLE PREPARE VERIFY CARBON_MODEL BUILD_ROOT WORK_DIR MODE)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "Missing -D${required} for interface GPUMD runtime test")
  endif()
endforeach()
if(NOT MODE MATCHES "^(nve|empty_shc|heat_lan|geometry_split|geometry_legacy)$")
  message(FATAL_ERROR "Unsupported interface fixture mode: ${MODE}")
endif()
foreach(file_var IN ITEMS GPUMD_EXECUTABLE PREPARE VERIFY CARBON_MODEL)
  if(NOT EXISTS "${${file_var}}" OR IS_DIRECTORY "${${file_var}}")
    message(FATAL_ERROR "${file_var} must be an existing file: ${${file_var}}")
  endif()
endforeach()

get_filename_component(build_root_real "${BUILD_ROOT}" REALPATH)
if(NOT EXISTS "${build_root_real}/CMakeCache.txt" OR IS_DIRECTORY "${build_root_real}/CMakeCache.txt")
  message(FATAL_ERROR "BUILD_ROOT must be the configured CMake binary directory")
endif()
set(allowed_runtime_root "${build_root_real}/runtime")
if(IS_SYMLINK "${allowed_runtime_root}")
  message(FATAL_ERROR "Build runtime root must not be a symlink")
endif()
file(MAKE_DIRECTORY "${allowed_runtime_root}")
get_filename_component(allowed_runtime_root_real "${allowed_runtime_root}" REALPATH)
if(NOT allowed_runtime_root_real STREQUAL allowed_runtime_root)
  message(FATAL_ERROR "Build runtime root resolves outside the configured build directory")
endif()
get_filename_component(work_dir_name "${WORK_DIR}" NAME)
get_filename_component(work_dir_parent_real "${WORK_DIR}/.." REALPATH)
if(NOT work_dir_name STREQUAL "interface_${MODE}" OR
   NOT work_dir_parent_real STREQUAL allowed_runtime_root_real)
  message(FATAL_ERROR "WORK_DIR must be the fixed mode directory under the build runtime root")
endif()
set(expected_work_dir "${allowed_runtime_root_real}/interface_${MODE}")
if(EXISTS "${expected_work_dir}")
  if(IS_SYMLINK "${expected_work_dir}")
    message(FATAL_ERROR "WORK_DIR must not be a symlink")
  endif()
  get_filename_component(work_dir_real "${expected_work_dir}" REALPATH)
  if(NOT work_dir_real STREQUAL expected_work_dir)
    message(FATAL_ERROR "WORK_DIR resolves outside its fixed build runtime directory")
  endif()
endif()

file(REMOVE_RECURSE "${expected_work_dir}")
file(MAKE_DIRECTORY "${expected_work_dir}")
execute_process(
  COMMAND "${PREPARE}" "${MODE}" "${expected_work_dir}" "${CARBON_MODEL}"
  RESULT_VARIABLE prepare_result
  OUTPUT_VARIABLE prepare_stdout
  ERROR_VARIABLE prepare_stderr)
if(NOT prepare_result EQUAL 0)
  message(FATAL_ERROR "Could not prepare interface fixture: ${prepare_stdout}${prepare_stderr}")
endif()

execute_process(
  COMMAND "${GPUMD_EXECUTABLE}"
  WORKING_DIRECTORY "${expected_work_dir}"
  RESULT_VARIABLE gpumd_result
  OUTPUT_VARIABLE gpumd_stdout
  ERROR_VARIABLE gpumd_stderr)
set(gpumd_output "${gpumd_stdout}\n${gpumd_stderr}")

if(MODE STREQUAL "empty_shc")
  if(gpumd_result EQUAL 0)
    message(FATAL_ERROR "SHC accepted an explicit empty sampling plane list")
  endif()
  string(FIND "${gpumd_output}" "requires shared geometry with at least one sampling plane" error_position)
  if(error_position EQUAL -1)
    message(FATAL_ERROR "SHC failed for an unexpected reason: ${gpumd_output}")
  endif()
  file(READ "${expected_work_dir}/native_shared_geometry.json" geometry_json)
  string(FIND "${geometry_json}" "\"sampling_planes_A\": []" empty_sampling_position)
  if(empty_sampling_position EQUAL -1)
    message(FATAL_ERROR "Explicit empty sampling plane metadata was not written before SHC rejection")
  endif()
  return()
endif()

if(NOT gpumd_result EQUAL 0)
  message(FATAL_ERROR "GPUMD fixture ${MODE} failed: ${gpumd_output}")
endif()
execute_process(
  COMMAND "${VERIFY}" "${MODE}"
  WORKING_DIRECTORY "${expected_work_dir}"
  RESULT_VARIABLE verify_result
  OUTPUT_VARIABLE verify_stdout
  ERROR_VARIABLE verify_stderr)
if(NOT verify_result EQUAL 0)
  message(FATAL_ERROR "Interface fixture verifier ${MODE} failed: ${verify_stdout}${verify_stderr}")
endif()
message(STATUS "${verify_stdout}")
