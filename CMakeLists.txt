cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA compute capability (e.g. native, 75, 80, 90)")
project(GPUMD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# MSVC host compiler flags: /utf-8 for Unicode source, /Zc:preprocessor for standards-conforming preprocessor.
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8 -Xcompiler /Zc:preprocessor")
endif()

# --- gpumd ---
file(GLOB GPUMD_SOURCES
    src/main_gpumd/*.cu
    src/minimize/*.cu
    src/phonon/*.cu
    src/integrate/*.cu
    src/mc/*.cu
    src/force/*.cu
    src/measure/*.cu
    src/model/*.cu
    src/utilities/*.cu)
add_executable(gpumd ${GPUMD_SOURCES})
target_include_directories(gpumd PRIVATE src)
target_link_libraries(gpumd PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GPUMD_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT GPUMD_GIT_COMMIT)
    set(GPUMD_GIT_COMMIT "unknown")
endif()
execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GPUMD_GIT_COMMIT_FULL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT GPUMD_GIT_COMMIT_FULL)
    set(GPUMD_GIT_COMMIT_FULL "unknown")
endif()
execute_process(
    COMMAND git describe --always --tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GPUMD_GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT GPUMD_GIT_DESCRIBE)
    set(GPUMD_GIT_DESCRIBE "unknown")
endif()
execute_process(
    COMMAND git status --porcelain --untracked-files=normal
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GPUMD_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(GPUMD_GIT_STATUS)
    set(GPUMD_GIT_DIRTY "yes")
else()
    set(GPUMD_GIT_DIRTY "no")
endif()
target_compile_definitions(gpumd PRIVATE
    GPUMD_GIT_COMMIT="${GPUMD_GIT_COMMIT}"
    GPUMD_GIT_COMMIT_FULL="${GPUMD_GIT_COMMIT_FULL}"
    GPUMD_GIT_DESCRIBE="${GPUMD_GIT_DESCRIBE}"
    GPUMD_GIT_DIRTY="${GPUMD_GIT_DIRTY}")

option(BUILD_TESTING "Build GPUMD tests" OFF)
include(CTest)
if(BUILD_TESTING)
    add_executable(quantum_heat_moments_cpu_test tests/quantum_heat_moments_cpu_test.cpp)
    target_include_directories(quantum_heat_moments_cpu_test PRIVATE src)
    add_test(NAME quantum_heat_moments_cpu_test COMMAND quantum_heat_moments_cpu_test)
endif()

# --- nep ---
file(GLOB NEP_SOURCES
    src/main_nep/*.cu
    src/utilities/*.cu)
add_executable(nep ${NEP_SOURCES})
target_include_directories(nep PRIVATE src)
target_link_libraries(nep PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)

# --- gnep (optional, build with: cmake --build . --target gnep) ---
file(GLOB GNEP_SOURCES
    src/main_gnep/*.cu
    src/utilities/*.cu)
add_executable(gnep EXCLUDE_FROM_ALL ${GNEP_SOURCES})
target_include_directories(gnep PRIVATE src)
target_link_libraries(gnep PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft Threads::Threads)
option(GNEP_TEST_DIAGNOSTICS "Export deterministic GNEP training diagnostics" OFF)
if(GNEP_TEST_DIAGNOSTICS)
    target_compile_definitions(gnep PRIVATE GNEP_TEST_DIAGNOSTICS)
endif()
option(GNEP_TEST_FIXED_SCALER "Keep the GNEP descriptor scaler fixed for gradient tests" OFF)
if(GNEP_TEST_FIXED_SCALER)
    target_compile_definitions(gnep PRIVATE USE_FIXED_SCALER)
endif()


# --- NNAP support (enable with -D PKG_NNAP=yes) ---
option(PKG_NNAP "Enable NNAP support" OFF)
if(PKG_NNAP)
    find_package(JNI REQUIRED)
    target_include_directories(gpumd PRIVATE ${JNI_INCLUDE_DIRS})
    target_link_libraries(gpumd PRIVATE ${JNI_LIBRARIES})
    target_compile_definitions(gpumd PRIVATE USE_NNAP)
    
    # delay load jvm.dll on windows
    if(WIN32 AND MSVC)
        target_link_libraries(gpumd PRIVATE delayimp)
        target_link_options(gpumd PRIVATE "/DELAYLOAD:jvm.dll")
        string(REPLACE "\\" "/" JSE_JVM_LIB_PATH_DEF "$ENV{JVM_LIB_PATH}")
        add_definitions(-DJVM_LIB_PATH="${JSE_JVM_LIB_PATH_DEF}")
    endif()

    string(REPLACE "\\" "/" JSE_JAR_PATH_DEF "$ENV{JSE_JAR_PATH}")
    target_compile_definitions(gpumd PRIVATE JVM_CLASS_PATH="-Djava.class.path=${JSE_JAR_PATH_DEF}")
endif()
