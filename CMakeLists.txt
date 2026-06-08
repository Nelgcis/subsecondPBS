cmake_minimum_required(VERSION 3.10)
project(RayTraceDicomIntegration)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures.
# CUDA 13.x no longer supports older targets such as compute_60, so keep the
# default narrow and allow caller override via -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
enable_language(CUDA)

# Keep the upstream nuclear-correction macro family available locally so the
# halo path can be compiled in a controlled way.
add_compile_definitions(SOUKUP=0)
add_compile_definitions(FLUKA=1)
add_compile_definitions(GAUSS_FIT=2)

set(NUCLEAR_CORR "GAUSS_FIT" CACHE STRING "Build halo / nuclear correction mode (OFF, SOUKUP, FLUKA, GAUSS_FIT)")
set_property(CACHE NUCLEAR_CORR PROPERTY STRINGS "OFF" "SOUKUP" "FLUKA" "GAUSS_FIT")
if(NUCLEAR_CORR STREQUAL "SOUKUP")
    add_compile_definitions(NUCLEAR_CORR=SOUKUP)
elseif(NUCLEAR_CORR STREQUAL "FLUKA")
    add_compile_definitions(NUCLEAR_CORR=FLUKA)
elseif(NUCLEAR_CORR STREQUAL "GAUSS_FIT")
    add_compile_definitions(NUCLEAR_CORR=GAUSS_FIT)
elseif(NOT NUCLEAR_CORR STREQUAL "OFF")
    message(FATAL_ERROR "Unsupported NUCLEAR_CORR='${NUCLEAR_CORR}'. Expected OFF, SOUKUP, FLUKA, or GAUSS_FIT.")
endif()

# Set CUDA properties (directory scope default)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY CUDA_ARCHITECTURES 86)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/core)
include_directories(${CMAKE_SOURCE_DIR}/include/algorithms)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)

# Core source files
set(CORE_SOURCES
    src/core/raytracedicom_wrapper.cu
    src/core/bev_ray_tracing.cu
)

# Algorithm source files
set(ALGORITHM_SOURCES
    src/algorithms/convolution.cu
    src/algorithms/superposition.cu
    src/algorithms/complete_superposition.cu
    src/algorithms/idd_sigma.cu
    src/algorithms/cuda_final_dose.cu
    src/algorithms/prim_transf_kernel.cu
    src/algorithms/transfer_param_struct_div3_impl.cu
    src/algorithms/transfer_param_helper.cu
)

# Utility source files
set(UTILITY_SOURCES
    src/utils/utils.cu
    src/utils/texture_ultra_optimized.cu
    src/utils/advanced_memory_pool.cu
    src/utils/precompiled_texture_manager.cu
    src/utils/energy_reader.cpp
    src/utils/backward_compatibility.cu
)

# Test source files
set(TEST_SOURCES
    src/tests/complete_dose_debug.cu
)

# All source files
set(ALL_SOURCES
    ${CORE_SOURCES}
    ${ALGORITHM_SOURCES}
    ${UTILITY_SOURCES}
)

# Create main library
add_library(raytracedicom_lib ${ALL_SOURCES})
target_include_directories(raytracedicom_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(raytracedicom_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------
# Optional: pybind11 Python bindings
# -----------------------------
option(RTD_BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" ON)
if(RTD_BUILD_PYTHON_BINDINGS)
    set(RTD_PYBIND11_DIR "${CMAKE_SOURCE_DIR}/owl-master/pyOWL/pybind11")
    if(EXISTS "${RTD_PYBIND11_DIR}/CMakeLists.txt")
        add_subdirectory("${RTD_PYBIND11_DIR}" "${CMAKE_BINARY_DIR}/pybind11")
        pybind11_add_module(raytracedicom_py src/bindings/raytracedicom_pybind.cpp)
        target_link_libraries(raytracedicom_py PRIVATE raytracedicom_lib)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
            target_link_libraries(raytracedicom_py PRIVATE stdc++fs)
        endif()
        target_include_directories(raytracedicom_py PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        set_target_properties(raytracedicom_py PROPERTIES CUDA_ARCHITECTURES 86)
        set_target_properties(raytracedicom_py PROPERTIES
            OUTPUT_NAME "cudaCalDoseRTD"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
        )
    else()
        message(WARNING "pybind11 not found at ${RTD_PYBIND11_DIR}; set RTD_BUILD_PYTHON_BINDINGS=OFF or provide pybind11")
    endif()
endif()


# Set CUDA properties for the library
set_property(TARGET raytracedicom_lib PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_property(TARGET raytracedicom_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link CUDA libraries
target_link_libraries(raytracedicom_lib ${CUDA_LIBRARIES})

# Create test executable
add_executable(raytracedicom_test ${TEST_SOURCES})
target_link_libraries(raytracedicom_test raytracedicom_lib)

# Create CPB debug executable
add_executable(cpb_debug_test src/tests/cpb_convolution_debug.cu)
target_link_libraries(cpb_debug_test raytracedicom_lib)

# Create texture coordinate test executable
add_executable(texture_coordinate_test src/tests/texture_coordinate_test.cu)
target_link_libraries(texture_coordinate_test raytracedicom_lib)

add_executable(simple_texture_debug src/tests/simple_texture_debug.cu)
target_link_libraries(simple_texture_debug raytracedicom_lib)

# Add CPB to Ray Tracing Integration Test
add_executable(cpb_ray_tracing_integration src/tests/cpb_ray_tracing_integration.cu)
target_link_libraries(cpb_ray_tracing_integration raytracedicom_lib)

# Wrapper integration test
add_executable(wrapper_integration_test src/tests/wrapper_integration_test.cu)
target_link_libraries(wrapper_integration_test raytracedicom_lib)

# GCC 8 and older require an explicit std::filesystem link library.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(wrapper_integration_test stdc++fs)
endif()

# CarbonPBS LUT layout example (documents expected tensor shapes)
add_executable(carbonpbs_data_layout_example src/tests/carbonpbs_data_layout_example.cu)
target_link_libraries(carbonpbs_data_layout_example raytracedicom_lib)

# Complete dose debug test
add_executable(complete_dose_debug src/tests/complete_dose_debug.cu)
target_link_libraries(complete_dose_debug raytracedicom_lib)

# Simple ray mapping test
add_executable(simple_ray_mapping_test src/tests/simple_ray_mapping_test.cu)
target_link_libraries(simple_ray_mapping_test raytracedicom_lib)

# Dose distribution analysis test
add_executable(dose_distribution_analysis src/tests/dose_distribution_analysis.cu)
target_link_libraries(dose_distribution_analysis raytracedicom_lib)

# Set output directories and CUDA architectures
set_target_properties(raytracedicom_lib raytracedicom_test cpb_debug_test texture_coordinate_test simple_texture_debug cpb_ray_tracing_integration wrapper_integration_test carbonpbs_data_layout_example complete_dose_debug simple_ray_mapping_test dose_distribution_analysis
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    CUDA_ARCHITECTURES 86
)

# Compiler-specific options
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(raytracedicom_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -use_fast_math
        -Xcompiler -Wall
        >
    )
endif()

# Print configuration summary
message(STATUS "RayTraceDicom Integration Configuration:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Toolkit: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  Nuclear correction mode: ${NUCLEAR_CORR}")
message(STATUS "  Core Sources: ${CORE_SOURCES}")
message(STATUS "  Algorithm Sources: ${ALGORITHM_SOURCES}")
message(STATUS "  Utility Sources: ${UTILITY_SOURCES}")
message(STATUS "  Test Sources: ${TEST_SOURCES}")
