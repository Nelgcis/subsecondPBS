
cmake_minimum_required(VERSION 3.15)
project(carbonPBS VERSION 0.0.1 LANGUAGES C CXX CUDA)
find_package(OpenMP)
if(OpenMP_FOUND)
    message(STATUS "OpenMP includes: ${OPENMP_INCLUDES}")
    message(STATUS "OpenMP libraries: ${OpenMP_CXX_LIBRARIES}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS} --extended-lambda")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_NVCC_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#set(temp2 -Xcompiler -O3 -lineinfo -DNDEBUG)
if(NOT MSVC)
    set(temp2 -Xcompiler -O3 -lineinfo -DNDEBUG)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE)
endif()

#set(temp2 -g -G -DDEBUG)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${temp2}>")

include_directories(${CMAKE_SOURCE_DIR} ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc)

link_directories(${CMAKE_BINARY_DIR}/primitives)

add_library(cudaCalDose1 MODULE cudaCalDose.cpp deviceCalDose.cu)
set_property(TARGET cudaCalDose1 PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(cudaCalDose1 PROPERTIES
        CUDA_ARCHITECTURES "80;89"
        CUDA_SEPARABLE_COMPILATION ON
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_include_directories(cudaCalDose1 PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/cuda/samples/common/inc ${PROJECT_SOURCE_DIR})
target_link_libraries(cudaCalDose1 PRIVATE pybind11::module)
target_link_libraries(cudaCalDose1 PRIVATE owl::owl)
target_link_libraries(cudaCalDose1 PUBLIC OpenMP::OpenMP_CXX)
if(MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_options(cudaCalDose1 PRIVATE /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrt.lib)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_link_options(cudaCalDose1 PRIVATE /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib)
    endif()
endif() 

add_library(cudaCalDose1_ cudaCalDose.cpp deviceCalDose.cu)
set_property(TARGET cudaCalDose1_ PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(cudaCalDose1_ PROPERTIES
        CUDA_ARCHITECTURES "80;89"
        INTERPROCEDURAL_OPTIMIZATION TRUE
)

target_include_directories(cudaCalDose1_ PUBLIC ${CUDA_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/cuda/samples/common/inc ${PROJECT_SOURCE_DIR})
target_link_libraries(cudaCalDose1_ PRIVATE pybind11::module)
target_link_libraries(cudaCalDose1_ PRIVATE owl::owl)
target_link_libraries(cudaCalDose1_ PUBLIC OpenMP::OpenMP_CXX)
if(MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_options(cudaCalDose1_ PRIVATE /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrt.lib)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_link_options(cudaCalDose1_ PRIVATE /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib)
    endif()
endif() 

# 安装cudaCalDose1动态库
install(TARGETS cudaCalDose1
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
if(WIN32)
    # 获取依赖并安装到指定位置
    install(CODE "set(LIB_DIRECTORIES \"${Python3_RUNTIME_LIBRARY_DIRS}\" \"${CUDAToolkit_BIN_DIR}\")")
    install(CODE [[
        file(GET_RUNTIME_DEPENDENCIES
            RESOLVED_DEPENDENCIES_VAR RESOLVED_DEPS
            UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPS
            LIBRARIES $<TARGET_FILE:cudaCalDose1>
            DIRECTORIES ${LIB_DIRECTORIES}
            POST_EXCLUDE_REGEXES "system32"
        )
        foreach(UNRESOLVED ${UNRESOLVED_DEPS})
            message(WARNING ${UNRESOLVED})
        endforeach()
        file(INSTALL ${RESOLVED_DEPS} DESTINATION ${CMAKE_INSTALL_PREFIX})
    ]]
    )
endif()