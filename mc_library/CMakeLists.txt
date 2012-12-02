cmake_minimum_required(VERSION 2.8)
# suppress warning from FindCUDA
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

FIND_PACKAGE(CUDA REQUIRED)

message("-- CUDA_TOOLKIT_ROOT_DIR is '${CUDA_TOOLKIT_ROOT_DIR}'")
message("-- CUDA_SDK_ROOT_DIR is '${CUDA_SDK_ROOT_DIR}'")
message("-- CUDA_LIB_DIR is '${CUDA_LIB_DIR}'")
message("-- CUDA_INCLUDE_DIRS is '${CUDA_INCLUDE_DIRS}'")
message("-- CUDA_LIBRARIES is '${CUDA_LIBRARIES}'")

set(CUDA_ARCHITECTURE "sm_20" CACHE STRING "Cuda architecture")
SET(CUDA_NVCC_FLAGS "-arch=${CUDA_ARCHITECTURE}")
set(CUDA_SDK_ROOT_DIR_ $ENV{CUDA_SDK_ROOT_DIR} CACHE STRING "CUDA SDK root folder")

set(CMAKE_FIND_LIBRARY_SUFFIXES ".a;.so")

cuda_add_library(marching_cubes SHARED marching_cubes.cu)
								
target_link_libraries(marching_cubes cuda)
