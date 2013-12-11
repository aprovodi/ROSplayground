cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

rosbuild_check_for_sse()

find_package(OpenCV REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

# SSE flags
rosbuild_check_for_sse()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -DEIGEN_NO_DEBUG -fpermissive -Wno-invalid-offsetof")

set( sources	
        src/kfusionCPU.cpp 
        src/tsdfVolume.cpp 
        src/CircularBuffer.cpp
        src/kinectfusion_app.cpp
) 

set( headers 
        include/kfusionCPU/internal.hpp 
        include/kfusionCPU/kfusionCPU.h 
        include/kfusionCPU/OptimizedSelfAdjointMatrix6x6f.h 
        include/kfusionCPU/tsdfVolume.h
        include/kfusionCPU/CircularBuffer.h
) 

#common commands for building c++ executables and libraries
rosbuild_add_library(kfusionCPU src/kfusionCPU.cpp src/tsdfVolume.cpp src/CircularBuffer.cpp)
rosbuild_add_executable(${PROJECT_NAME}_app src/kfusionCPU.cpp src/kinectfusion_app.cpp src/tsdfVolume.cpp src/CircularBuffer.cpp)
#rosbuild_add_boost_directories(${PROJECT_NAME} thread)
rosbuild_add_openmp_flags(${PROJECT_NAME})
