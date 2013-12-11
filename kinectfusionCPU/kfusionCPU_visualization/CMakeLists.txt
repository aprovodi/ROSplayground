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

find_package(OpenCV REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(Eigen REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEIGEN_NO_DEBUG -fpermissive -Wno-invalid-offsetof")

include_directories(/usr/include/vtk-5.8)
include_directories(mc_library)

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread boost_filesystem boost_system)
rosbuild_add_executable(${PROJECT_NAME} src/kfusion_vis.cpp src/pcl_trajectory_visualizer.cpp src/SceneCloudView.cpp)
target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/mc_library/libmarching_cubes.so /usr/lib/libboost_system.so.1.46.1 /usr/lib/libvtkCommon.so.5.8 /usr/lib/libvtkFiltering.so.5.8 /usr/lib/libvtkRendering.so.5.8 /usr/lib/libvtkGraphics.so.5.8 /usr/lib/libboost_signals.so.1.46.1)
