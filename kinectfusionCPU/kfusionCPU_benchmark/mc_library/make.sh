#nvcc -c -DGT520 -arch sm_20 marching_cubes.cu -o marching_cubes.o -lcuda -lcudart -Xcompiler -fPIC
nvcc -c -arch sm_11 marching_cubes.cu -o marching_cubes.o -lcuda -lcudart -Xcompiler -fPIC

g++ -m64 -O3 -shared -use_fast_math -L/usr/stud/provodin/ros_workspace/cvpr-ros-pkg/cvpr-cuda/cuda/cuda/lib64 -o libmarching_cubes.so marching_cubes.o -lcuda -lcudart

rm marching_cubes.o
