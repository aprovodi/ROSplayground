nvcc -c -DGT520 -arch sm_20 marching_cubes.cu -o marching_cubes.o -lcuda -lcudart -Xcompiler -fPIC

g++ -m64 -O3 -shared -use_fast_math -o libmarching_cubes.so marching_cubes.o -lcudart -lcuda

rm marching_cubes.o
