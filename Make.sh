#!/bin/sh
g++ -m32  -I/usr/local/cuda-5.0/include -I ../nvcc_test/inc -o main.o -c sle_solver_gpu.cpp
g++ -m32 -o sle_solver_gpu main.o -L/usr/local/cuda-5.0/lib -lcudart -lcublas -lcusparse
