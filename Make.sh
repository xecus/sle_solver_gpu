#!/bin/sh
g++ -m32  -I/usr/local/cuda-5.0/include -I ../nvcc_test/inc -o sle_solver_gpu.o -c sle_solver_gpu.cpp
g++ -m32 -o main.o -c main.cpp
g++ -m32 -o sle_solver_gpu sle_solver_gpu.o hoge.o -L/usr/local/cuda-5.0/lib -lcudart -lcublas -lcusparse
