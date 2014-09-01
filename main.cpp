#include <iostream>
#include <cstdlib>
#include "solver.h"

void genTridiag(int *I, int *J, double *val, int N, int nz){
	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (double)rand()/RAND_MAX + 10.0f;
	val[1] = (double)rand()/RAND_MAX;
	int start;
	for (int i = 1; i < N; i++){
		if (i > 1){
			I[i] = I[i-1]+3;
		}else{
			I[1] = 2;
		}
		start = (i-1)*3 + 2;
		J[start] = i - 1;
		J[start+1] = i;
		if (i < N-1){
			J[start+2] = i + 1;
		}
		val[start] = val[start-1];
		val[start+1] = (double)rand()/RAND_MAX + 10.0f;
		if (i < N-1){
			val[start+2] = (double)rand()/RAND_MAX;
		}
	}
	I[N] = nz;
	return;
}

int main(int argc, char **argv){
	int i;
	int M;
	int N;
	int nz;
	int *I;
	int *J;
	double *val;
	double *x;
	double *rhs;
	//Make Sparse Matrix
	M = N = 1048576;
	nz = (N-2)*3 + 4;
	I = new int[N+1];
	J = new int[nz];
	val = new double[nz];
	x = new double[N];
	rhs = new double[N];
	genTridiag(I, J, val, N, nz);
	for (int i = 0; i < N; i++) rhs[i] = 0.1;

	solver _sc;

	_sc.CallSetA( M , val , I , J );
	_sc.CallSetX( x );

	//CG
	for(i=0;i<N;i++) x[i] = 0.0;
	//CG(M,N,nz,I,J,val,x,rhs);
	_sc.CallCG( rhs );
	std::cout << "[CG]" << std::endl;
	for(i=0;i<5;i++) std::cout << x[i] << std::endl;

	//BICG-STAB
	for(i=0;i<N;i++) x[i] = 0.0;
	//BiCGSTAB(M,N,nz,I,J,val,x,rhs);
	_sc.CallBiCGSTAB( rhs );
	std::cout << "[BiCGSTAB]" << std::endl;
	for(i=0;i<5;i++) std::cout << x[i] << std::endl;

	//GCR Method
	for(i=0;i<N;i++) x[i] = 0.0;
	//GCR(M,N,nz,I,J,val,x,rhs);
	_sc.CallGCR( rhs );
	std::cout << "[GCR]" << std::endl;
	for(i=0;i<5;i++) std::cout << x[i] << std::endl;

	delete[] I;
	delete[] J;
	delete[] val;
	delete[] x;
	delete[] rhs;
	return 0;
}