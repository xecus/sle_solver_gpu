/*

	Hiroyuki Ootaguro
	2014/08/31

*/
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "sle_solver_gpu.h"

/* for BiCG-STAB */
double vector_norm(double a[],int size){
	double sum = 0.0;
	int i;
	for(i=0;i<size;i++) sum += fabs( a[i] );
	return sum;
}

/* for Debug */
void ShowVector(double *DeviceMemory,int size,char *fn){
	double *temp = new double[size];
	cudaMemcpy(temp, DeviceMemory, size*sizeof(double), cudaMemcpyDeviceToHost);
	printf("\t[%s]\n",fn);
	printf("\tvector_norm=%lf\n",vector_norm(temp,size));
	int i;
	for(i=0;i<5;i++) printf("\t[%d]=%lf\n",i,temp[i]);
	delete[] temp;
	return;
}


void CG(int M,int N,int nz,int *I,int *J,double *val,double *x,double *rhs){
	/* Var */
	clock_t start,end;
	const double tol = 1e-15f;
	double a, b, na, r0, r1;
	int *d_col, *d_row;
	double *d_val, *d_x, dot;
	double *d_r, *d_p, *d_Ax;
	int k;
	double alpha, beta, alpham1;
	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	/* Find Best Cuda Device */
	/*
	int devID = findCudaDevice(1, (const char **)"./CG");
	if (devID < 0){
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	*/
	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	if (checkCudaErrors(cublasStatus))exit(EXIT_FAILURE);
	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);
	if (checkCudaErrors(cusparseStatus))exit(EXIT_FAILURE);
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);
	if (checkCudaErrors(cusparseStatus))exit(EXIT_FAILURE);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(double)));
	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;
	//CSR Matrix Calc
	//d_Ax = alpha * op(CSR) * d_x + beta * d_AX
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
   N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
	//d_r =  alpham1 * d_Ax + d_r (Vector)
	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	//r1 = d_r * d_r;
	cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
	k = 1;
	float elapsed_time_ms = 0.0f;
	cudaEvent_t custart, custop;
	cudaEventCreate( &custart );
	cudaEventCreate( &custop );
	cudaEventRecord( custart, 0);
	while (r1 > tol*tol){

		if (k > 1){
			b = r1 / r0;
			// d_p = b * d_p;
			cublasStatus = cublasDscal(cublasHandle, N, &b, d_p, 1);
			// d_p = alpha * d_r + d_p
			cublasStatus = cublasDaxpy(cublasHandle, N, &alpha, d_r, 1, d_p, 1);
		}else{
			// d_p = d_r
			cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
		}
		// d_AX = alpha * op(CSR) * d_p + beta * d_Ax
		// d_AX = alpha * op(CSR) * d_p
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
   N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ax);
		// dot = d_p ^T * d_Ax
		cublasStatus = cublasDdot(cublasHandle, N, d_p, 1, d_Ax, 1, &dot);
		a = r1 / dot;
		//d_x = a * d_p + d_x;
		cublasStatus = cublasDaxpy(cublasHandle, N, &a, d_p, 1, d_x, 1);
		na = -a;
		//d_r = na * d_Ax + d_r;
		cublasStatus = cublasDaxpy(cublasHandle, N, &na, d_Ax, 1, d_r, 1);
		r0 = r1;
		//r1 = d_r^T * d_r;
		cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
		cudaThreadSynchronize();
		//printf("iteration = %3d, residual = %e\n", k, sqrt(r1));
		k++;
	}
	cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord( custop, 0);
	cudaEventSynchronize( custop );
	cudaEventElapsedTime( &elapsed_time_ms, custart, custop );
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);
	//cudaDeviceReset();
	return;
}
int GCR(int M,int N,int nz,int *I,int *J,double *val,double *x,double *rhs){
	/* Var */
	clock_t start,end;
	double a, b, na, r0, r1;
	int *d_col, *d_row;
	double *d_val, *d_x;
	double *d_r;
	double *d_p;
	double *d_q;
	double *d_s;
	double *d_Ax;
	double alpha, beta, alpham1;
	int MaxIter = 10;
	double BetaStack[ MaxIter ];
	// This will pick the best possible CUDA capable device
	cudaDeviceProp deviceProp;
	/* Find Best Cuda Device */
	/*
	int devID = findCudaDevice(1, (const char **)"./GCR");
	if (devID < 0){
		printf("exiting...\n");
		exit(EXIT_SUCCESS);
	}
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	if (checkCudaErrors(cublasStatus))exit(EXIT_FAILURE);
	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);
	if (checkCudaErrors(cusparseStatus))exit(EXIT_FAILURE);
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);
	if (checkCudaErrors(cusparseStatus))exit(EXIT_FAILURE);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*MaxIter*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_q, N*MaxIter*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_s, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(double)));
	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
	alpha = 1.0;
	alpham1 = -1.0;
	beta = 0.0;
	r0 = 0.;
	//d_Ax = A(CRS) * x
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
   N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
	// d_r = -d_Ax + d_r
	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
	//d_p = dr
	cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
	//d_q = A(CRS) * d_r
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
   N, N, nz, &alpha, descr, d_val, d_row, d_col, d_r, &beta, d_q);
	//d_s = d_q
	cublasStatus = cublasDcopy(cublasHandle, N, d_q, 1, d_s, 1);
	float elapsed_time_ms = 0.0f;
	cudaEvent_t custart, custop;
	cudaEventCreate( &custart );
	cudaEventCreate( &custop );
	cudaEventRecord( custart, 0);
	int k = 0;
	while(1){

                //convergence check
		/*
		double r1;
                cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r, 1, &r1);
                if (r1 < 1e-40) break;
                std::cout << "r1=" << r1 << std::endl;
		*/
		double r1;
		cublasStatus = cublasDnrm2(cublasHandle,N,d_r,1,&r1);
		//std::cout << "r1=" << r1 << std::endl;
		if( r1 < 1e-15 ) break;

		double temp1,temp2,dot_1,dot_1m;
		//temp1 = (d_q,d_r)
		cublasStatus = cublasDdot(cublasHandle, N, d_q+N*k, 1, d_r, 1, &temp1);
		if (checkCudaErrors(cublasStatus))exit(EXIT_FAILURE);
		//temp2 = (d_q,d_q)
		cublasStatus = cublasDdot(cublasHandle, N, d_q+N*k, 1, d_q+N*k, 1, &temp2);
		if (checkCudaErrors(cublasStatus))exit(EXIT_FAILURE);
		dot_1 = temp1 / temp2;
		dot_1m = -1.0 * dot_1;
		//d_x = dot_1 * d_p + d_x;
		cublasStatus = cublasDaxpy(cublasHandle, N, &dot_1, d_p+N*k, 1, d_x, 1);
		//d_r = dot_1m * d_q + d_r;
		cublasStatus = cublasDaxpy(cublasHandle, N, &dot_1m, d_q+N*k, 1, d_r, 1);
		//d_s = A(CRS) * d_r
		cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, nz, &alpha, descr, d_val, d_row, d_col, d_r, &beta, d_s);
		for(int i=0;i<k;i++){
			cublasStatus = cublasDdot(cublasHandle, N, d_q + N*i, 1, d_s, 1, &temp1);
			cublasStatus = cublasDdot(cublasHandle, N, d_q + N*i, 1, d_q + N*i, 1, &temp2);
			BetaStack[i] = -1.0 * temp1 / temp2;
		}
		//d_p = d_r
		cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p+N*(k+1), 1);
		for(int i=0;i<=k;i++){
			// d_p = BetaStack[i] * d_p;
			cublasStatus = cublasDaxpy(cublasHandle, N, &BetaStack[i], d_p+N*i, 1, d_p+N*(k+1), 1);
		}
		//d_q = d_s
		cublasStatus = cublasDcopy(cublasHandle, N, d_s, 1, d_q+N*(k+1), 1);
		for(int i=0;i<=k;i++){
			// d_p[k+1] = BetaStack[i] * d_p[i];
			cublasStatus = cublasDaxpy(cublasHandle, N, &BetaStack[i], d_q+N*i, 1, d_q+N*(k+1), 1);
		}

		//Incr
		k++;
		if(k==MaxIter){
			k=0;
			//d_Ax = A(CRS) * x
			cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
			// d_r = B (Host Memory)
			cudaMemcpy(d_r, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
			// d_r = -d_Ax + d_r
			cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r, 1);
			//d_p = dr
			cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
			//d_q = A(CRS) * d_r
			cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			N, N, nz, &alpha, descr, d_val, d_row, d_col, d_r, &beta, d_q);
			//d_s = d_q
			cublasStatus = cublasDcopy(cublasHandle, N, d_q, 1, d_s, 1);
		}

	}
	cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
	cudaEventRecord( custop, 0);
	cudaEventSynchronize( custop );
	cudaEventElapsedTime( &elapsed_time_ms, custart, custop );
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_q);
	cudaFree(d_s);
	cudaFree(d_Ax);
	//cudaDeviceReset();
	return 0;
}

int BiCGSTAB(int M,int N,int nz,int *I,int *J,double *val,double *x,double *rhs){
	/* cudaDeviceProp */
	cudaDeviceProp deviceProp;
	/*
	int devID = findCudaDevice(1, (const char **)"./BICGSTAB");
	if (devID < 0){
	printf("exiting...\n");
	exit(EXIT_SUCCESS);
	}
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
	*/
	/*
	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
	   deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
	int version = (deviceProp.major * 0x10 + deviceProp.minor);
	if (version < 0x11){
	printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
	}
	*/
	/* Get handle to the CUBLAS context */
	cublasHandle_t cublasHandle = 0;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);
	if (checkCudaErrors(cublasStatus)){
	exit(EXIT_FAILURE);
	}
	/* Get handle to the CUSPARSE context */
	cusparseHandle_t cusparseHandle = 0;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);
	if (checkCudaErrors(cusparseStatus)){
	exit(EXIT_FAILURE);
	}
	cusparseMatDescr_t descr = 0;
	cusparseStatus = cusparseCreateMatDescr(&descr);
	if (checkCudaErrors(cusparseStatus)){
	exit(EXIT_FAILURE);
	}
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
	/* Var */
	double *r;
	int *d_col;
	int  *d_row;
	double *d_val;
	double *d_x;
	double *d_rhs;
	double *d_r0;
	double *d_r;
	double *d_p;
	double *d_Ax;
	double *d_Ap;
	double *d_e;
	double *d_Ae;
	double *d_p2;
	r = (double *)malloc(sizeof(double)*N);
	checkCudaErrors(cudaMalloc((void **)&d_col, nz*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_row, (N+1)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&d_val, nz*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_x, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_rhs, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r0, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_r, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ax, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ap, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_e, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_Ae, N*sizeof(double)));
	checkCudaErrors(cudaMalloc((void **)&d_p2, N*sizeof(double)));
	cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r0, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
	double alpha = 1.0;
	double alpham1 = -1.0;
	double beta = 0.0;
	//Ax = A * x
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
	N, N, nz, &alpha, descr, d_val, d_row, d_col, d_x, &beta, d_Ax);
	//r0 = b - Ax;
	cublasDaxpy(cublasHandle, N, &alpham1, d_Ax, 1, d_r0, 1);
	//p = r = r0
	cublasStatus = cublasDcopy(cublasHandle, N, d_r0, 1, d_p, 1);   //d_p = d_r0
	cublasStatus = cublasDcopy(cublasHandle, N, d_r0, 1, d_r, 1);   //d_r = d_r0
	double c1;
	cublasStatus = cublasDdot(cublasHandle, N, d_r0, 1, d_r0, 1, &c1);
	int k;
	for(k=0;k<1000;k++){
	//d_Ap = A * p
	//printf("d_Ap = A * p\n");
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
	N, N, nz, &alpha, descr, d_val, d_row, d_col, d_p, &beta, d_Ap);
	//c2 = r0^T * ap
	//printf("c2 = r0^T * ap\n");
	double c2;
	cublasStatus = cublasDdot(cublasHandle, N, d_r0, 1, d_Ap, 1, &c2);
	// alpha = c1 / c2
	//printf("Calc Alpha\n");
	double alpha2 = c1 / c2;
	double alpha2m1 = -1.0 * alpha2;
	// e = r - alpha2 * ap;
	//printf("e = r - alpha2 * ap\n");
	cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_e, 1);
	cublasStatus = cublasDaxpy(cublasHandle, N, &alpha2m1, d_Ap, 1, d_e, 1);
	// ae = a * e
	cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
	N, N, nz, &alpha, descr, d_val, d_row, d_col, d_e, &beta, d_Ae);
	// e_dot_ae = e^T * ae;
	//printf("e_dot_ae = e^T * ae\n");
	double e_dot_ae;
	cublasStatus = cublasDdot(cublasHandle, N, d_e, 1, d_Ae, 1, &e_dot_ae);
	// ae_dot_ae = ae^T * ae;
	//printf("ae_dot_ae = ae^T * ae\n");
	double ae_dot_ae;
	cublasStatus = cublasDdot(cublasHandle, N, d_Ae, 1, d_Ae, 1, &ae_dot_ae);
	//c3 = e_dot_ae / ae_dot_ae
	//printf("c3 = e_dot_ae / ae_dot_ae\n");
	double c3 = e_dot_ae / ae_dot_ae;
	double c3m1 = -1.0 * c3;
	//x+= alpha2 * p + c3 * e
	//printf("x+= alpha2 * p + c3 * e\n");
	cublasStatus = cublasDaxpy(cublasHandle, N, &alpha2, d_p, 1, d_x, 1);
	cublasStatus = cublasDaxpy(cublasHandle, N, &c3, d_e, 1, d_x, 1);
	//Check d_x
	//ShowVector(d_x,N,"d_x");
	double x_norm;
	cublasStatus = cublasDdot(cublasHandle, N, d_x, 1, d_x, 1, &x_norm);
	//printf("x_norm = %lf \n",x_norm);
	// r = e - c3 * ae;
	//printf("r = e - c3 * ae\n");
	cublasStatus = cublasDcopy(cublasHandle, N, d_e, 1, d_r, 1);
	cublasStatus = cublasDaxpy(cublasHandle, N, &c3m1, d_Ae, 1, d_r, 1);
	//err
	cudaMemcpy(r, d_r, N*sizeof(double), cudaMemcpyDeviceToHost);
	//printf("vector_norm(r,N)=%lf\n",vector_norm(r,N));
	//printf("vector_norm(rhs,N)=%lf\n",vector_norm(rhs,N));
	double err = vector_norm(r,N) / vector_norm(rhs,N);
	//printf("[%d]err=%lf\n",k,err);
	//ShowVector(d_x,N,"d_x");
	if( 1e-15 > err ){
	//printf("\tOK!\n");
	break;
	}
	//c1 = r^T * r0;
	//printf("c1 = r^T * r0\n");
	cublasStatus = cublasDdot(cublasHandle, N, d_r, 1, d_r0, 1, &c1);
	//beta2 = c1 / (c2 * c3)
	//printf("beta2 = c1 / (c2 * c3)");
	double beta2 = c1 / (c2 * c3);
	// [ p = r + beta2 * ( p - c3 * ap ) ]
	// p2 = p - c3 * ap
	// p = r + beta2 * p2
	//printf("p2 = p - c3 * ap\n");
	cublasStatus = cublasDcopy(cublasHandle, N, d_p, 1, d_p2, 1);
	cublasStatus = cublasDaxpy(cublasHandle, N, &c3m1, d_Ap, 1, d_p2, 1);
	//printf("p = r + beta2 * p2\n");
	cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, d_p, 1);
	cublasStatus = cublasDaxpy(cublasHandle, N, &beta2, d_p2, 1, d_p, 1);
	}
	cudaMemcpy(x, d_x, N*sizeof(double), cudaMemcpyDeviceToHost);
	cusparseDestroy(cusparseHandle);
	cublasDestroy(cublasHandle);
	free(r);
	cudaFree(d_col);
	cudaFree(d_row);
	cudaFree(d_val);
	cudaFree(d_x);
	cudaFree(d_rhs);
	cudaFree(d_r0);
	cudaFree(d_r);
	cudaFree(d_p);
	cudaFree(d_Ax);
	cudaFree(d_Ap);
	cudaFree(d_e);
	cudaFree(d_Ae);
	cudaFree(d_p2);
	//cudaDeviceReset();
	return 0;
}