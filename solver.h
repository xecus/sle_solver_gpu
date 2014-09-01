#ifndef _SLESOLVERGPU
#define _SLESOLVERGPU
    #include "sle_solver_gpu.h"
#endif

class solver{
private:
	int Dim;
    int NonZero;
	double *A;
	int *rowA;
	int *colA;
	double *x;
public:
	solver(){
	}
	~solver(){
	}
	solver(const solver &obj){
	}
    void CallSetA(int _Dim,double *_A,int *_rowA,int *_colA);
    void CallSetX(double *_x);
    void CallCG(double *_rhs);
    void CallGCR(double *_rhs);
    void CallBiCGSTAB(double *_rhs);
    
};

void solver::CallSetA(int _Dim,double *_A,int *_rowA,int *_colA){
    this->Dim = _Dim;
    this->NonZero = _rowA[_Dim];
    this->A = _A;
    this->rowA = _rowA;
    this->colA = _colA;
    return;
}
void solver::CallSetX(double *_x){
    this->x = _x;
    return;
}
void solver::CallCG(double *_rhs){
    CG(this->Dim,this->Dim,this->NonZero,this->rowA,this->colA,this->A,this->x,_rhs);
}
void solver::CallGCR(double *_rhs){
    GCR(this->Dim,this->Dim,this->NonZero,this->rowA,this->colA,this->A,this->x,_rhs);
}
void solver::CallBiCGSTAB(double *_rhs){
    BiCGSTAB(this->Dim,this->Dim,this->NonZero,this->rowA,this->colA,this->A,this->x,_rhs);
}