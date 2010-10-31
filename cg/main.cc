#include <arbb.hpp>

#include <iostream>

using namespace arbb;

struct Matrix
{
  dense<i32> nrows;
  dense<i32> cols;
  dense<f64> vals;
};

void laplace(int N, Matrix &M)
{
  // N - number of segments
  // n - number of unknowns
  int n=(N-1)*(N-1);
  int nnz = (5*n-4*(N-1));
  int *nrows = new int[n];
  int *cols = new int[nnz];
  double *vals = new double[nnz];

  int k=0;
  for(int gi=0; gi<N-1; gi++) {
    for(int gj=0; gj<N-1; gj++) {
      int row_start = k;
      int i = gi*(N-1)+gj;
      nrows[i] = k;
      //std::cout<<k<<" "<<nnz<<std::endl;
      if (gi>0) {
	cols[k] = i-(N-1);
	vals[k] = -1.0;
	k++;
      }
      if (gj>0) {
	cols[k] = i-1;
	vals[k] = -1.0;
	k++;
      }
      cols[k] = i;
      vals[k] = 4.0;
      k++;
      if (gj<N-2) {
	cols[k] = i+1;
	vals[k] = -1.0;
	k++;
      }
      if (gi<N-2) {
	cols[k] = i+(N-1);
	vals[k] = -1.0;
	k++;
      }
    }
  }

  bind(M.nrows, nrows, n);
  bind(M.cols, cols, nnz);
  bind(M.vals, vals, nnz);
}

void Ax(const Matrix &A, const dense<f64> &x, dense<f64> &y)
{
  dense<f64> colvals = gather(x, A.cols);
  dense<f64> mvals = colvals * A.vals;
  nested<f64> nmvals = reshape_nested_offsets(mvals, A.nrows);
  y = add_reduce(nmvals);
}

void cg(const Matrix &A, const dense<f64> &b, dense<f64> &v)
{
  dense<f64> x(b.size());
  dense<f64> ax, p, q;
  call(Ax)(A,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev, alpha, beta;
  i32 it = 0;

  _while (it<1000 && add_reduce(r*r)>1E-10) {
    
    rho = add_reduce(r*r);
    
    _if (it==0) {
      p = r;
    } _else {
      beta = rho/rho_prev;
      p = r + beta*p;
    } _end_if;
    
    call(Ax)(A,p,q);
    alpha = rho/add_reduce(p*q);

    x += alpha*p;
    r -= alpha*q;

    rho_prev = rho;

    it += 1;
  } _end_while;

  v = x;
}

//int nrows[3] = {0,2,5};
//int cols[7] = {0,1,0,1,2,1,2};
//double vals[7] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

int main()
{
  int N=4;
  int n=(N-1)*(N-1);
  dense<f64> b(n);
  
  Matrix A;
  laplace(N,A);
  b[0] = 2.0;

  dense<f64> x(n);
  call(cg)(A, b, x);

  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;

  return 0;
}
