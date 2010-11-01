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

struct cg_info {
  i32 n_iter;
};

void cg(const Matrix &A, const dense<f64> &b, dense<f64> &v, cg_info &cgi)
{
  dense<f64> x(b.size());
  dense<f64> ax, p, q;
  call(Ax)(A,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev, alpha, beta;
  i32 it = 0;

  _while (it<10000 && sqrt(add_reduce(r*r))>1E-10) {
    
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
  cgi.n_iter = it;
}

void print_vector(dense<f64> &x)
{
  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;
}

// MAIN

int main(int argn, char **argv)
{
  int N=40;
  if (argn>1) {
    int d;
    std::istringstream s(argv[1]);
    s>>d;
    N = d;
  }
  
  std::cout<<"Grid size: "<<N<<std::endl;
  int n=(N-1)*(N-1);
  dense<f64> b(n);
  
  Matrix A;
  laplace(N,A);
  b[0] = 2.0;

  dense<f64> x(n);
  cg_info cgi;
  std::cout<<"Starting CG"<<std::endl;
  call(cg)(A, b, x, cgi);
  std::cout<<"End CG"<<std::endl;

  dense<f64> Ax_;
  Ax(A,x,Ax_);
  dense<f64> bmAx = b-Ax_;
  f64 r = add_reduce(bmAx*bmAx);
  std::cout<<"r = "<<value(sqrt(r))<<std::endl;
  std::cout<<"n_iter = "<<value(cgi.n_iter)<<std::endl;

  return 0;
}
