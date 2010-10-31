#include <arbb.hpp>

#include <iostream>

using namespace arbb;

struct Matrix
{
  dense<i32> nrows;
  dense<i32> cols;
  dense<f64> vals;
};

void laplace(int N)
{

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

int nrows[3] = {0,2,5};
int cols[7] = {0,1,0,1,2,1,2};
double vals[7] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

int main()
{
  dense<f64> b(3);
  
  Matrix A;
  bind(A.nrows, nrows, 3);
  bind(A.cols, cols, 7);
  bind(A.vals, vals, 7);
  b[0] = 0.2;

  dense<f64> x(3);
  call(cg)(A, b, x);

  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;

  return 0;
}
