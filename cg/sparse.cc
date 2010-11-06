#include "sparse.hh"

using namespace arbb;

void Ax(const Matrix &A, const dense<f64> &x, dense<f64> &y)
{
  dense<f64> colvals = gather(x, A.cols);
  dense<f64> mvals = colvals * A.vals;
  nested<f64> nmvals = reshape_nested_offsets(mvals, A.nrows);
  y = add_reduce(nmvals);
}

void Ax_orig(const Matrix_orig &A,
	     const double *x,
	     double *y)
{
  for (int i=0; i<A.n-1; i++) {
    y[i] = 0.0;
    for (int k=A.nrows[i]; k<A.nrows[i+1]; k++) {
      y[i] += x[A.cols[k]]*A.vals[k];
    }
  }
  y[A.n-1] = 0.0;
  for (int k=A.nrows[A.n-1]; k<A.nnz; k++) {
    y[A.n-1] += x[A.cols[k]]*A.vals[k];
  }
}

double dot_orig(const double *x, const double *y, int N)
{
  double s = 0.0;
  for (int i=0; i<N; i++) {
    s += x[i]*y[i];
  }
  return s;
}

void print_vector(const dense<f64> &x)
{
  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;
}
