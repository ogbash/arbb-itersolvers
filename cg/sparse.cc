#include "sparse.hh"

using namespace arbb;

void Ax(const Matrix &A, const dense<f64> &x, dense<f64> &y)
{
  dense<f64> colvals = gather(x, A.cols);
  dense<f64> mvals = colvals * A.vals;
  nested<f64> nmvals = reshape_nested_offsets(mvals, A.nrows);
  y = add_reduce(nmvals);
}

void print_vector(const dense<f64> &x)
{
  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;
}
