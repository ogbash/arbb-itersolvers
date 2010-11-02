#ifndef SPARSE_HH
#define SPARSE_HH

#include <arbb.hpp>

struct Matrix
{
  arbb::dense<arbb::i32> nrows;
  arbb::dense<arbb::i32> cols;
  arbb::dense<arbb::f64> vals;
};

void Ax(const Matrix &A,
	const arbb::dense<arbb::f64> &x,
	arbb::dense<arbb::f64> &y);

void print_vector(const arbb::dense<arbb::f64> &x);

#endif
