#ifndef SPARSE_HH
#define SPARSE_HH

#include <arbb.hpp>

struct Matrix
{
  arbb::dense<arbb::i32> nrows;
  arbb::dense<arbb::i32> cols;
  arbb::dense<arbb::f64> vals;
};

struct Matrix_orig
{
  int n, nnz;
  int *nrows;
  int *cols;
  double *vals;
};

void Ax(const Matrix &A,
	const arbb::dense<arbb::f64> &x,
	arbb::dense<arbb::f64> &y);

void Ax_orig(const Matrix_orig &A,
	     const double *x,
	     double *y);

double dot_orig(const double *x, const double *y, int N);

void print_vector(const arbb::dense<arbb::f64> &x);

#endif
