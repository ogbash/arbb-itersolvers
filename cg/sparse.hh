#ifndef SPARSE_HH
#define SPARSE_HH

#include <arbb.hpp>

typedef arbb::i64 ind_t;
typedef arbb::uncaptured<ind_t>::type ind_ct;

struct Matrix
{
  arbb::dense<ind_t> nrows;
  arbb::dense<ind_t> cols;
  arbb::dense<arbb::f64> vals;
};

struct Matrix_orig
{
  int n, nnz;
  ind_ct *nrows;
  ind_ct *cols;
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
