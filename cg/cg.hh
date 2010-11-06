#ifndef CG_HH
#define CG_HH

#include "sparse.hh"

#include <arbb.hpp>

struct cg_info {
  arbb::i32 n_iter;
};

void pcg(const Matrix &A, 
	 const arbb::dense<arbb::f64> &b,
	 arbb::dense<arbb::f64> &v,
	 cg_info &cgi);
void cg(const Matrix &A,
	const arbb::dense<arbb::f64> &b,
	arbb::dense<arbb::f64> &v,
	cg_info &cgi);
void cg_sliced(const Matrix &A,
	const arbb::dense<arbb::f64> &b,
	arbb::dense<arbb::f64> &v,
	cg_info &cgi);
void cg_orig(const Matrix_orig &A,
	const double *b,
	double *v,
	cg_info &cgi);

#endif
