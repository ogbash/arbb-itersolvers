#ifndef SGS_HH
#define SGS_HH

#include "sparse.hh"

#include <arbb.hpp>

void sgs(const Matrix &A,
	 const arbb::dense<arbb::f64> &r,
	 arbb::dense<arbb::f64> &z);

#endif
