#include "sparse.hh"
#include "laplace.hh"
#include "cg.hh"

#include <arbb.hpp>

#include <iostream>

using namespace arbb;

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
