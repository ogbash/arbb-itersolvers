#include "sparse.hh"
#include "laplace.hh"
#include "cg.hh"
#include "util.hh"

#include <arbb.hpp>

#include <iostream>

using namespace arbb;

// MAIN without ArBB

int main(int argn, char **argv)
{
  struct timespec start, end;
  double lapse;

  int N=40;
  if (argn>1) {
    int d;
    std::istringstream s(argv[1]);
    s>>d;
    N = d;
  }
  
  std::cout<<"Grid size: "<<N<<std::endl;
  int n=(N-1)*(N-1);    
  double *b = new double[n];
  
  Matrix_orig A;
  laplace_orig(N,A);
  for (int i=0; i<A.n; i++)
    b[i] = 0.0;
  b[0] = 2.0;

  double *x = new double[n];
  cg_info cgi;
  std::cout<<"Starting CG"<<std::endl;
  clock_gettime(CLOCK_REALTIME, &start);
  cg_orig(A, b, x, cgi);
  clock_gettime(CLOCK_REALTIME, &end);
  std::cout<<"End CG"<<std::endl;

  lapse = timespec_lapse(&start, &end);
  std::cout<<"lapse="<<lapse<<std::endl;

  double *Ax_ = new double[A.n];
  Ax_orig(A,x,Ax_);
  double *bmAx = new double[A.n];
  for (int i=0; i<A.n; i++)
    bmAx[i] = b[i]-Ax_[i];
  double r = dot_orig(bmAx, bmAx, A.n);
  std::cout<<"r = "<<sqrt(r)<<std::endl;
  std::cout<<"n_iter = "<<value(cgi.n_iter)<<std::endl;

  return 0;
}
