#include "sparse.hh"
#include "laplace.hh"
#include "cg.hh"
#include "util.hh"

#include <arbb.hpp>

#include <iostream>

using namespace arbb;

// MAIN Ax without ArBB

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
  b[0] = 2.0;

  double *x = new double[n];
  cg_info cgi;
  std::cout<<"Starting Ax"<<std::endl;
  clock_gettime(CLOCK_REALTIME, &start);
  for (int i=0; i<1000; i++)
    Ax_orig(A, b, x);
  clock_gettime(CLOCK_REALTIME, &end);
  std::cout<<"End Ax"<<std::endl;

  lapse = timespec_lapse(&start, &end);
  std::cout<<"lapse="<<lapse<<std::endl;

  return 0;
}
