#include "laplace.hh"

#include <arbb.hpp>

using namespace arbb;

void laplace(int N, Matrix &M)
{
  // N - number of segments
  // n - number of unknowns
  int n=(N-1)*(N-1);
  int nnz = (5*n-4*(N-1));
  int *nrows = new int[n];
  int *cols = new int[nnz];
  double *vals = new double[nnz];

  int k=0;
  for(int gi=0; gi<N-1; gi++) {
    for(int gj=0; gj<N-1; gj++) {
      int row_start = k;
      int i = gi*(N-1)+gj;
      nrows[i] = k;
      //std::cout<<k<<" "<<nnz<<std::endl;
      if (gi>0) {
	cols[k] = i-(N-1);
	vals[k] = -1.0;
	k++;
      }
      if (gj>0) {
	cols[k] = i-1;
	vals[k] = -1.0;
	k++;
      }
      cols[k] = i;
      vals[k] = 4.0;
      k++;
      if (gj<N-2) {
	cols[k] = i+1;
	vals[k] = -1.0;
	k++;
      }
      if (gi<N-2) {
	cols[k] = i+(N-1);
	vals[k] = -1.0;
	k++;
      }
    }
  }

  bind(M.nrows, nrows, n);
  bind(M.cols, cols, nnz);
  bind(M.vals, vals, nnz);
}
