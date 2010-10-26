#include <arbb.hpp>

#include <iostream>

using namespace arbb;

void Ax(const dense<i32> &nrows_c, const dense<i32> &cols_c, const dense<f64> &vals_c,
	const dense<f64> &x_c, dense<f64> &y_c)
{
  dense<f64> colvals_c = gather(x_c, cols_c);
  dense<f64> mvals_c = colvals_c * vals_c;
  nested<f64> mvals_nc = reshape_nested_offsets(mvals_c, nrows_c);
  y_c = add_reduce(mvals_nc);
}

void cg(const dense<i32> &nrows, const dense<i32> &cols, const dense<f64> &vals,
	const dense<f64> &b, dense<f64> &v)
{
  dense<f64> x(b.size());
  dense<f64> ax, p, q;
  call(Ax)(nrows,cols,vals,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev, alpha, beta;
  i32 it = 0;

  _while (it<1000 && add_reduce(r*r)>1E-10) {
    
    rho = add_reduce(r*r);
    
    _if (it==0) {
      p = r;
    } _else {
      beta = rho/rho_prev;
      p = r + beta*p;
    } _end_if;
    
    call(Ax)(nrows,cols,vals,p,q);
    alpha = rho/add_reduce(p*q);

    x += alpha*p;
    r -= alpha*q;

    rho_prev = rho;

    it += 1;
  } _end_while;

  v = x;
}

int nrows[3] = {0,2,5};
int cols[7] = {0,1,0,1,2,1,2};
double vals[7] = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};

int main()
{
  dense<i32> nrows_c;
  dense<i32> cols_c;
  dense<f64> vals_c;
  dense<f64> b(3);
  
  bind(nrows_c, nrows, 3);
  bind(cols_c, cols, 7);
  bind(vals_c, vals, 7);
  b[0] = 0.2;

  dense<f64> x(3);
  call(cg)(nrows_c, cols_c, vals_c, b, x);

  const_range<f64> r = x.read_only_range();
  for (const_range_iterator<f64> i = r.begin(); i!=r.end(); i++)
    std::cout<<value(*i)<<std::endl;

  return 0;
}
