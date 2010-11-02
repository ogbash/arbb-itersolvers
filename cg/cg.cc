#include "cg.hh"
#include "sgs.hh"

using namespace arbb;

void pcg(const Matrix &A, const dense<f64> &b, dense<f64> &v, cg_info &cgi)
{
  dense<f64> x = fill(0.0, b.size());
  dense<f64> ax, p, q, z;
  call(Ax)(A,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev, alpha, beta;
  i32 it = 0;

  _while (it<10000 && sqrt(add_reduce(r*r))>1E-10) {
    sgs(A, r, z);
    
    rho = add_reduce(r*z);
    
    _if (it==0) {
      p = z;
    } _else {
      beta = rho/rho_prev;
      p = z + beta*p;
    } _end_if;
    
    call(Ax)(A,p,q);
    alpha = rho/add_reduce(p*q);

    x += alpha*p;
    r -= alpha*q;

    rho_prev = rho;

    it += 1;
  } _end_while;

  v = x;
  cgi.n_iter = it;
}

void cg(const Matrix &A, const dense<f64> &b, dense<f64> &v, cg_info &cgi)
{
  dense<f64> x = fill(0.0, b.size());
  dense<f64> ax, p, q;
  call(Ax)(A,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev, alpha, beta;
  i32 it = 0;

  _while (it<10000 && sqrt(add_reduce(r*r))>1E-10) {
    rho = add_reduce(r*r);
    
    _if (it==0) {
      p = r;
    } _else {
      beta = rho/rho_prev;
      p = r + beta*p;
    } _end_if;
    
    call(Ax)(A,p,q);
    alpha = rho/add_reduce(p*q);

    x += alpha*p;
    r -= alpha*q;

    rho_prev = rho;

    it += 1;
  } _end_while;

  v = x;
  cgi.n_iter = it;
}

void cg_sliced_cond(const i32 &it, const dense<f64> &r, boolean &cond)
{
  cond = it<10000 && sqrt(add_reduce(r*r))>1E-10;
}

void cg_sliced_iter(const i32 &it, const Matrix &A,
		    dense<f64> &x, dense<f64> &r, dense<f64> &p,
		    f64 &rho, f64 &rho_prev)
{
  dense<f64> q;
  f64 beta, alpha;

  rho = add_reduce(r*r);

  _if (it==0) {
    p = r;
  } _else {
    beta = rho/rho_prev;
    p = r + beta*p;
  } _end_if;
    
  call(Ax)(A,p,q);
  alpha = rho/add_reduce(p*q);

  x += alpha*p;
  r -= alpha*q;
}

void cg_sliced(const Matrix &A, const dense<f64> &b, dense<f64> &v, cg_info &cgi)
{
  dense<f64> x = fill(0.0, b.size());
  dense<f64> ax, p;
  call(Ax)(A,x,ax);
  dense<f64> r = b - ax;
  f64 rho, rho_prev;
  i32 it = 0;
  boolean cond;
  
  cg_sliced_cond(it, r, cond);
  while (value(cond)) {
    call(cg_sliced_iter)(it, A, x, r, p, rho, rho_prev);
    
    rho_prev = rho;

    it += 1;
    call(cg_sliced_cond)(it, r, cond);
  }

  v = x;
  cgi.n_iter = it;
}
