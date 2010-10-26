#include <iostream>
#include <arbb.hpp>

using namespace arbb;

void my_function(arbb::f32& result, arbb::f32 a, arbb::f32 b)
{
  std::cout<<"called"<<std::endl;
  result = a + b;
}

void my_mul(arbb::dense<f32> values, arbb::f32& result)
{
  result = arbb::add_reduce(values);
}

int main()
{
  std::cout<<"Hello"<<std::endl;

  float f[] = {1.3, 2.3, 1.1, 4, 67, 5, 4.3, 4.5, 4.3, 3.3, 5.4};
  arbb::f32 x = 1.0f;
  arbb::f32 y = 2.0f;
  arbb::f32 z, w, s;
  arbb::dense<arbb::f32> c;

  // Compile and execute the function using Intel(R) Array Building Blocks.
  arbb::call(my_function)(z, x, y);
  arbb::call(my_function)(w, z, y);
  arbb::bind(c,f,sizeof(f));
  arbb::call(my_mul)(c,s);

  // Print out the value of z after the function execution.
  std::cout << "z = " << value(z) << std::endl;
  std::cout << "w = " << value(w) << std::endl;
  s.read_only_range();
  std::cout << "s = " << value(s) << std::endl;

  return 0;
}
