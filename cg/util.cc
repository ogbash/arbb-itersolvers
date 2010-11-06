#include "util.hh"

double timespec_lapse (timespec *from, timespec *to)
{
  double result = 0.0;

  result += (to->tv_sec - from->tv_sec);
  result += (to->tv_nsec - from->tv_nsec)/1E9;
     
  return result;
}
