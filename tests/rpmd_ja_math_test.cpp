#include "../src/measure/rpmd_ja_math.cuh"
#include <cassert>
#include <cmath>

int main()
{
  const double hac[] = {1.0, 3.0, 5.0, 10.0, 6.0, 2.0};
  double rtc[6] = {0.0};
  rpmd_ja_integrate_hac(3, 2, 0.5, hac, rtc);
  assert(rtc[0] == 0.0);
  assert(rtc[1] == 2.0);
  assert(rtc[2] == 6.0);
  assert(rtc[3] == 0.0);
  assert(rtc[4] == 8.0);
  assert(rtc[5] == 12.0);
  assert(std::isfinite(rtc[5]));
}
