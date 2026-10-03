#pragma once

inline void rpmd_ja_integrate_hac(
  const int number_of_lags,
  const int number_of_components,
  const double factor,
  const double* hac,
  double* rtc)
{
  for (int component = 0; component < number_of_components; ++component) {
    const int offset = number_of_lags * component;
    for (int lag = 1; lag < number_of_lags; ++lag) {
      const int index = offset + lag;
      rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index]) * factor;
    }
  }
}
