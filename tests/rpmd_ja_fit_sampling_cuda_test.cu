#include <cuda_runtime.h>
#include "../src/measure/rpmd_ja_fit_sampling.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void check(cudaError_t error)
{
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

static bool oracle_mic(const RpmdJAFitCell& box, double& x, double& y, double& z)
{
  double s[3] = {
    box.inverse[0] * x + box.inverse[1] * y + box.inverse[2] * z,
    box.inverse[3] * x + box.inverse[4] * y + box.inverse[5] * z,
    box.inverse[6] * x + box.inverse[7] * y + box.inverse[8] * z};
  for (double v : s) if (!std::isfinite(v)) return false;
  for (double& v : s) v -= std::nearbyint(v);
  const double a = box.cell[0] * s[0] + box.cell[1] * s[1] + box.cell[2] * s[2];
  const double b = box.cell[3] * s[0] + box.cell[4] * s[1] + box.cell[5] * s[2];
  const double c = box.cell[6] * s[0] + box.cell[7] * s[1] + box.cell[8] * s[2];
  x = a; y = b; z = c;
  return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

static std::vector<double> oracle(
  int n, int p, const std::vector<std::vector<double>>& position,
  const std::vector<std::vector<double>>& force, const RpmdJAFitCell& box)
{
  std::vector<double> out(10 * static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    double ring[3], prev[3], center[3] = {}, mic[3] = {}, mean[3] = {};
    double scale = 1.0;
    unsigned int error = 0;
    for (int b = 0; b < p; ++b) {
      double r[3], f[3];
      for (int d = 0; d < 3; ++d) {
        r[d] = position[b][i + d * n];
        f[d] = force[b][i + d * n];
        if (!std::isfinite(r[d])) error |= 1;
        if (!std::isfinite(f[d])) error |= 2;
        scale = std::max(scale, std::abs(r[d]));
        mean[d] += f[d] / p;
      }
      if (b == 0) {
        for (int d = 0; d < 3; ++d) ring[d] = prev[d] = r[d], center[d] = r[d] / p;
      } else {
        double delta[3] = {r[0] - prev[0], r[1] - prev[1], r[2] - prev[2]};
        if (!oracle_mic(box, delta[0], delta[1], delta[2])) error |= 4;
        for (int d = 0; d < 3; ++d) prev[d] += delta[d], center[d] += prev[d] / p;
      }
      double delta[3] = {r[0] - ring[0], r[1] - ring[1], r[2] - ring[2]};
      if (!oracle_mic(box, delta[0], delta[1], delta[2])) error |= 8;
      for (int d = 0; d < 3; ++d) mic[d] += delta[d] / p;
    }
    double close[3] = {ring[0] - prev[0], ring[1] - prev[1], ring[2] - prev[2]};
    if (!oracle_mic(box, close[0], close[1], close[2])) error |= 16;
    double winding2 = 0.0, branch2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      const double w = prev[d] - ring[d] + close[d];
      const double q = center[d] - (ring[d] + mic[d]);
      winding2 += w * w; branch2 += q * q;
      out[i + d * n] = center[d];
      out[3 * n + i + d * n] = mean[d];
    }
    out[6 * n + i] = scale;
    out[7 * n + i] = std::sqrt(winding2);
    out[8 * n + i] = std::sqrt(branch2);
    out[9 * n + i] = error;
  }
  return out;
}

static std::vector<double> run_case(int n, int p, RpmdJAFitCell box,
                     std::vector<std::vector<double>> position,
                     std::vector<std::vector<double>> force,
                     bool check_pointer_refresh = false,
                     unsigned int expected_error = 0)
{
  std::vector<double*> dp(p), df(p);
  std::vector<const double*> hp(p), hf(p);
  for (int b = 0; b < p; ++b) {
    check(cudaMalloc(reinterpret_cast<void**>(&dp[b]), position[b].size() * sizeof(double)));
    check(cudaMalloc(reinterpret_cast<void**>(&df[b]), force[b].size() * sizeof(double)));
    check(cudaMemcpy(dp[b], position[b].data(), position[b].size() * sizeof(double), cudaMemcpyHostToDevice));
    check(cudaMemcpy(df[b], force[b].data(), force[b].size() * sizeof(double), cudaMemcpyHostToDevice));
    hp[b] = dp[b]; hf[b] = df[b];
  }
  const double** dpp; const double** dfp;
  double* dout;
  check(cudaMalloc(reinterpret_cast<void**>(&dpp), p * sizeof(double*))); check(cudaMalloc(reinterpret_cast<void**>(&dfp), p * sizeof(double*)));
  check(cudaMalloc(reinterpret_cast<void**>(&dout), 10 * static_cast<std::size_t>(n) * sizeof(double)));
  check(cudaMemcpy(dpp, hp.data(), p * sizeof(double*), cudaMemcpyHostToDevice));
  check(cudaMemcpy(dfp, hf.data(), p * sizeof(double*), cudaMemcpyHostToDevice));

  std::vector<double> actual;
  for (int pass = 0; pass < (check_pointer_refresh ? 2 : 1); ++pass) {
    if (pass == 1) {
      for (int b = 0; b < p; ++b) {
        std::fill(force[b].begin(), force[b].end(), 100.0 + b);
        check(cudaMemcpy(df[b], force[b].data(), force[b].size() * sizeof(double), cudaMemcpyHostToDevice));
      }
      double* replacement;
      check(cudaMalloc(reinterpret_cast<void**>(&replacement), position[0].size() * sizeof(double)));
      for (double& value : position[0]) value += 0.01;
      check(cudaMemcpy(replacement, position[0].data(), position[0].size() * sizeof(double), cudaMemcpyHostToDevice));
      check(cudaFree(dp[0])); dp[0] = replacement; hp[0] = replacement;
      check(cudaMemcpy(dpp, hp.data(), p * sizeof(double*), cudaMemcpyHostToDevice));
    }
    auto expected = oracle(n, p, position, force, box);
    rpmd_ja_fit_sampling_kernel<<<(n + 127) / 128, 128>>>(n, p, dpp, dfp, box, dout);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    actual.resize(expected.size());
    check(cudaMemcpy(actual.data(), dout, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i) {
      assert((std::isnan(expected[i]) && std::isnan(actual[i])) ||
             std::abs(actual[i] - expected[i]) <= 2e-12 * std::max(1.0, std::abs(expected[i])));
    }
    if (expected_error) assert((static_cast<unsigned int>(actual[9 * n]) & expected_error) == expected_error);
  }
  check(cudaFree(dout)); check(cudaFree(dpp)); check(cudaFree(dfp));
  for (int b = 0; b < p; ++b) { check(cudaFree(dp[b])); check(cudaFree(df[b])); }
  return actual;
}

int main()
{
  RpmdJAFitCell orth = {{10,0,0, 0,8,0, 0,0,6}, {0.1,0,0, 0,0.125,0, 0,0,1.0/6}};
  RpmdJAFitCell skew = {{4,0.7,-0.3, 0,3.5,0.4, 0.2,0,5}, {0.2490535963339311, -0.04981071926678622, 0.018928073321378764, 0.001138530726097971, 0.28548657956906615, -0.022770614521959415, -0.009962143853357244, 0.0019924287706714486, 0.19924287706714489}};
  {
    std::vector<std::vector<double>> r(1, {0.2, -0.4, 1.1});
    std::vector<std::vector<double>> f(1, {1, 2, 3});
    run_case(1, 1, orth, r, f);
  }
  {
    std::vector<std::vector<double>> r(6, std::vector<double>(3, 0.0)), f(6, std::vector<double>(3, 1.0));
    for (int b = 0; b < 6; ++b) r[b][0] = 4.0 * b;
    const auto expected = oracle(1, 6, r, f, orth);
    assert(expected[7] > 1.0 && expected[8] > 1.0);
    const auto actual = run_case(1, 6, orth, r, f);
    assert(actual[7] > 128.0 * std::numeric_limits<double>::epsilon() * 6 * actual[6]);
    assert(actual[8] > 128.0 * std::numeric_limits<double>::epsilon() * 6 * actual[6]);
  }
  {
    std::vector<std::vector<double>> r(5, std::vector<double>(3, 0.0)), f(5, std::vector<double>(3, 1.0));
    const double x[] = {0, 4, 6, 4, 0};
    for (int b = 0; b < 5; ++b) r[b][0] = x[b];
    const auto expected = oracle(1, 5, r, f, orth);
    assert(expected[7] < 1e-12 && std::abs(expected[8] - 2.0) < 1e-12);
    const auto actual = run_case(1, 5, orth, r, f);
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * 5 * actual[6];
    assert(actual[7] <= tolerance && actual[8] > tolerance);
  }
  {
    const int n = 2, p = 32;
    for (const RpmdJAFitCell box : {orth, skew}) {
      std::vector<std::vector<double>> r(p, std::vector<double>(3 * n)), f(p, std::vector<double>(3 * n));
      for (int b = 0; b < p; ++b) for (int i = 0; i < 3 * n; ++i) {
        const int image_x = b % 5 - 2, image_y = (2 * b) % 5 - 2, image_z = (3 * b) % 5 - 2;
        const double shift[3] = {
          box.cell[0] * image_x + box.cell[1] * image_y + box.cell[2] * image_z,
          box.cell[3] * image_x + box.cell[4] * image_y + box.cell[5] * image_z,
          box.cell[6] * image_x + box.cell[7] * image_y + box.cell[8] * image_z};
        r[b][i] = 0.2 * i + 0.05 * std::sin(6.283185307179586 * b / p + i) + shift[i / n];
        f[b][i] = 0.2 * b - i;
      }
      const auto actual = run_case(n, p, box, r, f, box.cell[0] == skew.cell[0]);
      double box_scale = 1.0, coordinate_scale = 1.0;
      for (double h : box.cell) box_scale = std::max(box_scale, std::abs(h));
      for (int i = 0; i < n; ++i) coordinate_scale = std::max(coordinate_scale, actual[6 * n + i]);
      const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() * p *
        std::max(box_scale, coordinate_scale);
      for (int i = 0; i < n; ++i) {
        assert(static_cast<unsigned int>(actual[9 * n + i]) == 0);
        assert(actual[7 * n + i] <= tolerance);
        assert(actual[8 * n + i] <= tolerance);
      }
    }
  }
  {
    std::vector<std::vector<double>> r(3, std::vector<double>(3, 0.0)), f(3, std::vector<double>(3, 1.0));
    f[1][2] = std::numeric_limits<double>::quiet_NaN();
    run_case(1, 3, orth, r, f, false, 2);
  }
  {
    std::vector<std::vector<double>> r(3, std::vector<double>(3, 2.0e13)), f(3, std::vector<double>(3, 1.0));
    const auto expected = oracle(1, 3, r, f, orth);
    assert(expected[6] > 1.0e12 * 10.0);
    const auto actual = run_case(1, 3, orth, r, f);
    assert(actual[6] > 1.0e12 * 10.0);
  }
  return 0;
}
