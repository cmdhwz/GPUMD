#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace
{
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

Vector apply(const Matrix& matrix, const Vector& x)
{
  Vector result(matrix.size(), 0.0);
  for (std::size_t row = 0; row < matrix.size(); ++row)
    for (std::size_t column = 0; column < x.size(); ++column)
      result[row] += matrix[row][column] * x[column];
  return result;
}

Matrix multiply(const Matrix& a, const Matrix& b)
{
  Matrix result(a.size(), Vector(b[0].size(), 0.0));
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t k = 0; k < b.size(); ++k)
      for (std::size_t j = 0; j < b[0].size(); ++j) result[i][j] += a[i][k] * b[k][j];
  return result;
}

double dot(const Vector& left, const Vector& right)
{
  double result = 0.0;
  for (std::size_t i = 0; i < left.size(); ++i) result += left[i] * right[i];
  return result;
}

Vector add_scaled(const Vector& a, const Vector& b, const double scale)
{
  Vector result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) result[i] = a[i] + scale * b[i];
  return result;
}

std::vector<Vector> chebyshev(const Matrix& z, const Vector& seed, const int degree)
{
  std::vector<Vector> result{seed};
  if (degree == 0) return result;
  result.push_back(apply(z, seed));
  for (int m = 2; m <= degree; ++m) {
    const Vector z_current = apply(z, result[m - 1]);
    Vector next(seed.size());
    for (std::size_t i = 0; i < seed.size(); ++i)
      next[i] = 2.0 * z_current[i] - result[m - 2][i];
    result.push_back(next);
  }
  return result;
}

Vector modal(const std::vector<Vector>& terms, const Vector& coefficients)
{
  Vector result(terms[0].size(), 0.0);
  for (std::size_t m = 0; m < terms.size(); ++m)
    for (std::size_t i = 0; i < result.size(); ++i) result[i] += coefficients[m] * terms[m][i];
  return result;
}

std::vector<Vector> modal_ranks(const std::vector<Vector>& terms, const Matrix& coefficients)
{
  std::vector<Vector> result;
  for (const Vector& rank : coefficients) result.push_back(modal(terms, rank));
  return result;
}

double close(const double a, const double b)
{
  return std::fabs(a - b) < 1.0e-11 * std::max({1.0, std::fabs(a), std::fabs(b)});
}
} // namespace

int main()
{
  const Matrix D{{0.7, -0.2, 0.0, 0.0}, {-0.2, 1.1, 0.1, 0.0},
                 {0.0, 0.1, 0.9, -0.15}, {0.0, 0.0, -0.15, 0.8}};
  const std::array<Matrix, 3> BT{{
    Matrix{{0.2, 1.3, -0.4, 0.1}, {-0.7, 0.6, 0.2, 0.0},
           {0.8, -0.1, 0.5, -0.3}, {0.4, 0.0, 1.2, -0.6}},
    Matrix{{-0.3, 0.9, 0.5, 0.2}, {0.1, -0.8, 0.3, 0.4},
           {1.1, 0.2, -0.4, 0.6}, {-0.5, 0.3, 0.7, 0.1}},
    Matrix{{0.4, -0.6, 0.8, 0.0}, {0.2, 0.5, -0.9, 0.3},
           {-0.1, 0.7, 0.2, -0.5}, {0.6, -0.4, 0.1, 0.9}}
  }};
  const Vector y{0.3, -0.8, 1.2, 0.4};
  const Vector u{-0.6, 0.2, 0.7, -1.1};
  const Vector p_lambda{0.8, -0.35};
  const Vector q_lambda{-0.4, 0.9};
  const Matrix p_coeff{{0.6, -0.2, 0.07, 0.015}, {-0.1, 0.5, 0.11, -0.02}};
  const Matrix q_coeff{{-0.1, 0.5, 0.11, -0.02}, {0.6, -0.2, 0.07, 0.015}};
  Matrix z = D;
  for (std::size_t i = 0; i < z.size(); ++i) z[i][i] -= 1.0; // Z = 2D/Lambda - I, Lambda=2.

  const Vector dy = apply(D, y);
  const Vector du = apply(D, u);
  const auto dy_terms = chebyshev(z, dy, 3);
  const auto du_terms = chebyshev(z, du, 3);
  const auto u_terms = chebyshev(z, u, 3);
  // Independent polynomial identities catch recurrence mistakes in this CPU oracle.
  const Matrix z2 = multiply(z, z);
  const Matrix z3 = multiply(z2, z);
  const Vector z2y = apply(z2, y);
  Vector t2(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) t2[i] = 2.0 * z2y[i] - y[i];
  const Vector zy = apply(z, y);
  const Vector z3y = apply(z3, y);
  Vector t3(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) t3[i] = 4.0 * z3y[i] - 3.0 * zy[i];
  const auto check_terms = chebyshev(z, y, 3);
  for (std::size_t i = 0; i < y.size(); ++i) {
    assert(close(check_terms[2][i], t2[i]));
    assert(close(check_terms[3][i], t3[i]));
  }
  const auto lp = modal_ranks(dy_terms, p_coeff);
  const auto rp = modal_ranks(du_terms, p_coeff);
  const auto lq = modal_ranks(dy_terms, q_coeff);
  const auto rq = modal_ranks(u_terms, q_coeff);

  for (const Matrix& bt : BT) {
    bool nonzero_direction = false;
    for (const Vector& row : bt)
      for (double value : row) nonzero_direction = nonzero_direction || value != 0.0;
    double p = 0.0, q = 0.0, wrong_q = 0.0;
    for (std::size_t rank = 0; rank < p_lambda.size(); ++rank)
      p += 1.7 * 1.7 * p_lambda[rank] * dot(apply(bt, lp[rank]), rp[rank]);
    for (std::size_t rank = 0; rank < q_lambda.size(); ++rank) {
      q += 1.7 * q_lambda[rank] * dot(lq[rank], apply(bt, rq[rank]));
      wrong_q += 1.7 * q_lambda[rank] * dot(rq[rank], apply(bt, rq[rank]));
    }
    double p_expanded = 0.0, q_expanded = 0.0;
    for (std::size_t rank = 0; rank < p_lambda.size(); ++rank)
      for (std::size_t left_degree = 0; left_degree < p_coeff[rank].size(); ++left_degree)
        for (std::size_t right_degree = 0; right_degree < p_coeff[rank].size(); ++right_degree)
          for (std::size_t i = 0; i < y.size(); ++i)
            for (std::size_t j = 0; j < y.size(); ++j)
              p_expanded += 1.7 * 1.7 * p_lambda[rank] * p_coeff[rank][left_degree] * p_coeff[rank][right_degree] *
                dy_terms[left_degree][j] * bt[i][j] * du_terms[right_degree][i];
    for (std::size_t rank = 0; rank < q_lambda.size(); ++rank)
      for (std::size_t left_degree = 0; left_degree < q_coeff[rank].size(); ++left_degree)
        for (std::size_t right_degree = 0; right_degree < q_coeff[rank].size(); ++right_degree)
          for (std::size_t i = 0; i < y.size(); ++i)
            for (std::size_t j = 0; j < y.size(); ++j)
              q_expanded += 1.7 * q_lambda[rank] * q_coeff[rank][left_degree] * q_coeff[rank][right_degree] *
                dy_terms[left_degree][i] * bt[i][j] * u_terms[right_degree][j];
    assert(close(p, p_expanded));
    assert(close(q, q_expanded));
    if (nonzero_direction) assert(!close(q, wrong_q));
  }
}
