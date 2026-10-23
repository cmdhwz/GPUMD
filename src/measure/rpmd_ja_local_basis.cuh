#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace rpmd_ja_local_basis {

struct Vec3
{
  double x, y, z;
};

struct Site
{
  int atom;
  // Displacements are already on the frozen R0 branch; retain images for assembly and transport.
  int image[3];
};

struct Term
{
  int parameter;
  int atom_count;
  Site site[3];
  Vec3 coefficient[3];
  double weight;
  double energy_share[3];
};

static_assert(std::is_trivially_copyable<Term>::value, "local terms must be directly copyable to CUDA buffers");

struct EdgeInput
{
  Site first, second;
  Vec3 axis;
  int longitudinal_parameter;
  int transverse_parameter;
};

struct BridgeInput
{
  Site oxygen1, hydrogen, oxygen2;
  Vec3 axis;
  // Frozen geometric bridge polarity a0; oxygen exchange reverses it with the axis.
  double a0;
  int s_parameter, ell_parameter, cross_parameter, transverse_parameter;
};

struct Evaluation
{
  double energy = 0.0;
  std::vector<double> site_energy;
  std::vector<double> gradient;
};

struct HessianBlock
{
  Site row, column;
  int parameter;
  double value[9];
};

struct SparseEntry
{
  Site row, column;
  int parameter, row_component, column_component;
  double value;
};

struct OperatorNormalization
{
  int input_parameter_count = 0;
  int base_parameter_count = 0;
  int output_parameter_count = 0;
  std::vector<int> output_source_parameters;
  std::vector<int> redundant_parameters;
  std::vector<double> residual_relative_norm;
  // Row-major input-parameter by output-parameter map: theta_input = transform * theta_output.
  std::vector<double> transform;
  // Row-major input-parameter by redundant-parameter null-combination map.
  std::vector<double> redundant_combination;
};

namespace detail {

inline Vec3 add(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 subtract(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 scale(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline bool finite(Vec3 a) { return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z); }

inline Vec3 unit(Vec3 a)
{
  if(!finite(a)) throw std::runtime_error("non-finite local-basis axis");
  const double norm = std::sqrt(dot(a, a));
  if(!(norm > 0.0) || !std::isfinite(norm)) throw std::runtime_error("zero local-basis axis");
  return scale(a, 1.0 / norm);
}

inline void perpendicular_axes(Vec3 axis, Vec3& first, Vec3& second)
{
  const double absolute[3] = {std::abs(axis.x), std::abs(axis.y), std::abs(axis.z)};
  Vec3 reference = {1.0, 0.0, 0.0};
  if(absolute[1] < absolute[0] && absolute[1] <= absolute[2]) reference = {0.0, 1.0, 0.0};
  else if(absolute[2] < absolute[0] && absolute[2] < absolute[1]) reference = {0.0, 0.0, 1.0};
  first = unit(cross(axis, reference));
  second = unit(cross(axis, first));
}

inline void check_site(const Site& site)
{
  if(site.atom < 0) throw std::runtime_error("negative local-basis atom index");
}

inline Term make_term(const Site* sites, const Vec3* coefficients, int atom_count,
                      int parameter, double weight, const double* shares)
{
  if(parameter < 0 || atom_count < 1 || atom_count > 3 || !std::isfinite(weight))
    throw std::runtime_error("invalid local-basis term metadata");
  Term term{};
  term.parameter = parameter;
  term.atom_count = atom_count;
  term.weight = weight;
  Vec3 sum = {0.0, 0.0, 0.0};
  double coefficient_scale = 1.0;
  double share_sum = 0.0;
  for(int i = 0; i < atom_count; ++i) {
    check_site(sites[i]);
    if(!finite(coefficients[i]) || !std::isfinite(shares[i]) || shares[i] < 0.0)
      throw std::runtime_error("invalid local-basis coefficient or energy share");
    term.site[i] = sites[i];
    term.coefficient[i] = coefficients[i];
    term.energy_share[i] = shares[i];
    sum = add(sum, coefficients[i]);
    coefficient_scale += std::sqrt(dot(coefficients[i], coefficients[i]));
    share_sum += shares[i];
  }
  if(std::sqrt(dot(sum, sum)) > 1.0e-12 * coefficient_scale || std::abs(share_sum - 1.0) > 1.0e-12)
    throw std::runtime_error("local-basis term is not translation invariant or shares do not sum to one");
  return term;
}

inline void check_terms(const std::vector<Term>& terms, const double* theta, int parameter_count,
                        int atom_count, const double* masses)
{
  if(parameter_count < 0 || atom_count < 0 || (parameter_count && !theta))
    throw std::runtime_error("invalid local-basis dimensions");
  if(masses) for(int i = 0; i < atom_count; ++i)
    if(!(masses[i] > 0.0) || !std::isfinite(masses[i])) throw std::runtime_error("invalid local-basis mass");
  for(const Term& term : terms) {
    if(term.parameter < 0 || term.parameter >= parameter_count || term.atom_count < 1 || term.atom_count > 3 ||
       !std::isfinite(term.weight) || !std::isfinite(theta[term.parameter]))
      throw std::runtime_error("invalid local-basis term");
    for(int i = 0; i < term.atom_count; ++i)
      if(term.site[i].atom < 0 || term.site[i].atom >= atom_count || !detail::finite(term.coefficient[i]))
        throw std::runtime_error("local-basis site is outside the atom array");
  }
}

struct OperatorKey
{
  int row_atom, column_atom, row_component, column_component;
  bool operator<(const OperatorKey& other) const
  {
    if(row_atom != other.row_atom) return row_atom < other.row_atom;
    if(column_atom != other.column_atom) return column_atom < other.column_atom;
    if(row_component != other.row_component) return row_component < other.row_component;
    return column_component < other.column_component;
  }
};

using SparseOperator = std::map<OperatorKey, double>;

inline double component(Vec3 value, int axis)
{
  return axis == 0 ? value.x : axis == 1 ? value.y : value.z;
}

inline double sparse_dot(const SparseOperator& a, const SparseOperator& b)
{
  const SparseOperator* smaller = &a;
  const SparseOperator* larger = &b;
  if(smaller->size() > larger->size()) std::swap(smaller, larger);
  double sum = 0.0, correction = 0.0;
  for(const auto& entry : *smaller) {
    const auto found = larger->find(entry.first);
    if(found == larger->end()) continue;
    const double value = entry.second * found->second;
    const double adjusted = value - correction;
    const double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  return sum;
}

inline double coefficient_dot(const std::vector<double>& a, const std::vector<double>& b,
                              const std::vector<double>& gram, int parameter_count)
{
  double sum = 0.0, correction = 0.0;
  for(int i = 0; i < parameter_count; ++i) for(int j = 0; j < parameter_count; ++j) {
    const double value = a[i] * gram[static_cast<std::size_t>(i) * parameter_count + j] * b[j];
    const double adjusted = value - correction;
    const double next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
  }
  return sum;
}

inline double projection(const Term& term, const double* x, int atom_count, const double* masses)
{
  double value = 0.0;
  for(int i = 0; i < term.atom_count; ++i) {
    const int atom = term.site[i].atom;
    const double factor = masses ? 1.0 / std::sqrt(masses[atom]) : 1.0;
    value += term.coefficient[i].x * x[atom] * factor;
    value += term.coefficient[i].y * x[atom_count + atom] * factor;
    value += term.coefficient[i].z * x[2 * atom_count + atom] * factor;
  }
  return value;
}

inline std::vector<double> inverse_sqrt_masses(int atom_count, const double* masses)
{
  std::vector<double> result(static_cast<std::size_t>(atom_count), 1.0);
  if(masses) for(int i = 0; i < atom_count; ++i) result[i] = 1.0 / std::sqrt(masses[i]);
  return result;
}

} // namespace detail

class OperatorGramAccumulator
{
public:
  OperatorGramAccumulator(int parameter_count, int atom_count, const double* masses = nullptr)
    : parameter_count_(parameter_count), atom_count_(atom_count)
  {
    if(parameter_count < 0 || atom_count < 0 ||
       (parameter_count && static_cast<std::size_t>(parameter_count) >
         std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(parameter_count)))
      throw std::runtime_error("invalid local-basis operator Gram dimensions");
    inverse_sqrt_ = detail::inverse_sqrt_masses(atom_count, masses);
    operators_.resize(static_cast<std::size_t>(parameter_count));
    if(masses) for(int i = 0; i < atom_count; ++i)
      if(!(masses[i] > 0.0) || !std::isfinite(masses[i])) throw std::runtime_error("invalid local-basis mass");
  }

  void add(const Term& term)
  {
    if(term.parameter < 0 || term.parameter >= parameter_count_ || term.atom_count < 1 || term.atom_count > 3 ||
       !std::isfinite(term.weight))
      throw std::runtime_error("invalid local-basis operator term");
    for(int i = 0; i < term.atom_count; ++i) {
      detail::check_site(term.site[i]);
      if(term.site[i].atom >= atom_count_ || !detail::finite(term.coefficient[i]))
        throw std::runtime_error("local-basis operator site is outside the atom array");
      for(int j = 0; j < term.atom_count; ++j) for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) {
        const double value = term.weight * inverse_sqrt_[term.site[i].atom] * inverse_sqrt_[term.site[j].atom] *
          detail::component(term.coefficient[i], a) * detail::component(term.coefficient[j], b);
        if(!std::isfinite(value)) throw std::runtime_error("non-finite local-basis operator entry");
        if(value != 0.0) {
          const detail::OperatorKey key = {term.site[i].atom, term.site[j].atom, a, b};
          operators_[term.parameter][key] += value;
        }
      }
    }
  }

  std::vector<double> gram() const
  {
    const std::size_t count = static_cast<std::size_t>(parameter_count_);
    std::vector<double> result(count * count, 0.0);
    for(int i = 0; i < parameter_count_; ++i) for(int j = i; j < parameter_count_; ++j) {
      const double value = detail::sparse_dot(operators_[i], operators_[j]);
      if(!std::isfinite(value)) throw std::runtime_error("non-finite local-basis operator Gram entry");
      result[static_cast<std::size_t>(i) * count + j] = value;
      result[static_cast<std::size_t>(j) * count + i] = value;
    }
    return result;
  }

  int parameter_count() const { return parameter_count_; }

  std::size_t sparse_entry_count() const
  {
    std::size_t count = 0;
    for(const detail::SparseOperator& op : operators_) {
      if(op.size() > std::numeric_limits<std::size_t>::max() - count)
        throw std::runtime_error("local-basis sparse entry count overflows");
      count += op.size();
    }
    return count;
  }

  double linear_combination_norm(const std::vector<double>& coefficients) const
  {
    if(coefficients.size()!=static_cast<std::size_t>(parameter_count_)||
       !std::all_of(coefficients.begin(),coefficients.end(),[](double value){return std::isfinite(value);}))
      throw std::runtime_error("invalid local-basis operator combination");
    struct Sum{double value=0.0,correction=0.0;void add(const double x){
      const double next=value+x;correction+=std::abs(value)>=std::abs(x)?(value-next)+x:(x-next)+value;value=next;}};
    std::map<detail::OperatorKey,Sum> combined;
    for(int parameter=0;parameter<parameter_count_;++parameter)if(coefficients[parameter]!=0.0)
      for(const auto& entry:operators_[parameter]){
        const double value=coefficients[parameter]*entry.second;
        if(!std::isfinite(value))throw std::runtime_error("non-finite local-basis operator combination entry");
        combined[entry.first].add(value);}
    Sum norm2;
    for(const auto& entry:combined){const double value=entry.second.value+entry.second.correction;
      norm2.add(value*value);}
    const double square=norm2.value+norm2.correction;
    if(!(square>=0.0)||!std::isfinite(square))throw std::runtime_error("invalid local-basis operator combination norm");
    return std::sqrt(square);
  }

private:
  int parameter_count_, atom_count_;
  std::vector<double> inverse_sqrt_;
  std::vector<detail::SparseOperator> operators_;
};

inline std::vector<Term> make_edge_terms(const EdgeInput& edge)
{
  detail::check_site(edge.first);
  detail::check_site(edge.second);
  if(edge.first.atom == edge.second.atom) throw std::runtime_error("local-basis edge repeats an atom");
  const Vec3 axis = detail::unit(edge.axis);
  Vec3 transverse[2];
  detail::perpendicular_axes(axis, transverse[0], transverse[1]);
  const Site sites[2] = {edge.first, edge.second};
  const double shares[2] = {0.5, 0.5};
  std::vector<Term> terms;
  terms.reserve(3);
  for(int component = 0; component < 3; ++component) {
    const Vec3 direction = component == 0 ? axis : transverse[component - 1];
    const Vec3 coefficients[2] = {detail::scale(direction, -1.0), direction};
    const int parameter = component == 0 ? edge.longitudinal_parameter : edge.transverse_parameter;
    terms.push_back(detail::make_term(sites, coefficients, 2, parameter, 1.0, shares));
  }
  return terms;
}

inline std::vector<Term> make_edge_terms(const std::vector<EdgeInput>& edges)
{
  std::vector<Term> terms;
  terms.reserve(edges.size() * 3);
  for(const EdgeInput& edge : edges) {
    const std::vector<Term> local = make_edge_terms(edge);
    terms.insert(terms.end(), local.begin(), local.end());
  }
  return terms;
}

inline std::vector<Term> make_symmetric_edge_terms(const Site& first, const Site& second,
                                                   int parameter_offset)
{
  detail::check_site(first);
  detail::check_site(second);
  if(first.atom == second.atom || parameter_offset < 0 || parameter_offset > std::numeric_limits<int>::max() - 5)
    throw std::runtime_error("invalid local-basis symmetric edge");
  const Site sites[2] = {first, second};
  const double shares[2] = {0.5, 0.5};
  const Vec3 basis[3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  std::vector<Term> terms;
  terms.reserve(9);
  for(int axis = 0; axis < 3; ++axis) {
    const Vec3 coefficients[2] = {detail::scale(basis[axis], -1.0), basis[axis]};
    const int parameter = axis == 0 ? 0 : axis == 1 ? 3 : 5;
    terms.push_back(detail::make_term(sites, coefficients, 2, parameter_offset + parameter, 1.0, shares));
  }
  const int off_diagonal_parameter[3] = {1, 2, 4};
  const int left_axis[3] = {0, 0, 1};
  const int right_axis[3] = {1, 2, 2};
  constexpr double inverse_sqrt_two = 0.70710678118654752440;
  for(int pair = 0; pair < 3; ++pair) {
    Vec3 plus = detail::scale(detail::add(basis[left_axis[pair]], basis[right_axis[pair]]), inverse_sqrt_two);
    Vec3 minus = detail::scale(detail::subtract(basis[left_axis[pair]], basis[right_axis[pair]]), inverse_sqrt_two);
    const Vec3 plus_coefficients[2] = {detail::scale(plus, -1.0), plus};
    const Vec3 minus_coefficients[2] = {detail::scale(minus, -1.0), minus};
    terms.push_back(detail::make_term(sites, plus_coefficients, 2,
      parameter_offset + off_diagonal_parameter[pair], 1.0, shares));
    terms.push_back(detail::make_term(sites, minus_coefficients, 2,
      parameter_offset + off_diagonal_parameter[pair], -1.0, shares));
  }
  return terms;
}

inline std::vector<Term> make_bridge_terms(const BridgeInput& bridge)
{
  detail::check_site(bridge.oxygen1);
  detail::check_site(bridge.hydrogen);
  detail::check_site(bridge.oxygen2);
  if(bridge.oxygen1.atom == bridge.hydrogen.atom || bridge.oxygen1.atom == bridge.oxygen2.atom ||
     bridge.hydrogen.atom == bridge.oxygen2.atom || !std::isfinite(bridge.a0))
    throw std::runtime_error("invalid local-basis bridge");
  const Vec3 axis = detail::unit(bridge.axis);
  Vec3 transverse[2];
  detail::perpendicular_axes(axis, transverse[0], transverse[1]);
  const Site sites[3] = {bridge.oxygen1, bridge.hydrogen, bridge.oxygen2};
  const double shares[3] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
  const Vec3 s[3] = {detail::scale(axis, -0.5), axis, detail::scale(axis, -0.5)};
  const Vec3 ell[3] = {detail::scale(axis, -1.0), {0.0, 0.0, 0.0}, axis};
  std::vector<Term> terms;
  terms.reserve(6);
  terms.push_back(detail::make_term(sites, s, 3, bridge.s_parameter, 1.0, shares));
  terms.push_back(detail::make_term(sites, ell, 3, bridge.ell_parameter, 1.0, shares));
  Vec3 plus[3], minus[3];
  for(int i = 0; i < 3; ++i) {
    plus[i] = detail::add(s[i], ell[i]);
    minus[i] = detail::subtract(s[i], ell[i]);
  }
  terms.push_back(detail::make_term(sites, plus, 3, bridge.cross_parameter, 0.5 * bridge.a0, shares));
  terms.push_back(detail::make_term(sites, minus, 3, bridge.cross_parameter, -0.5 * bridge.a0, shares));
  for(const Vec3 direction : transverse) {
    const Vec3 coefficients[3] = {detail::scale(direction, -0.5), direction, detail::scale(direction, -0.5)};
    terms.push_back(detail::make_term(sites, coefficients, 3, bridge.transverse_parameter, 1.0, shares));
  }
  return terms;
}

inline OperatorNormalization normalize_added_operators(const OperatorGramAccumulator& operators,
                                                       int base_parameter_count,
                                                       double relative_tolerance = 1.0e-12)
{
  const int parameter_count=operators.parameter_count();
  const std::vector<double> gram=operators.gram();
  if(parameter_count < 0 || base_parameter_count < 0 || base_parameter_count > parameter_count ||
     parameter_count - base_parameter_count > 32 || !(relative_tolerance >= 0.0) ||
     !std::isfinite(relative_tolerance))
    throw std::runtime_error("invalid local-basis operator dimensions or tolerance");

  std::vector<std::vector<double>> orthonormal_coefficients;
  std::vector<std::vector<double>> normalized_additions;
  std::vector<std::vector<double>> redundant_combinations;
  OperatorNormalization result;
  result.input_parameter_count = parameter_count;
  result.base_parameter_count = base_parameter_count;
  result.residual_relative_norm.assign(static_cast<std::size_t>(parameter_count), 0.0);

  auto residualize = [&](std::vector<double> coefficients) {
    for(int pass = 0; pass < 2; ++pass) for(const std::vector<double>& q : orthonormal_coefficients) {
      const double projection = detail::coefficient_dot(q, coefficients, gram, parameter_count);
      for(int p = 0; p < parameter_count; ++p) coefficients[p] -= projection * q[p];
    }
    return coefficients;
  };
  const auto norm=[&](const std::vector<double>& coefficients){return operators.linear_combination_norm(coefficients);};

  for(int p = 0; p < base_parameter_count; ++p) {
    std::vector<double> coefficients(static_cast<std::size_t>(parameter_count), 0.0);
    coefficients[p] = 1.0;
    const double original_norm = norm(coefficients);
    coefficients = residualize(std::move(coefficients));
    const double residual_norm = norm(coefficients);
    const double ratio = original_norm > 0.0 ? residual_norm / original_norm : 0.0;
    result.residual_relative_norm[p] = ratio;
    if(original_norm > 0.0 && ratio > relative_tolerance) {
      for(double& value : coefficients) value /= residual_norm;
      orthonormal_coefficients.push_back(std::move(coefficients));
    }
  }

  for(int p = base_parameter_count; p < parameter_count; ++p) {
    std::vector<double> coefficients(static_cast<std::size_t>(parameter_count), 0.0);
    coefficients[p] = 1.0;
    const double original_norm = norm(coefficients);
    coefficients = residualize(std::move(coefficients));
    const double residual_norm = norm(coefficients);
    const double ratio = original_norm > 0.0 ? residual_norm / original_norm : 0.0;
    result.residual_relative_norm[p] = ratio;
    if(!(original_norm > 0.0) || ratio <= relative_tolerance) {
      result.redundant_parameters.push_back(p);
      redundant_combinations.push_back(std::move(coefficients));
      continue;
    }
    for(double& value : coefficients) value /= residual_norm;
    normalized_additions.push_back(coefficients);
    orthonormal_coefficients.push_back(std::move(coefficients));
  }

  result.output_parameter_count = base_parameter_count + static_cast<int>(normalized_additions.size());
  result.transform.assign(static_cast<std::size_t>(parameter_count) * result.output_parameter_count, 0.0);
  result.output_source_parameters.reserve(static_cast<std::size_t>(result.output_parameter_count));
  for(int p = 0; p < base_parameter_count; ++p) {
    result.transform[static_cast<std::size_t>(p) * result.output_parameter_count + p] = 1.0;
    result.output_source_parameters.push_back(p);
  }
  for(std::size_t addition = 0; addition < normalized_additions.size(); ++addition) {
    const int column = base_parameter_count + static_cast<int>(addition);
    for(int p = 0; p < parameter_count; ++p)
      result.transform[static_cast<std::size_t>(p) * result.output_parameter_count + column] = normalized_additions[addition][p];
    for(int p = base_parameter_count; p < parameter_count; ++p)
      if(normalized_additions[addition][p] != 0.0) { result.output_source_parameters.push_back(p); break; }
  }
  result.redundant_combination.assign(static_cast<std::size_t>(parameter_count) * redundant_combinations.size(), 0.0);
  for(std::size_t column = 0; column < redundant_combinations.size(); ++column)
    for(int p = 0; p < parameter_count; ++p)
      result.redundant_combination[static_cast<std::size_t>(p) * redundant_combinations.size() + column] =
        redundant_combinations[column][p];
  return result;
}

inline OperatorNormalization normalize_added_operators(const std::vector<Term>& terms,
                                                       int parameter_count, int base_parameter_count,
                                                       double relative_tolerance = 1.0e-12)
{
  OperatorGramAccumulator operators(parameter_count, [&]() {
    int atom_count = 0;
    for(const Term& term : terms) for(int i = 0; i < term.atom_count; ++i)
      atom_count = std::max(atom_count, term.site[i].atom + 1);
    return atom_count;
  }());
  for(const Term& term : terms) operators.add(term);
  return normalize_added_operators(operators,base_parameter_count,relative_tolerance);
}

inline std::vector<Term> transform_terms(const std::vector<Term>& terms,
                                         const OperatorNormalization& normalization)
{
  if(normalization.input_parameter_count < 0 || normalization.output_parameter_count < 0 ||
     normalization.transform.size() != static_cast<std::size_t>(normalization.input_parameter_count) *
                                         normalization.output_parameter_count)
    throw std::runtime_error("invalid local-basis operator transform");
  std::vector<Term> transformed;
  for(const Term& term : terms) {
    if(term.parameter < 0 || term.parameter >= normalization.input_parameter_count)
      throw std::runtime_error("local-basis term parameter is outside the operator transform");
    for(int p = 0; p < normalization.output_parameter_count; ++p) {
      const double scale = normalization.transform[static_cast<std::size_t>(term.parameter) *
                                                    normalization.output_parameter_count + p];
      if(scale == 0.0) continue;
      if(!std::isfinite(scale)) throw std::runtime_error("non-finite local-basis operator transform");
      Term copy = term;
      copy.parameter = p;
      copy.weight *= scale;
      transformed.push_back(copy);
    }
  }
  return transformed;
}

inline std::vector<double> operator_gram(const std::vector<Term>& terms, int parameter_count,
                                         int atom_count, const double* masses = nullptr)
{
  OperatorGramAccumulator operators(parameter_count, atom_count, masses);
  for(const Term& term : terms) operators.add(term);
  return operators.gram();
}

inline Evaluation evaluate(const std::vector<Term>& terms, const double* theta, int parameter_count,
                           const double* displacement_soa, int atom_count)
{
  detail::check_terms(terms, theta, parameter_count, atom_count, nullptr);
  if(atom_count && !displacement_soa) throw std::runtime_error("missing local-basis displacements");
  Evaluation result;
  result.site_energy.assign(static_cast<std::size_t>(atom_count), 0.0);
  result.gradient.assign(static_cast<std::size_t>(3 * atom_count), 0.0);
  for(const Term& term : terms) {
    const double q = detail::projection(term, displacement_soa, atom_count, nullptr);
    const double scale = theta[term.parameter] * term.weight;
    const double energy = 0.5 * scale * q * q;
    if(!std::isfinite(energy)) throw std::runtime_error("non-finite local-basis energy");
    result.energy += energy;
    for(int i = 0; i < term.atom_count; ++i) {
      const int atom = term.site[i].atom;
      result.site_energy[atom] += energy * term.energy_share[i];
      result.gradient[atom] += scale * q * term.coefficient[i].x;
      result.gradient[atom_count + atom] += scale * q * term.coefficient[i].y;
      result.gradient[2 * atom_count + atom] += scale * q * term.coefficient[i].z;
    }
  }
  return result;
}

inline std::vector<double> apply_hessian(const std::vector<Term>& terms, const double* theta,
                                         int parameter_count, const double* vector_soa, int atom_count,
                                         const double* masses = nullptr)
{
  detail::check_terms(terms, theta, parameter_count, atom_count, masses);
  if(atom_count && !vector_soa) throw std::runtime_error("missing local-basis action vector");
  const std::vector<double> inverse_sqrt = detail::inverse_sqrt_masses(atom_count, masses);
  std::vector<double> result(static_cast<std::size_t>(3 * atom_count), 0.0);
  for(const Term& term : terms) {
    const double q = detail::projection(term, vector_soa, atom_count, masses);
    const double scale = theta[term.parameter] * term.weight * q;
    for(int i = 0; i < term.atom_count; ++i) {
      const int atom = term.site[i].atom;
      const double factor = inverse_sqrt[atom] * scale;
      result[atom] += factor * term.coefficient[i].x;
      result[atom_count + atom] += factor * term.coefficient[i].y;
      result[2 * atom_count + atom] += factor * term.coefficient[i].z;
    }
  }
  return result;
}

inline std::vector<double> apply_hessian_sqrt_mass(const std::vector<Term>& terms, const double* theta,
                                                   int parameter_count, const double* vector_soa,
                                                   const double* sqrt_mass_soa, int atom_count)
{
  if(parameter_count < 0 || atom_count < 0 || (parameter_count && !theta) ||
     (atom_count && (!vector_soa || !sqrt_mass_soa)))
    throw std::runtime_error("invalid local-basis square-root-mass action dimensions");
  for(int i = 0; i < 3 * atom_count; ++i)
    if(!(sqrt_mass_soa[i] > 0.0) || !std::isfinite(sqrt_mass_soa[i]))
      throw std::runtime_error("invalid local-basis square-root mass");
  for(const Term& term : terms) {
    if(term.parameter < 0 || term.parameter >= parameter_count || term.atom_count < 1 || term.atom_count > 3 ||
       !std::isfinite(term.weight) || !std::isfinite(theta[term.parameter]))
      throw std::runtime_error("invalid local-basis square-root-mass term");
    for(int i = 0; i < term.atom_count; ++i)
      if(term.site[i].atom < 0 || term.site[i].atom >= atom_count || !detail::finite(term.coefficient[i]))
        throw std::runtime_error("local-basis square-root-mass site is outside the atom array");
  }
  std::vector<double> result(static_cast<std::size_t>(3 * atom_count), 0.0);
  for(const Term& term : terms) {
    double projection = 0.0;
    for(int i = 0; i < term.atom_count; ++i) {
      const int atom = term.site[i].atom;
      projection += term.coefficient[i].x * vector_soa[atom] / sqrt_mass_soa[atom];
      projection += term.coefficient[i].y * vector_soa[atom_count + atom] / sqrt_mass_soa[atom_count + atom];
      projection += term.coefficient[i].z * vector_soa[2 * atom_count + atom] / sqrt_mass_soa[2 * atom_count + atom];
    }
    const double scale = term.weight * theta[term.parameter] * projection;
    for(int i = 0; i < term.atom_count; ++i) {
      const int atom = term.site[i].atom;
      result[atom] += term.coefficient[i].x * scale / sqrt_mass_soa[atom];
      result[atom_count + atom] += term.coefficient[i].y * scale / sqrt_mass_soa[atom_count + atom];
      result[2 * atom_count + atom] += term.coefficient[i].z * scale / sqrt_mass_soa[2 * atom_count + atom];
    }
  }
  return result;
}

inline std::vector<double> apply_parameter_hessian(const std::vector<Term>& terms, int parameter,
                                                   const double* vector_soa, int atom_count,
                                                   const double* masses = nullptr)
{
  if(parameter < 0 || atom_count < 0 || (atom_count && !vector_soa))
    throw std::runtime_error("invalid local-basis parameter action dimensions");
  if(masses) for(int i = 0; i < atom_count; ++i)
    if(!(masses[i] > 0.0) || !std::isfinite(masses[i])) throw std::runtime_error("invalid local-basis mass");
  const std::vector<double> inverse_sqrt = detail::inverse_sqrt_masses(atom_count, masses);
  std::vector<double> result(static_cast<std::size_t>(3 * atom_count), 0.0);
  for(const Term& term : terms) if(term.parameter == parameter) {
    const double q = detail::projection(term, vector_soa, atom_count, masses);
    const double scale = term.weight * q;
    for(int i = 0; i < term.atom_count; ++i) {
      const int atom = term.site[i].atom;
      if(atom < 0 || atom >= atom_count || !std::isfinite(term.weight) || !detail::finite(term.coefficient[i]))
        throw std::runtime_error("invalid local-basis parameter action term");
      const double factor = inverse_sqrt[atom] * scale;
      result[atom] += factor * term.coefficient[i].x;
      result[atom_count + atom] += factor * term.coefficient[i].y;
      result[2 * atom_count + atom] += factor * term.coefficient[i].z;
    }
  }
  return result;
}

inline std::vector<HessianBlock> hessian_blocks(const std::vector<Term>& terms, const double* theta,
                                               int parameter_count, int atom_count = 0,
                                               const double* masses = nullptr)
{
  if(atom_count == 0) for(const Term& term : terms)
    for(int i = 0; i < term.atom_count; ++i) atom_count = std::max(atom_count, term.site[i].atom + 1);
  detail::check_terms(terms, theta, parameter_count, atom_count, masses);
  const std::vector<double> inverse_sqrt = detail::inverse_sqrt_masses(atom_count, masses);
  std::vector<HessianBlock> blocks;
  for(const Term& term : terms) for(int i = 0; i < term.atom_count; ++i) for(int j = 0; j < term.atom_count; ++j) {
    HessianBlock block{};
    block.row = term.site[i];
    block.column = term.site[j];
    block.parameter = term.parameter;
    const double scale = theta[term.parameter] * term.weight * inverse_sqrt[block.row.atom] * inverse_sqrt[block.column.atom];
    const double row[3] = {term.coefficient[i].x, term.coefficient[i].y, term.coefficient[i].z};
    const double column[3] = {term.coefficient[j].x, term.coefficient[j].y, term.coefficient[j].z};
    for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) block.value[3 * a + b] = scale * row[a] * column[b];
    blocks.push_back(block);
  }
  return blocks;
}

inline std::vector<SparseEntry> sparse_entries(const std::vector<Term>& terms, const double* theta,
                                               int parameter_count, int atom_count = 0,
                                               const double* masses = nullptr)
{
  const std::vector<HessianBlock> blocks = hessian_blocks(terms, theta, parameter_count, atom_count, masses);
  std::vector<SparseEntry> entries;
  entries.reserve(blocks.size() * 9);
  for(const HessianBlock& block : blocks) for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) {
    SparseEntry entry{};
    entry.row = block.row;
    entry.column = block.column;
    entry.parameter = block.parameter;
    entry.row_component = a;
    entry.column_component = b;
    entry.value = block.value[3 * a + b];
    entries.push_back(entry);
  }
  return entries;
}

} // namespace rpmd_ja_local_basis
