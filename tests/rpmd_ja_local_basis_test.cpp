#include "../src/measure/rpmd_ja_local_basis.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

using namespace rpmd_ja_local_basis;

static bool close(double a, double b, double tolerance = 2.0e-10)
{
  return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

static void compare(const std::vector<double>& a, const std::vector<double>& b, double tolerance = 2.0e-10)
{
  assert(a.size() == b.size());
  for(std::size_t i = 0; i < a.size(); ++i) assert(close(a[i], b[i], tolerance));
}

static std::vector<double> dense_action(const std::vector<HessianBlock>& blocks,
                                        const std::vector<double>& x, int n,
                                        const double* masses = nullptr)
{
  std::vector<double> y(3 * n, 0.0);
  for(const HessianBlock& block : blocks) {
    for(int a = 0; a < 3; ++a) for(int b = 0; b < 3; ++b) {
      double value = block.value[3 * a + b];
      if(masses) value /= std::sqrt(masses[block.row.atom] * masses[block.column.atom]);
      y[a * n + block.row.atom] += value * x[b * n + block.column.atom];
    }
  }
  return y;
}

static void check_pair_arithmetic()
{
  EdgeInput edge{};
  edge.first = {0, {0, 0, 0}};
  edge.second = {1, {1, -1, 0}};
  edge.axis = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};
  edge.longitudinal_parameter = 0;
  edge.transverse_parameter = 1;
  const std::vector<Term> terms = make_edge_terms(edge);
  assert(terms.size() == 3);

  const int n = 3;
  const double theta[] = {2.3, -0.7};
  const double masses[] = {1.3, 4.2, 8.1};
  std::vector<double> x = {0.2, -0.4, 0.7, -0.6, 0.3, 0.1, 0.5, -0.2, 0.9};
  const Evaluation value = evaluate(terms, theta, 2, x.data(), n);
  assert(value.gradient.size() == x.size());
  assert(value.site_energy.size() == static_cast<std::size_t>(n));
  assert(close(value.site_energy[0], value.energy * 0.5));
  assert(close(value.site_energy[1], value.energy * 0.5));
  assert(close(value.site_energy[2], 0.0));
  assert(close(value.gradient[0] + value.gradient[1] + value.gradient[2], 0.0));
  assert(close(value.gradient[n] + value.gradient[n + 1] + value.gradient[n + 2], 0.0));
  assert(close(value.gradient[2 * n] + value.gradient[2 * n + 1] + value.gradient[2 * n + 2], 0.0));

  for(std::size_t q = 0; q < x.size(); ++q) {
    const double original = x[q];
    x[q] = original + 1.0e-6;
    const double plus = evaluate(terms, theta, 2, x.data(), n).energy;
    x[q] = original - 1.0e-6;
    const double minus = evaluate(terms, theta, 2, x.data(), n).energy;
    x[q] = original;
    assert(close((plus - minus) / 2.0e-6, value.gradient[q], 2.0e-8));
  }

  const auto blocks = hessian_blocks(terms, theta, 2);
  assert(blocks.size() == terms.size() * 4);
  bool found_owned_image = false;
  for(const HessianBlock& block : blocks)
    if(block.row.atom == 0 && block.column.atom == 1 && block.column.image[0] == 1 &&
       block.column.image[1] == -1)
      found_owned_image = true;
  assert(found_owned_image);
  const auto actual = apply_hessian(terms, theta, 2, x.data(), n);
  compare(actual, dense_action(blocks, x, n));

  const auto weighted = apply_hessian(terms, theta, 2, x.data(), n, masses);
  compare(weighted, dense_action(blocks, x, n, masses));
  std::vector<double> sqrt_mass(3 * n);
  for(int axis = 0; axis < 3; ++axis) for(int atom = 0; atom < n; ++atom)
    sqrt_mass[axis * n + atom] = std::sqrt(masses[atom]);
  compare(apply_hessian_sqrt_mass(terms, theta, 2, x.data(), sqrt_mass.data(), n), weighted);
  const auto sparse = sparse_entries(terms, theta, 2, masses);
  assert(!sparse.empty());
  bool found_cross_atom = false;
  for(const SparseEntry& entry : sparse) {
    if(entry.row.atom != entry.column.atom) found_cross_atom = true;
    double unweighted = 0.0;
    for(const HessianBlock& block : blocks)
      if(block.row.atom == entry.row.atom && block.column.atom == entry.column.atom &&
         block.parameter == entry.parameter)
        unweighted += block.value[3 * entry.row_component + entry.column_component];
    assert(close(entry.value, unweighted / std::sqrt(masses[entry.row.atom] * masses[entry.column.atom])));
  }
  assert(found_cross_atom);
  const auto longitudinal_action = apply_parameter_hessian(terms, 0, x.data(), n);
  std::vector<Term> longitudinal_terms;
  for(const Term& term : terms) if(term.parameter == 0) longitudinal_terms.push_back(term);
  const double unit_theta[] = {1.0};
  compare(longitudinal_action, apply_hessian(longitudinal_terms, unit_theta, 1, x.data(), n));

  std::vector<double> shifted = x;
  for(int atom = 0; atom < n; ++atom) {
    shifted[atom] += 0.37;
    shifted[n + atom] -= 0.21;
    shifted[2 * n + atom] += 0.13;
  }
  const Evaluation translated = evaluate(terms, theta, 2, shifted.data(), n);
  assert(close(translated.energy, value.energy));
  compare(translated.gradient, value.gradient);
}

static void check_symmetric_edge_arithmetic()
{
  const Site first = {0, {0, 0, 0}};
  const Site second = {1, {-1, 2, 0}};
  const std::vector<Term> terms = make_symmetric_edge_terms(first, second, 0);
  const double theta[] = {1.2, -0.4, 0.7, 2.1, -1.3, 0.9};
  const double displacement[] = {0.5, -0.2, 0.0, -0.3, 0.8, 0.0};
  const double dx = displacement[1] - displacement[0];
  const double dy = displacement[4] - displacement[3];
  const double expected = 0.5 * theta[0] * dx * dx + theta[1] * dx * dy +
    theta[3] * dy * dy * 0.5;
  const Evaluation value = evaluate(terms, theta, 6, displacement, 2);
  assert(close(value.energy, expected));
  assert(close(value.site_energy[0], 0.5 * expected));
  assert(close(value.site_energy[1], 0.5 * expected));
  const std::vector<double> gram = operator_gram(terms, 6, 2);
  assert(close(gram[0], 4.0));
  assert(close(gram[1 * 6 + 1], 8.0));
  assert(close(gram[0 * 6 + 1], 0.0));
}

static void check_bridge_exchange_and_zero_extension()
{
  BridgeInput bridge{};
  bridge.oxygen1 = {0, {0, 0, 0}};
  bridge.hydrogen = {1, {0, -1, 0}};
  bridge.oxygen2 = {2, {1, 0, 0}};
  bridge.axis = {0.0, 0.0, 1.0};
  bridge.a0 = 0.37;
  bridge.s_parameter = 2;
  bridge.ell_parameter = 3;
  bridge.cross_parameter = 4;
  bridge.transverse_parameter = 5;
  const std::vector<Term> bridge_terms = make_bridge_terms(bridge);
  assert(bridge_terms.size() == 6);
  assert(bridge_terms[2].weight != 0.0 && bridge_terms[3].weight != 0.0);
  assert(bridge_terms[2].weight == -bridge_terms[3].weight);

  const int n = 3;
  const double theta[] = {0.0, 0.0, 1.4, -0.8, 2.1, 0.6};
  std::vector<double> x = {0.3, -0.2, 0.8, 0.6, 0.4, -0.1, -0.7, 0.2, 0.5};
  const Evaluation original = evaluate(bridge_terms, theta, 6, x.data(), n);
  assert(close(original.site_energy[0], original.energy / 3.0));
  assert(close(original.site_energy[1], original.energy / 3.0));
  assert(close(original.site_energy[2], original.energy / 3.0));
  assert(close(original.gradient[0] + original.gradient[1] + original.gradient[2], 0.0));

  BridgeInput swapped = bridge;
  std::swap(swapped.oxygen1, swapped.oxygen2);
  swapped.axis = {0.0, 0.0, -1.0};
  swapped.a0 = -bridge.a0;
  const Evaluation exchanged = evaluate(make_bridge_terms(swapped), theta, 6, x.data(), n);
  assert(close(exchanged.energy, original.energy));
  compare(exchanged.gradient, original.gradient);
  bool found_o1_o2_block = false;
  for(const HessianBlock& block : hessian_blocks(bridge_terms, theta, 6))
    if(block.row.atom == 0 && block.column.atom == 2 && std::abs(block.value[8]) > 1.0e-12)
      found_o1_o2_block = true;
  assert(found_o1_o2_block);

  EdgeInput edge{};
  edge.first = {0, {0, 0, 0}};
  edge.second = {2, {0, 0, 0}};
  edge.axis = {1.0, 0.0, 0.0};
  edge.longitudinal_parameter = 0;
  edge.transverse_parameter = 1;
  const std::vector<Term> base = make_edge_terms(edge);
  std::vector<Term> extended = base;
  extended.insert(extended.end(), bridge_terms.begin(), bridge_terms.end());
  const double zero_extended_theta[] = {1.1, 0.4, 0.0, 0.0, 0.0, 0.0};
  const Evaluation base_value = evaluate(base, zero_extended_theta, 6, x.data(), n);
  const Evaluation extended_value = evaluate(extended, zero_extended_theta, 6, x.data(), n);
  assert(close(base_value.energy, extended_value.energy));
  compare(base_value.site_energy, extended_value.site_energy);
  compare(base_value.gradient, extended_value.gradient);
  compare(apply_hessian(base, zero_extended_theta, 6, x.data(), n),
          apply_hessian(extended, zero_extended_theta, 6, x.data(), n));
}

static void check_operator_normalization()
{
  EdgeInput base_edge{};
  base_edge.first = {0, {0, 0, 0}};
  base_edge.second = {1, {1, 0, 0}};
  base_edge.axis = {1.0, 0.0, 0.0};
  base_edge.longitudinal_parameter = 0;
  base_edge.transverse_parameter = 1;
  std::vector<Term> terms = make_edge_terms(base_edge);

  EdgeInput duplicate = base_edge;
  duplicate.second.image[1] = 7;
  duplicate.longitudinal_parameter = 2;
  const std::vector<Term> duplicate_terms = make_edge_terms(duplicate);
  terms.insert(terms.end(), duplicate_terms.begin(), duplicate_terms.begin() + 1);
  terms.back().energy_share[0] = 0.0;
  terms.back().energy_share[1] = 1.0;

  EdgeInput independent{};
  independent.first = {1, {0, 0, 0}};
  independent.second = {2, {0, 0, 0}};
  independent.axis = {0.0, 1.0, 0.0};
  independent.longitudinal_parameter = 3;
  independent.transverse_parameter = 4;
  const std::vector<Term> independent_terms = make_edge_terms(independent);
  terms.insert(terms.end(), independent_terms.begin(), independent_terms.begin() + 1);

  const OperatorNormalization normalization = normalize_added_operators(terms, 4, 2);
  assert(normalization.output_parameter_count == 3);
  assert(normalization.output_source_parameters.size() == 3);
  assert(normalization.output_source_parameters[0] == 0);
  assert(normalization.output_source_parameters[1] == 1);
  assert(normalization.output_source_parameters[2] == 3);
  assert(normalization.redundant_parameters.size() == 1);
  assert(normalization.redundant_parameters[0] == 2);
  assert(normalization.redundant_combination.size() == 4);
  assert(normalization.residual_relative_norm.size() == 4);
  assert(normalization.residual_relative_norm[2] < 1.0e-12);
  assert(close(normalization.transform[0 * 3 + 0], 1.0));
  assert(close(normalization.transform[1 * 3 + 1], 1.0));

  const std::vector<Term> transformed = transform_terms(terms, normalization);
  const double transformed_theta[] = {0.3, -0.2, 0.7};
  double original_theta[4] = {};
  for(int input = 0; input < normalization.input_parameter_count; ++input)
    for(int output = 0; output < normalization.output_parameter_count; ++output)
      original_theta[input] += normalization.transform[
        static_cast<std::size_t>(input) * normalization.output_parameter_count + output] * transformed_theta[output];
  const std::vector<double> x = {0.2, -0.4, 0.7, -0.6, 0.3, 0.1, 0.5, -0.2, 0.9};
  const Evaluation original = evaluate(terms, original_theta, 4, x.data(), 3);
  const Evaluation normalized = evaluate(transformed, transformed_theta, 3, x.data(), 3);
  assert(close(normalized.energy, original.energy));
  compare(normalized.site_energy, original.site_energy);
  compare(normalized.gradient, original.gradient);

  const double unit_transformed_theta[] = {1.0, 1.0, 1.0};
  const auto normalized_blocks = hessian_blocks(transformed, unit_transformed_theta, 3);
  double operator_norm_squared = 0.0;
  for(const HessianBlock& block : normalized_blocks)
    if(block.parameter == 2) for(double value : block.value) operator_norm_squared += value * value;
  assert(close(std::sqrt(operator_norm_squared), 1.0));

  std::vector<Term> null_terms;
  for(const Term& term : terms) {
    const double coefficient = normalization.redundant_combination[
      static_cast<std::size_t>(term.parameter) * normalization.redundant_parameters.size()];
    if(coefficient != 0.0) {
      Term combined = term;
      combined.parameter = 0;
      combined.weight *= coefficient;
      null_terms.push_back(combined);
    }
  }
  const double null_theta[] = {1.0};
  const Evaluation gauge = evaluate(null_terms, null_theta, 1, x.data(), 3);
  assert(close(gauge.energy, 0.0));
  assert(close(gauge.gradient[0], 0.0) && close(gauge.gradient[1], 0.0));
  assert(!close(gauge.site_energy[0], 0.0) || !close(gauge.site_energy[1], 0.0));

  bool rejected_overflow = false;
  try { (void)normalize_added_operators(terms, 35, 2); }
  catch(const std::runtime_error&) { rejected_overflow = true; }
  assert(rejected_overflow);
}

static OperatorNormalization nearly_dependent_normalization(const double tilt)
{
  const double masses[] = {1.7, 9.3};
  EdgeInput base{};base.first={0,{0,0,0}};base.second={1,{0,0,0}};
  base.axis={1.0,0.0,0.0};base.longitudinal_parameter=0;base.transverse_parameter=1;
  EdgeInput added=base;added.second.image[0]=1;added.second.image[1]=-2;
  const double norm=std::sqrt(1.0+tilt*tilt);added.axis={1.0/norm,tilt/norm,0.0};
  added.longitudinal_parameter=1;added.transverse_parameter=2;
  const std::vector<Term> base_terms=make_edge_terms(base),added_terms=make_edge_terms(added);
  OperatorGramAccumulator operators(2,2,masses);
  operators.add(base_terms.front());operators.add(added_terms.front());
  return normalize_added_operators(operators,1,1.0e-12);
}

static void check_near_dependent_sparse_residual()
{
  const OperatorNormalization independent=nearly_dependent_normalization(1.0e-10);
  assert(independent.output_parameter_count==2);
  assert(independent.redundant_parameters.empty());
  assert(independent.residual_relative_norm[1]>1.0e-12);
  const OperatorNormalization redundant=nearly_dependent_normalization(1.0e-14);
  assert(redundant.output_parameter_count==1);
  assert(redundant.redundant_parameters.size()==1&&redundant.redundant_parameters[0]==1);
  assert(redundant.residual_relative_norm[1]<=1.0e-12);
}

int main()
{
  check_pair_arithmetic();
  check_symmetric_edge_arithmetic();
  check_bridge_exchange_and_zero_extension();
  check_operator_normalization();
  check_near_dependent_sparse_residual();
}
