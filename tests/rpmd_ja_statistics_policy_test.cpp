#ifdef NDEBUG
#undef NDEBUG
#endif
#include "rpmd_ja_statistics_policy.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
std::vector<double> ar1(const double phi, const std::size_t count, std::uint64_t seed)
{
  std::vector<double> values(count);
  double value = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    const double noise = static_cast<double>(seed >> 11) / 9007199254740992.0 - 0.5;
    value = phi * value + noise;
    values[i] = value;
  }
  return values;
}
}

int main()
{
  using namespace rpmd_ja_statistics_policy;
  const FrozenConfig config = frozen_config();
  TrainingChannels training{{"physical_mean_x", "physical_second_x"},
    {ar1(0.4, 8192, 10), ar1(0.2, 8192, 11)}};
  const BlockPlan plan = plan_training_blocks(training, config);
  assert(plan.status == "PASS");
  assert(plan.tau_frames > 1.0);
  assert(plan.adjacent_block_length == 2 * plan.block_length);
  assert(plan.block_count >= config.min_blocks && plan.adjacent_block_count >= config.min_blocks);
  assert(plan.common_covered_frame_count == plan.block_count * plan.block_length);
  assert(plan.common_covered_frame_count == plan.adjacent_block_count * plan.adjacent_block_length);
  assert(plan.discarded_tail_frames == plan.frame_count - plan.common_covered_frame_count);
  assert(plan.max_adjacent_se_relative_change <= config.adjacent_se_relative_tolerance);

  std::size_t budget_observations=0;
  bool budget_stop_observed=false;
  try {
    const TrainingChannels slow{{"slow"}, {ar1(0.995, 16384, 41)}};
    const BudgetObserver stop_after_twenty_two=[&](){
      if(++budget_observations==22)throw std::runtime_error("injected statistics budget stop");
    };
    (void)plan_training_blocks(slow,config,stop_after_twenty_two);
  } catch(const std::runtime_error& error) {
    budget_stop_observed=std::string(error.what())=="injected statistics budget stop";
  }
  assert(budget_stop_observed && budget_observations==22);

  const std::vector<double> truncated_values={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1000};
  const std::size_t common_prefix=detail::common_block_coverage(truncated_values.size(),4);
  assert(common_prefix==16);
  assert(detail::prefix_mean(truncated_values,common_prefix)==8.5);

  TrainingChannels tail_training{{"q_x"}, {ar1(0.1, 4099, 22)}};
  const BlockPlan tail_plan = plan_training_blocks(tail_training, config);
  assert(tail_plan.status == "PASS");
  const std::size_t tail_common_prefix = detail::common_block_coverage(tail_training.values[0].size(),
    tail_plan.block_length);
  assert(tail_common_prefix < tail_training.values[0].size());
  TrainingChannels tail_development = tail_training;
  for (std::size_t frame = tail_common_prefix; frame < tail_development.values[0].size(); ++frame)
    tail_development.values[0][frame] = 1000.0;
  const MeanDrift tail_drift = assess_fixed_r0_means(tail_training, tail_development, tail_plan, config);
  assert(tail_drift.development_covered_frame_count == tail_common_prefix);
  assert(tail_drift.mean_development[0] == detail::mean(tail_development.values[0]));
  assert(tail_drift.status == "DATA_INCONCLUSIVE");
  assert(tail_drift.channels[0].delta_z > tail_drift.z_tolerance);
  std::size_t drift_budget_observations = 0;
  bool drift_budget_stop_observed = false;
  try {
    const BudgetObserver stop_drift_after_four = [&]() {
      if (++drift_budget_observations == 4) throw std::runtime_error("injected drift budget stop");
    };
    (void)assess_fixed_r0_means(tail_training, tail_development, tail_plan, config, nullptr,
      stop_drift_after_four);
  } catch (const std::runtime_error& error) {
    drift_budget_stop_observed = std::string(error.what()) == "injected drift budget stop";
  }
  assert(drift_budget_stop_observed && drift_budget_observations == 4);

  TrainingChannels constant{{"zero"}, {std::vector<double>(1024, 0.0)}};
  const BlockPlan constant_plan = plan_training_blocks(constant, config);
  assert(constant_plan.status == "DATA_INCONCLUSIVE");
  assert(constant_plan.channels[0].state == "CONSTANT_ZERO");

  TrainingChannels constant_and_varying{{"constant_zero", "varying"},
    {std::vector<double>(8192, 0.0), ar1(0.1, 8192, 13)}};
  TrainingChannels constant_development{{"constant_zero", "varying"},
    {std::vector<double>(8192, 0.0), ar1(0.1, 8192, 14)}};
  const BlockPlan constant_and_varying_plan = plan_training_blocks(constant_and_varying, config);
  assert(constant_and_varying_plan.status == "PASS");
  const MeanDrift stable_constants = assess_fixed_r0_means(
    constant_and_varying, constant_development, constant_and_varying_plan, config);
  assert(stable_constants.status == "PASS");
  assert(stable_constants.channels[0].state == "CONSTANT_ZERO");
  for (double& value : constant_development.values[0]) value = 0.25;
  const MeanDrift changed_constant = assess_fixed_r0_means(
    constant_and_varying, constant_development, constant_and_varying_plan, config);
  assert(changed_constant.status == "DATA_INCONCLUSIVE");
  assert(changed_constant.channels[0].state == "CONSTANT_CHANGED");
  assert(std::isinf(changed_constant.channels[0].delta_z));

  TrainingChannels q_train{{"q_x"}, {ar1(0.1, 4096, 20)}};
  TrainingChannels q_development{{"q_x"}, {ar1(0.1, 4096, 21)}};
  const BlockPlan q_plan = plan_training_blocks(q_train, config);
  assert(q_plan.status == "PASS");
  const MeanDrift stable = assess_fixed_r0_means(q_train, q_development, q_plan, config);
  assert(stable.status == "PASS");
  assert(stable.dimension_count == 1);
  assert(stable.covariance_of_mean_difference.size() == 1);
  assert(stable.adjacent_covariance_of_mean_difference.size() == 1);
  assert(stable.development_adjacent_se_relative_change <= config.adjacent_se_relative_tolerance);
  assert(stable.channels[0].se_train > 0.0 && stable.channels[0].se_development > 0.0);

  for (double& value : q_development.values[0]) value += 0.4;
  const MeanDrift shifted = assess_fixed_r0_means(q_train, q_development, q_plan, config);
  assert(shifted.status == "DATA_INCONCLUSIVE");
  assert(shifted.channels[0].delta_z > config.mean_drift_z_tolerance);

  TrainingChannels q_slow_development{{"q_x"}, {ar1(0.95, 16384, 31)}};
  const BlockPlan q_long_plan = plan_training_blocks(TrainingChannels{{"q_x"}, {ar1(0.1, 16384, 30)}}, config);
  const MeanDrift slow_development = assess_fixed_r0_means(
    TrainingChannels{{"q_x"}, {ar1(0.1, 16384, 30)}}, q_slow_development, q_long_plan, config);
  assert(slow_development.status == "DATA_INCONCLUSIVE");
  assert(slow_development.reason == "DEVELOPMENT_BLOCK_SENSITIVE");

  const PairedChoice selected = assess_paired_choice(std::vector<double>(8, -1.0),
    std::vector<double>(4, -0.8), config);
  assert(selected.status == "A_WINS");
  const PairedChoice unstable = assess_paired_choice(std::vector<double>(8, -1.0),
    std::vector<double>(4, 1.0), config);
  assert(unstable.status == "DATA_INCONCLUSIVE");
}
