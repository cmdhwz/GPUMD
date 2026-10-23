#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpmd_ja_statistics_policy
{
struct FrozenConfig
{
  std::string version;
  std::string autocorrelation_rule;
  double max_lag_fraction;
  double window_c;
  double min_samples_per_tau;
  double block_factor;
  std::size_t min_blocks;
  double adjacent_se_relative_tolerance;
  double mean_drift_z_tolerance;
  double mean_drift_family_alpha;
  std::string mean_drift_threshold_rule;
  double constant_relative_tolerance;
  double selection_z_multiplier;
};

inline FrozenConfig frozen_config()
{
  return {"RPJASTAT2", "FIXED_TRAINING_PHYSICAL_AND_PARAMETER_CHANNEL_SOKAL_WINDOW_V2", 0.25, 5.0, 50.0, 2.0, 4,
    0.25, 2.0, 0.05, "MAX_Z_SUBGAUSSIAN_UNION_ENGINEERING_V1", 1.0e-12, 2.0};
}

struct TrainingChannels
{
  std::vector<std::string> names;
  std::vector<std::vector<double>> values;
};

using BudgetObserver=std::function<void()>;

struct ChannelBlockSummary
{
  std::string name;
  std::string state;
  double mean = 0.0;
  double variance = 0.0;
  double tau_frames = std::numeric_limits<double>::quiet_NaN();
  std::size_t window_lag = 0;
  double se_block = std::numeric_limits<double>::quiet_NaN();
  double se_adjacent_block = std::numeric_limits<double>::quiet_NaN();
  double adjacent_se_relative_change = std::numeric_limits<double>::quiet_NaN();
};

struct BlockPlan
{
  std::string policy_version = "RPJASTAT2";
  std::string status = "DATA_INCONCLUSIVE";
  std::string reason;
  std::size_t frame_count = 0;
  std::size_t max_lag = 0;
  double tau_frames = std::numeric_limits<double>::quiet_NaN();
  std::size_t window_lag_max = 0;
  std::size_t block_length = 0;
  std::size_t adjacent_block_length = 0;
  std::size_t common_covered_frame_count = 0;
  std::size_t discarded_tail_frames = 0;
  std::size_t block_count = 0;
  std::size_t adjacent_block_count = 0;
  double max_adjacent_se_relative_change = std::numeric_limits<double>::quiet_NaN();
  std::vector<ChannelBlockSummary> channels;
};

struct ChannelDriftSummary
{
  std::string name;
  std::string state;
  double mean_train = 0.0;
  double mean_development = 0.0;
  double mean_difference = 0.0;
  double se_train = 0.0;
  double se_development = 0.0;
  double se_difference = 0.0;
  double train_z = 0.0;
  double development_z = 0.0;
  double delta_z = 0.0;
};

struct MeanDrift
{
  std::string policy_version = "RPJASTAT2";
  std::string status = "DATA_INCONCLUSIVE";
  std::string reason;
  std::string scope = "DECLARED_PHYSICAL_CHANNELS_ONLY";
  std::size_t dimension_count = 0;
  double tau_frames = std::numeric_limits<double>::quiet_NaN();
  double effective_frames_train = std::numeric_limits<double>::quiet_NaN();
  double effective_frames_development = std::numeric_limits<double>::quiet_NaN();
  std::size_t block_length = 0;
  std::size_t adjacent_block_length = 0;
  std::size_t training_covered_frame_count = 0;
  std::size_t development_covered_frame_count = 0;
  std::size_t training_discarded_tail_frames = 0;
  std::size_t development_discarded_tail_frames = 0;
  double development_adjacent_se_relative_change = std::numeric_limits<double>::quiet_NaN();
  double z_tolerance = std::numeric_limits<double>::quiet_NaN();
  double family_alpha = std::numeric_limits<double>::quiet_NaN();
  std::size_t comparison_count = 0;
  std::string threshold_method = "MAX_Z_SUBGAUSSIAN_UNION_ENGINEERING_V1";
  double maximum_z = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> mean_train;
  std::vector<double> mean_development;
  std::vector<double> mean_difference;
  std::vector<std::vector<double>> covariance_of_mean_train;
  std::vector<std::vector<double>> covariance_of_mean_development;
  std::vector<std::vector<double>> covariance_of_mean_difference;
  std::vector<std::vector<double>> adjacent_covariance_of_mean_train;
  std::vector<std::vector<double>> adjacent_covariance_of_mean_development;
  std::vector<std::vector<double>> adjacent_covariance_of_mean_difference;
  std::vector<ChannelDriftSummary> channels;
  std::vector<ChannelDriftSummary> adjacent_channels;
};

struct PairedScaleSummary
{
  bool resolved = false;
  std::string winner;
  double mean_difference = std::numeric_limits<double>::quiet_NaN();
  double standard_error = std::numeric_limits<double>::quiet_NaN();
};

struct PairedChoice
{
  std::string policy_version = "RPJASTAT2";
  std::string status = "DATA_INCONCLUSIVE";
  std::string rule = "PAIRED_TWO_SE_SAME_WINNER_AT_B_AND_2B_V1";
  PairedScaleSummary block;
  PairedScaleSummary adjacent_block;
};

namespace detail
{
inline void validate_config(const FrozenConfig& config)
{
  const FrozenConfig frozen = frozen_config();
  if (config.version != frozen.version || config.autocorrelation_rule != frozen.autocorrelation_rule ||
      config.max_lag_fraction != frozen.max_lag_fraction || config.window_c != frozen.window_c ||
      config.min_samples_per_tau != frozen.min_samples_per_tau || config.block_factor != frozen.block_factor ||
      config.min_blocks != frozen.min_blocks ||
      config.adjacent_se_relative_tolerance != frozen.adjacent_se_relative_tolerance ||
      config.mean_drift_z_tolerance != frozen.mean_drift_z_tolerance ||
      config.mean_drift_family_alpha != frozen.mean_drift_family_alpha ||
      config.mean_drift_threshold_rule != frozen.mean_drift_threshold_rule ||
      config.constant_relative_tolerance != frozen.constant_relative_tolerance ||
      config.selection_z_multiplier != frozen.selection_z_multiplier)
    throw std::invalid_argument("RPMD-JA frozen statistics configuration is invalid");
}

inline void observe_budget(const BudgetObserver& observer,const std::size_t index)
{
  if(observer&&(index&4095u)==0)observer();
}

inline std::size_t validate_channels(const TrainingChannels& channels,const BudgetObserver& observer={})
{
  if (channels.names.empty() || channels.names.size() != channels.values.size())
    throw std::invalid_argument("RPMD-JA statistics channel names and values do not match");
  const std::size_t count = channels.values.front().size();
  if (count < 2) throw std::invalid_argument("RPMD-JA statistics channels require at least two frames");
  for (std::size_t channel = 0; channel < channels.values.size(); ++channel) {
    if (channels.names[channel].empty() || channels.values[channel].size() != count)
      throw std::invalid_argument("RPMD-JA statistics channels must be named and have equal length");
    for (std::size_t frame=0;frame<count;++frame){
      observe_budget(observer,frame);
      if (!std::isfinite(channels.values[channel][frame])) throw std::invalid_argument("RPMD-JA statistics channel is non-finite");}
  }
  return count;
}

inline double mean(const std::vector<double>& values,const BudgetObserver& observer={})
{
  double total = 0.0;
  for(std::size_t index=0;index<values.size();++index){observe_budget(observer,index);total+=values[index];}
  return total / static_cast<double>(values.size());
}

inline double variance(const std::vector<double>& values, const double center,const BudgetObserver& observer={})
{
  double sum = 0.0;
  for (std::size_t index=0;index<values.size();++index) {
    observe_budget(observer,index);const double value=values[index];
    const double delta = value - center;
    sum += delta * delta;
  }
  return sum / static_cast<double>(values.size() - 1);
}

inline double prefix_variance(const std::vector<double>& values,const std::size_t frame_count,const double center,
  const BudgetObserver& observer={})
{
  if(frame_count<2||frame_count>values.size())throw std::invalid_argument("RPMD-JA statistics variance prefix is invalid");
  double sum=0.0;for(std::size_t frame=0;frame<frame_count;++frame){observe_budget(observer,frame);const double delta=values[frame]-center;sum+=delta*delta;}
  return sum/static_cast<double>(frame_count-1);
}

inline std::string channel_state(const std::vector<double>& values, const double center,
  const double value_variance, const double constant_tolerance,const std::size_t frame_count=0,
  const BudgetObserver& observer={})
{
  if (value_variance == 0.0) return center == 0.0 ? "CONSTANT_ZERO" : "CONSTANT_NONZERO";
  const std::size_t count=frame_count==0?values.size():frame_count;
  if(count==0||count>values.size())throw std::invalid_argument("RPMD-JA statistics channel-state prefix is invalid");
  double scale = 0.0, minimum = values[0], maximum = minimum;
  for (std::size_t frame=0;frame<count;++frame) {
    observe_budget(observer,frame);
    const double value=values[frame];
    scale = std::max(scale, std::abs(value));
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
  }
  return scale > 0.0 && (maximum - minimum) / scale <= constant_tolerance ? "NEAR_CONSTANT" : "VARYING";
}

inline double autocorrelation(const std::vector<double>& values, const double center,
  const double variance, const std::size_t lag,const BudgetObserver& observer={})
{
  double sum = 0.0;
  for (std::size_t frame = 0; frame + lag < values.size(); ++frame){
    observe_budget(observer,frame);
    sum += (values[frame] - center) * (values[frame + lag] - center);
  }
  return (sum / static_cast<double>(values.size() - lag)) / variance;
}

inline std::size_t common_block_coverage(const std::size_t frame_count,const std::size_t block_length)
{
  if(block_length==0||block_length>std::numeric_limits<std::size_t>::max()/2)return 0;
  const std::size_t quantum=2*block_length;
  return frame_count/quantum*quantum;
}

inline double prefix_mean(const std::vector<double>& values,const std::size_t frame_count,const BudgetObserver& observer={})
{
  if(frame_count==0||frame_count>values.size())throw std::invalid_argument("RPMD-JA statistics prefix is invalid");
  double total=0.0;for(std::size_t frame=0;frame<frame_count;++frame){observe_budget(observer,frame);total+=values[frame];}
  return total/static_cast<double>(frame_count);
}

inline double block_mean_se(const std::vector<double>& values, const std::size_t block_length,
  const std::size_t covered_frame_count,const BudgetObserver& observer={})
{
  if(block_length==0||covered_frame_count>values.size()||covered_frame_count%block_length!=0)
    throw std::invalid_argument("RPMD-JA statistics block coverage is invalid");
  const std::size_t count = covered_frame_count / block_length;
  if(count<2)throw std::invalid_argument("RPMD-JA statistics require at least two blocks");
  std::vector<double> means(count, 0.0);
  for (std::size_t block = 0; block < count; ++block) {
    double total = 0.0;
    for (std::size_t frame = block * block_length; frame < (block + 1) * block_length; ++frame)
      {observe_budget(observer,frame);total += values[frame];}
    means[block] = total / static_cast<double>(block_length);
  }
  const double center = mean(means,observer);
  return std::sqrt(variance(means, center,observer) / static_cast<double>(count));
}

inline std::vector<std::vector<double>> channel_block_means(
  const TrainingChannels& channels, const std::size_t block_length,const std::size_t covered_frame_count,
  const BudgetObserver& observer={})
{
  if(block_length==0||covered_frame_count>channels.values.front().size()||covered_frame_count%block_length!=0)
    throw std::invalid_argument("RPMD-JA statistics channel block coverage is invalid");
  const std::size_t count = covered_frame_count / block_length;
  std::vector<std::vector<double>> result(channels.values.size(), std::vector<double>(count, 0.0));
  for (std::size_t channel = 0; channel < channels.values.size(); ++channel) {
    for (std::size_t block = 0; block < count; ++block) {
      double total = 0.0;
      for (std::size_t frame = block * block_length; frame < (block + 1) * block_length; ++frame)
        {observe_budget(observer,frame);total += channels.values[channel][frame];}
      result[channel][block] = total / static_cast<double>(block_length);
    }
  }
  return result;
}

inline double z_score(const double value, const double standard_error)
{
  if (standard_error == 0.0) return value == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
  return std::abs(value) / standard_error;
}

inline double covariance(const std::vector<double>& left, const double left_mean,
  const std::vector<double>& right, const double right_mean,const BudgetObserver& observer={})
{
  double sum = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index){
    observe_budget(observer,index);
    sum += (left[index] - left_mean) * (right[index] - right_mean);
  }
  return sum / static_cast<double>(left.size() - 1);
}

inline PairedScaleSummary paired_resolution(const std::vector<double>& differences,
  const FrozenConfig& config)
{
  PairedScaleSummary result;
  if (differences.size() < config.min_blocks) return result;
  for (const double value : differences)
    if (!std::isfinite(value)) return result;
  result.mean_difference = mean(differences);
  result.standard_error = std::sqrt(variance(differences, result.mean_difference) /
    static_cast<double>(differences.size()));
  result.resolved = std::abs(result.mean_difference) > config.selection_z_multiplier * result.standard_error;
  if (result.resolved) result.winner = result.mean_difference < 0.0 ? "A" : "B";
  return result;
}
} // namespace detail

inline BlockPlan plan_training_blocks(const TrainingChannels& channels, const FrozenConfig& config,
  const BudgetObserver& observer={})
{
  // The Sokal window is a finite-data heuristic, not a universal PIMD guarantee; an unclosed window is inconclusive.
  detail::validate_config(config);
  BlockPlan result;
  const std::size_t count = detail::validate_channels(channels,observer);
  result.frame_count = count;
  result.max_lag = static_cast<std::size_t>(count * config.max_lag_fraction);
  if (result.max_lag == 0) {
    result.reason = "TOO_FEW_FRAMES_FOR_MAX_LAG";
    return result;
  }

  std::vector<std::size_t> varying_channels;
  for (std::size_t channel = 0; channel < channels.values.size(); ++channel) {
    ChannelBlockSummary summary;
    summary.name = channels.names[channel];
    summary.mean = detail::mean(channels.values[channel],observer);
    summary.variance = detail::variance(channels.values[channel], summary.mean,observer);
    if (summary.variance == 0.0) {
      summary.state = summary.mean == 0.0 ? "CONSTANT_ZERO" : "CONSTANT_NONZERO";
    } else {
      double scale = 0.0;
      double minimum = channels.values[channel][0], maximum = minimum;
      for (std::size_t frame=0;frame<channels.values[channel].size();++frame) {
        detail::observe_budget(observer,frame);const double value=channels.values[channel][frame];
        scale = std::max(scale, std::abs(value));
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
      summary.state = scale > 0.0 && (maximum - minimum) / scale <= config.constant_relative_tolerance ?
        "NEAR_CONSTANT" : "VARYING";
    }
    if (summary.state == "VARYING") varying_channels.push_back(channel);
    result.channels.push_back(summary);
  }
  if (varying_channels.empty()) {
    result.reason = "NO_VARYING_TRAINING_CHANNEL";
    return result;
  }

  std::vector<std::size_t> windows;
  for (const std::size_t channel : varying_channels) {
    double half_tau = 0.5;
    std::size_t window = 0;
    const double center = result.channels[channel].mean;
    const double variance = result.channels[channel].variance;
    for (std::size_t lag = 1; lag <= result.max_lag; ++lag) {
      const double rho = detail::autocorrelation(channels.values[channel], center, variance, lag,observer);
      if (!std::isfinite(rho)) break;
      half_tau += rho;
      const double tau = 2.0 * half_tau;
      if (tau > 0.0 && static_cast<double>(lag) >= config.window_c * tau) {
        window = lag;
        break;
      }
    }
    if (window == 0 || half_tau <= 0.0) {
      result.reason = "SOKAL_WINDOW_UNREACHED";
      return result;
    }
    const double tau = 2.0 * half_tau;
    if (static_cast<double>(count) < config.min_samples_per_tau * tau) {
      result.reason = "INSUFFICIENT_SAMPLES_PER_TAU";
      result.tau_frames = tau;
      return result;
    }
    result.tau_frames = std::isfinite(result.tau_frames) ? std::max(result.tau_frames, tau) : tau;
    result.channels[channel].tau_frames = tau;
    result.channels[channel].window_lag = window;
    windows.push_back(window);
  }
  result.window_lag_max = *std::max_element(windows.begin(), windows.end());
  result.block_length = static_cast<std::size_t>(std::ceil(config.block_factor * result.tau_frames));
  result.block_length = std::max<std::size_t>(1, result.block_length);
  result.adjacent_block_length = 2 * result.block_length;
  result.adjacent_block_count = count / result.adjacent_block_length;
  result.block_count = 2 * result.adjacent_block_count;
  result.common_covered_frame_count = result.block_count * result.block_length;
  result.discarded_tail_frames = count-result.common_covered_frame_count;
  if (result.block_count < config.min_blocks || result.adjacent_block_count < config.min_blocks) {
    result.reason = "TOO_FEW_ADJACENT_BLOCKS";
    return result;
  }

  result.max_adjacent_se_relative_change = 0.0;
  for (const std::size_t channel : varying_channels) {
    ChannelBlockSummary& summary = result.channels[channel];
    summary.se_block = detail::block_mean_se(channels.values[channel], result.block_length,result.common_covered_frame_count,observer);
    summary.se_adjacent_block = detail::block_mean_se(channels.values[channel], result.adjacent_block_length,result.common_covered_frame_count,observer);
    const double denominator = std::max(summary.se_block, summary.se_adjacent_block);
    summary.adjacent_se_relative_change = denominator == 0.0 ? 0.0 :
      std::abs(summary.se_adjacent_block - summary.se_block) / denominator;
    result.max_adjacent_se_relative_change = std::max(result.max_adjacent_se_relative_change,
      summary.adjacent_se_relative_change);
  }
  if (result.max_adjacent_se_relative_change > config.adjacent_se_relative_tolerance) {
    result.reason = "ADJACENT_BLOCK_SENSITIVE";
    return result;
  }
  result.status = "PASS";
  return result;
}

inline MeanDrift assess_fixed_r0_means(const TrainingChannels& training,
  const TrainingChannels& development, const BlockPlan& block_plan, const FrozenConfig& config,
  const std::vector<std::string>* expected_zero_mean_channels = nullptr,const BudgetObserver& observer={})
{
  detail::validate_config(config);
  const std::size_t train_frames = detail::validate_channels(training,observer);
  const std::size_t development_frames = detail::validate_channels(development,observer);
  if (training.names != development.names)
    throw std::invalid_argument("RPMD-JA fixed-R0 training/development inputs do not match");
  std::vector<std::string> zero_mean_names = expected_zero_mean_channels == nullptr ?
    training.names : *expected_zero_mean_channels;
  for (const std::string& name : zero_mean_names)
    if (std::find(training.names.begin(), training.names.end(), name) == training.names.end())
      throw std::invalid_argument("RPMD-JA expected-zero mean channel is not declared");
  MeanDrift result;
  if (block_plan.status != "PASS" || block_plan.block_length == 0 ||
      block_plan.adjacent_block_length != 2 * block_plan.block_length) {
    result.reason = "TRAINING_BLOCK_PLAN_UNRESOLVED";
    return result;
  }
  result.tau_frames = block_plan.tau_frames;
  result.dimension_count = training.names.size();
  const std::size_t dimensions = result.dimension_count;
  result.block_length = block_plan.block_length;
  result.adjacent_block_length = block_plan.adjacent_block_length;
  result.training_covered_frame_count=detail::common_block_coverage(train_frames,result.block_length);
  result.development_covered_frame_count=detail::common_block_coverage(development_frames,result.block_length);
  result.training_discarded_tail_frames=train_frames-result.training_covered_frame_count;
  result.development_discarded_tail_frames=development_frames-result.development_covered_frame_count;
  if(result.training_covered_frame_count==0||result.development_covered_frame_count==0){
    result.reason="NO_COMMON_B_AND_2B_FRAME_COVERAGE";return result;}
  result.effective_frames_train = static_cast<double>(result.training_covered_frame_count) / result.tau_frames;
  result.effective_frames_development = static_cast<double>(result.development_covered_frame_count) / result.tau_frames;
  result.z_tolerance = config.mean_drift_z_tolerance;
  result.mean_train.resize(dimensions);
  result.mean_development.resize(dimensions);
  result.mean_difference.resize(dimensions);
  for (std::size_t i = 0; i < dimensions; ++i) {
    result.mean_train[i] = detail::mean(training.values[i],observer);
    result.mean_development[i] = detail::mean(development.values[i],observer);
    result.mean_difference[i] = result.mean_development[i] - result.mean_train[i];
  }

  const std::size_t lengths[2] = {result.block_length, result.adjacent_block_length};
  double maximum_z = 0.0;
  double maximum_development_se_change = 0.0;
  for (int scale = 0; scale < 2; ++scale) {
    const std::size_t length = lengths[scale];
    const std::size_t train_block_count = result.training_covered_frame_count / length;
    const std::size_t development_block_count = result.development_covered_frame_count / length;
    if (train_block_count < config.min_blocks || development_block_count < config.min_blocks) {
      result.reason = "TOO_FEW_TRAINING_OR_DEVELOPMENT_BLOCKS";
      return result;
    }
    const std::vector<std::vector<double>> train_blocks = detail::channel_block_means(training, length,result.training_covered_frame_count,observer);
    const std::vector<std::vector<double>> development_blocks = detail::channel_block_means(development, length,result.development_covered_frame_count,observer);
    std::vector<double> train_block_means(dimensions, 0.0), development_block_means(dimensions, 0.0);
    std::vector<std::vector<double>> covariance_train(dimensions, std::vector<double>(dimensions, 0.0));
    std::vector<std::vector<double>> covariance_development(dimensions, std::vector<double>(dimensions, 0.0));
    std::vector<std::vector<double>> covariance_difference(dimensions, std::vector<double>(dimensions, 0.0));
    for (std::size_t i = 0; i < dimensions; ++i) {
      train_block_means[i] = detail::mean(train_blocks[i],observer);
      development_block_means[i] = detail::mean(development_blocks[i],observer);
    }
    for (std::size_t i = 0; i < dimensions; ++i) {
      for (std::size_t j = 0; j < dimensions; ++j) {
        const double train_cov = detail::covariance(train_blocks[i], train_block_means[i],
          train_blocks[j], train_block_means[j],observer) / static_cast<double>(train_block_count);
        const double development_cov = detail::covariance(development_blocks[i], development_block_means[i],
          development_blocks[j], development_block_means[j],observer) / static_cast<double>(development_block_count);
        covariance_train[i][j] = train_cov;
        covariance_development[i][j] = development_cov;
        covariance_difference[i][j] = train_cov + development_cov;
      }
    }
    std::vector<ChannelDriftSummary>& summaries = scale == 0 ? result.channels : result.adjacent_channels;
    summaries.reserve(dimensions);
    for (std::size_t i = 0; i < dimensions; ++i) {
      ChannelDriftSummary summary;
      summary.name = training.names[i];
      summary.mean_train = result.mean_train[i];
      summary.mean_development = result.mean_development[i];
      summary.mean_difference = result.mean_difference[i];
      summary.se_train = std::sqrt(covariance_train[i][i]);
      summary.se_development = std::sqrt(covariance_development[i][i]);
      summary.se_difference = std::sqrt(covariance_difference[i][i]);
      const bool expected_zero_mean = std::find(zero_mean_names.begin(), zero_mean_names.end(), summary.name) !=
        zero_mean_names.end();
      summary.train_z = expected_zero_mean ? detail::z_score(summary.mean_train, summary.se_train) : 0.0;
      summary.development_z = expected_zero_mean ? detail::z_score(summary.mean_development, summary.se_development) : 0.0;
      summary.delta_z = detail::z_score(summary.mean_difference, summary.se_difference);
      const std::string train_state = detail::channel_state(training.values[i], result.mean_train[i],
        detail::variance(training.values[i],result.mean_train[i],observer),config.constant_relative_tolerance,
        training.values[i].size(),observer);
      const std::string development_state = detail::channel_state(development.values[i], result.mean_development[i],
        detail::variance(development.values[i],result.mean_development[i],observer),config.constant_relative_tolerance,
        development.values[i].size(),observer);
      if (train_state == "CONSTANT_ZERO" && development_state == "CONSTANT_ZERO") summary.state = "CONSTANT_ZERO";
      else if (train_state == "CONSTANT_NONZERO" && development_state == "CONSTANT_NONZERO") summary.state = "CONSTANT_NONZERO";
      else if ((train_state == "CONSTANT_ZERO" || train_state == "CONSTANT_NONZERO") &&
               (development_state == "CONSTANT_ZERO" || development_state == "CONSTANT_NONZERO"))
        summary.state = "CONSTANT_CHANGED";
      else if (train_state == "NEAR_CONSTANT" || development_state == "NEAR_CONSTANT")
        summary.state = "NEAR_CONSTANT";
      else summary.state = "VARYING";
      maximum_z = std::max(maximum_z,
        std::max(summary.train_z, std::max(summary.development_z, summary.delta_z)));
      summaries.push_back(summary);
    }
    if (scale == 0) {
      result.covariance_of_mean_train = covariance_train;
      result.covariance_of_mean_development = covariance_development;
      result.covariance_of_mean_difference = covariance_difference;
    } else {
      result.adjacent_covariance_of_mean_train = covariance_train;
      result.adjacent_covariance_of_mean_development = covariance_development;
      result.adjacent_covariance_of_mean_difference = covariance_difference;
    }
  }
  // Frozen max-z correction is an engineering screen; estimated PIMD block SEs do not prove sub-Gaussian coverage.
  const std::size_t comparison_count = 2 * (dimensions + 2 * zero_mean_names.size());
  result.comparison_count = comparison_count;
  result.family_alpha = config.mean_drift_family_alpha;
  result.z_tolerance = std::max(config.mean_drift_z_tolerance,
    std::sqrt(2.0 * std::log(2.0 * static_cast<double>(comparison_count) / config.mean_drift_family_alpha)));
  for (std::size_t i = 0; i < dimensions; ++i) {
    const double base = result.channels[i].se_development;
    const double adjacent = result.adjacent_channels[i].se_development;
    const double denominator = std::max(base, adjacent);
    const double change = denominator == 0.0 ? 0.0 : std::abs(adjacent - base) / denominator;
    maximum_development_se_change = std::max(maximum_development_se_change, change);
  }
  result.maximum_z = maximum_z;
  result.development_adjacent_se_relative_change = maximum_development_se_change;
  if (maximum_development_se_change > config.adjacent_se_relative_tolerance) {
    result.reason = "DEVELOPMENT_BLOCK_SENSITIVE";
  } else if (maximum_z > result.z_tolerance) {
    result.reason = "FIXED_R0_MEAN_OR_DRIFT";
  } else {
    result.status = "PASS";
  }
  return result;
}

inline PairedChoice assess_paired_choice(const std::vector<double>& score_a_minus_b_block,
  const std::vector<double>& score_a_minus_b_adjacent, const FrozenConfig& config)
{
  detail::validate_config(config);
  PairedChoice result;
  result.block = detail::paired_resolution(score_a_minus_b_block, config);
  result.adjacent_block = detail::paired_resolution(score_a_minus_b_adjacent, config);
  if (result.block.resolved && result.adjacent_block.resolved &&
      result.block.winner == result.adjacent_block.winner)
    result.status = result.block.winner + "_WINS";
  return result;
}
} // namespace rpmd_ja_statistics_policy
