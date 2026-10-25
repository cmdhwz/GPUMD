#pragma once

#include "rpmd_ja_sha256.h"
#include "rpmd_ja_utf8_file_io.h"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#ifdef GPUMD_WPE_ENABLED
#include "wisevolve_potential_api.h"
#endif
#include "utilities/compact_nep.cuh"
#include "utilities/run_input.cuh"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

namespace rpmd_ja_sample_provenance
{
struct ConfigurationSnapshot
{
  std::string run_input_sha256;
  std::string physical_force_configuration_sha256;
  std::string integrator_configuration_sha256;
  std::string active_configuration_scope;
  unsigned active_run_ordinal=0;
  bool active_configuration_identified=false;
  std::string model_sha256;
  std::string atom_signature_sha256;
  std::string thermostat_tokens_json;
  std::string model_path;
};

struct SourceIdentity
{
  std::string job_id;
  std::string job_id_status;
  std::string run_id;
};

inline std::string sha256_bytes(const unsigned char* bytes, const std::size_t count)
{
  rpmd_ja_sha256::Digest digest;
  digest.update(bytes, count);
  return digest.finish_hex();
}

inline std::string sha256_file(const std::string& path)
{
  if (path.empty()) return std::string();
  rpmd_ja_file_io::ifstream input(path, std::ios::binary);
  if (!input) return std::string();
  rpmd_ja_sha256::Digest digest;
  char buffer[65536];
  while (input) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    if (count > 0) digest.update(reinterpret_cast<const unsigned char*>(buffer), static_cast<std::size_t>(count));
  }
  if (input.bad()) return std::string();
  return digest.finish_hex();
}

inline std::string sha256_text(const std::string& text)
{
  return sha256_bytes(reinterpret_cast<const unsigned char*>(text.data()), text.size());
}

inline std::string canonical_token_lines(const std::vector<RunInputLine>& lines)
{
  std::ostringstream output;
  for (const RunInputLine& line : lines) {
    output << line.tokens.size() << ':';
    for (const std::string& token : line.tokens) output << token.size() << ':' << token;
    output << ';';
  }
  return output.str();
}

inline std::string canonical_token_lines_for_configuration(const std::vector<RunInputLine>& lines,
  const bool physical_force)
{
  std::ostringstream output;
  for (const RunInputLine& line : lines) {
    if (line.tokens.empty()) continue;
    const std::string& keyword = line.tokens[0];
    const bool selected = physical_force ?
      (keyword == "potential" || keyword == "pppm_mesh_spacing" || keyword == "kspace" || keyword == "dftd3") :
      (keyword == "ensemble" || keyword == "time_step" || keyword.compare(0, 5, "pimd_") == 0);
    if (!selected) continue;
    output << line.tokens.size() << ':';
    for (const std::string& token : line.tokens) output << token.size() << ':' << token;
    output << ';';
  }
  return output.str();
}

inline bool active_run_configuration_lines(const std::vector<RunInputLine>& lines,const int number_of_steps,
  const unsigned run_ordinal,std::vector<RunInputLine>& active_lines)
{
  unsigned ordinal=0;
  for(const RunInputLine& line:lines){
    if(!line.tokens.empty()&&line.tokens[0]=="run"){
      if(ordinal==run_ordinal){
        if(line.tokens.size()<2)return false;
        try{std::size_t used=0;const int steps=std::stoi(line.tokens[1],&used);
          if(used!=line.tokens[1].size()||steps!=number_of_steps)return false;}
        catch(...){return false;}
        return true;
      }
      ++ordinal;
    }
    active_lines.push_back(line);
  }
  return false;
}

inline std::string json_escape(const std::string& value)
{
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << static_cast<unsigned int>(character) << std::dec;
        else output << static_cast<char>(character);
    }
  }
  return output.str();
}

inline std::string json_token_lines(const std::vector<RunInputLine>& lines, const bool thermostat_only)
{
  std::ostringstream output;
  output << '[';
  bool first_line = true;
  for (const RunInputLine& line : lines) {
    if (line.tokens.empty()) continue;
    const std::string& keyword = line.tokens[0];
    if (thermostat_only && keyword != "ensemble" && keyword.compare(0, 5, "pimd_") != 0) continue;
    if (!first_line) output << ',';
    first_line = false;
    output << '[';
    for (std::size_t token = 0; token < line.tokens.size(); ++token) {
      if (token != 0) output << ',';
      output << '"' << json_escape(line.tokens[token]) << '"';
    }
    output << ']';
  }
  output << ']';
  return output.str();
}

inline std::vector<RunInputLine> effective_thermostat_configuration_lines(
  const std::vector<RunInputLine>& lines)
{
  std::vector<RunInputLine> effective;
  for (auto line = lines.rbegin(); line != lines.rend(); ++line) {
    if (line->tokens.empty()) continue;
    const std::string& keyword = line->tokens[0];
    if (keyword != "ensemble" && keyword.compare(0, 5, "pimd_") != 0) continue;
    const bool already_seen = std::any_of(
      effective.begin(), effective.end(), [&keyword](const RunInputLine& selected) {
        return !selected.tokens.empty() && selected.tokens[0] == keyword;
      });
    if (!already_seen) effective.push_back(*line);
  }
  std::reverse(effective.begin(), effective.end());
  return effective;
}

inline std::string atom_signature(const Atom& atom, const Box& box)
{
  if (atom.cpu_type.size() != atom.cpu_mass.size()) return std::string();
  std::ostringstream signature;
  signature.imbue(std::locale::classic());
  signature << std::setprecision(std::numeric_limits<double>::max_digits10);
  signature << "atoms=" << atom.cpu_type.size() << ';';
  for (std::size_t atom_id = 0; atom_id < atom.cpu_type.size(); ++atom_id)
    signature << atom.cpu_type[atom_id] << ':' << atom.cpu_mass[atom_id] << ';';
  signature << "cell=";
  for (int component = 0; component < 9; ++component) signature << box.cpu_h[component] << ';';
  return sha256_text(signature.str());
}

inline ConfigurationSnapshot snapshot(const Force& force,const Atom& atom,const Box& box,
  const int number_of_steps,const unsigned run_ordinal,const double time_step,const double temperature,
  const int beads,const bool internal_mass_com)
{
  ConfigurationSnapshot result;
  const RunInput& input = force.get_run_input();
  const std::vector<RunInputLine>& lines = input.lines();
  const std::string canonical = canonical_token_lines(lines);
  result.run_input_sha256 = sha256_text(canonical);
  std::vector<RunInputLine> active_lines;
  const bool active_run_identified =
    active_run_configuration_lines(lines, number_of_steps, run_ordinal, active_lines);
  result.active_run_ordinal=run_ordinal;
  std::ostringstream integrator_state;integrator_state.imbue(std::locale::classic());
  integrator_state<<std::setprecision(std::numeric_limits<double>::max_digits10)
    <<"time_step="<<time_step<<";temperature="<<temperature
    <<";beads="<<beads<<";internal_mass_com="<<(internal_mass_com?1:0)<<';';
  result.physical_force_configuration_sha256 = active_run_identified?
    sha256_text(canonical_token_lines_for_configuration(active_lines,true)):std::string();
  result.integrator_configuration_sha256 = active_run_identified?
    sha256_text(canonical_token_lines_for_configuration(active_lines,false)+integrator_state.str()):std::string();
  result.model_path = force.primary_nep_model_path();
  result.active_configuration_identified = active_run_identified;
#ifdef GPUMD_WPE_ENABLED
  if (result.model_path.empty()) result.model_path = get_first_potential_filename(input);
  const Force::WpeStageSnapshot& stage = force.get_wpe_stage_snapshot();
  if (!stage.valid || stage.stage_kind != WPE_STAGE_RUN ||
      stage.number_of_atoms != atom.number_of_atoms ||
      stage.pimd_beads != static_cast<std::uint32_t>(beads) ||
      (stage.facts & WPE_STAGE_PIMD_ECO_CONFIGURATION) != 0) {
    result.active_configuration_identified = false;
  } else if (active_run_identified) {
    std::ostringstream stage_configuration;
    stage_configuration.imbue(std::locale::classic());
    stage_configuration << std::setprecision(std::numeric_limits<double>::max_digits10)
      << ";wpe_stage_kind=" << stage.stage_kind
      << ";wpe_facts=" << stage.facts
      << ";wpe_number_of_atoms=" << stage.number_of_atoms
      << ";wpe_pimd_beads=" << stage.pimd_beads
      << ";wpe_qnep_kspace_method=" << stage.qnep_kspace_method
      << ";wpe_dftd3_enabled=" << stage.dftd3_enabled
      << ";wpe_dftd3_functional=" << stage.dftd3_functional
      << ";wpe_dftd3_rc_potential=" << stage.dftd3_rc_potential
      << ";wpe_dftd3_rc_coordination_number=" << stage.dftd3_rc_coordination_number
      << ";wpe_qnep_pppm_spacing=" << stage.qnep_pppm_spacing;
    result.physical_force_configuration_sha256 = sha256_text(
      result.physical_force_configuration_sha256 + stage_configuration.str());
    result.integrator_configuration_sha256 = sha256_text(
      result.integrator_configuration_sha256 + stage_configuration.str());
  }
#endif
  result.model_sha256 = sha256_file(result.model_path);
#ifdef GPUMD_WPE_ENABLED
  if (result.model_path.empty() || result.model_sha256.empty())
    result.active_configuration_identified = false;
#endif
  result.active_configuration_scope=result.active_configuration_identified?"PREFIX_BEFORE_ACTIVE_RUN":"ACTIVE_RUN_NOT_IDENTIFIED";
  result.atom_signature_sha256 = atom_signature(atom, box);
  result.thermostat_tokens_json = json_token_lines(
    effective_thermostat_configuration_lines(active_lines), true);
  return result;
}

inline SourceIdentity source_identity()
{
  static const char* job_variables[] = {
    "SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID", "CI_JOB_ID", "AZ_BATCH_JOB_ID"};
  SourceIdentity result;
  for (const char* variable : job_variables) {
    const char* value = std::getenv(variable);
    if (value != nullptr && value[0] != '\0') {
      result.job_id = std::string(variable) + ':' + value;
      result.job_id_status = "SCHEDULER_DECLARED";
      break;
    }
  }
  const std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
  std::ostringstream run_id;
  run_id << "run-local-unverified-" << now.time_since_epoch().count();
  result.run_id = run_id.str();
  if (result.job_id.empty()) {
    result.job_id = result.run_id;
    result.job_id_status = "RUN_LOCAL_UNVERIFIED";
  }
  return result;
}
} // namespace rpmd_ja_sample_provenance
