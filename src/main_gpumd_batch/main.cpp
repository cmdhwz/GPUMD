/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
*/

/*----------------------------------------------------------------------------80
Run independent NVE GPUMD simulations in a bounded number of child processes.
Each child runs in its sample directory, so model.xyz and all GPUMD outputs are
kept separate and qNEP/HAC state is isolated per trajectory.
------------------------------------------------------------------------------*/

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>

static const char path_separator = '/';

static std::string join_path(const std::string& directory, const std::string& name)
{
  if (directory.empty()) return name;
  return directory + path_separator + name;
}

static bool file_exists(const std::string& filename)
{
  std::ifstream input(filename, std::ios::binary);
  return input.good();
}

static bool read_file(const std::string& filename, std::string& contents)
{
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open()) return false;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (input.bad()) return false;
  contents = buffer.str();
  return true;
}

static bool file_equals(const std::string& filename, const std::string& expected)
{
  std::string actual;
  return read_file(filename, actual) && actual == expected;
}

static bool write_file(const std::string& filename, const std::string& contents)
{
  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  if (!output.is_open()) return false;
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  output.close();
  if (!output) {
    std::remove(filename.c_str());
    return false;
  }
  return true;
}

static bool parse_integer(const char* text, long long& value)
{
  if (text == nullptr || *text == '\0') return false;
  char* end = nullptr;
  errno = 0;
  const long long parsed = std::strtoll(text, &end, 10);
  if (errno == ERANGE || end == text || *end != '\0') return false;
  value = parsed;
  return true;
}

static bool validate_marker_name(const std::string& name)
{
  return !name.empty() && name != "." && name != ".." &&
         name != "run.in" && name != "model.xyz" &&
         name.find_first_of("/\\:") == std::string::npos;
}

static bool inspect_marker(const std::string& filename, bool& exists, bool& valid)
{
  exists = false;
  valid = false;
  struct stat information{};
  if (lstat(filename.c_str(), &information) != 0) {
    return errno == ENOENT;
  }
  exists = true;
  valid = S_ISREG(information.st_mode) && information.st_size == 0;
  return true;
}

static bool validate_nve_input(const std::string& contents)
{
  std::istringstream input(contents);
  std::string line;
  while (std::getline(input, line)) {
    const size_t comment = line.find('#');
    if (comment != std::string::npos) line.resize(comment);
    std::istringstream tokens(line);
    std::string keyword;
    tokens >> keyword;
    if (keyword != "ensemble") continue;

    std::string ensemble;
    tokens >> ensemble;
    if (ensemble != "nve") {
      std::cerr << "gpumd_batch: only ensemble nve is supported; found ensemble "
                << (ensemble.empty() ? "<missing>" : ensemble) << "." << std::endl;
      return false;
    }
  }
  return true;
}

static std::string current_directory()
{
  char buffer[PATH_MAX];
  return getcwd(buffer, sizeof(buffer)) == nullptr ? std::string() : std::string(buffer);
}

static bool is_absolute_path(const std::string& path)
{
  return !path.empty() && path[0] == '/';
}

static std::string absolute_path(const std::string& path)
{
  if (is_absolute_path(path)) return path;
  const std::string directory = current_directory();
  return directory.empty() ? path : join_path(directory, path);
}

static std::string gpumd_executable(const char* argv0)
{
  char buffer[PATH_MAX];
  const ssize_t length = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
  if (length > 0) {
    buffer[length] = '\0';
    const std::string self(buffer);
    const size_t separator = self.find_last_of('/');
    return (separator == std::string::npos ? std::string() : self.substr(0, separator + 1)) +
           "gpumd";
  }
  if (argv0 != nullptr && std::strchr(argv0, '/') != nullptr) {
    const std::string self = absolute_path(argv0);
    const size_t separator = self.find_last_of('/');
    return (separator == std::string::npos ? std::string() : self.substr(0, separator + 1)) +
           "gpumd";
  }
  return "gpumd";
}

static int run_child(const std::string& executable, const std::string& directory)
{
  const pid_t child = fork();
  if (child < 0) return 127;
  if (child == 0) {
    if (chdir(directory.c_str()) != 0) _exit(126);
    if (executable.find('/') != std::string::npos) {
      execl(executable.c_str(), executable.c_str(), static_cast<char*>(nullptr));
    } else {
      execlp(executable.c_str(), executable.c_str(), static_cast<char*>(nullptr));
    }
    _exit(127);
  }

  int status = 1;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) return 127;
  }
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
  return 127;
}

static void report(std::mutex& output_mutex, const std::string& message)
{
  std::lock_guard<std::mutex> lock(output_mutex);
  std::cout << message << std::endl;
}

static bool process_sample(
  const std::string& executable,
  const std::string& run_input_contents,
  const std::string& sample_directory,
  const std::string& marker_name,
  std::mutex& output_mutex)
{
  const std::string marker_path = join_path(sample_directory, marker_name);
  bool marker_exists = false;
  bool marker_valid = false;
  if (!inspect_marker(marker_path, marker_exists, marker_valid)) {
    report(output_mutex, "gpumd_batch: failed to inspect " + marker_path + ".");
    return false;
  }
  if (marker_exists) {
    if (!marker_valid) {
      report(output_mutex, "gpumd_batch: invalid completion marker " + marker_path +
                            "; expected a zero-byte regular file.");
      return false;
    }
    report(output_mutex, "gpumd_batch: skip " + sample_directory + " (" + marker_name + ").");
    return true;
  }

  if (!file_exists(join_path(sample_directory, "model.xyz"))) {
    report(output_mutex, "gpumd_batch: missing " + join_path(sample_directory, "model.xyz") + ".");
    return false;
  }

  const std::string sample_run_input = join_path(sample_directory, "run.in");
  bool copied_run_input = false;
  if (file_exists(sample_run_input)) {
    if (!file_equals(sample_run_input, run_input_contents)) {
      report(output_mutex, "gpumd_batch: " + sample_run_input +
                            " differs from the parent run.in; refusing to overwrite it.");
      return false;
    }
  } else {
    if (!write_file(sample_run_input, run_input_contents)) {
      report(output_mutex, "gpumd_batch: failed to prepare " + sample_run_input + ".");
      return false;
    }
    copied_run_input = true;
  }

  const int exit_code = run_child(executable, absolute_path(sample_directory));
  const bool cleaned = !copied_run_input || std::remove(sample_run_input.c_str()) == 0;
  if (exit_code != 0) {
    report(output_mutex, "gpumd_batch: " + sample_directory +
                          " failed with exit code " + std::to_string(exit_code) + ".");
    return false;
  }
  if (!cleaned) {
    report(output_mutex, "gpumd_batch: completed " + sample_directory +
                          ", but could not remove the temporary run.in; no marker written.");
    return false;
  }

  bool marker_created_during_run = false;
  bool marker_valid_after_run = false;
  if (!inspect_marker(marker_path, marker_created_during_run, marker_valid_after_run)) {
    report(output_mutex, "gpumd_batch: failed to inspect " + marker_path + " after the run.");
    return false;
  }
  if (marker_created_during_run) {
    const char* marker_type = marker_valid_after_run ? "completion marker" : "output file";
    report(output_mutex, "gpumd_batch: " + marker_path +
                          " was created during the run as a " + marker_type +
                          "; refusing to overwrite it.");
    return false;
  }

  const int marker_fd = open(marker_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
  if (marker_fd < 0) {
    report(output_mutex, "gpumd_batch: completed " + sample_directory +
                          ", but failed to create " + marker_name + ".");
    return false;
  }
  if (close(marker_fd) != 0) {
    std::remove(marker_path.c_str());
    report(output_mutex, "gpumd_batch: completed " + sample_directory +
                          ", but failed to finalize " + marker_name + ".");
    return false;
  }
  report(output_mutex, "gpumd_batch: completed " + sample_directory + ".");
  return true;
}

int main(int argc, char* argv[])
{
  const char* default_marker = "COMPLETED";
  if (argc != 5 && argc != 6) {
    std::cerr << "Usage: gpumd_batch <folder_prefix> <first> <last> <parallel> [marker]\n"
              << "       marker defaults to " << default_marker << "." << std::endl;
    return EXIT_FAILURE;
  }

  const std::string prefix(argv[1]);
  const std::string marker_name = argc == 6 ? argv[5] : default_marker;
  if (!validate_marker_name(prefix) || !validate_marker_name(marker_name)) {
    std::cerr << "gpumd_batch: folder prefix and marker must be direct-child names without "
                 "path separators."
              << std::endl;
    return EXIT_FAILURE;
  }

  long long first = 0;
  long long last = 0;
  long long parallel = 0;
  if (!parse_integer(argv[2], first) || !parse_integer(argv[3], last) ||
      !parse_integer(argv[4], parallel) || first < 0 || last < first || last == LLONG_MAX ||
      parallel <= 0) {
    std::cerr << "gpumd_batch: first/last must define a non-negative range and parallel must be "
                 "positive."
              << std::endl;
    return EXIT_FAILURE;
  }

  const std::string run_input("run.in");
  std::string run_input_contents;
  if (!read_file(run_input, run_input_contents)) {
    std::cerr << "gpumd_batch: failed to read " << run_input << "." << std::endl;
    return EXIT_FAILURE;
  }
  if (!validate_nve_input(run_input_contents)) return EXIT_FAILURE;

  const long long total = last - first + 1;
  const long long worker_count = std::min(parallel, total);
  const std::string executable = gpumd_executable(argv[0]);
  std::cout << "gpumd_batch: running " << total << " NVE sample(s) with up to " << worker_count
            << " concurrent gpumd process(es)." << std::endl;

  std::atomic<long long> next_sample(first);
  std::atomic<bool> failed(false);
  std::mutex output_mutex;
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(worker_count));
  for (long long worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        const long long index = next_sample.fetch_add(1);
        if (index > last) return;
        const std::string sample_directory = prefix + std::to_string(index);
        if (!process_sample(
              executable, run_input_contents, sample_directory, marker_name, output_mutex)) {
          failed.store(true);
        }
      }
    });
  }

  for (auto& worker : workers) worker.join();
  return failed.load() ? EXIT_FAILURE : EXIT_SUCCESS;
}
