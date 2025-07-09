#include <sys/wait.h>

#include <optional>
#include <random>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "barrier.h"

namespace {
void TestConstructor() {
  int pid = fork();
  CHECK(pid >= 0) << "Fork failed: " << strerror(errno);

  if (pid == 0) {
    SenseReversingBarrier barrier(2, "/TestBarrier");
    return;
  } else {
    SenseReversingBarrier barrier(2, "/TestBarrier");
    waitpid(pid, nullptr, 0);
    return;
  }
}

void WaitWithoutSleep(int num_processes, int num_iterations) {
  constexpr double MAX_WAIT_MS = 100.0;
  SenseReversingBarrier barrier(num_processes, "/TestBarrier");

  for (int j = 0; j < num_iterations; ++j) {
    LOG(INFO) << "Waiting at barrier iteration " << j;
    barrier.Wait();
    LOG(INFO) << "Passed barrier iteration " << j;
  }
}

void WaitWithRandomSleep(int num_processes, int num_iterations) {
  constexpr double MAX_WAIT_MS = 100.0;
  SenseReversingBarrier barrier(num_processes, "/TestBarrier");

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, MAX_WAIT_MS);

  for (int j = 0; j < num_iterations; ++j) {
    LOG(INFO) << "Waiting at barrier iteration " << j;
    barrier.Wait();
    double sleep_ms = dis(gen);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(static_cast<int>(sleep_ms)));
    LOG(INFO) << "Passed barrier iteration " << j;
  }
}

void TestWaitWithoutSleep(int num_processes, int num_iterations) {
  std::vector<int> pids;

  for (int i = 0; i < num_processes - 1; ++i) {
    int pid = fork();
    CHECK(pid >= 0) << "Fork failed: " << strerror(errno);

    if (pid == 0) {
      WaitWithoutSleep(num_processes, num_iterations);
      return;
    } else {
      pids.push_back(pid);
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  WaitWithoutSleep(num_processes, num_iterations);
  auto end_time = std::chrono::high_resolution_clock::now();

  for (int child_pid : pids) {
    waitpid(child_pid, nullptr, 0);
  }
  double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_time - start_time)
                        .count() /
                    10e6;

  LOG(INFO) << "Wait time: " << duration / num_iterations
            << " ms per iteration.";
  LOG(INFO) << "Wait time: " << duration / num_iterations / num_processes
            << " ms per iteration per process.";
  return;
}

void TestWaitWithRandomSleep(int num_processes, int num_iterations) {
  std::vector<int> pids;

  for (int i = 0; i < num_processes - 1; ++i) {
    int pid = fork();
    CHECK(pid >= 0) << "Fork failed: " << strerror(errno);

    if (pid == 0) {
      WaitWithRandomSleep(num_processes, num_iterations);
      return;
    } else {
      pids.push_back(pid);
    }
  }

  WaitWithRandomSleep(num_processes, num_iterations);

  for (int child_pid : pids) {
    waitpid(child_pid, nullptr, 0);
  }
  return;
}

} // namespace

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");
ABSL_FLAG(std::string, test_type, "constructor",
          "Type of test to run. Available types: constructor, "
          "wait_with_random_sleep");
ABSL_FLAG(int, num_processes, 2,
          "Number of processes to use in the wait test.");
ABSL_FLAG(int, num_iterations, 20,
          "Number of iterations to run in the wait test.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Sense Reversing Barrier Test");
  absl::ParseCommandLine(argc, argv);
  const std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }
  const std::string test_type = absl::GetFlag(FLAGS_test_type);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  if (test_type == "constructor") {
    TestConstructor();
  } else if (test_type == "wait_with_random_sleep") {
    int num_processes = absl::GetFlag(FLAGS_num_processes);
    int num_iterations = absl::GetFlag(FLAGS_num_iterations);
    TestWaitWithRandomSleep(num_processes, num_iterations);
  } else if (test_type == "wait_without_sleep") {
    int num_processes = absl::GetFlag(FLAGS_num_processes);
    int num_iterations = absl::GetFlag(FLAGS_num_iterations);
    TestWaitWithoutSleep(num_processes, num_iterations);
  } else {
    LOG(ERROR) << "Unknown test type: " << test_type
               << ". Available types: constructor, wait";
    return 1;
  }

  return 0;
}
