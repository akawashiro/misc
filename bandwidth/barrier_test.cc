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

void WaitWithSleep(int num_processes, int num_iterations) {
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

void TestWait(int num_processes, int num_iterations) {
  std::vector<int> pids;

  for (int i = 0; i < num_processes - 1; ++i) {
    int pid = fork();
    CHECK(pid >= 0) << "Fork failed: " << strerror(errno);

    if (pid == 0) {
      WaitWithSleep(num_processes, num_iterations);
      return;
    } else {
      pids.push_back(pid);
    }
  }

  WaitWithSleep(num_processes, num_iterations);

  for (int child_pid : pids) {
    waitpid(child_pid, nullptr, 0);
  }
  return;
}

} // namespace

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");
ABSL_FLAG(std::string, test_type, "constructor",
          "Type of test to run. Available types: constructor, wait");
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
  } else if (test_type == "wait") {
    int num_processes = absl::GetFlag(FLAGS_num_processes);
    int num_iterations = absl::GetFlag(FLAGS_num_iterations);
    TestWait(num_processes, num_iterations);
  } else {
    LOG(ERROR) << "Unknown test type: " << test_type
               << ". Available types: constructor, wait";
    return 1;
  }

  return 0;
}
