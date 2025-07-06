#include <algorithm>
#include <chrono>
#include <optional>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("memcpy");
  absl::ParseCommandLine(argc, argv);

  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  std::vector<uint8_t> src = generateDataToSend();
  std::vector<uint8_t> dst(DATA_SIZE, 0);
  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS;
       ++iteration) {
    std::fill(dst.begin(), dst.end(), 0);
    const auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst.data(), src.data(), DATA_SIZE);
    const auto end = std::chrono::high_resolution_clock::now();

    CHECK(verifyDataReceived(src)) << "Data verification failed before memcpy.";
    if (NUM_WARMUPS <= iteration) {
      const double duration =
          std::chrono::duration<double>(end - start).count();
      durations.push_back(duration);
    }
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec";
}
