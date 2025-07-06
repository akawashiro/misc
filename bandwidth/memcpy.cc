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

ABSL_FLAG(int, num_iterations, 10,
          "Number of measurement iterations (minimum 3)");
ABSL_FLAG(int, num_warmups, 3, "Number of warmup iterations");
ABSL_FLAG(uint64_t, data_size, 128 * (1 << 20),
          "Size of data to transfer in bytes");
ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("memcpy");
  absl::ParseCommandLine(argc, argv);

  // Get values from command line flags
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  int num_warmups = absl::GetFlag(FLAGS_num_warmups);
  uint64_t data_size = absl::GetFlag(FLAGS_data_size);

  // Validate num_iterations
  if (num_iterations < 3) {
    LOG(ERROR) << "num_iterations must be at least 3, got: " << num_iterations;
    return 1;
  }

  // Validate data_size
  if (data_size <= CHECKSUM_SIZE) {
    LOG(ERROR) << "data_size must be larger than CHECKSUM_SIZE ("
               << CHECKSUM_SIZE << "), got: " << data_size;
    return 1;
  }

  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  std::vector<uint8_t> src = generateDataToSend(data_size);
  std::vector<uint8_t> dst(data_size, 0);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    std::fill(dst.begin(), dst.end(), 0);
    const auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst.data(), src.data(), data_size);
    const auto end = std::chrono::high_resolution_clock::now();

    CHECK(verifyDataReceived(src, data_size))
        << "Data verification failed before memcpy.";
    if (num_warmups <= iteration) {
      const double duration =
          std::chrono::duration<double>(end - start).count();
      durations.push_back(duration);
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec";
}
