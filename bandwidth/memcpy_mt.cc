#include <chrono>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

ABSL_FLAG(uint64_t, data_size, 128 * (1 << 20),
          "Size of data to transfer in bytes");

void memcpyInMultiThread(uint64_t n_threads, int num_warmups,
                         int num_iterations, uint64_t data_size) {
  std::vector<uint8_t> src = generateDataToSend(data_size);
  std::vector<uint8_t> dst(data_size, 0x00);
  uint64_t chunk_size = data_size / n_threads;

  auto copy_chunk = [&](uint64_t thread_id) {
    uint64_t start = thread_id * chunk_size;
    uint64_t end =
        (thread_id == n_threads - 1) ? data_size : start + chunk_size;
    std::memcpy(dst.data() + start, src.data() + start, end - start);
  };

  std::vector<double> durations;
  for (size_t i = 0; i < num_warmups + num_iterations; ++i) {
    std::fill(dst.begin(), dst.end(), 0x00);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::thread> threads;
    for (uint64_t i = 0; i < n_threads; ++i) {
      threads.emplace_back(copy_chunk, i);
    }
    for (auto &t : threads) {
      t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();

    if (num_warmups <= i) {
      const double duration =
          std::chrono::duration<double>(end - start).count();
      durations.push_back(duration);
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30)
            << " GiByte/sec. Threads: " << n_threads;
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

ABSL_FLAG(int, num_iterations, 10,
          "Number of measurement iterations (minimum 3)");
ABSL_FLAG(int, num_warmups, 3, "Number of warmup iterations");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Multi-threaded Memcpy Bandwidth Benchmark");
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

  VLOG(1) << "Starting multi-threaded memcpy bandwidth test...";
  for (uint64_t n_threads = 1; n_threads <= 4; ++n_threads) {
    memcpyInMultiThread(n_threads, num_warmups, num_iterations, data_size);
  }
}
