#include <chrono>
#include <thread>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

void memcpyInMultiThread(uint64_t n_threads) {
  std::vector<uint8_t> src = generateDataToSend();
  std::vector<uint8_t> dst(DATA_SIZE, 0x00);
  uint64_t chunk_size = DATA_SIZE / n_threads;

  auto copy_chunk = [&](uint64_t thread_id) {
    uint64_t start = thread_id * chunk_size;
    uint64_t end =
        (thread_id == n_threads - 1) ? DATA_SIZE : start + chunk_size;
    std::memcpy(dst.data() + start, src.data() + start, end - start);
  };

  std::vector<double> durations;
  for (size_t i = 0; i < NUM_WARMUPS + NUM_ITERATIONS; ++i) {
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

    if (NUM_WARMUPS <= i) {
      const double duration =
          std::chrono::duration<double>(end - start).count();
      durations.push_back(duration);
    }
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30)
            << " GiByte/sec. Threads: " << n_threads;
}

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  VLOG(1) << "Starting multi-threaded memcpy bandwidth test...";
  for (uint64_t n_threads = 1; n_threads <= 4; ++n_threads) {
    memcpyInMultiThread(n_threads);
  }
}
