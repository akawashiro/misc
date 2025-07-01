#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

void memcpy_in_multi_thread(uint64_t size, uint64_t n_threads) {
  std::vector<uint8_t> src(size, 0xFF);
  std::vector<uint8_t> dst(size, 0x00);
  uint64_t chunk_size = size / n_threads;

  auto copy_chunk = [&](uint64_t thread_id) {
    uint64_t start = thread_id * chunk_size;
    uint64_t end = (thread_id == n_threads - 1) ? size : start + chunk_size;
    std::memcpy(dst.data() + start, src.data() + start, end - start);
  };

  // Perform warm-up runs to stabilize cache and memory subsystem
  for (int warmup = 0; warmup < 3; ++warmup) {
    std::fill(dst.begin(), dst.end(), 0x00);
    std::vector<std::thread> warmup_threads;
    for (uint64_t i = 0; i < n_threads; ++i) {
      warmup_threads.emplace_back(copy_chunk, i);
    }
    for (auto &t : warmup_threads) {
      t.join();
    }
  }

  // Perform 10 measurements
  std::vector<double> durations;
  for (int iteration = 0; iteration < 10; ++iteration) {
    // Reset destination buffer for each measurement
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
    auto elapsed = std::chrono::duration<double>(end - start).count();
    durations.push_back(elapsed);
  }

  // Sort durations and exclude min and max
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  // Calculate average of remaining 8 measurements
  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();

  double bandwidth = static_cast<double>(size) / (1 << 30) / average_duration;
  LOG(INFO) << "Bandwidth: " << bandwidth
            << " GiByte/sec. Threads: " << n_threads;
}

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  LOG(INFO) << "Starting multi-threaded memcpy bandwidth test...";
  constexpr uint64_t size = 128 * 1024 * 1024; // 128 MiB
  for (uint64_t n_threads = 1; n_threads <= 4; ++n_threads) {
    memcpy_in_multi_thread(size, n_threads);
  }
}
