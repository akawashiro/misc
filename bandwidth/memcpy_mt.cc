#include <chrono>
#include <cstdint>
#include <cstring>
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
    LOG(INFO) << "Thread " << thread_id << " copying from " << start << " to "
              << end << " (size: " << (end - start) << " bytes)";
    std::memcpy(dst.data() + start, src.data() + start, end - start);
  };

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
  LOG(INFO) << "Bandwidth: " << static_cast<double>(size) / (1 << 30) / elapsed
            << " GiByte/sec. Threads: " << n_threads;
}

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  constexpr uint64_t size = 128 * 1024 * 1024; // 128 MiB
  for (uint64_t n_threads = 1; n_threads <= 4; ++n_threads) {
    memcpy_in_multi_thread(size, n_threads);
  }
}
