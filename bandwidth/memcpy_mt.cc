#include <cstdint>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>

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

  clock_t start = clock();
  std::vector<std::thread> threads;
  for (uint64_t i = 0; i < n_threads; ++i) {
    threads.emplace_back(copy_chunk, i);
  }
  for (auto &t : threads) {
    t.join();
  }
  clock_t end = clock();
  double elapsed = static_cast<double>(end - start) / CLOCKS_PER_SEC;
  LOG(INFO) << "Memcpy in " << n_threads << " threads: " << elapsed
            << " seconds, bandwidth: "
            << static_cast<double>(size) / (elapsed * (1 << 30)) << " GiByte/s";
}

int main() {
  constexpr uint64_t size = (1 << 30);
  for (uint64_t n_threads = 1; n_threads <= 8; ++n_threads) {
    memcpy_in_multi_thread(size, n_threads);
  }
}
