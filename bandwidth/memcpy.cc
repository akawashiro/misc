#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  constexpr uint64_t size = (1 << 30); // 1 GiB
  constexpr uint64_t iterations = 10;
  void *src = malloc(size);
  void *dst = malloc(size);

  std::vector<double> durations;
  for (uint64_t i = 0; i < iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst, src, size);
    const auto end = std::chrono::high_resolution_clock::now();
    const double duration_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    if (i == 0) {
      continue; // Skip the first iteration
    }
    const double duration = duration_ns / 1e9;
    durations.push_back(duration);
  }
  double average_duration =
      std::accumulate(durations.begin(), durations.end(), 0.0) /
      durations.size();
  double bandwidth = size / average_duration;

  std::cout << "Bandwidth: " << bandwidth / (1 << 30) << " GiB/sec" << std::endl;
}
