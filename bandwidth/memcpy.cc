#include <chrono>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  constexpr uint64_t size = 128 * 1024 * 1024; // 128 MiB
  constexpr uint64_t iterations = 10;
  std::vector<uint8_t> src(size, 0xFF);
  std::vector<uint8_t> dst(size, 0x00);

  std::vector<double> durations;
  for (uint64_t i = 0; i < iterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst.data(), src.data(), size);
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

  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec";
}
