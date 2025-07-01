#include <algorithm>
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
  std::vector<uint8_t> src(size, 0xFF);
  std::vector<uint8_t> dst(size, 0x00);

  // Perform 10 measurements
  std::vector<double> durations;
  for (int iteration = 0; iteration < 10; ++iteration) {
    // Reset destination buffer for each measurement
    std::fill(dst.begin(), dst.end(), 0x00);

    const auto start = std::chrono::high_resolution_clock::now();
    std::memcpy(dst.data(), src.data(), size);
    const auto end = std::chrono::high_resolution_clock::now();
    const double duration = std::chrono::duration<double>(end - start).count();
    durations.push_back(duration);
  }

  // Sort durations and exclude min and max
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  // Calculate average of remaining 8 measurements
  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();
  double bandwidth = size / average_duration;

  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec";
}
