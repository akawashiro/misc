#include "common.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "absl/log/check.h"
#include "absl/log/log.h"

std::vector<uint8_t> calcChecksum(const std::vector<uint8_t> &data) {
  std::vector<uint8_t> checksum(CHECKSUM_SIZE, 0);
  for (size_t i = 0; i < CONTEXT_SIZE; ++i) {
    checksum[i % CHECKSUM_SIZE] ^= data[i];
  }
  return checksum;
}

std::vector<uint8_t> generateDataToSend() {
  VLOG(1) << "Generating data to send...";
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  std::vector<uint8_t> data(DATA_SIZE, 0);

  size_t i = 0;
  for (i = 0; i + 8 < CONTEXT_SIZE; i += 8) {
    uint64_t *d = reinterpret_cast<uint64_t *>(&data[i]);
    *d = dist(engine);
  }
  for (; i < CONTEXT_SIZE; ++i) {
    data[i] = static_cast<uint8_t>(dist(engine) & 0xFF);
  }
  VLOG(1) << "Context data generated. Size: " << CONTEXT_SIZE
          << " bytes. Filling checksum...";
  const std::vector<uint8_t> checksum = calcChecksum(data);
  for (size_t j = 0; j < CHECKSUM_SIZE; ++j) {
    data[CONTEXT_SIZE + j] = checksum[j];
  }
  VLOG(1) << "Data generation complete. Data size: " << static_cast<double>(data.size()) / (1 << 30)
          << " GiByte, Checksum size: " << CHECKSUM_SIZE << " bytes.";

  return data;
}

bool verifyDataReceived(const std::vector<uint8_t> &data) {
  if (data.size() != DATA_SIZE) {
    LOG(ERROR) << "Data size mismatch: expected " << DATA_SIZE << ", got "
               << data.size();
    return false;
  }

  std::vector<uint8_t> checksum = calcChecksum(data);
  for (size_t i = 0; i < CHECKSUM_SIZE; ++i) {
    if (data[CONTEXT_SIZE + i] != checksum[i]) {
      LOG(ERROR) << "Checksum mismatch at index " << i << ": expected "
                 << static_cast<int>(checksum[i]) << ", got "
                 << static_cast<int>(data[CONTEXT_SIZE + i]);
      return false;
    }
  }
  return true;
}

double calculateBandwidth(std::vector<double> durations) {
  CHECK(durations.size() == NUM_ITERATIONS);
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();
  double bandwidth = DATA_SIZE / average_duration;
  return bandwidth;
}
