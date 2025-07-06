#include <optional>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"
#include "memcpy_benchmark.h"
#include "memcpy_mt_benchmark.h"
#include "tcp_benchmark.h"
#include "udp_benchmark.h"
#include "uds_benchmark.h"
#include "pipe_benchmark.h"
#include "mmap_benchmark.h"
#include "shm_benchmark.h"

ABSL_FLAG(std::string, type, "", "Benchmark type to run (memcpy, memcpy_mt, tcp, udp, uds, pipe, mmap, shm)");
ABSL_FLAG(int, num_iterations, 10, "Number of measurement iterations (minimum 3)");
ABSL_FLAG(int, num_warmups, 3, "Number of warmup iterations");
ABSL_FLAG(uint64_t, data_size, 128 * (1 << 20), "Size of data to transfer in bytes");
ABSL_FLAG(std::optional<int>, vlog, std::nullopt, "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Bandwidth benchmark tool. Use --type to specify benchmark type.");
  absl::ParseCommandLine(argc, argv);

  // Get values from command line flags
  std::string type = absl::GetFlag(FLAGS_type);
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  int num_warmups = absl::GetFlag(FLAGS_num_warmups);
  uint64_t data_size = absl::GetFlag(FLAGS_data_size);

  // Validate type
  if (type.empty()) {
    LOG(ERROR) << "Must specify --type. Available types: memcpy, memcpy_mt, tcp, udp, uds, pipe, mmap, shm";
    return 1;
  }

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

  // Run the appropriate benchmark
  int result = 0;
  if (type == "memcpy") {
    result = run_memcpy_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "memcpy_mt") {
    result = run_memcpy_mt_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "tcp") {
    result = run_tcp_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "udp") {
    result = run_udp_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "uds") {
    result = run_uds_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "pipe") {
    result = run_pipe_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "mmap") {
    result = run_mmap_benchmark(num_iterations, num_warmups, data_size);
  } else if (type == "shm") {
    result = run_shm_benchmark(num_iterations, num_warmups, data_size);
  } else {
    LOG(ERROR) << "Unknown benchmark type: " << type 
               << ". Available types: memcpy, memcpy_mt, tcp, udp, uds, pipe, mmap, shm";
    return 1;
  }

  return result;
}
