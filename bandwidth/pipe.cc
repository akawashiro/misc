#include <algorithm> // For std::min
#include <chrono>
#include <cstring>
#include <numeric>
#include <optional>
#include <string>
#include <sys/wait.h> // For wait
#include <unistd.h>   // For pipe, fork, close
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

ABSL_FLAG(int, num_iterations, 10,
          "Number of measurement iterations (minimum 3)");
ABSL_FLAG(int, num_warmups, 3, "Number of warmup iterations");
ABSL_FLAG(uint64_t, data_size, 128 * (1 << 20),
          "Size of data to transfer in bytes");

const int BUFFER_SIZE = 4096; // 4KB buffer for read/write

void send_process(int write_fd, int num_warmups, int num_iterations,
                  uint64_t data_size) {
  // Perform warm-up runs
  VLOG(1) << "Sender: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    std::vector<char> send_buffer(BUFFER_SIZE, 'W'); // 'W' for warmup
    size_t total_sent = 0;
    while (total_sent < data_size) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, data_size - total_sent);
      ssize_t bytes_written =
          write(write_fd, send_buffer.data(), bytes_to_send);
      if (bytes_written == -1) {
        perror("send: write during warmup");
        break;
      }
      total_sent += bytes_written;
    }
    VLOG(1) << "Sender: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Sender: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Sender: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    std::vector<char> send_buffer(BUFFER_SIZE, 'A'); // Fill buffer with 'A'
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data until data_size is reached
    while (total_sent < data_size) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, data_size - total_sent);
      ssize_t bytes_written =
          write(write_fd, send_buffer.data(), bytes_to_send);
      if (bytes_written == -1) {
        perror("send: write");
        break;
      }
      total_sent += bytes_written;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  // Close the write end of the pipe
  close(write_fd);
  VLOG(1) << "Sender: Exiting.";
}

void receive_process(int read_fd, int num_warmups, int num_iterations,
                     uint64_t data_size) {
  // Perform warm-up runs
  VLOG(1) << "Receiver: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    while (total_received < data_size) {
      ssize_t bytes_read = read(read_fd, recv_buffer.data(), BUFFER_SIZE);
      if (bytes_read == -1) {
        perror("receive: read during warmup");
        break;
      }
      if (bytes_read == 0) {
        break; // End of file (sender closed the pipe)
      }
      total_received += bytes_read;
    }
    VLOG(1) << "Receiver: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Receiver: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Receiver: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data until data_size is reached
    while (total_received < data_size) {
      ssize_t bytes_read = read(read_fd, recv_buffer.data(), BUFFER_SIZE);
      if (bytes_read == -1) {
        perror("receive: read");
        break;
      }
      if (bytes_read == 0) {
        VLOG(1) << "Receiver: Sender closed the pipe prematurely.";
        break;
      }
      total_received += bytes_read;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Receiver: Time taken: " << elapsed_time.count() << " seconds.";
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Close the read end of the pipe
  close(read_fd);
  VLOG(1) << "Receiver: Exiting.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Pipe Bandwidth Benchmark");
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

  // Create a pipe
  int pipe_fds[2];
  if (pipe(pipe_fds) == -1) {
    perror("pipe");
    return 1;
  }

  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    close(read_fd);
    close(write_fd);
    return 1;
  }

  if (pid == 0) {
    // Child process (sender)
    close(read_fd); // Close unused read end
    send_process(write_fd, num_warmups, num_iterations, data_size);
  } else {
    // Parent process (receiver)
    close(write_fd); // Close unused write end
    receive_process(read_fd, num_warmups, num_iterations, data_size);

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
