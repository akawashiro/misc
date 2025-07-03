#include <algorithm> // For std::min
#include <chrono>
#include <cstring>
#include <numeric>
#include <string>
#include <sys/wait.h> // For wait
#include <unistd.h>   // For pipe, fork, close
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

const size_t DATA_SIZE = 128 * 1024 * 1024; // 128 MiB
const int BUFFER_SIZE = 4096;               // 4KB buffer for read/write
const int NUM_ITERATIONS = 10;              // Number of measurement iterations

void writer_process(int write_fd) {
  // Perform warm-up runs
  VLOG(1) << "Writer: Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    std::vector<char> send_buffer(BUFFER_SIZE, 'W'); // 'W' for warmup
    size_t total_sent = 0;
    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_written =
          write(write_fd, send_buffer.data(), bytes_to_send);
      if (bytes_written == -1) {
        perror("writer: write during warmup");
        break;
      }
      total_sent += bytes_written;
    }
    VLOG(1) << "Writer: Warm-up " << warmup + 1 << "/3 completed";
  }
  VLOG(1) << "Writer: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    VLOG(1) << "Writer: Starting iteration " << iteration + 1 << "/"
            << NUM_ITERATIONS;

    std::vector<char> send_buffer(BUFFER_SIZE, 'A'); // Fill buffer with 'A'
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data until DATA_SIZE is reached
    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_written =
          write(write_fd, send_buffer.data(), bytes_to_send);
      if (bytes_written == -1) {
        perror("writer: write");
        break;
      }
      total_sent += bytes_written;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Writer: Time taken: " << elapsed_time.count() << " seconds.";
  }

  // Sort durations and exclude min and max
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  // Calculate average of remaining 8 measurements
  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();

  if (average_duration > 0) {
    double bandwidth_gibps =
        DATA_SIZE / (average_duration * 1024.0 * 1024.0 * 1024.0);
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Writer";
  }

  // Close the write end of the pipe
  close(write_fd);
  VLOG(1) << "Writer: Exiting.";
}

void reader_process(int read_fd) {
  // Perform warm-up runs
  VLOG(1) << "Reader: Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    while (total_received < DATA_SIZE) {
      ssize_t bytes_read = read(read_fd, recv_buffer.data(), BUFFER_SIZE);
      if (bytes_read == -1) {
        perror("reader: read during warmup");
        break;
      }
      if (bytes_read == 0) {
        break; // End of file (writer closed the pipe)
      }
      total_received += bytes_read;
    }
    VLOG(1) << "Reader: Warm-up " << warmup + 1 << "/3 completed";
  }
  VLOG(1) << "Reader: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    VLOG(1) << "Reader: Starting iteration " << iteration + 1 << "/"
            << NUM_ITERATIONS;

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data until DATA_SIZE is reached
    while (total_received < DATA_SIZE) {
      ssize_t bytes_read = read(read_fd, recv_buffer.data(), BUFFER_SIZE);
      if (bytes_read == -1) {
        perror("reader: read");
        break;
      }
      if (bytes_read == 0) {
        VLOG(1) << "Reader: Writer closed the pipe prematurely.";
        break;
      }
      total_received += bytes_read;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Reader: Time taken: " << elapsed_time.count() << " seconds.";
  }

  // Sort durations and exclude min and max
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  // Calculate average of remaining 8 measurements
  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();

  if (average_duration > 0) {
    double bandwidth_gibps =
        DATA_SIZE / (average_duration * 1024.0 * 1024.0 * 1024.0);
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Reader";
  }

  // Close the read end of the pipe
  close(read_fd);
  VLOG(1) << "Reader: Exiting.";
}

int main() {
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
    // Child process (writer)
    close(read_fd); // Close unused read end
    writer_process(write_fd);
  } else {
    // Parent process (reader)
    close(write_fd); // Close unused write end
    reader_process(read_fd);

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
