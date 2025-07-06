#include "pipe_benchmark.h"

#include <algorithm> // For std::min
#include <chrono>
#include <cstring>
#include <numeric>
#include <string>
#include <sys/wait.h> // For wait
#include <unistd.h>   // For pipe, fork, close
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/log.h"

#include "common.h"

// BUFFER_SIZE is now passed as a parameter

void send_process(int write_fd, int num_warmups, int num_iterations,
                  uint64_t data_size, uint64_t buffer_size) {
  // Generate data to send once
  std::vector<uint8_t> data_to_send = generateDataToSend(data_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;
    int display_iteration =
        is_warmup ? iteration + 1 : iteration - num_warmups + 1;

    if (is_warmup) {
      VLOG(1) << "Sender: Warm-up " << display_iteration << "/" << num_warmups;
    } else {
      VLOG(1) << "Sender: Starting iteration " << display_iteration << "/"
              << num_iterations;
    }

    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data until data_size is reached
    while (total_sent < data_size) {
      size_t bytes_to_send = std::min(buffer_size, data_size - total_sent);
      ssize_t bytes_written;
      if (is_warmup) {
        std::vector<uint8_t> warmup_buffer(bytes_to_send, 'W');
        bytes_written = write(write_fd, warmup_buffer.data(), bytes_to_send);
      } else {
        bytes_written =
            write(write_fd, data_to_send.data() + total_sent, bytes_to_send);
      }
      if (bytes_written == -1) {
        LOG(ERROR) << "send: write: " << strerror(errno);
        break;
      }
      total_sent += bytes_written;
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  // Close the write end of the pipe
  close(write_fd);
  VLOG(1) << "Sender: Exiting.";
}

void receive_process(int read_fd, int num_warmups, int num_iterations,
                     uint64_t data_size, uint64_t buffer_size) {
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;
    int display_iteration =
        is_warmup ? iteration + 1 : iteration - num_warmups + 1;

    if (is_warmup) {
      VLOG(1) << "Receiver: Warm-up " << display_iteration << "/"
              << num_warmups;
    } else {
      VLOG(1) << "Receiver: Starting iteration " << display_iteration << "/"
              << num_iterations;
    }

    std::vector<uint8_t> recv_buffer(buffer_size);
    std::vector<uint8_t> received_data;
    if (!is_warmup) {
      received_data.reserve(data_size);
    }
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data until data_size is reached
    while (total_received < data_size) {
      ssize_t bytes_read = read(read_fd, recv_buffer.data(), buffer_size);
      if (bytes_read == -1) {
        LOG(ERROR) << "receive: read: " << strerror(errno);
        break;
      }
      if (bytes_read == 0) {
        if (!is_warmup) {
          VLOG(1) << "Receiver: Sender closed the pipe prematurely.";
        }
        break;
      }
      total_received += bytes_read;
      if (!is_warmup) {
        received_data.insert(received_data.end(), recv_buffer.begin(),
                             recv_buffer.begin() + bytes_read);
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());

      // Verify received data
      if (!verifyDataReceived(received_data, data_size)) {
        LOG(ERROR) << ReceivePrefix(display_iteration)
                   << "Data verification failed!";
      } else {
        VLOG(1) << ReceivePrefix(display_iteration)
                << "Data verification passed.";
      }

      VLOG(1) << "Receiver: Time taken: " << elapsed_time.count()
              << " seconds.";
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Close the read end of the pipe
  close(read_fd);
  VLOG(1) << "Receiver: Exiting.";
}

int run_pipe_benchmark(int num_iterations, int num_warmups, uint64_t data_size,
                       uint64_t buffer_size) {
  // Create a pipe
  int pipe_fds[2];
  if (pipe(pipe_fds) == -1) {
    LOG(ERROR) << "pipe: " << strerror(errno);
    return 1;
  }

  int read_fd = pipe_fds[0];
  int write_fd = pipe_fds[1];

  pid_t pid = fork();

  if (pid == -1) {
    LOG(ERROR) << "fork: " << strerror(errno);
    close(read_fd);
    close(write_fd);
    return 1;
  }

  if (pid == 0) {
    // Child process (sender)
    close(read_fd); // Close unused read end
    send_process(write_fd, num_warmups, num_iterations, data_size, buffer_size);
  } else {
    // Parent process (receiver)
    close(write_fd); // Close unused write end
    receive_process(read_fd, num_warmups, num_iterations, data_size,
                    buffer_size);

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
