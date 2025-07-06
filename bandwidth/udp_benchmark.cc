#include "udp_benchmark.h"

#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <numeric>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/log.h"

#include "common.h"

// --- Common Settings ---
constexpr int PORT = 12345;
constexpr const char *HOST = "127.0.0.1";
// CHUNK_SIZE is now passed as a parameter (buffer_size)

// --- Receiver (Child Process) Operations ---
void run_receiver(int pipe_write_fd, int num_warmups, int num_iterations,
                  uint64_t data_size, uint64_t buffer_size) {
  // Notify parent process that receiver is ready
  char signal_char = 'R'; // 'R' for Ready
  if (write(pipe_write_fd, &signal_char, 1) != 1) {
    LOG(ERROR) << "Receiver: pipe write failed" << ": " << strerror(errno);
    exit(1);
  }
  close(pipe_write_fd); // Close pipe after notification

  LOG(INFO) << "[Receiver] Waiting for data... (Port: " << PORT << ")";

  std::vector<uint8_t> buffer(buffer_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    int sockfd;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t cli_len = sizeof(cli_addr);

    // Create UDP socket for each iteration
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
      LOG(ERROR) << "Receiver: socket() failed" << ": " << strerror(errno);
      exit(1);
    }

    // Set receive address information
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(PORT);

    // Bind address to socket
    if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      LOG(ERROR) << "Receiver: bind() failed" << ": " << strerror(errno);
      close(sockfd);
      exit(1);
    }
    bool is_warmup = iteration < num_warmups;
    int display_iteration =
        is_warmup ? iteration + 1 : iteration - num_warmups + 1;

    uint64_t total_bytes_received = 0;
    ssize_t bytes_received;
    std::vector<uint8_t> received_data;
    if (!is_warmup) {
      received_data.reserve(data_size);
    }

    // Wait for start signal from sender
    bytes_received = recvfrom(sockfd, buffer.data(), buffer_size, 0,
                              (struct sockaddr *)&cli_addr, &cli_len);
    if (bytes_received < 0) {
      LOG(ERROR) << "Receiver: recvfrom() failed on start" << ": "
                 << strerror(errno);
      exit(1);
    }

    // Check signal type
    char expected_signal = is_warmup ? 'W' : 'S';
    if (buffer[0] == expected_signal) {
      total_bytes_received += bytes_received;
      received_data.insert(received_data.end(), buffer.begin(),
                           buffer.begin() + bytes_received);

      // Start measurement
      auto start_time = std::chrono::high_resolution_clock::now();
      if (is_warmup) {
        VLOG(1) << "[Receiver] Warm-up " << display_iteration << "/"
                << num_warmups;
      } else {
        LOG(INFO) << ReceivePrefix(display_iteration) << "Iteration started...";
      }

      // Receive data until total size is reached
      while (total_bytes_received < data_size) {
        bytes_received = recvfrom(sockfd, buffer.data(), buffer_size, 0,
                                  (struct sockaddr *)&cli_addr, &cli_len);
        if (bytes_received > 0) {
          total_bytes_received += bytes_received;
          received_data.insert(received_data.end(), buffer.begin(),
                               buffer.begin() + bytes_received);
        }
      }

      // End measurement
      auto end_time = std::chrono::high_resolution_clock::now();

      if (!is_warmup) {
        std::chrono::duration<double> elapsed = end_time - start_time;
        durations.push_back(elapsed.count());

        LOG(INFO) << ReceivePrefix(display_iteration)
                  << "Iteration completed in " << elapsed.count() << " seconds";
      }

      // Verify received data (always, even during warmup)
      if (!verifyDataReceived(received_data, data_size)) {
        LOG(ERROR) << ReceivePrefix(display_iteration)
                   << "Data verification failed!";
      } else {
        VLOG(1) << ReceivePrefix(display_iteration)
                << "Data verification passed.";
      }
    }

    // Close socket for this iteration
    close(sockfd);
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double gibytes_per_second = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << gibytes_per_second << " GiByte/sec";
}

// --- Sender (Parent Process) Operations ---
void run_sender(int num_warmups, int num_iterations, uint64_t data_size,
                uint64_t buffer_size) {
  // Generate data to send once
  std::vector<uint8_t> data_to_send = generateDataToSend(data_size);

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    int sockfd;
    struct sockaddr_in serv_addr;

    // Create UDP socket for each iteration
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
      LOG(ERROR) << "Sender: socket() failed" << ": " << strerror(errno);
      exit(1);
    }

    // Set destination (receive) address information
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    if (inet_aton(HOST, &serv_addr.sin_addr) == 0) {
      LOG(ERROR) << "Sender: inet_aton() failed" << ": " << strerror(errno);
      close(sockfd);
      exit(1);
    }
    bool is_warmup = iteration < num_warmups;
    int display_iteration =
        is_warmup ? iteration + 1 : iteration - num_warmups + 1;

    if (is_warmup) {
      VLOG(1) << "[Sender] Warm-up " << display_iteration << "/" << num_warmups;
    } else {
      LOG(INFO) << SendPrefix(display_iteration) << "Starting iteration...";
    }

    // First packet marked with signal character
    char signal_char = is_warmup ? 'W' : 'S';
    std::vector<uint8_t> first_packet(buffer_size);
    first_packet[0] = signal_char;
    // Fill rest of first packet with actual data
    size_t first_packet_data_size = std::min(buffer_size - 1, data_size);
    std::memcpy(first_packet.data() + 1, data_to_send.data(),
                first_packet_data_size);

    if (sendto(sockfd, first_packet.data(), buffer_size, 0,
               (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      LOG(ERROR) << "Sender: sendto() error on signal: " << strerror(errno);
    }

    // Send remaining packets using generated data
    uint64_t num_packets = data_size / buffer_size;
    for (uint64_t i = 1; i < num_packets; ++i) {
      if (i % 10 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      size_t offset = i * buffer_size;
      if (sendto(sockfd, data_to_send.data() + offset, buffer_size, 0,
                 (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        // Display error but continue
        LOG(ERROR) << "Sender: sendto() error: " << strerror(errno);
      }
    }

    // Close socket for this iteration
    close(sockfd);

    // Small delay between iterations
    if (iteration < num_warmups + num_iterations - 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  LOG(INFO) << "[Sender] All iterations complete.";
}

int run_udp_benchmark(int num_iterations, int num_warmups, uint64_t data_size,
                      uint64_t buffer_size) {
  // Pipe for parent-child synchronization
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    LOG(ERROR) << "pipe() failed" << ": " << strerror(errno);
    exit(1);
  }

  // Fork the process
  pid_t pid = fork();

  if (pid < 0) {
    LOG(ERROR) << "fork() failed" << ": " << strerror(errno);
    exit(1);
  }

  if (pid == 0) {
    // --- Child Process (Receiver) ---
    close(pipefd[0]); // Close read end of pipe (not used by child)
    run_receiver(
        pipefd[1], num_warmups, num_iterations, data_size,
        buffer_size); // Execute receiver process (pass write end of pipe)
    exit(0);

  } else {
    // --- Parent Process (Sender) ---
    close(pipefd[1]); // Close write end of pipe (not used by parent)

    // Wait for ready notification from child process
    char signal_buffer;
    LOG(INFO) << "[Main] Waiting for receiver process to be ready...";
    if (read(pipefd[0], &signal_buffer, 1) != 1) {
      LOG(ERROR) << "Main: pipe read failed. Receiver may have failed to start."
                 << ": " << strerror(errno);
      exit(1);
    }
    close(pipefd[0]); // Read complete

    if (signal_buffer == 'R') {
      LOG(INFO) << "[Main] Receiver is ready. Starting transmission.";
    }

    run_sender(num_warmups, num_iterations, data_size,
               buffer_size); // Execute sender process

    // Wait for child process to terminate (prevent zombie processes)
    wait(NULL);
    LOG(INFO) << "[Main] Measurement complete.";
  }

  return 0;
}
