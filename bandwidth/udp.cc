#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <numeric>
#include <optional>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

// --- Common Settings ---
constexpr int PORT = 12345;
constexpr const char *HOST = "127.0.0.1";
constexpr size_t CHUNK_SIZE = 8192;                        // 8 KB
constexpr uint64_t TOTAL_DATA_SIZE = 128ULL * 1024 * 1024; // 128 MiB
constexpr uint64_t NUM_PACKETS = TOTAL_DATA_SIZE / CHUNK_SIZE;
constexpr int NUM_ITERATIONS = 10; // Number of measurement iterations

// --- Error Handling Function ---
void die(const char *message) {
  perror(message);
  exit(1);
}

// --- Receiver (Child Process) Operations ---
void run_receiver(int pipe_write_fd) {
  int sockfd;
  struct sockaddr_in serv_addr, cli_addr;
  socklen_t cli_len = sizeof(cli_addr);

  // Create UDP socket
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    die("Receiver: socket() failed");
  }

  // Set server address information
  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_addr.s_addr = INADDR_ANY;
  serv_addr.sin_port = htons(PORT);

  // Bind address to socket
  if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    close(pipe_write_fd); // Close pipe before exiting
    die("Receiver: bind() failed");
  }

  // Notify parent process that receiver is ready
  char signal_char = 'R'; // 'R' for Ready
  if (write(pipe_write_fd, &signal_char, 1) != 1) {
    die("Receiver: pipe write failed");
  }
  close(pipe_write_fd); // Close pipe after notification

  LOG(INFO) << "[Receiver] Waiting for data... (Port: " << PORT << ")";

  std::vector<char> buffer(CHUNK_SIZE);

  // Perform warm-up runs
  VLOG(1) << "[Receiver] Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    uint64_t total_bytes_received = 0;
    ssize_t bytes_received;

    // Wait for warmup start signal from sender
    bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                              (struct sockaddr *)&cli_addr, &cli_len);
    if (bytes_received < 0) {
      die("Receiver: recvfrom() failed on warmup start");
    }

    if (buffer[0] == 'W') { // 'W' for warmup
      total_bytes_received += bytes_received;
      while (total_bytes_received < TOTAL_DATA_SIZE) {
        bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                                  (struct sockaddr *)&cli_addr, &cli_len);
        if (bytes_received > 0) {
          total_bytes_received += bytes_received;
        }
      }
      VLOG(1) << "[Receiver] Warm-up " << warmup + 1 << "/3 completed";
    }
  }
  VLOG(1) << "[Receiver] Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    uint64_t total_bytes_received = 0;
    ssize_t bytes_received;

    // Wait for iteration start signal from sender
    bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                              (struct sockaddr *)&cli_addr, &cli_len);
    if (bytes_received < 0) {
      die("Receiver: recvfrom() failed on iteration start");
    }

    // Check if this is a start signal (first packet should contain 'S')
    if (buffer[0] == 'S') {
      total_bytes_received += bytes_received;

      // Start measurement
      auto start_time = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "[Receiver] Iteration " << iteration + 1 << "/"
                << NUM_ITERATIONS << " started...";

      // Receive data until total size is reached
      while (total_bytes_received < TOTAL_DATA_SIZE) {
        bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                                  (struct sockaddr *)&cli_addr, &cli_len);
        if (bytes_received > 0) {
          total_bytes_received += bytes_received;
        }
      }

      // End measurement
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end_time - start_time;
      durations.push_back(elapsed.count());

      LOG(INFO) << "[Receiver] Iteration " << iteration + 1 << " completed in "
                << elapsed.count() << " seconds";
    }
  }

  // Sort durations and exclude min and max
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  // Calculate average of remaining 8 measurements
  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();

  double gibytes_per_second =
      (static_cast<double>(TOTAL_DATA_SIZE) / (1024.0 * 1024.0 * 1024.0)) /
      average_duration;
  LOG(INFO) << "Bandwidth: " << gibytes_per_second << " GiByte/sec";

  close(sockfd);
}

// --- Sender (Parent Process) Operations ---
void run_sender() {
  int sockfd;
  struct sockaddr_in serv_addr;

  // Create UDP socket
  if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    die("Sender: socket() failed");
  }

  // Set destination (server) address information
  memset(&serv_addr, 0, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);
  if (inet_aton(HOST, &serv_addr.sin_addr) == 0) {
    die("Sender: inet_aton() failed");
  }

  // Perform warm-up runs
  VLOG(1) << "[Sender] Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    // First packet marked with 'W' to signal warmup
    std::vector<char> warmup_buffer(CHUNK_SIZE, 'W');
    if (sendto(sockfd, warmup_buffer.data(), CHUNK_SIZE, 0,
               (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      perror("Sender: sendto() error on warmup signal");
    }

    // Send remaining packets
    std::vector<char> buffer(CHUNK_SIZE, 'w');
    for (uint64_t i = 1; i < NUM_PACKETS; ++i) {
      if (i % 10 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (sendto(sockfd, buffer.data(), CHUNK_SIZE, 0,
                 (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        // Display error but continue
        perror("Sender: sendto() error during warmup");
      }
    }
    VLOG(1) << "[Sender] Warm-up " << warmup + 1 << "/3 completed";
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  VLOG(1) << "[Sender] Warm-up complete. Starting measurements...";

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    LOG(INFO) << "[Sender] Starting iteration " << iteration + 1 << "/"
              << NUM_ITERATIONS;

    // First packet marked with 'S' to signal start
    std::vector<char> start_buffer(CHUNK_SIZE, 'S');
    if (sendto(sockfd, start_buffer.data(), CHUNK_SIZE, 0,
               (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      perror("Sender: sendto() error on start signal");
    }

    // Send remaining packets
    std::vector<char> buffer(CHUNK_SIZE, 'D');
    for (uint64_t i = 1; i < NUM_PACKETS; ++i) {
      if (i % 10 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (sendto(sockfd, buffer.data(), CHUNK_SIZE, 0,
                 (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        // Display error but continue
        perror("Sender: sendto() error");
      }
    }

    // Small delay between iterations
    if (iteration < NUM_ITERATIONS - 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  LOG(INFO) << "[Sender] All iterations complete.";
  close(sockfd);
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("UDP Bandwidth Benchmark");
  absl::ParseCommandLine(argc, argv);

  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  // Pipe for parent-child synchronization
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    die("pipe() failed");
  }

  // Fork the process
  pid_t pid = fork();

  if (pid < 0) {
    die("fork() failed");
  }

  if (pid == 0) {
    // --- Child Process (Receiver) ---
    close(pipefd[0]); // Close read end of pipe (not used by child)
    run_receiver(
        pipefd[1]); // Execute receiver process (pass write end of pipe)
    exit(0);

  } else {
    // --- Parent Process (Sender) ---
    close(pipefd[1]); // Close write end of pipe (not used by parent)

    // Wait for ready notification from child process
    char signal_buffer;
    LOG(INFO) << "[Main] Waiting for receiver process to be ready...";
    if (read(pipefd[0], &signal_buffer, 1) != 1) {
      die("Main: pipe read failed. Receiver may have failed to start.");
    }
    close(pipefd[0]); // Read complete

    if (signal_buffer == 'R') {
      LOG(INFO) << "[Main] Receiver is ready. Starting transmission.";
    }

    run_sender(); // Execute sender process

    // Wait for child process to terminate (prevent zombie processes)
    wait(NULL);
    LOG(INFO) << "[Main] Measurement complete.";
  }

  return 0;
}
