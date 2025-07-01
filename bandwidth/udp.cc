#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "absl/log/log.h"

// --- Common Settings ---
constexpr int PORT = 12345;
constexpr const char *HOST = "127.0.0.1";
constexpr size_t CHUNK_SIZE = 8192;                        // 8 KB
constexpr uint64_t TOTAL_DATA_SIZE = 128ULL * 1024 * 1024; // 128 MiB
constexpr uint64_t NUM_PACKETS = TOTAL_DATA_SIZE / CHUNK_SIZE;

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
  uint64_t total_bytes_received = 0;
  ssize_t bytes_received;

  // Receive the first packet
  bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                            (struct sockaddr *)&cli_addr, &cli_len);
  if (bytes_received < 0) {
    die("Receiver: recvfrom() failed on first packet");
  }
  total_bytes_received += bytes_received;

  // Start measurement
  auto start_time = std::chrono::high_resolution_clock::now();
  LOG(INFO) << "[Receiver] Receiving started...";

  // Receive data until total size is reached
  while (total_bytes_received < TOTAL_DATA_SIZE) {
    LOG(INFO) << "[Receiver] Received: " << total_bytes_received << " bytes in "
              << TOTAL_DATA_SIZE << " bytes ("
              << (static_cast<double>(total_bytes_received) / TOTAL_DATA_SIZE) *
                     100
              << "%)" << " remaining bytes: "
              << (TOTAL_DATA_SIZE - total_bytes_received) << " bytes";
    bytes_received = recvfrom(sockfd, buffer.data(), CHUNK_SIZE, 0,
                              (struct sockaddr *)&cli_addr, &cli_len);
    if (bytes_received > 0) {
      total_bytes_received += bytes_received;
    }
  }

  // End measurement
  auto end_time = std::chrono::high_resolution_clock::now();

  // Calculate and display results
  std::chrono::duration<double> elapsed = end_time - start_time;
  double gibytes_per_second =
      (static_cast<double>(total_bytes_received) / (1024.0 * 1024.0 * 1024.0)) /
      elapsed.count();

  LOG(INFO) << "--- Reception Results ---";
  LOG(INFO) << "Total Data Received: "
            << static_cast<double>(total_bytes_received) / (1024 * 1024)
            << " MiB";
  LOG(INFO) << "Time Elapsed:          " << elapsed.count() << " seconds";
  LOG(INFO) << "Average Bandwidth:     " << gibytes_per_second << " GiByte/sec";

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

  std::vector<char> buffer(CHUNK_SIZE, 'D');

  LOG(INFO) << "[Sender] Sending 1 GiB (" << TOTAL_DATA_SIZE
            << " bytes) of data...";

  // Loop to send data
  for (uint64_t i = 0; i < NUM_PACKETS; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (sendto(sockfd, buffer.data(), CHUNK_SIZE, 0,
               (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
      // Display error but continue
      perror("Sender: sendto() error");
    }
  }

  LOG(INFO) << "[Sender] Sending complete.";
  close(sockfd);
}

int main() {
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
