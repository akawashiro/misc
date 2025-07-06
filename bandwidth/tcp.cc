#include <algorithm>    // For std::min
#include <arpa/inet.h>  // For inet_addr
#include <chrono>       // For time measurement
#include <cstring>      // For memset, strerror
#include <netinet/in.h> // For sockaddr_in and IPPROTO_TCP
#include <numeric>      // For std::accumulate
#include <optional>
#include <string>
#include <sys/socket.h>
#include <unistd.h> // For fork, close
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

const int PORT = 12345; // Port number for TCP communication
const std::string LOOPBACK_IP = "127.0.0.1"; // Localhost IP address
const int BUFFER_SIZE = 4096;                // 4KB buffer for send/recv

void receive_process(int num_warmups, int num_iterations, uint64_t data_size) {
  int listen_fd, conn_fd;
  struct sockaddr_in receive_addr, send_addr;
  socklen_t send_len = sizeof(send_addr);

  // Create a TCP socket
  listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd == -1) {
    LOG(ERROR) << "receive: socket: " << strerror(errno);
    return;
  }

  // Allow immediate reuse of the port after the program exits
  int optval = 1;
  if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &optval,
                 sizeof(optval)) == -1) {
    LOG(ERROR) << "receive: setsockopt SO_REUSEADDR: " << strerror(errno);
    close(listen_fd);
    return;
  }

  // Configure receive address
  memset(&receive_addr, 0, sizeof(receive_addr));
  receive_addr.sin_family = AF_INET;
  receive_addr.sin_addr.s_addr = inet_addr(LOOPBACK_IP.c_str());
  receive_addr.sin_port = htons(PORT);

  // Bind the socket to the specified IP address and port
  if (bind(listen_fd, (struct sockaddr *)&receive_addr, sizeof(receive_addr)) ==
      -1) {
    LOG(ERROR) << "receive: bind: " << strerror(errno);
    close(listen_fd);
    return;
  }

  // Listen for incoming connections
  if (listen(listen_fd, 5) == -1) {
    LOG(ERROR) << "receive: listen: " << strerror(errno);
    close(listen_fd);
    return;
  }

  VLOG(1) << "Receiver: Listening on " << LOOPBACK_IP << ":" << PORT;

  // Perform warm-up runs
  VLOG(1) << "Receiver: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    conn_fd = accept(listen_fd, (struct sockaddr *)&send_addr, &send_len);
    if (conn_fd == -1) {
      LOG(ERROR) << "receive: accept during warmup: " << strerror(errno);
      close(listen_fd);
      return;
    }

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    while (total_received < data_size) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
      if (bytes_received == -1) {
        LOG(ERROR) << "receive: recv during warmup: " << strerror(errno);
        break;
      }
      if (bytes_received == 0) {
        break;
      }
      total_received += bytes_received;
    }
    close(conn_fd);
    VLOG(1) << "Receiver: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Receiver: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    // Accept a sender connection for each iteration
    conn_fd = accept(listen_fd, (struct sockaddr *)&send_addr, &send_len);
    if (conn_fd == -1) {
      LOG(ERROR) << "receive: accept: " << strerror(errno);
      close(listen_fd);
      return;
    }

    VLOG(1) << ReceivePrefix(iteration + 1) << "Sender connected from "
            << inet_ntoa(send_addr.sin_addr) << ":" << ntohs(send_addr.sin_port)
            << ". Receiving data...";

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Receive data until data_size is reached
    while (total_received < data_size) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
      if (bytes_received == -1) {
        LOG(ERROR) << "receive: recv: " << strerror(errno);
        break;
      }
      if (bytes_received == 0) {
        LOG(INFO) << "Receiver: Sender disconnected prematurely.";
        break;
      }
      total_received += bytes_received;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Receiver: Received "
            << total_received / (1024.0 * 1024.0 * 1024.0) << " GiB of data in "
            << elapsed_time.count() << " seconds.";

    // Close connection for this iteration
    close(conn_fd);
  }

  double bandwidth_gibps =
      calculateBandwidth(durations, num_iterations, data_size) /
      (1024.0 * 1024.0 * 1024.0);

  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Close sockets
  close(listen_fd);
  VLOG(1) << "Receiver: Exiting.";
}

void send_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Perform warm-up runs
  VLOG(1) << "Sender: Performing warm-up runs...";

  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    int sock_fd;
    struct sockaddr_in receive_addr;

    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd == -1) {
      LOG(ERROR) << "send: socket during warmup: " << strerror(errno);
      return;
    }

    memset(&receive_addr, 0, sizeof(receive_addr));
    receive_addr.sin_family = AF_INET;
    receive_addr.sin_addr.s_addr = inet_addr(LOOPBACK_IP.c_str());
    receive_addr.sin_port = htons(PORT);

    while (connect(sock_fd, (struct sockaddr *)&receive_addr,
                   sizeof(receive_addr)) == -1) {
      sleep(1);
    }

    std::vector<char> send_buffer(BUFFER_SIZE, 'W'); // 'W' for warmup
    size_t total_sent = 0;
    while (total_sent < data_size) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, data_size - total_sent);
      ssize_t bytes_sent = send(sock_fd, send_buffer.data(), bytes_to_send, 0);
      if (bytes_sent == -1) {
        LOG(ERROR) << "send: send during warmup: " << strerror(errno);
        break;
      }
      total_sent += bytes_sent;
    }
    shutdown(sock_fd, SHUT_WR);
    close(sock_fd);
    VLOG(1) << "Sender: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
    usleep(100000); // 100ms delay between warmup runs
  }
  VLOG(1) << "Sender: Warm-up complete. Starting measurements...";

  std::vector<uint8_t> data_to_send = generateDataToSend(data_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    int sock_fd;
    struct sockaddr_in receive_addr;

    // Create a TCP socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd == -1) {
      LOG(ERROR) << "send: socket: " << strerror(errno);
      return;
    }

    // Configure receive address to connect to
    memset(&receive_addr, 0, sizeof(receive_addr));
    receive_addr.sin_family = AF_INET;
    receive_addr.sin_addr.s_addr = inet_addr(LOOPBACK_IP.c_str());
    receive_addr.sin_port = htons(PORT);

    // Connect to the receiver
    VLOG(1) << SendPrefix(iteration + 1) << "Connecting to receiver at "
            << LOOPBACK_IP << ":" << PORT;
    while (connect(sock_fd, (struct sockaddr *)&receive_addr,
                   sizeof(receive_addr)) == -1) {
      LOG(ERROR) << "send: connect (retrying in 1 second): " << strerror(errno);
      sleep(1); // Wait a bit if receiver isn't ready yet
    }

    VLOG(1) << SendPrefix(iteration + 1)
            << "Connected to receiver. Sending data...";

    std::vector<char> send_buffer(BUFFER_SIZE, 'B'); // Fill buffer with 'B'
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Send data until data_size is reached
    while (total_sent < data_size) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, data_size - total_sent);
      // ssize_t bytes_sent = send(sock_fd, send_buffer.data(), bytes_to_send,
      // 0);
      ssize_t bytes_sent =
          send(sock_fd, data_to_send.data() + total_sent, bytes_to_send, 0);
      if (bytes_sent == -1) {
        LOG(ERROR) << "send: send: " << strerror(errno);
        break;
      }
      total_sent += bytes_sent;
    }

    // Ensure all data is sent before closing the socket
    shutdown(sock_fd, SHUT_WR);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";

    // Close the socket
    close(sock_fd);

    // Small delay between iterations to allow receiver to reset
    if (iteration < num_iterations - 1) {
      usleep(100000); // 100ms delay
    }
  }

  double bandwidth_gibps =
      calculateBandwidth(durations, num_iterations, data_size) /
      (1024.0 * 1024.0 * 1024.0);

  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  VLOG(1) << "Sender: Exiting.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("TCP Bandwidth Benchmark");
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

  pid_t pid = fork();

  if (pid == -1) {
    LOG(ERROR) << "fork: " << strerror(errno);
    return 1;
  }

  if (pid == 0) {
    // Child process (sender)
    send_process(num_warmups, num_iterations, data_size);
  } else {
    // Parent process (receiver)
    receive_process(num_warmups, num_iterations, data_size);
  }

  return 0;
}
