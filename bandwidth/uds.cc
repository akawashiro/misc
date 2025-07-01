#include <algorithm> // For std::min
#include <chrono>
#include <cstdio> // For remove()
#include <cstring>
#include <numeric>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

const std::string SOCKET_PATH = "/tmp/unix_domain_socket_test.sock";
const size_t DATA_SIZE = 128 * 1024 * 1024; // 128 MiB
const int BUFFER_SIZE = 4096;               // 4KB buffer for send/recv
const int NUM_ITERATIONS = 10;              // Number of measurement iterations

void server_process() {
  int listen_fd, conn_fd;
  struct sockaddr_un addr;

  // Create a Unix domain stream socket
  listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listen_fd == -1) {
    perror("server: socket");
    return;
  }

  // Remove the socket file if it already exists (from previous runs)
  remove(SOCKET_PATH.c_str());

  // Configure the socket address
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

  // Bind the socket to the specified path
  if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    perror("server: bind");
    close(listen_fd);
    return;
  }

  // Listen for incoming connections
  if (listen(listen_fd, 5) == -1) {
    perror("server: listen");
    close(listen_fd);
    return;
  }

  VLOG(1) << "Server: Waiting for client connection on " << SOCKET_PATH;

  // Perform warm-up runs
  VLOG(1) << "Server: Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    conn_fd = accept(listen_fd, NULL, NULL);
    if (conn_fd == -1) {
      perror("server: accept during warmup");
      close(listen_fd);
      return;
    }

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    while (total_received < DATA_SIZE) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
      if (bytes_received == -1) {
        perror("server: recv during warmup");
        break;
      }
      if (bytes_received == 0) {
        break;
      }
      total_received += bytes_received;
    }
    close(conn_fd);
    VLOG(1) << "Server: Warm-up " << warmup + 1 << "/3 completed";
  }
  VLOG(1) << "Server: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    // Accept a client connection for each iteration
    conn_fd = accept(listen_fd, NULL, NULL);
    if (conn_fd == -1) {
      perror("server: accept");
      close(listen_fd);
      return;
    }

    VLOG(1) << "Server: Client connected. Receiving data... (Iteration "
            << iteration + 1 << "/" << NUM_ITERATIONS << ")";

    std::vector<char> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Receive data until DATA_SIZE is reached
    while (total_received < DATA_SIZE) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
      if (bytes_received == -1) {
        perror("server: recv");
        break;
      }
      if (bytes_received == 0) {
        VLOG(1) << "Server: Client disconnected prematurely.";
        break;
      }
      total_received += bytes_received;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Server: Time taken: " << elapsed_time.count() << " seconds.";

    // Close connection for this iteration
    close(conn_fd);
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
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Server";
  }

  // Close sockets and remove the socket file
  close(listen_fd);
  remove(SOCKET_PATH.c_str());
  VLOG(1) << "Server: Exiting.";
}

void client_process() {
  // Perform warm-up runs
  VLOG(1) << "Client: Performing warm-up runs...";
  for (int warmup = 0; warmup < 3; ++warmup) {
    int sock_fd;
    struct sockaddr_un addr;

    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd == -1) {
      perror("client: socket during warmup");
      return;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

    while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      if (errno == ENOENT) {
        sleep(1);
      } else {
        perror("client: connect during warmup");
        close(sock_fd);
        return;
      }
    }

    std::vector<char> send_buffer(BUFFER_SIZE, 'W'); // 'W' for warmup
    size_t total_sent = 0;
    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_sent = send(sock_fd, send_buffer.data(), bytes_to_send, 0);
      if (bytes_sent == -1) {
        perror("client: send during warmup");
        break;
      }
      total_sent += bytes_sent;
    }
    close(sock_fd);
    VLOG(1) << "Client: Warm-up " << warmup + 1 << "/3 completed";
    usleep(100000); // 100ms delay between warmup runs
  }
  VLOG(1) << "Client: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    int sock_fd;
    struct sockaddr_un addr;

    // Create a Unix domain stream socket
    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd == -1) {
      perror("client: socket");
      return;
    }

    // Configure the socket address
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

    // Connect to the server
    VLOG(1) << "Client: Connecting to server on " << SOCKET_PATH
            << " (Iteration " << iteration + 1 << "/" << NUM_ITERATIONS << ")";
    while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      if (errno == ENOENT) {
        // Server socket not found, wait and retry
        LOG(ERROR)
            << "Client: Server socket not found, retrying in 1 second...";
        sleep(1);
      } else {
        perror("client: connect");
        close(sock_fd);
        return;
      }
    }

    VLOG(1) << "Client: Connected to server. Sending data...";

    std::vector<char> send_buffer(BUFFER_SIZE, 'A'); // Fill buffer with 'A'
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Send data until DATA_SIZE is reached
    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send =
          std::min((size_t)BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_sent = send(sock_fd, send_buffer.data(), bytes_to_send, 0);
      if (bytes_sent == -1) {
        perror("client: send");
        break;
      }
      total_sent += bytes_sent;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    // Close the socket
    close(sock_fd);

    // Small delay between iterations to allow server to reset
    if (iteration < NUM_ITERATIONS - 1) {
      usleep(100000); // 100ms delay
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

  if (average_duration > 0) {
    double bandwidth_gibps =
        DATA_SIZE / (average_duration * 1024.0 * 1024.0 * 1024.0);
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Client";
  }

  VLOG(1) << "Client: Exiting.";
}

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    return 1;
  }

  if (pid == 0) {
    // Child process (client)
    client_process();
  } else {
    // Parent process (server)
    server_process();
  }

  return 0;
}
