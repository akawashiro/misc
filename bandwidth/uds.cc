#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

const std::string SOCKET_PATH = "/tmp/unix_domain_socket_test.sock";
constexpr size_t BUFFER_SIZE = (1 << 20);

void server_process() {
  int listen_fd, conn_fd;
  struct sockaddr_un addr;

  listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(listen_fd != -1) << "Failed to create socket";

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

  std::vector<double> durations;
  std::vector<uint8_t> read_data(DATA_SIZE, 0x00);

  for (int iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS;
       ++iteration) {
    // Accept a client connection for each iteration
    conn_fd = accept(listen_fd, NULL, NULL);
    if (conn_fd == -1) {
      perror("server: accept");
      close(listen_fd);
      return;
    }

    VLOG(1) << "Server: Client connected. Receiving data... (Iteration "
            << iteration + 1 << "/" << NUM_ITERATIONS << ")";

    std::vector<uint8_t> recv_buffer(BUFFER_SIZE);
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
      memcpy(read_data.data() + total_received - bytes_received,
             recv_buffer.data(), bytes_received);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    verifyDataReceived(read_data);
    if (NUM_WARMUPS <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Server: Time taken: " << elapsed_time.count() << " seconds.";
    }
    close(conn_fd);
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec. Server";

  close(listen_fd);
  remove(SOCKET_PATH.c_str());
}

void client_process() {
  std::vector<uint8_t> data_to_send = generateDataToSend();
  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS;
       ++iteration) {
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

    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send = std::min(BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_sent =
          send(sock_fd, data_to_send.data() + total_sent, bytes_to_send, 0);
      if (bytes_sent == -1) {
        perror("client: send");
        break;
      }
      total_sent += bytes_sent;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (NUM_WARMUPS <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
    }
    close(sock_fd);

    // Small delay between iterations to allow server to reset
    if (iteration < NUM_ITERATIONS - 1) {
      usleep(100000); // 100ms delay
    }
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec. Client";
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
