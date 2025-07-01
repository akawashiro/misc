#include <algorithm>    // For std::min
#include <arpa/inet.h>  // For inet_addr
#include <chrono>       // For time measurement
#include <cstring>      // For memset
#include <netinet/in.h> // For sockaddr_in and IPPROTO_TCP
#include <string>
#include <sys/socket.h>
#include <unistd.h> // For fork, close
#include <vector>

#include "absl/log/log.h"

const int PORT = 12345; // Port number for TCP communication
const std::string LOOPBACK_IP = "127.0.0.1"; // Localhost IP address
const size_t DATA_SIZE = 128 * 1024 * 1024;  // 128 MiB
const int BUFFER_SIZE = 4096;                // 4KB buffer for send/recv

void server_process() {
  int listen_fd, conn_fd;
  struct sockaddr_in server_addr, client_addr;
  socklen_t client_len = sizeof(client_addr);

  // Create a TCP socket
  listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd == -1) {
    perror("server: socket");
    return;
  }

  // Allow immediate reuse of the port after the program exits
  int optval = 1;
  if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &optval,
                 sizeof(optval)) == -1) {
    perror("server: setsockopt SO_REUSEADDR");
    close(listen_fd);
    return;
  }

  // Configure server address
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = inet_addr(LOOPBACK_IP.c_str());
  server_addr.sin_port = htons(PORT);

  // Bind the socket to the specified IP address and port
  if (bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) ==
      -1) {
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

  LOG(INFO) << "Server: Listening on " << LOOPBACK_IP << ":" << PORT;

  // Accept a client connection
  conn_fd = accept(listen_fd, (struct sockaddr *)&client_addr, &client_len);
  if (conn_fd == -1) {
    perror("server: accept");
    close(listen_fd);
    return;
  }

  LOG(INFO) << "Server: Client connected from "
            << inet_ntoa(client_addr.sin_addr) << ":"
            << ntohs(client_addr.sin_port) << ". Receiving data...";

  std::vector<char> recv_buffer(BUFFER_SIZE);
  size_t total_received = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  // Receive data until DATA_SIZE is reached
  while (total_received < DATA_SIZE) {
    ssize_t bytes_received = recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
    if (bytes_received == -1) {
      perror("server: recv");
      break;
    }
    if (bytes_received == 0) {
      LOG(INFO) << "Server: Client disconnected prematurely.";
      break;
    }
    total_received += bytes_received;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  LOG(INFO) << "Server: Received "
            << total_received / (1024.0 * 1024.0 * 1024.0) << " GiB of data.";
  LOG(INFO) << "Server: Time taken: " << elapsed_time.count() << " seconds.";
  if (elapsed_time.count() > 0) {
    double bandwidth_gibps = total_received / (elapsed_time.count() * 1024.0 *
                                               1024.0 * 1024.0); // GiB/s
    LOG(INFO) << "Server: Bandwidth: " << bandwidth_gibps << " GiB/s";
  }

  // Close sockets
  close(conn_fd);
  close(listen_fd);
  LOG(INFO) << "Server: Exiting.";
}

void client_process() {
  int sock_fd;
  struct sockaddr_in server_addr;

  // Create a TCP socket
  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd == -1) {
    perror("client: socket");
    return;
  }

  // Configure server address to connect to
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = inet_addr(LOOPBACK_IP.c_str());
  server_addr.sin_port = htons(PORT);

  // Connect to the server
  LOG(INFO) << "Client: Connecting to server at " << LOOPBACK_IP << ":" << PORT;
  while (connect(sock_fd, (struct sockaddr *)&server_addr,
                 sizeof(server_addr)) == -1) {
    perror("client: connect (retrying in 1 second)");
    sleep(1); // Wait a bit if server isn't ready yet
  }

  LOG(INFO) << "Client: Connected to server. Sending data...";

  std::vector<char> send_buffer(BUFFER_SIZE, 'B'); // Fill buffer with 'B'
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

  // Ensure all data is sent before closing the socket (optional, but good
  // practice for benchmarks) shutdown(sock_fd, SHUT_WR);

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  LOG(INFO) << "Client: Sent " << total_sent / (1024.0 * 1024.0 * 1024.0)
            << " GiB of data.";
  LOG(INFO) << "Client: Time taken: " << elapsed_time.count() << " seconds.";
  if (elapsed_time.count() > 0) {
    double bandwidth_gibps =
        total_sent / (elapsed_time.count() * 1024.0 * 1024.0 * 1024.0); // GiB/s
    LOG(INFO) << "Client: Bandwidth: " << bandwidth_gibps << " GiB/s";
  }

  // Close the socket
  close(sock_fd);
  LOG(INFO) << "Client: Exiting.";
}

int main() {
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
