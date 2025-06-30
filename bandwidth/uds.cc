#include <algorithm> // For std::min
#include <chrono>
#include <cstdio> // For remove()
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>

const std::string SOCKET_PATH = "/tmp/unix_domain_socket_test.sock";
const size_t DATA_SIZE = 1024 * 1024 * 1024; // 1 GiB
const int BUFFER_SIZE = 4096;                // 4KB buffer for send/recv

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

  std::cout << "Server: Waiting for client connection on " << SOCKET_PATH
            << std::endl;

  // Accept a client connection
  conn_fd = accept(listen_fd, NULL, NULL);
  if (conn_fd == -1) {
    perror("server: accept");
    close(listen_fd);
    return;
  }

  std::cout << "Server: Client connected. Receiving data..." << std::endl;

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
      std::cout << "Server: Client disconnected prematurely." << std::endl;
      break;
    }
    total_received += bytes_received;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;

  std::cout << "Server: Received "
            << total_received / (1024.0 * 1024.0 * 1024.0) << " GiB of data."
            << std::endl;
  std::cout << "Server: Time taken: " << elapsed_time.count() << " seconds."
            << std::endl;
  if (elapsed_time.count() > 0) {
    double bandwidth_gibps = total_received / (elapsed_time.count() * 1024.0 *
                                               1024.0 * 1024.0); // GiB/s
    std::cout << "Server: Bandwidth: " << bandwidth_gibps << " GiB/s"
              << std::endl;
  }

  // Close sockets and remove the socket file
  close(conn_fd);
  close(listen_fd);
  remove(SOCKET_PATH.c_str());
  std::cout << "Server: Exiting." << std::endl;
}

void client_process() {
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
  std::cout << "Client: Connecting to server on " << SOCKET_PATH << std::endl;
  while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    if (errno == ENOENT) {
      // Server socket not found, wait and retry
      std::cerr << "Client: Server socket not found, retrying in 1 second..."
                << std::endl;
      sleep(1);
    } else {
      perror("client: connect");
      close(sock_fd);
      return;
    }
  }

  std::cout << "Client: Connected to server. Sending data..." << std::endl;

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

  std::cout << "Client: Sent " << total_sent / (1024.0 * 1024.0 * 1024.0)
            << " GiB of data." << std::endl;
  std::cout << "Client: Time taken: " << elapsed_time.count() << " seconds."
            << std::endl;
  if (elapsed_time.count() > 0) {
    double bandwidth_gibps =
        total_sent / (elapsed_time.count() * 1024.0 * 1024.0 * 1024.0); // GiB/s
    std::cout << "Client: Bandwidth: " << bandwidth_gibps << " GiB/s"
              << std::endl;
  }

  // Close the socket
  close(sock_fd);
  std::cout << "Client: Exiting." << std::endl;
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
