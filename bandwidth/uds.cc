#include <sys/socket.h>
#include <sys/un.h>

#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "common.h"

const std::string SOCKET_PATH = "/tmp/unix_domain_socket_test.sock";
constexpr size_t BUFFER_SIZE = (1 << 20);

void receive_process() {
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
    LOG(ERROR) << "Failed to bind socket to " << SOCKET_PATH;
    close(listen_fd);
    return;
  }

  // Listen for incoming connections
  if (listen(listen_fd, 5) == -1) {
    LOG(ERROR) << "Failed to listen on socket " << SOCKET_PATH;
    close(listen_fd);
    return;
  }

  VLOG(1) << "Receive: Waiting for client connection on " << SOCKET_PATH;

  std::vector<double> durations;
  std::vector<uint8_t> read_data(DATA_SIZE, 0x00);

  for (int iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS;
       ++iteration) {
    conn_fd = accept(listen_fd, NULL, NULL);
    CHECK(conn_fd != -1) << "Failed to accept connection";

    VLOG(1) << "Receive: Client connected.";
    VLOG(1) << "Reader: Begin receiving data.";
    std::vector<uint8_t> recv_buffer(BUFFER_SIZE);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (total_received < DATA_SIZE) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), BUFFER_SIZE, 0);
      CHECK(bytes_received >= 0) << "Failed to receive data";
      if (bytes_received == 0) {
        VLOG(1) << "Receive: Client disconnected prematurely.";
        break;
      }
      total_received += bytes_received;
      memcpy(read_data.data() + total_received - bytes_received,
             recv_buffer.data(), bytes_received);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    close(conn_fd);
    VLOG(1) << "Reader: Finished receiving data.";

    verifyDataReceived(read_data);
    if (NUM_WARMUPS <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Receive: Time taken: " << elapsed_time.count() << " seconds.";
    }
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << " Receive bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec.";

  close(listen_fd);
  remove(SOCKET_PATH.c_str());
}

void send_process() {
  std::vector<uint8_t> data_to_send = generateDataToSend();
  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_WARMUPS + NUM_ITERATIONS;
       ++iteration) {
    int sock_fd;
    struct sockaddr_un addr;

    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(sock_fd != -1) << "Failed to create socket";

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

    VLOG(1) << "Send: Connecting to reader on " << SOCKET_PATH;
    while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      if (errno == ENOENT) {
        LOG(ERROR)
            << "Send: Receive socket not found, retrying in 1 second...";
        sleep(1);
      } else {
        LOG(ERROR) << "Send: Failed to connect to server socket: "
                   << strerror(errno);
        close(sock_fd);
        return;
      }
    }

    VLOG(1) << "Send: Begin data transfer.";
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (total_sent < DATA_SIZE) {
      size_t bytes_to_send = std::min(BUFFER_SIZE, DATA_SIZE - total_sent);
      ssize_t bytes_sent =
          send(sock_fd, data_to_send.data() + total_sent, bytes_to_send, 0);
      if (bytes_sent == -1) {
        LOG(ERROR) << "Send: Failed to send data: " << strerror(errno);
        break;
      }
      total_sent += bytes_sent;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    VLOG(1) << "Send: Finish data transfer";

    if (NUM_WARMUPS <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Send: Time taken: " << elapsed_time.count() << " seconds.";
    }
    close(sock_fd);

    // Caution: We need this sleep to ensure the server has time to reset.
    // Otherwise, the bandwidth of send will be siginificantly lower than the
    // receive.
    sleep(1);
  }

  double bandwidth = calculateBandwidth(durations);
  LOG(INFO) << " Send bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Unix Domain Socket Benchmark");
  absl::ParseCommandLine(argc, argv);

  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  pid_t pid = fork();
  CHECK(pid != -1) << "Failed to fork process";

  if (pid == 0) {
    send_process();
  } else {
    receive_process();
  }

  return 0;
}
