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

#include "absl/strings/str_cat.h"
#include "common.h"

ABSL_FLAG(uint64_t, data_size, 128 * (1 << 20),
          "Size of data to transfer in bytes");

const std::string SOCKET_PATH = "/tmp/unix_domain_socket_test.sock";
constexpr size_t DEFAULT_BUFFER_SIZE = (1 << 20);

void receive_process(uint64_t buffer_size, int num_warmups, int num_iterations,
                     uint64_t data_size) {
  std::vector<double> durations;
  std::vector<uint8_t> read_data(data_size, 0x00);

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
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
    if (listen(listen_fd, 0) == -1) {
      LOG(ERROR) << "Failed to listen on socket " << SOCKET_PATH;
      close(listen_fd);
      return;
    }

    VLOG(1) << ReceivePrefix(iteration) << "Waiting for sender connection on "
            << SOCKET_PATH;

    conn_fd = accept(listen_fd, NULL, NULL);
    CHECK(conn_fd != -1) << "Failed to accept connection";

    VLOG(1) << ReceivePrefix(iteration) << "Sender connected.";
    VLOG(1) << ReceivePrefix(iteration) << "Begin receiving data.";
    std::vector<uint8_t> recv_buffer(buffer_size);
    size_t total_received = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (total_received < data_size) {
      ssize_t bytes_received =
          recv(conn_fd, recv_buffer.data(), buffer_size, 0);
      CHECK(bytes_received >= 0) << "Failed to receive data";
      if (bytes_received == 0) {
        VLOG(1) << ReceivePrefix(iteration)
                << "Sender disconnected prematurely.";
        break;
      }
      total_received += bytes_received;
      memcpy(read_data.data() + total_received - bytes_received,
             recv_buffer.data(), bytes_received);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    close(conn_fd);
    close(listen_fd);
    remove(SOCKET_PATH.c_str());
    VLOG(1) << ReceivePrefix(iteration) << "Finished receiving data.";

    verifyDataReceived(read_data, data_size);
    if (num_warmups <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << ReceivePrefix(iteration)
              << "Time taken: " << elapsed_time.count() << " seconds.";
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << " Receive bandwidth: " << bandwidth / (1 << 30)
            << " GiByte/sec.";
}

void send_process(uint64_t buffer_size, int num_warmups, int num_iterations,
                  uint64_t data_size) {
  std::vector<uint8_t> data_to_send = generateDataToSend(data_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    int sock_fd;
    struct sockaddr_un addr;

    sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    CHECK(sock_fd != -1) << "Failed to create socket";

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH.c_str(), sizeof(addr.sun_path) - 1);

    VLOG(1) << SendPrefix(iteration) << "Connecting to receiver on "
            << SOCKET_PATH;
    while (connect(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
      if (errno == ENOENT || errno == ECONNREFUSED) {
        LOG(ERROR) << SendPrefix(iteration)
                   << "Connection failed: " << strerror(errno)
                   << ". Retrying...";
        sleep(1);
      } else {
        LOG(ERROR) << SendPrefix(iteration)
                   << "Unexpected error: " << strerror(errno);
        close(sock_fd);
        return;
      }
    }

    VLOG(1) << SendPrefix(iteration) << "Begin data transfer.";
    size_t total_sent = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (total_sent < data_size) {
      size_t bytes_to_send = std::min(buffer_size, data_size - total_sent);
      ssize_t bytes_sent =
          send(sock_fd, data_to_send.data() + total_sent, bytes_to_send, 0);
      if (bytes_sent == -1) {
        LOG(ERROR) << "Send: Failed to send data: " << strerror(errno);
        break;
      }
      total_sent += bytes_sent;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    VLOG(1) << SendPrefix(iteration) << "Finish data transfer";

    if (num_warmups <= iteration) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << SendPrefix(iteration) << "Time taken: " << elapsed_time.count()
              << " seconds.";
    }
    close(sock_fd);
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << " Send bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");
ABSL_FLAG(int, buffer_size, DEFAULT_BUFFER_SIZE,
          "Size of the buffer used for sending and receiving data.");
ABSL_FLAG(int, num_iterations, 10,
          "Number of measurement iterations (minimum 3)");
ABSL_FLAG(int, num_warmups, 3, "Number of warmup iterations");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Unix Domain Socket Benchmark");
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
  int buffer_size = absl::GetFlag(FLAGS_buffer_size);

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  pid_t pid = fork();
  CHECK(pid != -1) << "Failed to fork process";

  if (pid == 0) {
    send_process(buffer_size, num_warmups, num_iterations, data_size);
  } else {
    receive_process(buffer_size, num_warmups, num_iterations, data_size);
  }

  return 0;
}
