#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <numeric>
#include <semaphore.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
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

const std::string SHM_NAME = "/shm_bandwidth_test";
const std::string SEM_WRITER_NAME = "/sem_writer_bandwidth";
const std::string SEM_READER_NAME = "/sem_reader_bandwidth";
constexpr size_t BUFFER_SIZE = (1 << 20);

struct SharedBuffer {
  size_t data_size;
  bool transfer_complete;
  char data[BUFFER_SIZE];
};

void cleanup_resources() {
  shm_unlink(SHM_NAME.c_str());
  sem_unlink(SEM_WRITER_NAME.c_str());
  sem_unlink(SEM_READER_NAME.c_str());
}

size_t send_data(SharedBuffer *shared_buffer, sem_t *sem_writer,
                 sem_t *sem_reader, const std::vector<char> &data,
                 uint64_t data_size) {
  size_t total_sent = 0;

  while (total_sent < data_size) {
    // Wait for permission to write
    sem_wait(sem_writer);

    size_t bytes_to_send = std::min(BUFFER_SIZE, data_size - total_sent);
    shared_buffer->data_size = bytes_to_send;
    memcpy(shared_buffer->data, data.data(), bytes_to_send);
    total_sent += bytes_to_send;

    if (total_sent >= data_size) {
      shared_buffer->transfer_complete = true;
    }

    // Signal reader that data is ready
    sem_post(sem_reader);
  }

  return total_sent;
}

size_t receive_data(SharedBuffer *shared_buffer, sem_t *sem_writer,
                    sem_t *sem_reader, uint64_t data_size) {
  size_t total_received = 0;

  while (total_received < data_size) {
    // Wait for writer to signal data is ready
    sem_wait(sem_reader);

    if (shared_buffer->transfer_complete) {
      break;
    }

    total_received += shared_buffer->data_size;

    // Signal writer that data has been read
    sem_post(sem_writer);
  }

  return total_received;
}

void receive_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Clean up any existing resources
  cleanup_resources();

  // Create shared memory
  int shm_fd = shm_open(SHM_NAME.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    perror("server: shm_open");
    return;
  }

  if (ftruncate(shm_fd, sizeof(SharedBuffer)) == -1) {
    perror("server: ftruncate");
    close(shm_fd);
    cleanup_resources();
    return;
  }

  SharedBuffer *shared_buffer = static_cast<SharedBuffer *>(
      mmap(NULL, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED,
           shm_fd, 0));
  if (shared_buffer == MAP_FAILED) {
    perror("server: mmap");
    close(shm_fd);
    cleanup_resources();
    return;
  }

  // Create semaphores for synchronization
  sem_t *sem_writer = sem_open(SEM_WRITER_NAME.c_str(), O_CREAT, 0666, 1);
  sem_t *sem_reader = sem_open(SEM_READER_NAME.c_str(), O_CREAT, 0666, 0);
  if (sem_writer == SEM_FAILED || sem_reader == SEM_FAILED) {
    perror("server: sem_open");
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    cleanup_resources();
    return;
  }

  VLOG(1) << "Receiver: Shared memory and semaphores initialized";

  // Perform warm-up runs
  VLOG(1) << "Receiver: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    shared_buffer->transfer_complete = false;

    sem_post(sem_writer);
    receive_data(shared_buffer, sem_writer, sem_reader, data_size);
    VLOG(1) << "Receiver: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Receiver: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Receiver: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    shared_buffer->transfer_complete = false;

    sem_post(sem_writer);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Receive data until data_size is reached
    receive_data(shared_buffer, sem_writer, sem_reader, data_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Receiver: Time taken: " << elapsed_time.count() << " seconds.";
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Cleanup
  sem_close(sem_writer);
  sem_close(sem_reader);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  cleanup_resources();
  VLOG(1) << "Receiver: Exiting.";
}

void send_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Give server time to initialize
  usleep(100000); // 100ms

  // Open existing shared memory
  int shm_fd = shm_open(SHM_NAME.c_str(), O_RDWR, 0666);
  if (shm_fd == -1) {
    perror("client: shm_open");
    return;
  }

  SharedBuffer *shared_buffer = static_cast<SharedBuffer *>(
      mmap(NULL, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED,
           shm_fd, 0));
  if (shared_buffer == MAP_FAILED) {
    perror("client: mmap");
    close(shm_fd);
    return;
  }

  // Open existing semaphores
  sem_t *sem_writer = sem_open(SEM_WRITER_NAME.c_str(), 0);
  sem_t *sem_reader = sem_open(SEM_READER_NAME.c_str(), 0);
  if (sem_writer == SEM_FAILED || sem_reader == SEM_FAILED) {
    perror("client: sem_open");
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    return;
  }

  // Perform warm-up runs
  VLOG(1) << "Sender: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    std::vector<char> data(BUFFER_SIZE, 'W'); // 'W' for warmup
    send_data(shared_buffer, sem_writer, sem_reader, data, data_size);
    VLOG(1) << "Sender: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
    usleep(100000); // 100ms delay between warmup runs
  }
  VLOG(1) << "Sender: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Sender: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    std::vector<char> data(BUFFER_SIZE, 'A'); // Fill buffer with 'A'

    auto start_time = std::chrono::high_resolution_clock::now();

    // Send data until data_size is reached
    send_data(shared_buffer, sem_writer, sem_reader, data, data_size);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());
    VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";

    // Small delay between iterations
    if (iteration < num_iterations - 1) {
      usleep(100000); // 100ms delay
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  // Cleanup
  sem_close(sem_writer);
  sem_close(sem_reader);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  VLOG(1) << "Sender: Exiting.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Shared Memory Bandwidth Benchmark");
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
  } else {
    // Default to level 1 for backward compatibility
    absl::SetGlobalVLogLevel(1);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    return 1;
  }

  if (pid == 0) {
    send_process(num_warmups, num_iterations, data_size);
  } else {
    receive_process(num_warmups, num_iterations, data_size);

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
