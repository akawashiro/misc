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

#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

const std::string SHM_NAME = "/shm_bandwidth_test";
const std::string SEM_WRITER_NAME = "/sem_writer_bandwidth";
const std::string SEM_READER_NAME = "/sem_reader_bandwidth";
constexpr size_t DATA_SIZE = 128 * (1 << 20);
constexpr size_t BUFFER_SIZE = (1 << 20);
constexpr int NUM_ITERATIONS = 10;

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
                 sem_t *sem_reader, const std::vector<char> &data) {
  size_t total_sent = 0;

  while (total_sent < DATA_SIZE) {
    // Wait for permission to write
    sem_wait(sem_writer);

    size_t bytes_to_send = std::min(BUFFER_SIZE, DATA_SIZE - total_sent);
    shared_buffer->data_size = bytes_to_send;
    memcpy(shared_buffer->data, data.data(), bytes_to_send);
    total_sent += bytes_to_send;

    if (total_sent >= DATA_SIZE) {
      shared_buffer->transfer_complete = true;
    }

    // Signal reader that data is ready
    sem_post(sem_reader);
  }

  return total_sent;
}

size_t receive_data(SharedBuffer *shared_buffer, sem_t *sem_writer,
                    sem_t *sem_reader) {
  size_t total_received = 0;

  while (total_received < DATA_SIZE) {
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

void receive_process() {
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
  for (int warmup = 0; warmup < 3; ++warmup) {
    shared_buffer->transfer_complete = false;

    sem_post(sem_writer);
    receive_data(shared_buffer, sem_writer, sem_reader);
    VLOG(1) << "Receiver: Warm-up " << warmup + 1 << "/3 completed";
  }
  VLOG(1) << "Receiver: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    VLOG(1) << "Receiver: Starting iteration " << iteration + 1 << "/"
            << NUM_ITERATIONS;

    shared_buffer->transfer_complete = false;

    sem_post(sem_writer);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Receive data until DATA_SIZE is reached
    receive_data(shared_buffer, sem_writer, sem_reader);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Receiver: Time taken: " << elapsed_time.count() << " seconds.";
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
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";
  }

  // Cleanup
  sem_close(sem_writer);
  sem_close(sem_reader);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  cleanup_resources();
  VLOG(1) << "Receiver: Exiting.";
}

void send_process() {
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
  for (int warmup = 0; warmup < 3; ++warmup) {
    std::vector<char> data(BUFFER_SIZE, 'W'); // 'W' for warmup
    send_data(shared_buffer, sem_writer, sem_reader, data);
    VLOG(1) << "Sender: Warm-up " << warmup + 1 << "/3 completed";
    usleep(100000); // 100ms delay between warmup runs
  }
  VLOG(1) << "Sender: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
    VLOG(1) << "Sender: Starting iteration " << iteration + 1 << "/"
            << NUM_ITERATIONS;

    std::vector<char> data(BUFFER_SIZE, 'A'); // Fill buffer with 'A'

    auto start_time = std::chrono::high_resolution_clock::now();

    // Send data until DATA_SIZE is reached
    send_data(shared_buffer, sem_writer, sem_reader, data);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());
    VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";

    // Small delay between iterations
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
    LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";
  }

  // Cleanup
  sem_close(sem_writer);
  sem_close(sem_reader);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  VLOG(1) << "Sender: Exiting.";
}

int main() {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::SetGlobalVLogLevel(1);
  absl::InitializeLog();

  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    return 1;
  }

  if (pid == 0) {
    send_process();
  } else {
    receive_process();

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
