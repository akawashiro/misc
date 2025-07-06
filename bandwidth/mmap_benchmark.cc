#include "mmap_benchmark.h"

#include <algorithm> // For std::min
#include <chrono>
#include <cstring>
#include <fcntl.h> // For open
#include <numeric>
#include <string>
#include <sys/mman.h> // For mmap, munmap
#include <sys/stat.h> // For fstat
#include <sys/wait.h> // For wait
#include <unistd.h>   // For fork, close, ftruncate
#include <vector>

#include "absl/log/globals.h"
#include "absl/log/log.h"

#include "common.h"

namespace {
const std::string MMAP_FILE_PATH = "/tmp/mmap_bandwidth_test.dat";
const int BUFFER_SIZE = 4096; // 4KB buffer for read/write

// Structure for synchronization between processes
struct sync_data {
  volatile bool sender_ready;
  volatile bool receiver_ready;
  volatile size_t bytes_written;
  volatile bool sender_done;
  volatile int current_iteration;
};

void send_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Create and open the memory-mapped file
  int fd = open(MMAP_FILE_PATH.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  if (fd == -1) {
    LOG(ERROR) << "send: open: " << strerror(errno);
    return;
  }

  // Size the file to accommodate data and sync structure
  size_t total_size = data_size + sizeof(sync_data);
  if (ftruncate(fd, total_size) == -1) {
    LOG(ERROR) << "send: ftruncate: " << strerror(errno);
    close(fd);
    return;
  }

  // Map the file into memory
  void *mapped_region =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped_region == MAP_FAILED) {
    LOG(ERROR) << "send: mmap: " << strerror(errno);
    close(fd);
    return;
  }

  // Set up pointers
  char *data_region = static_cast<char *>(mapped_region);
  sync_data *sync = reinterpret_cast<sync_data *>(data_region + data_size);

  // Initialize sync structure
  sync->sender_ready = false;
  sync->receiver_ready = false;
  sync->bytes_written = 0;
  sync->sender_done = false;
  sync->current_iteration = -1;

  // Perform warm-up runs
  VLOG(1) << "Sender: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    sync->current_iteration = warmup;
    sync->bytes_written = 0;
    sync->sender_ready = true;

    // Wait for receiver to be ready
    while (!sync->receiver_ready) {
      usleep(1); // Small delay to avoid busy waiting
    }

    // Write data in chunks
    for (size_t offset = 0; offset < data_size; offset += BUFFER_SIZE) {
      size_t chunk_size = std::min((size_t)BUFFER_SIZE, data_size - offset);
      memset(data_region + offset, 'W', chunk_size); // 'W' for warmup
      sync->bytes_written = offset + chunk_size;
    }

    sync->sender_ready = false;
    sync->receiver_ready = false;
    VLOG(1) << "Sender: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Sender: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Sender: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    sync->current_iteration = iteration;
    sync->bytes_written = 0;
    sync->sender_ready = true;

    // Wait for receiver to be ready
    while (!sync->receiver_ready) {
      usleep(1);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data in chunks
    for (size_t offset = 0; offset < data_size; offset += BUFFER_SIZE) {
      size_t chunk_size = std::min((size_t)BUFFER_SIZE, data_size - offset);
      memset(data_region + offset, 'A', chunk_size); // Fill with 'A'
      sync->bytes_written = offset + chunk_size;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";

    sync->sender_ready = false;
    sync->receiver_ready = false;
  }

  sync->sender_done = true;

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  // Clean up
  munmap(mapped_region, total_size);
  close(fd);
  VLOG(1) << "Sender: Exiting.";
}

void receive_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Give sender time to create the file
  usleep(100000); // 100ms

  // Open the memory-mapped file
  int fd = open(MMAP_FILE_PATH.c_str(), O_RDWR);
  if (fd == -1) {
    LOG(ERROR) << "receive: open: " << strerror(errno);
    return;
  }

  // Get file size
  struct stat file_stat;
  if (fstat(fd, &file_stat) == -1) {
    LOG(ERROR) << "receive: fstat: " << strerror(errno);
    close(fd);
    return;
  }

  size_t total_size = file_stat.st_size;

  // Map the file into memory
  void *mapped_region =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped_region == MAP_FAILED) {
    LOG(ERROR) << "receive: mmap: " << strerror(errno);
    close(fd);
    return;
  }

  // Set up pointers
  char *data_region = static_cast<char *>(mapped_region);
  sync_data *sync = reinterpret_cast<sync_data *>(data_region + data_size);

  // Perform warm-up runs
  VLOG(1) << "Receiver: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    // Wait for sender to be ready
    while (!sync->sender_ready || sync->current_iteration != warmup) {
      usleep(1);
    }

    sync->receiver_ready = true;

    // Read data by waiting for sender to complete each chunk
    size_t last_read = 0;
    while (last_read < data_size) {
      while (sync->bytes_written <= last_read && sync->sender_ready) {
        usleep(1); // Wait for more data
      }
      if (!sync->sender_ready)
        break; // Sender finished this iteration
      last_read = sync->bytes_written;
    }

    VLOG(1) << "Receiver: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Receiver: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Receiver: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    // Wait for sender to be ready
    while (!sync->sender_ready || sync->current_iteration != iteration) {
      usleep(1);
    }

    sync->receiver_ready = true;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data by waiting for sender to complete each chunk
    size_t last_read = 0;
    while (last_read < data_size) {
      while (sync->bytes_written <= last_read && sync->sender_ready) {
        usleep(1); // Wait for more data
      }
      if (!sync->sender_ready)
        break; // Sender finished this iteration
      last_read = sync->bytes_written;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Receiver: Time taken: " << elapsed_time.count() << " seconds.";
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Clean up
  munmap(mapped_region, total_size);
  close(fd);
  VLOG(1) << "Receiver: Exiting.";
}

} // namespace

int run_mmap_benchmark(int num_iterations, int num_warmups, uint64_t data_size) {
  // Remove the file if it exists from a previous run
  unlink(MMAP_FILE_PATH.c_str());

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

    // Wait for child process to complete
    int status;
    wait(&status);

    // Clean up the temporary file
    unlink(MMAP_FILE_PATH.c_str());
  }

  return 0;
}
