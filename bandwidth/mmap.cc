#include <algorithm> // For std::min
#include <chrono>
#include <cstring>
#include <fcntl.h> // For open
#include <numeric>
#include <optional>
#include <string>
#include <sys/mman.h> // For mmap, munmap
#include <sys/stat.h> // For fstat
#include <sys/wait.h> // For wait
#include <unistd.h>   // For fork, close, ftruncate
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

const std::string MMAP_FILE_PATH = "/tmp/mmap_bandwidth_test.dat";
const int BUFFER_SIZE = 4096; // 4KB buffer for read/write

// Structure for synchronization between processes
struct sync_data {
  volatile bool writer_ready;
  volatile bool reader_ready;
  volatile size_t bytes_written;
  volatile bool writer_done;
  volatile int current_iteration;
};

void writer_process(int num_warmups, int num_iterations) {
  // Create and open the memory-mapped file
  int fd = open(MMAP_FILE_PATH.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0666);
  if (fd == -1) {
    perror("writer: open");
    return;
  }

  // Size the file to accommodate data and sync structure
  size_t total_size = DATA_SIZE + sizeof(sync_data);
  if (ftruncate(fd, total_size) == -1) {
    perror("writer: ftruncate");
    close(fd);
    return;
  }

  // Map the file into memory
  void *mapped_region =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped_region == MAP_FAILED) {
    perror("writer: mmap");
    close(fd);
    return;
  }

  // Set up pointers
  char *data_region = static_cast<char *>(mapped_region);
  sync_data *sync = reinterpret_cast<sync_data *>(data_region + DATA_SIZE);

  // Initialize sync structure
  sync->writer_ready = false;
  sync->reader_ready = false;
  sync->bytes_written = 0;
  sync->writer_done = false;
  sync->current_iteration = -1;

  // Perform warm-up runs
  VLOG(1) << "Writer: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    sync->current_iteration = warmup;
    sync->bytes_written = 0;
    sync->writer_ready = true;

    // Wait for reader to be ready
    while (!sync->reader_ready) {
      usleep(1); // Small delay to avoid busy waiting
    }

    // Write data in chunks
    for (size_t offset = 0; offset < DATA_SIZE; offset += BUFFER_SIZE) {
      size_t chunk_size = std::min((size_t)BUFFER_SIZE, DATA_SIZE - offset);
      memset(data_region + offset, 'W', chunk_size); // 'W' for warmup
      sync->bytes_written = offset + chunk_size;
    }

    sync->writer_ready = false;
    sync->reader_ready = false;
    VLOG(1) << "Writer: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Writer: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Writer: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    sync->current_iteration = iteration;
    sync->bytes_written = 0;
    sync->writer_ready = true;

    // Wait for reader to be ready
    while (!sync->reader_ready) {
      usleep(1);
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Write data in chunks
    for (size_t offset = 0; offset < DATA_SIZE; offset += BUFFER_SIZE) {
      size_t chunk_size = std::min((size_t)BUFFER_SIZE, DATA_SIZE - offset);
      memset(data_region + offset, 'A', chunk_size); // Fill with 'A'
      sync->bytes_written = offset + chunk_size;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Writer: Time taken: " << elapsed_time.count() << " seconds.";

    sync->writer_ready = false;
    sync->reader_ready = false;
  }

  sync->writer_done = true;

  double bandwidth = calculateBandwidth(durations, num_iterations);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Writer";

  // Clean up
  munmap(mapped_region, total_size);
  close(fd);
  VLOG(1) << "Writer: Exiting.";
}

void reader_process(int num_warmups, int num_iterations) {
  // Give writer time to create the file
  usleep(100000); // 100ms

  // Open the memory-mapped file
  int fd = open(MMAP_FILE_PATH.c_str(), O_RDWR);
  if (fd == -1) {
    perror("reader: open");
    return;
  }

  // Get file size
  struct stat file_stat;
  if (fstat(fd, &file_stat) == -1) {
    perror("reader: fstat");
    close(fd);
    return;
  }

  size_t total_size = file_stat.st_size;

  // Map the file into memory
  void *mapped_region =
      mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped_region == MAP_FAILED) {
    perror("reader: mmap");
    close(fd);
    return;
  }

  // Set up pointers
  char *data_region = static_cast<char *>(mapped_region);
  sync_data *sync = reinterpret_cast<sync_data *>(data_region + DATA_SIZE);

  // Perform warm-up runs
  VLOG(1) << "Reader: Performing warm-up runs...";
  for (int warmup = 0; warmup < num_warmups; ++warmup) {
    // Wait for writer to be ready
    while (!sync->writer_ready || sync->current_iteration != warmup) {
      usleep(1);
    }

    sync->reader_ready = true;

    // Read data by waiting for writer to complete each chunk
    size_t last_read = 0;
    while (last_read < DATA_SIZE) {
      while (sync->bytes_written <= last_read && sync->writer_ready) {
        usleep(1); // Wait for more data
      }
      if (!sync->writer_ready)
        break; // Writer finished this iteration
      last_read = sync->bytes_written;
    }

    VLOG(1) << "Reader: Warm-up " << warmup + 1 << "/" << num_warmups
            << " completed";
  }
  VLOG(1) << "Reader: Warm-up complete. Starting measurements...";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_iterations; ++iteration) {
    VLOG(1) << "Reader: Starting iteration " << iteration + 1 << "/"
            << num_iterations;

    // Wait for writer to be ready
    while (!sync->writer_ready || sync->current_iteration != iteration) {
      usleep(1);
    }

    sync->reader_ready = true;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Read data by waiting for writer to complete each chunk
    size_t last_read = 0;
    while (last_read < DATA_SIZE) {
      while (sync->bytes_written <= last_read && sync->writer_ready) {
        usleep(1); // Wait for more data
      }
      if (!sync->writer_ready)
        break; // Writer finished this iteration
      last_read = sync->bytes_written;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    durations.push_back(elapsed_time.count());

    VLOG(1) << "Reader: Time taken: " << elapsed_time.count() << " seconds.";
  }

  double bandwidth = calculateBandwidth(durations, num_iterations);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Reader";

  // Clean up
  munmap(mapped_region, total_size);
  close(fd);
  VLOG(1) << "Reader: Exiting.";
}

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Memory-Mapped File Bandwidth Benchmark");
  absl::ParseCommandLine(argc, argv);

  // Get values from command line flags
  int num_iterations = absl::GetFlag(FLAGS_num_iterations);
  int num_warmups = absl::GetFlag(FLAGS_num_warmups);

  // Validate num_iterations
  if (num_iterations < 3) {
    LOG(ERROR) << "num_iterations must be at least 3, got: " << num_iterations;
    return 1;
  }

  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }

  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  // Remove the file if it exists from a previous run
  unlink(MMAP_FILE_PATH.c_str());

  pid_t pid = fork();

  if (pid == -1) {
    perror("fork");
    return 1;
  }

  if (pid == 0) {
    // Child process (writer)
    writer_process(num_warmups, num_iterations);
  } else {
    // Parent process (reader)
    reader_process(num_warmups, num_iterations);

    // Wait for child process to complete
    int status;
    wait(&status);

    // Clean up the temporary file
    unlink(MMAP_FILE_PATH.c_str());
  }

  return 0;
}
