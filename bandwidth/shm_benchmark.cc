#include "shm_benchmark.h"

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "absl/log/log.h"

#include "barrier.h"
#include "common.h"

namespace {
const std::string SHM_NAME = "/shm_bandwidth_test";
const std::string BARRIER_ID = "/shm_benchmark";
constexpr size_t BUFFER_SIZE = (1 << 20);

struct SharedBuffer {
  bool transfer_complete;
  size_t data_size[2];
  char data[2][BUFFER_SIZE];
};

void CleanupResources() {
  shm_unlink(SHM_NAME.c_str());
}

void ReceiveProcess(int num_warmups, int num_iterations, uint64_t data_size) {
  SenseReversingBarrier barrier(2, BARRIER_ID);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;

    if (is_warmup) {
      VLOG(1) << "Receiver: Warm-up " << iteration << "/" << num_warmups;
    } else {
      VLOG(1) << ReceivePrefix(iteration) << "Starting iteration...";
    }

    // Create shared memory
    int shm_fd = shm_open(SHM_NAME.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
      LOG(ERROR) << "receive: shm_open: " << strerror(errno);
      return;
    }

    if (ftruncate(shm_fd, sizeof(SharedBuffer)) == -1) {
      LOG(ERROR) << "receive: ftruncate: " << strerror(errno);
      close(shm_fd);
      CleanupResources();
      return;
    }

    SharedBuffer *shared_buffer = static_cast<SharedBuffer *>(
        mmap(NULL, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED,
             shm_fd, 0));
    if (shared_buffer == MAP_FAILED) {
      LOG(ERROR) << "receive: mmap: " << strerror(errno);
      close(shm_fd);
      CleanupResources();
      return;
    }

    VLOG(1) << "Receiver: Shared memory and semaphores initialized";
    barrier.Wait();

    shared_buffer->transfer_complete = false;
    std::vector<uint8_t> received_data(data_size, 0);

    barrier.Wait();
    auto start_time = std::chrono::high_resolution_clock::now();

    auto end_time = std::chrono::high_resolution_clock::now();
    barrier.Wait();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());

      VLOG(1) << "Receiver: Time taken: " << elapsed_time.count() * 1000
              << " ms.";
    }

    // Verify received data (always, even during warmup)
    if (!VerifyDataReceived(received_data, data_size)) {
      LOG(ERROR) << ReceivePrefix(iteration) << "Data verification failed!";
    } else {
      VLOG(1) << ReceivePrefix(iteration) << "Data verification passed.";
    }

    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    CleanupResources();
  }

  double bandwidth = CalculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Receive bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec.";
}

void SendProcess(int num_warmups, int num_iterations, uint64_t data_size,
                 uint64_t buffer_size) {
  SenseReversingBarrier barrier(2, BARRIER_ID);
  std::vector<uint8_t> data_to_send = GenerateDataToSend(data_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;

    if (is_warmup) {
      VLOG(1) << "Sender: Warm-up " << iteration << "/" << num_warmups;
    } else {
      VLOG(1) << SendPrefix(iteration) << "Starting iteration...";
    }

    barrier.Wait();
    // Open existing shared memory
    int shm_fd = shm_open(SHM_NAME.c_str(), O_RDWR, 0666);
    if (shm_fd == -1) {
      LOG(ERROR) << "send: shm_open: " << strerror(errno);
      return;
    }

    SharedBuffer *shared_buffer = static_cast<SharedBuffer *>(
        mmap(NULL, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED,
             shm_fd, 0));
    if (shared_buffer == MAP_FAILED) {
      LOG(ERROR) << "send: mmap: " << strerror(errno);
      close(shm_fd);
      return;
    }

    barrier.Wait();
    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    barrier.Wait();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Sender: Time taken: " << elapsed_time.count() * 1000
              << " ms.";
    }

    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
  }

  double bandwidth = CalculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Send bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec.";
}

} // namespace

int RunShmBenchmark(int num_iterations, int num_warmups, uint64_t data_size,
                    uint64_t buffer_size) {
  SenseReversingBarrier::ClearResource(BARRIER_ID);
  CleanupResources();

  pid_t pid = fork();

  if (pid == -1) {
    LOG(ERROR) << "Fork failed: " << strerror(errno);
    return 1;
  }

  if (pid == 0) {
    SendProcess(num_warmups, num_iterations, data_size, buffer_size);
    exit(0);
  } else {
    ReceiveProcess(num_warmups, num_iterations, data_size);
    waitpid(pid, nullptr, 0);
  }

  return 0;
}
