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
const std::string SEM_SENDER_NAME = "/sem_sender_bandwidth";
const std::string SEM_RECEIVER_NAME = "/sem_receiver_bandwidth";
const std::string BARRIER_ID = "/shm_benchmark";
constexpr size_t BUFFER_SIZE = (1 << 20);

struct SharedBuffer {
  size_t data_size;
  bool transfer_complete;
  char data[BUFFER_SIZE];
};

void CleanupResources() {
  shm_unlink(SHM_NAME.c_str());
  sem_unlink(SEM_SENDER_NAME.c_str());
  sem_unlink(SEM_RECEIVER_NAME.c_str());
}

size_t send_data(SharedBuffer *shared_buffer, sem_t *sem_sender,
                 sem_t *sem_receiver, const uint8_t *data, uint64_t data_size) {
  size_t total_sent = 0;

  while (total_sent < data_size) {
    sem_wait(sem_sender);

    size_t bytes_to_send = std::min(BUFFER_SIZE, data_size - total_sent);
    shared_buffer->data_size = bytes_to_send;
    memcpy(shared_buffer->data, data + total_sent, bytes_to_send);
    total_sent += bytes_to_send;

    if (total_sent >= data_size) {
      shared_buffer->transfer_complete = true;
    }

    sem_post(sem_receiver);
  }

  return total_sent;
}

size_t receive_data(SharedBuffer *shared_buffer, sem_t *sem_sender,
                    sem_t *sem_receiver, std::vector<uint8_t> *received_data,
                    uint64_t data_size) {
  size_t total_received = 0;

  while (total_received < data_size) {
    sem_wait(sem_receiver);

    memcpy(received_data->data() + total_received, shared_buffer->data,
           shared_buffer->data_size);
    total_received += shared_buffer->data_size;

    if (shared_buffer->transfer_complete) {
      break;
    }

    sem_post(sem_sender);
  }

  return total_received;
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

    // Create semaphores for synchronization
    sem_t *sem_sender = sem_open(SEM_SENDER_NAME.c_str(), O_CREAT, 0666, 1);
    sem_t *sem_receiver = sem_open(SEM_RECEIVER_NAME.c_str(), O_CREAT, 0666, 0);
    if (sem_sender == SEM_FAILED || sem_receiver == SEM_FAILED) {
      LOG(ERROR) << "receive: sem_open: " << strerror(errno);
      munmap(shared_buffer, sizeof(SharedBuffer));
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
    receive_data(shared_buffer, sem_sender, sem_receiver, &received_data,
                 data_size);
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

    // Cleanup
    sem_close(sem_sender);
    sem_close(sem_receiver);
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    CleanupResources();
  }

  double bandwidth = CalculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30)
            << " GiByte/sec. Receiver";
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

    // Open existing semaphores
    sem_t *sem_sender = sem_open(SEM_SENDER_NAME.c_str(), 0);
    if (sem_sender == SEM_FAILED) {
      LOG(ERROR) << "send: sem_open: " << strerror(errno);
      munmap(shared_buffer, sizeof(SharedBuffer));
      close(shm_fd);
      return;
    }
    sem_t *sem_receiver = sem_open(SEM_RECEIVER_NAME.c_str(), 0);
    if (sem_receiver == SEM_FAILED) {
      LOG(ERROR) << "send: sem_open: " << strerror(errno);
      munmap(shared_buffer, sizeof(SharedBuffer));
      close(shm_fd);
      return;
    }

    barrier.Wait();
    auto start_time = std::chrono::high_resolution_clock::now();
    send_data(shared_buffer, sem_sender, sem_receiver, data_to_send.data(),
              data_size);
    auto end_time = std::chrono::high_resolution_clock::now();
    barrier.Wait();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Sender: Time taken: " << elapsed_time.count() * 1000
              << " ms.";
    }

    sem_close(sem_sender);
    sem_close(sem_receiver);
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
  }

  double bandwidth = CalculateBandwidth(durations, num_iterations, data_size);
  LOG(INFO) << "Bandwidth: " << bandwidth / (1 << 30) << " GiByte/sec. Sender";
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
