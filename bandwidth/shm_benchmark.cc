#include "shm_benchmark.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "absl/log/log.h"

#include "common.h"

namespace {
const std::string SHM_NAME = "/shm_bandwidth_test";
const std::string SEM_SENDER_NAME = "/sem_sender_bandwidth";
const std::string SEM_RECEIVER_NAME = "/sem_receiver_bandwidth";
constexpr size_t BUFFER_SIZE = (1 << 20);

struct SharedBuffer {
  size_t data_size;
  bool transfer_complete;
  char data[BUFFER_SIZE];
};

void cleanup_resources() {
  shm_unlink(SHM_NAME.c_str());
  sem_unlink(SEM_SENDER_NAME.c_str());
  sem_unlink(SEM_RECEIVER_NAME.c_str());
}

size_t send_data(SharedBuffer *shared_buffer, sem_t *sem_sender,
                 sem_t *sem_receiver, const uint8_t *data, uint64_t data_size) {
  size_t total_sent = 0;

  while (total_sent < data_size) {
    // Wait for permission to write
    sem_wait(sem_sender);

    size_t bytes_to_send = std::min(BUFFER_SIZE, data_size - total_sent);
    shared_buffer->data_size = bytes_to_send;
    memcpy(shared_buffer->data, data + total_sent, bytes_to_send);
    total_sent += bytes_to_send;

    if (total_sent >= data_size) {
      shared_buffer->transfer_complete = true;
    }

    // Signal receiver that data is ready
    sem_post(sem_receiver);
  }

  return total_sent;
}

size_t receive_data(SharedBuffer *shared_buffer, sem_t *sem_sender,
                    sem_t *sem_receiver, std::vector<uint8_t> *received_data,
                    uint64_t data_size) {
  size_t total_received = 0;

  while (total_received < data_size) {
    // Wait for sender to signal data is ready
    sem_wait(sem_receiver);

    if (shared_buffer->transfer_complete) {
      break;
    }

    if (received_data) {
      received_data->insert(received_data->end(),
                            reinterpret_cast<uint8_t *>(shared_buffer->data),
                            reinterpret_cast<uint8_t *>(shared_buffer->data) +
                                shared_buffer->data_size);
    }
    total_received += shared_buffer->data_size;

    // Signal sender that data has been read
    sem_post(sem_sender);
  }

  return total_received;
}

void receive_process(int num_warmups, int num_iterations, uint64_t data_size) {
  // Clean up any existing resources
  cleanup_resources();

  // Create shared memory
  int shm_fd = shm_open(SHM_NAME.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    LOG(ERROR) << "receive: shm_open: " << strerror(errno);
    return;
  }

  if (ftruncate(shm_fd, sizeof(SharedBuffer)) == -1) {
    LOG(ERROR) << "receive: ftruncate: " << strerror(errno);
    close(shm_fd);
    cleanup_resources();
    return;
  }

  SharedBuffer *shared_buffer = static_cast<SharedBuffer *>(
      mmap(NULL, sizeof(SharedBuffer), PROT_READ | PROT_WRITE, MAP_SHARED,
           shm_fd, 0));
  if (shared_buffer == MAP_FAILED) {
    LOG(ERROR) << "receive: mmap: " << strerror(errno);
    close(shm_fd);
    cleanup_resources();
    return;
  }

  // Create semaphores for synchronization
  sem_t *sem_sender = sem_open(SEM_SENDER_NAME.c_str(), O_CREAT, 0666, 1);
  sem_t *sem_receiver = sem_open(SEM_RECEIVER_NAME.c_str(), O_CREAT, 0666, 0);
  if (sem_sender == SEM_FAILED || sem_receiver == SEM_FAILED) {
    LOG(ERROR) << "receive: sem_open: " << strerror(errno);
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    cleanup_resources();
    return;
  }

  VLOG(1) << "Receiver: Shared memory and semaphores initialized";

  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;

    if (is_warmup) {
      VLOG(1) << "Receiver: Warm-up " << iteration << "/" << num_warmups;
    } else {
      VLOG(1) << ReceivePrefix(iteration) << "Starting iteration...";
    }

    shared_buffer->transfer_complete = false;
    std::vector<uint8_t> received_data;
    received_data.reserve(data_size);

    sem_post(sem_sender);
    auto start_time = std::chrono::high_resolution_clock::now();

    // Receive data until data_size is reached
    receive_data(shared_buffer, sem_sender, sem_receiver, &received_data,
                 data_size);

    auto end_time = std::chrono::high_resolution_clock::now();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());

      VLOG(1) << "Receiver: Time taken: " << elapsed_time.count()
              << " seconds.";
    }

    // Verify received data (always, even during warmup)
    if (!verifyDataReceived(received_data, data_size)) {
      LOG(ERROR) << ReceivePrefix(iteration) << "Data verification failed!";
    } else {
      VLOG(1) << ReceivePrefix(iteration) << "Data verification passed.";
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Receiver";

  // Cleanup
  sem_close(sem_sender);
  sem_close(sem_receiver);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  cleanup_resources();
  VLOG(1) << "Receiver: Exiting.";
}

void send_process(int num_warmups, int num_iterations, uint64_t data_size,
                  uint64_t buffer_size) {
  // Give receiver time to initialize
  usleep(100000); // 100ms

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
  sem_t *sem_receiver = sem_open(SEM_RECEIVER_NAME.c_str(), 0);
  if (sem_sender == SEM_FAILED || sem_receiver == SEM_FAILED) {
    LOG(ERROR) << "send: sem_open: " << strerror(errno);
    munmap(shared_buffer, sizeof(SharedBuffer));
    close(shm_fd);
    return;
  }

  // Generate data to send once
  std::vector<uint8_t> data_to_send = generateDataToSend(data_size);
  std::vector<double> durations;

  for (int iteration = 0; iteration < num_warmups + num_iterations;
       ++iteration) {
    bool is_warmup = iteration < num_warmups;

    if (is_warmup) {
      VLOG(1) << "Sender: Warm-up " << iteration << "/" << num_warmups;
    } else {
      VLOG(1) << SendPrefix(iteration) << "Starting iteration...";
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Send data until data_size is reached (always use generated data)
    send_data(shared_buffer, sem_sender, sem_receiver, data_to_send.data(),
              data_size);

    auto end_time = std::chrono::high_resolution_clock::now();

    if (!is_warmup) {
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      durations.push_back(elapsed_time.count());
      VLOG(1) << "Sender: Time taken: " << elapsed_time.count() << " seconds.";
    }

    // Small delay between iterations
    if (iteration < num_warmups + num_iterations - 1) {
      usleep(100000); // 100ms delay
    }
  }

  double bandwidth = calculateBandwidth(durations, num_iterations, data_size);

  double bandwidth_gibps = bandwidth / (1024.0 * 1024.0 * 1024.0);
  LOG(INFO) << "Bandwidth: " << bandwidth_gibps << " GiByte/sec. Sender";

  // Cleanup
  sem_close(sem_sender);
  sem_close(sem_receiver);
  munmap(shared_buffer, sizeof(SharedBuffer));
  close(shm_fd);
  VLOG(1) << "Sender: Exiting.";
}

} // namespace

int run_shm_benchmark(int num_iterations, int num_warmups, uint64_t data_size,
                      uint64_t buffer_size) {
  pid_t pid = fork();

  if (pid == -1) {
    LOG(ERROR) << "fork: " << strerror(errno);
    return 1;
  }

  if (pid == 0) {
    send_process(num_warmups, num_iterations, data_size, buffer_size);
  } else {
    receive_process(num_warmups, num_iterations, data_size);

    // Wait for child process to complete
    int status;
    wait(&status);
  }

  return 0;
}
