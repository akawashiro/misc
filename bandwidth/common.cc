#include "common.h"

#include <algorithm>
#include <numeric>
#include <random>

#include "absl/log/check.h"
#include "absl/log/log.h"

std::vector<uint8_t> calcChecksum(const std::vector<uint8_t> &data) {
  std::vector<uint8_t> checksum(CHECKSUM_SIZE, 0);
  for (size_t i = 0; i < CONTEXT_SIZE; ++i) {
    checksum[i % CHECKSUM_SIZE] ^= data[i];
  }
  return checksum;
}

std::vector<uint8_t> generateDataToSend() {
  VLOG(1) << "Generating data to send...";
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);
  std::vector<uint8_t> data(DATA_SIZE, 0);

  size_t i = 0;
  for (i = 0; i + 8 < CONTEXT_SIZE; i += 8) {
    uint64_t *d = reinterpret_cast<uint64_t *>(&data[i]);
    *d = dist(engine);
  }
  for (; i < CONTEXT_SIZE; ++i) {
    data[i] = static_cast<uint8_t>(dist(engine) & 0xFF);
  }
  VLOG(1) << "Context data generated. Size: " << CONTEXT_SIZE
          << " bytes. Filling checksum...";
  const std::vector<uint8_t> checksum = calcChecksum(data);
  for (size_t j = 0; j < CHECKSUM_SIZE; ++j) {
    data[CONTEXT_SIZE + j] = checksum[j];
  }
  VLOG(1) << "Data generation complete. Data size: "
          << static_cast<double>(data.size()) / (1 << 30)
          << " GiByte, Checksum size: " << CHECKSUM_SIZE << " bytes.";

  return data;
}

bool verifyDataReceived(const std::vector<uint8_t> &data) {
  if (data.size() != DATA_SIZE) {
    LOG(ERROR) << "Data size mismatch: expected " << DATA_SIZE << ", got "
               << data.size();
    return false;
  }

  std::vector<uint8_t> checksum = calcChecksum(data);
  for (size_t i = 0; i < CHECKSUM_SIZE; ++i) {
    if (data[CONTEXT_SIZE + i] != checksum[i]) {
      LOG(ERROR) << "Checksum mismatch at index " << i << ": expected "
                 << static_cast<int>(checksum[i]) << ", got "
                 << static_cast<int>(data[CONTEXT_SIZE + i]);
      return false;
    }
  }
  return true;
}

double calculateBandwidth(std::vector<double> durations) {
  CHECK(durations.size() == NUM_ITERATIONS);
  std::sort(durations.begin(), durations.end());
  std::vector<double> filtered_durations(durations.begin() + 1,
                                         durations.end() - 1);

  double average_duration = std::accumulate(filtered_durations.begin(),
                                            filtered_durations.end(), 0.0) /
                            filtered_durations.size();
  double bandwidth = DATA_SIZE / average_duration;
  return bandwidth;
}

ProcessBarrier::ProcessBarrier(int total_processes, const std::string &shm_name,
                               const std::string &mutex_sem_name,
                               const std::string &barrier_sem_name)
    : shm_name_(shm_name), mutex_sem_name_(mutex_sem_name),
      barrier_sem_name_(barrier_sem_name), mutex_sem_(nullptr),
      barrier_sem_(nullptr), barrier_data_(nullptr), shm_fd_(-1),
      owner_(false) // Initially assume not the owner
{
  VLOG(1) << "Process " << getpid()
          << ": Initializing ProcessBarrier with total_processes: "
          << total_processes;
  // 1. Create or open shared memory segment
  // O_CREAT: Create if it doesn't exist.
  // O_RDWR: Read/write access.
  // S_IRUSR | S_IWUSR: Read/write permissions for the user.
  shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  if (shm_fd_ == -1) {
    throw std::runtime_error("shm_open failed: " +
                             std::string(strerror(errno)));
  }

  // Check if this process is the first to create the shared memory segment.
  // We try to open exclusively. If it succeeds, we are the first.
  // If it fails with EEXIST, it means another process already created it.
  int test_fd =
      shm_open(shm_name_.c_str(), O_CREAT | O_EXCL | O_RDWR, S_IRUSR | S_IWUSR);
  if (test_fd != -1) {
    // Exclusive creation successful, so this is the owner.
    close(test_fd); // Close the temporary FD
    owner_ = true;
  } else if (errno != EEXIST) {
    // Any other error means a problem.
    close(shm_fd_); // Close the shm_open FD too.
    throw std::runtime_error("shm_open (exclusive check) failed: " +
                             std::string(strerror(errno)));
  }

  // If this instance is the owner (first creator), set the size of shared
  // memory.
  if (owner_) {
    if (ftruncate(shm_fd_, sizeof(BarrierData)) == -1) {
      close(shm_fd_);
      throw std::runtime_error("ftruncate failed: " +
                               std::string(strerror(errno)));
    }
  }

  // Map shared memory into the process's address space.
  barrier_data_ = static_cast<BarrierData *>(mmap(NULL, sizeof(BarrierData),
                                                  PROT_READ | PROT_WRITE,
                                                  MAP_SHARED, shm_fd_, 0));
  if (barrier_data_ == MAP_FAILED) {
    close(shm_fd_);
    throw std::runtime_error("mmap failed: " + std::string(strerror(errno)));
  }

  // 2. Open or create mutex semaphore (for protecting the shared counter)
  // O_CREAT | O_EXCL: Create exclusively. If it exists, sem_open returns
  // SEM_FAILED with errno EEXIST. 1: Initial value (binary semaphore for mutual
  // exclusion).
  mutex_sem_ =
      sem_open(mutex_sem_name_.c_str(), O_CREAT | O_EXCL, S_IRUSR | S_IWUSR, 1);
  if (mutex_sem_ == SEM_FAILED) {
    if (errno == EEXIST) {
      // Already exists, just open it.
      mutex_sem_ = sem_open(mutex_sem_name_.c_str(), 0);
      if (mutex_sem_ == SEM_FAILED) {
        munmap(barrier_data_, sizeof(BarrierData));
        close(shm_fd_);
        throw std::runtime_error("sem_open mutex existing failed: " +
                                 std::string(strerror(errno)));
      }
    } else {
      // Other error
      munmap(barrier_data_, sizeof(BarrierData));
      close(shm_fd_);
      throw std::runtime_error("sem_open mutex failed: " +
                               std::string(strerror(errno)));
    }
  } else {
    // This process exclusively created the mutex semaphore, so it's the owner
    // for initial setup. It's also the responsibility of this process to
    // initialize barrier_data.
    barrier_data_->count = 0;
    barrier_data_->num_processes = total_processes;
    VLOG(1) << "Process " << getpid()
            << ": Initialized shared memory and mutex semaphore." << std::endl;
  }

  // 3. Open or create barrier semaphore (for waiting on all processes)
  // 0: Initial value (all processes wait until posted by the last arriving
  // process).
  barrier_sem_ = sem_open(barrier_sem_name_.c_str(), O_CREAT | O_EXCL,
                          S_IRUSR | S_IWUSR, 0);
  if (barrier_sem_ == SEM_FAILED) {
    if (errno == EEXIST) {
      // Already exists, just open it.
      barrier_sem_ = sem_open(barrier_sem_name_.c_str(), 0);
      if (barrier_sem_ == SEM_FAILED) {
        sem_close(mutex_sem_); // Cleanup previously opened semaphore
        munmap(barrier_data_, sizeof(BarrierData));
        close(shm_fd_);
        throw std::runtime_error("sem_open barrier existing failed: " +
                                 std::string(strerror(errno)));
      }
    } else {
      // Other error
      sem_close(mutex_sem_); // Cleanup previously opened semaphore
      munmap(barrier_data_, sizeof(BarrierData));
      close(shm_fd_);
      throw std::runtime_error("sem_open barrier failed: " +
                               std::string(strerror(errno)));
    }
  } else {
    // This process exclusively created the barrier semaphore.
    VLOG(1) << "Process " << getpid() << ": Initialized barrier semaphore."
            << std::endl;
  }

  VLOG(1) << "Process " << getpid() << ": Barrier initialized";
}

ProcessBarrier::~ProcessBarrier() {
  // Close semaphores
  if (mutex_sem_ != nullptr) {
    if (sem_close(mutex_sem_) == -1) {
      VLOG(1) << "Process " << getpid()
              << ": Error closing mutex semaphore: " << strerror(errno)
              << std::endl;
    }
  }
  if (barrier_sem_ != nullptr) {
    if (sem_close(barrier_sem_) == -1) {
      VLOG(1) << "Process " << getpid()
              << ": Error closing barrier semaphore: " << strerror(errno)
              << std::endl;
    }
  }
  // Unmap shared memory
  if (barrier_data_ != nullptr) {
    if (munmap(barrier_data_, sizeof(BarrierData)) == -1) {
      VLOG(1) << "Process " << getpid()
              << ": Error unmapping shared memory: " << strerror(errno)
              << std::endl;
    }
  }
  // Close shared memory file descriptor
  if (shm_fd_ != -1) {
    close(shm_fd_);
  }
  // Note: unlinking of resources should be done explicitly by one process
  // (e.g., in unlink_all()), not automatically in the destructor, as other
  // processes might still be using them.
}

void ProcessBarrier::wait() {
  // 1. Acquire mutex to safely update the shared counter.
  if (sem_wait(mutex_sem_) == -1) {
    throw std::runtime_error("sem_wait (mutex) failed: " +
                             std::string(strerror(errno)));
  }

  barrier_data_->count++;
  VLOG(1) << "Process " << getpid()
          << ": Reached barrier. Count: " << barrier_data_->count << "/"
          << barrier_data_->num_processes << std::endl;

  if (barrier_data_->count == barrier_data_->num_processes) {
    // All processes have arrived. Signal all waiting processes.
    VLOG(1) << "Process " << getpid()
            << ": All processes reached barrier. Releasing others."
            << std::endl;
    for (int i = 0; i < barrier_data_->num_processes; ++i) {
      if (sem_post(barrier_sem_) == -1) {
        // Critical error: try to release mutex before throwing to prevent
        // deadlock.
        sem_post(mutex_sem_);
        throw std::runtime_error("sem_post (barrier) failed: " +
                                 std::string(strerror(errno)));
      }
    }
    barrier_data_->count =
        0; // Reset counter for the next barrier synchronization.
  }

  // 2. Release mutex after updating the counter.
  if (sem_post(mutex_sem_) == -1) {
    throw std::runtime_error("sem_post (mutex) failed: " +
                             std::string(strerror(errno)));
  }

  // 3. Wait on the barrier semaphore until all processes have arrived and been
  // released.
  if (sem_wait(barrier_sem_) == -1) {
    throw std::runtime_error("sem_wait (barrier) failed: " +
                             std::string(strerror(errno)));
  }
  VLOG(1) << "Process " << getpid() << ": Passed barrier." << std::endl;
}

void ProcessBarrier::unlink_all() {
  // Unlink semaphores and shared memory from the system.
  // These calls remove the resources from the system. They should be called by
  // only one process (typically the last one to exit or a designated manager).
  // ENOENT means the resource doesn't exist, which is fine if it was already
  // unlinked.
  if (sem_unlink(mutex_sem_name_.c_str()) == -1 && errno != ENOENT) {
    VLOG(1) << "Process " << getpid()
            << ": Error unlinking mutex semaphore: " << strerror(errno)
            << std::endl;
  }
  if (sem_unlink(barrier_sem_name_.c_str()) == -1 && errno != ENOENT) {
    VLOG(1) << "Process " << getpid()
            << ": Error unlinking barrier semaphore: " << strerror(errno)
            << std::endl;
  }
  if (shm_unlink(shm_name_.c_str()) == -1 && errno != ENOENT) {
    VLOG(1) << "Process " << getpid()
            << ": Error unlinking shared memory: " << strerror(errno)
            << std::endl;
  }
  VLOG(1) << "Process " << getpid() << ": Unlinked all shared resources."
          << std::endl;
}
