#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string>
#include <vector>

constexpr uint64_t DATA_SIZE = 128 * (1 << 20);
constexpr uint64_t CHECKSUM_SIZE = 128;
constexpr uint64_t CONTEXT_SIZE = DATA_SIZE - CHECKSUM_SIZE;
static_assert(DATA_SIZE > CHECKSUM_SIZE,
              "DATA_SIZE must be larger than CHECKSUM_SIZE");

constexpr int NUM_WARMUPS = 1;
constexpr int NUM_ITERATIONS = 10;

std::vector<uint8_t> generateDataToSend();
bool verifyDataReceived(const std::vector<uint8_t> &data);
double calculateBandwidth(std::vector<double> durations);

class ProcessBarrier {
public:
  // Constructor: Initializes or opens the barrier resources.
  // total_processes: The total number of processes that will synchronize at
  // this barrier. shm_name: Name for the shared memory segment. mutex_sem_name:
  // Name for the mutex semaphore protecting the shared counter.
  // barrier_sem_name: Name for the barrier semaphore used for waiting.
  ProcessBarrier(
      int total_processes, const std::string &shm_name = "/process_barrier_shm",
      const std::string &mutex_sem_name = "/process_barrier_mutex_sem",
      const std::string &barrier_sem_name = "/process_barrier_barrier_sem");

  // Destructor: Cleans up resources. Follows RAII principles.
  ~ProcessBarrier();

  // Waits at the barrier until all participating processes have arrived.
  void wait();

  // Unlinks (deletes) all shared resources from the system.
  // This should typically be called by the last process to finish or a
  // dedicated manager process. Calling this while other processes are still
  // using the barrier can lead to issues.
  void unlink_all();

private:
  // Nested struct to hold shared data within the shared memory segment.
  // This makes BarrierData a part of ProcessBarrier's internal implementation.
  struct BarrierData {
    int count;         // Counter for processes that have reached the barrier.
    int num_processes; // Total number of processes expected at the barrier.
  };

  std::string shm_name_;
  std::string mutex_sem_name_;
  std::string barrier_sem_name_;

  sem_t *mutex_sem_;          // Pointer to the mutex semaphore.
  sem_t *barrier_sem_;        // Pointer to the barrier semaphore.
  BarrierData *barrier_data_; // Pointer to the shared barrier data.
  int shm_fd_;                // File descriptor for the shared memory.
  bool owner_; // True if this instance created the shared resources.

  // Disable copy constructor and assignment operator to prevent unintended
  // copying.
  ProcessBarrier(const ProcessBarrier &) = delete;
  ProcessBarrier &operator=(const ProcessBarrier &) = delete;
};
