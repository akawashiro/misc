#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include <optional>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

class SenseReversingBarrier {
public:
  SenseReversingBarrier(int n, const std::string &id)
      : n_(n), init_sem_id_(id + "_init_sem"), shm_sem_id_(id + "_shm_sem"),
        shm_id_(id + "_shm") {
    init_sem_ = sem_open(init_sem_id_.c_str(), O_CREAT, 0644, 1);
    CHECK(init_sem_ != SEM_FAILED) << "Failed to create semaphore with id '"
                                   << init_sem_id_ << "': " << strerror(errno);
    shm_sem_ = sem_open(shm_sem_id_.c_str(), O_CREAT, 0644, 1);
    CHECK(shm_sem_ != SEM_FAILED) << "Failed to create semaphore with id '"
                                  << shm_sem_id_ << "': " << strerror(errno);

    // Critical section to ensure all processes hold initialized shared memory.
    sem_wait(init_sem_);
    shm_fd_ = shm_open(shm_id_.c_str(), O_CREAT | O_RDWR | O_EXCL, 0644);
    if (shm_fd_ >= 0) {
      VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
              << " - Created shared memory with id '" << shm_id_ << "'";
      CHECK(ftruncate(shm_fd_, sizeof(ShmData)) == 0)
          << "Failed to set size of shared memory with id '" << shm_id_
          << "': " << strerror(errno);
      shm_data_ = static_cast<ShmData *>(mmap(nullptr, sizeof(ShmData),
                                              PROT_READ | PROT_WRITE,
                                              MAP_SHARED, shm_fd_, 0));
      CHECK(shm_data_ != MAP_FAILED) << "Failed to map shared memory with id '"
                                     << shm_id_ << "': " << strerror(errno);
      shm_data_->count_.store(0);
    } else if (shm_fd_ < 0) {
      if (errno == EEXIST) {
        VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
                << " - Shared memory with id '" << shm_id_
                << "' already exists, opening it instead.";
        shm_fd_ = shm_open(shm_id_.c_str(), O_RDWR, 0644);
        CHECK(shm_fd_ >= 0) << "Failed to open existing shared memory with id '"
                            << shm_id_ << "': " << strerror(errno);
        shm_data_ = static_cast<ShmData *>(mmap(nullptr, sizeof(ShmData),
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED, shm_fd_, 0));
        CHECK(shm_data_ != MAP_FAILED)
            << "Failed to map existing shared memory with id '" << shm_id_
            << "': " << strerror(errno);
      } else {
        CHECK(false) << "Failed to create shared memory with id '" << shm_id_
                     << "': " << strerror(errno);
      }
    }
    sem_post(init_sem_);
    // Critical section ends here.

    shm_data_->n_users_.fetch_add(1);

    VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
            << " - SenseReversingBarrier initialized with id '" << shm_id_
            << "' for " << n_ << " users. Waiting for all users to join.";
    while (shm_data_->n_users_.load() < n_) {
      std::this_thread::yield();
    }
    VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
            << " - All users have joined the barrier with id '" << shm_id_
            << "'. Proceeding.";
  }

  ~SenseReversingBarrier() {
    uint64_t remaining_users = shm_data_->n_users_.fetch_sub(1);
    if (remaining_users == 1) {
      VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
              << " - Last user of shared memory with id '" << shm_id_
              << "' is exiting. Unlinking shared memory.";
      if (init_sem_) {
        sem_close(init_sem_);
        sem_unlink(init_sem_id_.c_str());
      }
      if (shm_sem_) {
        sem_close(shm_sem_);
        sem_unlink(shm_sem_id_.c_str());
      }
      if (shm_fd_ >= 0) {
        munmap(shm_data_, sizeof(ShmData));
        close(shm_fd_);
        shm_unlink(shm_id_.c_str());
      }
    } else {
      VLOG(1) << "PID: " << getpid() << " TID: " << pthread_self()
              << " - Not the last user of shared memory with id '" << shm_id_
              << " " << remaining_users << " users remaining.";
      if (init_sem_) {
        sem_close(init_sem_);
      }
      if (shm_sem_) {
        sem_close(shm_sem_);
      }
      if (shm_fd_ >= 0) {
        munmap(shm_data_, sizeof(ShmData));
        close(shm_fd_);
      }
    }
  }

private:
  struct ShmData {
    std::atomic<uint64_t> count_;
    std::atomic<bool> shared_sense_{false};
    std::atomic<uint64_t> n_users_{0};
  };

  sem_t *init_sem_;
  sem_t *shm_sem_;
  int shm_fd_;
  ShmData *shm_data_;
  bool sense_ = true;

  const uint64_t n_;
  const std::string init_sem_id_;
  const std::string shm_sem_id_;
  const std::string shm_id_;
};

namespace {
void TestConstructor() {
  int pid = fork();
  CHECK(pid >= 0) << "Fork failed: " << strerror(errno);

  if (pid == 0) {
    SenseReversingBarrier barrier(2, "/TestBarrier");
    return;
  } else {
    SenseReversingBarrier barrier(2, "/TestBarrier");
    waitpid(pid, nullptr, 0);
    return;
  }
}
} // namespace

ABSL_FLAG(std::optional<int>, vlog, std::nullopt,
          "Show VLOG messages lower than this level.");

int main(int argc, char *argv[]) {
  absl::SetProgramUsageMessage("Sense Reversing Barrier Test");
  absl::ParseCommandLine(argc, argv);
  std::optional<int> vlog = absl::GetFlag(FLAGS_vlog);
  if (vlog.has_value()) {
    int v = *vlog;
    absl::SetGlobalVLogLevel(v);
  }
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();

  TestConstructor();

  return 0;
}
