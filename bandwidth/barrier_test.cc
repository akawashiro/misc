#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include <optional>

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
      : n_(n), sem_id_(id + "_sem"), shm_id_(id + "_shm") {
    init_sem_ = sem_open(sem_id_.c_str(), O_CREAT, 0644, 1);
    CHECK(init_sem_ != SEM_FAILED) << "Failed to create semaphore with id '"
                                   << sem_id_ << "': " << strerror(errno);

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
        VLOG(1) << "Shared memory with id '" << shm_id_
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
  }
  ~SenseReversingBarrier() {
    if (init_sem_) {
      sem_close(init_sem_);
      sem_unlink(sem_id_.c_str());
    }
  }

private:
  struct ShmData {
    std::atomic<uint64_t> count_;
    std::atomic<bool> sense_{false};
    std::atomic<uint64_t> n_users_{0};
  };

  sem_t *init_sem_;
  int shm_fd_;
  ShmData *shm_data_;

  const uint64_t n_;
  const std::string sem_id_;
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
