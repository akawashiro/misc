#include <fcntl.h>
#include <semaphore.h>
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
  SenseReversingBarrier(int n, const std::string &id) : n_(n), id_(id) {
    init_sem_ = sem_open((id + "_sem").c_str(), O_CREAT, 0644, 1);
    CHECK(init_sem_ != SEM_FAILED) << "Failed to create semaphore with id '"
                                   << id_ << "': " << strerror(errno);
    sem_wait(init_sem_);
    VLOG(1) << "SenseReversingBarrier initialized with id: " << id_
            << ", n: " << n_;
    sem_post(init_sem_);
  }
  ~SenseReversingBarrier() {
    if (init_sem_) {
      sem_close(init_sem_);
      sem_unlink(id_.c_str());
    }
  }

private:
  sem_t *init_sem_;
  const uint64_t n_;
  const std::string id_;
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
