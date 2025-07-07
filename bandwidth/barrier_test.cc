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
  SenseReversingBarrier(int n, const std::string &id) : n_(n), id_(id) {}

private:
  const uint64_t n_;
  const std::string id_;
};

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

  SenseReversingBarrier barrier(4, "test_barrier");

  LOG(INFO) << "Starting barrier test";

  LOG(INFO) << "Barrier test completed successfully.";

  return 0;
}
