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
