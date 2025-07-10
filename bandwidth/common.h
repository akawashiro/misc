#include <cstdint>
#include <string>
#include <vector>

constexpr uint64_t CHECKSUM_SIZE = 128;

std::vector<uint8_t> generateDataToSend(uint64_t data_size);
bool verifyDataReceived(const std::vector<uint8_t> &data, uint64_t data_size);
double calculateBandwidth(std::vector<double> durations, int num_iterations,
                          uint64_t data_size);

// Logging prefix functions
std::string ReceivePrefix(int iteration);
std::string SendPrefix(int iteration);
