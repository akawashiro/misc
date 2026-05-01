#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kNumElements = 8;

__global__ void add_kernel(const int* a, const int* b, int* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

void check_cuda(cudaError_t status, const char* msg) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

int main() {
  std::array<int, kNumElements> host_a{};
  std::array<int, kNumElements> host_b{};
  std::array<int, kNumElements> host_c{};

  for (int i = 0; i < kNumElements; ++i) {
    host_a[i] = i;
    host_b[i] = i * 10;
  }

  int* dev_a = nullptr;
  int* dev_b = nullptr;
  int* dev_c = nullptr;

  check_cuda(cudaMalloc(&dev_a, sizeof(int) * kNumElements), "cudaMalloc dev_a");
  check_cuda(cudaMalloc(&dev_b, sizeof(int) * kNumElements), "cudaMalloc dev_b");
  check_cuda(cudaMalloc(&dev_c, sizeof(int) * kNumElements), "cudaMalloc dev_c");

  check_cuda(cudaMemcpy(dev_a, host_a.data(), sizeof(int) * kNumElements, cudaMemcpyHostToDevice),
             "cudaMemcpy host_a");
  check_cuda(cudaMemcpy(dev_b, host_b.data(), sizeof(int) * kNumElements, cudaMemcpyHostToDevice),
             "cudaMemcpy host_b");

  constexpr int threads_per_block = 256;
  int blocks = (kNumElements + threads_per_block - 1) / threads_per_block;
  add_kernel<<<blocks, threads_per_block>>>(dev_a, dev_b, dev_c, kNumElements);
  check_cuda(cudaGetLastError(), "kernel launch");

  check_cuda(cudaMemcpy(host_c.data(), dev_c, sizeof(int) * kNumElements, cudaMemcpyDeviceToHost),
             "cudaMemcpy host_c");

  for (int i = 0; i < kNumElements; ++i) {
    std::printf("%d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
  }

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return 0;
}
