#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <immintrin.h>
#include <numeric>
#include <string>
#include <vector>

void print_header() {
  printf("Type            \tSize (MiB)  \tTime (ms) \tBandwidth (GiB/s)   \t"
         "Cycles/Byte \n");
}

void print_results(const char *type, size_t size, double average_time,
                   double bandwidth) {
  double cycles_per_byte = 5000.0 * 1000 * 1000 * average_time / size;
  printf("%-16s\t%-12zu\t%-10.2f\t%-20.2f\t%-12.2f\n", type,
         size / (1024 * 1024), average_time * 1000,
         bandwidth / (1024 * 1024 * 1024), cycles_per_byte);
}

void shuffle_src(void *src, size_t size_byte) {
  uint32_t *src_bytes = (uint32_t *)src;
  int N = size_byte / sizeof(uint32_t);
  for (int i = N - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    uint32_t temp = src_bytes[i];
    src_bytes[i] = src_bytes[j];
    src_bytes[j] = temp;
  }
}

void set_src(void *src, size_t size_byte) {
  for (uint32_t i = 0; i < size_byte / sizeof(uint32_t); i++) {
    ((uint32_t *)(src))[i] = (uint32_t)i;
  }
  shuffle_src(src, size_byte);
}

void check_dest(void *dest, size_t size_byte) {
  std::vector<uint32_t> count(size_byte / sizeof(uint32_t), 0);
  uint32_t *dest_uint32 = (uint32_t *)dest;
  for (size_t i = 0; i < size_byte / sizeof(uint32_t); i++) {
    if (count[dest_uint32[i]] == 0) {
      count[dest_uint32[i]] = 1;
    } else {
      printf("Error: Duplicate value %u found in destination.\n",
             dest_uint32[i]);
    }
  }
  return;
}

void shuffle_indices(uint32_t *indices, size_t size_byte) {
  int N = size_byte / 4;
  for (int i = N - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    uint32_t temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
}

using memcpy_with_shuffle_t = void (*)(void *dest, const void *src,
                                       size_t size_byte,
                                       const uint32_t *indices);

enum class IndexSortType { SORTED, SHUFFLED };

void benchmark_memcpy(memcpy_with_shuffle_t memcpy_func, size_t size_byte,
                      int warmup, int iterations, std::string name,
                      IndexSortType index_type) {
  void *src = aligned_alloc(32, size_byte);
  void *dest = aligned_alloc(32, size_byte);
  uint32_t *indices = reinterpret_cast<uint32_t *>(
      aligned_alloc(32, size_byte / 4 * sizeof(uint32_t)));
  for (size_t i = 0; i < size_byte / 4; i++) {
    indices[i] = i;
  }
  if (index_type == IndexSortType::SHUFFLED) {
    shuffle_indices(indices, size_byte);
  }

  set_src(src, size_byte);
  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; i++) {
    clock_t start = clock();
    memcpy_func(dest, src, size_byte, indices);
    clock_t end = clock();
    if (i == 0) {
      check_dest(dest, size_byte);
    } else if (i >= warmup) {
      times.push_back((double)(end - start) / CLOCKS_PER_SEC);
    }
  }

  double average_time =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  double bandwidth = (double)(size_byte) / average_time;
  print_results(name.c_str(), size_byte, average_time, bandwidth);
  free(src);
  free(dest);
  free(indices);
}

void memcpy_copy(size_t size_byte, int warmup, int iterations) {
  void *src = aligned_alloc(32, size_byte);
  void *dest = aligned_alloc(32, size_byte);

  set_src(src, size_byte);
  std::vector<double> times;
  for (int i = 0; i < warmup + iterations; i++) {
    clock_t start = clock();
    memcpy(dest, src, size_byte);
    clock_t end = clock();
    if (i == 0) {
      check_dest(dest, size_byte);
    } else if (i >= warmup) {
      times.push_back((double)(end - start) / CLOCKS_PER_SEC);
    }
  }

  double average_time =
      std::accumulate(times.begin(), times.end(), 0.0) / times.size();
  double bandwidth = (double)(size_byte) / average_time;
  print_results("memcpy", size_byte, average_time, bandwidth);
  free(src);
  free(dest);
}

void avx2_copy(void *dest_void, const void *src_void, size_t size_byte,
               const uint32_t *indices) {
  uint8_t *src = reinterpret_cast<uint8_t *>(const_cast<void *>(src_void));
  uint8_t *dest = reinterpret_cast<uint8_t *>(dest_void);
  for (size_t j = 0; j < size_byte / 32; j++) {
    __m256i *src_vec = (__m256i *)(src + j * 32);
    __m256i *dest_vec = (__m256i *)(dest + j * 32);
    _mm256_storeu_si256(dest_vec, _mm256_loadu_si256(src_vec));
  }
}

void avx2_gather_copy(void *dest_void, const void *src_void, size_t size_byte,
                      const uint32_t *indices) {
  uint8_t *src = reinterpret_cast<uint8_t *>(const_cast<void *>(src_void));
  uint8_t *dest = reinterpret_cast<uint8_t *>(dest_void);
  for (size_t j = 0; j < size_byte / 32; j++) {
    __m256i vindices = _mm256_loadu_si256((__m256i *)(indices + j * 8));
    __m256i *dest_vec = (__m256i *)(dest + j * 32);
    __m256i gathered = _mm256_i32gather_epi32(src, vindices, 4);
    _mm256_storeu_si256(dest_vec, gathered);
  }
}

void avx512_copy(void *dest_void, const void *src_void, size_t size_byte,
                 const uint32_t *indices) {
  uint8_t *src = reinterpret_cast<uint8_t *>(const_cast<void *>(src_void));
  uint8_t *dest = reinterpret_cast<uint8_t *>(dest_void);
  for (size_t j = 0; j < size_byte / 64; j++) {
    __m512i *src_vec = (__m512i *)(src + j * 64);
    __m512i *dest_vec = (__m512i *)(dest + j * 64);
    _mm512_storeu_si512(dest_vec, _mm512_loadu_si512(src_vec));
  }
}

void avx512_gather_copy(void *dest_void, const void *src_void, size_t size_byte,
                        const uint32_t *indices) {
  uint8_t *src = reinterpret_cast<uint8_t *>(const_cast<void *>(src_void));
  uint8_t *dest = reinterpret_cast<uint8_t *>(dest_void);
  for (size_t j = 0; j < size_byte / 64; j++) {
    __m512i vindices = _mm512_loadu_si512((__m512i *)(indices + j * 16));
    __m512i *dest_vec = (__m512i *)(dest + j * 64);
    __m512i gathered = _mm512_i32gather_epi32(vindices, src, 4);
    _mm512_storeu_si512(dest_vec, gathered);
  }
}

int main() {
  const int WARMUP = 10;
  const int ITERATIONS = 10;
  const int N_SIZE = 11;
  const int N_TEST_SIZE = 8;
  size_t sizes[N_SIZE] = {1UL << 20,  // 1 MiB
                          1UL << 21,  // 2 MiB
                          1UL << 22,  // 4 MiB
                          1UL << 23,  // 8 MiB
                          1UL << 24,  // 16 MiB
                          1UL << 25,  // 32 MiB
                          1UL << 26,  // 64 MiB
                          1UL << 27,  // 128 MiB
                          1UL << 28,  // 256 MiB
                          1UL << 29,  // 512 MiB
                          1UL << 30}; // 1024 MiB
  print_header();
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(
        [](void *dest, const void *src, size_t size_byte,
           const uint32_t *indices) { memcpy(dest, src, size_byte); },
        sizes[i], WARMUP, ITERATIONS, "memcpy", IndexSortType::SORTED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx2_copy, sizes[i], WARMUP, ITERATIONS, "avx2",
                     IndexSortType::SORTED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx2_gather_copy, sizes[i], WARMUP, ITERATIONS,
                     "avx2_gather_shuffled", IndexSortType::SHUFFLED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx2_gather_copy, sizes[i], WARMUP, ITERATIONS,
                     "avx2_gather_sorted", IndexSortType::SORTED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx512_copy, sizes[i], WARMUP, ITERATIONS, "avx512",
                     IndexSortType::SORTED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx512_gather_copy, sizes[i], WARMUP, ITERATIONS,
                     "avx512_gather_shuffled", IndexSortType::SHUFFLED);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    benchmark_memcpy(avx512_gather_copy, sizes[i], WARMUP, ITERATIONS,
                     "avx512_gather_sorted", IndexSortType::SORTED);
  }
}
