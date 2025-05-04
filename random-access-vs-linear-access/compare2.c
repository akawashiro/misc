#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_header() {
  printf("Function\tSize (MiB)\tAverage Time (s)\tBandwidth (GiB/s)\n");
}

void print_results(char *function_name, size_t size, double average_time,
                   double bandwidth) {
  printf("%s\t%zu\t%.6f\t%.6f\n", function_name, size / (1024 * 1024),
         average_time, bandwidth / (1024 * 1024 * 1024));
}

void set_src(void *src, size_t size_byte) {
  for (size_t i = 0; i < size_byte; i++) {
    ((uint8_t *)(src))[i] = (uint8_t)(i % 256);
  }
}

void check_dest(void *dest, size_t size_byte) {
  int *count = malloc(sizeof(int) * 256);
  for (int i = 0; i < size_byte; i++) {
    count[((uint8_t *)(dest))[i]]++;
  }
  for (int i = 0; i < 256; i++) {
    if (count[i] != size_byte / 256) {
      printf("Error at value %d: expected %zu, got %d\n", i,
             size_byte / 256, count[i]);
    }
  }
}

void memcpy_copy(size_t size_byte, int warmup, int iterations) {
  void *src = aligned_alloc(32, size_byte);
  void *dest = aligned_alloc(32, size_byte);

  set_src(src, size_byte);
  for (int i = 0; i < warmup; i++) {
    memcpy(dest, src, size_byte);
  }
  check_dest(dest, size_byte);

  clock_t start = clock();
  for (int i = 0; i < iterations; i++) {
    memcpy(dest, src, size_byte);
  }
  clock_t end = clock();
  double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
  double average_time = time_taken / iterations;
  double bandwidth =
      (double)(size_byte * sizeof(int32_t) * iterations) / time_taken;
  print_results("memcpy", size_byte, average_time, bandwidth);
}

void avx_copy(size_t size_byte, int warmup, int iterations) {
  void *src = aligned_alloc(32, size_byte);
  void *dest = aligned_alloc(32, size_byte);

  set_src(src, size_byte);
  for (int i = 0; i < warmup; i++) {
    for (size_t j = 0; j < size_byte / 32; j++) {
      __m256i *src_vec = (__m256i *)(src + j * 32);
      __m256i *dest_vec = (__m256i *)(dest + j * 32);
      _mm256_storeu_si256(dest_vec, _mm256_loadu_si256(src_vec));
    }
    if(i == 0) {
      check_dest(dest, size_byte);
    }
  }

  clock_t start = clock();
  for (int i = 0; i < iterations; i++) {
    for (size_t j = 0; j < size_byte / 32; j++) {
      __m256i *src_vec = (__m256i *)(src + j * 8);
      __m256i *dest_vec = (__m256i *)(dest + j * 8);
      _mm256_storeu_si256(dest_vec, _mm256_loadu_si256(src_vec));
    }
  }
  clock_t end = clock();
  double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
  double average_time = time_taken / iterations;
  double bandwidth =
      (double)(size_byte * sizeof(int32_t) * iterations) / time_taken;
  print_results("avx", size_byte, average_time, bandwidth);
}

void avx_gather_copy(size_t size_byte, int warmup, int iterations) {
  void *src = aligned_alloc(32, size_byte);
  uint32_t *indices = aligned_alloc(32, size_byte / 4 * sizeof(uint32_t));
  for (size_t i = 0; i < size_byte / 4; i++) {
    indices[i] = i;
  }
  void *dest = aligned_alloc(32, size_byte);

  set_src(src, size_byte);
  for (int i = 0; i < warmup; i++) {
    for (size_t j = 0; j < size_byte / 32; j++) {
      __m256i vindices = _mm256_loadu_si256((__m256i *)(indices + j * 8));
      // printf("indices[%zu] = %zu\n", j * 8, indices[j * 8]);
      __m256i *dest_vec = (__m256i *)(dest + j * 32);
      __m256i gathered = _mm256_i32gather_epi32(src, vindices, 4);
      _mm256_storeu_si256(dest_vec, gathered);
    }
    if (i == 0) {
      check_dest(dest, size_byte);
    }
  }

  clock_t start = clock();
  for (int i = 0; i < warmup; i++) {
    for (size_t j = 0; j < size_byte / 32; j++) {
      __m256i vindices = _mm256_loadu_si256((__m256i *)(indices + j * 8));
      __m256i *dest_vec = (__m256i *)(dest + j * 32);
      __m256i gathered = _mm256_i32gather_epi32(src, vindices, 4);
      _mm256_storeu_si256(dest_vec, gathered);
    }
  }
  clock_t end = clock();
  double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
  double average_time = time_taken / iterations;
  double bandwidth =
      (double)(size_byte * sizeof(int32_t) * iterations) / time_taken;
  print_results("avx_gather", size_byte, average_time, bandwidth);
}

int main() {
  const int WARMUP = 10;
  const int ITERATIONS = 100;
#define N_SIZE 11
  size_t sizes[N_SIZE] = {1UL << 20, 1UL << 21, 1UL << 22, 1UL << 23,
                          1UL << 24, 1UL << 25, 1UL << 26, 1UL << 27,
                          1UL << 28, 1UL << 29, 1UL << 30};
#define N_TEST_SIZE 4
  print_header();
  for (int i = 0; i < N_TEST_SIZE; i++) {
    memcpy_copy(sizes[i], WARMUP, ITERATIONS);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    avx_copy(sizes[i], WARMUP, ITERATIONS);
  }
  for (int i = 0; i < N_TEST_SIZE; i++) {
    avx_gather_copy(sizes[i], WARMUP, ITERATIONS);
  }
}
