#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Function to perform linear access copy
void linear_copy(int *src, int *dest, int size) {
  for (int i = 0; i < size; i++) {
    dest[i] = src[i];
  }
}

// Function to perform strided random access copy
void strided_copy(int *src, int *dest, int size, int stride) {
  for (int s = 0; s < stride; s++) {
    for (int i = s; i < size; i += stride) {
      dest[i] = src[i];
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <size> <access_type (linear or random)> <stride (for "
           "random)> <iterations>\n",
           argv[0]);
    printf("  <size>: Size of the array.\n");
    printf("  <access_type>: 'linear' for linear access, 'random' for strided "
           "random access.\n");
    printf("  <stride>: Stride value for random access. Use 1 for linear "
           "access.\n");
    printf("  <iterations>: Number of times to perform the copy operation.\n");
    return 1;
  }

  int size = atoi(argv[1]);
  char *access_type = argv[2];
  int stride = atoi(argv[3]);
  int iterations = atoi(argv[4]);

  if (size <= 0 || iterations <= 0 ||
      (strcmp(access_type, "random") == 0 && stride <= 0)) {
    printf("Invalid input parameters.\n");
    return 1;
  }

  int *src = (int *)malloc(size * sizeof(int));
  int *dest = (int *)malloc(size * sizeof(int));

  if (src == NULL || dest == NULL) {
    perror("Failed to allocate memory");
    return 1;
  }

  // Initialize the source array
  for (int i = 0; i < size; i++) {
    src[i] = i;
  }

  clock_t start, end;
  double cpu_time_used;

  start = clock();

  for (int iter = 0; iter < iterations; iter++) {
    if (strcmp(access_type, "linear") == 0) {
      linear_copy(src, dest, size);
    } else if (strcmp(access_type, "random") == 0) {
      strided_copy(src, dest, size, stride);
    } else {
      printf("Invalid access type: %s\n", access_type);
      free(src);
      free(dest);
      return 1;
    }
  }

  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; // in seconds
  double memory_bandwidth =
      (double)(size * sizeof(int) * iterations) / cpu_time_used;

  printf("Array size: %d\n", size);
  printf("Access type: %s\n", access_type);
  if (strcmp(access_type, "random") == 0) {
    printf("Stride: %d\n", stride);
  }
  printf("Iterations: %d\n", iterations);
  printf("Execution time for one copy: %.2f milliseconds\n",
         cpu_time_used / iterations * 1000);
  printf("Memory bandwidth: %.2f MB/s\n",
         memory_bandwidth / (1024 * 1024)); // Convert to MB/s

  free(src);
  free(dest);

  return 0;
}
