#! /bin/bash

set -eu -o pipefail

gcc -o compare compare.c

array_size=$((2 ** 20))
n_iterations=100

taskset --cpu-list 0 ./compare ${array_size} linear 1 ${n_iterations}
for stride in 4 16 64 256 1023 1024 4096 16384; do
    taskset --cpu-list 0 ./compare ${array_size} random $stride ${n_iterations}
done

for array_size in $((2 ** 10)) $((2 ** 12)) $((2 ** 14)) $((2 ** 16)) $((2 ** 18)) $((2 ** 20)) $((2 ** 22)) $((2 ** 24)); do
    taskset --cpu-list 0 ./compare ${array_size} linear 1 ${n_iterations}
done

for array_size in $((2 ** 10)) $((2 ** 12)) $((2 ** 14)) $((2 ** 16)) $((2 ** 18)) $((2 ** 20)) $((2 ** 22)) $((2 ** 24)) $((2 ** 26)) $((2 ** 28)); do
    taskset --cpu-list 0 ./compare ${array_size} memcpy 1 ${n_iterations}
done
