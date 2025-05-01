#! /bin/bash

set -eux -o pipefail

gcc -o compare compare.c

array_size=$((2 ** 20))
n_iterations=100

./compare ${array_size} linear 1 ${n_iterations}
for stride in 4 16 64 256 1023 1024 4096 16384; do
    ./compare ${array_size} random $stride ${n_iterations}
done
