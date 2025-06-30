#! /bin/bash

set -eux -o pipefail

cmake -S . -B build
cmake --build build
./build/memcpy
# ./build/memcpy_mt
./build/udp
