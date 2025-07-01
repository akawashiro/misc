#! /bin/bash

set -eux -o pipefail

cmake -S . -B build -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS=-stdlib=libc++
cmake --build build
# ./build/memcpy
# ./build/memcpy_mt
# ./build/udp
./build/uds
./build/tcp
