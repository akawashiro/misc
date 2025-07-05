#! /bin/bash

set -eux -o pipefail

cmake -S . \
    -B build \
    -D CMAKE_CXX_COMPILER=clang++-18 \
    -D CMAKE_CXX_FLAGS=-stdlib=libc++ \
    -G Ninja
cmake --build build
