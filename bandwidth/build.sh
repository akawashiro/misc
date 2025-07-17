#! /bin/bash

set -eux -o pipefail

git submodule update --init --recursive
cmake -S . \
    -B build \
    -D CMAKE_CXX_COMPILER=clang++ \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -D CMAKE_CXX_FLAGS="-stdlib=libc++" \
    -G Ninja
cmake --build build
