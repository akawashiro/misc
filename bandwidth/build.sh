#! /bin/bash

set -eux -o pipefail

git submodule update --init --recursive
cmake -S . \
    -B build \
    -D CMAKE_CXX_COMPILER=clang++-18 \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -G Ninja
cmake --build build
