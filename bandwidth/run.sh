#! /bin/bash

set -eux -o pipefail

./build.sh
./build/memcpy
./build/memcpy_mt
# ./build/udp
./build/uds
./build/tcp
./build/pipe
./build/shm
