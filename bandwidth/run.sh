#! /bin/bash

set -eux -o pipefail

./build/memcpy
./build/memcpy_mt
# ./build/udp
./build/uds
./build/tcp
