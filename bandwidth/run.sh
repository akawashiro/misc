#! /bin/bash

set -eux -o pipefail

g++ -o memcpy memcpy.cc
./memcpy
g++ -o memcpy_mt memcpy_mt.cc
./memcpy_mt
