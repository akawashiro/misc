#! /bin/bash

set -eux -o pipefail
g++ -o memcpy memcpy.cc
./memcpy
