#! /bin/bash

set -eux -o pipefail

g++ -o compare2 compare2.cc -mavx2 -Wall -mavx512f
./compare2
