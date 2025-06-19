#! /bin/bash

set -eux -o pipefail

g++ main.cc -o main
./main
