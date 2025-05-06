#! /bin/bash

set -eux -o pipefail

gcc -o compare2 compare2.c -mavx2
./compare2
