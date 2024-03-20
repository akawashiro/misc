#! /bin/bash

set -eux -o pipefail

aarch64-linux-gnu-gcc -o hello-aarch64 -static hello.c
