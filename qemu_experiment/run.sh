#! /bin/bash

set -eux -o pipefail

aarch64-linux-gnu-gcc -o hello-aarch64 -static hello.c
${HOME}/tmp/qemu-install/bin/qemu-aarch64 -d in_asm,out_asm,op -D ./qemu_experiment.log ./hello-aarch64
cat qemu_experiment.log
