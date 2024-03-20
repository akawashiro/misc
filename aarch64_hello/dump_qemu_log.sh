#! /bin/bash

set -eux -o pipefail

${HOME}/tmp/qemu-install/bin/qemu-aarch64 -d in_asm,out_asm,op -D ./log ./hello-aarch64 && cat log
