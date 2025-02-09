#! /bin/bash

set -eux -o pipefail

iverilog -g 2012 -o test_cpu test_cpu.sv -W all
vvp test_cpu
