#! /bin/bash

set -eux -o pipefail

iverilog -g 2012 -o test_flop test_flop.sv
vvp test_flop
