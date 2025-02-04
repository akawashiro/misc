#! /bin/bash

set -eux -o pipefail

iverilog -g 2012 -o test_halfadder test_halfadder.sv
vvp test_halfadder
