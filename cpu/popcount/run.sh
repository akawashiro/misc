#! /bin/bash

set -eux -o pipefail

iverilog -g 2012 -o test_popcnt test_popcnt.sv
vvp test_popcnt
