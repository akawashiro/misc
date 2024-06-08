#! /bin/bash

set -eux -o pipefail

N_INSTALLED=$(dpkg -l | grep -e bpfcc-tools -e python3-bpfcc | wc -l)
if [ $N_INSTALLED -ne 2 ]; then
    echo "Installing bpfcc-tools and python3-bpfcc"
    sudo apt update
    sudo apt install bpfcc-tools python3-bpfcc
fi

# sudo python3 ./hello.py
sudo python3 ./execsnoop.py
