#! /bin/bash

set -eux -o pipefail

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip3 install torch transformers datasets

python3 train.py
