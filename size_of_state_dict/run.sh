#! /bin/bash

set -eux -o pipefail

python3 -m venv venv
source venv/bin/activate
pip install transformers torch datasets pillow accelerate

python3 main.py
