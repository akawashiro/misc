#! /bin/bash

set -eux -o pipefail

python3 -m venv venv
source venv/bin/activate
pip3 install torch matplotlib pandas
python3 llama.py
