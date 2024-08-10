#! /bin/bash

set -eux -o pipefail

VENV_DIR=/tmp/fixed-kv-cache-venv

if [[ ! -d $VENV_DIR ]]; then
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
else
    source $VENV_DIR/bin/activate
fi

pip3 install accelerate transformers torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
python3 main.py
