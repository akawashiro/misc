#! /bin/bash

set -eux -o pipefail

SCRIPT_DIR=$(cd $(dirname $0); pwd)
VENV_DIR=${SCRIPT_DIR}/venv

if [[ ! -d $VENV_DIR ]]; then
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate
pip3 install torch

python3 main.py
