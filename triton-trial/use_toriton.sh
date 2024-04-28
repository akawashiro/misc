#! /bin/bash

set -eux -o pipefail

SCRIPT_DIR=$(cd $(dirname $0); pwd)
source ${SCRIPT_DIR}/build_and_install_triton_venv/bin/activate

python3 ${SCRIPT_DIR}/01-vector-add.py
