#!/bin/bash -eux

python3 -m venv venv
source venv/bin/activate
pip install transformers accelerate
python3 ./calm2-7b.py
