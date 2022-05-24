#! /bin/bash -eux

# python3 -m venv /tmp/brax_ant_myenv
source /tmp/brax_ant_myenv/bin/activate
# pip install -U pip
# pip install git+https://github.com/google/brax.git@main torch isort black

python3 main.py
