#!/bin/bash -eux

maturin develop
RUST_LOG=info python3 ./use_pyrjit.py
