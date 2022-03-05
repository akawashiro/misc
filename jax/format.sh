#! /bin/bash -eux

mypy xla_to_onnx.py --ignore-missing-imports
isort xla_to_onnx.py
black xla_to_onnx.py
