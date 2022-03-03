#! /bin/bash -eux

isort xla_to_onnx.py
black xla_to_onnx.py
