#! /bin/bash -eux

# if [ ! -d myenv ]; then
#     python3 -m venv myenv
# fi
# 
# source myenv/bin/activate
# pip install tf2onnx git+https://github.com/deepmind/dm-haiku "jax[cpu]" tensorflow
# python3 main.py
# python3 resnet.py
python3 maxpool.py
