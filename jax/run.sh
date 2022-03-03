#! /bin/bash -eux

# Generate mnist.hlo_proto
# python3 mnist.py

# Generate hlo_proto
# rm -rf hlo_proto
# mkdir hlo_proto
# protoc --python_out=hlo_proto -I=hlo_proto_def hlo_proto_def/xla_data.proto
# protoc --python_out=hlo_proto -I=hlo_proto_def hlo_proto_def/hlo.proto

# python3 load_hlo_proto.py
# dot -Tps mnist_as_hlo_dot_graph.dot -o mnist_as_hlo_dot_graph.ps

# isort xla_to_onnx.py
# black xla_to_onnx.py
pytest -s xla_to_onnx.py
dot -Tps add_as_hlo_dot_graph.dot -o add_as_hlo_dot_graph.ps
dot -Tps sum_as_hlo_dot_graph.dot -o sum_as_hlo_dot_graph.ps
