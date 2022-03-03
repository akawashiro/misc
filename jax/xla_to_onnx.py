import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

sys.path.append("hlo_proto")

import hlo_pb2
import onnx
import onnxruntime as ort
from onnx import AttributeProto, GraphProto, TensorProto, helper


# See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#creating-an-onnx-model-using-helper-functions
def hlo_proto_to_onnx(hlo_proto, onnx_filename):
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [32, 32])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32])
    Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [32, 32])

    node_def = helper.make_node("Add", ["X", "Y"], ["Z"])
    graph_def = helper.make_graph([node_def], "test-model", [X, Y], [Z])

    op = onnx.OperatorSetIdProto()
    op.version = 13
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op]
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, onnx_filename)


def test_add():
    x = np.random.normal(size=(32, 32)).astype(np.float32)
    y = np.random.normal(size=(32, 32)).astype(np.float32)
    z = jnp.add(x, y)

    print(x)
    print(z)

    # TODO(akawashiro): Use inline=True to remove call
    add_jit = jit(jnp.add, inline=True)
    xla = jax.xla_computation(add_jit)(x, y)
    print(xla.as_hlo_text())

    with open("add_as_hlo_text.txt", "w") as f:
        f.write(xla.as_hlo_text())
    with open("add_as_hlo_dot_graph.dot", "w") as f:
        f.write(xla.as_hlo_dot_graph())

    hlo_proto = xla.as_serialized_hlo_module_proto()
    with open("add.hlo_proto", "wb") as f:
        f.write(hlo_proto)

    with open("add.hlo_proto", "rb") as f:
        add_hlo_proto_data = f.read()
        add_hlo_proto = hlo_pb2.HloModuleProto()
        add_hlo_proto.ParseFromString(add_hlo_proto_data)

    with open("add_hlo_proto.txt", "w") as f:
        f.write(str(add_hlo_proto))

    hlo_proto_to_onnx(add_hlo_proto, "add.onnx")
    ort_sess = ort.InferenceSession("add.onnx")
    outputs = ort_sess.run(None, {"X": x, "Y": y})
    assert np.allclose(z, outputs[0])


# def test_sum():
#     x = np.random.normal(size=(32, 32)).astype(np.float32)
#     y = jnp.sum(x)
#
#     print(x)
#
#     sum_jit = jit(jnp.sum, inline=True)
#     xla = jax.xla_computation(sum_jit)(x)
#     print(xla.as_hlo_text())
#
#     with open("sum_as_hlo_text.txt", "w") as f:
#         f.write(xla.as_hlo_text())
#     with open("sum_as_hlo_dot_graph.dot", "w") as f:
#         f.write(xla.as_hlo_dot_graph())
