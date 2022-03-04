import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

sys.path.append("hlo_proto")

import hlo_pb2
import onnx
import onnxruntime as ort
import xla_data_pb2
from onnx import AttributeProto, GraphProto, TensorProto, helper


def translate_dtype(element_type):
    assert element_type in [xla_data_pb2.F32]
    if element_type == xla_data_pb2.F32:
        return TensorProto.FLOAT


def shape_proto_to_value_info_proto(name: str, shape_proto):
    dims = shape_proto.dimensions
    dtype = translate_dtype(shape_proto.element_type)
    return helper.make_tensor_value_info(name, dtype, dims)


def translate_inputs(parameters):
    names = []
    values = []
    for i in range(len(parameters)):
        names.append("input" + str(i))
        values.append(shape_proto_to_value_info_proto("input" + str(i), parameters[i]))
    return (names, values)


def translate_outputs(tuple_shapes):
    names = []
    values = []
    for i in range(len(tuple_shapes)):
        names.append("output" + str(i))
        values.append(
            shape_proto_to_value_info_proto("output" + str(i), tuple_shapes[i])
        )
    return (names, values)


# Instruction -> (name, ValueInfo, Node)
def t_instruction(instruction):
    assert(instruction.opcode in ["parameter", "constant", "add", "tuple"])
    if instruction.opcode == "parameter":
        name = str(instruction.id)
        value = shape_proto_to_value_info_proto(str(instruction.id), instruction.shape)
        return (name, value, None)
    elif instruction.opcode == "constant":
        # TODO:
        return (str(instruction.id), None, None)
    elif instruction.opcode == "add":
        inputs = list(map(lambda x:str(x), instruction.operand_ids))
        node = helper.make_node("Add", inputs, [str(instruction.id)])
        return (str(instruction.id), None, node)
    elif instruction.opcode == "tuple":
        # TODO:
        assert(len(instruction.operand_ids) == 1)
        inputs = list(map(lambda x:str(x), instruction.operand_ids))
        node = helper.make_node("Identity", inputs, [str(instruction.id)])
        return (str(instruction.id), None, node)


def t_computation(computation):
    name_value_nodes = list(map(t_instruction, computation.instructions))
    input_values = []
    for n, v, _ in name_value_nodes:
        if v is not None:
            input_values.append(n)
    print(name_value_nodes)
    print("input_values = ", input_values)


# See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#creating-an-onnx-model-using-helper-functions
def hlo_proto_to_onnx(hlo_proto, onnx_filename):
    t_computation(hlo_proto.computations[0])

    input_names, input_values = translate_inputs(
        hlo_proto.host_program_shape.parameters
    )
    output_names, output_values = translate_outputs(
        hlo_proto.host_program_shape.result.tuple_shapes
    )

    node_def = helper.make_node("Add", input_names, output_names)
    graph_def = helper.make_graph([node_def], "test-model", input_values, output_values)

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
    input_values = [x, y]
    z = jnp.add(x, y)

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

    m = onnx.load("add.onnx")
    input_names = list(map(lambda x: x.name, m.graph.input))
    inputs = {}
    assert len(input_names) == len(input_values)
    for n, v in zip(input_names, input_values):
        inputs[n] = v
    hlo_proto_to_onnx(add_hlo_proto, "add.onnx")
    ort_sess = ort.InferenceSession("add.onnx")
    outputs = ort_sess.run(None, inputs)
    assert np.allclose(z, outputs[0])


def test_sum():
    x = np.random.normal(size=(32, 32)).astype(np.float32)
    # y = jnp.sum(x)

    print(x)

    sum_jit = jit(jnp.sum, inline=True)
    xla = jax.xla_computation(sum_jit)(x)
    print(xla.as_hlo_text())

    with open("sum_as_hlo_text.txt", "w") as f:
        f.write(xla.as_hlo_text())
    with open("sum_as_hlo_dot_graph.dot", "w") as f:
        f.write(xla.as_hlo_dot_graph())

    hlo_proto = xla.as_serialized_hlo_module_proto()
    with open("sum.hlo_proto", "wb") as f:
        f.write(hlo_proto)

    with open("sum.hlo_proto", "rb") as f:
        add_hlo_proto_data = f.read()
        add_hlo_proto = hlo_pb2.HloModuleProto()
        add_hlo_proto.ParseFromString(add_hlo_proto_data)

    with open("sum_hlo_proto.txt", "w") as f:
        f.write(str(add_hlo_proto))

