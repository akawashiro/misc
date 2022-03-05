import subprocess
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import AttributeProto, GraphProto, TensorProto, TypeProto, helper

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

sys.path.append("hlo_proto")  # nopep8

import hlo_pb2  # nopep8
import xla_data_pb2  # nopep8


def translate_dtype(element_type):
    assert element_type in [xla_data_pb2.F32]
    if element_type == xla_data_pb2.F32:
        return TensorProto.FLOAT


def shape_proto_to_zeros(name: str, shape_proto):
    dims = shape_proto.dimensions
    dtype = translate_dtype(shape_proto.element_type)
    zeros = np.zeros(dims)
    return helper.make_tensor(name, data_type=dtype, dims=dims, vals=zeros)


def shape_proto_to_value_info_proto(name: str, shape_proto):
    dims = shape_proto.dimensions
    dtype = translate_dtype(shape_proto.element_type)
    return helper.make_tensor_value_info(name, dtype, dims)


def translate_inputs(parameters: List[Any]) -> Tuple[List[str], List[Any]]:
    names = []
    values = []
    for i in range(len(parameters)):
        names.append("input" + str(i))
        values.append(shape_proto_to_value_info_proto("input" + str(i), parameters[i]))
    return (names, values)


def translate_outputs(tuple_shapes: List[Any]) -> Tuple[List[str], List[Any]]:
    names = []
    values = []
    for i in range(len(tuple_shapes)):
        names.append("output" + str(i))
        values.append(
            shape_proto_to_value_info_proto("output" + str(i), tuple_shapes[i])
        )
    return (names, values)


gensym_id = 0


def gensym(prefix: str = "") -> str:
    global gensym_id
    gensym_id += 1
    return prefix + "gensym_" + str(gensym_id)


# Instruction -> [(name, ValueInfo, Node)]
def t_instruction(
    hlo_proto, instruction
) -> List[Tuple[str, Optional[Any], Optional[Any]]]:
    # XLA: https://www.tensorflow.org/xla/operation_semantics
    # ONNX: https://github.com/onnx/onnx/blob/main/docs/Operators.md
    if instruction.opcode == "parameter":
        name = str(instruction.id)
        value = shape_proto_to_value_info_proto(str(instruction.id), instruction.shape)
        return [(name, value, None)]
    elif instruction.opcode == "constant":
        # TODO:
        return [(str(instruction.id), None, None)]
    elif instruction.opcode == "add":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Add", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "subtract":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Sub", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "maximum":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Max", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "exponential":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Exp", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "log":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Log", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "dot":
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Gemm", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "tuple":
        # TODO:
        assert len(instruction.operand_ids) == 1
        inputs = list(map(lambda x: str(x), instruction.operand_ids))
        node = helper.make_node("Identity", inputs, [str(instruction.id)])
        return [(str(instruction.id), None, node)]
    elif instruction.opcode == "broadcast":
        # TODO: Adding dummy broadcasted value is wasteful clearly. I hope
        # post-process remove this dummy value with constant propagation.
        zero_id = gensym("broadcast_zero_")
        dummy_zeros = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[zero_id],
            value=shape_proto_to_zeros(
                gensym("broadcast_shape_proto_to_zeros_"), instruction.shape
            ),
        )
        inputs = list(map(lambda x: str(x), instruction.operand_ids)) + [zero_id]
        node = helper.make_node("Add", inputs, [str(instruction.id)])
        # Note: Nodes must be topologically sorted
        return [(zero_id, None, dummy_zeros), (str(instruction.id), None, node)]
    elif instruction.opcode == "reshape":
        shape_id = gensym("reshape_shape_")
        shape_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=[shape_id],
            value=helper.make_tensor(
                gensym("reshape_tensor_"),
                data_type=TensorProto.INT64,
                dims=[len(instruction.shape.dimensions)],
                vals=instruction.shape.dimensions,
            ),
        )
        inputs = list(map(lambda x: str(x), instruction.operand_ids)) + [shape_id]
        node = helper.make_node("Reshape", inputs=inputs, outputs=[str(instruction.id)])
        return [(shape_id, None, shape_node), (str(instruction.id), None, node)]
    elif instruction.opcode == "reduce":
        assert (
            len(instruction.called_computation_ids) == 1
        ), "Calling multiple computations in reduce opcode. It must be strange."
        reduce_op = get_computation(hlo_proto, instruction.called_computation_ids[0])
        if is_sum_reduce_op(reduce_op):
            assert len(instruction.operand_ids) == 2
            # TODO: The second oprand of reduce_sum must be 0 as the identity of monoid. We can ignore it for now.
            inputs = list(map(lambda x: str(x), instruction.operand_ids[:1]))
            node = helper.make_node("ReduceSum", inputs, [str(instruction.id)])
            return [(str(instruction.id), None, node)]
        if is_max_reduce_op(reduce_op):
            assert len(instruction.operand_ids) == 2
            # TODO: The second oprand of reduce_max must be -inf as the
            # identity of monoid. We can ignore it for now.
            inputs = list(map(lambda x: str(x), instruction.operand_ids[:1]))
            node = helper.make_node("ReduceMax", inputs, [str(instruction.id)])
            return [(str(instruction.id), None, node)]
        raise RuntimeError()
    else:
        raise RuntimeError(instruction.opcode + " is not supported yet!")


def is_sum_reduce_op(reduce_op):
    return (
        len(reduce_op.instructions) == 4 and reduce_op.instructions[3].opcode == "add"
    )


def is_max_reduce_op(reduce_op):
    return (
        len(reduce_op.instructions) == 4
        and reduce_op.instructions[3].opcode == "maximum"
    )


def get_computation(hlo_proto, computation_id):
    for c in hlo_proto.computations:
        if c.id == computation_id:
            return c
    raise RuntimeError("Cannot find computation of " + computation_id)


# See https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#creating-an-onnx-model-using-helper-functions
# Pass hlo_proto also because some operators such as reduce call other sub-computation.
def t_computation(hlo_proto, computation, onnx_filename):
    name_value_nodes = []
    for i in computation.instructions:
        name_value_nodes.extend(t_instruction(hlo_proto, i))
    input_values = []
    nodes = []
    for n, v, node in name_value_nodes:
        if v is not None:
            input_values.append(v)
        if node is not None:
            nodes.append(node)
    output_values = [
        shape_proto_to_value_info_proto(
            str(computation.root_id), computation.program_shape.result.tuple_shapes[0]
        )
    ]
    # print(name_value_nodes)
    # print("input_values = ", input_values)

    graph_def = helper.make_graph(nodes, "test-model", input_values, output_values)

    op = onnx.OperatorSetIdProto()
    op.version = 13
    model_def = helper.make_model(
        graph_def, producer_name="onnx-example", opset_imports=[op]
    )
    onnx.checker.check_model(model_def)
    onnx.save(model_def, onnx_filename)


def hlo_proto_to_onnx(hlo_proto, onnx_filename):
    main_computation = hlo_proto.computations[-1]
    assert (
        hlo_proto.entry_computation_name == main_computation.name
    ), "TODO: Translate only the main computation"
    t_computation(hlo_proto, main_computation, onnx_filename)


def gen_onnx_inputs(onnx_name, input_values):
    m = onnx.load(onnx_name)
    input_names = list(map(lambda x: x.name, m.graph.input))
    inputs = {}
    assert len(input_names) == len(input_values)
    for n, v in zip(input_names, input_values):
        inputs[n] = v
    return inputs


def translate_and_run(fn, input_values, test_name):
    onnx_name = test_name + ".onnx"

    # TODO(akawashiro): Use inline=True to remove call
    fn_jit = jit(fn, inline=True)
    xla = jax.xla_computation(fn_jit)(*input_values)

    with open(test_name + "_as_hlo_text.txt", "w") as f:
        f.write(xla.as_hlo_text())
    with open(test_name + "_as_hlo_dot_graph.dot", "w") as f:
        f.write(xla.as_hlo_dot_graph())
    dot_cmd = [
        "dot",
        "-Tps",
        test_name + "_as_hlo_dot_graph.dot",
        "-o",
        test_name + "_as_hlo_dot_graph.ps",
    ]
    subprocess.run(dot_cmd)

    hlo_proto = xla.as_serialized_hlo_module_proto()
    with open(test_name + ".hlo_proto", "wb") as f:
        f.write(hlo_proto)

    with open(test_name + ".hlo_proto", "rb") as f:
        hlo_proto_data = f.read()
        hlo_proto = hlo_pb2.HloModuleProto()
        hlo_proto.ParseFromString(hlo_proto_data)

    with open(test_name + "_hlo_proto.txt", "w") as f:
        f.write(str(hlo_proto))

    hlo_proto_to_onnx(hlo_proto, onnx_name)

    inputs = gen_onnx_inputs(onnx_name, input_values)
    ort_sess = ort.InferenceSession(onnx_name)
    outputs = ort_sess.run(None, inputs)
    return outputs
