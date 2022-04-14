import functools

import onnx
import tensorflow as tf
import tf2onnx
from jax import random as random
from jax.example_libraries.stax import BatchNorm, MaxPool
from jax.experimental import jax2tf
from jax import grad

rng_key = random.PRNGKey(0)

batch_size = 4
image_size = 16
channel_size = 1
input_shape = (batch_size, image_size, image_size, channel_size)

init_fun, predict_fun = BatchNorm()
_, init_params = init_fun(rng_key, input_shape)

print(init_params)
param_shape = []
tf_param_shape = []
for p in init_params:
    param_shape.append(p.shape)
    tf_param_shape.append(tf.TensorSpec(p.shape))


inference_tf = jax2tf.convert(predict_fun, enable_xla=False)
inference_tf = tf.function(inference_tf, autograph=False)

print(inference_tf)

tf_input0 = tf.ones((1,))
tf_input1 = tf.ones((1,))
tf_input = tf.ones(input_shape)
tf_output = inference_tf((tf_input0, tf_input1), tf_input)
print(tf_output)

inference_onnx = tf2onnx.convert.from_function(
    inference_tf,
    input_signature=[
        tf_param_shape,
        tf.TensorSpec(input_shape),
    ],
)
model_proto, external_tensor_storage = inference_onnx
onnx.save(model_proto, "bn.onnx")
