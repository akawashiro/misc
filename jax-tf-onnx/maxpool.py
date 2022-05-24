from jax.example_libraries.stax import MaxPool, BatchNorm
from jax import random as random
import functools
import onnx
import tensorflow as tf
from jax.experimental import jax2tf
import tf2onnx

rng_key = random.PRNGKey(0)

batch_size = 4
image_size = 16
channel_size = 1
input_shape = (batch_size, image_size, image_size, channel_size)

init_fun, predict_fun = MaxPool((3, 3))
_, init_params = init_fun(rng_key, input_shape)

print(init_params)


inference = predict_fun

inference_tf = jax2tf.convert(inference, enable_xla=False)
inference_tf = tf.function(inference_tf, autograph=False)

print(inference_tf)

tf_input = tf.ones(input_shape)
tf_output = inference_tf((), tf_input)
print(tf_output)

inference_onnx = tf2onnx.convert.from_function(
    inference_tf, input_signature=[tf.TensorSpec(()), tf.TensorSpec(input_shape)]
)
model_proto, external_tensor_storage = inference_onnx
onnx.save(model_proto, "maxpool.onnx")
