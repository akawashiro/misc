# Copied from https://github.com/google/jax/issues/7629 and modified by Akira Kawata

# The actual JAX program you stage out is not important. I am using a Haiku
# neural network, but this will work with any JAX NN library.

import jax
import jax.numpy as jnp
import haiku as hk


def f(x):
    return hk.nets.MLP([300, 100, 10])(x)


f = hk.transform(f)
rng = jax.random.PRNGKey(42)
x = jnp.ones([1, 1])
params = f.init(rng, x)

# If you want to save an "inference only" version of your function just close
# over the params. Some TF users refer to this as a "frozen graph".
import functools

inference = functools.partial(f.apply, params, None)

# jax2tf enables us to easily go from JAX -> TensorFlow.
import tensorflow as tf
from jax.experimental import jax2tf

inference_tf = jax2tf.convert(inference, enable_xla=False)
inference_tf = tf.function(inference_tf, autograph=False)

# tf2onnx allows TF programs to be staged out as onnx protos.
import tf2onnx

inference_onnx = tf2onnx.convert.from_function(
    inference_tf, input_signature=[tf.TensorSpec([1, 1])]
)
model_proto, external_tensor_storage = inference_onnx

print(model_proto)
