import functools

import onnx
import tensorflow as tf
import tf2onnx
from jax import grad
from jax import random as random
from jax.example_libraries.stax import (AvgPool, BatchNorm, Conv, Dense,
                                        FanInSum, FanOut, Flatten, GeneralConv,
                                        Identity, LogSoftmax, MaxPool, Relu)
from jax.experimental import jax2tf

rng_key = random.PRNGKey(0)


def ConvBlock(kernel_size, filters, strides=(2, 2)):
    ks = kernel_size
    filters1, filters2, filters3 = filters
    Main = stax.serial(
        Conv(filters1, (1, 1), strides),
        BatchNorm(),
        Relu,
        Conv(filters2, (ks, ks), padding="SAME"),
        BatchNorm(),
        Relu,
        Conv(filters3, (1, 1)),
        BatchNorm(),
    )
    Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
    return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
    ks = kernel_size
    filters1, filters2 = filters

    def make_main(input_shape):
        # the number of output channels depends on the number of input channels
        return stax.serial(
            Conv(filters1, (1, 1)),
            BatchNorm(),
            Relu,
            Conv(filters2, (ks, ks), padding="SAME"),
            BatchNorm(),
            Relu,
            Conv(input_shape[3], (1, 1)),
            BatchNorm(),
        )

    Main = stax.shape_dependent(make_main)
    return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks


def ResNet50(num_classes):
    return stax.serial(
        GeneralConv(("HWCN", "OIHW", "NHWC"), 64, (7, 7), (2, 2), "SAME"),
        BatchNorm(),
        Relu,
        MaxPool((3, 3), strides=(2, 2)),
        ConvBlock(3, [64, 64, 256], strides=(1, 1)),
        IdentityBlock(3, [64, 64]),
        IdentityBlock(3, [64, 64]),
        ConvBlock(3, [128, 128, 512]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        IdentityBlock(3, [128, 128]),
        ConvBlock(3, [256, 256, 1024]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        IdentityBlock(3, [256, 256]),
        ConvBlock(3, [512, 512, 2048]),
        IdentityBlock(3, [512, 512]),
        IdentityBlock(3, [512, 512]),
        AvgPool((7, 7)),
        Flatten,
        Dense(num_classes),
        LogSoftmax,
    )


batch_size = 128
image_size = 224
channel_size = 3
input_shape = (image_size, image_size, channel_size, batch_size)

init_fun, predict_fun = AvgPool((3, 3))
_, init_params = init_fun(rng_key, input_shape)

param_shape = []
tf_param_shape = []
for p in init_params:
    param_shape.append(p.shape)
    tf_param_shape.append(tf.TensorSpec(p.shape))


inference_tf = jax2tf.convert(predict_fun, enable_xla=False)
inference_tf = tf.function(inference_tf, autograph=False)

inference_onnx = tf2onnx.convert.from_function(
    inference_tf,
    input_signature=[
        tf_param_shape,
        tf.TensorSpec(input_shape),
    ],
)
model_proto, external_tensor_storage = inference_onnx
onnx.save(model_proto, "resnet.onnx")
