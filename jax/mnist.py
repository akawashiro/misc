# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.
The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""


import time
import itertools

import numpy.random as npr

import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
# from examples import datasets
import datasets


def loss(params, batch):
  inputs, targets = batch
  l = predict_jit.lower(params, inputs)
  print(l)
  print(type(l))
  print(l.compile())
  print(l.compile().compiler_ir())

  c = jax.xla_computation(predict)(params, inputs)
  print(type(c))
  print(c.as_hlo_text())  # doctest: +SKIP
  with open("mnist_as_hlo_text.txt", "w") as f:
      f.write(c.as_hlo_text())
  with open("mnist_as_hlo_dot_graph.dot", "w") as f:
      f.write(c.as_hlo_dot_graph())
  # print(c.as_serialized_hlo_module_proto())
  # hlo_proto = c.encode()
  hlo_proto = c.as_serialized_hlo_module_proto()
  print(type(hlo_proto))
  # print(hlo_proto)
  with open("mnist.hlo_proto", "wb") as f:
      f.write(hlo_proto)
  import sys
  sys.exit()

  preds = predict_jit(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)
predict_jit = jit(predict, inline=True)
print(predict_jit)
print(type(predict_jit))


if __name__ == "__main__":
  rng = random.PRNGKey(0)

  step_size = 0.001
  num_epochs = 2
  batch_size = 128
  momentum_mass = 0.9

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  print("\nStarting training...")
  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      opt_state = update(next(itercount), opt_state, next(batches))
    epoch_time = time.time() - start_time

    params = get_params(opt_state)
    train_acc = accuracy(params, (train_images, train_labels))
    test_acc = accuracy(params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
