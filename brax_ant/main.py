import functools
import time
import subprocess

import brax
# from IPython.display import HTML, Image
import gym
import jax
import torch
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html, image
from jax import numpy as jnp

v = torch.ones(1, device="cuda")  # init torch cuda before jax

environment = "ant"  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'reacher', 'walker2d', 'fetch', 'grasp', 'ur5e']
env = envs.create(env_name=environment)
state = env.reset(rng=jp.random_prngkey(seed=0))

with open("vis.html", "w") as f:
    f.write(html.render(env.sys, [state.qp]))

rollout = []
for i in range(100):
    # wiggle sinusoidally with a phase shift per actuator
    action = jp.sin(i * jp.pi / 15 + jp.arange(0, env.action_size) * jp.pi)
    state = env.step(state, action)
    rollout.append(state)

state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))

for _ in range(100):
    state = jax.jit(env.step)(state, jnp.ones((env.action_size,)))

with open("rollout.html", "w") as f:
    f.write(html.render(env.sys, [s.qp for s in rollout]))


fn_jit = jax.jit(env.step, inline=True)
xla = jax.xla_computation(fn_jit)(state, jnp.ones((env.action_size,)))

with open("step_as_hlo_text.txt", "w") as f:
    f.write(xla.as_hlo_text())
with open("step_as_hlo_dot_graph.dot", "w") as f:
    f.write(xla.as_hlo_dot_graph())
    dot_cmd = [
        "dot",
        "-Tsvg",
        "step_as_hlo_dot_graph.dot",
        "-o",
        "step_as_hlo_dot_graph.svg",
    ]
    subprocess.run(dot_cmd)

