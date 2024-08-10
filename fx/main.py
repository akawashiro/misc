import torch

def func_to_trace(x):
    return torch.relu(x) + torch.tanh(x)

traced = torch.fx.symbolic_trace(func_to_trace)
print(f"{type(traced)=}")
print(f"traced.graph: {traced.graph}")

def slice_input(x):
    return x[0]

traced = torch.fx.symbolic_trace(slice_input)
print(f"{type(traced)=}")
print(f"traced.graph: {traced.graph}")
