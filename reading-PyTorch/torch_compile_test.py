import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + x

module = M()

opt_module = torch.compile(module)
print(opt_module)
