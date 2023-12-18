import torch
import triton
import xformers

def compile(model, mode='max-autotune'):
    return torch.compile(model, mode, fullgraph=True)
