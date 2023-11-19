import torch
import triton
import xformers

from sfast.compilers.stable_diffusion_pipeline_compiler import (compile as sf_compile,
                                                               CompilationConfig)
def compile(model, method='torch'):
    if method == 'stable-fast':
        config = CompilationConfig.Default()
        config.enable_xformers = True
        config.enable_triton = True
        config.enable_cuda_graph = True

        return sf_compile(model, config)
    else:
        return torch.compile(model, mode="max-autotune", fullgraph=True)
