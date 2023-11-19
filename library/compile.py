# from sfast.compilers.stable_diffusion_pipeline_compiler import (compile as sf_compile,
#                                                                CompilationConfig)
def compile(model):
    return torch.compile(model, mode="max-autotune", fullgraph=True)
    # config = CompilationConfig.Default()

    # # xformers and Triton are suggested for achieving best performance.
    # # It might be slow for Triton to generate, compile and fine-tune kernels.
    # try:
    #     import xformers
    #     config.enable_xformers = True
    # except ImportError:
    #     print('xformers not installed, skip')
    # # NOTE:
    # # When GPU VRAM is insufficient or the architecture is too old, Triton might be slow.
    # # Disable Triton if you encounter this problem.
    # try:
    #     import triton
    #     config.enable_triton = True
    # except ImportError:
    #     print('Triton not installed, skip')
    # # NOTE:
    # # CUDA Graph is suggested for small batch sizes and small resolutions to reduce CPU overhead.
    # # My implementation can handle dynamic shape with increased need for GPU memory.
    # # But when your GPU VRAM is insufficient or the image resolution is high,
    # # CUDA Graph could cause less efficient VRAM utilization and slow down the inference,
    # # especially when on Windows or WSL which has the "shared VRAM" mechanism.
    # # If you meet problems related to it, you should disable it.
    # config.enable_cuda_graph = True

    # return sf_compile(model, config)
