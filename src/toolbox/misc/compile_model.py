import torch

from src.toolbox.misc.get_logger import get_logger

logger = get_logger(__name__)


def compile_model(model, use_compile, backend=None, *args, **kwargs):
    if use_compile and backend:
        logger.warning(
            "Optimizing the model by torch.compile(). This process may not work OOTB in some cases because of unsupported devices, out-of-date graphic drivers, wrong triton installation, specific model design, pytorch bugs, etc."
        )

        return torch.compile(
            model, *args, backend=backend, dynamic=False, fullgraph=True, mode='max-autotune-no-cudagraphs', disable=not use_compile, **kwargs
        )
    return model


def conditional_compile_func(func, compile_or_not, backend, *args, **kwargs):
    return torch.compile(func, backend=backend, dynamic=False, mode='max-autotune', disable=not compile_or_not, *args, **kwargs)
