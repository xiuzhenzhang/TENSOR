import gc

import torch


def free_model_from_gpu(model):
    using_cuda = next(model.parameters()).is_cuda
    del model

    gc.collect()
    if using_cuda:
        torch.cuda.empty_cache()
