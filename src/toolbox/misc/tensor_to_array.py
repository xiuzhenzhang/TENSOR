import torch


def move_from_tensor_to_ndarray(*args):
    '''
    This function converts an arbitrary number of torch.tensors to np.arrays.
    This function can automaticly move cuda tensors to cpu.
    '''
    def move_tensor(x):
        if torch.is_tensor(x):
            if x.dtype == torch.bfloat16:
                return x.detach().to(torch.float32).cpu().numpy()
            return x.detach().cpu().numpy()
        return x

    if len(args) == 1:
        tmp_results = move_tensor(args[0])
    else:
        tmp_results = []
        for item in args:
            tmp_results.append(move_tensor(item))

    return tmp_results


def move_from_tensor_to_list(*args):
    '''
    This function converts an arbitrary number of torch.tensors to lists.
    This function can automaticly move cuda tensors to cpu.
    '''
    def move_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        return x

    if len(args) == 1:
        tmp_results = move_tensor(args[0])
    else:
        tmp_results = []
        for item in args:
            tmp_results.append(move_tensor(item))

    return tmp_results
