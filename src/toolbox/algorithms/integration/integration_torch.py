import torch
from einops import rearrange

'''
Approximate an integral based on its definition.
dim refers to the dimension index of expanded_func_value where the integration should be performed.
'''
def approximate_integration_torch(expanded_func_value, expanded_x, dim, only_integral = False, func_val_x_having_same_shape = False):
    # tensor check
    func_val_number_of_dim = len(expanded_func_value.shape)
    device = expanded_func_value.device

    if not func_val_x_having_same_shape:
        the_number_of_dimensions_after_integration_dim = abs(dim) - 1 if dim < 0 else func_val_number_of_dim - dim - 1
        einop = f'... -> ... {"() " * the_number_of_dimensions_after_integration_dim}'
        expanded_x = rearrange(expanded_x, einop)                              # [..., integration_sample_rate - 1, ...]

    if only_integral:
        integral_of_all_events = torch.trapezoid(y = expanded_func_value, x = expanded_x, dim = dim)
                                                                               # [...]
    else:
        integral_of_all_events = torch.cumulative_trapezoid(y = expanded_func_value, x = expanded_x, dim = dim)
                                                                               # [..., integration_sample_rate, ...]
        # Prepend 0 to integral_of_all_events because \\int_{t_l}^{t_l}{\\lambda^*(\\tau)d\\tau} = 0
        # We have to check the shape.
        integral_start_from_zero = torch.zeros(
            ( *(integral_of_all_events.shape[:dim]), 1, *(integral_of_all_events.shape[dim + 1:] if dim != -1 else []) ), 
            device = device)                                                   # [..., 1, ...]
        integral_of_all_events = torch.concat((integral_start_from_zero, integral_of_all_events), dim = dim)
                                                                               # [..., integration_sample_rate, ...]

    return integral_of_all_events
