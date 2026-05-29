from operator import itemgetter

import numpy as np
import torch

from src.toolbox.misc import move_from_tensor_to_ndarray


def expand_true_intensity(time, intensity, opt):
    """
    Entry function of calculating the true intensity functions.
    """
    return true_intensity_dict[opt.info_dict["dataset_name"]](time, intensity, opt, device=opt.device)
    # [batch_size, seq_len, resolution]


def expand_true_probability(time, intensity, opt):
    """
    Entry function of calculating the true probability distribution.
    """
    functions = true_probability_dict[opt.info_dict["dataset_name"]]

    if len(functions) == 2:
        """
        Two functions means you should combine the intensity function and corresponding integral function to
        obtain the final probability distribution.
        """
        expand_true_intensity = functions[0](
            time, intensity, opt, device=opt.device
        )  # [batch_size, seq_len, resolution]
        expand_true_integral = functions[1](
            time, intensity, opt, device=opt.device
        )  # [batch_size, seq_len, resolution]
        return expand_true_intensity * torch.exp(-expand_true_integral)  # [batch_size, seq_len, resolution]

    """
    While for several special tpps defined by probability distributions instead of intensity functions, thing are quite
    easier: go find the distribution and the task is done.
    """
    return functions[0](time, intensity, opt, device=opt.device)
    # [batch_size, seq_len, resolution]


def hawkes_1(time, intensity, opt, device):
    """
    Hawkes_1 process: \\lambda(t) = \\mu + a * b * exp(-b(t - t_l))
    The dataset must provide the value of \\mu, a, and b in their dataset_card.yml.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, a, b = itemgetter("mu", "a", "b")(opt.info_dict)
    resolution = opt.resolution

    batch_size = time.shape[0]
    intensity = torch.cat((torch.zeros((batch_size, 1), device=device), intensity[:, :-1]), dim=-1)

    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    true_intensity = intensity.unsqueeze(-1).repeat(1, 1, resolution) - mu + a * b
    # [batch_size, seq_len, resolution]
    intensity_multiplier_matrix = torch.exp(-b * expand_time)  # [batch_size, seq_len, resolution]
    expand_true_intensity = true_intensity * intensity_multiplier_matrix + mu  # [batch_size, seq_len, resolution]
    expand_true_intensity[:, 0, :] = mu
    return expand_true_intensity


def hawkes_1_integral(time, intensity, opt, device):
    """
    Hawkes_1 process: \\Lambda(t) = \\mu * (t - t_l) + a - a * exp(-b(t - t_l)). When t = t_l, \\Lambda(t) = 0.
    The dataset must provide the value of \\mu, a, and b in their dataset_card.yml.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, a, b = itemgetter("mu", "a", "b")(opt.info_dict)
    resolution = opt.resolution

    # Integral part 1
    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    mu_integral = mu * expand_time  # [batch_size, seq_len, resolution]
    basic_exponential_integral = a - a * torch.exp(-b * expand_time)  # [batch_size, seq_len, resolution]

    # Integral part 2
    table = torch.diag_embed(time[:, 1:-1], offset=-2)  # [batch_size, seq_len, seq_len]
    table = torch.cumsum(table, dim=-2)  # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table = torch.cumsum(table.flip(-1), dim=-1).flip(-1)  # [batch_size, seq_len, seq_len]
    table_mask = (table != 0).int()  # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table *= table_mask  # [batch_size, seq_len, seq_len]
    historical_multiplier = torch.exp(-b * reversed_cumsum_of_table)  # [batch_size, seq_len, seq_len]
    historical_multiplier *= table_mask  # [batch_size, seq_len, seq_len]

    historical_integral = basic_exponential_integral.unsqueeze(-1) * historical_multiplier.unsqueeze(-2)
    # [batch_size, seq_len, resolution, seq_len]
    historical_integral = torch.sum(historical_integral, dim=-1)  # [batch_size, seq_len, resolution]

    # Get the integral
    expand_true_integral = mu_integral + basic_exponential_integral + historical_integral
    # [batch_size, seq_len, resolution]
    expand_true_integral[:, 0, :] = mu_integral[:, 0, :]  # [batch_size, seq_len, resolution]

    return expand_true_integral


"""
Hawkes process whose intensity function has multiple kernels.
"""


def hawkes_2(time, intensity, opt, device):
    """
    Hawkes_2 process: \\lambda(t) = \\mu + a_1 * b_1 * exp(-b_1(t - t_l)) + a_2 * b_2 * exp(-b_2(t - t_l))
    The dataset must provide the value of \\mu, a_1, , a_2, b_1, and b_2 in their dataset_card.yml.

    It seems that we have no choice but to solve the intensity iteratively.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, a_1, a_2, b_1, b_2 = itemgetter("mu", "a_1", "a_2", "b_1", "b_2")(opt.info_dict)
    resolution = opt.resolution

    batch_size, seq_len = time.shape
    """
    Imitate the original input_time
    """
    time = torch.cat((torch.zeros((batch_size, 1), device=device), time), dim=-1)
    # [batch_size, seq_len + 1]
    p1d = (1, 0, 0, 0)

    expand_true_intensity = torch.ones((batch_size, seq_len, resolution), device=device) * mu
    # [batch_size, seq_len, resolution]
    expand_time = (time.unsqueeze(-1) / (resolution - 1)).repeat(1, 1, resolution - 1)
    # [batch_size, seq_len + 1, resolution - 1]
    expand_time = torch.nn.functional.pad(expand_time, p1d)  # [batch_size, seq_len + 1, resolution]

    time_cumsum = torch.cumsum(expand_time.reshape(batch_size, -1), dim=-1)  # [batch_size, (seq_len + 1) * resolution]
    time_cumsum = time_cumsum.reshape(batch_size, seq_len + 1, resolution)  # [batch_size, (seq_len + 1), resolution]
    for seq_index in range(2, seq_len + 1):
        expand_batch_time = time_cumsum[:, seq_index:, :] - time_cumsum[:, seq_index, 0].reshape(batch_size, 1, 1)
        # [batch_size, seq_len - seq_index + 1, resolution]
        expand_intensity_add = a_1 * b_1 * torch.exp(-b_1 * expand_batch_time) + a_2 * b_2 * torch.exp(
            -b_2 * expand_batch_time
        )
        # [batch_size, seq_len - seq_index + 1, resolution]
        p2d = (0, 0, seq_index - 1, 0)
        expand_true_intensity += torch.nn.functional.pad(expand_intensity_add, p2d)
        # [batch_size, seq_len, resolution]

    expand_true_intensity[:, 0, :] = mu
    return expand_true_intensity


def hawkes_2_integral(time, intensity, opt, device):
    """
    Hawkes_2 process: \\Lambda(t) = \\mu * (t - t_l) + a_1 - a_1 * exp(-b_1(t - t_l)) + a_2 - a_2 * exp(-b_2(t - t_l)).
    When t = t_l, \\Lambda(t) = 0.
    The dataset must provide the value of \\mu, a_1, , a_2, b_1, and b_2 in their dataset_card.yml.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, a_1, a_2, b_1, b_2 = itemgetter("mu", "a_1", "a_2", "b_1", "b_2")(opt.info_dict)
    resolution = opt.resolution

    # Integral part 1
    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    mu_integral = mu * expand_time  # [batch_size, seq_len, resolution]
    basic_exponential_integral_1 = a_1 - a_1 * torch.exp(-b_1 * expand_time)  # [batch_size, seq_len, resolution]
    basic_exponential_integral_2 = a_2 - a_2 * torch.exp(-b_2 * expand_time)  # [batch_size, seq_len, resolution]
    basic_exponential_integral = basic_exponential_integral_1 + basic_exponential_integral_2
    # [batch_size, seq_len, resolution]

    # Integral part 2
    table = torch.diag_embed(time[:, 1:-1], offset=-2)  # [batch_size, seq_len, seq_len]
    table = torch.cumsum(table, dim=-2)  # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table = torch.cumsum(table.flip(-1), dim=-1).flip(-1)  # [batch_size, seq_len, seq_len]
    table_mask = (table != 0).int()  # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table *= table_mask  # [batch_size, seq_len, seq_len]
    historical_multiplier_1 = torch.exp(-b_1 * reversed_cumsum_of_table)  # [batch_size, seq_len, seq_len]
    historical_multiplier_1 *= table_mask  # [batch_size, seq_len, seq_len]
    historical_multiplier_2 = torch.exp(-b_2 * reversed_cumsum_of_table)  # [batch_size, seq_len, seq_len]
    historical_multiplier_2 *= table_mask  # [batch_size, seq_len, seq_len]

    historical_integral_1 = basic_exponential_integral_1.unsqueeze(-1) * historical_multiplier_1.unsqueeze(-2)
    # [batch_size, seq_len, resolution, seq_len]
    historical_integral_2 = basic_exponential_integral_2.unsqueeze(-1) * historical_multiplier_2.unsqueeze(-2)
    # [batch_size, seq_len, resolution, seq_len]
    historical_integral = torch.sum(historical_integral_1, dim=-1) + torch.sum(historical_integral_2, dim=-1)
    # [batch_size, seq_len, resolution]

    # Get the integral
    expand_true_integral = mu_integral + basic_exponential_integral + historical_integral
    # [batch_size, seq_len, resolution]
    expand_true_integral[:, 0, :] = mu_integral[:, 0, :]  # [batch_size, seq_len, resolution]

    return expand_true_integral


"""
Time-independent poisson process
"""


def poisson(time, intensity, opt, device):
    """
    Poisson process: \\lambda(t) = lam
    The intensity function of poisson process is a constant.
    The dataset must provide the value of lam in their dataset_card.yml.

    Args:
    time       : [batch_size, seq_len]  (not used in this function)
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    lam = itemgetter("lam")(opt.info_dict)
    resolution = opt.resolution

    batch_size, seq_len = intensity.shape
    return torch.ones((batch_size, seq_len, resolution), device=device) * lam
    # [batch_size, seq_len, resolution]


def poisson_integral(time, intensity, opt, device):
    """
    Poisson process: \\lambda(t) = lam and \\Lambda(t) = lam * t (\\Lambda(t) is the integral of \\lambda(t))
    The dataset must provide the value of lam in their dataset_card.yml.

    Args:
    time       : [batch_size, seq_len]  (not used in this function)
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    lam = itemgetter("lam")(opt.info_dict)
    resolution = opt.resolution

    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]

    return expand_time * lam  # [batch_size, seq_len, resolution]


"""
Stationary renewal process, whose probability distribution instead of intensity function is defined.
"""


def stationary_renewal(time, intensity, opt, device):
    """
    The stationary renewal process: \\lambda(t) = -0.797885*exp(-0.5*(log(t))**2) / (-t + t * erf(0.707107 * log(t)))
    The intensity function only matches the explicitly given lognorm distribution used in the synthetic data generator.
    Please check and modify this function if you want to use another hyperparameter set for the stationary renewal process during data generation.

    Timestamp 0 will be shifted to a very small value.

    Args:
    time       : [batch_size, seq_len]
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    resolution = opt.resolution

    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    expand_time[:, :, 0] += 1e-8
    return (
        -0.797885
        * torch.exp(-0.5 * (torch.log(expand_time)) ** 2)
        / (-expand_time + expand_time * torch.erf(0.707107 * torch.log(expand_time)))
    )
    # [batch_size, seq_len, resolution]


def stationary_renewal_probability(time, intensity, opt, device):
    """
    We won't implement the integral of stationary renewal's intensity function.
    We will directly use the distribution, instead.

    Args:
    time       : [batch_size, seq_len]
                 The original timestamp sequence.
    intensity  : [batch_size, seq_len]
                 The value of ture intensity function at all event points.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameter
    # stationary renewal doesn't support custom hyperparamters, otherwise calculating its intensity function would be impossible.
    resolution = opt.resolution
    s = np.sqrt(1)
    mu = 0

    from scipy.stats import lognorm

    time_multiplier = torch.linspace(0, 1, resolution, device=device)  # [resolution]
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    distribution_values = lognorm.pdf(move_from_tensor_to_ndarray(expand_time), s=s, scale=np.exp(mu))
    # [batch_size, seq_len, resolution]
    return torch.from_numpy(distribution_values)  # [batch_size, seq_len, resolution]


"""
Self-correct process, which the latest events would drastically decrease the probability of next events in a small time period.
"""


def self_correct(time, intensity, opt, device):
    """
    Self correct process has a iterative intensity function. \\lambda(t) = exp(mu * tau - alpha * N)
    N is the number of happened events.
    The dataset must provide the value of \\mu and \\alpha in their dataset_card.yml.

    Args:
    time       : [batch_size, seq_len]
    intensity  : [batch_size, seq_len]
               The value of true intensity function when a event happens.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, alpha = itemgetter("mu", "alpha")(opt.info_dict)
    resolution = opt.resolution

    batch_size = time.shape[0]
    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    shift_intensity = torch.cat((torch.ones(batch_size, 1, device=device), intensity[:, :-1]), dim=-1)
    # [batch_size, seq_len]
    start_intensity = shift_intensity / torch.exp(torch.tensor(alpha))  # [batch_size, seq_len]
    expand_time = time_multiplier * time.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    start_intensity = start_intensity.unsqueeze(-1).repeat(1, 1, resolution)  # [batch_size, seq_len, resolution]
    return start_intensity * torch.exp(mu * expand_time)  # [batch_size, seq_len, resolution]


def self_correct_integral(time, intensity, opt, device):
    """
    self correct process has intensity function: \\lambda(t) = exp(mu * tau - alpha * N)
    N is the number of happened events. Omi et al. claim self correct process doesn't aggregate intensity functions of all
    historical events, but it does just like the Hawkes process.
    The dataset must provide the value of \\mu and \\alpha in their dataset_card.yml.

    Args:
    time       : [batch_size, seq_len + 1]
    intensity  : [batch_size, seq_len]
               The value of true intensity function when a event happens.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    """
    # hyperparameters
    mu, alpha = itemgetter("mu", "alpha")(opt.info_dict)
    resolution = opt.resolution

    batch_size, _ = time.shape
    time_interval = time  # [batch_size, seq_len]
    time_history = torch.cat((torch.zeros((batch_size, 1), device=device), time[:, :-1]), dim=-1)
    # [batch_size, seq_len]

    batch_size, seq_len = time_interval.shape
    N = (
        torch.arange(0, seq_len, device=device)
        .repeat(batch_size, 1)
        .repeat_interleave(resolution, dim=-1)
        .reshape(batch_size, seq_len, -1)
    )  # [batch_size, seq_len, resolution]
    time_multiplier = torch.linspace(0, 1, resolution, device=device)
    expand_time = time_multiplier * time_interval.unsqueeze(-1)  # [batch_size, seq_len, resolution]
    historical_part = torch.exp(mu * time_history)  # [batch_size, seq_len]
    historical_part = torch.cumprod(historical_part, dim=-1)  # [batch_size, seq_len]
    historical_part = historical_part.repeat_interleave(resolution, dim=-1)  # [batch_size, seq_len * resolution]
    historical_part = historical_part.reshape(batch_size, seq_len, resolution)  # [batch_size, seq_len, resolution]
    return (torch.exp(mu * expand_time - alpha * N) - torch.exp(-alpha * N)) / mu * historical_part
    # [batch_size, seq_len, resolution]


true_intensity_dict = {
    "hawkes_1": hawkes_1,
    "hawkes_2": hawkes_2,
    "poisson": poisson,
    "stationary_renewal": stationary_renewal,
    "self_correct": self_correct,
}

true_probability_dict = {
    "hawkes_1": [hawkes_1, hawkes_1_integral],
    "hawkes_2": [hawkes_2, hawkes_2_integral],
    "poisson": [poisson, poisson_integral],
    "stationary_renewal": [stationary_renewal_probability],
    "self_correct": [self_correct, self_correct_integral],
}
