from torch import nn


class ScaledTanh(nn.Module):
    """
    An extension to tanh.
    Original tanh will limit the input into [-1, 1].
    This tanh can limit the input into [-parameter, parameter].
    """

    def __init__(self, parameter=1, device=None):
        super().__init__()
        self.device = device
        self.parameter = parameter

    def forward(self, x):
        return self.parameter * nn.functional.tanh(x)
