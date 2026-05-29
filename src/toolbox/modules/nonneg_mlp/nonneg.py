import torch
import torch.nn.functional as F
from torch import nn

# From Babylon's neuralTPPs: https://github.com/babylonhealth/neuralTPPs


class NonNegLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True, eps=0.0):
        super().__init__(in_features, out_features, bias, device=device)
        self.eps = eps
        self.device = device
        self.positivify_weights()

    def positivify_weights_old(self):
        mask = (self.weight < 0) * (-1)
        mask = mask + (self.weight >= 0)
        self.weight.data = self.weight.data * mask

    def positivify_weights(self):
        self.weight.data = F.relu(self.weight.data)

    def forward(self, inputs):
        self.positivify_weights()
        return F.linear(inputs, self.weight, self.bias)
