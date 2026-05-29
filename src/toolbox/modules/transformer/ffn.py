import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    '''
    Feedforward module next to the Transformers layer.
    '''
    def __init__(self, d_input, d_hidden, device, dropout = 0.1):
        super().__init__()
        self.device = device

        self.w_1 = nn.Linear(d_input, d_hidden, device = self.device)
        self.w_2 = nn.Linear(d_hidden, d_input, device = self.device)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.RMSNorm(d_input, eps = 1e-6, device = self.device)

    def forward(self, x):
        '''
        Args:
        1. x: input tensor. shape: [..., d_input]
        Outputs:
        1. output: result tensor. shape: [..., d_input]
        '''
        residual = x

        x = self.norm(x)                                                       # [..., d_input]
        x = self.dropout(F.gelu(self.w_1(x)))                                  # [..., d_hidden]
        x = self.dropout(self.w_2(x))                                          # [..., d_input]
        x += residual
        return self.norm(x)                                                    # [..., d_input]
