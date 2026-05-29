import torch
import torch.nn as nn
from einops import rearrange

from src.toolbox.modules.transformer.ffn import FFN
from src.toolbox.modules.transformer.selfattn import SelfAttn


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, dropout = 0.1):
        super().__init__()
        self.device = device

        self.attn = MultiheadAttention(n_head = n_head, d_input = d_input, d_qk = d_qk,
                                       d_v = d_v, device = self.device, dropout = dropout)
        self.ffn = FFN(d_input = d_input, d_hidden = d_hidden, device = self.device, dropout = dropout)


    def forward(self, q, k = None, v = None, self_attn_mask = None, non_pad_mask = None):
        '''
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [batch_size, seq_len, d_attn_input]
        Outputs:
        '''
        if k is None and v is None:
            output, attn = self.attn(q, q, q, mask = self_attn_mask)           # [batch_size, seq_len, d_input] & [batch_size, n_head, seq_len, seq_len]
        else:
            output, attn = self.attn(q, k, v, mask = self_attn_mask)           # [batch_size, seq_len, d_input] & [batch_size, n_head, seq_len, seq_len]

        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, '... -> ... 1')                  # [batch_size, seq_len, d_input]

        output = self.ffn(output)                                              # [batch_size, seq_len, d_input]
        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, '... -> ... 1')                  # [batch_size, seq_len, d_input]

        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, dropout = 0.1):
        '''
        Template self-attention module with multihead-attention type 2: this module concatenates original outputs and
        compress high-dimensional vectors into d_input
        '''
        super().__init__()
        self.device = device

        self.d_input = d_input
        self.n_head = n_head
        self.d_q = d_qk
        self.d_k = d_qk
        self.d_v = d_v
        self.dropout = dropout

        # Linear: d_input -> d_q, d_k, or d_v
        self.w_q = nn.Linear(d_input, self.d_q * self.n_head, bias = False, device = self.device)
        self.w_k = nn.Linear(d_input, self.d_k * self.n_head, bias = False, device = self.device)
        self.w_v = nn.Linear(d_input, self.d_v * self.n_head, bias = False, device = self.device)

        # Self-attention module
        self.self_attn = SelfAttn(temperature = d_qk ** 0.5, attn_dropout = self.dropout, device = self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_input
        self.fc_attn_output = nn.Linear(self.n_head * d_v, self.d_input, bias = True, device = self.device)

        # Dropout
        self.dropout = nn.Dropout(self.dropout)

        # layer normalization
        self.layer_norm_for_q = nn.RMSNorm(self.d_input, eps = 1e-6, device = self.device, dtype = torch.get_default_dtype())
        self.layer_norm_for_output = nn.RMSNorm(self.d_input, eps = 1e-6, device = self.device, dtype = torch.get_default_dtype())


    def forward(self, q, k, v, mask = None):
        '''
        Args:
        1. q: input tensor. shape: [batch_size, seq_len, d_input]
        2. k: input tensor. shape: [batch_size, seq_len, d_input]
        3. v: input tensor. shape: [batch_size, seq_len, d_input]
        4. mask: the mask tensor used by self attention. shape: [seq_len, seq_len]
        Output:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_output]
        2. attn: self attention value. shape: [batch_size, n_head, seq_len, seq_len]
        '''

        residual = q
        q = self.layer_norm_for_q(q)                                           # [batch_size, seq_len, n_head, d_qk]

        # preparing for q, k, and v.
        q = rearrange(self.w_q(q), '... (nh dq) -> ... nh dq', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_qk]
        k = rearrange(self.w_k(k), '... (nh dk) -> ... nh dk', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_qk]
        v = rearrange(self.w_v(v), '... (nh dv) -> ... nh dv', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_v]

        output, attn = self.self_attn(q, k, v, mask = mask)                    # [batch_size, seq_len, n_head, d_v] & [batch_size, n_head, seq_len, seq_len]
        output = rearrange(output, '...  nh dv -> ... (nh dv)', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head * d_v]
        output = self.dropout(self.fc_attn_output(output))                     # [batch_size, seq_len, d_output]
        output += residual

        output = self.layer_norm_for_output(output)                            # [batch_size, seq_len, d_output]

        return output, attn
