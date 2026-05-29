import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class SelfAttn(nn.Module):
    '''
    SelfAttn module, the heart of transformers' layer
    '''
    def __init__(self, temperature, attn_dropout, device):
        super().__init__()
        self.device = device
        self.temperature = temperature

        self.dropout = nn.Dropout(attn_dropout)


    def forward(self, q, k, v, mask = None):
        '''
        Args:

        1. q: input tensor. shape: [batch_size, seq_len_q, n_head, d_qk]
        2. k: input tensor. shape: [batch_size, seq_len_k, n_head, d_qk]
        3. v: input tensor. shape: [batch_size, seq_len_k, n_head, d_v]
        4. mask: mask_out several values in the attention matrices. shape: [batch_size, seq_len_q, seq_len_k]

        Output:
        1. output: the result of self attention. shape: [batch_size, seq_len, n_head, d_v]
        '''

        q /= self.temperature                                                  # [batch_size, seq_len_q, n_head, d_qk]

        attn = einsum(q, k, '... slq nh dqk, ... slk nh dqk -> ... nh slq slk')# [batch_size, n_head, seq_len_q, seq_len_k]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-3) == 0, -1e9)             # [batch_size, n_head, seq_len_q, seq_len_k]

        # F.softmax() uses float32 by default.
        # This means it will upcast the input to float32 and the output is also float32.
        # We send it the dtype of attn to force it to respect the precision cast.
        # We also have reports saying low precision softmax is GPU only.
        # Please check https://github.com/huggingface/transformers/issues/27341 for further information.
        attn = self.dropout(F.softmax(attn, dim = -1, dtype = attn.dtype))     # [batch_size, n_head, seq_len_q, seq_len_k]
        out = einsum(attn, v, '... nh slq slk, ... slk nh dv -> ... slq nh dv')# [batch_size, seq_len_q, n_head, d_v]

        return out, attn
