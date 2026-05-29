import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class Patchifier(nn.Module):
    def __init__(self, d_input, patch_length, device):
        super(Patchifier, self).__init__()
        self.device = device
        self.d_input = d_input
        self.patch_length = patch_length
        
        self.patchifier = nn.Conv1d(self.d_input, self.d_input, kernel_size = patch_length, 
                                    stride = patch_length, bias = False, device = self.device)
    
    def forward(self, input):
        # size of input: [..., seq_len, d_input]
        output = input                                                         # [..., seq_len, d_input]
        input = rearrange(input, '... sl di -> ... di sl')                     # [..., d_input, seq_len]
        seq_len = input.shape[-1]
        initial_pad_length = self.patch_length - (seq_len % self.patch_length)
        input = F.pad(input, (initial_pad_length, 0, 0, 0, 0, 0))              # [..., d_input, (seq_len // patch_length + 1) * patch_length]
        
        patch_embeddings = []
        for i in range(self.patch_length - 1, -1, -1):
            patch_embeddings.append(self.patchifier(F.pad(input, (i, 0, 0, 0, 0, 0))))
                                                                               # [..., (seq_len // patch_length) + 1, d_input] * patch_length
        patch_embeddings = torch.concat(patch_embeddings, dim = -2)            # [..., d_input * patch_length, (seq_len // patch_length) + 1, d_input]
        patch_embeddings = rearrange(patch_embeddings.transpose(-1, -2), '... np (pl di) -> ... (np pl) di', pl = self.patch_length)[:, initial_pad_length:]
                                                                               # [..., seq_len, d_input]
        output = output + patch_embeddings
        
        return output