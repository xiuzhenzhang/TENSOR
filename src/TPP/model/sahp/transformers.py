import torch
import torch.nn as nn
from einops import rearrange

from src.toolbox.modules import BiasedPositionalEmbedding, TransformerLayer, get_subsequent_mask


class TransformerEncoder(nn.Module):
    def __init__(self, num_marks, device, d_input, d_rnn, d_hidden, n_layers, n_head, d_qk, d_v, dropout):
        """
        This function builds a Transformer encoder.

        ### Args
          * ```int``` num_marks
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_rnn
            The dimension of RNN's hidden state.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        """
        super().__init__()
        self.device = device
        self.num_marks = num_marks if num_marks > 0 else 1

        self.encoder = Encoder(
            num_marks=self.num_marks,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            dropout=dropout,
            device=self.device,
        )

    def forward(self, time_history, mark_history, non_pad_mask, custom_mark_history=False):
        """
        Encode the input continuous-time event stream using Transformer.

        ### Args
          * ```torch.tensor``` time_history
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent mark.
          * ```torch.tensor``` mark_history
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event.
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event.
          * ```bool``` custom_mark_history
            This argument should be true if the event_history has already been converted into embeddings.
        ### Outputs
            * ```torch.tensor``` enc_output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        """
        return self.encoder(time_history, mark_history, non_pad_mask, custom_mark_history)
        # [batch_size, seq_len, d_input]

    def get_event_embedding(self, input_event):
        """
        Convert the inputted event marks into embeddings

        ### Args
          * ```torch.tensor``` input_event
            shape: ```[batch_size, seq_len]```
            The mark of observed mark.

        ### Outputs
            * ```torch.tensor```
              shape: ```[batch_size, seq_len, d_input]```
              The representation of marks.
        """
        return self.encoder.get_event_embedding(input_event)  # [batch_size, seq_len, d_input]


class Encoder(nn.Module):
    def __init__(self, num_marks, d_input, d_hidden, n_layers, n_head, d_qk, d_v, dropout, device):
        """
        This function builds a Transformer encoder.

        ### Args
          * ```int``` num_marks
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        """
        super().__init__()
        self.device = device
        self.d_input = d_input
        self.num_marks = num_marks

        # position vector, used for temporal encoding
        # TODO(me): set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len=4096, device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_marks + 1, d_input, padding_idx=num_marks, device=self.device)

        self.layer_stack = nn.ModuleList(
            [
                TransformerLayer(
                    d_input=d_input,
                    d_hidden=d_hidden,
                    n_head=n_head,
                    d_qk=d_qk,
                    d_v=d_v,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, time_history, mark_history, non_pad_mask, custom_mark_history):
        """
        Encode the input continuous-time event stream using Transformer.

        ### Args
          * ```torch.tensor``` time_history
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent mark.
          * ```torch.tensor``` mark_history
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event.
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event.
          * ```bool``` custom_mark_history
            This argument should be true if the event_history has already been converted into embeddings.
        ### Outputs
            * ```torch.tensor``` mingled_emb
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        """
        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        seq_len = mark_history.shape[-1]
        self_attn_mask_subseq = get_subsequent_mask(seq_len, device=self.device)
        # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = rearrange(non_pad_mask, "b s -> b () s")  # [batch_size, seq_len, seq_len]
        self_attn_mask = self_attn_mask_keypad & self_attn_mask_subseq  # [batch_size, seq_len, seq_len]

        # Time Embedding
        time_emb = self.position_emb(seq_len, time_history)  # [batch_size, seq_len, d_input]

        # Event Embedding
        if mark_history is not None:
            mark_emb = mark_history if custom_mark_history else self.event_emb(mark_history)
        # [batch_size, seq_len, d_input]
        else:
            mark_emb = torch.zeros_like(time_emb, device=self.device)  # [batch_size, seq_len, d_input]
        mingled_emb = time_emb + mark_emb  # [batch_size, seq_len, d_input]

        for enc_layer in self.layer_stack:
            mingled_emb, _ = enc_layer(
                mingled_emb, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask
            )  # [batch_size, seq_len, d_input]

        return mingled_emb

    def get_event_embedding(self, input_event):
        """
        Convert the inputted event marks into embeddings

        ### Args
          * ```torch.tensor``` input_event
            shape: ```[batch_size, seq_len]```
            The mark of observed mark.

        ### Outputs
            * ```torch.tensor```
              shape: ```[batch_size, seq_len, d_input]```
              The representation of marks.
        """
        return self.event_emb(input_event)  # [batch_size, seq_len, d_input]
