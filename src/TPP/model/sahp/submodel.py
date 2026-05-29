import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.stats import spearmanr

from src.toolbox.algorithms import approximate_integration
from src.toolbox.metrics import L1_distance_across_marks
from src.toolbox.misc import move_from_tensor_to_ndarray
from src.TPP.model.sahp.transformers import TransformerEncoder


class SAHP(nn.Module):
    def __init__(
        self,
        device,
        num_marks,
        d_input,
        d_rnn,
        d_hidden,
        n_layers,
        n_head,
        d_qk,
        d_v,
        dropout,
        integration_sample_rate,
    ):
        """
        This function creates a SAHP model.

        ### Args
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
            * ```int``` d_rnn
              The dimension of RNN's hidden state.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        """
        super().__init__()
        self.num_marks = num_marks
        self.device = device
        self.integration_sample_rate = integration_sample_rate

        # The original paper makes people believe SAHP is an RMTPP-like model.
        # However, this model in fact decays the hidden embedding so it is akin to CTLSTM.
        # The following three layers find the \\eta_{u, i+1}, \\mu_{u, i+1}, and \\gamma_{u i+1}
        self.gelu = nn.GELU()

        self.start_layer = nn.Sequential(nn.Linear(d_input, d_input, bias=True, device=self.device), self.gelu)

        self.converge_layer = nn.Sequential(nn.Linear(d_input, d_input, bias=True, device=self.device), self.gelu)

        self.decay_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias=True, device=self.device), nn.Softplus(beta=10.0).to(torch.bfloat16)
        )

        # This layer translates decayed hidden states into intensity function values.
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_input, self.num_marks, bias=True, device=self.device), nn.Softplus(beta=1.0).to(torch.bfloat16)
        )

        # History encoder. SAHP employs a plain transformer to encode marked temporal history
        self.history_encoder = TransformerEncoder(
            num_marks,
            device=self.device,
            d_input=d_input,
            d_rnn=d_rnn,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            dropout=dropout,
        )

    def state_decay(self, mu, eta, gamma, duration_t, num_dimension_prior_batch):
        """
        This function decays the hidden state using a Hawkes-like rule by time.

        ### Args:
          * ```torch.tensor``` mu
            shape: ```[..., batch_size, seq_len, d_hidden]```
          * ```torch.tensor``` eta
            shape: ```[..., batch_size, seq_len, d_hidden]```
          * ```torch.tensor``` gamma
            shape: ```[..., batch_size, seq_len, d_hidden]```
            mu, eta, and gamma for state decay.
          * ```torch.tensor``` duration_t
            shape: ```[batch_size, seq_len, (integration_sample_rate, num_marks)]```
            Decay by how much time?
          * ```int``` num_dimension_prior_batch
            How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
        ### Outputs
            * ```torch.tensor``` cell_t
              shape: ```[..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]```
              The decayed state at duration_t.
        """

        def get_cell_states(mu, eta, gamma, duration_t):
            return torch.tanh(
                mu + (eta - mu) * torch.exp(-gamma * duration_t)
            )  # [..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]

        if len(duration_t.shape) - 2 - num_dimension_prior_batch < 0:
            raise ValueError("Too few dimensions in duration_t!")

        # add additional dimension to mu, eta, and gamma.
        mu = rearrange(
            mu,
            f"... d_i -> {'() ' * num_dimension_prior_batch}... {'() ' * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i",
        )
        # [..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]
        eta = rearrange(
            eta,
            f"... d_i -> {'() ' * num_dimension_prior_batch}... {'() ' * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i",
        )
        # [..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]
        gamma = rearrange(
            gamma,
            f"... d_i -> {'() ' * num_dimension_prior_batch}... {'() ' * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i",
        )
        # [..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]

        duration_t = duration_t.unsqueeze(
            dim=-1
        )  # [..., batch_size, seq_len, (integration_sample_rate, num_marks), 1]
        return get_cell_states(
            mu, eta, gamma, duration_t
        )  # [..., batch_size, seq_len, (integration_sample_rate, num_marks), d_input]

    def forward(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        custom_marks_history=False,
        num_dimension_prior_batch=0,
    ):
        """
        SAHP's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding marks from the attention map.
            * ```bool``` custom_marks_history
              when true, the marks_history will be the mark embedding of historical marks.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
        ### Outputs
            * ```torch.tensor``` integral_all_marks
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_marks
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        history = self.history_encoder(time_history, marks_history, mask_history, custom_marks_history)
        # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)  # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)  # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)  # [batch_size, seq_len, d_input]

        hidden_state_at_t = self.state_decay(
            mu=mu, eta=eta, gamma=gamma, duration_t=time_next, num_dimension_prior_batch=num_dimension_prior_batch
        )
        # [..., batch_size, seq_len, d_input]
        # calculate the intensity.
        intensity_all_marks = self.intensity_layer(hidden_state_at_t)  # [..., batch_size, seq_len, num_marks]
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(
            mu=mu, eta=eta, gamma=gamma, duration_t=expanded_time, num_dimension_prior_batch=num_dimension_prior_batch
        )
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_intensity_all_marks = self.intensity_layer(expanded_hidden_state_at_t)
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]

        integral_all_marks = approximate_integration(
            expanded_intensity_all_marks, expanded_time, dim=-2, only_integral=True
        )
        # [..., batch_size, seq_len, num_marks]

        return integral_all_marks, intensity_all_marks

    def get_mark_embedding(self, input_mark):
        return self.history_encoder.get_mark_embedding(input_mark)  # [batch_size, seq_len, d_history]

    def integral_intensity_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        integration_sample_rate,
        num_dimension_prior_batch=0,
        time_next_start=None
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_mark.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding marks from the attention map.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
            * ```torch,tensor``` time_next_start
              shape: ```[..., batch_size, seq_len]``` if not None
              When given, this function computes the integral between [time_next_start, t_i]. time_next_start are expected to be non-negative.
              This affects the integral, intensity, and timestamp.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)  # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)  # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)  # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = (time_next - time_next_start).unsqueeze(dim=-1) * time_multiplier + time_next_start.unsqueeze(
            dim=-1
        )
        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(
            mu=mu, eta=eta, gamma=gamma, duration_t=expanded_time, num_dimension_prior_batch=num_dimension_prior_batch
        )
        # [..., batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_marks = self.intensity_layer(expanded_hidden_state_at_t)
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_integral_all_marks = approximate_integration(expanded_intensity_all_marks, expanded_time, dim=-2)
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]

        return expanded_integral_all_marks, expanded_intensity_all_marks, expanded_time

    def integral_intensity_time_next_3d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        integration_sample_rate,
        num_dimension_prior_batch=0,
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len, num_marks]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding marks from the attention map.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)  # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)  # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)  # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        # [integration_sample_rate]
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, num_marks, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(
            mu=mu, eta=eta, gamma=gamma, duration_t=expanded_time, num_dimension_prior_batch=num_dimension_prior_batch
        )
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, d_input]
        expanded_intensity_all_marks = self.intensity_layer(expanded_hidden_state_at_t)
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]
        expanded_integral_all_marks = approximate_integration(expanded_intensity_all_marks, expanded_time, dim=-2)
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

        return expanded_integral_all_marks, expanded_intensity_all_marks, expanded_time

    def model_probe_function(
        self, time_history, time_next, marks_history, mask_history, mask_next, integration_sample_rate
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding marks from the attention map.
            * ```torch.tensor``` mask_next
              shape: ```[..., batch_size, seq_len]```
              Tell which mark in *_next is the real mark so should be considered in metric calculation.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```dict``` data
              Probed data used for plot drawing.
            * ```torch.tensor``` expanded_time
              shape: ```[batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)  # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)  # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)  # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(
            mu=mu, eta=eta, gamma=gamma, duration_t=expanded_time, num_dimension_prior_batch=0
        )
        # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_marks = self.intensity_layer(expanded_hidden_state_at_t)
        # [batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_integral_all_marks = approximate_integration(expanded_intensity_all_marks, expanded_time, dim=-2)
        # [batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

        # construct the plot dict
        data = {}
        data["expand_intensity_for_each_mark"] = (
            expanded_intensity_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )
        data["expand_integral_for_each_mark"] = (
            expanded_integral_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )

        expand_intensity = rearrange(expanded_intensity_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]
        expand_integral = rearrange(expanded_integral_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]

        spearman_matrix = []
        pearson_matrix = []
        l1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, expanded_time_per_seq) in enumerate(
            zip(expand_intensity, expand_integral, mask_next, expanded_time)
        ):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_marks == 1:
                spearman_matrix_per_seq = np.array([[1.0]])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[: seq_len * integration_sample_rate])[0]
                if self.num_marks == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(
                probability_distribution[: seq_len * integration_sample_rate], rowvar=False
            )
            if self.num_marks == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), " -> () ()")

            # L^1 metric
            l1_matrix_per_seq = L1_distance_across_marks(
                probability_distribution[: seq_len * integration_sample_rate],
                time_next=expanded_time_per_seq[:seq_len],
                has_flatten=True,
            )
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            l1_matrix.append(l1_matrix_per_seq)

        data["spearman_matrix"] = spearman_matrix
        data["pearson_matrix"] = pearson_matrix
        data["L1_matrix"] = l1_matrix

        return data, expanded_time
