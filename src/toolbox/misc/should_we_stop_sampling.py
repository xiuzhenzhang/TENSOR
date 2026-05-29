import torch


def check_should_we_stop_sampling(tmp_time_history_for_sampling, end_sampling_requirement, **kwargs):
    should_we_stop = False

    if end_sampling_requirement == 'time':

        tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)  # [number_of_sampled_sequences]
        if tmp_sum_of_sampled_time.min() > kwargs['end_time']:
            should_we_stop = True
            sampled_mask = (tmp_time_history_for_sampling.cumsum(dim = -1) < kwargs['end_time']).int()

            return should_we_stop, sampled_mask
        return should_we_stop, None

    if end_sampling_requirement == 'event_num':
        current_seq_len = tmp_time_history_for_sampling.shape[-1]
        if current_seq_len > kwargs['max_seq_len']:
            should_we_stop = True
            sampled_mask = torch.ones_like(tmp_time_history_for_sampling, dtype = int)

            return should_we_stop, sampled_mask
        return should_we_stop, None

    if end_sampling_requirement == 'time_and_event_num':
        current_seq_len = tmp_time_history_for_sampling.shape[-1]
        tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)  # [number_of_sampled_sequences]
        if tmp_sum_of_sampled_time.min() > kwargs['end_time'] or current_seq_len > kwargs['max_seq_len']:
            should_we_stop = True
            sampled_mask = (tmp_time_history_for_sampling.cumsum(dim = -1) < kwargs['end_time']).int()

            return should_we_stop, sampled_mask
        return should_we_stop, None

    raise Exception('Unrecognized sampling termination requirement.')
