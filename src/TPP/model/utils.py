import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import pack, rearrange, repeat
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.toolbox.metrics import evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    break_batched_inputs_into_seqs,
    check_should_we_stop_sampling,
    move_from_tensor_to_ndarray,
)
from src.TPP.resources import expand_true_probability

default_figure_kwargs = {"font.size": 18, "figure.figsize": (8, 4)}


"""
This function returns a list consisting of the step size of each operation.
For example:
(total_rate: 40, step_size: 15) -> [15, 15, 10]
"""


def step_split(total_rate, step_size):
    substep_rate_list = []
    while total_rate > 0:
        substep_rate_list.append(step_size)
        total_rate -= step_size
    substep_rate_list[-1] += total_rate

    return substep_rate_list


"""
Thinning algorithm.
"""


def thinning_sampling(
    maximum_thinning_loops,
    max_sample_time_limit,
    sample_output_shape,
    device,
    intensity_func,
    find_maximum_intensity_values_in_one_interval,
    *args,
    **kwargs,
):
    sample_rate, batch_size, seq_len = sample_output_shape
    thinning_unit_interval_length = max_sample_time_limit / maximum_thinning_loops

    predicted_time = torch.zeros(sample_rate, batch_size, seq_len, dtype=torch.int32, device=device)
    # [sample_rate, batch_size, seq_len]
    # The initial mask tensor contains only zero.
    # Zero means we have got a valid time sample.
    # One means we need a resample
    rejected_mask = torch.ones(sample_rate, batch_size, seq_len, dtype=torch.int32, device=device)
    # [sample_rate, batch_size, seq_len]
    thinning_loops = 0
    while rejected_mask.sum() > 0:
        thinning_loops += 1
        if thinning_loops > maximum_thinning_loops:
            break

        sampling_interval_left_side = (
            torch.ones_like(rejected_mask) * thinning_unit_interval_length * (thinning_loops - 1)
        )
        # [sample_rate, batch_size, seq_len]
        sampling_interval_right_side = torch.ones_like(rejected_mask) * thinning_unit_interval_length * thinning_loops
        # [sample_rate, batch_size, seq_len]
        intensity_values_for_thinning_upper_bound = (
            find_maximum_intensity_values_in_one_interval(
                sampling_interval_left_side, sampling_interval_right_side, *args, **kwargs
            )
            * 1.05
        )
        # [sample_rate, batch_size, seq_len]
        # Exponential distribution: F(x) = 1 - exp(-\\lambda x) => x = ln(1 - F(x)) / (-\\lambda)
        probability_threshold_for_exp = torch.zeros_like(intensity_values_for_thinning_upper_bound)
        # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_exp)  # [sample_rate, batch_size, seq_len]
        probability_threshold_for_thinning = torch.zeros_like(intensity_values_for_thinning_upper_bound)
        # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_thinning)  # [sample_rate, batch_size, seq_len]
        sampled_time = -torch.log(1 - probability_threshold_for_exp) / (
            intensity_values_for_thinning_upper_bound + 1e-20
        )
        # [sample_rate, batch_size, seq_len]
        # Part 1: exclude time exceeding the limit.
        sampled_time_exceeding_limit = sampled_time > thinning_unit_interval_length
        # [sample_rate, batch_size, seq_len]
        # Part 2: exclude time rejected by the learned MTPP.
        intensity_values_at_sampled_time = intensity_func(sampled_time, *args, **kwargs)
        # [sample_rate, batch_size, seq_len]
        sampled_time_rejected = (
            probability_threshold_for_thinning
            > intensity_values_at_sampled_time / intensity_values_for_thinning_upper_bound
        )
        # [sample_rate, batch_size, seq_len]
        rejected_in_this_loop = sampled_time_rejected | sampled_time_exceeding_limit
        # [sample_rate, batch_size, seq_len]
        accept_mask = rejected_mask & (~rejected_in_this_loop)  # [sample_rate, batch_size, seq_len]
        rejected_mask = rejected_mask & rejected_in_this_loop  # [sample_rate, batch_size, seq_len]
        predicted_time = predicted_time + accept_mask * sampled_time + rejected_mask * thinning_unit_interval_length
        # [sample_rate, batch_size, seq_len]
    return predicted_time


"""
resolution_inf and resolution_between_marks.
"""


def decide_resolution_inf_and_resolution_between_events(time, memory_ceiling, num_marks, mean, std):
    # Suggested batch_size: 1

    max_ = time.mean() + 10 * time.std() if mean == 0 and std == 1 else mean + 10 * std

    if mean == 0:
        resolution_between_marks = max(min(int(time.mean().item() // 0.005), 500), 10)
    else:
        resolution_between_marks = max(min(int(mean // 0.005), 500), 10)

    max_ = min(1e6, max_)
    resolution_inf = max(int(max_ // 0.005), 100)

    batch_size, seq_len = time.shape
    if batch_size * seq_len * resolution_inf * num_marks > memory_ceiling:
        resolution_inf = int(memory_ceiling // (seq_len * num_marks * batch_size))

    if batch_size * seq_len * resolution_between_marks * num_marks * num_marks > memory_ceiling:
        resolution_between_marks = int(memory_ceiling // (seq_len * num_marks * num_marks * batch_size))

    return max_, resolution_inf, resolution_between_marks


"""
custom metrics
"""


def get_f1_and_top_k_acc_in_mae_e(marks_true, p_m, input_mask, num_marks):
    f1 = []
    top_k_acc = []
    for marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq in zip(marks_true, p_m, input_mask):
        marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq = move_from_tensor_to_ndarray(
            marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq
        )
        y_pred = np.argmax(probability_integral_per_seq, axis=-1)

        selected_marks_true_per_seq = marks_true_per_seq[input_mask_per_seq == 1]
        selected_y_pred = y_pred[input_mask_per_seq == 1]
        selected_probability_integral_per_seq = probability_integral_per_seq[input_mask_per_seq == 1]

        f1.append(f1_score(y_true=selected_marks_true_per_seq, y_pred=selected_y_pred, average="macro"))
        top_k_acc_single_mark_seq = []
        if num_marks > 2:
            for k in range(1, num_marks):
                top_k_acc_single_mark_seq.append(
                    top_k_accuracy_score(
                        y_true=selected_marks_true_per_seq,
                        y_score=selected_probability_integral_per_seq,
                        k=k,
                        labels=np.arange(num_marks),
                    )
                )
        else:
            top_k_acc_single_mark_seq.append(accuracy_score(y_true=selected_marks_true_per_seq, y_pred=selected_y_pred))
        top_k_acc.append(top_k_acc_single_mark_seq)

    return f1, top_k_acc


"""
Plotting.
"""


def draw_intensity_integral_and_probability(
    df, df_mark, annotation, figure_type, color_palette, num_marks, figure_kwargs={}
):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)
    no_ground_truth = len(df.columns) == 2

    df_plot = pd.melt(df, "Time")
    df_plot.columns = ["Time", " ", figure_type]

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.lineplot(x="Time", y=figure_type, hue=" ", data=df_plot, ax=ax)

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(handles=handles, labels=labels, loc="lower left")
        ax.add_artist(lineplot_legend)

        sns.scatterplot(
            x="Time",
            y="Point",
            data=df_mark,
            palette=color_palette,
            hue="Mark",
            hue_order=[f"Mark {item}" for item in range(num_marks)],
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(
            handles=handles[1 if no_ground_truth else 2 :], labels=labels[1 if no_ground_truth else 2 :]
        )
        lineplot_legend.set_title("Mark")
        ax.add_artist(lineplot_legend)

        if annotation is not None:
            props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5}
            ax.text(0.05, 0.95, annotation, transform=ax.transAxes, fontsize=14, verticalalignment="top", bbox=props)

    return fig


def legend_format(num_marks):
    import math

    format_parameter = {"ncol": 1, "fontsize": 18}

    if num_marks > 10:
        format_parameter["ncol"] = 2

    num_marks_per_column = math.ceil(num_marks / format_parameter["ncol"])
    format_parameter["fontsize"] = format_parameter["fontsize"] * (-0.1 * max(num_marks_per_column - 5, 0) + 1)

    return format_parameter


def draw_intensity_integral_per_mark(df, df_mark, figure_type, color_palette, num_marks, figure_kwargs={}):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()

        sns.lineplot(
            x="Time",
            y=figure_type,
            hue="Mark",
            data=df,
            palette=color_palette,
            hue_order=[f"Mark {item}" for item in range(num_marks)],
            ax=ax,
        )

        sns.scatterplot(
            x="Time",
            y="Point",
            data=df_mark,
            palette=color_palette,
            hue="Mark",
            hue_order=[f"Mark {item}" for item in range(num_marks)],
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(
            handles=[(handles[idx], handles[idx + num_marks]) for idx in range(num_marks)],
            labels=labels[:num_marks],
            **legend_format(num_marks),
            handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
        )
        lineplot_legend.set_title("Mark")

    return fig


def draw_heatmap(df_matrix, index_name, column_name, value_name, figure_kwargs):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)
    index, column = df_matrix.shape

    # The index and column list
    index_list = [ele for ele in range(index) for _ in range(column)]
    column_list = list(range(column)) * index

    df = pd.DataFrame.from_dict({index_name: index_list, column_name: column_list, value_name: df_matrix.flatten()})
    df = df.pivot(index=index_name, columns=column_name, values=value_name)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.heatmap(data=df, cmap="YlGnBu", vmin=0, vmax=max(1, np.max(df_matrix)), annot=False, ax=ax)

    return fig


def draw_lineplot(*args, figure_kwargs={}, **kwargs):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.lineplot(*args, **kwargs, ax=ax)

    return fig


"""
For EHD.
"""


def pick_log_probability(log_probability, last_index, seq_len_x):
    device = last_index.device
    batch_size = last_index.shape[0]

    start_idx = torch.clamp(last_index - 1 - seq_len_x, min=0)
    # [batch_size]
    index_indices = torch.arange(seq_len_x, device=device)  # [seq_len_x]
    index_indices = repeat(index_indices, "... -> b ...", b=batch_size) + start_idx.unsqueeze(dim=-1)
    # [batch_size, seq_len_x]
    return log_probability.gather(-1, index_indices)  # [batch_size, seq_len_x]


"""
Mixins
"""
class SeqGenTimeMarkMixin:
    def sample_time_mark(self, time_history_for_sampling, marks_history_for_sampling, mean, std, end_sampling_requirement="time", **kwargs):
        """
        This function will sample x sequences by the learned probability distribution following the time-mark prediction procedure.
        Steps:
        1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
        2. Judge the mark of this mark by comparing \\(\\lambda^*(m, t_s)\\).
        """
        if time_history_for_sampling is None and marks_history_for_sampling is None:
            number_of_sampled_sequences = kwargs["number_of_sampled_sequences"]
            time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device=self.device)
            # [number_of_sampled_sequences, 1]
            marks_history_for_sampling = (
                torch.ones((number_of_sampled_sequences, 1), device=self.device, dtype=torch.int32) * self.num_marks
            )
            # [number_of_sampled_sequences, 1]
        else:
            if not (time_history_for_sampling is not None and marks_history_for_sampling is not None):
                raise ValueError("How is it possible that one input history is not None while another one is?")

            if not marks_history_for_sampling.shape[0] == time_history_for_sampling.shape[0]:
                raise ValueError(
                    f"time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while marks_history_for_sampling suggests {marks_history_for_sampling.shape[0]}. So, how many sequences should we sample?"
                )
            number_of_sampled_sequences = marks_history_for_sampling.shape[0]

        sampled_mask = None

        while True:
            should_we_stop, sampled_mask = check_should_we_stop_sampling(
                time_history_for_sampling, end_sampling_requirement, **kwargs
            )

            if should_we_stop:
                break

            sampled_time, sampled_marks = self.next_one_event_prediction_time_mark(
                time_history_for_sampling,
                marks_history_for_sampling,
                number_of_sampled_sequences,
                mean,
                std,
            )

            time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], "nss *")
            # [number_of_sampled_sequences, history_length + 1]
            marks_history_for_sampling, _ = pack([marks_history_for_sampling, sampled_marks], "nss *")
            # [number_of_sampled_sequences, history_length + 1]

        return time_history_for_sampling, marks_history_for_sampling, sampled_mask


class SeqGenMarkTimeMixin:
    def sample_mark_time(self, time_history_for_sampling, marks_history_for_sampling, mean, std, end_sampling_requirement="time", **kwargs):
        """
        This function will sample x sequences by the learned probability distribution following the time-mark prediction procedure.
        Steps:
        1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
        2. Judge the mark of this mark by comparing \\(\\lambda^*(m, t_s)\\).
        """
        if time_history_for_sampling is None and marks_history_for_sampling is None:
            number_of_sampled_sequences = kwargs["number_of_sampled_sequences"]
            time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device=self.device)
            # [number_of_sampled_sequences, 1]
            marks_history_for_sampling = (
                torch.ones((number_of_sampled_sequences, 1), device=self.device, dtype=torch.int32) * self.num_marks
            )
            # [number_of_sampled_sequences, 1]
        else:
            if not (time_history_for_sampling is not None and marks_history_for_sampling is not None):
                raise ValueError("How is it possible that one input history is not None while another one is?")

            if not marks_history_for_sampling.shape[0] == time_history_for_sampling.shape[0]:
                raise ValueError(
                    f"time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while marks_history_for_sampling suggests {marks_history_for_sampling.shape[0]}. So, how many sequences should we sample?"
                )
            number_of_sampled_sequences = marks_history_for_sampling.shape[0]

        sampled_mask = None

        while True:
            should_we_stop, sampled_mask = check_should_we_stop_sampling(
                time_history_for_sampling, end_sampling_requirement, **kwargs
            )

            if should_we_stop:
                break

            sampled_time, sampled_marks = self.next_one_event_prediction_mark_time(
                time_history_for_sampling,
                marks_history_for_sampling,
                number_of_sampled_sequences,
                mean,
                std,
            )

            time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], "nss *")
            # [number_of_sampled_sequences, history_length + 1]
            marks_history_for_sampling, _ = pack([marks_history_for_sampling, sampled_marks], "nss *")
            # [number_of_sampled_sequences, history_length + 1]

        return time_history_for_sampling, marks_history_for_sampling, sampled_mask


class SpearmanL1EvaluationMixin:
    # Evaluation over the entire dataset.
    def get_spearman_and_l1(self, input_data, opt):
        """
        Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```float``` spearman
              The spearman coefficient between the predicted and real distribution.
            * ```float``` l1
              The l1 distance between the predicted and real distribution.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_probability, timestamp = self.probability_time_next_2d(
            time_history=time_history,
            time_next=time_next,
            marks_history=marks_history,
            mask_history=mask_history,
            integration_sample_rate=opt.resolution,
            mean=mean,
            std=std
        )  # [batch_size, seq_len, resolution, num_marks] + [batch_size, seq_len, resolution]
        expand_probability = expand_probability.sum(dim=-1)  # [batch_size, seq_len, resolution]
        true_probability = expand_true_probability(time_next, input_intensity, opt)
        # [batch_size, seq_len, resolution] or batch_size * None

        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(
            expand_probability, true_probability, timestamp
        )
        spearman = evaluate_on_one_batch(expand_probability, true_probability, mask_next, "spearman", -2, -2, -1)
        l1 = evaluate_on_one_batch(
            expand_probability,
            true_probability,
            mask_next,
            "l1",
            -2,
            -2,
            -1,
            additional_inputs=[
                timestamp,
            ],
        )

        return spearman.tolist(), l1.tolist()


class NextEventPredictionTimeMarkMixin:
    def get_mae_and_f1(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` mae
              shape: ```[batch_size, seq_len]```
              The MAE value, which is the time gap between each predicted and real event.
            * ```float``` f1_1
              The f1 value shows the accuracy of the predicted marks.
            * ```torch.tensor``` dist
              shape: ```[batch_size, seq_len, num_marks]```
              The mark distribution at when the real event happens.
            * ```np.ndarray``` marks_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed marks.
        """
        argument_check(opt, **{"sample_rate": int, "mae_step": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history=time_history,
            time_next=time_next,
            marks_history=marks_history,
            mask_history=mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=opt.mae_step,
        )
        # [batch_size, seq_len] + [batch_size, seq_len, num_marks]
        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae = mae.sum(dim=-1) / mask_next.sum(dim=-1)
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, marks_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        mae, marks_next, mark_dist, acc, macro_f1, micro_f1, mask_next = move_from_tensor_to_ndarray(
            mae, marks_next, mark_dist, acc, macro_f1, micro_f1, mask_next
        )
        pred_time, marks_next, mark_dist = break_batched_inputs_into_seqs(mask_next, pred_time, marks_next, mark_dist)
        return mae.tolist(), acc.tolist(), macro_f1.tolist(), micro_f1.tolist(), pred_time, mark_dist, marks_next


class NextEventPredictionMarkTimeMixin:
    def get_mae_e_and_f1(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_marks]```
              The value of calculated p(m).
            * ```np.ndarray``` marks_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed marks.
        """
        argument_check(opt, **{"sample_rate": int, "mae_e_step": int})

        input_time, input_marks, _, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        pred_time_all_marks, mark_dist = self.next_event_prediction_mark_time(
            time_history=time_history,
            marks_history=marks_history,
            marks_next=marks_next,
            mask_history=mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_marks] + [batch_size, seq_len, num_marks]
        marks_next_mask = torch.nn.functional.one_hot(marks_next, num_classes=self.num_marks)
        # [batch_size, seq_len, num_marks]
        pred_time = (pred_time_all_marks * marks_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_marks]
        mae_e = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae_e = mae_e.sum(dim=-1) / mask_next.sum(dim=-1)
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, marks_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        (
            mae_e,
            marks_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        ) = move_from_tensor_to_ndarray(
            mae_e,
            marks_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        )
        mark_dist, pred_time_all_marks, time_next, marks_next = break_batched_inputs_into_seqs(
            mask_next, mark_dist, pred_time_all_marks, time_next, marks_next
        )

        return (
            mae_e.tolist(),
            acc.tolist(),
            macro_f1.tolist(),
            micro_f1.tolist(),
            mark_dist,
            pred_time_all_marks,
            time_next,
            marks_next,
        )


class GetWhichEventFirstMixin:
    def get_which_event_first(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        Instead of picking the most probable event, we pick the event predicted to happen first.

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` which_event_first_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE values when we pick predicted times using real marks.
            * ```float``` f1
              The f1 value shows the accuracy of the predicted marks.
        """
        argument_check(opt, **{"sample_rate": int, "sample_substep": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        pred_time_all_marks, _ = self.next_event_prediction_mark_time(
            time_history=time_history,
            marks_history=marks_history,
            marks_next=marks_next,
            mask_history=mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_marks]

        predicted_time, predicted_mark = pred_time_all_marks.min(dim=-1)  # [batch_size, seq_len] + [batch_size, seq_len]
        maes = torch.abs(time_next - predicted_time) * mask_next  # [batch_size, seq_len]
        maes = maes.sum(dim=-1) / mask_next.sum(dim=-1)  # [batch_size]

        results = evaluate_on_one_batch(predicted_mark, marks_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]
        maes = move_from_tensor_to_ndarray(maes)

        return maes.tolist(), acc.tolist(), macro_f1.tolist(), micro_f1.tolist()


class BalancedSamplingFromDistributionMixin:
    def balanced_sampling_from_distribution(self, input_data, opt):
        """This function samples from the distribution p(m, t) by sampling the mark first from p(m) then time from p(t|m).
          All samples can later be used to draw the distribution plot.

          You should declare the following arguments in your config file:
          1. ```int``` sample_rate: how many time samples from the time distribution are needed.
          2. ```int``` sample_substep: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.

        Returns:
            tuple[Any]: the results: mae_e, acc, macro-f1, micro-f1,
                        distribution of mark at the true time (evaluation = True),
                        predicted time of all marks, the true time of the next mark,
                        the true mark of the next mark
        """
        argument_check(opt, **{"sample_rate": int, "sample_substep": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        sampled_time_all_marks, mark_dist = self.next_event_prediction_mark_time(
            time_history=time_history,
            marks_history=marks_history,
            marks_next=marks_next,
            mask_history=mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            get_time_sample=True,
        )
        # [sample_rate, batch_size, seq_len, num_marks], [batch_size, seq_len, num_marks]

        mark_dist, sampled_time_all_marks, mask_next = move_from_tensor_to_ndarray(
            mark_dist, sampled_time_all_marks, mask_next
        )

        probability_integral_to_inf, tau_pred_all_mark = break_batched_inputs_into_seqs(
            mask_next, mark_dist, rearrange(sampled_time_all_marks, "sr b sl ne -> b sl ne sr")
        )  # batch_size * [seq_len, num_marks] + batch_size * [seq_len, num_marks, sample_rate]

        tau_pred_all_mark = [rearrange(item, "sl ne sr -> sr sl ne") for item in tau_pred_all_mark]

        return tau_pred_all_mark, probability_integral_to_inf
