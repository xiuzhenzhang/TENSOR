import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_between_two_funcs
from src.toolbox.misc import get_logger, move_from_tensor_to_ndarray, save_fig, stable_palette
from src.TPP.model.utils import (
    draw_heatmap,
    draw_intensity_integral_and_probability,
    draw_intensity_integral_per_mark,
    draw_lineplot,
)
from src.TPP.resources.syn_tpp_utils import expand_true_intensity, expand_true_probability

logger = get_logger(__name__)


def generate_intensity_figure(data, opt):
    """
    This function draws the intensity function given one sequence and store them in the "result" folder.

    ### Args
        * ```dict``` data
          All the data we need to draw the plot. The data type can vary from item to item, so be careful.
        * ```namespace``` opt
          Task arguments.
    """
    timestamp = data["timestamp"]
    num_marks = opt.info_dict["num_marks"]
    color_palette = stable_palette([f"Mark {i}" for i in range(num_marks)])

    # Part 1: the sum of intensity functions over all markers.
    expand_intensity = data["expand_intensity"]  # [batch_size, seq_len, resolution, num_marks]
    mask_next = data["mask_next"]  # [batch_size, seq_len]
    marks_next = data["marks_next"]  # [batch_size, seq_len]
    time_next = data["time_next"]  # [batch_size, seq_len]
    input_intensity = data["input_intensity"]  # [batch_size, seq_len + 1]

    expand_intensity = expand_intensity.sum(dim=-1)  # [batch_size, seq_len, resolution]
    true_intensity = expand_true_intensity(time_next, input_intensity, opt)  # [batch_size, seq_len, resolution]

    packed_data = zip(
        *move_from_tensor_to_ndarray(expand_intensity, marks_next, time_next, mask_next, timestamp, true_intensity)
    )
    for idx, (
        expand_intensity_per_seq,
        marks_next_per_seq,
        time_next_per_seq,
        mask_next_per_seq,
        timestamp_per_seq,
        true_intensity_per_seq,
    ) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()
        start_time = time_next_per_seq[:seq_len].cumsum(axis=-1)
        timestamp_offset = np.concatenate((np.array([0.0]), start_time[:-1]), axis=-1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq + np.expand_dims(timestamp_offset, axis=-1)

        df_mark = pd.DataFrame.from_dict(
            {
                "Time": start_time,
                "Point": np.zeros_like(marks_next_per_seq),
                "Mark": [f"Mark {item}" for item in marks_next_per_seq],
            }
        )

        annotation = None
        if true_intensity_per_seq is not None:
            df_intensity_plot = pd.DataFrame.from_dict(
                {
                    "Time": timestamp_per_seq.flatten(),
                    "Predicted": expand_intensity_per_seq[:seq_len, :].flatten(),
                    "Truth": true_intensity_per_seq[:seq_len, :].flatten(),
                }
            )

            # Spearman correlation
            rho = spearmanr(
                a=true_intensity_per_seq[:seq_len, :].flatten(), b=expand_intensity_per_seq[:seq_len, :].flatten()
            )[0]
            # Pearson correlation
            r = np.corrcoef(
                x=true_intensity_per_seq[:seq_len, :].flatten(), y=expand_intensity_per_seq[:seq_len, :].flatten()
            )[0, 1]
            # L1 distance
            l1 = L1_distance_between_two_funcs(
                x=true_intensity_per_seq[:seq_len, :],
                y=expand_intensity_per_seq[:seq_len, :],
                timestamp=timestamp_per_seq,
            )

            annotation = "\n".join((rf"$r = {r}$", rf"$\rho = {rho}$", rf"$L^1 = {l1}$"))
        else:
            df_intensity_plot = pd.DataFrame.from_dict(
                {"Time": timestamp_per_seq.flatten(), "Predicted": expand_intensity_per_seq[:seq_len, :].flatten()}
            )

        fig = draw_intensity_integral_and_probability(
            df_intensity_plot, df_mark, annotation, "Intensity", color_palette, num_marks
        )
        save_fig(fig, opt.plot_store_dir_for_this_batch, f"intensity_{idx}.pdf")
        logger.info(f"intensity_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    return 0


def generate_integral_figure(data, opt):
    """
    This function draws the integral of the intensity function given one sequence and store them in the "result" folder.

    ### Args
        * ```dict``` data
          All the data we need to draw the plot. The data type can vary from item to item, so be careful.
        * ```namespace``` opt
          Task arguments.
    """
    timestamp = data["timestamp"]
    num_marks = opt.info_dict["num_marks"]
    color_palette = stable_palette([f"Mark {i}" for i in range(num_marks)])

    # Part 1: the sum of intensity integrals over all markers.
    expand_integral = data["expand_integral"]  # [batch_size, seq_len, resolution]
    mask_next = data["mask_next"]  # [batch_size, seq_len]
    marks_next = data["marks_next"]  # [batch_size, seq_len]
    time_next = data["time_next"]  # [batch_size, seq_len]
    expand_integral = expand_integral.sum(dim=-1)  # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(expand_integral, marks_next, time_next, mask_next, timestamp))
    for idx, (
        expand_integral_per_seq,
        marks_next_per_seq,
        time_next_per_seq,
        mask_next_per_seq,
        timestamp_per_seq,
    ) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()
        start_time = time_next_per_seq[:seq_len].cumsum(axis=-1)
        timestamp_offset = np.concatenate((np.array([0.0]), start_time[:-1]), axis=-1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq + np.expand_dims(timestamp_offset, axis=-1)

        df_mark = pd.DataFrame.from_dict(
            {
                "Time": start_time,
                "Point": np.zeros_like(marks_next_per_seq),
                "Mark": [f"Mark {item}" for item in marks_next_per_seq],
            }
        )

        df_integral_plot = pd.DataFrame.from_dict(
            {"Time": timestamp_per_seq.flatten(), "Predicted": expand_integral_per_seq[:seq_len, :].flatten()}
        )

        fig = draw_intensity_integral_and_probability(
            df_integral_plot, df_mark, None, "Integral", color_palette, num_marks
        )
        save_fig(fig, opt.plot_store_dir_for_this_batch, f"integral_{idx}.pdf")
        logger.info(f"integral_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    return 0


def generate_probability_figure(data, opt):
    """
    This function draws the probability distribution p^*(m, t) given one sequence and store them in the "result" folder.

    ### Args
        * ```dict``` data
          All the data we need to draw the plot. The data type can vary from item to item, so be careful.
        * ```namespace``` opt
          Task arguments.
    """
    timestamp = data["timestamp"]
    num_marks = opt.info_dict["num_marks"]
    color_palette = stable_palette([f"Mark {i}" for i in range(num_marks)])

    # Part 1: the sum of probability distributions over all markers.
    expand_probability = data["expand_probability"]  # [batch_size, seq_len, resolution, num_marks]
    mask_next = data["mask_next"]  # [batch_size, seq_len]
    marks_next = data["marks_next"]  # [batch_size, seq_len]
    time_next = data["time_next"]  # [batch_size, seq_len]
    input_intensity = data["input_intensity"]  # [batch_size, seq_len + 1]

    expand_probability = expand_probability.sum(dim=-1)  # [batch_size, seq_len, resolution]
    true_probability = expand_true_probability(
        time_next, input_intensity, opt
    )  # [batch_size, seq_len, resolution] or batch_size * None

    packed_data = zip(
        *move_from_tensor_to_ndarray(expand_probability, marks_next, time_next, mask_next, timestamp, true_probability)
    )
    for idx, (
        expand_probability_per_seq,
        marks_next_per_seq,
        time_next_per_seq,
        mask_next_per_seq,
        timestamp_per_seq,
        true_probability_per_seq,
    ) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()
        start_time = time_next_per_seq[:seq_len].cumsum(axis=-1)
        timestamp_offset = np.concatenate((np.array([0.0]), start_time[:-1]), axis=-1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq + np.expand_dims(timestamp_offset, axis=-1)

        df_mark = pd.DataFrame.from_dict(
            {
                "Time": start_time,
                "Point": np.zeros_like(marks_next_per_seq),
                "Mark": [f"Mark {item}" for item in marks_next_per_seq],
            }
        )

        annotation = None
        if true_probability_per_seq is not None:
            df_probability_plot = pd.DataFrame.from_dict(
                {
                    "Time": timestamp_per_seq.flatten(),
                    "Predicted": expand_probability_per_seq[:seq_len, :].flatten(),
                    "Truth": true_probability_per_seq[:seq_len, :].flatten(),
                }
            )

            # Spearman correlation
            rho = spearmanr(
                a=true_probability_per_seq[:seq_len, :].flatten(), b=expand_probability_per_seq[:seq_len, :].flatten()
            )[0]
            # Pearson correlation
            r = np.corrcoef(
                x=true_probability_per_seq[:seq_len, :].flatten(), y=expand_probability_per_seq[:seq_len, :].flatten()
            )[0, 1]
            # L1 distance
            l1 = L1_distance_between_two_funcs(
                x=true_probability_per_seq[:seq_len, :],
                y=expand_probability_per_seq[:seq_len, :],
                timestamp=timestamp_per_seq,
            )

            annotation = "\n".join((rf"$r = {r}$", rf"$\rho = {rho}$", rf"$L^1 = {l1}$"))
        else:
            df_probability_plot = pd.DataFrame.from_dict(
                {"Time": timestamp_per_seq.flatten(), "Predicted": expand_probability_per_seq[:seq_len, :].flatten()}
            )

        fig = draw_intensity_integral_and_probability(
            df_probability_plot, df_mark, annotation, "Probability", color_palette, num_marks
        )
        save_fig(fig, opt.plot_store_dir_for_this_batch, f"probability_{idx}.pdf")
        logger.info(f"probability_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    return 0


def generate_debug_figure(data, opt):
    """
    This function draws plots for deeper insight of intensity functions and other metrics.

    ### Args
        * ```dict``` data
          All the data we need to draw the plot. The data type can vary from item to item, so be careful.
        * ```namespace``` opt
          Task arguments.
    """
    timestamp = data["timestamp"]
    num_marks = opt.info_dict["num_marks"]
    resolution = opt.resolution
    color_palette = stable_palette([f"Mark {i}" for i in range(num_marks)])

    # Part 1: expand intensity and expand integral
    # Required plots: lineplot and scatterplot
    marks_next = data["marks_next"]  # [batch_size, seq_len]
    time_next = data["time_next"]  # [batch_size, seq_len]
    mask_next = data["mask_next"]  # [batch_size, seq_len]
    expand_intensity = data[
        "expand_intensity_for_each_mark"
    ]  # [batch_size, seq_len, resolution, num_marks] if self.mark_toggle else [batch_size, seq_len, resolution, 1]
    expand_integral = data[
        "expand_integral_for_each_mark"
    ]  # [batch_size, seq_len, resolution, num_marks] if self.mark_toggle else [batch_size, seq_len, resolution, 1]
    expand_timestamp = timestamp  # [batch_size, seq_len, resolution]

    packed_data = zip(
        *move_from_tensor_to_ndarray(
            marks_next, time_next, mask_next, expand_intensity, expand_integral, expand_timestamp
        )
    )
    for idx, (
        marks_next_per_seq,
        time_next_per_seq,
        mask_next_per_seq,
        expand_intensity_per_seq,
        expand_integral_per_seq,
        timestamp_per_seq,
    ) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()
        start_time = time_next_per_seq[:seq_len].cumsum(axis=-1)
        timestamp_offset = np.concatenate((np.array([0.0]), start_time[:-1]), axis=-1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq[:seq_len] + np.expand_dims(timestamp_offset, axis=-1)

        # Figure 1 and 2: Mark-wise intensity and integral function.
        # Required plots: lineplot
        df_mark = pd.DataFrame.from_dict(
            {
                "Time": start_time,
                "Point": np.zeros_like(marks_next_per_seq[:seq_len]),
                "Mark": [f"Mark {item}" for item in marks_next_per_seq[:seq_len]],
            }
        )

        mark_list = [f"Mark {i}" for i in range(num_marks)]

        df_intensity = pd.DataFrame.from_dict(
            {
                "Time": timestamp_per_seq.flatten().repeat(num_marks),
                "Intensity": expand_intensity_per_seq[:seq_len, :, :].flatten(),
                "Mark": mark_list * (seq_len * resolution),
            }
        )
        df_integral = pd.DataFrame.from_dict(
            {
                "Time": timestamp_per_seq.flatten().repeat(num_marks),
                "Integral": expand_integral_per_seq[:seq_len, :, :].flatten(),
                "Mark": mark_list * (seq_len * resolution),
            }
        )

        fig1 = draw_intensity_integral_per_mark(df_intensity, df_mark, "Intensity", color_palette, num_marks)
        save_fig(fig1, opt.plot_store_dir_for_this_batch, f"mark_wise_intensity_{idx}.pdf")
        logger.info(f"mark_wise_intensity_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

        fig2 = draw_intensity_integral_per_mark(df_integral, df_mark, "Integral", color_palette, num_marks)
        save_fig(fig2, opt.plot_store_dir_for_this_batch, f"mark_wise_integral_{idx}.pdf")
        logger.info(f"mark_wise_integral_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    # Part 3, 4, 5: plot for spearman, pearson, and L1 distance matrix
    # Required plots: heatmap
    for value in ["spearman", "pearson", "L1"]:
        matrices = data[f"{value}_matrix"]
        for idx, matrix in enumerate(matrices):
            fig = draw_heatmap(matrix, "Mark type", "Mark type ", value, {"font.size": 18, "figure.figsize": (5, 5)})
            save_fig(fig, opt.plot_store_dir_for_this_batch, f"{value}_{idx}.pdf")
            logger.info(f"{value}_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    # Part 6: plot for Top-K accuracy
    # Required plots: lineplot
    top_k = data["top_k"]  # [batch_size, num_marks - 1]
    for idx, top_k_per_seq in enumerate(top_k):
        data_top_k_per_seq = {
            "K": np.arange(1, max(num_marks, 2)),
            "Accuracy": top_k_per_seq,
        }

        fig6 = draw_lineplot(
            data=data_top_k_per_seq, x="K", y="Accuracy", figure_kwargs={"font.size": 18, "figure.figsize": (5, 5)}
        )
        ax = fig6.gca()
        ax.set_ylim(bottom=-0.05, top=1.25)

        save_fig(fig6, opt.plot_store_dir_for_this_batch, f"topk_{idx}.pdf")
        logger.info(f"Top-K_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    # Part 7: Logarithm of MAE at each mark
    maes_ptm = data["maes_ptm"]  # [batch_size, seq_len]
    mae_pt = data["mae_pt"]  # [batch_size, seq_len]
    mask_next = data["mask_next"]  # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(mae_pt, maes_ptm, mask_next))
    for idx, (mae_pt_per_seq, maes_ptm_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_maes_per_seq = {
            "Event Index": list(range(seq_len)) * 2,
            r"$\log(1 + \mathrm{MAE})$": np.concatenate(
                (np.log(1 + mae_pt_per_seq[:seq_len]), np.log(1 + maes_ptm_per_seq[:seq_len]))
            ),
            "Mark": ["MAE"] * seq_len + ["MAE-E"] * seq_len,
        }

        fig7 = draw_lineplot(
            data=data_maes_per_seq,
            x="Event Index",
            y=r"$\log(1 + \mathrm{MAE})$",
            hue="Mark",
            figure_kwargs={"font.size": 18, "figure.figsize": (5, 5)},
        )
        save_fig(fig7, opt.plot_store_dir_for_this_batch, f"MAE_{idx}.pdf")
        logger.info(f"MAE_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    # Part 8: the value of \\sum_{m \\in M}{p^*(m)} given different history.
    probability_sum = data["probability_sum"]  # [batch_size, seq_len]
    mask_next = data["mask_next"]  # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(probability_sum, mask_next))
    for idx, (probability_sum_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_probability_sum_per_seq = {
            "Event Index": np.arange(1, seq_len + 1),
            r"$\sum_{m \in M}{p(m)}$": probability_sum_per_seq[:seq_len],
        }

        fig8 = draw_lineplot(
            data=data_probability_sum_per_seq,
            x="Event Index",
            y=r"$\sum_{m \in M}{p(m)}$",
            figure_kwargs={"font.size": 18, "figure.figsize": (5, 5)},
        )
        ax = fig8.gca()
        ax.set_ylim(bottom=-0.05, top=1.05)
        save_fig(fig8, opt.plot_store_dir_for_this_batch, f"sum_of_p_{idx}.pdf")
        logger.info(f"sum_of_p_m_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    """
    Part 9: The Logarithm of time prediction against all marks
    """
    tau_pred_all_mark = data["tau_pred_all_mark"]  # [batch_size, seq_len, num_marks]
    mask_next = data["mask_next"]  # [batch_size, seq_len]
    tau_pred_all_mark, mask_next = move_from_tensor_to_ndarray(tau_pred_all_mark, mask_next)
    # [batch_size, seq_len, num_marks] + [batch_size, seq_len]
    for idx, (tau_pred_all_mark_per_seq, mask_next) in enumerate(zip(tau_pred_all_mark, mask_next)):
        seq_len = mask_next_per_seq.sum()

        data_tau_pred_all_mark_per_seq = {
            "Event Index": [ele for ele in range(seq_len) for _ in range(num_marks)],
            r"$\log(1 + t_p)$": np.log(1 + tau_pred_all_mark_per_seq[:seq_len, :]).flatten(),
            "Mark": [f"Mark {i}" for i in range(num_marks)] * seq_len,
        }

        fig9 = draw_lineplot(
            data=data_tau_pred_all_mark_per_seq,
            x="Event Index",
            y=r"$\log(1 + t_p)$",
            hue="Mark",
            figure_kwargs={"font.size": 18, "figure.figsize": (5, 5)},
        )
        save_fig(fig9, opt.plot_store_dir_for_this_batch, f"log_pred_time_{idx}.pdf")
        logger.info(f"log_pred_time_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!")

    return 0
