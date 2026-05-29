import numpy as np
from tqdm import tqdm

from src.toolbox.dict_operation import dict_merge
from src.toolbox.misc import dump_to_pkl, get_logger, mkdir_if_not_exist, write_to_txt, write_yaml

logger = get_logger(name=__file__)


def spearman_and_l1_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    """
    spearman, l1 = all_evaluation_results
    spearman = np.mean(spearman)
    l1 = np.mean(l1)

    result_file = opt.store_dir / f"{desc}_spearman_and_l1.txt"
    strings = f"For the {desc} of {opt.dataset_name}, we announce that the average spearman coefficient is {spearman} and average L1 distance is {l1}."
    write_to_txt(strings, result_file)

    result_dist_file = opt.store_dir / f"{desc}_spearman_and_l1_result.pkl"
    dump_to_pkl({"spearman": spearman, "l1": l1}, result_dist_file, compression="bz2")


def mae_and_f1_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = mae_and_f1.

    This function calculates the average of mae and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae values for calculating Q1, Q2, and Q3 later.
    """
    mae, acc, macro_f1, micro_f1, pred_time, dist, events_next = all_evaluation_results

    acc = np.mean(acc)
    macro_f1 = np.mean(macro_f1)
    micro_f1 = np.mean(micro_f1)
    mean_mae = np.mean(mae)

    mae_dist_file = opt.store_dir / f"{desc}_mae_data.pkl"
    data = {"pred_time": pred_time, "dist": dist, "events_next": events_next}
    dump_to_pkl(data, mae_dist_file, compression="bz2")

    result_file = opt.store_dir / f"{desc}_mae_and_macro-f1.txt"
    strings = [
        f"For the {desc} of {opt.dataset_name}, the average MAE is {mean_mae}.",
        f"The average acc is {acc}.",
        f"The average macro-F1 is {macro_f1}.",
        f"The average micro-F1 is {micro_f1}.",
    ]
    write_to_txt(strings, result_file)
    result_dist_file = opt.store_dir / f"{desc}_mae_and_f1_result.pkl"
    dump_to_pkl(
        {"mae": mean_mae, "acc": acc, "macro-F1": macro_f1, "micro-F1": micro_f1}, result_dist_file, compression="bz2"
    )


def mae_e_and_f1_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    """

    mae_e, acc, macro_f1, micro_f1, mark_dist, pred_time_all_marks, time_next, events_next = all_evaluation_results

    acc = np.mean(acc)
    macro_f1 = np.mean(macro_f1)
    micro_f1 = np.mean(micro_f1)
    mean_mae_e = np.mean(mae_e)

    mae_e_dist_file = opt.store_dir / f"{desc}_mae_e_data.pkl"
    data = {
        "t_m": pred_time_all_marks,
        "events_next": events_next,
        "pm": mark_dist,
        "time_next": time_next,
    }
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")

    # Report the average of mae-e and f1.
    result_file = opt.store_dir / f"{desc}_mae_e_and_macro-f1.txt"
    strings = [
        f"For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e}.",
        f"The average acc is {acc}.",
        f"The average macro-F1 is {macro_f1}.",
        f"The average micro-F1 is {micro_f1}.",
    ]
    write_to_txt(strings, result_file)
    result_dist_file = opt.store_dir / f"{desc}_mae_e_and_f1_result.pkl"
    dump_to_pkl(
        {"mae": mean_mae_e, "acc": acc, "macro-F1": macro_f1, "micro-F1": micro_f1}, result_dist_file, compression="bz2"
    )


def which_mark_occurs_first_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = which_mark_occurs_first.
    """
    mae, acc, macro_f1, micro_f1 = all_evaluation_results
    mae = np.mean(mae)
    acc = np.mean(acc)
    macro_f1 = np.mean(macro_f1)
    micro_f1 = np.mean(micro_f1)

    """
    Report the average of mae-e and f1.
    """
    result_file = opt.store_dir / f"{desc}_which_event_first.txt"
    strings = [
        f"For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mae}.",
        f"The average acc is {acc}.",
        f"The average macro-F1 is {macro_f1}.",
        f"The average micro-F1 is {micro_f1}.",
    ]
    write_to_txt(strings, result_file)
    result_dist_file = opt.store_dir / f"{desc}_which_mark_occurs_first_result.pkl"
    dump_to_pkl(
        {"mae": mae, "acc": acc, "macro-F1": macro_f1, "micro-F1": micro_f1}, result_dist_file, compression="bz2"
    )


def balanced_sampling_from_distribution_postprocess(all_evaluation_results, desc, opt):
    """
    Dump the detailed distribution of mae-e for further usage.
    """
    samples, p_ms = all_evaluation_results

    mae_e_dist_file = opt.store_dir / f"{desc}_samples_at_every_point.pkl"
    data = {"samples": samples, "p_ms": p_ms}
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")


def cppod_evaluation_postprocess(all_evaluation_results, desc, opt):
    rocs = all_evaluation_results
    rocs = np.nanmean(rocs).item()

    result_file = opt.store_dir / f"{desc}_roc_mean.txt"
    strings = f"For the {desc} of {opt.dataset_name}, we announce that the average roc of outlier detection is {rocs}."
    write_to_txt(strings, result_file)

    mae_e_dist_file = opt.store_dir / f"{desc}_cppod_rocs.pkl"
    data = {"rocs": rocs}
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")


def cppod_commission_evaluation_postprocess(all_evaluation_results, desc, opt):
    rocs = all_evaluation_results
    rocs = np.nanmean(rocs).item()

    result_file = opt.store_dir / f"{desc}_roc_commission_mean.txt"
    strings = f"For the {desc} of {opt.dataset_name}, we announce that the average roc of outlier detection is {rocs}."
    write_to_txt(strings, result_file)

    mae_e_dist_file = opt.store_dir / f"{desc}_cppod_commission_rocs.pkl"
    data = {"rocs": rocs}
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")


def generate_hypro_dataset_postprocess(all_evaluation_results, desc, opt):
    """
    Dump the detailed distribution of mae-e for further usage.
    """
    input_time, input_events, tau_sampled, events_sampled = all_evaluation_results

    mae_e_dist_file = opt.store_dir / f"{desc}_hypro_sample.pkl"
    data = {
        "input_time": input_time,
        "input_events": input_events,
        "tau_sampled": tau_sampled,
        "events_sampled": events_sampled,
    }
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")
    write_yaml(
        {
            **opt.info_dict,
            "hypro_length": opt.number_of_events_hypro,
            "hypro_negative_samples": opt.number_of_negative_samples,
        },
        opt.store_dir,
        "dataset_card.yml",
    )


def llm_vs_mtpp_ranking_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = llm_ranking.

    This function computes the average ranking of the real event within a lot of sampled events.
    We expected that the real event should have a generally higher ranking (lower in the value) than randomly sampled events.
    """
    ranking_from_mtpp, ranking_from_llm = all_evaluation_results

    average_ranking_from_mtpp = np.mean(ranking_from_mtpp)
    average_ranking_from_llm = np.mean(ranking_from_llm)

    result_file = opt.store_dir / f"{desc}_ranking.txt"
    strings = f"For the {desc} of {opt.dataset_name}, we announce that the average ranking of true events decided by the MTPP model is {average_ranking_from_mtpp} and ranking of true events decided by the {opt.llm_model} is {average_ranking_from_llm}."
    write_to_txt(strings, result_file)

    ranking_results_file = opt.store_dir / f"{desc}_ranking.pkl"
    data = {"ranking_from_mtpp": ranking_from_mtpp, "ranking_from_llm": ranking_from_llm}
    dump_to_pkl(data, ranking_results_file, compression="bz2")


def mae_and_f1_of_imputated_events(model, dataset, desc, opt, early_offload):
    """
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    """
    elapsed_time = 0
    data_size = 0
    list_mae = []
    f1 = []

    with tqdm(dataset, desc=f"MAE and macro-f1 for imputated events in {desc}:") as progress_bar:
        for minibatch in progress_bar:
            mae_per_seq, f1_per_seq = model("mae_and_f1_imputated_events", minibatch, opt)
            # [batch_size, seq_len]
            list_mae.append(mae_per_seq.flatten().tolist())
            f1.append(f1_per_seq)

        elapsed_time = progress_bar.format_dict["elapsed"]
        data_size = progress_bar.format_dict["total"]

    f1 = np.array(f1).mean()
    mean_mae = np.concatenate(list_mae).mean().item()

    mkdir_if_not_exist(opt.store_dir)
    """
    Report the average of mae-e and f1.
    """
    result_file = opt.store_dir / f"{desc}_mae_e_and_macro-f1_of_imputated_events.txt"
    strings = [
        f"For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.\n",
        f"Evaluation speed: {elapsed_time / data_size}s per sequence.",
    ]
    write_to_txt(strings, result_file)

    """
    Dump the detailed distribution of mae-e for further usage.
    """
    mae_e_dist_file = opt.store_dir / f"{desc}_mae_e_of_imputated_events.pkl"
    data = {"mae_e": list_mae, "f1": f1}
    dump_to_pkl(data, mae_e_dist_file, compression="bz2")


def llm_mtpp_classification_postprocess(all_evaluation_results, desc, opt):
    """
    This function is called when task_name = llm_mtpp_classification.

    This function computes the average ranking of the real event within a lot of sampled events.
    We expected that the real event should have a generally higher ranking (lower in the value) than randomly sampled events.
    """
    ranking_from_mtpp, ranking_from_llm = all_evaluation_results

    average_ranking_from_mtpp = np.mean(ranking_from_mtpp)
    average_ranking_from_llm = np.mean(ranking_from_llm)

    result_file = opt.store_dir / f"{desc}_ranking.txt"
    strings = f"For the {desc} of {opt.dataset_name}, we announce that the average ranking of true events decided by the MTPP model is {average_ranking_from_mtpp} and ranking of true events decided by the {opt.llm_model} is {average_ranking_from_llm}."
    write_to_txt(strings, result_file)

    ranking_results_file = opt.store_dir / f"{desc}_ranking.pkl"
    data = {"ranking_from_mtpp": ranking_from_mtpp, "ranking_from_llm": ranking_from_llm}
    dump_to_pkl(data, ranking_results_file, compression="bz2")


def nll_with_label_postprocess(all_evaluation_results, desc, opt):
    nll_per_seq, labels, mask, passenger_data = all_evaluation_results

    # process the passenger.
    batched_passenger_data = {key: [] for key in passenger_data[0]}
    for item in passenger_data:
        for key in batched_passenger_data:
            batched_passenger_data[key].extend(item.get(key, None))

    categorized_nll = {}
    for nll, label in zip(nll_per_seq, labels):
        if categorized_nll.get(label) is None:
            categorized_nll[label] = [nll]
        else:
            categorized_nll[label].append(nll)

    nll_results_file = opt.store_dir / f"{desc}_nll.pkl"
    dump_to_pkl(
        {
            "categorized_nll": categorized_nll,
            "nll_per_seq": nll_per_seq,
            "labels": labels,
            "mask": mask,
            **batched_passenger_data,
        },
        nll_results_file,
        compression="bz2",
    )

    string_list = [f"For dataset {desc}:\n"]
    for label, nll in categorized_nll.items():
        string_list.append(f"There are {len(nll)} sequences with label {label}.")
        string_list.append(
            f"Sequences with label {label} has mean NLL value as {np.mean(nll)}. Standard deviation is {np.std(nll)}."
        )

    result_file = opt.store_dir / f"{desc}_nll.txt"
    write_to_txt(string_list, result_file)


desc_funcs = {
    "spearman_and_l1": {"desc_string": "Spearman and L1 for {0}", "postprocess_func": spearman_and_l1_postprocess},
    "mae_and_f1": {"desc_string": "MAE and macro-f1 for {0}", "postprocess_func": mae_and_f1_postprocess},
    "mae_e_and_f1": {"desc_string": "MAE-E and macro-f1 for {0}", "postprocess_func": mae_e_and_f1_postprocess},
    "which_mark_occurs_first": {
        "desc_string": "Predict the next event by finding which mark occurs first for {0}",
        "postprocess_func": which_mark_occurs_first_postprocess,
    },
    "balanced_sampling_from_distribution": {
        "desc_string": "Samples of {0} for each mark",
        "postprocess_func": balanced_sampling_from_distribution_postprocess,
    },
    "nll_with_label": {
        "desc_string": "Measure and group NLL based on labels for {0}",
        "postprocess_func": nll_with_label_postprocess,
    },
    "generate_hypro_dataset": {
        "desc_string": "Generate HYPRO dataset for {0}",
        "postprocess_func": generate_hypro_dataset_postprocess,
    },
    # experiment 1: real event classification
    "llm_mtpp_classification": {
        "desc_string": "Comparing real event classification accuracy of MTPP and LLM for {0}",
        "postprocess_func": llm_mtpp_classification_postprocess,
    },
    # CPPOD task.
    "cppod_evaluation": {
        "desc_string": "Obtaining CPPOD score for {0}",
        "postprocess_func": cppod_evaluation_postprocess,
    },
    "cppod_commission_evaluation": {
        "desc_string": "Obtaining CPPOD score on commission outlier for {0}",
        "postprocess_func": cppod_commission_evaluation_postprocess,
    },
    # Custom evaluation function.
    "mae_and_f1_of_imputated_events": mae_and_f1_of_imputated_events,
}


def task_running_on_the_entire_dataset_or_samples(task_name):
    available_tasks = desc_funcs.keys()
    if task_name not in available_tasks:
        logger.exception(f"Unknown task {task_name}. Available tasks are {available_tasks}.")
