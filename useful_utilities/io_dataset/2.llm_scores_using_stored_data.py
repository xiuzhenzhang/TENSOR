#!/usr/bin/env python

'''
This file evaluates the performance of TENSOR.
Default temperature is 0.5.
If you want to do temperature sensitivity testing as shown in section 5.4, change temperature to np.arange(0.1, 2.05, 0.05).
then get the best temperature using best_temp.py
'''


import multiprocessing
import os

from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from llm_info_sglang import description_abnormal_category, description_normal_category
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from vllm import LLM

default_figure_kwargs = {
    "font.size": 18,
    "figure.figsize": (8, 4),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    "mathtext.fontset": "dejavusans",
    "text.latex.preamble": "\\usepackage{amsmath}",
}

# root is the io_dataset folder.
root_path = Path(__file__).parent.resolve()
result_dir = "results"
# result_indexes = ['1', '2', '3', '4', '5', '6']
result_indexes = ["1", "2", "3"]
procedure_name = "TPP"
dataset_names = [
    # "Armenia",
    # "Qatar",
    # "Ecuador",
    "Egypt",
    "Russia_1",
    # "Thailand",
    # "Catalonia",
    "China_1",
    # "Iran_6",
    "Iran_1",
    # "Ghana",
    # "Russia_4",
    # "Iran_2",
    # "Iran_5",
    # "Venezuela_1",
    # "Spain",
    # "China_2",
    # "Iran_4",
    # "Iran_3",
    # "Russia_5",
    # "Russia_2",
    # "Cuba",
    "UAE",
]
task_name = "nll_with_label"
seed = 42

model_name = "sahp"
lr = 0.002
batch_size = 32
nts = 20000
train_procedure_config = None
train_dataloader_config = "sahp_dl.yml"
train_model_config = "sahp.yml"

# temperatures = np.arange(0.1, 2.05, 0.05)
temperatures = [0.5]

# llm_name = "openai/gpt-oss-120b"
# llm_name = "meta-llama/Llama-3.3-70B-Instruct"
# llm_name = "Qwen/Qwen3-Next-80B-A3B-Thinking"
# llm_name = "zai-org/GLM-4.5-Air"
llm_name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

used_scoring_model = "BAAI/bge-reranker-v2-m3"

prompt_version = "v2"
local_llm = True
api_key = "sk-or-v1-YOUR_KEY"

# ============ Code Blocks ============

BATCH_SIZE = 10000


def calculate_ap_random_chunk(args):
    seed, n_iters, y_label, size = args
    rng = np.random.default_rng(seed)
    ap_random = []
    for _ in range(n_iters):
        random_scores = rng.uniform(low=0.0, high=1.0, size=size)
        ap_random.append(average_precision_score(y_true=y_label, y_score=random_scores, pos_label=0))
    return ap_random


def load_from_pkl(filepath, compression=None):
    import importlib
    import pathlib
    import pickle as pkl

    dict_compression_algorithms = {
        # Is it a good choice?
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }

    # If compression is None, we will guess the compression algorithm.
    # ***.lzma -> lzma
    # ***.bz2 -> bz2
    # ***.gz -> gz
    # others -> no compression.
    if compression is None:
        selected_open_function = dict_compression_algorithms.get(pathlib.Path(filepath).suffixes[-1][1:], open)
    else:
        selected_open_function = dict_compression_algorithms[compression]

    f = selected_open_function(filepath, "rb")
    data = pkl.load(f)
    f.close()

    return data


def write_to_txt(strings, filepath):
    from pathlib import Path

    f = Path(filepath).open("w")

    if isinstance(strings, list):
        strings = [item + "\n" for item in strings]
        f.writelines(strings)
    else:
        f.write(strings)

    f.close()

    return 0


def find_optimal_threshold(y_label, y_score, pos_label=0):
    """
    Find the optimal threshold that maximizes F1 score for binary classification.

    Args:
        y_label: True labels
        y_score: Predicted scores
        pos_label: The label of the positive class (default 0 for abnormal)

    Returns:
        optimal_threshold: The threshold that maximizes F1 score
        max_f1: The maximum F1 score achieved
    """
    # Get precision-recall curve and thresholds
    precision, recall, thresholds = precision_recall_curve(y_label, y_score, pos_label=pos_label)

    # Calculate F1 scores for each threshold
    # Note: precision_recall_curve returns n+1 precision and recall values but n thresholds
    # The last precision/recall values correspond to threshold approaching infinity
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)

    # Find the index of the maximum F1 score
    max_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[max_f1_idx]
    optimal_threshold = thresholds[max_f1_idx]

    return optimal_threshold, max_f1


llm_assigned_name = llm_name.replace("/", "_")
used_scoring_assigned_name = used_scoring_model.replace("/", "_")
# dataset_files = ['train_nll.pkl.bz2', 'evaluation_nll.pkl.bz2', 'test_nll.pkl.bz2']
# dataset_files = ["evaluation_nll.pkl.bz2", "test_nll.pkl.bz2"]
dataset_files = ["test_nll.pkl.bz2"]
target_files = [f"AP_{llm_assigned_name}_{sampled_dataset_file}.pkl.lzma" for sampled_dataset_file in dataset_files]

if local_llm:
    llm_scoring = LLM(model=used_scoring_model, runner="pooling", gpu_memory_utilization=0.95, trust_remote_code=True)
else:
    llm_embedding = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

label_translator = {0: "Abnormal", 1: "Normal"}

for sampled_dataset in dataset_names:
    print(f"Processing {sampled_dataset}...")

    for sampled_dataset_file, sampled_target_file in zip(dataset_files, target_files):
        y_score_len = 0
        ap_all_evaluations = {}
        ap_no_llm_all_evaluations = {}
        ap_only_llms = {}
        ap_with_random_all_evaluations = {}
        precision_all_evaluations = {}
        recall_all_evaluations = {}
        f1_all_evaluations = {}
        auc_all_evaluations = {}

        for temperature in temperatures:
            ap_all_evaluations[temperature] = []
            ap_no_llm_all_evaluations[temperature] = []
            ap_only_llms[temperature] = []
            ap_with_random_all_evaluations[temperature] = []
            precision_all_evaluations[temperature] = []
            recall_all_evaluations[temperature] = []
            f1_all_evaluations[temperature] = []
            auc_all_evaluations[temperature] = []

        for result_index in result_indexes:
            target_result_dir = root_path / result_dir / sampled_dataset / result_index

            results = load_from_pkl(target_result_dir / sampled_target_file)
            eval_file = f"AP_{llm_assigned_name}_evaluation_nll.pkl.bz2.pkl.lzma"
            evaluation_results = load_from_pkl(target_result_dir / eval_file)

            if llm_name == "meta-llama/Llama-3.3-70B-Instruct":
                responses = [results["responses"]]
                eval_responses = [evaluation_results["responses"]]
            elif llm_name == "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
                # the shape is [seq_len, n]
                responses = results["responses"]
                n = len(responses[0])
                print(f'n: {n}')
                reshaped_responses = [[] for _ in range(n)]
                for item in responses:
                    for idx, response in enumerate(item):
                        reshaped_responses[idx].append(response[:8000])
                responses = reshaped_responses

                eval_responses = evaluation_results["responses"]
                reshaped_responses = [[] for _ in range(n)]
                for item in eval_responses:
                    for idx, response in enumerate(item):
                        reshaped_responses[idx].append(response[:8000])
                eval_responses = reshaped_responses
            else:
                responses = results["responses"]
                eval_responses = evaluation_results["responses"]

            description_abnormal = description_abnormal_category[prompt_version]
            description_normal = description_normal_category[prompt_version]

            print(description_abnormal)
            print(description_normal)

            y_label = results["y_label"]
            y_score = results["y_score"]
            y_score_len = len(y_score)

            eval_y_label = evaluation_results["y_label"]
            eval_y_score = evaluation_results["y_score"]

            abnormal_scores = []
            normal_scores = []
            eval_abnormal_scores = []
            eval_normal_scores = []

            if local_llm:
                # Process test responses
                for each_abnormal_description in description_abnormal:
                    tmp = []
                    for idx, responses_per_retry in enumerate(responses):
                        print(f"test abnormal: {idx}")
                        tmp.append(
                            [
                                item.outputs.score
                                for item in llm_scoring.score(
                                    each_abnormal_description, responses_per_retry, use_tqdm=False
                                )
                            ]
                        )
                    abnormal_scores.append(tmp)
                # [description_abnormal, n, n_query]
                abnormal_scores = torch.tensor(abnormal_scores).mean(dim=-2).mean(dim=-2)

                for each_normal_description in description_normal:
                    tmp = []
                    for idx, responses_per_retry in enumerate(responses):
                        print(f"test normal: {idx}")
                        tmp.append(
                            [
                                item.outputs.score
                                for item in llm_scoring.score(
                                    each_normal_description, responses_per_retry, use_tqdm=False
                                )
                            ]
                        )
                    normal_scores.append(tmp)
                # [description_abnormal, n, n_query]
                normal_scores = torch.tensor(normal_scores).mean(dim=-2).mean(dim=-2)

                # Process evaluation responses for threshold optimization
                for each_abnormal_description in description_abnormal:
                    tmp = []
                    for idx, responses_per_retry in enumerate(eval_responses):
                        print(f"eval abnormal: {idx}")
                        tmp.append(
                            [
                                item.outputs.score
                                for item in llm_scoring.score(
                                    each_abnormal_description, responses_per_retry, use_tqdm=False
                                )
                            ]
                        )
                    eval_abnormal_scores.append(tmp)
                # [description_abnormal, n, n_query]
                eval_abnormal_scores = torch.tensor(eval_abnormal_scores).mean(dim=-2).mean(dim=-2)

                for each_normal_description in description_normal:
                    tmp = []
                    for idx, responses_per_retry in enumerate(eval_responses):
                        print(f"eval normal: {idx}")
                        tmp.append(
                            [
                                item.outputs.score
                                for item in llm_scoring.score(
                                    each_normal_description, responses_per_retry, use_tqdm=False
                                )
                            ]
                        )
                    eval_normal_scores.append(tmp)
                # [description_abnormal, n, n_query]
                eval_normal_scores = torch.tensor(eval_normal_scores).mean(dim=-2).mean(dim=-2)
            else:
                abnormal_description_embeddings = llm_embedding.embeddings.create(
                    model=used_scoring_model, input=description_abnormal, encoding_format="float"
                )
                abnormal_description_embeddings = torch.tensor(
                    [item.embedding for item in abnormal_description_embeddings.data]
                )  # [num_abnormal_archors, emb_size]

                normal_description_embeddings = llm_embedding.embeddings.create(
                    model=used_scoring_model, input=description_normal, encoding_format="float"
                )
                normal_description_embeddings = torch.tensor(
                    [item.embedding for item in normal_description_embeddings.data]
                )  # [num_normal_archors, emb_size]

                # Process test responses
                response_embeddings = llm_embedding.embeddings.create(
                    model=used_scoring_model, input=responses, encoding_format="float"
                )
                response_embeddings = torch.tensor(
                    [item.embedding for item in response_embeddings.data]
                )  # [num_responses, emb_size]

                abnormal_scores = torch.nn.functional.cosine_similarity(
                    abnormal_description_embeddings.unsqueeze(dim=-2), response_embeddings.unsqueeze(dim=0), dim=-1
                )
                # [num_abnormal_archors, num_responses]
                normal_scores = torch.nn.functional.cosine_similarity(
                    normal_description_embeddings.unsqueeze(dim=-2), response_embeddings.unsqueeze(dim=0), dim=-1
                )
                # [num_normal_archors, num_responses]

                abnormal_scores = abnormal_scores.max(dim=0).values
                normal_scores = normal_scores.mean(dim=0)

                # Process evaluation responses for threshold optimization
                eval_response_embeddings = llm_embedding.embeddings.create(
                    model=used_scoring_model, input=eval_responses, encoding_format="float"
                )
                eval_response_embeddings = torch.tensor(
                    [item.embedding for item in eval_response_embeddings.data]
                )  # [num_eval_responses, emb_size]

                eval_abnormal_scores = torch.nn.functional.cosine_similarity(
                    abnormal_description_embeddings.unsqueeze(dim=-2), eval_response_embeddings.unsqueeze(dim=0), dim=-1
                )
                # [num_abnormal_archors, num_eval_responses]
                eval_normal_scores = torch.nn.functional.cosine_similarity(
                    normal_description_embeddings.unsqueeze(dim=-2), eval_response_embeddings.unsqueeze(dim=0), dim=-1
                )
                # [num_normal_archors, num_eval_responses]

                eval_abnormal_scores = eval_abnormal_scores.max(dim=0).values
                eval_normal_scores = eval_normal_scores.mean(dim=0)

            # Compute test set scores
            merged_score = torch.stack([abnormal_scores, normal_scores], dim=-1)
            abnormal_score = torch.nn.functional.softmax(merged_score, dim=-1)[:, 0]
            random_abnormal_score = torch.zeros_like(abnormal_score).uniform_(0, 1)

            # Compute evaluation set scores for threshold optimization
            eval_merged_score = torch.stack([eval_abnormal_scores, eval_normal_scores], dim=-1)
            eval_abnormal_score = torch.nn.functional.softmax(eval_merged_score, dim=-1)[:, 0]

            for temperature in temperatures:
                # Compute scores for test set
                y_score_llm_only = abnormal_score / temperature
                y_score_with_llm = torch.exp(torch.tensor(y_score)) * torch.exp(y_score_llm_only)
                y_score_random = random_abnormal_score / temperature
                y_score_with_random = torch.exp(torch.tensor(y_score)) * torch.exp(y_score_random)
                # Avoid ridiculously high score.
                y_score_with_llm = y_score_with_llm.clamp(max=1e6)

                # Compute scores for evaluation set (for threshold finding)
                eval_y_score_llm_only = eval_abnormal_score / temperature
                eval_y_score_with_llm = torch.exp(torch.tensor(eval_y_score)) * torch.exp(eval_y_score_llm_only)
                eval_y_score_with_llm = eval_y_score_with_llm.clamp(max=1e6)

                # scale the score to [0, 1]
                y_score_with_llm = torch.nn.functional.tanh(y_score_with_llm)
                y_score_with_random = torch.nn.functional.tanh(y_score_with_random)
                eval_y_score_with_llm = torch.nn.functional.tanh(eval_y_score_with_llm)

                # Find optimal threshold on evaluation set
                optimal_threshold, eval_max_f1 = find_optimal_threshold(
                    eval_y_label, eval_y_score_with_llm, pos_label=0
                )

                # Apply threshold to test set
                y_pred = (y_score_with_llm <= optimal_threshold).int().numpy()

                # print(eval_y_score_with_llm)
                # print(y_score_with_llm)
                # print(optimal_threshold)

                # Compute classification metrics on test set
                precision = precision_score(y_label, y_pred, pos_label=0, zero_division=0)
                recall = recall_score(y_label, y_pred, pos_label=0, zero_division=0)
                f1 = f1_score(y_label, y_pred, pos_label=0, zero_division=0)

                # Compute AUC score (ROC-AUC)
                # Convert labels so that pos_label=0 (abnormal) becomes 1 for AUC calculation
                y_label_binary = [1 if label == 0 else 0 for label in y_label]
                auc = roc_auc_score(y_label_binary, y_score_with_llm)

                ap_with_llm = average_precision_score(y_true=y_label, y_score=y_score_with_llm, pos_label=0)
                ap_llm_only = average_precision_score(y_true=y_label, y_score=torch.exp(y_score_llm_only), pos_label=0)
                ap_no_llm = average_precision_score(y_true=y_label, y_score=y_score, pos_label=0)
                ap_with_random = average_precision_score(y_true=y_label, y_score=y_score_with_random, pos_label=0)

                ap_all_evaluations[temperature].append(ap_with_llm)
                ap_only_llms[temperature].append(ap_llm_only)
                ap_no_llm_all_evaluations[temperature].append(ap_no_llm)
                ap_with_random_all_evaluations[temperature].append(ap_with_random)
                precision_all_evaluations[temperature].append(precision)
                recall_all_evaluations[temperature].append(recall)
                f1_all_evaluations[temperature].append(f1)
                auc_all_evaluations[temperature].append(auc)

                # Generate precision-recall plot for this result_index
                plt.rcParams.update(default_figure_kwargs)
                fig, ax = plt.subplots()

                # Calculate precision-recall curves for each scoring method
                precision_with_llm, recall_with_llm, _ = precision_recall_curve(y_label, y_score_with_llm, pos_label=0)
                precision_llm_only, recall_llm_only, _ = precision_recall_curve(
                    y_label, torch.exp(y_score_llm_only), pos_label=0
                )
                precision_no_llm, recall_no_llm, _ = precision_recall_curve(y_label, y_score, pos_label=0)

                # Plot the precision-recall curves
                ax.plot(recall_with_llm, precision_with_llm, label=f"w/ LLM (AP={ap_with_llm:.3f})", linewidth=2)
                ax.plot(recall_llm_only, precision_llm_only, label=f"LLM only (AP={ap_llm_only:.3f})", linewidth=2)
                ax.plot(recall_no_llm, precision_no_llm, label=f"w/o LLM (AP={ap_no_llm:.3f})", linewidth=2)

                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Curve - {sampled_dataset}")
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)

                # Save the plot in the target_result_dir
                plot_filename = (
                    target_result_dir
                    / f"PR_{llm_assigned_name}_{used_scoring_assigned_name}_{sampled_dataset_file[:-8]}_{str(temperature)}.pdf"
                )
                plt.tight_layout()
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(f"Saved precision-recall plot to {plot_filename}")

                # Write result-specific text file
                write_to_txt(
                    [
                        f"Average Precision Score w/ LLM: {ap_with_llm}",
                        f"Average Precision Score w/o LLM: {ap_no_llm}",
                        f"Average Precision Score w LLM only: {ap_llm_only}",
                        f"Average Precision Score w random: {ap_with_random}",
                        f"",
                        f"Threshold-based Classification Metrics (optimized on eval set):",
                        f"Optimal Threshold: {optimal_threshold}",
                        f"Evaluation Set Max F1: {eval_max_f1}",
                        f"Test Set Precision: {precision}",
                        f"Test Set Recall: {recall}",
                        f"Test Set F1: {f1}",
                        f"Test Set AUC: {auc}",
                    ],
                    target_result_dir
                    / f"AP_{llm_assigned_name}_{used_scoring_assigned_name}_{sampled_dataset_file}_{str(temperature)}.txt",
                )

        # What is the performance of random guess?
        rng = np.random.default_rng(seed)
        n_total = 10000
        n_processes = multiprocessing.cpu_count()
        chunk_size = n_total // n_processes
        remainder = n_total % n_processes

        tasks = []
        for i in range(n_processes):
            iters = chunk_size + (1 if i < remainder else 0)
            if iters > 0:
                chunk_seed = rng.integers(0, 2**32)
                tasks.append((chunk_seed, iters, y_label, y_score_len))

        ap_random = []
        with (
            multiprocessing.Pool(processes=n_processes) as pool,
            tqdm(total=n_total, desc="Random AP Calculation") as pbar,
        ):
            for result in pool.imap_unordered(calculate_ap_random_chunk, tasks):
                ap_random.extend(result)
                pbar.update(len(result))

        mean_ap_random = np.mean(ap_random)
        std_ap_random = np.std(ap_random)

        results = []
        for temperature in temperatures:
            mean_ap_all_evaluations = np.mean(ap_all_evaluations[temperature])
            print(mean_ap_all_evaluations)
            std_ap_all_evaluations = np.std(ap_all_evaluations[temperature])

            mean_ap_no_llm_all_evaluations = np.mean(ap_no_llm_all_evaluations[temperature])
            std_ap_no_llm_all_evaluations = np.std(ap_no_llm_all_evaluations[temperature])

            mean_ap_only_llms = np.mean(ap_only_llms[temperature])
            std_ap_only_llms = np.std(ap_only_llms[temperature])

            mean_ap_only_random = np.mean(ap_with_random_all_evaluations[temperature])
            std_ap_only_random = np.std(ap_with_random_all_evaluations[temperature])

            mean_precision = np.mean(precision_all_evaluations[temperature])
            std_precision = np.std(precision_all_evaluations[temperature])

            mean_recall = np.mean(recall_all_evaluations[temperature])
            std_recall = np.std(recall_all_evaluations[temperature])

            mean_f1 = np.mean(f1_all_evaluations[temperature])
            std_f1 = np.std(f1_all_evaluations[temperature])

            mean_auc = np.mean(auc_all_evaluations[temperature])
            std_auc = np.std(auc_all_evaluations[temperature])

            results.extend(
                [
                    f"For temperature {temperature}",
                    f"Average Precision Score w/ LLM: {mean_ap_all_evaluations}±{std_ap_all_evaluations}",
                    f"Average Precision Score w/o LLM: {mean_ap_no_llm_all_evaluations}±{std_ap_no_llm_all_evaluations}",
                    f"Average Precision Score w LLM only: {mean_ap_only_llms}±{std_ap_only_llms}",
                    f"Average Precision Score w random: {mean_ap_only_random}±{std_ap_only_random}",
                    f"Random Guess: {mean_ap_random}±{std_ap_random}",
                    f"Test Set Precision (threshold optimized on eval): {mean_precision}±{std_precision}",
                    f"Test Set Recall (threshold optimized on eval): {mean_recall}±{std_recall}",
                    f"Test Set F1 (threshold optimized on eval): {mean_f1}±{std_f1}",
                    f"Test Set AUC: {mean_auc}±{std_auc}",
                ]
            )

        write_to_txt(
            results,
            root_path
            / f"results/{sampled_dataset}/AP_{llm_assigned_name}_{used_scoring_assigned_name}_{sampled_dataset_file}.txt",
        )
