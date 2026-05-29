"""
Use LLMs to verify if one user is IO or not based on their timeline.
This file evaluates the performance of the LLM-only baseline.

Please note that we only use local LLMs in our work.
LLM API calls might work but untested.
"""

import argparse
import gc
import importlib
import math
import multiprocessing
import os
import pickle as pkl
from pathlib import Path
from typing import Any

# Environment configuration
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["SGLANG_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm


def load_from_pkl(filepath: Path, compression: str | None = None) -> Any:
    """Load data from a pickle file with optional compression.

    Args:
        filepath: Path to the pickle file
        compression: Compression method ("lzma", "bz2", "gz", or None)

    Returns:
        Loaded data
    """
    compression_algorithms = {
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }

    if compression is None:
        # Auto-detect compression from extension
        suffix = Path(filepath).suffixes[-1][1:]
        selected_open_function = compression_algorithms.get(suffix, open)
    else:
        selected_open_function = compression_algorithms[compression]

    with selected_open_function(filepath, "rb") as f:
        data = pkl.load(f)

    return data


def evaluate_io_user(responses_list: list[list[str]]) -> list[float]:
    """Decide whether a user is an IO user or not by how much of posts are IO posts.

    Returns a list of scores (ratio of IO posts) for each user.
    """
    user_scores = []
    for user_responses in responses_list:
        if not user_responses:
            user_scores.append(0.0)
            continue

        io_count = 0
        for resp in user_responses:
            # Based on inspection, responses are 'True' or 'False' strings
            if str(resp).lower() == "true":
                io_count += 1

        user_scores.append(io_count / len(user_responses))
    return user_scores


def main() -> None:
    """Main execution function."""
    root_path = Path(__file__).parent.parent.parent.resolve()
    dataset_dir = root_path / "useful_utilities/io_dataset/results"
    dataset_names = ["Russia_1", "China_1", "Egypt"]
    indices = ["1", "2", "3"]

    # Support multiple LLM models
    llm_models = ["openai_gpt-oss-120b", "llama3.3"]

    for llm_model in llm_models:
        filename = f"llm_baseline_AP_{llm_model}_test_nll.pkl.bz2.pkl.lzma"
        all_results = {}
        found_any = False

        for dataset in dataset_names:
            dataset_metrics = {"precision": [], "recall": [], "f1": [], "auc": [], "pr_auc": []}

            for idx in indices:
                filepath = dataset_dir / dataset / idx / filename
                if not filepath.exists():
                    # print(f"Warning: File {filepath} not found.")
                    continue

                found_any = True
                print(f"Processing {llm_model} - {dataset} index {idx}...")
                data = load_from_pkl(filepath, compression="lzma")

                responses = data["responses"]
                # Based on inspection, y_label=0 is IO and y_label=1 is normal.
                # We want to evaluate IO as the positive class.
                y_true_raw = np.array(data["y_label"])
                y_true = (y_true_raw == 0).astype(int)

                # Calculate scores (ratio of IO posts)
                y_scores = np.array(evaluate_io_user(responses))

                # Decide threshold (e.g., 0.1) for binary classification
                y_pred = (y_scores > 0.1).astype(int)

                # Evaluate
                if len(np.unique(y_true)) < 2:
                    print(f"Warning: Only one class present in y_true for {dataset} index {idx}. Skipping AUC.")
                    auc = 0.0
                    pr_auc = 0.0
                else:
                    auc = roc_auc_score(y_true, y_scores)
                    pr_auc = average_precision_score(y_true, y_scores)

                dataset_metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
                dataset_metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
                dataset_metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))
                dataset_metrics["auc"].append(auc)
                dataset_metrics["pr_auc"].append(pr_auc)

            if dataset_metrics["precision"]:
                all_results[dataset] = dataset_metrics

        if found_any:
            # Calculate average and std
            output_lines = []
            output_lines.append(f"LLM Model: {llm_model}")
            output_lines.append("=" * 30)
            for dataset, metrics in all_results.items():
                output_lines.append(f"Dataset: {dataset}")
                for metric_name, values in metrics.items():
                    avg = np.mean(values)
                    std = np.std(values)
                    output_lines.append(f"  {metric_name:10}: {avg:.4f} +/- {std:.4f}")
                output_lines.append("-" * 20)

            # Write to final result text file for this LLM
            output_path = root_path / f"useful_utilities/io_dataset/final_results_{llm_model}.txt"
            with open(output_path, "w") as f:
                f.write("\n".join(output_lines))

            print(f"Results for {llm_model} written to {output_path}")
        else:
            print(f"No results found for {llm_model}")


if __name__ == "__main__":
    main()
