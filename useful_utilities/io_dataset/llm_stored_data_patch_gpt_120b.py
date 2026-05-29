#!/usr/bin/env python

"""
Sometimes LLMs return an empty string as the IO user detection results, which can break the llm scoring scripts.
Someone can refer to this patch script to fix them.
"""


import multiprocessing
import os

from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from llm_info_sglang import description_abnormal_category, description_normal_category
from sklearn.metrics import average_precision_score, precision_recall_curve
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
result_indexes = ['1', '2', '3']
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
    "Iran_4",
    # "Iran_3",
    # "Russia_5",
    # "Russia_2",
    "Cuba",
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

temperature = 0.5
llm_name = 'openai/gpt-oss-120b'
# used_scoring_model="nomic-ai/nomic-embed-text-v2-moe"
# used_scoring_model="google/embeddinggemma-300m"
used_scoring_model="BAAI/bge-reranker-v2-m3"
# used_scoring_model="Qwen/Qwen3-Embedding-8B"
# used_scoring_model = "google/gemini-embedding-001"

prompt_version = 'v2'
local_llm = True
api_key = "sk-or-v1-YOUR_KEYS"

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


def dump_to_pkl(data, filepath, compression=None):
    import importlib
    import os
    import pickle as pkl
    from pathlib import Path

    dict_compression_algorithms = {
        None: open,
        "lzma": importlib.import_module("lzma").open,
        "bz2": importlib.import_module("bz2").open,
        "gz": importlib.import_module("gzip").open,
    }
    head, tail = os.path.split(filepath)
    tail = tail + f"{'.' + compression if compression is not None else ''}"
    filepath = Path(head, tail)

    selected_open_function = dict_compression_algorithms[compression]
    f = selected_open_function(filepath, "wb")
    pkl.dump(data, f)
    f.close()

    return 0


llm_assigned_name = llm_name.replace("/", "_")
used_scoring_assigned_name = used_scoring_model.replace("/", "_")
# dataset_files = ['train_nll.pkl.bz2', 'evaluation_nll.pkl.bz2', 'test_nll.pkl.bz2']
# dataset_files = ["evaluation_nll.pkl.bz2", "test_nll.pkl.bz2"]
dataset_files = ["test_nll.pkl.bz2"]

target_files = [
    f"AP_{llm_assigned_name}_{sampled_dataset_file}.pkl.lzma"
    for sampled_dataset_file in dataset_files
]

label_translator = {0: "Abnormal", 1: "Normal"}

for sampled_dataset in dataset_names:
    print(f"Processing {sampled_dataset}...")

    for sampled_dataset_file, sampled_target_file in zip(dataset_files, target_files):
        y_score_len = 0
        ap_all_evaluations = []
        ap_no_llm_all_evaluations = []
        ap_only_llms = []

        for result_index in result_indexes:
            target_result_dir = root_path / result_dir / sampled_dataset / result_index

            results = load_from_pkl(target_result_dir / sampled_target_file)

            responses = results["responses"]
            print(responses)
            description_abnormal = description_abnormal_category[prompt_version]
            description_normal = description_normal_category[prompt_version]

            print(description_abnormal)
            print(description_normal)

            if sampled_dataset == "Russia_1":
                if result_index == '1':
                    # Russia_1 index 1
                    responses[0][202] = 'Control account'
                    responses[1][1484] = 'Control account'
                    responses[2][1858] = 'Control account'
                    responses[3][771] = 'Control account'
                elif result_index == '2':
                    # Russia_1 index 2
                    responses[1][162] = 'Control account'
                    responses[1][1484] = 'Control account'
                    responses[2][1008] = 'IO account'
                    responses[3][1429] = 'Control account'
                    responses[4][771] = 'IO account'

                elif result_index == '3':
                    responses[0][459] = 'Control account'
                    responses[0][1484] = 'Control account'
                    responses[0][2206] = 'Control account'
                    responses[1][1501] = 'IO account'
                    responses[2][771] = 'Control account'
                    responses[3][1501] = 'IO account'
                    responses[3][1839] = 'Control account'
                    responses[3][1972] = 'Control account'
                    responses[4][1972] = 'Control account'
                    responses[4][2050] = 'IO account'
            if sampled_dataset == 'China_1':
                if result_index == '1':
                    responses[0][1041] = 'Control account'
                    responses[1][1041] = 'Control account'
                elif result_index == '2':
                    responses[0][1042] = 'Control account'
                    responses[4][1041] = 'Control account'
                elif result_index == '3':
                    responses[0][228] = 'Control account'
                    responses[3][1687] = 'IO account'
            if sampled_dataset == 'Cuba':
                if result_index == '1':
                    pass
                if result_index == '3':
                    responses[4][1326] = 'Control account'
                    responses[4][1555] = 'IO account'
                    responses[4][1735] = 'IO account'
                    responses[4][1748] = 'IO account'
            if sampled_dataset == 'UAE':
                if result_index == '1':
                    pass
                if result_index == '3':
                    responses[0][594] = 'Control account'

            results["responses"] = responses
            dump_to_pkl(results, target_result_dir / sampled_target_file[:-5], compression='lzma')
            print('stop')
