#!/usr/bin/env python

"""
This scripts shows that the average NLL of IO users timeline is different from normal users.
These results support we use TPP models to do IO user detection.
"""


import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import PermutationMethod, anderson_ksamp, ks_2samp

default_figure_kwargs = {
    "font.size": 18,
    "figure.figsize": (8, 4),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    "mathtext.fontset": "dejavusans",
    "text.latex.preamble": "\\usepackage{amsmath}"
}

root_path = Path('/scratch/s3963046/project/workflow_next')
result_dir = 'results'
result_index = '2'
procedure_name = 'TPP'
dataset_names = ['Qatar', 'Armenia', 'Ecuador', 'Egypt', 'Russia_1', 'Thailand', 'Catalonia', 'China_1', 'Iran_6', 'Iran_1', 'Ghana', 'Russia_4', 'Iran_2', 'Iran_5', 'Venezuela_1', 'Spain', 'China_2', 'Iran_4', 'Iran_3', 'Russia_5', 'Russia_2', 'Cuba', 'UAE']
task_name = 'nll_with_label'

model_name = 'sahp'
lr = 0.002
batch_size = 32
nts = 20000
train_procedure_config = None
train_dataloader_config = 'sahp_dl.yml'
train_model_config = 'sahp.yml'

evaluate_procedure_config = None
evaluate_dataloader_config = 'sahp_ldl.yml'
evaluate_task_config = None

model_dir = f'results_{model_name}_lr{lr}_bs{batch_size}_nts{nts}_{train_procedure_config}_{train_dataloader_config}_{train_model_config}'
task_conf_dir = f'{evaluate_procedure_config}_{evaluate_dataloader_config}_{evaluate_task_config}'

def load_from_pkl(filepath, compression = None):
    import importlib
    import pathlib
    import pickle as pkl

    dict_compression_algorithms = {
        # Is it a good choice?
        'lzma': importlib.import_module('lzma').open,
        'bz2': importlib.import_module('bz2').open,
        'gz': importlib.import_module('gzip').open
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

    f = selected_open_function(filepath, 'rb')
    data = pkl.load(f)
    f.close()

    return data

def save_fig(fig, file_location, file_name):
    fig.savefig(Path(file_location, file_name), bbox_inches = "tight")
    fig.clear()
    plt.close(fig = fig)
    del fig
    gc.collect()

def write_to_txt(strings, filepath):
    from pathlib import Path

    f = Path(filepath).open("w")

    if isinstance(strings, list):
        f.writelines(strings)
    else:
        f.write(strings)

    f.close()

    return 0

label_translator = {0: 'Abnormal', 1: 'Normal'}

results = None

dataset_files = ['train_nll.pkl.bz2', 'evaluation_nll.pkl.bz2', 'test_nll.pkl.bz2']
for sampled_dataset in dataset_names:
    print(f'Processing {sampled_dataset}...')
    target_result_dir = root_path / result_dir / procedure_name  / result_index / sampled_dataset / model_dir / task_name / task_conf_dir

    for sampled_dataset_file in dataset_files:
        results = load_from_pkl(target_result_dir / sampled_dataset_file)
        results = results["categorized_nll"]

        fig, ax = plt.subplots()
        # Prepare data for plotting
        plot_data = []
        for label, values in results.items():
            label_name = label_translator.get(label, label)
            for value in values:
                plot_data.append({'label': label_name, 'value': value})

        df = pd.DataFrame(plot_data)

        # Plot
        sns.kdeplot(data=df, x='value', hue='label', fill=True, common_norm=False, warn_singular=False, ax=ax)
        ax.set_title(f'Distribution of NLL for {sampled_dataset}')
        ax.set_xlabel('Negative Log Likelihood')

        if not Path(f"./results/{sampled_dataset}").exists():
            Path(f"./results/{sampled_dataset}").mkdir(parents=True)

        save_fig(fig, f"./results/{sampled_dataset}", f"{sampled_dataset_file}.pdf")

        # KS test: if p(results[1]) = p(results[0])
        # Distributions are different.
        statistic, pvalue = ks_2samp(results[1], results[0], method='asymp')
        write_to_txt([f"statistic: {statistic}\n", f"pvalue: {pvalue}\n"], f"./results/{sampled_dataset}/ks_two_sided_test_{sampled_dataset_file}.txt")

        # KS test: if F(results[1]) > F(results[0])
        # Model performs better on normal data.
        statistic, pvalue = ks_2samp(results[1], results[0], alternative="greater", method='asymp')
        write_to_txt([f"statistic: {statistic}\n", f"pvalue: {pvalue}\n"], f"./results/{sampled_dataset}/ks_greater_{sampled_dataset_file}.txt")

        # KS test requires large samples to reject the NULL.
        # We try Anderson-Darling test
        valid_results = all(len(group) >= 2 for group in results.values())
        if valid_results:
            statistic, quantiles, pvalue = anderson_ksamp([results[1], results[0]], method=PermutationMethod())
            write_to_txt([f"statistic: {statistic}\n", f"pvalue: {pvalue}\n"], f"./results/{sampled_dataset}/ad_test_{sampled_dataset_file}.txt")
        else:
            write_to_txt("No enough data for AD test.", f"./results/{sampled_dataset}/ad_test_{sampled_dataset_file}.txt")

