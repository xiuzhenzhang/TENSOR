import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

'''
This file computes the average precision of IO user detection results.
'''

default_figure_kwargs = {
    "font.size": 18,
    "figure.figsize": (8, 4),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times",
    "mathtext.fontset": "dejavusans",
    "text.latex.preamble": "\\usepackage{amsmath}"
}

root_path = Path('/home/undesired/coderepo/workflow_next')
result_dir = 'results'
procedure_name = 'TPP'
dataset_names = ['Qatar', 'Armenia', 'Ecuador', 'Egypt', 'Russia_1', 'Thailand', 'Catalonia', 'China_1', 'Iran_6', 'Iran_1', 'Ghana', 'Russia_4', 'Iran_2', 'Iran_5', 'Venezuela_1', 'Spain', 'China_2', 'Iran_4', 'Iran_3', 'Russia_5', 'Russia_2', 'Cuba', 'UAE']
task_name = 'nll_with_label'
seed=42

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

dataset_files = ['train_nll.pkl.bz2', 'evaluation_nll.pkl.bz2', 'test_nll.pkl.bz2']
for sampled_dataset in dataset_names:
    print(f'Processing {sampled_dataset}...')
    target_result_dir = root_path / result_dir / procedure_name / sampled_dataset / model_dir / task_name / task_conf_dir

    for sampled_dataset_file in dataset_files:
        results = load_from_pkl(target_result_dir / sampled_dataset_file)
        results = results["categorized_nll"]

        # We have to reverse the label here to make roc_auc_score happy.
        y_score, y_label = [], []
        for key, value in results.items():
            for item in value:
                y_score.append(item)
                y_label.append(key)

        ap = average_precision_score(y_true=y_label, y_score=y_score, pos_label=0)
        # What is the performance of random guess?
        rng = np.random.default_rng(seed)
        ap_random = []
        for _ in range(10000):
            random_scores = rng.uniform(low=0.0, high=1.0, size=len(y_score))
            ap_random.append(average_precision_score(y_true=y_label, y_score=random_scores, pos_label=0))
        mean_ap_random = np.mean(ap_random)
        std_ap_random = np.std(ap_random)

        if not Path(f"./results/{sampled_dataset}").exists():
            Path(f"./results/{sampled_dataset}").mkdir(parents=True)

        # The performance of TPP-based classifier.
        write_to_txt([f"Average Precision Score: {ap}\n", f"Random Guess: {mean_ap_random}±{std_ap_random}"], f"./results/{sampled_dataset}/AP_{sampled_dataset_file}.txt")
