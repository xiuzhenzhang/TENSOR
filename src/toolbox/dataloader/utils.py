import random

import numpy as np
import torch


# Referring to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)


def check_exist(file_list: list[str], file_type: str, **file_names) -> dict[str, str]:
    existing_files = {}
    for file_usage, file_name in file_names.items():
        if file_name is None:
            existing_files[file_usage] = None
            continue

        complete_file_name = f'{file_name}.{file_type}'
        if complete_file_name not in file_list:
            raise FileExistsError(f'{file_name} is not found! Found files are: {file_list}.')

        existing_files[file_usage] = complete_file_name

    return existing_files
