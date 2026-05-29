import os


def possible_checkpoint_detect(opt, idx):
    # We check if the checkpoint and related checkpoint.csv exist.
    # If checkpoint.csv exists, the training process should successfully complete, so checkpoint should exist.
    # If only the checkpoint, we might meet a runtime error during training.
    # We count runs that leaves a legit checkpoint.

    if opt.model_index is not None:
        return opt.model_index

    folder_name = 'model_' + idx

    print(folder_name)

    # Scan valid folders.
    tmp_path = opt.root_path / 'model' / opt.procedure
    files = os.scandir(tmp_path)
    possible_valid_dir_names = [int(dir_item.name) for dir_item in filter(lambda x: not x.is_file() and x.name.isdigit(), files)]
    valid_dir_indexes = []

    for possible_valid_dir_name in possible_valid_dir_names:
        possible_checkpoint = tmp_path / str(possible_valid_dir_name) / (opt.training_dataset_name if opt.training_dataset_name is not None else opt.dataset_name) / folder_name / 'checkpoint.chkpt'
        # possible_checkpoint_log = os.path.join(tmp_path, str(possible_valid_dir_name), opt.dataset_name, folder_name, 'checkpoint.csv')
        if possible_checkpoint.exists():
            valid_dir_indexes.append(possible_valid_dir_name)

    return sorted(valid_dir_indexes)
