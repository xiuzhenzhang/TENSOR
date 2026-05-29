import argparse
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.toolbox.dataloader.prefetch_loader import PrefetchLoader
from src.toolbox.dataloader.utils import check_exist, seed_worker
from src.toolbox.misc import get_logger, read_yaml

logger = get_logger(__name__)


def prepare_dataloaders(opt: argparse.ArgumentParser, find_dataset: Callable) -> dict[str, DataLoader]:
    """Create the required dataloader against custom dataloader settings.

    Args:
        opt (argparse.ArgumentParser): the arguments
        find_dataset (Callable): a function responsible for looking for the data_loader and how we load the dataset.

    Returns:
        dict[str, DataLoader]: the training, evaluation, and test dataset.
    """
    available_file_names = [item.name for item in Path(opt.data_path).glob(f"*.{opt.dataset_type}")]
    logger.info(f"Dataset path: {opt.data_path}.")

    # find if required dataset files exists.
    file_names = check_exist(
        available_file_names,
        opt.dataset_type,
        training=opt.training_data_name,
        evaluation=opt.evaluate_data_name,
        test=opt.test_data_name,
    )

    if len(file_names) == 0:
        logger.exception(f"No available dataset file in {opt.data_path}!")
    else:
        logger.info(
            f"We are going to read {len(file_names)} files in {opt.data_path}.\n{''.join([f'{key} dataset: {item}\n' for key, item in file_names.items()])}Is that right?"
        )

    dataloader_config_dict = read_yaml(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}

    # Read in the used_dataloader_config
    used_dataloader_config_dict = {}
    try:
        if opt.combine_used_and_current_dataloader_config:
            used_dataloader_config_dict = (
                read_yaml(opt.abs_used_dataloader_config) if opt.abs_used_dataloader_config else {}
            )
    except AttributeError:
        logger.warning(
            "combine_used_and_current_dataloader_config unset! Possibly we are training a model. We will ignore it."
        )
    # apply used_dataloader_config to current dataloader config if opt.combine_used_and_current_dataloader_config is True
    dataloader_config_dict.update(used_dataloader_config_dict)

    if dataloader_config_dict == {}:
        logger.info("No custom dataloader settings! We will use the default dataloader settings.")
    else:
        logger.info(f"Custom dataloader settings are loaded from this config file {opt.abs_dataloader_config}.")
        logger.info(f"Custom dataloader settings are: {dataloader_config_dict}.")

    dataset, read_data = find_dataset(opt)

    # Now, dataset_card.yml is mandatory for every dataset.
    # This YAML file should contain useful information about this dataset, like the number of classes it has.
    opt.dataloader_config_dict = dataloader_config_dict
    opt.info_dict = read_yaml(Path(opt.data_path, "dataset_card.yml"))

    # ========= Preparing dataloaders =========#
    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed)

    if getattr(opt, "training_data_name") is not None:
        train_dataset = dataset(
            read_data(opt.data_path, file_names["training"]),
            property_dict=opt.info_dict,
            device=opt.device,
            **dataloader_config_dict,
        )
        train_iterator = PrefetchLoader(
            DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=opt.training_batch_size,
                collate_fn=train_dataset.data_collator,
                num_workers=opt.n_worker,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=True,
            ),
            device=opt.device,
        )
    if getattr(opt, "evaluate_data_name", True) is not None:
        evaluate_dataset = dataset(
            read_data(opt.data_path, file_names["evaluation"]),
            property_dict=opt.info_dict,
            device=opt.device,
            **dataloader_config_dict,
        )
        evaluation_iterator = PrefetchLoader(
            DataLoader(
                evaluate_dataset,
                batch_size=opt.evaluation_batch_size,
                collate_fn=evaluate_dataset.data_collator,
                num_workers=opt.n_worker,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=True,
            ),
            device=opt.device,
        )
    if getattr(opt, "test_data_name", True) is not None:
        test_dataset = dataset(
            read_data(opt.data_path, file_names["test"]),
            property_dict=opt.info_dict,
            device=opt.device,
            **dataloader_config_dict,
        )
        test_iterator = PrefetchLoader(
            DataLoader(
                test_dataset,
                batch_size=opt.evaluation_batch_size,
                collate_fn=test_dataset.data_collator,
                num_workers=opt.n_worker,
                worker_init_fn=seed_worker,
                generator=g,
                pin_memory=True,
            ),
            device=opt.device,
        )

    return {
        "training": train_iterator,
        "evaluation": evaluation_iterator,
        "test": test_iterator,
    }
