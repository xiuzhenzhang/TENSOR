import argparse
from typing import Self

from src.TPP.utils import suffix
from src.trainer_arguments import BasicTrainerArguments


class TrainerArguments(BasicTrainerArguments):
    def __init__(self: Self, parser: argparse.ArgumentParser, root_path: str) -> Self:
        """Inheriting the basic trainer argument to create the trainer argument class for the TPP procedure.

        Args:
            self (Self): the trainer argument
            parser (argparse.ArgumentParser): the item storing all arguments
            root_path (str): the root dir of this project

        Returns:
            Self: the trainer argument
        """
        super().__init__(parser)
        self.root_path = root_path

        # self identification.
        self.parser.add_argument("--procedure", type=str, default="TPP", help=argparse.SUPPRESS)
        self.parser.add_argument(
            "--displayed_procedure_name", type=str, default="Temporal Point Process", help=argparse.SUPPRESS
        )
        self.parser.add_argument("--required_worker", type=str, default="Trainer", help=argparse.SUPPRESS)
        self.parser.add_argument(
            "--displayed_task_category", type=str, default="Model Training", help=argparse.SUPPRESS
        )


def Trainer_postprocess(opt: argparse.Namespace, root_path: str) -> argparse.Namespace:
    """postprocess the trainer arguments.

    Args:
        opt (argparse.Namespace): the original argument dict
        root_path (str): the root dir of this project

    Returns:
        argparse.Namespace: the processed argument dict
    """
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step
        opt.n_evaluation_steps *= opt.agg_update_step
        opt.n_report_steps *= opt.agg_update_step
        opt.n_warmup_steps *= opt.agg_update_step

    opt.abs_model_config = \
        root_path / "config" / opt.procedure / opt.model_name / opt.model_config if opt.model_config else None

    opt.model_config = opt.abs_model_config.name if opt.model_config else None

    opt.optim_config = root_path / "config" / opt.procedure / opt.optim_config

    opt.abs_dataloader_config = \
        root_path / "config" / opt.procedure / opt.model_name / opt.dataloader_config if opt.dataloader_config else None

    opt.dataloader_config = opt.abs_dataloader_config.name if opt.dataloader_config else None

    opt.abs_procedure_config = \
        root_path / "config" / opt.procedure / opt.procedure_config if opt.procedure_config else None

    opt.procedure_config = opt.abs_procedure_config.name if opt.procedure_config else None

    opt.data_path = root_path / "data" / opt.procedure / opt.dataset_name
    opt.model_identifier = suffix(
        opt,
        "model_name",
        "lr",
        "training_batch_size",
        "n_training_steps",
        "procedure_config",
        "dataloader_config",
        "model_config",
    )

    return opt
