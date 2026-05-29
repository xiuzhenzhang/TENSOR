import argparse
from typing import Self

from src.evaluator_arguments import BasicEvaluatorArguments
from src.TPP.utils import suffix


class EvaluatorArguments(BasicEvaluatorArguments):
    def __init__(self: Self, parser: argparse.ArgumentParser, root_path: str) -> Self:
        """Inheriting the basic evaluator argument to create the evaluator argument class for the TPP procedure.

        Args:
            self (Self): the evaluator argument
            parser (argparse.ArgumentParser): the item storing all arguments
            root_path (str): the root dir of this project

        Returns:
            Self: the evaluator argument
        """
        super().__init__(parser)
        self.root_path = root_path

        # self identification.
        self.parser.add_argument("--procedure", type=str, default="TPP", help=argparse.SUPPRESS)
        self.parser.add_argument(
            "--displayed_procedure_name", type=str, default="Temporal Point Process", help=argparse.SUPPRESS
        )
        self.parser.add_argument("--required_worker", type=str, default="Evaluator", help=argparse.SUPPRESS)
        self.parser.add_argument(
            "--displayed_task_category", type=str, default="Model Evaluation", help=argparse.SUPPRESS
        )


def Evaluator_postprocess(opt: argparse.Namespace, root_path: str) -> argparse.Namespace:
    """postprocess the evaluator arguments.

    Args:
        opt (argparse.Namespace): the original argument dict
        root_path (str): the root dir of this project

    Returns:
        argparse.Namespace: the processed argument dict
    """
    # Gradient aggergation check
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step

    opt.data_path = root_path / "data" / opt.procedure / opt.dataset_name

    opt.abs_dataloader_config = root_path / "config" / opt.procedure / opt.model_name / opt.dataloader_config if opt.dataloader_config else None
    opt.abs_procedure_config = root_path / "config" / opt.procedure / opt.procedure_config if opt.procedure_config else None

    opt.procedure_config = opt.abs_procedure_config.name if opt.procedure_config else None
    opt.dataloader_config = opt.abs_dataloader_config.name if opt.dataloader_config else None

    opt.abs_model_config = root_path / "config" / opt.procedure / opt.model_name / opt.model_config if opt.model_config else None
    opt.model_config = opt.abs_model_config.name if opt.model_config else None

    opt.abs_task_config = (
        root_path / "config" / opt.procedure / opt.model_name / opt.task_config if opt.task_config else None
    )
    opt.task_config = opt.abs_task_config.name if opt.abs_task_config else None

    opt.abs_used_dataloader_config = root_path / "config" / opt.procedure / opt.model_name / opt.used_dataloader_config if opt.used_dataloader_config else None
    opt.used_dataloader_config = opt.abs_used_dataloader_config.name if opt.used_dataloader_config else None

    opt.checkpoint_of_this_procedure = root_path / "model" / opt.procedure
    opt.results_of_this_procedure = root_path / "results" / opt.procedure
    opt.model_identifier = suffix(
        opt,
        "model_name",
        "lr",
        "used_batch_size",
        "n_training_steps",
        "used_procedure_config",
        "used_dataloader_config",
        "model_config",
    )
    opt.task_identifier = suffix(opt, "procedure_config", "dataloader_config", "task_config")

    return opt
