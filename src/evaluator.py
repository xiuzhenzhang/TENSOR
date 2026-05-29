import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Self

from src.toolbox.dataloader import prepare_dataloaders
from src.toolbox.evaluation import basic_evaluation, basic_evaluation_loop, load_checkpoint, possible_checkpoint_detect
from src.toolbox.misc import argument_check, get_logger, print_args, read_yaml

logger = get_logger(__name__)


class Evaluator:
    def __init__(self: Self, opt: argparse.ArgumentParser, procedure: str) -> Self:
        """Create an evaluator

        Args:
            self (Self): the evaluator
            opt (argparse.ArgumentParser): the parser which stores all arguments.
            procedure (str): The name of the procedure

        Returns:
            Self: the created evaluator
        """
        self.opt = opt
        self.opt.replace_index = [""] if opt.replace else possible_checkpoint_detect(opt, opt.model_identifier)
        logger.info(f"Available replace_index: {self.opt.replace_index}.")

        # load the model.
        self.get_model = getattr(procedure, "get_model")
        self.get_dataloader = getattr(procedure, "get_dataloader")
        self.task_dict = getattr(procedure, "get_evaluation_funcs")

        # ========= Restore Model from the checkpoint =========
        self.checkpoint_folder = "model_" + opt.model_identifier
        self.results_folder = "results_" + opt.model_identifier

    def work(self: Self) -> int:
        """where the evaluator evaluates the model.

        Args:
            self (Self): the evaluator

        Raises:
            logger.exception: Errors out when an exception is raised.
        """
        # ========= Load Dataset =========
        if self.opt.data_path:
            self.raw_data = prepare_dataloaders(self.opt, self.get_dataloader)
        else:
            raise logger.exception("Wrong input data path.")

        procedure_param = read_yaml(self.opt.abs_procedure_config) if self.opt.abs_procedure_config else {}
        self.opt.procedure_param = procedure_param
        logger.info(f"The hyperparameters for all tasks under procedure {self.opt.procedure} are {procedure_param}")

        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        logger.info(f"The input model hyperparameters are {model_param}.")
        self.model_class = self.get_model(self.opt)
        model = self.model_class(training=False, device=self.opt.device, opt=self.opt, **model_param, **procedure_param)
        self.opt.__dict__.update(model_param)
        self.opt.__dict__.update(procedure_param)

        if len(self.opt.replace_index) == 0:
            logger.warning("The evaluation exited because NO checkpoint has been found.")
            logger.warning("Perhaps, you have forgot the --replace in your script.")

        task_param = read_yaml(self.opt.abs_task_config) if self.opt.abs_task_config else {}
        logger.info(f"The input task settings are {task_param}.")
        self.opt.__dict__.update(task_param)

        for index in self.opt.replace_index:
            # locate where checkpoints are stored.
            self.opt.checkpoint_folder = Path(
                self.opt.checkpoint_of_this_procedure,
                str(index),
                self.opt.training_dataset_name if self.opt.training_dataset_name is not None else self.opt.dataset_name,
                self.checkpoint_folder,
            )
            # where figures, records are stored.
            self.opt.store_dir = Path(
                self.opt.results_of_this_procedure,
                str(index),
                self.opt.dataset_name,
                self.results_folder,
                self.opt.task_name,
                self.opt.task_identifier,
            )
            logger.info(f"We will load the model checkpoint in {self.opt.checkpoint_folder}.")
            logger.info(f"Results will be stored in {self.opt.store_dir}.")

            # Here, we need to restore the model weights from the checkpoint
            self.model = load_checkpoint(
                logger, self.opt.checkpoint_folder / "checkpoint.chkpt", model, device=self.opt.device, compile=self.opt.compile
            )
            logger.info(print_args(self.opt, "Evaluation Info"))

            # Fix module behaviours during evaluation.
            self.model.eval()
            self.task()

        self.finish_task()

    def finish_task(self: Self) -> int:
        """Housekeeping after the training finished.

        Args:
            self (Self): The trainer.

        Returns:
            int: 0
        """
        logger.info(f"Task {self.opt.task_name} finished!")
        return 0

    def task(self: Self) -> None:
        """Where we evaluate the model.

        Args:
            self (Self): the evaluator.
        """
        if self.opt.task_name in self.task_dict:
            return self.evaluation_on_entire_dataset(self.task_dict[self.opt.task_name])
        return self.evaluation_per_seq()

    def evaluation_per_seq(self: Self) -> None:
        """evaluate the model on several picked sequences.

        Args:
            self (Self): the evaluator.
        """
        argument_check(self.opt, **{"num_data_samples": int})

        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.training_data_name is not None:
            for idx, train_data in enumerate(self.raw_data["training"]):
                basic_evaluation(self.model, train_data, "train", batch_idx=idx, opt=self.opt)
                if idx >= self.opt.num_data_samples - 1:
                    break

        if self.opt.evaluate_data_name is not None:
            for idx, evaluation_data in enumerate(self.raw_data["evaluation"]):
                basic_evaluation(self.model, evaluation_data, "evaluation", batch_idx=idx, opt=self.opt)
                if idx >= self.opt.num_data_samples - 1:
                    break

        if self.opt.test_data_name is not None:
            for idx, test_data in enumerate(self.raw_data["test"]):
                basic_evaluation(self.model, test_data, "test", batch_idx=idx, opt=self.opt)
                if idx >= self.opt.num_data_samples - 1:
                    break

    def evaluation_on_entire_dataset(self: Self, evaluation_func: dict | list | Callable) -> None:
        """evaluate the model on the entire dataset.

        Args:
            self (Self): the evaluator
            evaluation_func (Union[dict, list, Callable]): the function used to postprocess the evaluation result.

        Raises:
            Exception: raise Exceptions when evaluation_func is unexpected.
        """
        if isinstance(evaluation_func, list):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                basic_evaluation_loop(
                    self.model, self.raw_data["training"], "train", opt=self.opt, early_offload=False, *evaluation_func
                )
            if self.opt.evaluate_data_name is not None:
                basic_evaluation_loop(
                    self.model,
                    self.raw_data["evaluation"],
                    "evaluation",
                    opt=self.opt,
                    early_offload=False,
                    *evaluation_func,
                )

            if self.opt.test_data_name is not None:
                basic_evaluation_loop(
                    self.model, self.raw_data["test"], "test", opt=self.opt, early_offload=True, *evaluation_func
                )
        elif isinstance(evaluation_func, dict):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                basic_evaluation_loop(
                    self.model, self.raw_data["training"], "train", opt=self.opt, early_offload=False, **evaluation_func
                )

            if self.opt.evaluate_data_name is not None:
                basic_evaluation_loop(
                    self.model,
                    self.raw_data["evaluation"],
                    "evaluation",
                    opt=self.opt,
                    early_offload=False,
                    **evaluation_func,
                )

            if self.opt.test_data_name is not None:
                basic_evaluation_loop(
                    self.model, self.raw_data["test"], "test", opt=self.opt, early_offload=True, **evaluation_func
                )
        elif isinstance(evaluation_func, Callable):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                evaluation_func(self.model, self.raw_data["training"], "train", opt=self.opt, early_offload=False)

            if self.opt.evaluate_data_name is not None:
                evaluation_func(
                    self.model, self.raw_data["evaluation"], "evaluation", opt=self.opt, early_offload=False
                )

            if self.opt.test_data_name is not None:
                evaluation_func(self.model, self.raw_data["test"], "test", opt=self.opt, early_offload=True)
        else:
            raise Exception("Unknown evaluation func!")
