import argparse
import os
from pathlib import Path
from typing import Any, Self

import pandas as pd
import torch
import torch._dynamo
from tqdm import tqdm

from src.toolbox.dataloader import prepare_dataloaders
from src.toolbox.evaluation import get_evaluation_results
from src.toolbox.list_operation import list_add, list_div
from src.toolbox.metrics import Metric
from src.toolbox.misc import (
    conditional_compile_func,
    cycle,
    get_logger,
    mkdir_if_not_exist,
    only_keep_data,
    pack_one_value_to_dict,
    print_args,
    read_yaml,
    write_yaml,
)
from src.toolbox.optimizer import (
    generate_optimizer_scheduler,
    get_lr,
    state_dict,
    step_and_update_lr,
    zero_grad,
)
from src.toolbox.training import print_performances, replace_check

logger = get_logger(__name__)


class Trainer:
    def __init__(self: Self, opt: argparse.ArgumentParser, procedure: str) -> Self:
        """Spawn a Trainer.

        Args:
            self (Self): The trainer
            opt (argparse.ArgumentParser): The parser which stores all arguments
            procedure (str): The name of the procedure
        """
        # We use pd.DataFrame to record training records.
        self.df_records = {
            "training": None,
            "evaluation": None,
            "test": None,
            "checkpoint": None,
        }

        # Store required initial information.
        self.opt = opt

        # Insert the model index if needed.
        continue_running, replace_index = \
            replace_check(
                self.opt,
                log=["checkpoint_record.csv", "evaluation_record.csv", "test_record.csv", "training_record.csv"],
                model="model_card.yml",
        )

        if not continue_running:
            logger.warning("The files are already here! No training is needed. Exiting now.")
            os._exit(0)

        self.opt.log = Path(
            self.opt.root_path,
            "log",
            self.opt.procedure,
            replace_index,
            self.opt.dataset_name,
        )
        self.opt.save_model = Path(
            self.opt.root_path,
            "model",
            self.opt.procedure,
            replace_index,
            self.opt.dataset_name,
        )

        # Load the entry of the model and dataloader.
        self.get_model = getattr(procedure, "get_model")
        self.get_dataloader = getattr(procedure, "get_dataloader")

        # Directory preparation.
        # Create log and model-saving dirs if they are not present.
        self.output_checkpoint_folder = "model_" + self.opt.model_identifier
        self.log_folder = "log_" + self.opt.model_identifier
        self.checkpoint_saved_steps = 0

    def get_procedure_monitor_dict(self: Self, additional_info: dict[str, dict] = {}) -> dict[str, dict]:
        """Pack all metric values into a dict for reporting.

        Args:
            self (Self): The Trainer
            additional_info (dict[str, dict], optional): Additional metrics provided by the model. Defaults to {}.

        Returns:
            dict[str, dict]: The packed values, ready to be parsed and reported.
        """
        monitored_info = {
            "lr": pack_one_value_to_dict(get_lr(self.optimizer), "8.5f"),
            "tensor_memory_consumption": pack_one_value_to_dict(
                torch.cuda.memory_allocated(self.opt.device) / 1024 / 1024 if self.opt.cuda else 0,
                "5f",
                "MiB",
            ),
            "reserved_memory": pack_one_value_to_dict(
                torch.cuda.memory_reserved(self.opt.device) / 1024 / 1024 if self.opt.cuda else 0,
                "5f",
                "MiB",
            ),
        }
        for key, value in additional_info.items():
            monitored_info[key] = value

        return monitored_info

    def work(self: Self) -> None:
        """Where the trainer trains the model.

        Args:
            self (Self): The trainer

        Raises:
            logger.exception: Errors out when an exception is raised.
        """
        # For unknown reason the self.opt.log and self.opt.save_model is empty or None.
        # This is undesired.
        if not self.opt.log and not self.opt.save_model:
            logger.exception("No model or log save path. Usually this shouldn't happen. Please check you environment.")
        mkdir_if_not_exist(self.opt.save_model / self.output_checkpoint_folder)
        mkdir_if_not_exist(self.opt.log / self.log_folder)

        logger.warning("Loading Dataset...")
        if self.opt.data_path:
            self.raw_data = prepare_dataloaders(self.opt, self.get_dataloader)
            self.opt.training_size = len(self.raw_data["training"])
        else:
            raise logger.exception("Wrong input data path.")

        logger.warning("Loading Model...")
        procedure_param = read_yaml(self.opt.abs_procedure_config) if self.opt.abs_procedure_config else {}
        self.opt.procedure_param = procedure_param
        logger.info(f"The hyperparameters for all tasks under procedure {self.opt.procedure} are {procedure_param}")

        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.opt.model_params = model_param
        logger.info(f"The input model hyperparameters are {model_param}")

        # We load the required model by get_model()
        self.model_class = self.get_model(self.opt)
        self.model = self.model_class(training=True, device=self.opt.device, opt=self.opt, **model_param, **procedure_param)
        self.opt.__dict__.update(model_param)
        self.opt.__dict__.update(procedure_param)

        # Metric checker for choosing the best model during training.
        self.metric_checker = Metric(
            self.model.metric_number,
            getattr(self.model, "smaller_is_better", None),
        )
        self.format_dict_length = self.model.format_dict_length
        self.report_sum = [0] * self.format_dict_length

        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in self.model.parameters())
        self.opt.trainable_parameters = trainable_parameters
        self.opt.epoch = self.opt.n_training_steps / self.opt.training_size
        logger.info(print_args(self.opt, "Training Info"))
        logger.info(f"For someone who needs the number of training epoches, the number is {self.opt.epoch:5.5f}")
        logger.info(
            f"The number of trainable model parameters is {self.opt.trainable_parameters} out of {total_parameters}."
        )

        # Due to the complexity of learning rate scheduler, the scheduler is fixed.
        # If you want to use another learning rate scheduler, plz modify it in src.optim.
        self.optimizer, self.scheduler = generate_optimizer_scheduler(self.opt, self.model)
        self.step_and_update_lr = conditional_compile_func(step_and_update_lr, self.opt.compile, self.opt.compile_backend, fullgraph=False)

        self.task()

    def task(self: Self) -> None:
        """Where we do the training and evaluation loop

        Args:
            self (Self): The trainer.

        Returns:
            int: returns 0 when we finish.
        """
        # Setting up file loggers and a wandb online logger.
        if self.opt.wandb:
            import wandb

            wandb.require("core")
            wandb.init(
                project=f"{self.opt.displayed_procedure_name} {self.opt.displayed_task_category}",
                config=vars(self.opt),
                group=self.opt.dataset_name,
                name="-".join(
                    [
                        self.opt.model_name,
                        str(self.opt.model_config),
                        self.opt.dataset_name,
                        str(self.opt.dataloader_config),
                    ]
                ),
                dir=str(self.opt.log / self.log_folder),
                resume="never",
                notes=f"Training {self.opt.model_name} with config file {str(self.opt.model_config)} on dataset {self.opt.dataset_name}.",
            )
            wandb.watch(self.model, log="all", log_freq=self.opt.n_report_steps, log_graph=True)

        desc = "  - (Training)   "
        step_range = range(1, self.opt.n_training_steps + 1)
        training_iter = cycle(self.raw_data["training"])
        zero_grad(self.optimizer)

        # Start training.
        self.evaluation_report(0)
        # Avoid crash.
        # torch._dynamo.reset()
        for current_step in tqdm(step_range, desc=desc, leave=False):
            data = next(training_iter)

            step_result = self.model.train_step(data)
            if current_step % self.opt.agg_update_step == 0:
                if self.opt.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
                self.step_and_update_lr(self.optimizer, self.scheduler)
                zero_grad(self.optimizer)

            self.report_sum = list_add(self.report_sum, step_result)

            # A short report about training.
            if current_step % self.opt.n_report_steps == 0:
                self.report_sum = list_div(self.report_sum, self.opt.n_report_steps)
                self.train_report(current_step)

            # A short report about evaluation and testing.
            if current_step % self.opt.n_evaluation_steps == 0:
                self.evaluation_report(current_step)

        self.finish_task()
        logger.warning("Training finished!")

    def finish_task(self: Self) -> int:
        """Housekeeping after the training finished.

        Args:
            self (Self): The trainer.

        Returns:
            int: 0
        """
        for key, value in self.df_records.items():
            if value is None:
                logger.warning(f"You require us to track the {key} process, but nothing is recorded!")
                continue

            log_filepath = str(self.opt.log / self.log_folder / f"{key}_record.csv")

            logger.info(f"Logs of {key} process are stored in {log_filepath}.")
            df_value = pd.DataFrame.from_dict(value)
            df_value.to_csv(log_filepath, index=False)

        # Write hyperparameters into the model dir.
        hyperparameters = vars(self.opt)
        hyperparameters["device"] = hyperparameters["device"].type
        write_yaml(
            {**hyperparameters, "checkpoint saved at": self.checkpoint_saved_steps},
            self.opt.save_model / self.output_checkpoint_folder,
            "model_card.yml",
        )

        if self.opt.wandb:
            import wandb

            wandb.finish()

        return 0

    def train_report(self: Self, current_step: int) -> None:
        """Report metrics of the training process at current_step.

        Args:
            self (Self): The trainer
            current_step (int): The training step when we do the report.
        """
        logger.warning(f"Brief training status report at step {current_step}.")
        report_sum = self.model.postprocess(self.report_sum, procedure="training")

        log_print_format_dict = self.model.log_print_format(report_sum, procedure="training")
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        plain_training_results = only_keep_data(log_print_format_dict)

        log_print_format_dict.update(procedure_monitor_dict)
        print_performances(logger=logger, procedure="training", data_dict=log_print_format_dict)

        self.transform_report_sum_into_recording_df(
            procedure="training", current_step=current_step, data=plain_training_results
        )
        if self.opt.wandb:
            import wandb

            wandb.log({"training": plain_training_results}, commit=False, step=current_step)
            wandb.log({"lr": get_lr(self.optimizer)}, step=current_step)

        self.report_sum = [0] * self.format_dict_length

    def evaluation(self: Self, dataset_name: str, current_step: int) -> dict[str, Any]:
        """Evaluate the current model at current_step on one dataset.

        Args:
            self (Self): The trainer
            dataset_name (str): The name of the dataset, either evaluation or test.
            current_step (int): At which step we do the evaluation.

        Returns:
            dict[str, Any]: The evaluation results.
        """
        evaluation_results = get_evaluation_results(
            self.raw_data[dataset_name],
            self.model,
            output_length=self.format_dict_length,
            desc=f"  - ({dataset_name})   ",
        )
        # dict_flops = {'FLOPS': {'data': evaluation_results['flops'] / 1000**4, 'num_format': '8.5f', 'suffix': 'TFlops'}}
        report = self.model.postprocess(evaluation_results["results"], procedure=dataset_name)

        log_print_format_dict = self.model.log_print_format(report, procedure=dataset_name)
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        plain_evaluation_results = only_keep_data(log_print_format_dict)

        log_print_format_dict.update(procedure_monitor_dict)
        print_performances(logger=logger, procedure=dataset_name, data_dict=log_print_format_dict)

        self.transform_report_sum_into_recording_df(
            procedure=dataset_name,
            current_step=current_step,
            data=plain_evaluation_results,
        )
        if self.opt.wandb:
            import wandb

            wandb.log({dataset_name: plain_evaluation_results}, step=current_step)

        return plain_evaluation_results

    def evaluation_report(self: Self, current_step: int) -> None:
        """Model evaluation on evaluation and test datasets then save the model if necessary.

        Args:
            self (Self): The trainer.
            current_step (int): At which step we do the evaluation.
        """
        logger.warning(f"Model evaluation and checkpoint saving at step {current_step}.")

        # Evaluation and checkpoint saving.
        evaluation_results = self.evaluation("evaluation", current_step)
        test_results = self.evaluation("test", current_step)
        self.save(current_step, evaluation_results, test_results)

    def should_we_save_model(
        self: Self,
        mode: str,
        metric_data: dict[str, Any],
        current_step: int,
        warmup: int,
    ) -> tuple[bool, str]:
        """Decide if we should save the model following the model saving strategy and other information.

        Args:
            self (Self): The trainer.
            mode (str): The model saving strategy.
            metric_data (dict[str, Any]): Evaluation result picked by self.model.choose_metric() from the evaluation result computed on the evaluation and test dataset.
            current_step (int): Current training step.
            warmup (int): The number of warmup steps. No checkpoint is saved during warmup.

        Returns:
            tuple[bool, str]: should we save the model, and the file name of the saved checkpoint.
        """

        def checker_for_mode_all(metric_data: dict[str, Any]) -> bool:
            return True

        def checker_for_mode_bests(metric_data: dict[str, Any]) -> bool:
            return self.metric_checker.compare(metric_data.values())

        dict_save_model_checkers_and_checkpoint_names = {
            "all": [checker_for_mode_all, f"checkpoint_at_step_{current_step}.chkpt"],
            "best": [checker_for_mode_bests, "checkpoint.chkpt"],
            "last": [checker_for_mode_all, "checkpoint.chkpt"],
        }

        save_should_or_not = False
        checker, checkpoint_name = dict_save_model_checkers_and_checkpoint_names[mode]
        if current_step >= warmup and checker(metric_data):
            save_should_or_not = True

        return save_should_or_not, checkpoint_name

    def save(
        self: Self,
        current_step: int,
        evaluation_results: dict[str, Any],
        test_results: dict[str, Any],
    ) -> None:
        """Save the model if needed.

        Args:
            self (Self): The trainer.
            current_step (int): Current training step.
            evaluation_results (dict[str, Any]): The evaluation result computed on the evaluation dataset.
            test_results (dict[str, Any]): The evaluation result computed on the test dataset.
        """
        metric_values, metric_names = self.model.choose_metric(evaluation_results, test_results)
        if len(metric_values) != len(metric_names):
            raise ValueError("metric_values mismatches metric_names!")
        metric_data = dict(zip(metric_names, metric_values))

        should_save_or_not, checkpoint_name = self.should_we_save_model(
            mode=self.opt.save_mode,
            metric_data=metric_data,
            current_step=current_step,
            warmup=self.opt.n_warmup_steps,
        )

        if should_save_or_not:
            # Save the model.
            model_name = self.opt.save_model / self.output_checkpoint_folder / checkpoint_name
            torch.save(self.model.state_dict(), model_name)
            # Save the optimizer.
            optimizer_name = self.opt.save_model / self.output_checkpoint_folder / "optimizer.chkpt"
            torch.save(state_dict(self.optimizer, self.scheduler), optimizer_name)
            self.transform_report_sum_into_recording_df(
                procedure="checkpoint", current_step=current_step, data=metric_data
            )
            logger.warning(f"----> We stored the model in {checkpoint_name} at step {current_step}. <----")

    def transform_report_sum_into_recording_df(self: Self, procedure: str, current_step: int, data: dict[str, Any]):
        """Append the computed metrics into the metric database.

        Args:
            self (Self): The trainer.
            procedure (str): The name of the database.
            current_step (int): Current training step.
            data (dict[str, Any]): The computed metrics.
        """
        new_df_perline_dict = {"current_step": current_step}
        new_df_perline_dict.update(data)

        if self.df_records[procedure] is None:
            empty_execution_log_dict = {}
            for key in new_df_perline_dict:
                empty_execution_log_dict[key] = []
            self.df_records[procedure] = empty_execution_log_dict

        for key in self.df_records[procedure]:
            self.df_records[procedure][key].append(new_df_perline_dict[key])
