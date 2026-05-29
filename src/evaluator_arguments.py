import argparse
from typing import Self


class BasicEvaluatorArguments:
    def __init__(self: Self, parser: argparse.ArgumentParser) -> Self:
        """Create a basic argument parser for the evaluator.

        Args:
            self (Self): The evaluator argument.
            parser (argparse.ArgumentParser): The parser.

        Returns:
            Self: The evaluator argument.
        """
        self.parser = parser

        # The Ultimate
        self.parser.add_argument(
            "--no_seed",
            action="store_true",
            help="This argument tells our code to randomly select a seed. You can use this option to explore your model's robustness.",
        )
        self.parser.add_argument("--seed", type=int, default=32, help="Set global random seed.")
        self.parser.add_argument(
            "--cuda",
            action="store_true",
            help="Set it to true if you want to use GPU to accelerate model training.",
        )
        self.parser.add_argument(
            "--cuda_device",
            type=int,
            default=0,
            help="Select which CUDA device you want to use. Default number is 0. This argument does nothing if --cuda is not set.",
        )
        self.parser.add_argument(
            "--replace",
            action="store_true",
            help="True: Replace existing everything, such as logs, model checkpoints, and results with the new one.\n False: Do not replace.",
        )
        self.parser.add_argument(
            "--model_index",
            nargs="+",
            default=None,
            help="Pick the model by its index.",
        )
        self.parser.add_argument(
            "--compile",
            action="store_true",
            help="True: Use torch.compile() to compile models or functions to speed up training and evaluation.\n False: disable torch.compile().",
        )
        self.parser.add_argument(
            "--dtype",
            type=str,
            default="float32",
            help="Train the MTPP model in different precision. Useful when training MTPP on devices fast on lower precision like bfloat16 or float16 but very slow on float32, for example A100.",
        )

        # The number of Dataloader worker
        self.parser.add_argument(
            "--n_worker",
            default=4,
            type=int,
            help="The number of dataloader workers. For most datasets, multiprocessing might speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet 'received 0 items of ancdata' exception.",
        )
        self.parser.add_argument(
            "--sleep",
            default=0,
            type=int,
            help="This task is delayed and will start in the amount of time you have set.",
        )

        # Input data
        self.parser.add_argument(
            "--training_dataset_name",
            type=str,
            default=None,
            help="Name of the used dataset where we train the model on. All datasets should be placed in {root}/data/input. By default, training_dataset_name is dataset_name if training_dataset_name is not provided.",
        )
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="Name of the used dataset. All datasets should be placed in {root}/data/input.",
        )
        self.parser.add_argument(
            "--dataset_type",
            type=str,
            default="pkl.lzma",
            help="File type of the selected dataset.",
        )
        self.parser.add_argument(
            "--dataloader_name",
            default=None,
            help="Name of the used dataloader. All dataloaders are stored in {root}/src/TPP/dataloader.",
        )
        self.parser.add_argument(
            "--dataloader_config",
            type=str,
            default=None,
            help="Relative path to the custom dataloader config file. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{dataloader_config}.",
        )
        self.parser.add_argument(
            "--used_dataloader_config",
            type=str,
            default=None,
            help="The name of dataloader config file used during training. We only need the filename, not the relative path.",
        )
        self.parser.add_argument(
            "--combine_used_and_current_dataloader_config",
            action="store_true",
            help="Combine the settings defined in used_dataloader_config and dataloader_config when set. Settings in dataloader_config will be overwritten by used_dataloader_config if exists in used_dataloader_config.",
        )

        self.parser.add_argument(
            "--training_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--evaluate_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--test_data_name",
            type=str,
            default=None,
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/{main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )

        # Evaluation related hyperparameters
        self.parser.add_argument(
            "--n_training_steps",
            type=int,
            default=10000,
            help="How many steps did we use to train this model?",
        )
        self.parser.add_argument(
            "--agg_update_step",
            type=int,
            default=1,
            help="The number of minibatches between two adjacent optimizer steps.\
                                                                                 The number of practical training steps is agg_update_step * n_training_steps.",
        )

        # Model save and log management
        self.parser.add_argument(
            "--save_mode",
            type=str,
            choices=["all", "best"],
            default="best",
            help="Store all model checkpoints or only store the best one.",
        )

        # Training procedure related hyperparameters
        self.parser.add_argument(
            "-ub",
            "--used_batch_size",
            type=int,
            default=2048,
            help="Batch size used for training the model.",
        )
        self.parser.add_argument(
            "-tb",
            "--training_batch_size",
            type=int,
            default=1,
            help="Batch size used for training set.",
        )
        self.parser.add_argument(
            "-eb",
            "--evaluation_batch_size",
            type=int,
            default=1,
            help="Batch size used for evaluation set.",
        )
        self.parser.add_argument(
            "--used_procedure_config",
            type=str,
            default=None,
            help="Relative path to the custom setting file, in which settings are applied to all tasks under the procedure. The absolute file path is {root}/config/${main_procedure_name}/${procedure_config}",
        )

        # Model-related hyperparameters
        self.parser.add_argument("--model_name", default=None, help="The model name.")
        self.parser.add_argument(
            "--model_config",
            type=str,
            default=None,
            help="Relative path to the custom model config file used for training. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{model_config}.",
        )

        # Optimizer-related hyperparameters
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.1,
            help="The learning rate used when training the model.",
        )

        # Which task you'd like to run and where is the task config file?
        self.parser.add_argument(
            "--procedure_config",
            type=str,
            default=None,
            help="Relative path to the custom setting file, in which settings are applied to all tasks under the procedure. The absolute file path is {root}/config/${main_procedure_name}/${procedure_config}",
        )
        self.parser.add_argument(
            "--task_name",
            type=str,
            help="Define which evaluation task you'd like to start.",
        )
        self.parser.add_argument(
            "--task_config",
            type=str,
            help="Relative path to the custom subtask config file used for training. This absolute file path is {root}/config/{main_procedure_name}/{model_name}/{task_config}.",
        )
