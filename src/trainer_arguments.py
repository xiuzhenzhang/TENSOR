import argparse
from typing import Self


class BasicTrainerArguments:
    def __init__(self: Self, parser: argparse.ArgumentParser) -> Self:
        """Create a basic argument parser for the trainer.

        Args:
            self (Self): The trainer argument.
            parser (argparse.ArgumentParser): The parser.

        Returns:
            Self: The trainer argument.
        """
        self.parser = parser
        # The Ultimate
        self.parser.add_argument(
            "--no_seed",
            action="store_true",
            help="This argument tells our code to randomly select a seed. You can use this option to explore your model's robustness.",
        )
        self.parser.add_argument(
            "--seed", type=int, default=32, help="Set global random seed."
        )
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
            type=int,
            default=None,
            help="Needed when replace=False. The log, model checkpoints, and results will be placed in <log/model/results>/model_index/. This value will be ignored when replace=True.",
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
            default=8,
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

        # Datasets.
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            default=None,
            help="Name of the used dataset. All datasets should be placed in {root}/data/${main_procedure_name}.",
        )
        self.parser.add_argument(
            "--dataset_type",
            type=str,
            default="pkl.lzma",
            help="The format of the required dataset.",
        )
        self.parser.add_argument(
            "--dataloader_name",
            default=None,
            help="Name of the used dataloader. All dataloaders are stored in *root*/src/TPP/dataloader.",
        )
        self.parser.add_argument(
            "--dataloader_config",
            type=str,
            default=None,
            help="Relative path to the custom dataloader config file. This absolute file path is {root}/config/${main_procedure_name}/{model_name}/{dataloader_config}.",
        )
        self.parser.add_argument(
            "--training_data_name",
            type=str,
            default="train",
            help="Name of the dataset used for training the model. This file should be placed in {root}/data/${main_procedure_name}/{dataset_name}/{training_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--evaluate_data_name",
            type=str,
            default="evaluate",
            help="Name of the dataset used for evaluating the model. This file should be placed in {root}/data/${main_procedure_name}/{dataset_name}/{evaluate_data_name}.{dataset_type}.",
        )
        self.parser.add_argument(
            "--test_data_name",
            type=str,
            default="test",
            help="Name of the dataset used for testing the model. This file should be placed in {root}/data/${main_procedure_name}/{dataset_name}/{test_data_name}.{dataset_type}.",
        )

        # Training procedure related hyperparameters
        self.parser.add_argument(
            "--n_training_steps",
            type=int,
            default=10000,
            help="Training steps used for training the model.",
        )
        self.parser.add_argument(
            "--n_evaluation_steps",
            type=int,
            default=200,
            help="Evaluate the model on evaluation and test datasets per {n_evaluation_steps} steps.",
        )
        self.parser.add_argument(
            "--n_report_steps",
            type=int,
            default=200,
            help="Report the training metrics per {n_report_steps} steps.",
        )
        self.parser.add_argument(
            "--agg_update_step",
            type=int,
            default=1,
            help="The number of minibatches between two adjacent optimizer steps. The number of practical training steps is \
                                                                            agg_update_step * n_training_steps",
        )
        self.parser.add_argument(
            "--n_warmup_steps",
            type=int,
            default=2000,
            help="The number of warmup steps. We won't store any checkpoints during warmup.",
        )
        self.parser.add_argument(
            "--procedure_config",
            type=str,
            default=None,
            help="Relative path to the custom setting file, in which settings are applied to all tasks under the procedure. The absolute file path is {root}/config/${main_procedure_name}/${procedure_config}",
        )

        # wandb support
        self.parser.add_argument(
            "--wandb",
            action="store_true",
            help="Use wandb to record and visualize the training procedure.",
        )

        # Model save and log management
        self.parser.add_argument(
            "--save_mode",
            type=str,
            choices=["all", "best", "last"],
            default="best",
            help="Store all model checkpoints or only store the best one.",
        )

        # Training procedure related hyperparameters
        self.parser.add_argument(
            "-tb",
            "--training_batch_size",
            type=int,
            default=2048,
            help="Batch size of training data.",
        )
        self.parser.add_argument(
            "-eb",
            "--evaluation_batch_size",
            type=int,
            default=2048,
            help="Batch size of evaluation and test data.",
        )
        self.parser.add_argument(
            "--grad_clip",
            type=float,
            default=0.0,
            help="Clips gradient norm of an iterable of parameters. It only comes info effect when the argument \
                                                                          value is bigger than 0.",
        )

        # Model-related hyperparameters
        self.parser.add_argument("--model_name", default=None, help="The model name.")
        self.parser.add_argument(
            "--model_config",
            type=str,
            default=None,
            help="Relative path to the custom model config file used for training. This absolute file path is {root}/config/{model_name}/{dataset_name}/{model_config}.",
        )

        # Optimizer-related hyperparameters
        self.parser.add_argument(
            "--optim_config",
            type=str,
            default=None,
            help="The config file that contains optimizer and scheduler settings.",
        )
        self.parser.add_argument(
            "--custom_op",
            action="store_true",
            help="Set it to true if you want to use your own optimizer or that from third-party packages.",
        )
        self.parser.add_argument(
            "--op_name",
            type=str,
            default="AdamW",
            help="The name of optimizer. All optimizer hyperparameters are set as default.",
        )
        self.parser.add_argument(
            "--lr_sched",
            action="store_true",
            help="Do you want to use learning rate scheduler? If scheduler is disabled, the warmup settings won't come into effect.",
        )
        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.1,
            help="Input learning rate. The real learning rate could change due to the lr scheduler.",
        )
        self.parser.add_argument("--n_cycles", type=float, default=0.5)
        self.parser.add_argument("--last_epoch", type=int, default=-1)
