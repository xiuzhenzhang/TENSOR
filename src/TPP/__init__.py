"""
TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
"""

from src.TPP.dataloader import get_dataloader
from src.TPP.evaluate_arguments import Evaluator_postprocess, EvaluatorArguments
from src.TPP.evaluate_functions import desc_funcs as get_evaluation_funcs
from src.TPP.model import get_model
from src.TPP.train_arguments import Trainer_postprocess, TrainerArguments
from src.TPP.utils import easy_model_load

__all__ = [
    "get_dataloader",
    "Evaluator_postprocess",
    "EvaluatorArguments",
    "get_evaluation_funcs",
    "get_model",
    "Trainer_postprocess",
    "TrainerArguments",
    "easy_model_load",
]

pytorch_version_warnings = {
    "==1.4.0": [
        """
It is known that several learning rate schedulers shipped by PyTorch 1.4.0 are buggy and fail to run. Please update PyTorch to 1.5.0 or above.
Detailed information is available at https://github.com/pytorch/pytorch/issues/36313
""".replace("\n", ""),
        "stop",
    ],
}
