import os
from pathlib import Path

root_path = Path(__file__).parent.resolve()

main_procedure_translator = {
    # Temporal Point Process
    "TPP_train": "TPP",
    "TPP_evaluate": "TPP",
    # Noted Temporal Point Process
    "NTPP_train": "NTPP",
    "NTPP_evaluate": "NTPP",
    # Long-horizon Temporal Point Process
    "LH_train": "LH",
    "LH_evaluate": "LH",
    # Missing Data Imputation with Temporal Point Process
    "MDI_train": "MDI",
    "MDI_evaluate": "MDI",
    # Outlier Detection with Temporal Point Process
    "OD_train": "OD",
    "OD_evaluate": "OD",
    # Explanation for MTPP
    "ehd_train": "ehd",
    "ehd_evaluate": "ehd",
    "Transformers": "Transformers",
    "fakenews": "fakenews",
}

sub_procedure_translator = {
    # Temporal Point Process
    "TPP_train": "Trainer",
    "TPP_evaluate": "Evaluator",
    # Noted Temporal Point Process
    "NTPP_train": "Trainer",
    "NTPP_evaluate": "Evaluator",
    # Long-horizon Temporal Point Process
    "LH_train": "Trainer",
    "LH_evaluate": "Evaluator",
    # Missing Data Imputation with Temporal Point Process
    "MDI_train": "Trainer",
    "MDI_evaluate": "Evaluator",
    # Outlier Detection with Temporal Point Process
    "OD_train": "Trainer",
    "OD_evaluate": "Evaluator",
    # Explanation for MTPP
    "ehd_train": "Trainer",
    "ehd_evaluate": "Evaluator",
    "Transformers": "Trainer",
    "fakenews": "Trainer",
}


def environment_var_settings():
    """
    Set up custom environment variables.
    """
    env_dict = {}

    if Path(root_path / "config" / "matplotlibrc").exists():
        env_dict["MATPLOTLIBRC"] = str(root_path / "config" / "matplotlibrc")

    # set up PYTORCH_CUDA_ALLOC_CONF to mitigate GPU memory fragmentation.
    env_dict["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env_dict["TOKENIZERS_PARALLELISM"] = "true"

    # Reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.environ.update(env_dict)


if __name__ == "__main__":
    # We should configure all environment variables here before ```from src import TaskHost```
    # imports everything.
    environment_var_settings()

    # Do NOT move these import codes to the beginning of this file.
    import argparse
    import importlib

    from src import TaskHost

    # Enumerate subparsers from procedure_names
    # we need main_procedure_translator and sub_procedure_translator to translate procedure names
    # into correct argument classes.
    parser = argparse.ArgumentParser()
    procedure_names = [
        # Temporal point process
        ["TPP_train", "Train a (marked) temporal point process model."],
        ["TPP_evaluate", "Evaluate (marked) temporal point process models."],
        # Noted Temporal point process
        ["NTPP_train", ""],
        ["NTPP_evaluate", ""],
        # Explanation for MTPP
        ["ehd_train", ""],
        ["ehd_evaluate", ""],
        # Long-horizon Temporal point process
        ["LH_train", ""],
        ["LH_evaluate", ""],
        # Missing Data Imputation with Temporal Point Process
        ["MDI_train", ""],
        ["MDI_evaluate", ""],
        # Outlier Detection with Temporal Point Process
        ["OD_train", ""],
        ["OD_evaluate", ""],
    ]

    subparsers = parser.add_subparsers()
    for procedure_name, help_info in procedure_names:
        # Fetch the argument class and attach them to the main parser.
        # All source files related to the specific procedure should be in `src`.
        # The folder name should match the name of the procedure so importlib can find and load them.
        # Each procedure should define two classes, TrainerArgument and EvaluatorArgument, each inheriting
        # the BasicTrainerArguments and BasicEvaluatorArguments, to add custom hyperparameters.
        main_procedure = main_procedure_translator[procedure_name]
        sub_procedure_argument_prefix = sub_procedure_translator[procedure_name]

        tmp_parser_hook = subparsers.add_parser(procedure_name, help=f"{help_info}")
        procedure = importlib.import_module("src." + main_procedure)
        argument_class_name = sub_procedure_argument_prefix + "Arguments"
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    # Call TaskHost to start the task.
    agent = TaskHost(parser=parser, root_path=root_path)
    agent.start()
