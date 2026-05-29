import argparse
import importlib
import sys
import time
from pathlib import Path
from time import localtime, strftime
from typing import Any

from src.toolbox.bulk_task_runner import monitor_and_automaticly_run_tasks, parameter_parser
from src.toolbox.misc import get_logger, mkdir_if_not_exist, read_yaml

# Get logger.
logger = get_logger(__name__)
# Define the project root dir.
root_path = Path(__file__).parent.resolve()
logger.info(f"project root is {root_path}.")
logger.info("Please confirm the root_path is correct!")
# Start time of the task.
current_local_time = strftime("%Y-%m-%d %H:%M:%S", localtime())
logger.info(f"Task started at {current_local_time}. ")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--script_type",
    type=str,
    choices=["normal", "previous_failed_tasks"],
    default="normal",
    help="Use this argument to select worker mode.\n \
          normal: In this mode, the script will pick job dict according to the received job_name. Failed tasks will be recorded in {model_name}_previous_failed_tasks.txt. \n \
          previous_failed_tasks: In this mode, this script will read in tasks from parameter_set/{procedure_name}/{model}_previous_failed_tasks.txt and execute these tasks one by one.",
)
parser.add_argument(
    "--procedure_name",
    type=str,
    choices=["TPP", "NTPP", "LH", "OD", "MDI"],
    help="You need this argument to select the proper parameter set.",
)
parser.add_argument(
    "--model",
    type=str,
    help="We use this model name to select correct parameter collection.",
)
parser.add_argument(
    "--job_name",
    nargs="+",
    default=None,
    help="Tell us which job do you want to execute. \
          This argument accepts multiple inputs so you can use '--job_name A B C' to run job A B C one by one. \n \
          None (Default): no job will be executed. \n \
          ALL (special): execute all jobs. \n \
          This argument will be ignored in the previous_failed_tasks mode.",
)
parser.add_argument(
    "--GPU",
    nargs="+",
    default=None,
    help="How many GPU do you want to use? Tell us the ID of available GPUs, \
          or set it to a negative number or None to go CPU-only.",
)
parser.add_argument(
    "--num_task_parallel",
    type=int,
    default=-1,
    help="The number of tasks we should run in parallel. \
          In GPU mode this number should not bigger than the number of available GPUs. \
          The default value, -1, will automatically use all GPUs, one GPU for one task. \
          This argument is mandatory when executing tasks on CPU or submitted through slurm.",
)
parser.add_argument("--slurm", action="store_true", help="Submit tasks through slurm.")
parser.add_argument(
    "--slurm_config",
    type=str,
    help="This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. \
          We will use the default quota if no config is given.",
)
parser.add_argument(
    "--sleep",
    type=int,
    default=0,
    help="This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. \
          We will use the default quota if no config is given.",
)
parser.add_argument(
    "--interpreter",
    type=str,
    nargs="+",
    default="python3",
    help="This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. \
          We will use the default quota if no config is given.",
)
parser.add_argument(
    "--start_from_this_index",
    type=int,
    default=1,
    help="This argument directs the script to start from the work with the given index. Useful during debug.",
)
parser.add_argument(
    "--dry_run",
    type=bool,
    default=False,
    help="When true, we jump over the task running process. Useful when you want to check if the task parsing is correct.",
)

# Preprocess
opt = parser.parse_args()

# Sleep opt.sleep seconds.
time.sleep(opt.sleep)

# Is dry_run open?
if opt.dry_run:
    logger.warning("We are in dry run mode!")

# Get GPU devices.
use_gpu = False
slurm_arguments = {}
if opt.GPU is not None:
    if not opt.slurm:
        gpu_pool = [int(gpu_id) for gpu_id in opt.GPU]
        if len([gpu_id for gpu_id in gpu_pool if gpu_id < 0]) == 0:
            if opt.num_task_parallel > len(gpu_pool):
                raise ValueError(
                    f"You are trying to run {opt.num_task_parallel} tasks simultaneously but we only have {len(gpu_pool)} GPUs."
                )
            use_gpu = True
            if opt.num_task_parallel == -1:
                opt.num_task_parallel = len(gpu_pool)
    else:
        use_gpu = True
        gpu_pool = [int(gpu_id) for gpu_id in opt.GPU] * opt.num_task_parallel
        slurm_arguments = {}
        if opt.slurm_config is not None:
            slurm_arguments = read_yaml(root_path / opt.slurm_config)


if not use_gpu:
    gpu_pool = []


# stdout dir
# where we store logs of tasks.
stdout_dir = root_path / "stdout" / opt.procedure_name / opt.script_type / opt.model


def task_generator(hyperparameter_list: dict[str, Any]) -> tuple[list[list], int]:
    """
    Generate tasks from hyperparameter list.
    Supports both old structure (with zip_style and counting_style) and
    new structure (with sequential and combinatorial).
    """
    hyperparameter_list["file_name"] = str(root_path / hyperparameter_list["worker"])
    hyperparameter_list["argparser"] = opt.procedure_name + "_" + hyperparameter_list["job_type"]
    if isinstance(opt.interpreter, list):
        hyperparameter_list["interpreter"] = opt.interpreter
    else:
        hyperparameter_list["interpreter"] = [
            opt.interpreter,
        ]

    generated_commands = parameter_parser(hyperparameter_list)

    # attach index.
    if hyperparameter_list.get("repeat") is not None:
        generated_commands = [
            commands + ["--model_index", str(index)]
            for index in range(opt.start_from_this_index, hyperparameter_list.get("repeat")+1)
            for commands in generated_commands
        ]

    logger.info(f"We have planned {len(generated_commands)} tasks!")
    return generated_commands, len(generated_commands)


if opt.script_type == "previous_failed_tasks":
    logger.info(
        f"We are in previous_failed_tasks mode. We will read in and rerun failed commands recorded in {opt.model}_previous_failed_tasks.txt."
    )
    try:
        f_previous_failed_tasks = (
            root_path / "parameter_set" / opt.procedure_name / f"{opt.model}_previous_failed_tasks.txt"
        ).open("r")
    except FileNotFoundError:
        logger.exception(
            f"File {str('parameter_set' / opt.procedure_name / f'{opt.model}_previous_failed_tasks.txt')} not found!"
        )
    except Exception as e:
        raise e

    generated_tasks = []
    for command in f_previous_failed_tasks:
        generated_tasks.append(command.strip().split(" "))
    the_number_of_task = len(generated_tasks)

    # Append Time to stdout_dir
    stdout_dir = stdout_dir / current_local_time

    mkdir_if_not_exist(stdout_dir)
    failed_tasks = monitor_and_automaticly_run_tasks(
        opt.dry_run,
        generated_tasks,
        use_gpu,
        gpu_pool,
        opt.num_task_parallel,
        stdout_dir,
        opt.slurm,
        slurm_arguments=slurm_arguments,
    )

    # Report the execution sumamry:
    logger.warning("==========================================")
    logger.warning("                Summary                   ")
    logger.warning("==========================================")
    failed_commands = []
    if len(failed_tasks) == 0:
        logger.info(f"All {the_number_of_task} tasks have successfully completed.")
    else:
        logger.warning(
            f"{len(failed_tasks)} tasks have failed. Please check what is wrong according to logs in directory stdout/and fix them!"
        )
        for index, command in failed_tasks.items():
            logger.warning(f"----> Task No.{index} has failed. <----")
            logger.warning(f"Task Command: {command}.")
            failed_commands.append(" ".join(command) + "\n")

    """
    Only in previous_failed_tasks mode we can rewrite the previous_failed_tasks.txt.
    By this we can avoid missing failed tasks in the previous task sets if the execution script calls bulk_task_runner.py
    multiple times.
    """
    f_previous_failed_tasks = (
        root_path / "parameter_set" / opt.procedure_name,
        f"{opt.model}_previous_failed_tasks.txt",
    ).open("w")
    f_previous_failed_tasks.writelines(failed_commands)
    f_previous_failed_tasks.close()
else:
    parameter_lib = importlib.import_module(f".{opt.procedure_name}", package="parameter_set")
    parameter_retriver = getattr(parameter_lib, "parameter_retriver")
    full_job_list = parameter_retriver(opt)

    if opt.job_name is None:
        logger.warning("No job selected! Exiting...")
        sys.exit(0)
    elif opt.job_name == [
        "ALL",
    ]:
        opt.job_name = full_job_list.keys()

    logger.info(f"We will execute the following jobs: {opt.job_name}.")
    logger.info(f"All available jobs: {full_job_list.keys()}.")

    for job in opt.job_name:
        job_content = full_job_list[job]

        # If we only have a single dict, we pack it into a list.
        if isinstance(job_content, dict):
            job_content = [
                job_content,
            ]
        logger.info(f"Current executing the job: {job}. It has {len(job_content)} subjobs.")

        # Extract the list and run the tasks one by one.
        for idx, sub_job in enumerate(job_content):
            logger.warning(f"============ subjob No. {idx + 1} started ============")
            stdout_dir_for_this_subjob = stdout_dir / current_local_time / job / f"subjob_{idx + 1}"
            mkdir_if_not_exist(stdout_dir_for_this_subjob)
            generated_tasks, the_number_of_task = task_generator(sub_job)

            failed_tasks = monitor_and_automaticly_run_tasks(
                opt.dry_run,
                generated_tasks,
                use_gpu,
                gpu_pool,
                opt.num_task_parallel,
                stdout_dir_for_this_subjob,
                opt.slurm,
                slurm_arguments=slurm_arguments,
            )

            # Report the execution sumamry:
            logger.warning("==========================================")
            logger.warning("                Summary                   ")
            logger.warning("==========================================")
            failed_commands = []
            if len(failed_tasks) == 0:
                logger.info(f"All {the_number_of_task} tasks have successfully completed.")
            else:
                logger.warning(
                    f"{len(failed_tasks)} tasks have failed. Please check what is wrong according to logs in directory stdout/ and fix them!"
                )
                for index, command in failed_tasks.items():
                    logger.warning(f"----> Task No.{index} has failed. <----")
                    logger.warning(f"Task Command: {command}.")
                    failed_commands.append(" ".join(command) + "\n")

            # Only in previous_failed_tasks mode we rewrite the previous_failed_tasks.txt.
            # By this we can avoid missing failed tasks in the previous task sets if the execution script calls
            # bulk_task_runner.py multiple times.
            f_previous_failed_tasks = (
                root_path / "parameter_set" / opt.procedure_name / f"{opt.model}_previous_failed_tasks.txt"
            ).open("a")
            f_previous_failed_tasks.writelines(failed_commands)
            f_previous_failed_tasks.close()

            logger.warning(f"============ subjob No. {idx + 1} ended ============")
