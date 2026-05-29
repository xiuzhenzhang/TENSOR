import subprocess
import time
from pathlib import Path
from typing import TextIO

from termcolor import colored

from src.toolbox.misc import get_logger

logger = get_logger(__name__)

default_slurm_kwargs = {
    "slurm_partition": "SCT",
    "slurm_job_name": "slurm_task",
    "slurm_cpus_per_task": 8,
    "slurm_time": 1400,
    "slurm_mem": "16GB",
    "slurm_gres": "gpu:1",
    "slurm_qos": "normal",
}

monitor_frequency = 10
wait_after_task_finished = 3


def monitor_and_automaticly_run_tasks(
    dry_run: bool,
    tasks: list[list],
    use_gpu: bool,
    available_gpus: list,
    num_task_parallel: int,
    stdout_dir: str,
    use_slurm: bool,
    **kwargs,
) -> dict:
    if use_slurm:
        if use_gpu:
            return monitor_and_automaticly_run_tasks_on_slurm_gpu_node(
                tasks, available_gpus, num_task_parallel, stdout_dir, **kwargs
            )
        return monitor_and_automaticly_run_tasks_on_slurm_cpu_node(tasks, num_task_parallel, stdout_dir, **kwargs)
    if use_gpu:
        return monitor_and_automaticly_run_tasks_on_gpu(tasks, available_gpus, num_task_parallel, stdout_dir)
    return monitor_and_automaticly_run_tasks_on_cpu(tasks, num_task_parallel, stdout_dir)


def monitor_and_automaticly_run_tasks_on_cpu(
    tasks: list[list], num_task_parallel: int, stdout_dir: str
) -> dict[int, list]:
    number_of_tasks = len(tasks)

    def run_task(task: list, task_id: int) -> tuple[subprocess.Popen, TextIO]:
        # Replace this command with your actual task command
        logger.warning(f"----> Task No.{task_id}/{number_of_tasks} started. <----")
        logger.info(f"Command of task {task_id}/{number_of_tasks}: {task}")
        f_log = Path(stdout_dir / f"stdout_log_{task_id}.txt").open("w")
        process = subprocess.Popen(task, stdout=f_log, stderr=f_log, universal_newlines=True)

        return process, f_log

    task_id = 1
    running_tasks = []
    number_of_running_tasks = 0
    completed_tasks = set()
    all_task_executed = False
    failed_tasks = {}

    while True:
        if task_id > number_of_tasks:
            all_task_executed = True

        if number_of_running_tasks < num_task_parallel and not all_task_executed:
            command = tasks[task_id - 1]
            process, log_file = run_task(command, task_id)
            running_tasks.append(
                {
                    "task_id": task_id,
                    "command": command,
                    "process": process,
                    "stdout": log_file,
                }
            )
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for task in running_tasks:
            if task["task_id"] not in completed_tasks and task["process"].poll() is not None:
                if task["process"].poll() != 0:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} failed!. <----")
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} completed!. <----")

                completed_tasks.add(task["task_id"])
                task["stdout"].close()
                number_of_running_tasks -= 1
                time.sleep(wait_after_task_finished)

        # If the task id is bigger than the the number of tasks, quit the loop.
        if all_task_executed and len(completed_tasks) == number_of_tasks:
            break

        time.sleep(1 / monitor_frequency)

    return failed_tasks


def monitor_and_automaticly_run_tasks_on_gpu(
    tasks: list[list], available_gpus: list, num_task_parallel: int, stdout_dir
) -> dict[int, list]:
    gpu_pool = set(available_gpus)
    ticket_pool = set(range(num_task_parallel))
    number_of_gpus = len(gpu_pool)
    number_of_tasks = len(tasks)

    def run_task(task: list, task_id: int, gpu_id: int) -> tuple[subprocess.Popen, TextIO]:
        task_list = task + ["--cuda", "--cuda_device", f"{gpu_id}"]

        logger.warning(f"----> Task No.{task_id}/{number_of_tasks} started. <----")
        logger.info(f"Command of task {task_id}/{number_of_tasks}: {' '.join(task_list)}")
        f_log = Path(stdout_dir / f"stdout_log_{task_id}.txt").open("w")
        process = subprocess.Popen(task_list, stdout=f_log, stderr=f_log, universal_newlines=True)

        return process, f_log

    unique_task_id = 1
    running_tasks = {}
    all_task_executed = False
    failed_tasks = {}

    while True:
        if unique_task_id > number_of_tasks:
            all_task_executed = True

        if len(gpu_pool) != 0 and len(ticket_pool) != 0 and not all_task_executed:
            available_gpu = gpu_pool.pop()
            ticket = ticket_pool.pop()
            command = tasks[unique_task_id - 1]
            process, log_file = run_task(command, unique_task_id, available_gpu)
            running_tasks[ticket] = {
                "task_id": unique_task_id,
                "gpu_id": available_gpu,
                "command": command,
                "process": process,
                "stdout": log_file,
            }
            unique_task_id += 1

        # Check if one task has finished. If so, get the result and do some housekeeping
        # Add the allocated gpu_id and ticket back to the gpu_pool and ticket pool, saying we can start a new task if the GPU resources are sufficient.
        for ticket, task in running_tasks.items():
            if task != {} and task["process"].poll() is not None:
                if task["process"].poll() != 0:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} failed!. <----")
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} completed!. <----")

                task["stdout"].close()
                gpu_pool.add(task["gpu_id"])
                ticket_pool.add(ticket)
                running_tasks[ticket] = {}
                time.sleep(wait_after_task_finished)

        # If all GPUs are free again and the task id is bigger than the the number of tasks, quit the loop.
        if len(gpu_pool) == number_of_gpus and all_task_executed:
            break

        time.sleep(1 / monitor_frequency)

    return failed_tasks


def monitor_and_automaticly_run_tasks_on_slurm_cpu_node(
    tasks: list[list], num_task_parallel: int, stdout_dir: str, slurm_arguments: dict
) -> dict[int, list]:
    import submitit

    number_of_tasks = len(tasks)
    if slurm_arguments is None:
        slurm_arguments = {}

    if len(slurm_arguments) > 0:
        logger.info("The following slurm environment variables will be updated.")
        for key in slurm_arguments:
            logger.info(
                f"{key}: {colored(default_slurm_kwargs.get(key), 'blue')} -> {colored(slurm_arguments[key], 'red')}"
            )

    def run_task(task: list, task_id: int) -> submitit.Job:
        # Replace this command with your actual task command
        logger.warning(f"----> Task No.{task_id}/{number_of_tasks} started. <----")
        logger.info(f"Command of task {task_id}/{number_of_tasks}: {task}")

        executor = submitit.AutoExecutor(folder=str(stdout_dir / str(task_id)))

        default_slurm_kwargs.update(slurm_arguments)
        executor.update_parameters(**default_slurm_kwargs)
        function = submitit.helpers.CommandFunction(task)
        return executor.submit(function)

    task_id = 1
    running_tasks = []
    number_of_running_tasks = 0
    completed_tasks = set()
    all_task_executed = False
    failed_tasks = {}

    logger.warning(
        'Tasks submitted to slurm are out of our control. We can check if one job has finished, but submitit does not tell us if a task has successfully finished or errored out.\n \
         Our method is checking if the output log contains "Submitted job triggered an exception". If it has then we decide the job failed, otherwise succeed. \n \
         This method is unrealiable. You may need to manually check the results and identify any tasks that have failed.'
    )

    while True:
        if task_id > number_of_tasks:
            all_task_executed = True

        if number_of_running_tasks < num_task_parallel and not all_task_executed:
            command = tasks[task_id - 1]
            job = run_task(command, task_id)
            running_tasks.append(
                {
                    "task_id": task_id,
                    "command": command,
                    "job": job,
                    "slurm_id": job.job_id,
                }
            )
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for task in running_tasks:
            if task["task_id"] not in completed_tasks and task["job"].done():
                if "Submitted job triggered an exception" in task["job"].stderr():
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} failed!. <----")
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} completed!. <----")

                completed_tasks.add(task["task_id"])
                number_of_running_tasks -= 1
                time.sleep(wait_after_task_finished)

        # If the task id is bigger than the the number of tasks, quit the loop.
        if all_task_executed and len(completed_tasks) == number_of_tasks:
            break

        time.sleep(1 / monitor_frequency)

    return failed_tasks


def monitor_and_automaticly_run_tasks_on_slurm_gpu_node(
    tasks: list[list], available_gpus: list, num_task_parallel: int, stdout_dir: str, slurm_arguments: dict
) -> dict[int, list]:
    import submitit

    # I don't quite know how the GPU allocation works in slurm.
    # Due to this, we temporarily disable gpu_pool in this function.
    gpu_pool = list(available_gpus)
    ticket_pool = set(range(num_task_parallel))
    number_of_gpus = len(gpu_pool)
    number_of_tasks = len(tasks)

    if slurm_arguments is None:
        slurm_arguments = {}

    if len(slurm_arguments) > 0:
        logger.info("The following slurm environment variables will be updated.")
        for key in slurm_arguments:
            logger.info(
                f"{key}: {colored(default_slurm_kwargs.get(key), 'blue')} -> {colored(slurm_arguments[key], 'red')}"
            )

    def run_task(task: list, task_id: int, gpu_id: int) -> submitit.Job:
        task_list = task + ["--cuda", "--cuda_device", f"{gpu_id}"]

        logger.warning(f"----> Task No.{task_id}/{number_of_tasks} started. <----")
        logger.info(f"Command of task {task_id}/{number_of_tasks}: {' '.join(task_list)}")
        executor = submitit.AutoExecutor(folder=str(stdout_dir / str(task_id)))

        default_slurm_kwargs.update(slurm_arguments)
        executor.update_parameters(**default_slurm_kwargs)
        function = submitit.helpers.CommandFunction(task_list)
        return executor.submit(function)

    unique_task_id = 1
    running_tasks = {}
    all_task_executed = False
    failed_tasks = {}

    logger.warning(
        'Tasks submitted to slurm are out of our control. We can check if one job has finished, but submitit does not tell us if a task has successfully finished or errored out.\n \
         Our method is checking if the output log contains "Submitted job triggered an exception". If it has then we decide the job failed, otherwise succeed. \n \
         This method is unrealiable. You may need to manually check the results and identify any tasks that have failed.'
    )

    while True:
        if unique_task_id > number_of_tasks:
            all_task_executed = True

        if len(gpu_pool) != 0 and len(ticket_pool) != 0 and not all_task_executed:
            available_gpu = gpu_pool.pop()
            ticket = ticket_pool.pop()
            command = tasks[unique_task_id - 1]
            job = run_task(command, unique_task_id, available_gpu)
            running_tasks[ticket] = {
                "task_id": unique_task_id,
                "gpu_id": available_gpu,
                "command": command,
                "job": job,
                "slurm_id": job.job_id,
            }
            unique_task_id += 1

        # Check if one task has finished. If so, get the result and do some housekeeping
        # Add the allocated gpu_id and ticket back to the gpu_pool and ticket pool, saying we can start a new task if the GPU resources are sufficient.
        for ticket, task in running_tasks.items():
            if task != {} and task["job"].done():
                if "Submitted job triggered an exception" in task["job"].stderr():
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} failed!. <----")
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f"----> Task No.{task['task_id']}/{number_of_tasks} completed!. <----")

                gpu_pool.append(task["gpu_id"])
                ticket_pool.add(ticket)
                running_tasks[ticket] = {}
                time.sleep(wait_after_task_finished)

        # If all GPUs are free again and the task id is bigger than the the number of tasks, quit the loop.
        if len(gpu_pool) == number_of_gpus and all_task_executed:
            break

        time.sleep(1 / monitor_frequency)

    return failed_tasks
