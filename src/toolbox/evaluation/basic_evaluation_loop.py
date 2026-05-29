from pathlib import Path

from tqdm import tqdm

from src.toolbox.misc import free_model_from_gpu, get_logger, mkdir_if_not_exist, write_to_txt

logger = get_logger(name = __file__)

def basic_evaluation_loop(model, dataset, desc, opt, early_offload = True, desc_string = '{0}', postprocess_func = None):
    task_name = opt.task_name

    elapsed_time = 0
    list_output_results = None

    with tqdm(dataset, desc = desc_string.format(desc)) as progress_bar:
        for minibatch in progress_bar:
            results_per_minibatch = model(task_name, minibatch, opt)

            if results_per_minibatch is None:
                continue

            if list_output_results is None:
                result_length = len(results_per_minibatch)
                list_output_results = [[] for _ in range(result_length)]

            [a.extend(b) if isinstance(b, list) else a.append(b) for a, b in zip(list_output_results, results_per_minibatch)]

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    if early_offload:
        # How to remove a model and free its memory immediately?
        free_model_from_gpu(model)

    mkdir_if_not_exist(opt.store_dir)
    result_file = Path(opt.store_dir, f'{desc}_{task_name}_misc.txt')
    strings = [f'Evaluation speed: {elapsed_time/data_size}s per sequence.']
    write_to_txt(strings, result_file)

    logger.info(f'Entering {postprocess_func.__name__} for result postprocess...')
    # call user's postprocess function for evaluation results.
    postprocess_func(list_output_results, desc, opt)
