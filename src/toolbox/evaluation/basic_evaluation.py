from pathlib import Path

from src.toolbox.misc import get_logger, mkdir_if_not_exist

logger = get_logger(name = __file__)

def basic_evaluation(model, minibatch, desc, batch_idx, opt):
    '''
    This function will be called when task_name = graph

    This function only accepts figure name-matplotlib figure object pairs and saves these figures in the predefined location
    with the correct name and format.
    '''
    # Create the plot storing directory if not exist.
    plot_store_dir_for_this_batch = Path(opt.store_dir, desc, str(batch_idx))
    opt.plot_store_dir_for_this_batch = plot_store_dir_for_this_batch
    mkdir_if_not_exist(plot_store_dir_for_this_batch)

    logger.info(f'Start {opt.task_name} for the No.{batch_idx} minibatch in {desc} dataset!')
    model(opt.task_name, minibatch, opt)
