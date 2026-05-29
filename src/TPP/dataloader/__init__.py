import importlib

from src.toolbox.misc import get_logger

logger = get_logger(__name__)


def get_dataloader(opt):
    try:
        module = importlib.import_module('.' + opt.dataloader_name, package = f'src.{opt.procedure}.dataloader')
    except ImportError as e:
        logger.exception(f"Dataloader named {opt.dataloader_name} is not found! Please try again.")
        logger.exception(repr(e))

    logger.info(f"Dataloader name: {opt.dataloader_name}")
    return module.get_dataloader()
