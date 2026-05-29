import logging
import sys


def get_logger(name=None, root=True, mode="logging"):
    if mode == "logging":
        return get_logger_logging(name, root)
    return None


class LogFormat(logging.Formatter):
    @staticmethod
    def gen_format(x):
        from termcolor import colored

        fmt_dict = {
            "debug": {"color": "light_grey"},
            "info": {},
            "warning": {"color": "red"},
            "error": {
                "color": "red",
                "on_color": "on_white",
                "attrs": [
                    "bold",
                ],
            },
            "critical": {"color": "red", "on_color": "on_white", "attrs": ["bold", "blink"]},
        }

        return (
            colored("%(asctime)s", "green")
            + " "
            + colored("[%(filename)s:%(lineno)d]:", "cyan")
            + " "
            + colored("%(message)s", **fmt_dict[x])
        )

    FORMATS = {
        logging.DEBUG: gen_format("debug"),
        logging.INFO: gen_format("info"),
        logging.WARNING: gen_format("warning"),
        logging.ERROR: gen_format("error"),
        logging.CRITICAL: gen_format("critical"),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def get_logger_logging(name=None, root=True):
    """
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    """

    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    # add formatter to ch
    ch.setFormatter(LogFormat())
    # add ch to logger
    logger.addHandler(ch)

    return logger


if __name__ == "__main__":
    logger = get_logger(name=f"{__name__}", root=True, mode="logging")
    logger.debug("DEBUG.")
    logger.info("INFO.")
    logger.warning("WARNING.")
    logger.error("ERROR.")
    logger.critical("ERROR.")
