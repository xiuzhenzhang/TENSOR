import gc
from pathlib import Path

import matplotlib.pyplot as plt


def save_fig(fig, file_location, file_name):
    fig.savefig(Path(file_location, file_name), bbox_inches = "tight")
    fig.clear()
    plt.close(fig = fig)
    del fig
    gc.collect()
