from pathlib import Path


def mkdir_if_not_exist(dir_path):
    item = Path(dir_path)
    if not item.exists():
        item.mkdir(parents=True)
    return 0
