import pathlib
from pathlib import Path

import yaml


# Read and convert a YAML file into a dict object.
def read_yaml(yaml_path: str) -> dict:
    """Read a yaml file

    Args:
        yaml_path (str): The path to the yaml file.

    Returns:
        Dict: The data
    """
    a = {}
    if yaml_path is not None:
        item = Path(yaml_path)
        with item.open("r") as f:
            try:
                a = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)

    if a is None:
        a = {}

    return a


def write_yaml(data: dict, yaml_path: str, yaml_file: str) -> None:
    """Write a dict into a yaml file.

    Args:
        data (dict): The data
        yaml_path (str): Folder where we place the yaml file.
        yaml_file (str): The name of this yaml file.
    """
    # Convert UnixPath to str.
    for key, value in data.items():
        if isinstance(value, pathlib.PosixPath):
            data[key] = str(value.resolve())

    with Path(yaml_path, yaml_file).open("w") as outfile:
        yaml.safe_dump(data, outfile, default_flow_style=False, allow_unicode=True)
