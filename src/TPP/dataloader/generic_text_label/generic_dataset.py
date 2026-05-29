from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.utils as utils

from src.toolbox.dict_operation import dict_merge
from src.toolbox.misc import load_from_pkl


def prepend(array: np.array, number: float) -> np.array:
    """insert a float number at the start of one numpy array.

    Args:
        per_line (np.array): the numpy array.
        number (float): the number inserted into the array.

    Returns:
        np.array: the result.
    """
    return np.concatenate([np.array([number]), array])


def append(per_line: np.array, number: float) -> np.array:
    """append a float number at the end of one numpy array.

    Args:
        per_line (np.array): the numpy array.
        number (float): the number inserted into the array.

    Returns:
        np.array: the result.
    """
    return np.concatenate([per_line, np.array([number])])


def diff(per_line: np.array, prepend: float = np._NoValue, append: float = np._NoValue) -> np.array:
    """Calculate the diff of the input array

    Args:
        per_line (np.array): The input numpy array
        prepend (float, optional): Defaults to np._NoValue.
        append (float, optional): Defaults to np._NoValue.
        Values to prepend or append to a along axis prior to performing the difference.

    Returns:
        np.array: the result
    """
    return np.diff(per_line, prepend=prepend, append=append)


class GenericDatasetWithSeqLabel(utils.data.Dataset):
    def __init__(
        self: Self,
        data: dict[str, Any],
        device: torch.device,
        property_dict: dict[str, Any],
        relative_time: bool = True,
        evaluate: bool = False,
        shift: bool = False,
        input_norm_data: bool = False,
    ) -> Self:
        """Create a generic dataset

        Args:
            self (Self): the dataset item,
            data (dict[str, Any]): the raw data
            device (torch.device): the batched data will be moved to this device.
            property_dict (dict[str, Any]): the property of the dataset.
            evaluate (bool, optional): enable or disable the evaluate mode. Defaults to False.
            shift (bool, optional): if enabled, we shift the time interval by a small number to avoid 0. Defaults to False.
            input_norm_data (bool, optional): output the mean and std of the time interval if true. Defaults to False.

        Returns:
            Self: the dataset item.
        """
        super().__init__()
        self.device = device
        self.float_dtype = torch.get_default_dtype()
        self.evaluate = evaluate
        self.number_of_mark = property_dict["num_marks"]
        self.number_of_seq_lebel = property_dict["num_label"]
        self.start_time = property_dict["t_0"]
        self.end_time = property_dict["T"]
        self.mean = property_dict["mean"] if input_norm_data else 0
        self.std = property_dict["std"] if input_norm_data else 1

        # Convert data from list to np.array.
        self.time_seq = data["time_seq"]
        self.score = data["score"]
        # compatible with old names.
        self.marks = data["event"]
        self.label = data["label"]
        # embedding of the text.
        self.text_emb = data["text_emb"]

        self.passenger_keys = [
            "text",
            "post_time",
            "account_profile_description",
            "follower_count",
            "following_count",
            "account_creation_date",
            "reply_post",
            "reply_post_time",
            "reply_account_profile_description",
            "reply_follower_count",
            "reply_following_count",
            "reply_account_creation_date",
            "repost_post_time",
            "repost_account_profile_description",
            "repost_follower_count",
            "repost_following_count",
            "repost_account_creation_date",
        ]
        self.passenger = {key: data[key] for key in self.passenger_keys}

        self.dataset_size = len(self.time_seq)
        if len(self.time_seq) != len(self.label):
            raise Exception("Not all sequences have a label!")

        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True
        if relative_time:
            if self.evaluate:
                self.time_seq = [np.diff(seq, prepend=self.start_time) for seq in self.time_seq]
            else:
                # Use T
                # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
                # Do not use T
                self.time_seq = [np.diff(seq, prepend=self.start_time) for seq in self.time_seq]
                self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
                self.marks = [append(seq, self.number_of_mark) for seq in self.marks]
                self.text_emb = [seq + [[0] * property_dict["emb_dim"]] for seq in self.text_emb]

            self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
            self.time_seq = [prepend(seq, 0) for seq in self.time_seq]
            self.marks = [prepend(seq, self.number_of_mark) for seq in self.marks]
            self.text_emb = [[[0] * property_dict["emb_dim"]] + seq for seq in self.text_emb]
        else:
            if self.evaluate:
                self.time_seq = [prepend(seq, self.start_time) for seq in self.time_seq]
            else:
                # Use T
                # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
                # Do not use T
                self.time_seq = [prepend(seq, self.start_time) for seq in self.time_seq]
                self.time_seq = [append(seq, seq[-1] + 0.1) for seq in self.time_seq]
                self.marks = [append(seq, self.number_of_mark) for seq in self.marks]

            self.marks = [prepend(seq, self.number_of_mark) for seq in self.marks]

        # Fix datatype
        self.time_seq = [np.array(seq, dtype=np.float32) for seq in self.time_seq]
        self.score = [np.array(seq, dtype=np.float32) for seq in self.score]
        self.marks = [np.array(seq, dtype=np.int64) for seq in self.marks]
        self.text_emb = [np.array(seq, dtype=np.float32) for seq in self.text_emb]
        self.label = np.array(self.label, dtype=np.int64)

        # Caveat: self.time_seq and self.marks have dummy mark while self.score and self.intensity do not.
        self.max_seq_len = max([len(item) for item in self.time_seq])

    def __getitem__(self: Self, index: int | Iterable) -> list:
        """Get the batched data.

        Args:
            self (Self): the dataset
            index (Union[int, Iterable]): the index of the selected data.

        Returns:
            list: the output.
        """
        """
        Synthetic dataloader is very simple. It doesn't have any event infomation at each timestamp,
        and only the time differences between two neighboring mark are available.
        """
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)]

        return (
            self.time_seq[index],
            self.marks[index],
            self.text_emb[index],
            self.score[index],
            self.label[index],
        ), {key: value[index] for key, value in self.passenger.items()}

    def __len__(self: Self):
        """return the length of the dataset.

        Args:
            self (Self): the dataset item.

        Returns:
            int: the length of the dataset.
        """
        return self.dataset_size

    def data_collator(self: Self, data: list) -> tuple[list, tuple[float, float]]:
        """
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        """
        mask = []
        padded_data = []
        passenger_data = []

        for item in data:
            batched_data, passenger = item

            pad_length = self.max_seq_len - batched_data[0].size
            mask = np.array([1] * batched_data[0].size + [0] * pad_length, dtype=np.bool)
            padded_time_seq = np.pad(batched_data[0], (0, pad_length), mode="constant", constant_values=0)
            padded_event = np.pad(
                batched_data[1],
                (0, pad_length),
                mode="constant",
                constant_values=self.number_of_mark,
            )
            padded_text_emb = np.pad(batched_data[2], ((0, pad_length), (0, 0)), mode="constant", constant_values=0)
            padded_score = np.pad(batched_data[3], (0, pad_length), mode="constant", constant_values=0)
            label_of_the_seq = batched_data[4].item()

            padded_item = [padded_time_seq, padded_event, padded_text_emb, padded_score, mask, label_of_the_seq]
            padded_data.append(tuple(padded_item))
            passenger_data.append(passenger)

        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)

        # process the passenger.
        batched_passenger_data = dict_merge(passenger_data)

        return padded_data, (
            self.mean,
            self.std,
        ), batched_passenger_data


def read_data(path: str, file_name: str) -> dict:
    """Load the dataset.

    Args:
        path (str): the folder where the dataset locates.
        file_name (str): the name of the dataset file.

    Returns:
        dict: the loaded data.
    """
    return load_from_pkl(Path(path, file_name))


def generic_labeled_dataloader() -> tuple[utils.data.Dataset, Callable]:
    return [GenericDatasetWithSeqLabel, read_data]
