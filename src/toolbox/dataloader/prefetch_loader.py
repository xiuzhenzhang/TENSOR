# Copied from https://pytorch-geometric.readthedocs.io/en/2.7.0/_modules/torch_geometric/loader/prefetch.html.
# Original code is licensed under MIT.

import warnings
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
from torch.utils.data import DataLoader


class PrefetchLoader:
    r"""A GPU prefetcher class for asynchronously transferring data of a
    :class:`torch.utils.data.DataLoader` from host memory to device memory.

    Args:
        loader (torch.utils.data.DataLoader): The data loader.
        device (torch.device, optional): The device to load the data to.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device | None = None,
    ):
        self.loader = loader
        self.device_helper = DeviceHelper(device)

    def non_blocking_transfer(self, batch: Any) -> Any:
        if not self.device_helper.is_gpu:
            return batch
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        if torch.is_tensor(batch):
            batch = batch.pin_memory()
            return batch.to(device = self.device_helper.device, non_blocking=True)

        return batch

    def __iter__(self) -> Any:
        first = True
        self.device_helper.maybe_init_stream()

        batch = None
        for next_batch in self.loader:

            with self.device_helper.stream_context():
                next_batch = self.non_blocking_transfer(next_batch)

            if not first:
                yield batch
            else:
                first = False

            self.device_helper.maybe_wait_stream()

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.loader})'

class DeviceHelper:
    def __init__(self, device: torch.device | None = None):
        with_cuda = torch.cuda.is_available()

        if device is None:
            device = 'cuda' if with_cuda else 'cpu'

        self.device = torch.device(device)
        self.is_gpu = self.device.type in ['cuda', 'xpu']

        if (self.device.type == 'cuda' and not with_cuda):
            warnings.warn(
                f"Requested device '{self.device.type}' is not "
                f"available, falling back to CPU", stacklevel=2)
            self.device = torch.device('cpu')

        self.stream = None
        self.stream_context = nullcontext
        self.module = getattr(torch, self.device.type) if self.is_gpu else None

    def maybe_init_stream(self) -> None:
        if self.is_gpu:
            self.stream = self.module.Stream()
            self.stream_context = partial(
                self.module.stream,
                stream=self.stream,
            )

    def maybe_wait_stream(self) -> None:
        if self.stream is not None:
            self.module.current_stream().wait_stream(self.stream)
