# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, List, Tuple, Union
from time import time
import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T

import multiprocessing as mp
from doctr.transforms import Resize
from doctr.utils.multithreading import multithread_exec

__all__ = ["PreProcessor"]


class PreProcessor(nn.Module):
    """Implements an abstract preprocessor object which performs casting, resizing, batching and normalization.

    Args:
    ----
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int,
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.resize: T.Resize = Resize(output_size, **kwargs)
        self.resize.to(torch.device("cuda:0"))
        # Perform the division by 255 at the same time
        self.normalize = T.Normalize(mean, std)
        self.normalize.to(torch.device("cuda:0"))
        # self.resize_times = 0.0
        # self.dtype_times = 0.0

    def batch_inputs(self, samples: List[torch.Tensor]) -> List[torch.Tensor]:
        """Gather samples into batches for inference purposes

        Args:
        ----
            samples: list of samples of shape (C, H, W)

        Returns:
        -------
            list of batched samples (*, C, H, W)
        """
        num_batches = int(math.ceil(len(samples) / self.batch_size))
        batches = [
            torch.stack(samples[idx * self.batch_size : min((idx + 1) * self.batch_size, len(samples))], dim=0)
            for idx in range(int(num_batches))
        ]

        return batches

    def sample_transforms(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise AssertionError("expected list of 3D Tensors")
        if isinstance(x, np.ndarray):
            if x.dtype not in (np.uint8, np.float32):
                raise TypeError("unsupported data type for numpy.ndarray")
            x = torch.from_numpy(x.copy()).permute(2, 0, 1)
        elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
            raise TypeError("unsupported data type for torch.Tensor")
        # Resizing
        # resize_start_time = time()
        x = self.resize(x)
        # resize_end_time = time()
        # resize_time = resize_end_time - resize_start_time
        # self.resize_times+=resize_time

        

        # Data type
        # start_time = time()
        if x.dtype == torch.uint8:
            x = x.to(dtype=torch.float32).div(255).clip(0, 1)  # type: ignore[union-attr]
        else:
            x = x.to(dtype=torch.float32)  # type: ignore[union-attr]
        # to device
        # end_time = time()
        # self.dtype_times+=end_time - start_time

        return x
    

    def __call__(self, x: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]) -> List[torch.Tensor]:
        """Prepare document data for model forwarding

        Args:
        ----
            x: list of images (np.array) or tensors (already resized and batched)

        Returns:
        -------
            list of page batches
        """
        
        # Input type check
        if isinstance(x, (np.ndarray, torch.Tensor)):
            if x.ndim != 4:
                raise AssertionError("expected 4D Tensor")
            if isinstance(x, np.ndarray):
                if x.dtype not in (np.uint8, np.float32):
                    raise TypeError("unsupported data type for numpy.ndarray")
                x = torch.from_numpy(x.copy()).permute(0, 3, 1, 2)
            elif x.dtype not in (torch.uint8, torch.float16, torch.float32):
                raise TypeError("unsupported data type for torch.Tensor")
            # Resizing
            # start_time = time()
            if x.shape[-2] != self.resize.size[0] or x.shape[-1] != self.resize.size[1]:
                x = F.resize(
                    x, self.resize.size, interpolation=self.resize.interpolation, antialias=self.resize.antialias
                )
            # end_time = time()
            # print(f"Time taken for resizing: {end_time - start_time}")
            # Data type
            if x.dtype == torch.uint8:  # type: ignore[union-attr]
                x = x.to(dtype=torch.float32).div(255).clip(0, 1)  # type: ignore[union-attr]
            else:
                x = x.to(dtype=torch.float32)  # type: ignore[union-attr]
            batches = [x]

        elif isinstance(x, list) and all(isinstance(sample, (np.ndarray, torch.Tensor)) for sample in x):
            # Sample transform (to tensor, resize)
            # start_time = time()
            samples = []
            for sample in x:
                samples.append(self.sample_transforms(sample))
            # samples = list(multithread_exec(self.sample_transforms, x, threads= mp.cpu_count()))
            # end_time = time()
            # n = mp.cpu_count()
            # print(f"Number of CPUs: {n}")

            # print(f"Time taken for sample transforms: {end_time - start_time}")
            # print(f"Time taken for resize: {self.resize_times}")
            # print(f"Time taken for dtype change: {self.dtype_times}")
            self.resize_times = []
            self.dtype_times = []
            # Batching
            batches = self.batch_inputs(samples)
        else:
            raise TypeError(f"invalid input type: {type(x)}")

        # Batch transforms (normalize)
        # start_time = time()
        
        for idx, batch in enumerate(batches):
            batches[idx] = self.normalize(batch)
        # batches = list(multithread_exec(self.normalize, batches))
        # end_time = time()
        # print(f"Time taken for normalizing: {end_time - start_time}")

        return batches
