import gc
import os
from typing import Any, Callable, Iterable, List, Tuple, Union

from pathlib import Path

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import glob

import environ

DEFAULT_INPUT_FILE_NAME = "input.data"
DEFAULT_LABELS_FILE_NAME = "labels.data"


def Xy_iter(X_or_y = "X", which_set="trn", sub_set="first_half", sz_thresh=2.0):
    if X_or_y == "X":
        npy_paths = glob.glob("./cache/preprocessed/%s/%s/*_X.npy" % \
                                                (which_set, sub_set))

    elif X_or_y == "y":
        npy_paths = glob.glob("./cache/preprocessed/%s/%s/*_y.npy" % \
                                                (which_set, sub_set))


    for npy_path in npy_paths:
        with open(npy_path, "rb") as f:
            res = np.load(f)
        if X_or_y == "y":
            res = (np.array(res) > sz_thresh).astype(int)
        yield res


def get_dataset_size(num_windows_per_file, 
                     which_set="trn", 
                     sub_set="first_half"):
    npy_paths = glob.glob("./cache/preprocessed/%s/%s/*_X.npy" % \
                                                (which_set, sub_set))
    return len(npy_paths) * num_windows_per_file


class MMAPDataset(Dataset):
    def __init__(
        self,
        input_iter: Iterable[np.ndarray],
        labels_iter: Iterable[np.ndarray],
        mmap_path: str = None,
        mmap_X_path: str = None,
        mmap_y_path: str = None,
        size: int = None,
        transform_fn: Callable[..., Any] = None,
        num_windows_per_file: int = None,
    ) -> None:
        super().__init__()

        self.mmap_inputs: np.ndarray = None
        self.mmap_labels: np.ndarray = None
        self.transform_fn = transform_fn

        if mmap_path is None:
            mmap_path = os.path.abspath(os.getcwd())
        self._mkdir(mmap_path)

        self.mmap_input_path = os.path.join(mmap_path, mmap_X_path)
        self.mmap_labels_path = os.path.join(mmap_path, mmap_y_path)

        # get num of windows per cached 
        self.num_windows_per_file = num_windows_per_file

        # If the total size is not known we load the dataset in memory first
        if size is None:
            input_iter, labels_iter = self._consume_iterable(input_iter, labels_iter)
            size = len(input_iter)
        self.length = size

        for idx, (input, label) in enumerate(zip(input_iter, labels_iter)):
            if self.mmap_inputs is None:
                self.mmap_inputs = self._init_mmap(
                    self.mmap_input_path, input.dtype, (self.length, *input.shape)
                )
                self.mmap_labels = self._init_mmap(
                    self.mmap_labels_path, label.dtype, (self.length, *label.shape)
                )
            self.mmap_inputs[idx][:] = input[:]
            self.mmap_labels[idx][:] = label[:]

        del input_iter
        del labels_iter
        gc.collect()


    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, torch.Tensor]]:
        file_dx = int(idx / self.num_windows_per_file)
        win_dx  = idx % self.num_windows_per_file
        X = self.mmap_inputs[file_dx][win_dx]
        y = self.mmap_labels[file_dx][win_dx]
        if self.transform_fn:
            return self.transform_fn(eeg=X)['eeg'], \
                    torch.tensor(y) 
        # return self.mmap_inputs[idx], self.mmap_labels[idx]


    def __len__(self) -> int:
        return self.length


    def _consume_iterable(self, input_iter: Iterable[np.ndarray], labels_iter: Iterable[np.ndarray]) -> Tuple[List[np.ndarray]]:
        inputs = []
        labels = []

        for input, label in zip(input_iter, labels_iter):
            inputs.append(input)
            labels.append(label)

        if len(inputs) != len(labels):
            raise Exception(
                f"Input samples count {len(inputs)} is different than the labels count {len(labels)}"
            )

        if not isinstance(inputs[0], np.ndarray):
            raise TypeError("Inputs and labels must be of type np.ndarray")

        return inputs, labels


    def _mkdir(self, path: str) -> None:
        if os.path.exists(path):
            return

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return
        except:
            raise ValueError(
                "Failed to create the path (check the user write permissions)."
            )


    def _init_mmap(self, path: str, dtype: np.dtype, shape: Tuple[int], remove_existing: bool = False) -> np.ndarray:
        open_mode = "w+" if remove_existing else "r+"
        return np.memmap(
            path,
            dtype=dtype,
            mode=open_mode,
            shape=shape,
        )