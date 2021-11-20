# coding=utf-8

import json
from itertools import islice
from math import ceil
from os import path
from typing import List, Dict, Union, Optional

import torch
import torchtext
from logzero import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm

from instances import (
    AS,
    AS_PZERO,
    ClozeNotMaskedInstance,
    PzeroMaskedInstance,
    AsTrainingInstance,
    AsPzeroTrainingInstance,
    create_pzero_batch_instance,
    create_as_batch_instance,
    create_as_pzero_batch_instance,
)
from preprocess import reconfigure_sw2w_dict
from utils import read_lines, count_n_lines


class PaddingBucketIterator(object):
    """Generate padded mini-batches to minimize padding as much as possible.
    Args:
        * dataset: The list of the instances. (e.g. dataset = [(x, y), ...])
        * sort_key: The function for sorting the instances. For example,
            ```
            def sort_key(instance):
                return len(instance[0])
            ```
        * batch_size (int): The size of mini-batch
        * n_max_tokens (Optional[int]): The maximum number of tokens per batch
        * padding_value (int): The value for padding
        * shuffle (bool): Whether to shuffle instance
    """

    def __init__(
            self,
            dataset=None,
            sort_key=None,
            batch_size: int = 128,
            n_max_tokens: Optional[int] = None,
            padding_value: int = 0,
            shuffle: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sort_key = sort_key
        self.batch_size = n_max_tokens if n_max_tokens else batch_size
        self.padding_value = padding_value
        self.shuffle = shuffle
        self.use_max_tokens = True if n_max_tokens else False
        self.iterator = None

        if self.use_max_tokens:
            logger.info(f"a mini-batch size is determined by the number of tokens.")
            logger.info(f"Number of max tokens per batch: {self.batch_size}")
        else:
            logger.info(f"Number of instance per batch: {self.batch_size}")

        if self.dataset is not None:
            self.create_iterator()
            self.create_batches()

    def __len__(self) -> int:
        if self.use_max_tokens:
            length = ceil(len(self.dataset) / (self.batch_size / self.sort_key(self.dataset[0])))
        else:
            length = ceil(len(self.dataset) / self.batch_size)

        return length

    def __iter__(self):
        return self

    def __next__(self):
        return self.padding(next(self.iterator.batches))

    def create_iterator(self) -> None:
        def batch_size_fn_with_n_tokens(new, count, sofar):
            if sofar == 0:
                assert count == 1
                return self.sort_key(new)
            n_max_tokens = sofar / (count - 1)
            n_new_tokens = self.sort_key(new)
            if n_max_tokens < n_new_tokens:
                return n_new_tokens * count
            else:
                return n_max_tokens * count

        self.iterator = torchtext.data.BucketIterator(
            self.dataset,
            batch_size=self.batch_size,
            sort_key=self.sort_key,
            batch_size_fn=batch_size_fn_with_n_tokens if self.use_max_tokens else None,
            shuffle=self.shuffle,
            sort_within_batch=True
        )
        self.create_batches()

    def create_batches(self) -> None:
        self.iterator.create_batches()

    def padding(self, batch):
        """Return a padded mini-batch
        Example:
            The example of using 'torch.nn.utils.rnn.pad_sequence':
            Args:
                batch: [[xs, ys], ...], length = batch size
            Returns:
                padded_xs: torch.Tensor, shape = (batch, seq_length)
                ys: [y, ...], length = batch size

            xs, ys = zip(*batch)
            padded_xs = pad_sequence(xs, batch_first=True, padding_value=self.padding_value)

            return [padded_xs, ys]
        """
        raise NotImplementedError


class ClozeTaskDataset(Dataset):
    def __init__(self, file_path: Optional[str] = None, instances: Optional[List[ClozeNotMaskedInstance]] = None):
        """
        Args:
            * file_path (Optional[str]):
                Path to the file that contains 'instances.ClozeNotMaskedInstance' per line.
            * instances (Optional[List[ClozeNotMaskedInstance]]):
                Instead of file_path (default=None)
        """
        assert file_path is not None or instances is not None, "either argument must be specified"

        if instances:
            self.examples = [torch.LongTensor(instance["input_ids"]) for instance in tqdm(instances)]
        else:
            assert path.isfile(file_path)
            logger.info("Load file: {}".format(file_path))
            self.examples = [torch.LongTensor(json.loads(line)["input_ids"]) for line in tqdm(read_lines(file_path))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


def create_cloze_task_dataloader(
        batch_size: int = 2,
        n_gpu: int = 0,
        is_eval: bool = False,
        pad_token_id: Optional[int] = None,
        file_path: Optional[str] = None,
        dataset: Optional[ClozeTaskDataset] = None,
) -> DataLoader:
    """
    Args:
        * batch_size (int): The number of instances in one mini-batch.
        * n_gpu (int): The number of GPUs
        * is_eval (bool): If true, use 'SequentialSampler' as sampler
        * pad_token_id (Optional[int]): The embedding id of a padding token.
        * file_path (str): Path to the file containing 'instances.ClozeNotMaskedInstance' for each line.
        * dataset (Optional[ClozeTaskDataset]): dataset to be passed directly instead of file path (default=None)

    Returns:
        * data_loader (DataLoader)

    """

    def collate(examples: List[torch.Tensor]):
        if pad_token_id is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)

    batch_size = batch_size * max(1, n_gpu)

    if not dataset:
        dataset = ClozeTaskDataset(file_path)

    if is_eval:
        sampler = SequentialSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate
    )

    return data_loader


class PzeroBucketIterator(PaddingBucketIterator):
    """Generate padded mini-batches for pzero task"""

    def __init__(
            self,
            file_path: Optional[str] = None,
            batch_size: int = 128,
            n_max_tokens: int = None,
            padding_value: int = 0,
            shuffle: bool = False,
            n_instances: int = -1,
            max_seq_length: int = 512,
            dataset: Optional[List[PzeroMaskedInstance]] = None,
    ):
        """
        Args:
            * file_path (Optional[str]): Path to the file containing 'instances.PzeroMaskedInstance' for each line.
            * batch_size (int): The number of instances in one mini-batch
            * n_max_tokens (int): The maximum number of tokens contained in one mini-batch
            * padding_value (int): The embedding id of a padding token.
            * shuffle (bool): Whether to shuffle the dataset
            * n_instances (int): The number of instances to use for training
            * max_seq_length (int): The maximum number of input tokens for the model
            * dataset (Optional[List[PzeoMaskedInstance]]): Instead of file_path
        """
        super(PzeroBucketIterator, self).__init__(
            sort_key=self.sort_key,
            batch_size=batch_size,
            n_max_tokens=n_max_tokens,
            padding_value=padding_value,
            shuffle=shuffle,
        )
        self.file_path = file_path
        self.n_instances = n_instances
        self.max_seq_length = max_seq_length
        if dataset:
            self.dataset = dataset
        else:
            self.load_dataset()
        self.create_iterator()

    @staticmethod
    def sort_key(instance):
        return len(instance["input_ids"])

    def load_dataset(self):
        assert path.exists(self.file_path), f"not found: {self.file_path}"
        self.dataset: List[PzeroMaskedInstance] = []
        logger.info("The number of instances: {}".format("full" if self.n_instances == -1 else self.n_instances))
        logger.info("File: {}".format(self.file_path))
        n_max = None if self.n_instances == -1 else self.n_instances
        for instance in tqdm(islice(read_lines(self.file_path), 0, n_max)):
            self.dataset.append(json.loads(instance))

    def padding(self, batch: List[PzeroMaskedInstance]):
        return create_pzero_batch_instance(batch, self.padding_value)


class PasBucketIterator(PaddingBucketIterator):
    """Generate padded mini-batches for ZAR (PAS) task"""

    def __init__(
            self,
            file_path: Optional[str] = None,
            batch_size: int = 128,
            n_max_tokens: int = None,
            padding_value: int = 0,
            shuffle: bool = False,
            data_size_percentage: float = 100,
            max_seq_length: int = 512,
            model_type: str = AS_PZERO,  # 'as' or 'as-pzero'
            dataset: Optional[List[Union[AsTrainingInstance, AsPzeroTrainingInstance]]] = None,
    ):
        """
        Args:
            * file_path (Optional[str]): Path to the file containing 'instances.PzeroMaskedInstance' for each line.
            * batch_size (int): The number of instances in one mini-batch
            * n_max_tokens (int): The maximum number of tokens contained in one mini-batch
            * padding_value (int): The embedding id of a padding token.
            * shuffle (bool): Whether to shuffle the dataset
            * data_size_percentage (float): The percentage of the dataset to use for training
            * max_seq_length (int): The maximum number of input tokens for the model
            * model_type (str): Model type ("as" or "as-pzero")
            * dataset (Optional[List[Union[AsTrainingInstance, AsPzeroTrainingInstance]]]): Instead of file_path
        """
        super(PasBucketIterator, self).__init__(
            sort_key=self.sort_key,
            batch_size=batch_size,
            n_max_tokens=n_max_tokens,
            padding_value=padding_value,
            shuffle=shuffle,
        )
        self.file_path = file_path
        self.data_size_percentage = data_size_percentage
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        assert self.model_type in [AS, AS_PZERO], f"unsupported value: {self.model_type}, select '{AS}' or '{AS_PZERO}'"

        if dataset:
            self.dataset = dataset
        else:
            self.load_dataset()
        self.create_iterator()

    @staticmethod
    def sort_key(instance: Dict):
        return len(instance["input_ids"])

    def load_dataset(self):
        assert path.exists(self.file_path), f"not found: {self.file_path}"
        self.dataset: List[Union[AsTrainingInstance, AsPzeroTrainingInstance]] = []
        logger.info("Loading file: '{}'".format(self.file_path))
        len_file = count_n_lines(self.file_path)
        logger.info("The number of lines: {}".format(len_file))
        logger.info("The percentage of using data: {}%".format(self.data_size_percentage))

        for jsonl in tqdm(islice(read_lines(self.file_path), int(len_file * self.data_size_percentage / 100))):
            instance = json.loads(jsonl)
            instance['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(instance['eval_info']['sw2w_position'])
            self.dataset.append(instance)

        logger.info("The number of instances: {}".format(len(self.dataset)))
        logger.info("The number of batches: {}".format(len(self)))

    def padding(self, batch: List[Union[AsTrainingInstance, AsPzeroTrainingInstance]]):
        if self.model_type == AS:
            return create_as_batch_instance(batch, self.padding_value)
        elif self.model_type == AS_PZERO:
            return create_as_pzero_batch_instance(batch, self.padding_value)
        else:
            raise ValueError(f"unsupported value: {self.model_type}")
