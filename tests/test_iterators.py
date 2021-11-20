# coding=utf-8

import json
import random
import unittest

import numpy as np
import torch
from logzero import logger

from src.instances import AS, AS_PZERO
from src.iterators import (
    ClozeTaskDataset,
    create_cloze_task_dataloader,
    PzeroBucketIterator,
    PasBucketIterator,
)

random.seed(172)
np.random.seed(190)
torch.manual_seed(238)


class TestCreateClozeTaskDataLoader(unittest.TestCase):
    file = "tests/samples/cloze.instances.jsonl"

    instances = [json.loads(line) for line in open(file)]
    assert len(instances) == 2

    instances = instances * 10
    assert len(instances) == 20

    dataset = ClozeTaskDataset(instances=instances)

    def test_batch_size_with_single_gpu(self):
        batch_size = 7

        data_loader = create_cloze_task_dataloader(
            batch_size=batch_size,
            pad_token_id=0,
            n_gpu=1,
            is_eval=False,
            dataset=self.dataset,
        )

        for idx, batch in enumerate(data_loader, 1):
            if idx == len(data_loader):
                assert len(batch) == 6  # 20 % 7
            else:
                assert len(batch) == batch_size
        assert idx == len(data_loader)

    def test_batch_size_with_multi_gpu(self):
        batch_size = 2
        n_gpu = 3

        data_loader = create_cloze_task_dataloader(
            batch_size=batch_size,
            pad_token_id=0,
            n_gpu=n_gpu,
            is_eval=False,
            dataset=self.dataset,
        )

        for idx, batch in enumerate(data_loader, 1):
            if idx == len(data_loader):
                assert len(batch) == 2  # 20 % 6
            else:
                assert len(batch) == batch_size * n_gpu
        assert idx == len(data_loader)


class TestPzeroBucketIterator(unittest.TestCase):
    file = "tests/samples/pzero.instances.jsonl"
    datasets = [json.loads(line) for line in open(file)]
    assert len(datasets) == 15

    def test_fixed_batch_size(self):
        batch_size = 4
        data_loader = PzeroBucketIterator(
            batch_size=batch_size,
            shuffle=True,
            dataset=self.datasets,
        )
        logger.debug(f"fixed batch size: {batch_size}")
        for idx, batch in enumerate(data_loader, 1):
            batch_size, seq_length = batch['input_ids'].shape
            logger.debug(f"{batch['input_ids'].shape} (batch size = {batch_size})")
            if idx == 1:
                assert len(batch['input_ids']) == 3  # 15 % 4
            else:
                assert len(batch['input_ids']) == batch_size
        assert idx == len(data_loader)

    def test_max_tokens(self):
        n_max_tokens = 1000
        data_loader = PzeroBucketIterator(
            n_max_tokens=n_max_tokens,
            shuffle=True,
            dataset=self.datasets,
        )
        logger.debug(f"max tokens: {n_max_tokens}")
        for batch in data_loader:
            batch_size, seq_length = batch['input_ids'].shape
            n_tokens = batch_size * seq_length
            logger.debug(f"{batch['input_ids'].shape} (number of tokens = {n_tokens})")
            assert n_tokens <= n_max_tokens


class TestPasBucketIterator(unittest.TestCase):
    as_file = "tests/samples/as.instances.jsonl"
    as_dataset = [json.loads(line) for line in open(as_file)]
    assert len(as_dataset) == 5

    as_pzero_file = "tests/samples/as-pzero.instances.jsonl"
    as_pzero_dataset = [json.loads(line) for line in open(as_pzero_file)]
    assert len(as_pzero_dataset) == 15

    def test_as_fixed_batch_size(self):
        batch_size = 4
        data_loader = PasBucketIterator(
            batch_size=batch_size,
            shuffle=True,
            dataset=self.as_dataset,
            model_type=AS,
        )

        logger.debug(f"fixed batch size: {batch_size}")
        for idx, batch in enumerate(data_loader, 1):
            batch_size, seq_length = batch['input_ids'].shape
            logger.debug(f"{batch['input_ids'].shape} (batch size = {batch_size})")
            if idx == len(data_loader):
                assert len(batch['input_ids']) == 1  # 5 % 4
            else:
                assert len(batch['input_ids']) == batch_size
        assert idx == len(data_loader)

    def test_as_pzero_fixed_batch_size(self):
        batch_size = 4
        data_loader = PasBucketIterator(
            batch_size=batch_size,
            shuffle=True,
            dataset=self.as_pzero_dataset,
            model_type=AS_PZERO,
        )

        logger.debug(f"fixed batch size: {batch_size}")
        for idx, batch in enumerate(data_loader, 1):
            batch_size, seq_length = batch['input_ids'].shape
            logger.debug(f"{batch['input_ids'].shape} (batch size = {batch_size})")
            if idx == 1:
                assert len(batch['input_ids']) == 3  # 15 % 4
            else:
                assert len(batch['input_ids']) == batch_size
        assert idx == len(data_loader)

    def test_as_max_tokens(self):
        n_max_tokens = 200
        data_loader = PasBucketIterator(
            n_max_tokens=n_max_tokens,
            shuffle=True,
            dataset=self.as_dataset,
            model_type=AS,
        )
        logger.debug(f"max tokens: {n_max_tokens}")
        for batch in data_loader:
            batch_size, seq_length = batch['input_ids'].shape
            n_tokens = batch_size * seq_length
            logger.debug(f"{batch['input_ids'].shape} (number of tokens = {n_tokens})")
            assert n_tokens <= n_max_tokens

    def test_as_pzero_max_tokens(self):
        n_max_tokens = 200
        data_loader = PasBucketIterator(
            n_max_tokens=n_max_tokens,
            shuffle=True,
            dataset=self.as_pzero_dataset,
            model_type=AS_PZERO,
        )
        logger.debug(f"max tokens: {n_max_tokens}")
        for batch in data_loader:
            batch_size, seq_length = batch['input_ids'].shape
            n_tokens = batch_size * seq_length
            logger.debug(f"{batch['input_ids'].shape} (number of tokens = {n_tokens})")
            assert n_tokens <= n_max_tokens
