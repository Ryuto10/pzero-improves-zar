import sys
from os import path

import unittest
from logzero import logger

from src.schedulers import TrainingScheduler, MyLRScheduler


class TestTrainingScheduler(unittest.TestCase):
    def test_not_over_max_epoch(self):
        start_epoch = 0
        max_epoch = 30
        early_stopping_thres = 3

        performances = [
            0.3, 0.6, 0.83,  # best
            0.826, 0.80, 0.82,  # over threshold
            0.84,  # best
            0.83, 0.835, 0.826,  # over threshold
            0.832, 0.831, 0.83,  # over threshold
        ]
        expected = [
            (True, False, False), (True, False, False), (True, False, False),
            (False, False, False), (False, False, False), (False, True, False),
            (True, False, False),
            (False, False, False), (False, False, False), (False, True, False),
            (False, False, False), (False, False, False), (False, True, False),
        ]

        training_scheduler = TrainingScheduler(
            start_epoch=start_epoch,
            max_epoch=max_epoch,
            early_stopping_thres=early_stopping_thres,
        )
        for perf, (exp_is_best, exp_is_over_thres, exp_is_max_epoch) in zip(performances, expected):
            act_is_best, act_is_over_thre, act_is_max_epoch = training_scheduler(perf)
            assert act_is_best == exp_is_best
            assert act_is_over_thre == exp_is_over_thres
            assert act_is_max_epoch == exp_is_max_epoch

    def test_over_max_epoch(self):
        start_epoch = 0
        max_epoch = 10
        early_stopping_thres = 3

        performances = [
            0.3, 0.6, 0.83,  # best
            0.82, 0.825, 0.829,  # over threshold
            0.826, 0.80, 0.82,  # over threshold
            0.81,  # end of epoch
            0.88,  # end of epoch & best & warning message is printed
        ]
        expected = [
            (True, False, False), (True, False, False), (True, False, False),
            (False, False, False), (False, False, False), (False, True, False),
            (False, False, False), (False, False, False), (False, True, False),
            (False, False, True),
            (True, False, True),
        ]

        training_scheduler = TrainingScheduler(
            start_epoch=start_epoch,
            max_epoch=max_epoch,
            early_stopping_thres=early_stopping_thres,
        )

        for perf, (exp_is_best, exp_is_over_thres, exp_is_max_epoch) in zip(performances, expected):
            act_is_best, act_is_over_thre, act_is_max_epoch = training_scheduler(perf)
            assert act_is_best == exp_is_best
            assert act_is_over_thre == exp_is_over_thres
            assert act_is_max_epoch == exp_is_max_epoch


class TestMyLRScheduler(unittest.TestCase):
    def test_reduce_half(self):
        start_lr = 1e-3
        min_lr = 5e-5
        lr_scheduler = MyLRScheduler(
            optimizer_states={},
            start_lr=start_lr,
            min_lr=min_lr,
            ratio_reduce_lr=0.5,
        )
        expected = [5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 5e-5, None]

        for idx, exp_lr in enumerate(expected):
            act_lr = lr_scheduler.get_state()
            if act_lr is None:
                assert act_lr == exp_lr
            else:
                assert act_lr["lr"] == exp_lr
