# coding=utf-8

import unittest

import torch
from logzero import logger

from src.instances import (
    PzeroOutputForLoss,
    AsOutputForLoss,
    AsPzeroOutputForLoss,
)
from src.loss import (
    LossFunctionForPzero,
    LossFunctionForAs,
    LossFunctionForAsPzero,
    loss_cross_entropy,
    loss_kl_divergence,
    loss_bce_with_logits,
    create_gold_sequences,
)
from tests.mock_instances import mock_pzero_batch_instance, mock_as_batch_instance, mock_as_pzero_batch_instance


class TestLossFunctionForPzero(unittest.TestCase):
    loss_function = LossFunctionForPzero()

    batch_size = 2
    max_seq_length = 15
    batch = mock_pzero_batch_instance

    selection_scores_all_ones = torch.ones(batch_size, max_seq_length)
    selection_scores_close_to_gold: torch.Tensor = selection_scores_all_ones * 1e-04
    selection_scores_close_to_gold[0, 1] = 1e+4
    selection_scores_close_to_gold[1, 3] = 1e+4
    predicts_1 = PzeroOutputForLoss(selection_scores=selection_scores_all_ones)
    predicts_2 = PzeroOutputForLoss(selection_scores=selection_scores_close_to_gold)

    def test_on_cpu(self):
        device = torch.device('cpu')
        loss_1 = self.loss_function(self.batch, self.predicts_1, device)
        loss_2 = self.loss_function(self.batch, self.predicts_2, device)
        torch.testing.assert_allclose(loss_1, 5.4161)
        torch.testing.assert_allclose(loss_2, 0)

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            loss_1 = self.loss_function(self.batch, self.predicts_1, device)
            loss_2 = self.loss_function(self.batch, self.predicts_2, device)
            torch.testing.assert_allclose(loss_1, 5.4161)
            torch.testing.assert_allclose(loss_2, 0)
        else:
            logger.warning('Not tested on GPU')


class TestLossFunctionForAs(unittest.TestCase):
    loss_function = LossFunctionForAs()

    batch_size = 2
    max_seq_length = 21
    batch = mock_as_batch_instance

    label_scores_all_ones = torch.ones(batch_size, max_seq_length, 4)
    exo_scores_all_ones = torch.ones(batch_size, 12)
    label_scores_close_to_golds: torch.Tensor = label_scores_all_ones * 1e-4
    for p in [(0, 6, 0), (0, 2, 1), (0, 0, 2), (1, 6, 0), (1, 12, 1), (1, 0, 2)]:
        label_scores_close_to_golds[p] = 1e+4
    exo_scores_close_to_golds: torch.Tensor = exo_scores_all_ones * 1e-4
    for p in [(0, 8), (1, 8)]:
        exo_scores_close_to_golds[p] = 1e+4

    predicts_1 = AsOutputForLoss(label_scores=label_scores_all_ones, exo_scores=exo_scores_all_ones)
    predicts_2 = AsOutputForLoss(label_scores=label_scores_close_to_golds, exo_scores=exo_scores_close_to_golds)

    def test_on_cpu(self):
        device = torch.device('cpu')
        loss_1 = self.loss_function(self.batch, self.predicts_1, device)
        loss_2 = self.loss_function(self.batch, self.predicts_2, device)
        torch.testing.assert_allclose(loss_1, 19.6534)
        torch.testing.assert_allclose(loss_2, 0)

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            loss_1 = self.loss_function(self.batch, self.predicts_1, device)
            loss_2 = self.loss_function(self.batch, self.predicts_2, device)
            torch.testing.assert_allclose(loss_1, 19.6534)
            torch.testing.assert_allclose(loss_2, 0)
        else:
            logger.warning('Not tested on GPU')


class TestLossFunctionForAsPzero(unittest.TestCase):
    loss_function = LossFunctionForAsPzero()
    batch_size = 2
    max_seq_length = 24
    batch = mock_as_pzero_batch_instance

    selection_scores_all_ones = torch.ones(batch_size, max_seq_length)
    exo_scores_all_ones = torch.ones(batch_size, 4)
    selection_scores_close_to_golds: torch.Tensor = selection_scores_all_ones * 1e-4
    selection_scores_close_to_golds[0, 6] = 1e+4
    selection_scores_close_to_golds[1, 2] = 1e+4

    predicts_1 = AsPzeroOutputForLoss(selection_scores=selection_scores_all_ones, exo_scores=exo_scores_all_ones)
    predicts_2 = AsPzeroOutputForLoss(selection_scores=selection_scores_close_to_golds, exo_scores=exo_scores_all_ones)

    def test_on_cpu(self):
        device = torch.device('cpu')
        loss_1 = self.loss_function(self.batch, self.predicts_1, device)
        loss_2 = self.loss_function(self.batch, self.predicts_2, device)
        torch.testing.assert_allclose(loss_1, 6.3561)
        torch.testing.assert_allclose(loss_2, 0)

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            loss_1 = self.loss_function(self.batch, self.predicts_1, device)
            loss_2 = self.loss_function(self.batch, self.predicts_2, device)
            torch.testing.assert_allclose(loss_1, 6.3561)
            torch.testing.assert_allclose(loss_2, 0)
        else:
            logger.warning('Not tested on GPU')


class TestLossCrossEntropy(unittest.TestCase):
    def test_loss_is_zero(self):
        predicts = torch.ones(2, 3, 4) * 1e-4
        gold_positions = [[0, 1, 2], [3, 2, 1]]
        for batch_idx, sequence in enumerate(gold_positions):
            for seq_idx, embed_idx in enumerate(sequence):
                predicts[batch_idx, seq_idx, embed_idx] = 1e+4
        golds = torch.LongTensor(gold_positions)

        loss = loss_cross_entropy(predicts, golds)
        torch.testing.assert_allclose(loss, 0)

    def test_loss_is_not_zero(self):
        predicts = torch.ones(2, 3, 4)
        golds = torch.LongTensor([[0, 1, 2], [3, 2, 1]])

        loss = loss_cross_entropy(predicts, golds)
        torch.testing.assert_allclose(loss, 1.3863)


class TestLossKLDivergence(unittest.TestCase):
    def test_loss_is_zero_one_flag(self):
        predicts = torch.ones(2, 3) * 1e-4
        golds = torch.zeros(2, 3)

        for p in [(0, 1), (1, 2)]:
            predicts[p] = 1e+4
            golds[p] = 1

        loss = loss_kl_divergence(predicts, golds)
        torch.testing.assert_allclose(loss, 0)

    def test_loss_is_zero_two_flag(self):
        predicts = torch.ones(2, 3) * 1e-4
        golds = torch.zeros(2, 3)

        for p in [(0, 1), (0, 2), (1, 0), (1, 2)]:
            predicts[p] = 1e+4
            golds[p] = 0.5

        loss = loss_kl_divergence(predicts, golds)
        torch.testing.assert_allclose(loss, 0)

    def test_loss_is_not_zero(self):
        predicts = torch.ones(2, 3)
        golds = torch.zeros(2, 3)

        for p in [(0, 1), (1, 2)]:
            golds[p] = 1

        loss = loss_kl_divergence(predicts, golds)
        torch.testing.assert_allclose(loss, 2.1972)


class TestLossBceWithLogits(unittest.TestCase):
    def test_loss_is_zero(self):
        predicts = torch.ones(2, 3) * -1e+4
        golds = torch.zeros(2, 3)

        for p in [(0, 1), (1, 2)]:
            predicts[p] = 1e+4
            golds[p] = 1

        loss = loss_bce_with_logits(predicts, golds)
        torch.testing.assert_allclose(loss, 0)

    def test_loss_is_not_zero(self):
        predicts = torch.ones(2, 3)
        golds = torch.zeros(2, 3)

        for p in [(0, 1), (1, 2)]:
            golds[p] = 1

        loss = loss_bce_with_logits(predicts, golds)
        torch.testing.assert_allclose(loss, 5.8796)


class TestCreateGoldSequence(unittest.TestCase):
    batch_gold_ids = [[1, 2], [0]]
    seq_length = 3

    def test_create_golds_for_kl(self):
        actual = create_gold_sequences(self.batch_gold_ids, self.seq_length, prob=True)
        expected = torch.Tensor([[0, 0.5, 0.5], [1, 0, 0]])
        torch.testing.assert_allclose(actual, expected)

    def test_create_golds_for_bce(self):
        actual = create_gold_sequences(self.batch_gold_ids, self.seq_length, prob=False)
        expected = torch.Tensor([[0, 1, 1], [1, 0, 0]])
        torch.testing.assert_allclose(actual, expected)
