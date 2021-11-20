# coding: utf-8

from typing import List

import torch
import torch.nn as nn

from instances import (
    PzeroBatchInstance,
    PzeroOutputForLoss,
    AsBatchInstance,
    AsOutputForLoss,
    AsPzeroBatchInstance,
    AsPzeroOutputForLoss,
    TARGET_CASES,
    LossFunction,
)


class LossFunctionForPzero(LossFunction):
    def __init__(self):
        super().__init__()

    def compute_loss(
            self,
            batch: PzeroBatchInstance,
            predicts: PzeroOutputForLoss,
            device: torch.device = None,
    ) -> torch.Tensor:
        """Calculate the loss for pzero task"""

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output = predicts["selection_scores"].to(device)
        golds = create_gold_sequences(
            batch_gold_ids=batch["gold_ids"],
            seq_length=batch["input_ids"].shape[-1],
            prob=True,
        )
        golds = golds.to(device)
        loss = loss_kl_divergence(output, golds)

        return loss


class LossFunctionForAs(LossFunction):
    def __init__(self):
        super().__init__()

    def compute_loss(
            self,
            batch: AsBatchInstance,
            predicts: AsOutputForLoss,
            device: torch.device = None,
    ) -> torch.Tensor:
        """Calculate the loss for as model"""

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        label_scores = predicts["label_scores"].to(device)
        exo_scores = predicts["exo_scores"].to(device)

        # padding
        batch_size, seq_length = batch["input_ids"].shape
        label_scores = label_scores[:, :, :-1].transpose(1, 2).reshape(-1, seq_length)

        # golds for labels
        gold_positions = [
            gold_position_dict[case_name] for gold_position_dict in batch["gold_positions"] for case_name in
            TARGET_CASES
        ]
        golds = create_gold_sequences(
            batch_gold_ids=gold_positions,
            seq_length=seq_length,
            prob=True,
        )
        golds = golds.to(device)
        loss = loss_kl_divergence(label_scores, golds)

        # golds for exophora
        exo_scores = exo_scores.view(-1, 4)
        exo_ids = torch.LongTensor([exo_dic[case_name] for exo_dic in batch["exo_ids"] for case_name in TARGET_CASES])
        exo_ids = exo_ids.to(device)
        exo_loss = loss_cross_entropy(exo_scores, exo_ids)

        loss = loss + exo_loss

        return loss


class LossFunctionForAsPzero(LossFunction):
    def __init__(self):
        super().__init__()

    def compute_loss(
            self,
            batch: AsPzeroBatchInstance,
            predicts: AsPzeroOutputForLoss,
            device: torch.device = None,
    ) -> torch.Tensor:
        """Calculate the loss for as-pzero model"""

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sequence_scores = predicts["selection_scores"].to(device)
        exo_scores = predicts["exo_scores"].to(device)

        # golds for selection
        batch_size, seq_length = batch["input_ids"].shape
        golds = create_gold_sequences(
            batch_gold_ids=batch["gold_positions"],
            seq_length=seq_length,
            prob=True,
        )
        golds = golds.to(device)
        loss = loss_kl_divergence(sequence_scores, golds)

        # golds for exophora
        exo_ids = batch["exo_ids"].to(device)
        exo_loss = loss_cross_entropy(exo_scores, exo_ids)

        loss = loss + exo_loss

        return loss


def loss_cross_entropy(model_output, golds):
    assert model_output.shape[:-1] == golds.shape
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(
        model_output.view(-1, model_output.shape[-1]), golds.view(-1))

    return loss


def loss_kl_divergence(model_output, golds) -> torch.Tensor:
    assert model_output.shape == golds.shape
    dim = len(model_output.shape) - 1
    log_softmax = nn.LogSoftmax(dim=dim)
    loss_function = nn.KLDivLoss(reduction="sum")

    predicts = log_softmax(model_output)
    loss = loss_function(predicts, golds)

    return loss


def loss_bce_with_logits(model_output, golds) -> torch.Tensor:
    assert model_output.shape == golds.shape
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    loss = loss_function(model_output, golds)

    return loss


def create_gold_sequences(
        batch_gold_ids: List[List[int]],
        seq_length: int,
        prob: bool = False,
        cuda: bool = False
) -> torch.Tensor:
    """create gold sequences"""
    gold_sequences = torch.zeros(len(batch_gold_ids), seq_length)
    for batch_idx, gold_ids in enumerate(batch_gold_ids):
        if prob:
            gold_sequences[batch_idx, gold_ids] = 1 / len(gold_ids)
        else:
            gold_sequences[batch_idx, gold_ids] = 1
    if cuda:
        gold_sequences = gold_sequences.cuda()

    return gold_sequences
