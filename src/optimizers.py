from itertools import tee
from typing import Dict, List

import torch.nn as nn
from torch.optim import Adam, AdamW


def get_optimizer(
        model: nn.Module,
        optimizer_type: str = "adam",
        adam_epsilon: float = 1e-08,
        learning_rate: float = 2e-05,
        weight_decay: float = 0.0,
        additional_params_dicts: List[Dict] = None,
):
    """Create an optimizer"""
    additional_params = []
    if additional_params_dicts:
        for p_dict in additional_params_dicts:
            copy_generator, ori_generator = tee(p_dict["params"], 2)
            additional_params += list(copy_generator)
            p_dict['params'] = ori_generator
    additional_params = set(additional_params)

    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        grouped_params = [
            {
                'params': [p for n, p in model.named_parameters() if p not in additional_params]
            }
        ]
        if additional_params_dicts:
            grouped_params += additional_params_dicts

        optimizer = Adam(
            grouped_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=adam_epsilon,
            weight_decay=weight_decay,
            amsgrad=False
        )

    elif optimizer_type == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_params = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p not in no_decay
                ],
                'weight_decay': weight_decay
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p not in no_decay
                ],
                'weight_decay': 0.0
            }
        ]
        if additional_params_dicts:
            grouped_params += additional_params_dicts

        optimizer = AdamW(
            grouped_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=weight_decay,
            amsgrad=False
        )

    else:
        raise ValueError("unsupported value: {}".format(optimizer_type))

    return optimizer
