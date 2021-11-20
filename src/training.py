# coding=utf-8

import argparse
import logging
import os
import random
import shutil
from dataclasses import dataclass
from glob import glob
from os import path
from typing import Dict, List, Any, Set, Union

import logzero
import numpy as np
import pytorch_warmup
import torch
import torch.nn as nn
import yaml
from logzero import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from evaluating import compute_pzero_f_score, compute_pas_f_score, generate_pas_evaluation_instance
from instances import CLOZE, PZERO, AS, AS_PZERO
from iterators import create_cloze_task_dataloader, PzeroBucketIterator, PasBucketIterator
from loss import LossFunctionForPzero, LossFunctionForAs, LossFunctionForAsPzero
from models import PzeroModelForPreTraining, AsModel, AsPzeroModel, BertForPAS, CL_TOHOKU_BERT
from optimizers import get_optimizer
from schedulers import TrainingScheduler, MyLRScheduler
from tokenizer import load_tokenizer
from utils_pytorch import DataParallel as MyDataParallel

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

CHECKPOINT_PREFIX = "checkpoint"
TENSOR_BOARD_LOGFILE = "tensorboard_log"
CLOZE_MODEL_BASENAME = "pytorch_model.bin"
PZERO_MODEL_BASENAME = "best_pzero_model.bin"
PAS_MODEL_BASENAME = "best.model"
PAS_CHECKPOINT_BASENAME = "checkpoint.last"
EVAL_RESULT_BASENAME = "eval_results.txt"
LOG_FILE_BASENAME = "logzero.logs.txt"


@dataclass
class PretrainingArgs:
    """The parameters will be loaded from a yaml file"""
    # load and save
    output_dir: path.abspath
    train_data_file: path.abspath
    dev_data_file: path.abspath

    cache_dir: str = None
    model_name_or_path: str = None
    config_name: str = None
    tokenizer_name: str = None
    should_continue: bool = False
    overwrite_output_dir: bool = False
    overwrite_cache: bool = False
    logging_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = None

    # device
    fp16: bool = False
    fp16_opt_level: str = "O1"
    no_cuda: bool = False

    # hyper-parameters
    n_instances: int = -1  # the number of instances for training
    block_size: int = -1  # length of model inputs
    per_gpu_train_batch_size: int = 4
    per_gpu_eval_batch_size: int = 4
    per_gpu_train_max_tokens: int = None
    per_gpu_eval_max_tokens: int = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_steps: int = 0
    seed: int = 42

    # model specific parameters
    model_type: str = CLOZE  # select 'cloze' or 'pzero'
    mlm_probability: float = 0.15  # cloze task

    # additional parameters
    n_gpu: int = None
    device: torch.device = None
    model_max_length: int = None
    n_vocab: int = None
    cls_token_id: int = None
    sep_token_id: int = None
    mask_token_id: int = None
    pad_token_id: int = None
    special_token_ids: Set[int] = None
    t_total: float = None

    assert model_type in [CLOZE, PZERO], f"'model_type' must be '{CLOZE}' or '{PZERO}', not {model_type}"

    def set_additional_parameters(self) -> None:
        # Error checking
        assert path.exists(self.train_data_file), f"not found: {self.train_data_file}"
        assert path.exists(self.dev_data_file), f"not found: {self.dev_data_file}"

        if self.should_continue:
            sorted_checkpoints = sort_checkpoints(self.output_dir)
            if len(sorted_checkpoints) == 0:
                raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
            else:
                self.model_name_or_path = sorted_checkpoints[-1]

        if (
                path.exists(self.output_dir)
                and os.listdir(self.output_dir)
                and not self.overwrite_output_dir
                and not self.should_continue
        ):
            raise ValueError(f"Output directory already exists and is not empty: {self.output_dir}. "
                             " Use --overwrite_output_dir to overcome.")
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()

        # Setup logging
        logzero.loglevel(logging.INFO)
        logzero.logfile(path.join(self.output_dir, LOG_FILE_BASENAME))
        logger.warning(
            f"device: {self.device}, "
            f"n_gpu: {self.n_gpu}, "
            f"16-bits training: {self.fp16}",
        )

        # tokenize params
        if self.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, cache_dir=self.cache_dir)
        elif self.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir)
        else:
            raise ValueError("You are instantiating a new tokenizer from scratch. "
                             "This is not supported, but you can do it from another script, save it,"
                             "and load it from here, using --tokenizer_name")
        self.model_max_length = tokenizer.model_max_length
        self.n_vocab = len(tokenizer.vocab)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.special_token_ids = {self.cls_token_id, self.sep_token_id}

        if self.block_size <= 0:
            self.block_size = self.model_max_length
        else:
            self.block_size = min(self.block_size, self.model_max_length)

        if self.per_gpu_train_max_tokens:
            self.per_gpu_train_batch_size = self.per_gpu_train_max_tokens
        if self.per_gpu_eval_max_tokens:
            self.per_gpu_eval_batch_size = self.per_gpu_eval_max_tokens

        self.learning_rate = float(self.learning_rate)
        self.adam_epsilon = float(self.adam_epsilon)


@dataclass
class FinetuningArgs:
    """The parameters will be loaded from a yaml file"""
    # load and save
    output_dir: path.abspath
    train_data_file: path.abspath
    dev_data_file: path.abspath

    pretrained_model_path: path.abspath = None
    should_continue: bool = False
    skip_save_checkpoint: bool = False

    # device
    fp16: bool = False
    fp16_opt_level: str = "O1"
    no_cuda: bool = False

    # hyper-parameters
    seed: int = 0
    num_train_epochs: int = 150
    data_size_percentage: float = 100
    per_gpu_train_batch_size: int = 32
    per_gpu_eval_batch_size: int = 32
    per_gpu_train_max_tokens: int = None
    per_gpu_eval_max_tokens: int = None
    gradient_accumulation_steps: int = 1
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    learning_rate: float = 2e-5
    bert_learning_rate: float = None
    logit_learning_rate: float = None
    min_learning_rate: float = None
    learning_scheduler: str = "reduce_half"  # please select 'reduce_half' or 'warmup'
    early_stopping: int = 5  # if 0, don't stop training until maximum epoch
    embed_dropout: float = 0.0
    layer_norm_eps: float = 1e-5

    # model specific parameters
    model_type: str = AS  # select 'as' or 'as-pzero'

    # additional parameters
    n_gpu: int = None
    device: torch.device = None
    pad_token_id: int = None

    assert learning_scheduler in ["reduce_half", "warmup"], f"Unsupported value: {learning_scheduler}"
    assert model_type in [AS, AS_PZERO], f"Unsupported value: {learning_scheduler}"

    def set_additional_parameters(self) -> None:
        assert path.exists(self.train_data_file), f"not found: {self.train_data_file}"
        assert path.exists(self.dev_data_file), f"not found: {self.dev_data_file}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = 0 if self.no_cuda else torch.cuda.device_count()

        # Setup logging
        logzero.loglevel(logging.INFO)
        logzero.logfile(path.join(self.output_dir, LOG_FILE_BASENAME))
        logger.warning(
            f"device: {self.device}, "
            f"n_gpu: {self.n_gpu}, "
            f"16-bits training: {self.fp16}",
        )

        self.learning_rate = float(self.learning_rate)
        self.adam_epsilon = float(self.adam_epsilon)
        if self.min_learning_rate is None:
            self.min_learning_rate = self.learning_rate / 20
        tokenizer = load_tokenizer()
        self.pad_token_id = tokenizer.pad_token_id


class TrainingComponents:
    def __init__(self):
        self.args = None
        self.writer = None
        self.train_data_loader = None
        self.dev_data_loader = None
        self.loss_function = None

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.warmup_scheduler = None
        self.training_scheduler = None
        self.amp = None

    def set_zero_grad(self) -> None:
        self.model.zero_grad()
        self.optimizer.zero_grad()

    def backward_loss(self, loss) -> torch.Tensor:
        if self.args.fp16:
            with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss

    def update_optimizer(self, use_clip_gradient: bool = True) -> None:
        if use_clip_gradient:
            if self.args.fp16:
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), self.args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()

        if self.warmup_scheduler:
            self.lr_scheduler.step(self.lr_scheduler.last_epoch + 1)
            self.warmup_scheduler.dampen()

        elif self.args.model_type in [CLOZE, PZERO]:
            self.lr_scheduler.step()

        self.model.zero_grad()

    def set_fp16(self, opt_level: str = "O1"):
        """Set to half-precision floating point number"""
        assert self.model is not None
        assert self.optimizer is not None

        if self.amp is None:
            try:
                from apex import amp
                self.amp = amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

    def set_pretraining_components(self, args: PretrainingArgs) -> None:
        args.set_additional_parameters()
        set_seed(n_seed=args.seed, n_gpu=args.n_gpu)

        self.writer = SummaryWriter(log_dir=path.join(args.output_dir, TENSOR_BOARD_LOGFILE))

        logger.info(f"Training/evaluation parameters {args}")

        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError("You are instantiating a new config instance from scratch. "
                             "This is not supported, but you can do it from another script, save it,"
                             "and load it from here, using --config_name")

        # loading data loader
        if args.model_type == PZERO:
            self.train_data_loader = PzeroBucketIterator(
                file_path=args.train_data_file,
                batch_size=args.per_gpu_train_batch_size,
                n_max_tokens=args.per_gpu_train_max_tokens,
                padding_value=args.pad_token_id,
                shuffle=True,
                n_instances=args.n_instances,
                max_seq_length=args.model_max_length,
            )
            self.dev_data_loader = PzeroBucketIterator(
                file_path=args.dev_data_file,
                batch_size=args.per_gpu_eval_batch_size,
                n_max_tokens=args.per_gpu_eval_max_tokens,
                padding_value=args.pad_token_id,
                max_seq_length=args.model_max_length,
            )

            if args.model_name_or_path:
                bert_model = AutoModel.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir
                )
            else:
                logger.info("Training new model from scratch")
                bert_model = AutoModel.from_config(config)

            bert_model.resize_token_embeddings(args.n_vocab)
            model = PzeroModelForPreTraining(
                loss_function=LossFunctionForPzero(),
                bert_model=bert_model,
                device=args.device,
            )

        elif args.model_type == CLOZE:
            self.train_data_loader = create_cloze_task_dataloader(
                file_path=args.train_data_file,
                batch_size=args.per_gpu_train_batch_size,
                pad_token_id=args.pad_token_id,
                n_gpu=args.n_gpu,
            )
            self.dev_data_loader = create_cloze_task_dataloader(
                file_path=args.dev_data_file,
                batch_size=args.per_gpu_eval_batch_size,
                pad_token_id=args.pad_token_id,
                n_gpu=args.n_gpu,
                is_eval=True,
            )

            if args.model_name_or_path:
                model = AutoModelWithLMHead.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    cache_dir=args.cache_dir,
                )
            else:
                logger.info("Training new model from scratch")
                model = AutoModelWithLMHead.from_config(config)
            model.resize_token_embeddings(args.n_vocab)

        else:
            raise ValueError(f"unsupported value: {args.model_type}")

        model.to(args.device)

        # load model, optimizer, learning_rate_scheduler
        self.model = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training

        # Prepare optimizer and schedule (linear warmup and decay)
        self.optimizer = get_optimizer(
            model=self.model,
            optimizer_type="adamw",
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # update args params !
        num_batches = len(self.train_data_loader)
        if args.max_steps > 0:
            args.t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (num_batches // args.gradient_accumulation_steps) + 1
        else:
            args.t_total = num_batches // args.gradient_accumulation_steps * args.num_train_epochs
        self.args = args

        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.t_total
        )

        # Check if saved optimizer or scheduler states exist
        if (
                args.model_name_or_path
                and path.isfile(path.join(args.model_name_or_path, "optimizer.pt"))
                and path.isfile(path.join(args.model_name_or_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(path.join(args.model_name_or_path, "optimizer.pt")))
            self.lr_scheduler.load_state_dict(torch.load(path.join(args.model_name_or_path, "scheduler.pt")))

        if args.fp16:
            self.set_fp16(args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            self.model = MyDataParallel(self.model) if args.model_type == PZERO else torch.nn.DataParallel(self.model)

    def set_finetuning_components(self, args: FinetuningArgs) -> None:
        args.set_additional_parameters()
        set_seed(n_seed=args.seed, n_gpu=args.n_gpu)

        self.args = args
        self.writer = SummaryWriter(log_dir=path.join(args.output_dir, TENSOR_BOARD_LOGFILE))

        # load dataset
        self.train_data_loader = PasBucketIterator(
            file_path=args.train_data_file,
            batch_size=args.per_gpu_train_batch_size,
            n_max_tokens=args.per_gpu_train_max_tokens,
            padding_value=args.pad_token_id,
            shuffle=True,
            data_size_percentage=args.data_size_percentage,
            model_type=args.model_type,
        )
        self.dev_data_loader = PasBucketIterator(
            file_path=args.dev_data_file,
            batch_size=args.per_gpu_eval_batch_size,
            n_max_tokens=args.per_gpu_eval_max_tokens,
            padding_value=args.pad_token_id,
            model_type=args.model_type,
        )

        # load model
        if args.pretrained_model_path:
            bert_model = None
        else:
            bert_model = BertForPAS.from_pretrained(CL_TOHOKU_BERT)

        if args.model_type == AS:
            model = AsModel(
                loss_function=LossFunctionForAs(),
                bert_model=bert_model,
                embed_dropout=args.embed_dropout,
                layer_norm_eps=args.layer_norm_eps
            )
        elif args.model_type == AS_PZERO:
            model = AsPzeroModel(
                loss_function=LossFunctionForAsPzero(),
                bert_model=bert_model,
                embed_dropout=args.embed_dropout,
                layer_norm_eps=args.layer_norm_eps
            )
        else:
            raise ValueError(f"unsupported value: {args.model_type}")

        if args.pretrained_model_path:
            self._load_pretrained_model(args.pretrained_model_path, model)

        if torch.cuda.is_available():
            model = model.cuda()

        # Optimizer
        additional_params_dicts = []
        if args.logit_learning_rate:
            if args.model_type == AS:
                additional_params_dicts.append(
                    {"params": model.linear.parameters(), "lr": args.logit_learning_rate}
                )
            elif args.model_type == AS_PZERO:
                additional_params_dicts.append(
                    {"params": model.sentence_self_attn.parameters(), "lr": args.logit_learning_rate}
                )
                additional_params_dicts.append(
                    {"params": model.cls.parameters(), "lr": args.logit_learning_rate}
                )
                additional_params_dicts.append(
                    {"params": model.exo_linear_layer.parameters(), "lr": args.logit_learning_rate}
                )
        if args.bert_learning_rate:
            additional_params_dicts.append(
                {"params": model.bert_model.parameters(), "lr": args.bert_learning_rate}
            )

        optimizer = get_optimizer(
            model=model,
            optimizer_type="adam",
            adam_epsilon=args.adam_epsilon,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            additional_params_dicts=additional_params_dicts,
        )

        self.model = model
        self.optimizer = optimizer

        if args.fp16:
            self.set_fp16()

        # learning rate scheduler
        if args.learning_scheduler == 'reduce_half':
            self.lr_scheduler = MyLRScheduler(
                optimizer_states={"lr": args.learning_rate, "weight_decay": args.weight_decay},
                start_lr=args.learning_rate,
                min_lr=args.min_learning_rate
            )
            self.warmup_scheduler = None

        elif args.learning_scheduler == 'warmup':
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=args.num_train_epochs * len(self.train_data_loader),
                eta_min=args.min_learning_rate
            )
            self.warmup_scheduler = pytorch_warmup.UntunedLinearWarmup(self.optimizer)
        else:
            raise ValueError("unsupported value: '{}'".format(args.learning_scheduler))

        # training scheduler
        self.training_scheduler = TrainingScheduler(
            start_epoch=0,
            max_epoch=args.num_train_epochs,
            early_stopping_thres=args.early_stopping
        )

        if args.should_continue:
            self._load_finetuning_components(path.join(args.output_dir, PAS_CHECKPOINT_BASENAME))

    def save_pretraining_components(self, output_dir: str, only_model: bool = False) -> None:
        os.makedirs(output_dir, exist_ok=True)

        if self.args.model_type == PZERO:
            torch.save(
                self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict(),
                path.join(output_dir, PZERO_MODEL_BASENAME)
            )
            model_to_save = self.model.module.bert_model if hasattr(self.model, "module") else self.model.bert_model
            model_to_save.save_pretrained(output_dir)

        elif self.args.model_type == CLOZE:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            model_to_save.save_pretrained(output_dir)

        torch.save(self.args, path.join(output_dir, "training_args.bin"))
        logger.info(f"Saving model checkpoint to {output_dir}")

        if not only_model:
            torch.save(self.optimizer.state_dict(), path.join(output_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), path.join(output_dir, "scheduler.pt"))
            logger.info(f"Saving optimizer and scheduler states to {output_dir}")

    def save_finetuning_components(self, output_dir: str) -> None:
        logger.info(f"Save last checkpoint to {output_dir}")

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler,
            "warmup_scheduler": self.warmup_scheduler,
            "training_scheduler": self.training_scheduler
        }
        if self.args.fp16:
            checkpoint["amp"] = self.amp.state_dict()

        torch.save(checkpoint, path.join(output_dir, PAS_CHECKPOINT_BASENAME))

    def _load_finetuning_components(self, file_path: str) -> None:
        assert path.exists(file_path), f"Not found: {file_path}"
        logger.debug("Load: {}".format(file_path))
        checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint["model"])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler = checkpoint["lr_scheduler"]
        self.warmup_scheduler = checkpoint["warmup_scheduler"]
        self.training_scheduler = checkpoint["training_scheduler"]
        if self.args.fp16:
            self.amp.load_state_dict(checkpoint['amp'])

    def _load_pretrained_model(self, checkpoint_file: str, model: Union[AsModel, AsPzeroModel]) -> None:
        """
        Args:
            checkpoint_file: file path (or dir path)
            model: SelectionModelForPAS or SequentialLabelingModelForPAS or SelectionModelForPreTraining
        """
        # update checkpoint_file
        if path.isdir(checkpoint_file):
            if path.exists(path.join(checkpoint_file, PZERO_MODEL_BASENAME)):
                checkpoint_file = path.join(checkpoint_file, PZERO_MODEL_BASENAME)
            elif path.exists(path.join(checkpoint_file, CLOZE_MODEL_BASENAME)):
                checkpoint_file = path.join(checkpoint_file, CLOZE_MODEL_BASENAME)
            else:
                raise RuntimeError("unsupported file: {}".format(checkpoint_file))

        logger.info("load checkpoint: {}".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

        if path.basename(checkpoint_file) == PZERO_MODEL_BASENAME:
            bert_params = extract_model_params(checkpoint, "bert_model.")
            if len(bert_params) != len(model.bert_model.state_dict()):
                bert_params = self._adjust_bert_params(bert_params, model, "embeddings.position_ids")
            assert len(bert_params) == len(model.bert_model.state_dict())

            model.bert_model.load_state_dict(bert_params)

            if isinstance(model, AsPzeroModel):
                selection_attn_params = extract_model_params(checkpoint, "selection_self_attn.")
                assert len(selection_attn_params) == len(model.selection_self_attn.state_dict())
                model.selection_self_attn.load_state_dict(selection_attn_params)
                logger.info("load: bert & logit layer parameters")

            elif isinstance(model, AsModel):
                logger.info("load: bert parameters")

            else:
                raise ValueError(f"unsupported model type: {type(model)}")

        elif path.basename(checkpoint_file) == CLOZE_MODEL_BASENAME:
            if len(checkpoint) == len(model.bert_model.state_dict()):
                bert_params = checkpoint
            else:
                bert_params = extract_model_params(checkpoint, "bert.")

            if len(bert_params) != len(model.bert_model.state_dict()):
                bert_params = self._adjust_bert_params(bert_params, model, "embeddings.position_ids")
            if len(bert_params) != len(model.bert_model.state_dict()):
                bert_params = self._adjust_bert_params(bert_params, model, "pooler.dense.weight")
                bert_params = self._adjust_bert_params(bert_params, model, "pooler.dense.bias")
            assert len(bert_params) == len(model.bert_model.state_dict())

            model.bert_model.load_state_dict(bert_params)
            logger.info("load: bert parameters")

        else:
            raise ValueError("unsupported file: {}".format(checkpoint_file))

    @staticmethod
    def _adjust_bert_params(bert_params: Dict[str, Any], model: nn.Module, target_key: str) -> Dict[str, Any]:
        if target_key in bert_params:
            bert_params.pop(target_key)
        else:
            bert_params[target_key] = model.bert_model.state_dict()[target_key]

        return bert_params


def extract_model_params(checkpoint: nn.Module, prefix: str) -> Dict[str, Any]:
    model_params = {
        param_name.replace(prefix, "", 1): params
        for param_name, params in checkpoint.items()
        if param_name.startswith(prefix)
    }

    return model_params


def set_seed(n_seed: int, n_gpu: int) -> None:
    random.seed(n_seed)
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(n_seed)
    logger.info(f"seed: {n_seed}")


def sort_checkpoints(output_dir: str, checkpoint_prefix: str = CHECKPOINT_PREFIX) -> List[str]:
    # sort by 'getmtime'
    ordering_and_checkpoint_path = [
        (path.getmtime(file_path), file_path)
        for file_path in glob(path.join(output_dir, "{}*".format(checkpoint_prefix)))
    ]
    checkpoints_sorted = [file_path for file_time, file_path in sorted(ordering_and_checkpoint_path)]

    return checkpoints_sorted


def rotate_checkpoints(
        output_dir: str,
        save_total_limit: int = None,
        checkpoint_prefix: str = CHECKPOINT_PREFIX
) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sort_checkpoints(output_dir, checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = len(checkpoints_sorted) - save_total_limit
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint: {}".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(
        inputs: torch.Tensor,
        not_replaced_embed_ids: Set[int],
        pad_token_id: int,
        mask_token_id: int,
        n_vocab: int,
        mlm_probability: float = 0.15
) -> (torch.Tensor, torch.Tensor):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    Args:
        * inputs (torch.Tensor): tensor with shape ('batch_size', 'seq_length')
                                 which is embedding ids without a masked token embedding id
        * not_replaced_embed_ids (Set[int]): embedding ids that will not be replaced
        * pad_token_id (int): embedding id of a padding token
        * mask_token_id (int): embedding id of a mask token
        * n_vocab (int): A number of vocabulary
        * mlm_probability (float): The probability of replacing tokens

    Returns:
        *  inputs (torch.Tensor): a mini-batch as inputs
        *  labels (torch.Tensor): gold labels
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [[1 if idx in not_replaced_embed_ids else 0 for idx in ids] for ids in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    padding_mask = labels.eq(pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(n_vocab, labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def pretraining(tc: TrainingComponents) -> None:
    """ Pretraining the model """
    args: PretrainingArgs = tc.args

    # Train!
    total_batch_size = args.per_gpu_train_batch_size * max(args.n_gpu, 1) * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(tc.train_data_loader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            loader_len = len(tc.train_data_loader)
            epochs_trained = global_step // (loader_len // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (loader_len // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {global_step}")
            logger.info(f"  Will skip the first {steps_trained_in_current_epoch} steps in the first epoch")
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    tc.model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
    )

    for _ in train_iterator:
        batches = []  # use in Pzero task
        if args.model_type == PZERO:
            tc.train_data_loader.create_batches()
        epoch_iterator = tqdm(tc.train_data_loader, desc="Iteration", disable=False)

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            tc.model.train()

            if args.model_type == PZERO:
                batches.append(batch)
                if len(batches) < args.n_gpu:
                    continue
                if args.n_gpu <= 1:
                    assert len(batches) == 1
                    batches = batches[0]
                loss = tc.model(batches)
                batches = []

            elif args.model_type == CLOZE:
                inputs, labels = mask_tokens(
                    inputs=batch,
                    not_replaced_embed_ids=args.special_token_ids,
                    pad_token_id=args.pad_token_id,
                    mask_token_id=args.mask_token_id,
                    n_vocab=args.n_vocab,
                    mlm_probability=args.mlm_probability,
                )
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                outputs = tc.model(inputs, masked_lm_labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            else:
                raise ValueError(f"unsupported value: {args.model_type}")

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss = tc.backward_loss(loss)
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                tc.update_optimizer()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info("**** logging ****")
                    logger.info(f"**** current steps = {global_step} ****")
                    results = evaluate_pretraining(tc)
                    for key, value in results.items():
                        tc.writer.add_scalar("eval_{}".format(key), value, global_step)
                    tc.writer.add_scalar("lr", tc.lr_scheduler.get_lr()[0], global_step)
                    tc.writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    logger.info("**** save ****")
                    logger.info(f"**** current steps = {global_step} ****")

                    # Save model checkpoint
                    output_dir = path.join(args.output_dir, "{}-{}".format(CHECKPOINT_PREFIX, global_step))
                    tc.save_pretraining_components(output_dir)
                    rotate_checkpoints(
                        args.output_dir,
                        save_total_limit=args.save_total_limit,
                        checkpoint_prefix=CHECKPOINT_PREFIX
                    )

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tc.writer.close()

    logger.info(f" global_step = {global_step}, average loss = {tr_loss / global_step}")
    tc.save_pretraining_components(args.output_dir, only_model=True)


def evaluate_pretraining(tc: TrainingComponents) -> Dict[str, Any]:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    args: PretrainingArgs = tc.args

    if args.model_type == PZERO:
        tc.dev_data_loader.create_batches()
    eval_dataloader = tc.dev_data_loader

    eval_output_dir = args.output_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataloader.dataset))
    eval_loss = 0.0
    nb_eval_steps = 0
    pzero_preds = []
    pzero_golds = []
    batches = []
    special_token_ids = {args.cls_token_id, args.sep_token_id}

    tc.model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            if args.model_type == PZERO:
                batches.append(batch)
                if len(batches) < args.n_gpu:
                    continue
                if args.n_gpu <= 1:
                    assert len(batches) == 1
                    batches = batches[0]
                loss = tc.model(batches)
                batches = []

                preds = tc.model.module.inference(batch) if hasattr(tc.model, "module") else tc.model.inference(batch)
                pzero_preds += preds["selection_positions"].tolist()
                pzero_golds += batch["gold_ids"]

            elif args.model_type == CLOZE:
                inputs, labels = mask_tokens(
                    inputs=batch,
                    not_replaced_embed_ids=special_token_ids,
                    pad_token_id=args.pad_token_id,
                    mask_token_id=args.mask_token_id,
                    n_vocab=args.n_vocab,
                    mlm_probability=args.mlm_probability,
                )
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                outputs = tc.model(inputs, masked_lm_labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            else:
                raise ValueError(f"unsupported value: {args.model_type}")

            eval_loss += loss.mean().item()
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    if args.model_type == PZERO:
        f1_score = compute_pzero_f_score(pzero_preds, pzero_golds)
        result = {
            "loss": eval_loss,
            "perplexity": perplexity,
            "f1_score": f1_score,
        }
    else:
        result = {
            "perplexity": perplexity,
            "loss": eval_loss,
        }

    output_eval_file = os.path.join(eval_output_dir, EVAL_RESULT_BASENAME)
    with open(output_eval_file, "w") as fo:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info(f"  {key} = {str(result[key])}")
            fo.write(f"{key} = {str(result[key])}\n")

    return result


def finetuning(tc: TrainingComponents) -> None:
    args: FinetuningArgs = tc.args
    total_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps * args.n_gpu
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Max epoch: {args.num_train_epochs}")

    for epoch in range(tc.training_scheduler.epoch, tc.training_scheduler.max_epoch):
        epoch += 1
        logger.info(f"Epoch {epoch}")

        # Training
        logger.info("Training")
        tc.model.train()
        logger.debug("calculating loss on train set")
        train_loss = compute_loss_finetuning(tc)
        logger.debug("evaluating on train set")
        train_eval_score = evaluate_finetuning(tc)
        logger.info("Train loss: {}, Train eval score: {}".format(train_loss, train_eval_score))

        # Evaluation
        logger.info('Evaluating')
        tc.model.eval()
        logger.debug("calculating loss on dev set")
        with torch.no_grad():
            dev_loss = compute_loss_finetuning(tc)
        logger.debug("evaluating on dev set")
        dev_eval_score = evaluate_finetuning(tc)
        logger.info("Dev loss: {}, Dev eval score: {}".format(dev_loss, dev_eval_score))

        # log
        lr_groups = {"group{}".format(idx): group["lr"] for idx, group in enumerate(tc.optimizer.param_groups)}
        tc.writer.add_scalars('lr', lr_groups, epoch)
        logger.info("\n{}".format(torch.cuda.memory_summary()))

        # learning scheduler
        is_best, is_over_thres, finish_training = tc.training_scheduler(dev_eval_score)

        # Save best model
        if is_best:
            torch.save(tc.model.state_dict(), path.join(tc.args.output_dir, PAS_MODEL_BASENAME))
            logger.info("Save best model")

        # over threshold of early stopping
        if is_over_thres:
            if tc.warmup_scheduler is None:
                optim_state = tc.lr_scheduler.get_state()
                if optim_state is None:
                    finish_training = True
                else:
                    # best model is loaded
                    logger.info("Load best model:")
                    logger.info(
                        f"best_epoch={tc.training_scheduler.best_epoch}, "
                        f"best_eval_score={tc.training_scheduler.best_performance}"
                    )
                    tc.model.load_state_dict(torch.load(path.join(tc.args.output_dir, PAS_MODEL_BASENAME)))

                    # the optimizer is reset when If learning scheduler type is 'half_reduce'
                    logger.info("Set new optimizer: lr={}".format(optim_state["lr"]))
                    tc.optimizer = Adam(filter(lambda p: p.requires_grad, tc.model.parameters()), **optim_state)
                    if tc.args.fp16:
                        tc.set_fp16()
            else:
                finish_training = True

        logger.info("Current early stopping count: {}".format(tc.training_scheduler.early_stopping_count))
        logger.info(
            f"Best performance on dev: {tc.training_scheduler.best_performance} ,"
            f"best_epoch={tc.training_scheduler.best_epoch}"
        )

        if not args.skip_save_checkpoint:
            tc.save_finetuning_components(args.output_dir)

        if finish_training:
            break


def compute_loss_finetuning(tc: TrainingComponents) -> float:
    args: FinetuningArgs = tc.args
    data_loader = tc.train_data_loader if tc.model.training else tc.dev_data_loader
    data_loader.create_batches()
    tc.set_zero_grad()

    total_loss = 0
    n_iter = None

    for n_iter, batch in enumerate(tqdm(data_loader), 1):
        loss = tc.model(batch)
        if tc.model.training:
            loss = tc.backward_loss(loss)
            if n_iter % args.gradient_accumulation_steps == 0 or n_iter == len(data_loader):
                tc.update_optimizer(use_clip_gradient=False)
        else:
            loss = float(loss.cpu())
        total_loss += float(loss)

    assert n_iter == len(data_loader)
    total_loss /= n_iter
    dataset_name = "train" if tc.model.training else "dev"
    epoch = tc.training_scheduler.epoch + 1
    tc.writer.add_scalars('loss', {dataset_name: total_loss}, epoch)

    return total_loss


def evaluate_finetuning(tc: TrainingComponents) -> float:
    dataset_name = "train" if tc.model.training else "dev"
    data_loader = tc.train_data_loader if tc.model.training else tc.dev_data_loader
    epoch = tc.training_scheduler.epoch + 1

    eval_instances = generate_pas_evaluation_instance(tc.model, data_loader)
    f_score, exo_f_score = compute_pas_f_score(eval_instances)
    tc.writer.add_scalars('f_score', {dataset_name: f_score}, epoch)
    tc.writer.add_scalars('exo_f_score', {dataset_name: exo_f_score}, epoch)

    return f_score


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--yaml_file', type=path.abspath, required=True, help='Path to yaml file')

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    params_dict = yaml.safe_load(open(args.yaml_file))
    assert "model_type" in params_dict, "error: 'model_type' doesn't exist."

    if params_dict["model_type"] in [CLOZE, PZERO]:
        params = PretrainingArgs(**params_dict)
        training_mode = "pretraining"
    elif params_dict["model_type"] in [AS, AS_PZERO]:
        params = FinetuningArgs(**params_dict)
        training_mode = "finetuning"
    else:
        raise ValueError(f"unsupported value: {params_dict['model_type']}")

    training_components = TrainingComponents()

    logger.info(f"training mode: {training_mode}")
    if training_mode == "pretraining":
        training_components.set_pretraining_components(params)
        pretraining(training_components)

    elif training_mode == "finetuning":
        training_components.set_finetuning_components(params)
        finetuning(training_components)

    logger.info("done")


if __name__ == '__main__':
    main()
