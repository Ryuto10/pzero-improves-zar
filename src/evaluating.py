# coding: utf-8

from collections import defaultdict
from itertools import chain
from typing import Dict, Generator, List, Union, Iterable

from logzero import logger
from tqdm import tqdm

from instances import TARGET_CASES, PasOutputForInference, PasEvaluationInstance
from iterators import PasBucketIterator
from models import AsModel, AsPzeroModel


def compute_pas_f_score(eval_instances: Iterable[PasEvaluationInstance]) -> (float, float):
    """Evaluation of the model prediction on the train/dev set
    Args:
        * eval_instances (Iterable[PasEvaluationInstance]): the instances to be used for evaluation
    Returns:
        * f (float): f1 score of dep, intra, inter
        * exo_f (float): f1 score of exophoric
    """
    results = {
        "argument_in_input": {case: defaultdict(int) for case in TARGET_CASES},
        "exophora": {case: defaultdict(int) for case in TARGET_CASES},
    }

    for eval_inst in eval_instances:
        for predict, gold, case_name in zip(eval_inst["predicts"], eval_inst["golds"], eval_inst["case_names"]):
            _compute_f_factor(predict, gold, results["argument_in_input"][case_name])
        for predict, gold, case_name in zip(eval_inst["exo_predicts"], eval_inst["exo_golds"], eval_inst["case_names"]):
            if gold == -100:
                continue
            _compute_f_factor(predict, [gold], results["exophora"][case_name])

    for arg_category in results.keys():
        logger.info(f"{arg_category} F1:")
        results[arg_category]["all"] = defaultdict(int)

        for case_name in TARGET_CASES:
            for factor in ["pp", "np", "pn", "nn"]:
                results[arg_category]["all"][factor] += results[arg_category][case_name][factor]
        for case_name in chain(TARGET_CASES, ["all"]):
            logger.info("{}:".format(case_name))
            _compute_f_score_from_factor(results[arg_category][case_name])

    f = results["argument_in_input"]["all"]["f1"]
    exo_f = results["exophora"]["all"]["f1"]
    logger.info("F1 score: {} (exo: {})".format(f, exo_f))

    return f, exo_f


def compute_pzero_f_score(predicts: List[int], golds: List[List[int]]) -> float:
    """Compute the f1 score"""
    results = defaultdict(int)
    for predict, gold in zip(predicts, golds):
        _compute_f_factor(predict, gold, results)
    f1 = _compute_f_score_from_factor(results)

    return f1


def _compute_f_factor(predict: int, gold: List[int], results: Dict[str, int]) -> None:
    """Compute the value for calculating the f1 score"""
    # if gold label is [0], it indicates that there is no correct answer in input sentences.
    if gold == [0]:
        if predict == 0:
            results["nn"] += 1
        else:
            results["np"] += 1
    elif predict != 0:
        if predict in gold:
            results["pp"] += 1
        else:
            results["np"] += 1
            results["pn"] += 1
    else:
        results["pn"] += 1


def _compute_f_score_from_factor(results: Dict[str, Union[int, float]]) -> float:
    """Compute the f1 score from values generated by 'calc_f_factor'"""
    p_p, n_p, p_n, n_n = results["pp"], results["np"], results["pn"], results["nn"]
    prec = p_p / (p_p + n_p) if p_p > 0 else 0
    rec = p_p / (p_p + p_n) if p_p > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0

    log_format = "prec: {:.4f}\trec: {:.4f}\tf1: {:.4f}\tpp {}\tnp {}\tpn {}\tnn {}"
    logger.info(log_format.format(prec, rec, f1, p_p, n_p, p_n, n_n))

    results["prec"], results["rec"], results["f1"] = prec, rec, f1

    return f1


def generate_pas_evaluation_instance(
        model: Union[AsModel, AsPzeroModel],
        data_loader: PasBucketIterator,
) -> Generator[PasEvaluationInstance, None, None]:
    """
    Args:
        * model (Union[ASModel, AsPzeroModel]): the model for inference
        * data_loader (PasBucketIterator): data loader
    Return:
        eval_instance (PasEvaluationInstance)
    """
    data_loader.create_batches()
    for batch in tqdm(data_loader):
        batch_size: int = len(batch["input_ids"])
        output: PasOutputForInference = model.inference(batch)

        predicts = output["predicts"]
        exo_predicts = output["exo_predicts"]

        if isinstance(model, AsPzeroModel):
            golds = batch["gold_positions"]
            exo_golds = batch["exo_ids"]
            case_names = batch["case_names"]
            eval_infos = batch["eval_info"]

        elif isinstance(model, AsModel):
            golds = [gold[case_name] for gold in batch["gold_positions"] for case_name in TARGET_CASES]
            exo_golds = [exo_gold[case_name] for exo_gold in batch["exo_ids"] for case_name in TARGET_CASES]
            case_names = [case_name for _ in range(batch_size) for case_name in TARGET_CASES]
            eval_infos = [info for info in batch["eval_info"] for _ in range(3)]

        else:
            raise ValueError(f"unsupported type of the model: {type(model)}")

        assert len(predicts) == len(exo_predicts) == len(golds) == len(exo_golds) == len(case_names) == len(eval_infos)

        eval_instance = PasEvaluationInstance(
            predicts=predicts,
            exo_predicts=exo_predicts,
            golds=golds,
            exo_golds=exo_golds,
            case_names=case_names,
            eval_infos=eval_infos,
        )

        yield eval_instance