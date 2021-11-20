# coding=utf-8

import argparse
import json
from collections import defaultdict
from os import path, makedirs
from typing import Dict, List, Union

import logzero
import torch
import yaml
from logzero import logger

from evaluating import generate_pas_evaluation_instance
from instances import PasDecodeInstance, PREDICATE, FILE
from iterators import PasBucketIterator
from models import AsModel, AsPzeroModel
from training import FinetuningArgs, PAS_MODEL_BASENAME, AS, AS_PZERO

EVAL_DIRNAME = "evaluation"
LOG_FILE_BASENAME = "logzero.logs.txt"
DECODE_FILE_PREFIX = "predict-"


def decode_for_pas(
        model: Union[AsModel, AsPzeroModel],
        data_loader: PasBucketIterator
) -> List[Dict[str, PasDecodeInstance]]:
    """Decoding for evaluation using a test set
    Args:
        model (Union[AsModel, AsPzeroModel]): The model for decoding
        data_loader (PasBucketIterator): data loader
    Returns:
        results (List[Dict[str, PasDecodeInstance]]):
            ```
            results = [
                {
                    "file": path to each file for evaluation (str),
                    "pred": PasDecodeInstance(sent=sent_index, id=word_index),
                    "ga": PasDecodeInstance(sent=sent_index, id=word_index),
                    ...
                },
                ...
            ]
            ```
            where "pred" indicates target predicate
    """

    # In the final result, values with the same key are combined into a single dictionary.
    # key: '{file_name}-{predicate_sentence_index}-{predicate_word_index}'
    # value: prediction for each label
    decodes = defaultdict(dict)

    for instance in generate_pas_evaluation_instance(model, data_loader):
        zip_iter = zip(instance["predicts"], instance["exo_predicts"], instance["case_names"], instance["eval_infos"])
        for p_idx, exo_p_idx, case_name, eval_info in zip_iter:
            file_path = eval_info["file_path"]
            predicate_sent_idx = eval_info["prd_sent_idx"]
            predicate_word_idx = eval_info["prd_word_ids"][-1]

            key = f"{file_path}-{predicate_sent_idx}-{predicate_word_idx}"

            # 'file'
            if FILE in decodes[key]:
                assert decodes[key][FILE] == file_path
            else:
                decodes[key][FILE] = file_path

            # 'target predicate'
            if PREDICATE in decodes[key]:
                assert decodes[key][PREDICATE] == PasDecodeInstance(sent=predicate_sent_idx, id=predicate_word_idx)
            else:
                decodes[key][PREDICATE] = PasDecodeInstance(sent=predicate_sent_idx, id=predicate_word_idx)

            assert case_name not in decodes[key]

            # 'intra', 'inter'
            if p_idx != 0 and p_idx in eval_info["sw2w_position"]:
                sent_idx, word_idx = eval_info["sw2w_position"][p_idx]  # convert subword idx to sentence/word idx
                decodes[key][case_name] = PasDecodeInstance(sent=sent_idx, id=word_idx)

            # 'exophoric'
            elif p_idx == 0 and exo_p_idx != 0:
                decodes[key][case_name] = PasDecodeInstance(sent=-exo_p_idx, id=-1)  # EXO1 = -1, EXO2 = -2, EXOG = -3

    # sort values by keys and remove keys
    decode_results = [result for _, result in sorted(decodes.items(), key=lambda x: x[0])]

    return decode_results


def create_arg_parser():
    parser = argparse.ArgumentParser(description='decoding')
    parser.add_argument('--data', type=path.abspath, required=True, help='Path to test data')
    parser.add_argument('--yaml_file', type=path.abspath, required=True, help='Path to yaml file')

    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # file check
    if not path.exists(args.data):
        raise FileNotFoundError("not found: {}".format(args.data))
    if not path.exists(args.yaml_file):
        raise FileNotFoundError("not found: {}".format(args.yaml_file))

    # load yaml file
    params_dict = yaml.safe_load(open(args.yaml_file))
    assert "model_type" in params_dict, "error: 'model_type' doesn't exist."
    params = FinetuningArgs(**params_dict)
    params.set_additional_parameters()

    # file check
    if not path.exists(params.output_dir):
        raise FileNotFoundError("not found: {}".format(params.output_dir))

    model_file = path.join(params.output_dir, PAS_MODEL_BASENAME)

    if not path.exists(model_file):
        raise FileNotFoundError("not found: {}".format(params.output_dir))

    # new files and directory
    eval_dir = path.join(params.output_dir, EVAL_DIRNAME)
    log_file = path.join(eval_dir, LOG_FILE_BASENAME)
    result_file = path.join(eval_dir, f"{DECODE_FILE_PREFIX}{path.basename(args.data)}")

    # create directory for evaluation and set logfile
    makedirs(eval_dir, exist_ok=True)
    logzero.logfile(log_file)

    # load dataset
    eval_data_loader = PasBucketIterator(
        file_path=args.data,
        batch_size=params.per_gpu_eval_batch_size,
        n_max_tokens=params.per_gpu_eval_max_tokens,
        padding_value=params.pad_token_id,
        model_type=params.model_type,
    )

    # load model
    if params.model_type == AS:
        model = AsModel()
    elif params.model_type == AS_PZERO:
        model = AsPzeroModel()
    else:
        raise ValueError(f"unsupported value: {params.model_type}")
    model.load_state_dict(torch.load(model_file))
    if torch.cuda.is_available():
        model = model.cuda()

    # decode
    logger.info("Eval file: {}".format(args.data))
    logger.debug("Start decoding")
    model.eval()
    results = decode_for_pas(model=model, data_loader=eval_data_loader)
    logger.info("save to: {}".format(result_file))
    with open(result_file, "w") as fo:
        for result in results:
            print(json.dumps(result), file=fo)
    logger.info("done")


if __name__ == '__main__':
    main()
