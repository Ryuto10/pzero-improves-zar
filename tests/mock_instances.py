# coding=utf-8

from src.instances import (
    PzeroMaskedInstance,
    PasEvalInfo,
    AsGoldPositions,
    AsGoldExo,
    AsTrainingInstance,
    AsPzeroTrainingInstance,
    create_pzero_batch_instance,
    create_as_batch_instance,
    create_as_pzero_batch_instance,
)

pzero_instances = [
    PzeroMaskedInstance(
        input_ids=[2, 1317, 14, 11152, 29, 4609, 34, 8, 4, 9, 11152, 5, 29, 8, 3],
        masked_idx=8,
        gold_ids=[1],
    ),
    PzeroMaskedInstance(
        input_ids=[2, 171, 9, 4609, 5, 82, 5, 4, 5, 214, 2992, 3],
        masked_idx=7,
        gold_ids=[3],
    ),
]

mock_pzero_batch_instance = create_pzero_batch_instance(
    pzero_masked_instances=pzero_instances,
    padding_value=0,
)

as_instances = [
    AsTrainingInstance(
        input_tokens=[
            "[CLS]", "デフレ", "色", "を", "深める", "日本", "経済", "は", "いよいよ", "容易", "なら", "ざる",
            "局面", "を", "迎え", "##た", "と", "思わ", "れる", "。", "[SEP]",
        ],
        input_ids=[
            2, 27831, 1232, 11, 22005, 91, 994, 9, 27014, 4446, 737, 6143,
            21470, 11, 2033, 28447, 13, 2649, 62, 8, 3,
        ],
        predicate_position_ids=[4],
        xs_len=21,
        gold_positions=AsGoldPositions(ga=[6], o=[2], ni=[0]),
        exo_idx=AsGoldExo(ga=-100, o=-100, ni=0),
        eval_info=PasEvalInfo(
            prd_word_ids=[3],
            prd_sent_idx=0,
            file_path="dummy",
            sw2w_position={
                "19": [0, 17],
                "18": [0, 16],
                "17": [0, 15],
                "16": [0, 14],
                "15": [0, 13],
                "14": [0, 13],
                "13": [0, 12],
                "12": [0, 11],
                "11": [0, 10],
                "10": [0, 9],
                "9": [0, 8],
                "8": [0, 7],
                "7": [0, 6],
                "6": [0, 5],
                "5": [0, 4],
                "4": [0, 3],
                "3": [0, 2],
                "2": [0, 1],
                "1": [0, 0]
            }
        )
    ),
    AsTrainingInstance(
        input_tokens=[
            "[CLS]", "デフレ", "色", "を", "深める", "日本", "経済", "は", "いよいよ", "容易", "なら", "ざる",
            "局面", "を", "迎え", "##た", "と", "思わ", "れる", "。", "[SEP]",
        ],
        input_ids=[
            2, 27831, 1232, 11, 22005, 91, 994, 9, 27014, 4446, 737, 6143,
            21470, 11, 2033, 28447, 13, 2649, 62, 8, 3,
        ],
        predicate_position_ids=[14, 15],
        xs_len=21,
        gold_positions=AsGoldPositions(ga=[6], o=[12], ni=[0]),
        exo_idx=AsGoldExo(ga=-100, o=-100, ni=0),
        eval_info=PasEvalInfo(
            prd_word_ids=[13],
            prd_sent_idx=0,
            file_path="dummy",
            sw2w_position={
                "19": [0, 17],
                "18": [0, 16],
                "17": [0, 15],
                "16": [0, 14],
                "15": [0, 13],
                "14": [0, 13],
                "13": [0, 12],
                "12": [0, 11],
                "11": [0, 10],
                "10": [0, 9],
                "9": [0, 8],
                "8": [0, 7],
                "7": [0, 6],
                "6": [0, 5],
                "5": [0, 4],
                "4": [0, 3],
                "3": [0, 2],
                "2": [0, 1],
                "1": [0, 0]
            }
        )
    ),
]

mock_as_batch_instance = create_as_batch_instance(
    training_instances=as_instances,
    padding_value=0,
)

# for creating inputs
as_pzero_instances = [
    AsPzeroTrainingInstance(
        input_tokens=[
            "[CLS]", "デフレ", "色", "を", "深める", "日本", "経済", "は", "いよいよ", "容易", "なら", "ざる",
            "局面", "を", "迎え", "##た", "と", "思わ", "れる", "。", "[SEP]", "[MASK]", "が", "深める"
        ],
        input_ids=[
            2, 27831, 1232, 11, 22005, 91, 994, 9, 27014, 4446, 737, 6143,
            21470, 11, 2033, 28447, 13, 2649, 62, 8, 3, 4, 14, 22005
        ],
        predicate_position_ids=[4],
        mask_position_id=21,
        xs_len=24,
        gold_positions=[6],
        exo_idx=-100,
        case_name="ga",
        eval_info=PasEvalInfo(
            prd_word_ids=[3],
            prd_sent_idx=0,
            file_path="dummy",
            sw2w_position={
                "19": [0, 17],
                "18": [0, 16],
                "17": [0, 15],
                "16": [0, 14],
                "15": [0, 13],
                "14": [0, 13],
                "13": [0, 12],
                "12": [0, 11],
                "11": [0, 10],
                "10": [0, 9],
                "9": [0, 8],
                "8": [0, 7],
                "7": [0, 6],
                "6": [0, 5],
                "5": [0, 4],
                "4": [0, 3],
                "3": [0, 2],
                "2": [0, 1],
                "1": [0, 0]
            }
        ),
    ),
    AsPzeroTrainingInstance(
        input_tokens=[
            "[CLS]", "デフレ", "色", "を", "深める", "日本", "経済", "は", "いよいよ", "容易", "なら", "ざる",
            "局面", "を", "迎え", "##た", "と", "思わ", "れる", "。", "[SEP]", "[MASK]", "を", "深める"
        ],
        input_ids=[
            2, 27831, 1232, 11, 22005, 91, 994, 9, 27014, 4446, 737, 6143,
            21470, 11, 2033, 28447, 13, 2649, 62, 8, 3, 4, 11, 22005
        ],
        predicate_position_ids=[4],
        mask_position_id=21,
        xs_len=24,
        gold_positions=[2],
        exo_idx=-100,
        case_name="o",
        eval_info=PasEvalInfo(
            prd_word_ids=[3],
            prd_sent_idx=0,
            file_path="dummy",
            sw2w_position={
                "19": [0, 17],
                "18": [0, 16],
                "17": [0, 15],
                "16": [0, 14],
                "15": [0, 13],
                "14": [0, 13],
                "13": [0, 12],
                "12": [0, 11],
                "11": [0, 10],
                "10": [0, 9],
                "9": [0, 8],
                "8": [0, 7],
                "7": [0, 6],
                "6": [0, 5],
                "5": [0, 4],
                "4": [0, 3],
                "3": [0, 2],
                "2": [0, 1],
                "1": [0, 0]
            }
        ),
    ),
]

mock_as_pzero_batch_instance = create_as_pzero_batch_instance(
    training_instances=as_pzero_instances,
    padding_value=0,
)
