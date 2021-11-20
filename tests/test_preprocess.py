# coding=utf-8

import json
import unicodedata
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from src.instances import (
    ChunkInstance,
    PasGoldLabel,
    PasGoldCase,
    DEP,
    INTER,
    EXOG,
    NULL,
    GA,
    WO,
    EXO_INDEX,
)
from src.preprocess import (
    extract_chunk_from_parsed_text,
    PreprocessForPretrainingCloze,
    PreprocessForPretrainingPzero,
    preprocess_ntc_dataset,
    PreprocessForFinetuningAS,
    PreprocessForFinetuningASPzero,
    reconfigure_sw2w_dict,
)

SAMPLE_DIR = Path(__file__).absolute().parent.joinpath("samples")


class TestPreprocessForPretrainingChunk(unittest.TestCase):
    def test_preprocess_from_parsed_to_chunk(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.txt")
        expected_file = SAMPLE_DIR.joinpath("raw.parsed.chunk.jsonl")
        preprocessor = extract_chunk_from_parsed_text

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            if not expected_line.rstrip():
                continue
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)

    def test_reverse_from_chunk_to_raw(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.txt")
        expected_file = SAMPLE_DIR.joinpath("raw.txt")
        expected = [sentence.rstrip("\n") for sentence in expected_file.open() if sentence.rstrip()]

        output = []
        for chunk_instances in extract_chunk_from_parsed_text(str(input_file)):
            if len(chunk_instances) == 0:
                continue
            else:
                sentence = "".join("".join(chunk["surfs"]) for chunk in chunk_instances)
                output.append(sentence)

        self.assertEqual(expected, output)


class TestPreprocessForPretrainingCloze(unittest.TestCase):
    def test_preprocess_from_parsed_text_to_cloze(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.txt")
        expected_file = SAMPLE_DIR.joinpath("cloze.instances.jsonl")
        preprocessor = PreprocessForPretrainingCloze()

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)

    def test_preprocess_from_chunk_to_cloze(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.chunk.jsonl")
        expected_file = SAMPLE_DIR.joinpath("cloze.instances.jsonl")
        preprocessor = PreprocessForPretrainingCloze(is_chunk=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)

    def test_reverse_from_cloze_to_raw(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.txt")
        expected = ""
        for line in input_file.open():
            if line.startswith("*") or line.startswith("EOS"):
                continue
            token = unicodedata.normalize("NFKC", line.split("\t")[0])
            expected += token

        preprocessor = PreprocessForPretrainingCloze()
        n_max = preprocessor.tokenizer.model_max_length
        cls_id = preprocessor.tokenizer.cls_token_id
        sep_id = preprocessor.tokenizer.sep_token_id

        output = ""
        for cloze_instance in preprocessor(str(input_file)):
            embed_ids = cloze_instance["input_ids"]
            assert len(embed_ids) <= n_max, f"Over model max length: {len(embed_ids)} > {n_max}"
            assert embed_ids[0] == cls_id, f"The first token is not cls: {embed_ids[0]}"
            assert embed_ids[-1] == sep_id, f"The last token is not sep: {embed_ids[-1]}"
            tokens = preprocessor.tokenizer.decode(embed_ids[1:-1]).replace(" ", "")
            output += tokens

        self.assertEqual(expected, output)


class TestPreprocessForPretrainingPzero(unittest.TestCase):
    def test_preprocess_from_parsed_text_to_pzero(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.txt")
        expected_file = SAMPLE_DIR.joinpath("pzero.instances.jsonl")
        preprocessor = PreprocessForPretrainingPzero()

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)

    def test_preprocess_from_chunk_to_pzero(self):
        input_file = SAMPLE_DIR.joinpath("raw.parsed.chunk.jsonl")
        expected_file = SAMPLE_DIR.joinpath("pzero.instances.jsonl")
        preprocessor = PreprocessForPretrainingPzero(is_chunk=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)


class TestPreprocessForPretrainingPzeroComponents(unittest.TestCase):
    preprocessor = PreprocessForPretrainingPzero()

    def test_reduce_previous_tokens(self):
        n_model_max = self.preprocessor.tokenizer.model_max_length
        n_sents = self.preprocessor.max_n_sentences

        cases = [
            {
                "name": "no change (single sentence)",
                "input": [list(range(n_model_max - 2))],
                "expected": [list(range(n_model_max - 2))],
            },
            {
                "name": "remove the first word of the first sentence (single sentence)",
                "input": [list(range(n_model_max - 1))],
                "expected": [list(range(1, n_model_max - 1))],
            },
            {
                "name": "no change (two sentences)",
                "input": [list(range(n_model_max // 2)), list(range(round(n_model_max / 2) - 3))],
                "expected": [list(range(n_model_max // 2)), list(range(round(n_model_max / 2) - 3))],
            },
            {
                "name": "remove the first word of the first sentence (two sentences)",
                "input": [list(range(n_model_max // 2)), list(range(round(n_model_max / 2) - 2))],
                "expected": [list(range(1, n_model_max // 2)), list(range(round(n_model_max / 2) - 2))],
            },
            {
                "name": "remove the first sentence & the head words of the second sentence",
                "input": [list(range(n_model_max // 2)), list(range(n_model_max))],
                "expected": [list(range(2, n_model_max))],
            },
        ]

        for case in cases:
            input_deque = deque(case["input"], maxlen=n_sents)
            expected = deque(case["expected"], maxlen=n_sents)
            output = self.preprocessor._reduce_previous_tokens(input_deque)
            self.assertEqual(expected, output, case["name"])

    def test_create_noun_positions(self):
        cases = [
            {
                "name": "included a noun phrase",
                "chunk": ChunkInstance(
                    idx="",
                    head="",
                    surfs=["生物", "多様", "性", "が"],
                    poss=["名詞", "形容詞", "接尾辞", "助詞"],
                    pos_details=["普通名詞", "*", "名詞性名詞接尾辞", "格助詞"]
                ),
                "assign_number": 23,
                "expected_np": "生物多様性",
                "expected_ids": [23, 23, 23, 0],
            },
            {
                "name": "not included a noun phrase",
                "chunk": ChunkInstance(
                    idx="",
                    head="",
                    surfs=["定義", "さ", "れる", "。"],
                    poss=["名詞", "動詞", "接尾辞", "特殊"],
                    pos_details=["サ変名詞", "*", "動詞性接尾辞", "句点"]
                ),
                "assign_number": 14,
                "expected_np": "",
                "expected_ids": [0, 0, 0, 0],
            },
            {
                "name": "included a noun phrase (in parentheses)",
                "chunk": ChunkInstance(
                    idx="",
                    head="",
                    surfs=["（", "コミュニケーション", "）", "、"],
                    poss=["特殊", "名詞", "特殊", "特殊"],
                    pos_details=["括弧始", "普通名詞", "括弧終", "読点"]
                ),
                "assign_number": 31,
                "expected_np": "コミュニケーション",
                "expected_ids": [0, 31, 0, 0],
            },
            {
                "name": "included a noun phrase (exclude parentheses)",
                "chunk": ChunkInstance(
                    idx="",
                    head="",
                    surfs=["言語", "（", "げんご", "）", "、"],
                    poss=["名詞", "特殊", "名詞", "特殊", "特殊"],
                    pos_details=["普通名詞", "括弧始", "普通名詞", "括弧終", "読点"]
                ),
                "assign_number": 1,
                "expected_np": "言語",
                "expected_ids": [1, 0, 0, 0, 0],
            },
        ]

        for case in cases:
            output_np, output_ids = self.preprocessor._create_noun_phrase_positions(
                chunk=case["chunk"],
                assign_number=case["assign_number"],
            )
            self.assertEqual(case["expected_np"], output_np, case["name"])
            self.assertEqual(case["expected_ids"], output_ids, case["name"])

    def test_subwords(self):
        cases = [
            {
                "name": "not divided",
                "word": "コミュニケーション",
                "expected": ["コミュニケーション"],
            },
            {
                "name": "divided",
                "word": "東京タワー",
                "expected": ["東京", "##タワー"],
            },
        ]

        for case in cases:
            subwords = self.preprocessor._subwords(case["word"])
            self.assertEqual(case["expected"], subwords, case["name"])

    def test_insert_sep(self):
        cases = [
            {
                "name": "insert a single sep",
                "input": deque([["これ", "は", "テスト", "です", "。"]]),
                "sep": "[SEP]",
                "expected": ["これ", "は", "テスト", "です", "。", "[SEP]"],
            },
            {
                "name": "insert two seps",
                "input": deque([["これ", "は"], ["テスト", "です", "。"]]),
                "sep": "[SEP]",
                "expected": ["これ", "は", "[SEP]", "テスト", "です", "。", "[SEP]"],
            },
        ]

        for case in cases:
            output = self.preprocessor._insert_sep(case["input"], case["sep"])
            self.assertEqual(case["expected"], output, case["name"])

    def test_replace_mask(self):
        cases = [
            {
                "name": "replace a single token",
                "input": np.array([5, 6, 7, 8, 9, 10]),
                "replace_ids": np.array([3]),
                "mask": 0,
                "expected": np.array([5, 6, 7, 0, 9, 10]),
            },
            {
                "name": "replace two tokens",
                "input": np.array([5, 6, 7, 8, 9, 10]),
                "replace_ids": np.array([3, 4]),
                "mask": 0,
                "expected": np.array([5, 6, 7, 0, 10]),
            },
        ]

        for case in cases:
            output = self.preprocessor._replace_mask(
                input_ids=case["input"],
                replace_ids=case["replace_ids"],
                mask_idx=case["mask"],
            )
            self.assertEqual(case["expected"].tolist(), output.tolist(), case["name"])


class TestPreprocessNTCCorpus(unittest.TestCase):
    def test_preprocess_ntc(self):
        input_file = SAMPLE_DIR.joinpath("dummy_ntc")
        expected_file = SAMPLE_DIR.joinpath("ntc.instances.jsonl")
        preprocessor = preprocess_ntc_dataset

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            self.assertEqual(expected, output)


class TestPreprocessForFinetuningAs(unittest.TestCase):
    def test_preprocess_from_ntc_to_as(self):
        input_dir = SAMPLE_DIR.joinpath("dummy_ntc")
        expected_file = SAMPLE_DIR.joinpath("as.instances.jsonl")
        preprocessor = PreprocessForFinetuningAS(is_processed_path=False)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_dir))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)

    def test_preprocess_from_doc_to_as(self):
        input_file = SAMPLE_DIR.joinpath("ntc.instances.jsonl")
        expected_file = SAMPLE_DIR.joinpath("as.instances.jsonl")
        preprocessor = PreprocessForFinetuningAS(is_processed_path=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)

    def test_preprocess_from_doc_to_as_intra(self):
        input_file = SAMPLE_DIR.joinpath("ntc.instances.jsonl")
        expected_file = SAMPLE_DIR.joinpath("as.intra.instances.jsonl")
        preprocessor = PreprocessForFinetuningAS(is_processed_path=True, is_intra=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)


class TestPreprocessForFinetuningAsComponents(unittest.TestCase):
    preprocessor = PreprocessForFinetuningAS()

    def test_convert_word_to_subword(self):
        cases = [
            {
                "name": "same to max length",
                "sents": [
                    ['これ', 'は', 'テスト', 'の', '文', 'です'],
                    ['一番', '後ろ', 'の', '文', 'に', '対象', 'と', 'なる', '述語', 'が', '存在', 'します'],
                ],
                "prd_sent_idx": 1,
                "max_length": 22,
                "expected_subwords": [
                    '[CLS]',
                    'これ', 'は', 'テスト', 'の', '文', 'です',
                    '[SEP]',
                    '一番', '後ろ', 'の', '文', 'に', '対象', 'と', 'なる', '述語', 'が', '存在', 'しま', '##す',
                    '[SEP]',
                ],
            },
            {
                "name": "over max length",
                "sents": [
                    ['これ', 'は', 'テスト', 'の', '文', 'です'],
                    ['一番', '後ろ', 'の', '文', 'に', '対象', 'と', 'なる', '述語', 'が', '存在', 'します'],
                ],
                "prd_sent_idx": 1,
                "max_length": 20,
                "expected_subwords": [
                    '[CLS]',
                    'テスト', 'の', '文', 'です',
                    '[SEP]',
                    '一番', '後ろ', 'の', '文', 'に', '対象', 'と', 'なる', '述語', 'が', '存在', 'しま', '##す',
                    '[SEP]',
                ],
            },
        ]

        for case in cases:
            subwords, sw2w_position, w2sw_position = self.preprocessor._convert_word_to_subword(
                case['sents'],
                case['prd_sent_idx'],
                case['max_length']
            )
            self.assertEqual(case["expected_subwords"], subwords, case["name"])

            for (sent_idx, word_idx), sw_ids in w2sw_position.items():
                for sw_idx in sw_ids:
                    assert sw2w_position[sw_idx] == (sent_idx, word_idx)

            for sw_idx, (sent_idx, word_idx) in sw2w_position.items():
                sw_ids = w2sw_position[(sent_idx, word_idx)]
                assert sw_idx in set(sw_ids)

    def test_create_gold_label(self):
        cases = [
            {
                "name": "with an answer and exists in 'input_ids'",
                "pas_gold_label": PasGoldLabel(
                    gold_cases=[PasGoldCase(sent_idx=0, word_idx=4, case_type=DEP)],
                    case_name=WO,
                    case_type=DEP,
                ),
                "w2sw_position": {(0, 4): [4, 5]},
                "expected_gold_positions": [5],
                "expected_exo_idx": -100,
            },
            {
                "name": "with answers and exists in 'input_ids'",
                "pas_gold_label": PasGoldLabel(
                    gold_cases=[
                        PasGoldCase(sent_idx=0, word_idx=4, case_type=INTER),
                        PasGoldCase(sent_idx=1, word_idx=12, case_type=DEP),
                    ],
                    case_name=GA,
                    case_type=DEP,
                ),
                "w2sw_position": {(0, 4): [4, 5], (1, 12): [14]},
                "expected_gold_positions": [5, 14],
                "expected_exo_idx": -100,
            },
            {
                "name": "with answers and not exists in 'input_ids'",
                "pas_gold_label": PasGoldLabel(
                    gold_cases=[
                        PasGoldCase(sent_idx=0, word_idx=4, case_type=INTER),
                        PasGoldCase(sent_idx=1, word_idx=12, case_type=DEP),
                    ],
                    case_name=GA,
                    case_type=DEP,
                ),
                "w2sw_position": {},
                "expected_gold_positions": [0],
                "expected_exo_idx": -100,
            },
            {
                "name": "without an answer (exophora)",
                "pas_gold_label": PasGoldLabel(
                    gold_cases=[],
                    case_name=WO,
                    case_type=EXOG,
                ),
                "w2sw_position": {},
                "expected_gold_positions": [0],
                "expected_exo_idx": EXO_INDEX[EXOG],
            },
            {
                "name": "without an answer (none)",
                "pas_gold_label": PasGoldLabel(
                    gold_cases=[],
                    case_name=WO,
                    case_type=NULL,
                ),
                "w2sw_position": {},
                "expected_gold_positions": [0],
                "expected_exo_idx": EXO_INDEX[NULL],
            },
        ]

        for case in cases:
            gold_positions, exo_idx = self.preprocessor._create_gold_label(
                pas_gold_label=case['pas_gold_label'],
                w2sw_position=case['w2sw_position'],
            )
            self.assertEqual(case["expected_gold_positions"], gold_positions, case["name"])
            self.assertEqual(case["expected_exo_idx"], exo_idx, case["name"])


class TestPreprocessForFinetuningAsPzero(unittest.TestCase):
    def test_preprocess_from_ntc_to_as_pzero(self):
        input_dir = SAMPLE_DIR.joinpath("dummy_ntc")
        expected_file = SAMPLE_DIR.joinpath("as-pzero.instances.jsonl")
        preprocessor = PreprocessForFinetuningASPzero(is_processed_path=False)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_dir))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)

    def test_preprocess_from_doc_to_as_pzero(self):
        input_file = SAMPLE_DIR.joinpath("ntc.instances.jsonl")
        expected_file = SAMPLE_DIR.joinpath("as-pzero.instances.jsonl")
        preprocessor = PreprocessForFinetuningASPzero(is_processed_path=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)

    def test_preprocess_from_doc_to_as_pzero_intra(self):
        input_file = SAMPLE_DIR.joinpath("ntc.instances.jsonl")
        expected_file = SAMPLE_DIR.joinpath("as-pzero.intra.instances.jsonl")
        preprocessor = PreprocessForFinetuningASPzero(is_processed_path=True, is_intra=True)

        for expected_line, output in zip(expected_file.open(), preprocessor(str(input_file))):
            expected = json.loads(expected_line)
            expected['eval_info']['sw2w_position'] = reconfigure_sw2w_dict(expected['eval_info']['sw2w_position'])
            self.assertEqual(expected, output)
