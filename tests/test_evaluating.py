# coding=utf-8

import unittest
from collections import defaultdict

from logzero import logger

from src.evaluating import compute_pas_f_score, compute_pzero_f_score, _compute_f_score_from_factor, _compute_f_factor
from src.instances import PasEvaluationInstance


class TestComputePasFScore(unittest.TestCase):
    eval_instance = PasEvaluationInstance(
        predicts=[1, 34, 0, 2, 18, 4, 17, 8, 23, 0, 0],
        exo_predicts=[2, 0, 3, 3, 0, 0, 2, 1, 1, 0, 0],
        golds=[[1, 13], [0], [5, 12], [2], [18], [4], [19, 20], [8, 23], [0], [12], [0]],
        exo_golds=[-100, 0, -100, -100, -100, -100, -100, -100, 1, -100, 3],
        case_names=["ga", "ga", "ga", "ga", "ga", "ga", "ga", "ni", "ga", "ga", "ga"],
        eval_infos=[],
    )

    def test_compute_pas_f_score(self):
        actual_f, actual_exo_f = compute_pas_f_score([self.eval_instance])
        expected_f = 5 / 8
        expected_exo_f = 2 / 3

        assert actual_f == expected_f
        assert actual_exo_f == expected_exo_f

    def test_compute_f_score(self):
        results = defaultdict(int)
        for predict, gold in zip(self.eval_instance["predicts"], self.eval_instance["golds"]):
            _compute_f_factor(predict, gold, results)

        assert results["pp"] == 5
        assert results["np"] == 3
        assert results["pn"] == 3
        assert results["nn"] == 1

    def test_compute_f_score_from_factor(self):
        results = {"pp": 5, "np": 4, "pn": 3, "nn": 1}
        actual = _compute_f_score_from_factor(results)

        prec = 5 / 9
        rec = 5 / 8
        expected = 2 * prec * rec / (prec + rec)

        assert actual == expected


class TestComputePzeroFScore(unittest.TestCase):
    def test_compute_pzero_f_score(self):
        logger.info("---test_selection_evaluation---")
        predicts = [1, 0, 18, 4, 23, 0]
        golds = [[1], [0], [2, 18], [0], [12, 16], [12]]

        actual_f = compute_pzero_f_score(predicts, golds)
        expected_f = 0.5

        assert actual_f == expected_f
