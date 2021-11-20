# coding=utf-8

import unittest

import yaml
from src.training import PretrainingArgs, FinetuningArgs, TrainingComponents


class TestPretrainingArgs(unittest.TestCase):
    def test_set_additional_parameters_cloze(self):
        file = "tests/samples/cloze.params.yml"
        params_dict = yaml.safe_load(open(file))
        args = PretrainingArgs(**params_dict)

        args.set_additional_parameters()

    def test_set_additional_parameters_pzero(self):
        file = "tests/samples/pzero.params.yml"
        params_dict = yaml.safe_load(open(file))
        args = PretrainingArgs(**params_dict)

        args.set_additional_parameters()


class TestFineTuningArgs(unittest.TestCase):
    def test_set_additional_parameters_as(self):
        file = "tests/samples/as.params.yml"
        params_dict = yaml.safe_load(open(file))
        args = FinetuningArgs(**params_dict)

        args.set_additional_parameters()

    def test_set_additional_parameters_as_pzero(self):
        file = "tests/samples/as-pzero.params.yml"
        params_dict = yaml.safe_load(open(file))
        args = FinetuningArgs(**params_dict)

        args.set_additional_parameters()


class TestTrainingComponents(unittest.TestCase):
    tc = TrainingComponents()

    file = "tests/samples/cloze.params.yml"
    params_dict = yaml.safe_load(open(file))
    cloze_args = PretrainingArgs(**params_dict)

    file = "tests/samples/pzero.params.yml"
    params_dict = yaml.safe_load(open(file))
    pzero_args = PretrainingArgs(**params_dict)

    file = "tests/samples/as.params.yml"
    params_dict = yaml.safe_load(open(file))
    as_args = FinetuningArgs(**params_dict)

    file = "tests/samples/as-pzero.params.yml"
    params_dict = yaml.safe_load(open(file))
    as_pzero_args = FinetuningArgs(**params_dict)

    def test_set_pretraining_components_cloze(self):
        self.tc.set_pretraining_components(self.cloze_args)

    def test_set_pretraining_components_pzero(self):
        self.tc.set_pretraining_components(self.pzero_args)

    def test_set_finetuning_components_as(self):
        self.tc.set_finetuning_components(self.as_args)

    def test_set_finetuning_components_as_pzero(self):
        self.tc.set_finetuning_components(self.as_pzero_args)
