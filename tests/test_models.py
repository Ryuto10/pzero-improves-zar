# coding=utf-8

import unittest

import torch
from logzero import logger

from src.instances import (
    LossFunction,
)
from src.models import (
    BertEmbeddingsForPAS,
    BertForPAS,
    BertConfig,
    CL_TOHOKU_BERT,
    AttentionWeight,
    BertClsClassificationLayer,
    PzeroModelForPreTraining,
    AsModel,
    AsPzeroModel,
    create_padding_mask,
)
from tests.mock_instances import mock_pzero_batch_instance, mock_as_batch_instance, mock_as_pzero_batch_instance


class DummyLossFunction(LossFunction):
    def __init__(self):
        super().__init__()

    def compute_loss(self, device, batch, predicts) -> torch.Tensor:
        return torch.rand(1)[0]


class TestBertEmbeddingsForPAS(unittest.TestCase):
    config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
    bert_embeddings = BertEmbeddingsForPAS(config)

    # init for test
    with torch.no_grad():
        bert_embeddings.token_type_embeddings.weight[0] = 0
        bert_embeddings.token_type_embeddings.weight[1] = 1
        for i in range(len(bert_embeddings.position_embeddings.weight)):
            bert_embeddings.position_embeddings.weight[i] = i

    # inputs
    input_shape = torch.Size([2, 6])
    input_ids = torch.LongTensor([[2, 171, 9, 4609, 2992, 3], [2, 214, 11, 2762, 34, 3]])
    token_type_ids = torch.zeros(input_shape, dtype=torch.long)
    position_ids = torch.arange(input_shape[1], dtype=torch.long).unsqueeze(0).expand(input_shape)
    predicate_position_ids = [[4], [3, 4]]
    xs_len = torch.LongTensor([6, 6])

    def test_on_cpu(self):
        device = torch.device("cpu")
        bert_embeddings = self.bert_embeddings.to(device)

        # normal
        output = bert_embeddings(
            input_ids=self.input_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            position_ids=self.position_ids.to(device),
            model_type="normal",
        )
        assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])

        # AS model
        output = bert_embeddings(
            input_ids=self.input_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            position_ids=self.position_ids.to(device),
            predicate_position_ids=self.predicate_position_ids,
            model_type="as",
        )
        assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])

        # AS-Pzero model
        output = bert_embeddings(
            input_ids=self.input_ids.to(device),
            token_type_ids=self.token_type_ids.to(device),
            position_ids=self.position_ids.to(device),
            predicate_position_ids=self.predicate_position_ids,
            xs_len=self.xs_len,
            model_type="as-pzero",
        )
        assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            bert_embeddings = self.bert_embeddings.to(device)

            # normal
            output = bert_embeddings(
                input_ids=self.input_ids.to(device),
                token_type_ids=self.token_type_ids.to(device),
                position_ids=self.position_ids.to(device),
                model_type="normal",
            )
            assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])

            # AS model
            output = bert_embeddings(
                input_ids=self.input_ids.to(device),
                token_type_ids=self.token_type_ids.to(device),
                position_ids=self.position_ids.to(device),
                predicate_position_ids=self.predicate_position_ids,
                model_type="as",
            )
            assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])

            # AS-Pzero model
            output = bert_embeddings(
                input_ids=self.input_ids.to(device),
                token_type_ids=self.token_type_ids.to(device),
                position_ids=self.position_ids.to(device),
                predicate_position_ids=self.predicate_position_ids,
                xs_len=self.xs_len,
                model_type="as-pzero",
            )
            assert output.shape == torch.Size([*self.input_shape, self.config.hidden_size])
        else:
            logger.warning('Not tested on GPU')


class TestBertEmbeddingsForPASComponents(unittest.TestCase):
    config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
    bert_embeddings = BertEmbeddingsForPAS(config)

    # init for test
    with torch.no_grad():
        bert_embeddings.token_type_embeddings.weight[0] = 0
        bert_embeddings.token_type_embeddings.weight[1] = 1
        for i in range(len(bert_embeddings.position_embeddings.weight)):
            bert_embeddings.position_embeddings.weight[i] = i

    def test_create_predicate_embeddings_cpu(self):
        device = torch.device("cpu")
        bert_embeddings = self.bert_embeddings.to(device=device)
        input_shape = torch.Size([2, 7])

        # inputs
        predicate_position_ids = [[4, 5], [6]]
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # output
        output = bert_embeddings.create_predicate_embeddings(
            predicate_position_ids=predicate_position_ids,
            token_type_ids=token_type_ids,
        )

        expected = torch.zeros(
            *input_shape, self.config.hidden_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        expected[0][4] = 1
        expected[0][5] = 1
        expected[1][6] = 1

        torch.testing.assert_allclose(output, expected)

    def test_create_predicate_embeddings_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            bert_embeddings = self.bert_embeddings.to(device=device)
            input_shape = torch.Size([2, 7])

            # input
            predicate_position_ids = [[4, 5], [6]]
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            # output
            output = bert_embeddings.create_predicate_embeddings(
                predicate_position_ids=predicate_position_ids,
                token_type_ids=token_type_ids,
            )

            expected = torch.zeros(
                *input_shape, self.config.hidden_size,
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )
            expected[0][4] = 1
            expected[0][5] = 1
            expected[1][6] = 1

            torch.testing.assert_allclose(output, expected)
        else:
            logger.warning('Not tested on GPU')

    def test_create_additional_position_embeddings_cpu(self):
        """
        Here assume the following 'input_ids'

        input_ids = [
            [w, w, w, w, p, p, s, m, a, p, p],
            [w, w, w, w, w, w, p, w, w, s, m, a, p]
        ]

        where
            w -> word
            p -> target predicate
            s -> sep token
            m -> mask token
            a -> argument label token (e.g. 'ga', 'wo', 'ni')
        """

        device = torch.device("cpu")
        bert_embeddings = self.bert_embeddings.to(device=device)

        # inputs
        input_shape = torch.Size([2, 13])
        predicate_position_ids = [[4, 5], [6]]
        xs_len = torch.LongTensor([11, 13])
        position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # output
        output = bert_embeddings.create_additional_position_embeddings(
            predicate_position_ids=predicate_position_ids,
            xs_len=xs_len,
            position_ids=position_ids,
        )

        expected = torch.zeros(
            *input_shape, self.config.hidden_size,
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        with torch.no_grad():
            expected[0][9] += 4
            expected[0][10] += 5
            expected[1][12] += 6

        torch.testing.assert_allclose(output, expected)

    def test_create_additional_position_embeddings_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            bert_embeddings = self.bert_embeddings.to(device=device)

            # inputs
            input_shape = torch.Size([2, 13])
            predicate_position_ids = [[4, 5], [6]]
            xs_len = torch.LongTensor([11, 13])
            position_ids = torch.arange(input_shape[1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

            # output
            output = bert_embeddings.create_additional_position_embeddings(
                predicate_position_ids=predicate_position_ids,
                xs_len=xs_len,
                position_ids=position_ids,
            )

            expected = torch.zeros(
                *input_shape, self.config.hidden_size,
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )
            with torch.no_grad():
                expected[0][9] += 4
                expected[0][10] += 5
                expected[1][12] += 6

            torch.testing.assert_allclose(output, expected)
        else:
            logger.warning('Not tested on GPU')


class TestAttentionWeight(unittest.TestCase):
    batch_size = 2
    query_seq_len = 1
    key_seq_len = 3
    hidden_size = 9

    model = AttentionWeight(embed_dim=hidden_size)

    # init for test
    with torch.no_grad():
        model.query_weights.weight[:] = torch.eye(hidden_size)
        model.query_weights.bias[:] = 0
        model.key_weights.weight[:] = torch.eye(hidden_size)
        model.key_weights.bias[:] = 0

    # inputs
    query = torch.ones(batch_size, query_seq_len, hidden_size)
    key = torch.Tensor([[0, 1, 2]]).unsqueeze(-1).expand(batch_size, key_seq_len, hidden_size)
    key_padding_mask = torch.ByteTensor([0, 1, 0]).unsqueeze(0).expand(batch_size, key_seq_len)

    # expected
    expected = torch.Tensor([[[0, 3, 6]]]).expand(batch_size, query_seq_len, key_seq_len)

    def test_without_key_padding_mask_cpu(self):
        device = torch.device('cpu')
        model = self.model.to(device)

        output = model(
            query=self.query.to(device),
            key=self.key.to(device),
        )

        torch.testing.assert_allclose(output, self.expected.to(device))

    def test_without_key_padding_mask_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            model = self.model.to(device)

            output = model(
                query=self.query.to(device),
                key=self.key.to(device),
            )

            torch.testing.assert_allclose(output, self.expected.to(device))
        else:
            logger.warning('Not tested on GPU')

    def test_with_key_padding_mask_cpu(self):
        device = torch.device('cpu')
        model = self.model.to(device)

        output = model(
            query=self.query.to(device),
            key=self.key.to(device),
            key_padding_mask=self.key_padding_mask.to(device),
        )

        expected = self.expected.clone()
        expected[:, :, 1] = -1e04

        torch.testing.assert_allclose(output, expected.to(device))

    def test_with_key_padding_mask_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            model = self.model.to(device)

            output = model(
                query=self.query.to(device),
                key=self.key.to(device),
                key_padding_mask=self.key_padding_mask.to(device),
            )

            expected = self.expected.clone()
            expected[:, :, 1] = -1e04

            torch.testing.assert_allclose(output, expected.to(device))
        else:
            logger.warning('Not tested on GPU')


class TestBertClsClassificationLayer(unittest.TestCase):
    config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
    out_size = 12
    model = BertClsClassificationLayer(config=config, out_size=out_size)

    # shape
    batch_size = 2

    # inputs
    hxs = torch.rand(batch_size, 1, config.hidden_size)

    def test_on_cpu(self):
        device = torch.device('cpu')
        model = self.model.to(device)
        output = model(self.hxs.to(device))
        assert output.shape == (self.batch_size, 1, self.out_size)

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            model = self.model.to(device)
            output = model(self.hxs.to(device))
            assert output.shape == (self.batch_size, 1, self.out_size)
        else:
            logger.warning('Not tested on GPU')


class TestBertForPAS(unittest.TestCase):
    config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
    model = BertForPAS(config)

    # shape
    batch_size = 2
    max_seq_length = 6

    # inputs
    input_ids = torch.LongTensor([[2, 171, 9, 4609, 2992, 3], [2, 214, 11, 2762, 34, 3]])
    predicate_position_ids = [[4], [3, 4]]
    xs_len = torch.LongTensor([6, 6])

    def test_on_cpu(self):
        device = torch.device('cpu')
        model = self.model.to(device)
        output, _ = model(
            input_ids=self.input_ids,
            predicate_position_ids=self.predicate_position_ids,
            xs_len=self.xs_len,
        )
        assert output.shape == torch.Size([self.batch_size, self.max_seq_length, self.config.hidden_size])

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            model = self.model.to(device)
            output, _ = model(
                input_ids=self.input_ids.to(device),
                predicate_position_ids=self.predicate_position_ids,
                xs_len=self.xs_len,
            )
            assert output.shape == torch.Size([self.batch_size, self.max_seq_length, self.config.hidden_size])
        else:
            logger.warning('Not tested on GPU')


class TestPzeroModelForPreTraining(unittest.TestCase):
    dummy_loss_function = DummyLossFunction()
    model = PzeroModelForPreTraining(dummy_loss_function)

    # shape
    batch_size = 2
    max_seq_length = 15
    batch_instance = mock_pzero_batch_instance

    # def test_on_cpu(self):
    #     device = torch.device('cpu')
    #     model = self.model.to(device)
    #
    #     # forward
    #     loss = model(self.batch_instance)
    #     assert loss.shape == torch.Size([])
    #
    #     # prediction
    #     output = model.prediction(self.batch_instance)
    #     assert output.shape == torch.Size([self.batch_size, self.max_seq_length])
    #
    #     # inference
    #     output = model.inference(self.batch_instance)
    #     output = output["selection_positions"]
    #     assert output.shape == torch.Size([self.batch_size])

    def test_on_gpu(self):
        if torch.cuda.is_available():
            device = torch.device('cuda', index=0)
            model = self.model.to(device)

            # forward
            loss = model(self.batch_instance)
            assert loss.shape == torch.Size([])

            # prediction
            output = model.prediction(self.batch_instance)
            assert output.shape == torch.Size([self.batch_size, self.max_seq_length])

            # inference
            output = model.inference(self.batch_instance)
            output = output["selection_positions"]
            assert output.shape == torch.Size([self.batch_size])
        else:
            logger.warning('Not tested on GPU')


# class TestAsModel(unittest.TestCase):
#     dummy_loss_function = DummyLossFunction()
#     model = AsModel(dummy_loss_function)
#
#     # shape
#     batch_size = 2
#     max_seq_length = 21
#     batch_instance = mock_as_batch_instance
#
#     def test_on_cpu(self):
#         device = torch.device('cpu')
#         model = self.model.to(device)
#
#         # forward
#         loss = model(self.batch_instance)
#         assert loss.shape == torch.Size([])
#
#         # prediction
#         label_scores, exo_scores = model.prediction(self.batch_instance)
#         assert label_scores.shape == torch.Size([self.batch_size, self.max_seq_length, 4])
#         assert exo_scores.shape == torch.Size([self.batch_size, 3 * 4])
#
#         # inference
#         output = model.inference(self.batch_instance)
#         predicts = output["predicts"]
#         exo_predicts = output["exo_predicts"]
#         assert torch.Tensor(predicts).shape == torch.Size([self.batch_size * 3])
#         assert torch.Tensor(exo_predicts).shape == torch.Size([self.batch_size * 3])
#
#     def test_on_gpu(self):
#         if torch.cuda.is_available():
#             device = torch.device('cuda', index=0)
#             model = self.model.to(device)
#
#             # forward
#             loss = model(self.batch_instance)
#             assert loss.shape == torch.Size([])
#
#             # prediction
#             label_scores, exo_scores = model.prediction(self.batch_instance)
#             assert label_scores.shape == torch.Size([self.batch_size, self.max_seq_length, 4])
#             assert exo_scores.shape == torch.Size([self.batch_size, 3 * 4])
#
#             # inference
#             output = model.inference(self.batch_instance)
#             predicts = output["predicts"]
#             exo_predicts = output["exo_predicts"]
#             assert torch.Tensor(predicts).shape == torch.Size([self.batch_size * 3])
#             assert torch.Tensor(exo_predicts).shape == torch.Size([self.batch_size * 3])
#
#         else:
#             logger.warning('Not tested on GPU')


# class TestAsPzeroModel(unittest.TestCase):
#     dummy_loss_function = DummyLossFunction()
#     model = AsPzeroModel(dummy_loss_function)
#
#     # shape
#     batch_size = 2
#     max_seq_length = 24
#     batch_instance = mock_as_pzero_batch_instance
#
#     def test_on_cpu(self):
#         device = torch.device('cpu')
#         model = self.model.to(device)
#
#         # forward
#         loss = model(self.batch_instance)
#         assert loss.shape == torch.Size([])
#
#         # prediction
#         selection_scores, exo_scores = model.prediction(self.batch_instance)
#         assert selection_scores.shape == torch.Size([self.batch_size, self.max_seq_length])
#         assert exo_scores.shape == torch.Size([self.batch_size, 4])
#
#         # inference
#         output = model.inference(self.batch_instance)
#         predicts = output["predicts"]
#         exo_predicts = output["exo_predicts"]
#         assert torch.Tensor(predicts).shape == torch.Size([self.batch_size])
#         assert torch.Tensor(exo_predicts).shape == torch.Size([self.batch_size])
#
#     def test_on_gpu(self):
#         if torch.cuda.is_available():
#             device = torch.device('cuda', index=0)
#             model = self.model.to(device)
#
#             # forward
#             loss = model(self.batch_instance)
#             assert loss.shape == torch.Size([])
#
#             # prediction
#             selection_scores, exo_scores = model.prediction(self.batch_instance)
#             assert selection_scores.shape == torch.Size([self.batch_size, self.max_seq_length])
#             assert exo_scores.shape == torch.Size([self.batch_size, 4])
#
#             # inference
#             output = model.inference(self.batch_instance)
#             predicts = output["predicts"]
#             exo_predicts = output["exo_predicts"]
#             assert torch.Tensor(predicts).shape == torch.Size([self.batch_size])
#             assert torch.Tensor(exo_predicts).shape == torch.Size([self.batch_size])
#
#         else:
#             logger.warning('Not tested on GPU')


class TestCreatePaddingMask(unittest.TestCase):
    def test_padding_exists(self):
        padding_mask = create_padding_mask(
            xs_len=torch.LongTensor([3, 5]),
            batch_size=2,
            seq_length=5,
        )
        expected = torch.BoolTensor([[False] * 3 + [True] * 2, [False] * 5])
        torch.testing.assert_allclose(padding_mask.float(), expected.float())

    def test_padding_not_exists(self):
        padding_mask = create_padding_mask(
            xs_len=torch.LongTensor([5, 5]),
            batch_size=2,
            seq_length=5,
        )
        expected = torch.BoolTensor([[False] * 5, [False] * 5])
        torch.testing.assert_allclose(padding_mask.float(), expected.float())
