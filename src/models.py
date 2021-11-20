# coding: utf-8

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers.file_utils import add_start_docstrings_to_callable
from transformers.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
    BertPooler,
    BertPredictionHeadTransform,
    BERT_INPUTS_DOCSTRING,
    BertModel,
    BertConfig,
)

from instances import (
    CASE_INDEX,
    PzeroBatchInstance,
    PzeroOutputForLoss,
    PzeroOutputForInference,
    AsBatchInstance,
    AsOutputForLoss,
    AsPzeroBatchInstance,
    AsPzeroOutputForLoss,
    PasOutputForInference,
    LossFunction,
)

BertLayerNorm = torch.nn.LayerNorm
CL_TOHOKU_BERT = 'cl-tohoku/bert-base-japanese-whole-word-masking'


class BertEmbeddingsForPAS(nn.Module):
    """Embedding Layer customized to embed the positions of the target predicates.
    (1) Normal
        if 'predicate_position_ids' is None, the positions of the predicates are not embedded.

        embeddings = word_embeddins + position_embeddins

    (2) AS model
        The following arguments are required:
        - 'predicate_position_ids' is not None
        - 'xs_len' is None

        embeddings = word_embeddins + position_embeddins + predicate_embeddings (token_type_embeddings)

    (3) AS-Pzero model
        The following arguments are required:
        - 'predicate_position_ids' is not None
        - 'xs_len' is not None

        embeddings = word_embeddins + position_embeddins + additional_position_embeddings
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            predicate_position_ids=None,
            xs_len=None,
            model_type=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embed_dim = self.position_embeddings.embedding_dim

        # embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        predicate_embeddings = self.token_type_embeddings(token_type_ids)
        additional_position_embeddings = match_device(self, torch.zeros(*input_shape, embed_dim))

        # normal
        if predicate_position_ids is None:
            if model_type is not None:
                assert model_type == "normal"

        # AS model
        elif xs_len is None:
            predicate_embeddings = self.create_predicate_embeddings(
                predicate_position_ids=predicate_position_ids,
                token_type_ids=token_type_ids,
            )
            if model_type is not None:
                assert model_type == "as"

        # AS-Pzero model
        else:
            additional_position_embeddings = self.create_additional_position_embeddings(
                predicate_position_ids=predicate_position_ids,
                xs_len=xs_len,
                position_ids=position_ids,
            )
            if model_type is not None:
                assert model_type == "as-pzero"

        embeddings = inputs_embeds + position_embeddings + predicate_embeddings + additional_position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def create_predicate_embeddings(
            self,
            predicate_position_ids: List[List[int]],
            token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create predicate embeddings for AS model

        Args:
            predicate_position_ids (List[List[int]]) : Indices to point to where the predicate is located
            token_type_ids (torch.Tensor) : Indices to extract embeddings from token type embeddings

        Return:
            predicate_embeddings (torch.Tensor) : Tensor with shape (batch_size, seq_length, embed_dim)
        """
        input_shape = token_type_ids.size()
        assert len(predicate_position_ids) == input_shape[0]

        # Assign 1 to the predicate positions for each batch
        for idx, p_ids in enumerate(predicate_position_ids):
            token_type_ids[idx, p_ids] = 1

        predicate_embeddings = self.token_type_embeddings(token_type_ids)
        assert predicate_embeddings.size()[:-1] == input_shape
        assert predicate_embeddings.size()[-1] == self.token_type_embeddings.embedding_dim

        return predicate_embeddings

    def create_additional_position_embeddings(
            self,
            predicate_position_ids: List[List[int]],
            xs_len: torch.LongTensor,
            position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Create predicate embeddings for AS-Pzero model

        Args:
            predicate_position_ids (List[List[int]]) : Indices to point to where the predicate is located
            xs_len (torch.LongTensor) : LongTensor with batch size length.
                                        Each element indicates to the length of 'input_ids' before padding.
            position_ids (torch.Tensor) : Indices to extract embeddings from position embeddings

        Return:
            additional_position_embeddings (torch.Tensor) : Tensor with shape (batch_size, seq_length, embed_dim)
        """
        input_shape = position_ids.size()
        batch_size, seq_length = input_shape
        embed_dim = self.position_embeddings.embedding_dim

        assert len(predicate_position_ids) == batch_size
        assert xs_len.size() == input_shape[:-1]

        # Add the position embeddings of a target predicate to a predicate in query chunk
        additional_position_embeddings = torch.stack([
            torch.cat([
                match_device(self, torch.zeros(x_len - len(p_ids), embed_dim)),  # before a predicate in query chunk
                self.position_embeddings(position_ids[idx][p_ids]),  # a predicate in query chunk
                match_device(self, torch.zeros(seq_length - x_len, embed_dim))  # padding
            ])
            for idx, (p_ids, x_len) in enumerate(zip(predicate_position_ids, xs_len))
        ])

        assert additional_position_embeddings.size()[:-1] == input_shape
        assert additional_position_embeddings.size()[-1] == embed_dim

        return additional_position_embeddings


class BertForPAS(BertPreTrainedModel):
    """The embedding layer is modified for predicate argument structure analysis (PAS)"""

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddingsForPAS(config)  # Customized for PAS
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            predicate_position_ids=None,
            xs_len=None
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            predicate_position_ids=predicate_position_ids,
            xs_len=xs_len
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertClsClassificationLayer(nn.Module):
    def __init__(self, config, out_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, out_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


class AttentionWeight(nn.Module):
    """Compute Self-Attention Weights."""

    def __init__(self, embed_dim: int):
        super(AttentionWeight, self).__init__()
        self.embed_dim = embed_dim

        self.query_weights = nn.Linear(embed_dim, embed_dim)
        self.key_weights = nn.Linear(embed_dim, embed_dim)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            key_padding_mask: torch.ByteTensor = None
    ) -> torch.Tensor:
        """
        Args:
            - query: Tensor with shape ('batch_size', 'query_seq_length', 'embed_dim')
            - key: Tensor with shape ('batch_size', 'key_seq_length', 'embed_dim')
            - key_padding_mask: ByteTensor with shape ('batch_size', 'key_seq_length')
                if provided, specified padding elements in the key will be ignored by the attention.
                This is an binary mask. When the value is True, the corresponding value on the attention
                layer will be filled with -inf.
        Return:
            - attn_weights: Tensor with shape ('batch_size', 'query_seq_length', 'key_seq_length')
        """
        if len(query.shape) == 2:
            query = query.unsqueeze(1)
        batch_size, query_seq_length, embed_dim = query.shape
        key_seq_length = key.size(1)
        assert key.shape == (batch_size, key_seq_length, embed_dim)

        matmul_w = torch.matmul(self.query_weights(query), self.key_weights(key).transpose(1, 2)) / np.sqrt(embed_dim)
        assert matmul_w.shape == (batch_size, query_seq_length, key_seq_length)

        if key_padding_mask is not None:
            attn_weights = matmul_w.transpose(1, 2)
            assert attn_weights.shape == (batch_size, key_seq_length, query_seq_length)
            assert key_padding_mask.shape == (batch_size, key_seq_length)
            attn_weights[key_padding_mask] = -1e+04  # -np.inf
            attn_weights = attn_weights.transpose(1, 2)
        else:
            attn_weights = matmul_w

        assert attn_weights.shape == (batch_size, query_seq_length, key_seq_length)

        return attn_weights


class PzeroModelForPreTraining(nn.Module):
    """Pzero (Pseudo Zero Pronoun Resolution)

    Given sentences with a masked token, this model predicts a token with the same surface form as the masked token.
    The model is trained by pseudo labeled data created from raw corpus.
    The parameters to be trained are as follows:
        (1) model parameters of BERT
        (2) Selection Layer (two linear layers, W_1 and W_2)
    """

    def __init__(
            self,
            loss_function: Optional[LossFunction] = None,
            bert_model: Optional[BertModel] = None,
            device: Optional[torch.device] = None
    ):
        super(PzeroModelForPreTraining, self).__init__()
        self.loss_function = loss_function
        self.bert_model = bert_model
        self.device = device

        if self.bert_model is None:
            config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
            self.bert_model = BertForPAS(config)
        self.config = self.bert_model.config
        self.layer_norm = BertLayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.selection_self_attn = AttentionWeight(embed_dim=self.config.hidden_size)

    def forward(self, batch_instance: PzeroBatchInstance) -> torch.Tensor:
        """Calculate the loss"""
        selection_scores = self.prediction(batch_instance)
        output = PzeroOutputForLoss(selection_scores=selection_scores)
        loss = self.loss_function(batch_instance, output, self.device)

        return loss

    def prediction(self, batch_instance: PzeroBatchInstance) -> torch.Tensor:
        """
        Args:
            batch_instance: Please see 'instances.PzeroBatchInstance'

        Returns:
            selection_score: Tensor with shape ('batch_size', 'seq_length')
        """
        if self.device is None:
            try:
                self.device = get_model_device(self)
            except():
                "when using multi gpus, please set torch.device('cuda') to self.device"

        input_ids: torch.LongTensor = batch_instance["input_ids"].to(self.device)
        masked_ids: torch.LongTensor = batch_instance["masked_ids"]
        xs_len: torch.LongTensor = batch_instance["xs_len"]

        batch_size, seq_length = input_ids.shape
        assert batch_size == masked_ids.size(0) == xs_len.size(0)

        # embedding layer
        last_hidden_state, _ = self.bert_model(input_ids)

        last_hidden_state = self.layer_norm(last_hidden_state)
        masked_hidden_states = last_hidden_state[range(batch_size), masked_ids]
        assert masked_hidden_states.shape == torch.Size([batch_size, self.config.hidden_size])

        key_padding_mask = create_padding_mask(xs_len, batch_size, seq_length).to(self.device)
        assert key_padding_mask.shape == torch.Size([batch_size, seq_length])

        selection_scores = self.selection_self_attn(
            query=masked_hidden_states,
            key=last_hidden_state,
            key_padding_mask=key_padding_mask
        )
        selection_scores = selection_scores.squeeze(1)
        assert selection_scores.shape == torch.Size([batch_size, seq_length])

        return selection_scores

    def inference(self, batch_instance: PzeroBatchInstance) -> PzeroOutputForInference:
        """decoding for evaluation"""
        with torch.no_grad():
            selection_scores = self.prediction(batch_instance)
            selection_positions = torch.argmax(selection_scores, dim=1)
            output = PzeroOutputForInference(selection_positions=selection_positions)

        return output


class AsModel(nn.Module):
    """AS (Argument Selection with label probabilty)

    Given a target predicate and sentences, this model predicts its predicate argument structure (PAS) by AS method.
    This model requires model parameters of BERT. The model is fine-tuned by labeled data.
    The parameters to be trained are as follows:
        (1) model parameters of BERT
        (2) Softmax Layer (a linear layers, W_l)
        (3) Exophoric Layer (BertClsClassificationLayer)
    """

    def __init__(
            self,
            loss_function: Optional[LossFunction] = None,
            bert_model: Optional[BertForPAS] = None,
            embed_dropout: float = 0.0,
            layer_norm_eps: float = 1e-5,
    ):
        super(AsModel, self).__init__()
        self.loss_function = loss_function
        self.bert_model = bert_model
        self.embed_dropout = embed_dropout
        self.layer_norm_eps = layer_norm_eps

        if self.bert_model is None:
            config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
            self.bert_model = BertForPAS(config)
        self.embed_dim = self.bert_model.config.hidden_size
        self.max_seq_length = self.bert_model.config.max_position_embeddings
        self.vocab_size = self.bert_model.config.vocab_size
        self.out_dim = len(CASE_INDEX)

        self.embed_dropout_layer = nn.Dropout(self.embed_dropout)
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.linear = nn.Linear(self.embed_dim, self.out_dim)
        self.exo_linear_layer = BertClsClassificationLayer(self.bert_model.config, 12)  # 4 exos * 3 labels

    def forward(self, batch_instance: AsBatchInstance) -> torch.Tensor:
        """Calculate the loss"""
        device = get_model_device(self)
        label_scores, exo_scores = self.prediction(batch_instance)
        output = AsOutputForLoss(label_scores=label_scores, exo_scores=exo_scores)
        loss = self.loss_function(batch_instance, output, device)

        return loss

    def prediction(self, batch_instance: AsBatchInstance) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            * batch_instance: Please see 'instances.AsBatchInstance'

        Returns:
            * label_scores:
                Tensor with shape ('batch_size', 'seq_length', 4),
                which is scores of assigning the label over the input tokens.
                When calculating the loss with cross entropy, please calculate log_softmax.

            * exo_scores:
                Tensor with shape ('batch_size', 12) which is scores of exophoric.
                When the model selects the cls token as a predicate-argument
                (the model considers that the argument doesn't appear in the given sentences),
                the model classify the argument into the following four categories: (author, reader, general, none).
                Since there are three types of labels ('ga', 'wo', 'ni'),
                the model classifies arguments into 12 classes (4 * 3 = 12).
        """
        input_ids: torch.LongTensor = match_device(self, batch_instance["input_ids"])
        predicate_position_ids: List[List[int]] = batch_instance["predicate_position_ids"]
        xs_len: torch.LongTensor = batch_instance["xs_len"]

        batch_size, seq_length = input_ids.shape
        assert batch_size == len(predicate_position_ids) == xs_len.size(0)

        # embedding layer
        last_hidden_state, _ = self.bert_model(
            input_ids=input_ids,
            predicate_position_ids=predicate_position_ids
        )
        last_hidden_state = self.layer_norm(last_hidden_state)
        last_hidden_state = self.embed_dropout_layer(last_hidden_state)
        assert last_hidden_state.shape == torch.Size([batch_size, seq_length, self.embed_dim])

        # label scores
        label_scores = self.linear(last_hidden_state)
        # padding
        padding_mask = create_padding_mask(xs_len, batch_size, seq_length)
        label_scores[padding_mask] = -1e+04
        assert label_scores.shape == torch.Size([batch_size, seq_length, self.out_dim])

        # exo
        cls_hidden_states = last_hidden_state[:, 0, :]
        exo_scores = self.exo_linear_layer(cls_hidden_states)
        assert exo_scores.shape == torch.Size([batch_size, 12])

        return label_scores, exo_scores

    def inference(self, batch_instance: AsBatchInstance) -> PasOutputForInference:
        """For inference (decoding)

        Args:
            batch_instance (AsBatchInstance)

        Returns:
            output (PasInferenceOutput)
        """
        with torch.no_grad():
            label_scores, exo_scores = self.prediction(batch_instance)
            batch_size = len(label_scores)

            # predict: arguments in sentences
            predicts = torch.argmax(label_scores[:, :, :3], dim=1).view(-1)  # shape is (batch * 3, )
            assert predicts.shape == torch.Size([batch_size * 3])
            predicts = predicts.tolist()

            # predict: exophoric
            reshape_exo_scores = exo_scores.view(-1, 4)
            assert reshape_exo_scores.shape == torch.Size([batch_size * 3, 4])
            exo_predicts = torch.argmax(reshape_exo_scores, dim=1)
            assert exo_predicts.shape == torch.Size([batch_size * 3])
            exo_predicts = exo_predicts.tolist()

            output = PasOutputForInference(predicts=predicts, exo_predicts=exo_predicts)

        return output


class AsPzeroModel(nn.Module):
    """AS-Pzero (Argument Selection as Pseudo Zero Pronoun Resolution)

    Given a target predicate and sentences,
    this model predicts its predicate argument structure (PAS) by AS-Pzero method.
    This model requires model parameters of BERT. The model is fine-tuned by labeled data.
    The parameters to be trained are as follows:
        (1) model parameters of BERT
        (2) Selection Layer (two linear layers, W_1 and W_2)
        (3) Exophoric Layer (BertClsClassificationLayer)
    """

    def __init__(
            self,
            loss_function: Optional[LossFunction] = None,
            bert_model: Optional[BertForPAS] = None,
            embed_dropout: float = 0.0,
            layer_norm_eps: float = 1e-5,
    ):
        super(AsPzeroModel, self).__init__()
        self.loss_function = loss_function
        self.bert_model = bert_model
        self.embed_dropout = embed_dropout
        self.layer_norm_eps = layer_norm_eps

        if self.bert_model is None:
            config = BertConfig.from_pretrained(CL_TOHOKU_BERT)
            self.bert_model = BertForPAS(config)
        self.embed_dim = self.bert_model.config.hidden_size
        self.max_seq_length = self.bert_model.config.max_position_embeddings
        self.vocab_size = self.bert_model.config.vocab_size

        self.embed_dropout_layer = nn.Dropout(self.embed_dropout)
        self.layer_norm = BertLayerNorm(self.embed_dim, eps=self.layer_norm_eps)

        self.selection_self_attn = AttentionWeight(embed_dim=self.embed_dim)
        self.exo_linear_layer = BertClsClassificationLayer(self.bert_model.config, 4)  # (None, exo1, exo2, exog)

    def forward(self, batch_instance: AsPzeroBatchInstance) -> torch.Tensor:
        """Calculate the loss"""
        device = get_model_device(self)
        selection_scores, exo_scores = self.prediction(batch_instance)
        output = AsPzeroOutputForLoss(selection_scores=selection_scores, exo_scores=exo_scores)
        loss = self.loss_function(batch_instance, output, device)

        return loss

    def prediction(self, batch_instance: AsPzeroBatchInstance) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            * batch_instance: Please see 'instances.AsPzeroBatchInstance'

        Return:
            * selection_scores:
                Tensors with shape ('batch_size', 'seq_length'),
                which is score indicates whether a word is the argument of the given predicate or not.

            * exo_scores:
                Tensors with shape ('batch_size', 4) which is scores of exophoric.
                When the model selects the cls token as a predicate-argument
                (the model condiders that the argument doesn't appear in the given sentences),
                the model classify the argument into the following four categories: (author, reader, general, none).
        """
        input_ids: torch.LongTensor = match_device(self, batch_instance["input_ids"])
        predicate_position_ids: List[List[int]] = batch_instance["predicate_position_ids"]
        mask_position_ids: List[int] = batch_instance["mask_position_ids"]
        xs_len: torch.LongTensor = batch_instance["xs_len"]

        batch_size, seq_length = input_ids.shape
        assert batch_size == len(predicate_position_ids) == len(mask_position_ids) == xs_len.size(0)

        # embedding layer
        last_hidden_state, _ = self.bert_model(
            input_ids=input_ids,
            predicate_position_ids=predicate_position_ids,
            xs_len=xs_len,
        )
        last_hidden_state = self.layer_norm(last_hidden_state)
        last_hidden_state = self.embed_dropout_layer(last_hidden_state)
        assert last_hidden_state.shape == (batch_size, seq_length, self.embed_dim)
        masked_hidden_states = last_hidden_state[range(batch_size), mask_position_ids]
        assert masked_hidden_states.shape == (batch_size, self.embed_dim)

        # selection
        key_padding_mask = torch.arange(seq_length).expand(batch_size, seq_length) >= xs_len.unsqueeze(1)
        key_padding_mask = match_device(self, key_padding_mask)
        assert key_padding_mask.shape == (batch_size, seq_length)
        selection_scores = self.selection_self_attn(query=masked_hidden_states,
                                                    key=last_hidden_state,
                                                    key_padding_mask=key_padding_mask)
        selection_scores = selection_scores.squeeze(1)
        assert selection_scores.shape == (batch_size, seq_length)

        # exo
        cls_hidden_states = last_hidden_state[:, 0, :]
        exo_scores = self.exo_linear_layer(cls_hidden_states)
        assert exo_scores.shape == (batch_size, 4)

        return selection_scores, exo_scores

    def inference(self, batch_instance: AsPzeroBatchInstance) -> PasOutputForInference:
        """For inference (decoding)

        Args:
            batch_instance (AsPzeroBatchInstance)

        Returns:
            output (PasInferenceOutput)
        """
        with torch.no_grad():
            selection_scores, exo_scores = self.prediction(batch_instance)

            # predict: arguments in sentences
            predicts = torch.argmax(selection_scores, dim=1).tolist()

            # predict: exophoric
            exo_predicts = torch.argmax(exo_scores, dim=1).tolist()

            output = PasOutputForInference(predicts=predicts, exo_predicts=exo_predicts)

        return output


def get_model_device(model: nn.Module):
    return next(model.parameters()).device


def match_device(model: nn.Module, inputs):
    device = get_model_device(model)
    return inputs.to(device)


def create_padding_mask(xs_len: torch.LongTensor, batch_size: int, seq_length: int):
    """
    Args:
        xs_len: torch.LongTensor with shape (batch_size, )
        batch_size: int
        seq_length: int

    Returns:
        torch.Tensor
    """
    assert xs_len.shape == (batch_size,)
    return torch.arange(seq_length).expand(batch_size, seq_length) >= xs_len.cpu().unsqueeze(1)
