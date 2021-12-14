# coding=utf-8

from abc import ABC
from typing import Dict, List, Tuple, Union, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence

FILE = "file"
PREDICATE = "pred"
GA = 'ga'
WO = 'o'
NI = 'ni'
TARGET_CASES = [GA, WO, NI]

DEP = 'dep'
INTRA = 'intra'
INTER = 'inter'
EXO1 = "exo1"
EXO2 = "exo2"
EXOG = "exog"
INB = 'bunsetsu'  # an argument and its predicate are contained in the same chunk
NULL = "null"

EXO = [EXO1, EXO2, EXOG]
NO_ANSWERS = [NULL, EXO1, EXO2, EXOG]
CASE_TYPES = [DEP, INTRA, INTER, EXO1, EXO2, EXOG, INB, NULL]

CASE_SURFACES = {GA: "が", WO: "を", NI: "に"}
CASE_INDEX = {GA: 0, WO: 1, NI: 2, NULL: 3}
EXO_INDEX = {NULL: 0, EXO1: 1, EXO2: 2, EXOG: 3}

CLOZE = "cloze"
PZERO = "pzero"
AS = "as"
AS_PZERO = "as-pzero"


# For pretraining
class ChunkInstance(TypedDict):
    """This instance is generated by 'preprocess.extract_chunk_from_parsed_text'

    * idx (str) : A number indicating the order of the chunks in a sentence
    * head (str) : A string indicating the head of the chunk
    * surfs (List[str]) : The list of surface forms in a chunk
    * poss (List[str]) : The list of PoS tags in a chunk
    * poss-details (List[str]) : The list of PoS tag details in a chunk
    """

    idx: str
    head: str
    surfs: List[str]
    poss: List[str]
    pos_details: List[str]


class ClozeNotMaskedInstance(TypedDict):
    """Instance for Cloze task

    * input_ids (Union[List[int], List[int]]) : Embedding indices as an input (not masked)
    """

    input_ids: Union[torch.LongTensor, List[int]]


class PzeroMaskedInstance(TypedDict):
    """Instance for Pzero task

    * input_ids (Union[List[int], List[int]]) : Embedding indices as an input that contains a masked token.
    * masked_idx (int) : The position of a masked token.
    * gold_ids: (List[int]) : The positions of pseudo antecedents that a masked token refers to.
    """

    input_ids: Union[torch.LongTensor, List[int]]
    masked_idx: int
    gold_ids: List[int]


class PzeroBatchInstance(TypedDict):
    """Batch instance for Pzero task

     * input_ids: Tensor with shape ('batch_size', 'seq_length')
     * masked_ids: Tensor with shape ('batch_size', )
     * xs_len: Tensor with shape ('batch_size', )
     * gold_ids: List of 'PzeroMaskedInstance.gold_ids', (Length of list = 'batch_size')
    """
    input_ids: torch.LongTensor
    masked_ids: torch.LongTensor
    xs_len: torch.LongTensor
    gold_ids: List[List[int]]


class PzeroOutputForLoss(TypedDict):
    """
    * selection_scores:
        Tensor with shape ('batch_size', 'seq_length') where the value of the prediction location is larger.
    """
    selection_scores: torch.Tensor


class PzeroOutputForInference(TypedDict):
    """
    * selection_positions: Tensor with shape ('batch_size', ) which indicates the position of the prediction.
    """
    selection_positions: torch.Tensor


# For finetuning
class PasGoldCase(TypedDict):
    """Gold case (argument)

    * sent_idx (int) : the index of sentences
    * word_idx (int) : the index of words
    * case_type (str) : one of the items in the list 'CASE_TYPES'
    """

    sent_idx: int
    word_idx: int
    case_type: str


class PasGoldLabel(TypedDict):
    """PAS gold label for each case

    * gold_cases List[PasGoldCase] : If there are multiple gold arguments due to co-reference relations,
                                     there are multiple gold cases in the list.
    * case_name (str) : The target case name (one of 'GA', 'WO', or 'NI')
    * case_type (str) : The type closest to the predicate in 'gold_cases' is applied, except for the case 'INB'
                        ('DEP' > 'INTRA' > 'INTER' > 'EXO > 'NULL')
    """

    gold_cases: List[PasGoldCase]
    case_name: str
    case_type: str


class PasGoldLabels(TypedDict):
    """Gold Labels of Predicate Argument Structure (PAS).

    * ga (PasGoldLabel) : gold labels of nominative
    *  o (PasGoldLabel) : gold labels of accusative
    * ni (PasGoldLabel) : gold labels of dative
    """

    ga: PasGoldLabel
    o: PasGoldLabel
    ni: PasGoldLabel


class Pas(TypedDict):
    """Predicate Argument Structure (PAS).

    * prd_sent_idx (int) : The index of the sentence containing a target predicate.
    * prd_word_ids (List[int]) : The indices of a target predicate.
                                 If the target predicate consists of a noun + "suru" ("サ変名詞 + する"),
                                 the length of the list is greater than 1. The numbers must be continuous.
    * gold_labels List[PasGoldLabel]) : The list of gold labels for a target predicate.
    * alt_type (str) : 'passive', 'active' or 'causative'.
    """

    prd_sent_idx: int
    prd_word_ids: List[int]
    gold_labels: PasGoldLabels
    alt_type: str


class NTCPasDocument(TypedDict):
    """A document in NAIST Text corpus (NTC) for predicate argument structure (PAS) analysis

    * sents (List[List[str]]) : The list of sentences, where a sentence is the list of the words.
    * pas_list (List[PAS]) : The list of PAS in a document.
    * file_path (str) : The path to a document file in NTC
    * sent_ids (List[int]) : The indices indicating the positions of the sentences in a document
    """

    file_path: str
    sents: List[List[str]]
    sent_ids: List[int]
    pas_list: List[Pas]


class PasEvalInfo(TypedDict):
    """Items for evaluation

    * prd_word_ids (List[int]) : The word's position of a target predicate
    * prd_sent_idx (List[int]) : The sentence's position of a target predicate
    * file_path (str) : The path to the document file in NTC
    * sw2w_position (Dict[int, Tuple[int, int]) : The dictionary for converting subword's position to word's position
    """
    prd_word_ids: List[int]
    prd_sent_idx: int
    file_path: str
    sw2w_position: Dict[int, Tuple[int, int]]  # convert a subword position to a word position (sent_idx, word_idx)


class AsGoldPositions(TypedDict):
    """The positions of predicate-arguments in 'input_ids' for training

    * ga (List[int]) : positions of nominative
    *  o (List[int]) : positions of accusative
    * ni (List[int]) : positions of dative
    """

    ga: List[int]
    o: List[int]
    ni: List[int]


class AsGoldExo(TypedDict):
    """The gold labels of exophoric

    When the model considers that an argument does not appear in the given sentences,
    the model selects the CLS token as the argument.
    In this case, the model classifies the argument into the following four categories:
        * 0 -> none           : the argument does not exists
        * 1 -> author (EXO1)  : the type of argument is exophoric named 'author'
        * 2 -> reader (EXO2)  : the type of argument is exophoric named 'reader'
        * 3 -> general (EXOG) : the type of argument is exophoric named 'general' (the rest of exophoric)
    Here, if the argument exists in the given sentences, -100 is assigned to the gold label to avoid computing the loss.

    * ga (int) : gold label of nominative
    *  o (int) : gold label of accusative
    * ni (int) : gold label of dative
    """

    ga: int
    o: int
    ni: int


class AsTrainingInstance(TypedDict):
    """Training instance for AS Model.

    * input_tokens (List[str]) : This must be the following: [CLS] + sentence + [SEP] + ... + [SEP]

    (for inputs)
    * input_ids (Union[torch.LongTensor, List[int]]) : Embedding ids as input
    * predicate_position_ids (List[int]) : Positions of a target predicate in 'input_ids'
    * xs_len (int) : Length of 'input_ids'

    (for golds)
    * gold_positions (AsGoldPositions) : Position of gold labels in 'input_ids'
    * exo_idx (AsGoldExo) : Exophoric gold labels

    (for evaluation)
    * eval_info (PasEvalInfo) : Items for evaluation
    """

    input_tokens: List[str]

    # inputs
    input_ids: Union[torch.LongTensor, List[int]]
    predicate_position_ids: List[int]
    xs_len: int

    # golds
    gold_positions: AsGoldPositions
    exo_idx: AsGoldExo

    # evaluation
    eval_info: PasEvalInfo


class AsBatchInstance(TypedDict):
    """Batch instance for AS model

    (for inputs)
    * input_ids (LongTensor):
        Tensor with shape ('batch_size', 'seq_length'), which is embedding ids
    * predicate_position_ids (List[List[int]]):
        This is the list of positions of predicates, the length of which is 'batch_size'
    * xs_len (LongTensor):
        Tensor with shape ('batch_size', ), which is the numbers of tokens before being padded.

    (for golds)
    * gold_positions (List[AsGoldPositions]): The list of gold positions
    * exo_ids (List[AsGoldExo): The list of exo ids

    (for evaluation)
    * eval_info (List[PasEvalInfo]): The list of eval_info
   """
    # for inputs
    input_ids: torch.LongTensor
    predicate_position_ids: List[List[int]]
    xs_len: torch.LongTensor

    # for golds
    gold_positions: List[AsGoldPositions]
    exo_ids: List[AsGoldExo]

    # for evaluation
    eval_info: List[PasEvalInfo]


class AsOutputForLoss(TypedDict):
    """
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
    label_scores: torch.Tensor
    exo_scores: torch.Tensor


class AsPzeroTrainingInstance(TypedDict):
    """Training instance for AS-Pzero Model.

    * input_tokens (List[str]) : This must be the following: [CLS] + sentence + [SEP] + ... + [SEP]

    (for inputs)
    * input_ids (Union[torch.LongTensor, List[int]]) : Embedding ids as input
    * predicate_position_ids (List[int]) : Positions of a target predicate in 'input_ids'
    * mask_position_id (int) : Position of a masked token
    * xs_len (int) : Length of 'input_ids'

    (for golds)
    * gold_positions (List[int]) : Position of gold labels in 'input_ids'
    * exo_idx (int) : Exophoric gold label

    (for evaluation)
    * eval_info (PasEvalInfo) : Items for evaluation
    * case_name (str) : Target case name (one of 'GA', 'WO', or 'NI')
    """

    input_tokens: List[str]

    # input
    input_ids: Union[torch.LongTensor, List[int]]
    predicate_position_ids: List[int]
    mask_position_id: int
    xs_len: int

    # golds
    gold_positions: List[int]
    exo_idx: int

    # evaluation
    eval_info: PasEvalInfo
    case_name: str


class AsPzeroBatchInstance(TypedDict):
    """Batch instance for AsPzero model

    (for inputs)
    * input_ids (torch.LongTensor):
        Tensor with shape ('batch_size', 'seq_length'), which is embedding ids
    * predicate_position_ids (List[List[int]]):
        This is the list of positions of predicates, the length of which is 'batch_size'
    * mask_position_ids (List[int]):
        The length of mask_position_ids is 'batch_size'. Each number is the position of mask token.
    * xs_len (torch.LongTensor):
        Tensor with shape ('batch_size', ), which is the numbers of tokens before being padded.

    (for golds)
    * gold_positions (List[List[int]]) : Positions of gold labels in 'input_ids'
    * exo_ids (torch.LongTensor) : Exophoric gold labels

    (for evaluation)
    * eval_info (List[PasEvalInfo]) : Items for evaluation
    * case_names (List[str]) : Target case names (one of 'GA', 'WO', or 'NI')
    """
    # for inputs
    input_ids: torch.LongTensor
    predicate_position_ids: List[List[int]]
    mask_position_ids: List[int]
    xs_len: torch.LongTensor

    # for golds
    gold_positions: List[List[int]]
    exo_ids: torch.LongTensor

    # for evaluation
    eval_info: List[PasEvalInfo]
    case_names: List[str]


class AsPzeroOutputForLoss(TypedDict):
    """
    * selection_scores:
        Tensors with shape ('batch_size', 'seq_length'),
        which is score indicates whether a word is the argument of the given predicate or not.

    * exo_scores:
        Tensors with shape ('batch_size', 4) which is scores of exophoric.
        When the model selects the cls token as a predicate-argument
        (the model condiders that the argument doesn't appear in the given sentences),
        the model classify the argument into the following four categories: (author, reader, general, none).
    """
    selection_scores: torch.Tensor
    exo_scores: torch.Tensor


class PasOutputForInference(TypedDict):
    """
    * predicts (List[int]): Predictions about 'intra', 'inter'
    * exo_predicts ([List[int]): Predictions about 'exophoric'
    """
    predicts: List[int]
    exo_predicts: List[int]


class PasEvaluationInstance(TypedDict):
    """
    * predicts (List[int]): the prediction of the model for 'dep', 'intra', and 'inter'
    * exo_predicts (List[int]): the prediction of the model for 'exophora'
    * golds (List[List[int]]): the gold labels for 'dep', 'intra', and 'inter'
    * exo_golds (List[int]): the gold labels for 'exophora'
    * case_names (List[str]): the list of case names (e.g. 'ga', 'o', 'ni')
    * eval_infos (List[PasEvalInfo]): the list of 'PasEvalInfo'
    """
    predicts: List[int]
    exo_predicts: List[int]
    golds: List[List[int]]
    exo_golds: List[int]
    case_names: List[str]
    eval_infos: List[PasEvalInfo]


class PasDecodeInstance(TypedDict):
    """
    * sent (int): index of sentences
    * id   (int): index of words

    For the exophoric case:
        * EXO1  (author): sent = -1, id = -1
        * EXO2  (reader): sent = -2, id = -1
        * EXOG (general): sent = -3, id = -1
    """
    sent: int
    id: int


# batch functions
def create_pzero_batch_instance(
        pzero_masked_instances: List[PzeroMaskedInstance],
        padding_value: int = 0
) -> PzeroBatchInstance:
    """
    Args:
        pzero_masked_instances (List[PzeroMaskedInstance]): instances to create a batch
        padding_value (int): Pad token embed id

    Return:
        PzeroBatchInstance
    """

    input_ids = pad_sequence(
        [torch.LongTensor(inst["input_ids"]) for inst in pzero_masked_instances],
        batch_first=True,
        padding_value=padding_value
    )
    masked_ids = torch.LongTensor(
        [inst["masked_idx"] for inst in pzero_masked_instances]
    )
    xs_len = torch.LongTensor(
        [len(inst["input_ids"]) for inst in pzero_masked_instances]
    )
    gold_ids = [inst["gold_ids"] for inst in pzero_masked_instances]

    batch = PzeroBatchInstance(
        input_ids=input_ids,
        masked_ids=masked_ids,
        xs_len=xs_len,
        gold_ids=gold_ids,
    )

    return batch


def create_as_batch_instance(
        training_instances: List[AsTrainingInstance],
        padding_value: int = 0
) -> AsBatchInstance:
    """
    Args:
        training_instances (List[AsTrainingInstance): instances to create a batch
        padding_value (int):  Pad token embed id

    Returns:
        batch (AsBatchInstance)
    """
    input_ids = pad_sequence(
        [torch.LongTensor(inst["input_ids"]) for inst in training_instances],
        batch_first=True,
        padding_value=padding_value,
    )

    batch = AsBatchInstance(
        input_ids=input_ids,
        predicate_position_ids=[inst["predicate_position_ids"] for inst in training_instances],
        xs_len=torch.LongTensor([inst["xs_len"] for inst in training_instances]),
        gold_positions=[inst["gold_positions"] for inst in training_instances],
        exo_ids=[inst["exo_idx"] for inst in training_instances],
        eval_info=[inst["eval_info"] for inst in training_instances]
    )

    return batch


def create_as_pzero_batch_instance(
        training_instances: List[AsPzeroTrainingInstance],
        padding_value: int = 0
) -> AsPzeroBatchInstance:
    """
    Args:
        training_instances (List[AsPzeroTrainingInstance]): instances to create a batch
        padding_value (int): Pad token embed id

    Return:
        batch (AsPzeroBatchInstance)
    """

    input_ids = pad_sequence(
        [torch.LongTensor(inst["input_ids"]) for inst in training_instances],
        batch_first=True,
        padding_value=padding_value
    )

    batch = AsPzeroBatchInstance(
        input_ids=input_ids,
        predicate_position_ids=[inst["predicate_position_ids"] for inst in training_instances],
        mask_position_ids=[inst["mask_position_id"] for inst in training_instances],
        xs_len=torch.LongTensor([inst["xs_len"] for inst in training_instances]),
        gold_positions=[inst["gold_positions"] for inst in training_instances],
        exo_ids=torch.LongTensor([inst["exo_idx"] for inst in training_instances]),
        case_names=[inst["case_name"] for inst in training_instances],
        eval_info=[inst["eval_info"] for inst in training_instances],
    )

    return batch


class LossFunction(ABC):
    def __call__(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)

    def compute_loss(self, batch: TypedDict, predicts: TypedDict, device: torch.device = None) -> torch.Tensor:
        """
        Args:
            batch (TypedDict): a mini-batch containing inputs and golds
            predicts (TypedDict): prediction of a model
            device (torch.device): a device to send data

        Returns:
            loss (torch.Tensor)
        """
        raise NotImplementedError()
