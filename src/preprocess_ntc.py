# coding=utf-8

import re
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional, Any

import numpy as np
from logzero import logger

from instances import (
    GA,
    WO,
    NI,
    DEP,
    INTRA,
    INTER,
    EXO,
    INB,
    NULL,
    NO_ANSWERS,
    TARGET_CASES,
    CASE_TYPES,
    PasGoldCase,
    PasGoldLabel,
    PasGoldLabels,
    Pas,
    NTCPasDocument,
)
from utils import read_lines

PRED = 'type="pred"'
ALT = 'alt="(.*?)"'

ID = '(?:\s|^)id="(.*?)"'
GA_ID = 'ga="(.*?)"'
WO_ID = 'o="(.*?)"'
NI_ID = 'ni="(.*?)"'

GA_TYPE = 'ga_type="(.*?)"'
WO_TYPE = 'o_type="(.*?)"'
NI_TYPE = 'ni_type="(.*?)"'


def extract_single_value(values: List[Any]):
    """extract the value from a list with only one value"""
    assert len(values) == 1
    return values[0]


class Morph:
    """Morpheme class"""

    def __init__(self, line: str) -> None:
        self.ids: List[int] = []
        self.index: int = -1
        self.bunsetsu_index: int = -1
        self.bunsetsu_head: int = -1
        self.sent_index: int = -1

        self.surface_form = ''
        self.base_form = ''
        self.pos = ''
        self.prd_arg_info = ''

        self.is_predicate: bool = False
        self.has_sahen: bool = False
        self.alt_type: str = ""
        self.case_dict: Dict[str, Tuple[Optional[int], Optional[str]]] = {}  # (case_name: (idx, case_type), ...}
        self.gold_labels: Optional[PasGoldLabels] = None  # will be created by 'Document.create_gold_labels'

        self._set_morph_info(line)
        self._set_id()
        self._set_is_predicate()
        self._set_case_dict()

    def __repr__(self) -> str:
        return self.surface_form

    def _set_morph_info(self, line: str) -> None:
        """Parse the line containing a morpheme and set the information
        e.g.
            目指す[TAB]動詞,*,子音動詞サ行,基本形,目指す,...[TAB]O[TAB]alt="active" ga="10" ga_type="zero" type="pred"
        """

        line = line.split('\t')
        self.surface_form = line[0]
        self.base_form = line[1].split(',')[4] if line[1].split(',')[4] != '*' else self.surface_form
        self.pos = '-'.join(line[1].split(',')[0:3])
        self.prd_arg_info = line[-1] if len(line) == 4 else ''

    def _set_id(self) -> None:
        """Set entity ids"""

        entity_ids = re.findall(pattern=ID, string=self.prd_arg_info)
        self.ids = [int(idx) for idx in entity_ids] if entity_ids else []

    def _set_is_predicate(self) -> None:
        """Set whether or not the morpheme is a predicate"""

        prd_types = []
        for arg_info in self.prd_arg_info.split('&&'):
            if re.findall(pattern=PRED, string=arg_info):
                prd_types.append(arg_info)

        if len(prd_types) == 1:
            self.is_predicate = True

            # active or passive or causative
            self.prd_arg_info = extract_single_value(prd_types)
            alt_type = re.findall(pattern=ALT, string=self.prd_arg_info)
            self.alt_type = extract_single_value(alt_type)

        elif len(prd_types) > 1:
            logger.warning('Double predicates: {surf}\t{info}'.format(surf=self.surface_form, info=self.prd_arg_info))
            self.prd_arg_info = ''

        else:
            self.prd_arg_info = ''

    def _set_case_dict(self) -> None:
        """Set the target cases (arguments)"""

        ga_id, ga_type = self._find_case_info(GA_ID, GA_TYPE)
        wo_id, wo_type = self._find_case_info(WO_ID, WO_TYPE)
        ni_id, ni_type = self._find_case_info(NI_ID, NI_TYPE)
        self.case_dict = {
            GA: (ga_id, ga_type),
            WO: (wo_id, wo_type),
            NI: (ni_id, ni_type),
        }

    def _find_case_info(self, case_name_ptn: str, case_type_ptn: str) -> (Optional[int], Optional[str]):
        """find the index of the case and its type"""

        case_id = re.findall(pattern=case_name_ptn, string=self.prd_arg_info)
        case_type = re.findall(pattern=case_type_ptn, string=self.prd_arg_info)

        case_id = extract_single_value(case_id) if case_id else None
        case_type = extract_single_value(case_type) if case_type else None

        # if the type of zero anaphora is 'exophoric', 'case_id' contains the type of exophoric instead of index.
        # For example,
        #   * intra/inter: ga="12" ga_type="zero"
        #   * exophoric  : ga="exog" ga_type="zero"
        if case_id in EXO:
            case_type = case_id
            case_id = -1
        elif case_id is not None:
            case_id = int(case_id)

        return case_id, case_type


class Bunsetsu:
    """Bunsetsu (chunk) class"""

    def __init__(self, index: int, head: int) -> None:
        self.index = index
        self.head = head
        self.morphs = []

    def __repr__(self) -> str:
        return " ".join(str(morph) for morph in self.morphs)

    def add_morph(self, morph: Morph) -> None:
        """Add a morpheme to the list of morphemes"""
        self.morphs.append(morph)


class Sentence:
    """Sentence class"""

    def __init__(self, index: int) -> None:
        self.index = index
        self.bunsetsus: List[Bunsetsu] = []
        self.morphs: List[Morph] = []
        self.predicates: List[Morph] = []

        self.n_bunsetsus = 0
        self.n_morphs = 0
        self.n_cases = np.zeros(shape=(len(TARGET_CASES), len(CASE_TYPES)), dtype='int32')

    def __repr__(self) -> str:
        return " / ".join(str(bunsetsu) for bunsetsu in self.bunsetsus)

    def add_bunsetsu(self, bunsetsu: Bunsetsu) -> None:
        """Add a bunsetsu (chunk) to the list of bunsetsus (chunks)"""

        self.bunsetsus.append(bunsetsu)
        self.morphs.extend(bunsetsu.morphs)
        self.n_bunsetsus += 1

        for morph in bunsetsu.morphs:
            morph.index = self.n_morphs
            morph.bunsetsu_index = bunsetsu.index
            morph.bunsetsu_head = bunsetsu.head
            morph.sent_index = self.index
            self.n_morphs += 1

            if morph.is_predicate:
                self.predicates.append(morph)

    def set_sahen(self) -> None:
        """Set whether or not a predicate consists of multiple morphemes"""

        for prd in self.predicates:
            previous_word = self.morphs[prd.index - 1]
            if prd.index != 0 and prd.pos.startswith('動詞') and previous_word.pos.split('-')[1] == 'サ変名詞':
                prd.has_sahen = True

    def count_cases(self) -> None:
        """This operation must be performed after the 'Document.update_case_dict' is done."""

        for morph in self.predicates:
            for case_name, gold_label in morph.gold_labels.items():
                case_name_index = TARGET_CASES.index(case_name)
                case_type_index = CASE_TYPES.index(gold_label["case_type"])
                self.n_cases[case_name_index][case_type_index] += 1


class Document:
    """Document class"""

    def __init__(self, file_path: str) -> None:
        self.sents = []
        self.file_path = '/'.join(file_path.split('/')[-2:])

    def __repr__(self) -> str:
        return "\n".join(str(sent) for sent in self.sents)

    def add_sent(self, sent: Sentence) -> None:
        """Add a sentence to the list of sentences"""

        sent.set_sahen()
        self.sents.append(sent)

    def create_gold_labels(self) -> None:
        """Create gold labels of a predicate"""

        for target_sent in self.sents:
            for prd in target_sent.predicates:

                gold_labels = {}
                for case_name, (case_id, prd_case_type) in prd.case_dict.items():
                    # if a predicate-argument exists in the input sequence
                    if case_id is not None and prd_case_type not in EXO:
                        gold_cases = self._create_gold_cases(prd, case_id)
                        case_types = [gold_case["case_type"] for gold_case in gold_cases]
                        prd_case_type = self._select_pref_case_type(case_types)

                    # if a predicate-argument doesn't exist in the input sequence
                    else:
                        gold_cases = []

                    # create a new case dictionary
                    gold_label = PasGoldLabel(
                        gold_cases=gold_cases,
                        case_name=case_name,
                        case_type=NULL if prd_case_type is None else prd_case_type,
                    )
                    gold_labels[case_name] = gold_label

                prd.gold_labels = gold_labels

    def _create_gold_cases(self, target_prd: Morph, target_case_id: int) -> List[PasGoldCase]:
        """Create the list of gold cases (arguments)"""

        gold_cases: List[PasGoldCase] = []

        for morph in self._find_target_morph(target_case_id):
            if target_prd.sent_index != morph.sent_index:
                gold_case_type = INTER
            elif target_prd.bunsetsu_head == morph.bunsetsu_index:
                gold_case_type = DEP
            elif target_prd.bunsetsu_index == morph.bunsetsu_head:
                gold_case_type = DEP
            elif target_prd.bunsetsu_index == morph.bunsetsu_index:
                gold_case_type = INB
            else:
                gold_case_type = INTRA

            gold_case = PasGoldCase(
                sent_idx=morph.sent_index,
                word_idx=morph.index,
                case_type=gold_case_type,
            )
            gold_cases.append(gold_case)

        return gold_cases

    def _find_target_morph(self, case_id: int) -> Generator[Morph, None, None]:
        """This function looks for a morpheme that has the same id as the given id."""

        for sentence in self.sents:
            for morph in sentence.morphs:
                if case_id != -1 and case_id in morph.ids:
                    yield morph

    @staticmethod
    def _select_pref_case_type(case_types: List[str]) -> str:
        """
        If multiple case types exist, determine the case type in the following order of priority:
            DEP -> INTRA -> INTER -> INB
        """

        if DEP in case_types:
            pref_case_type = DEP
        elif INTRA in case_types:
            pref_case_type = INTRA
        elif INTER in case_types:
            pref_case_type = INTER
        elif INB in case_types:
            pref_case_type = INB
        else:
            raise ValueError("Cannot find preferred case type: {}".format(case_types))

        return pref_case_type


def create_corpus(file_paths: List[str]) -> List[Document]:
    """Read a file and create a document class"""

    corpus: List[Document] = []

    for file_path in file_paths:
        assert Path(file_path).exists(), f"Not found: {file_path}"

        doc = Document(file_path)
        n_sent = 0
        sent = Sentence(n_sent)
        bunsetsu = None

        for line in read_lines(file_path, print_log=False):
            line = line.rstrip()

            # End of sentence
            if line.startswith('EOS'):
                sent.add_bunsetsu(bunsetsu)
                doc.add_sent(sent)
                n_sent += 1
                sent = Sentence(n_sent)
                bunsetsu = None

            # Bunsetsu begins
            elif line.startswith('*'):
                if bunsetsu:
                    sent.add_bunsetsu(bunsetsu)
                _, sent_index, head, *_ = line.split()
                bunsetsu = Bunsetsu(index=int(sent_index),
                                    head=int(head[:-1]))

            # Morphological (Token) unit
            else:
                bunsetsu.add_morph(Morph(line))

        doc.create_gold_labels()
        corpus.append(doc)

    return corpus


def print_stats(corpus: List[Document]) -> None:
    """Print the status of corpus"""

    n_docs = len(corpus)
    n_sents = 0
    n_prds = 0
    n_cases = np.zeros(shape=(len(TARGET_CASES), len(CASE_TYPES)), dtype='int32')
    for doc in corpus:
        n_sents += len(doc.sents)
        for sent in doc.sents:
            sent.count_cases()
            n_prds += len(sent.predicates)
            n_cases += sent.n_cases

    # print out
    logger.info(f'# of documents: {n_docs}')
    logger.info(f'# of sentences: {n_sents}')
    logger.info(f'# of predicates: {n_prds}')
    for case_index, case_name in enumerate(TARGET_CASES):
        n_case_types = ', '.join(f'{case_type}: {n_cases[case_index][i]:>5}' for i, case_type in enumerate(CASE_TYPES))
        logger.info(f'  - [{case_name}] ' + n_case_types + f', TOTAL: {sum(n_cases[case_index]):>5}')


def create_ntc_pas_document(doc: Document) -> NTCPasDocument:
    """Create 'NTCPasDocument' from 'Document'"""

    file_path: str = doc.file_path
    sents: List[List[str]] = [[morph.surface_form for morph in sent.morphs] for sent in doc.sents]
    sent_ids: List[int] = [sent.index for sent in doc.sents]
    pas_list: List[Pas] = []

    for sent in doc.sents:
        for prd in sent.predicates:
            prd_word_ids = [prd.index - 1, prd.index] if prd.has_sahen else [prd.index]

            for case_name, gold_label in prd.gold_labels.items():

                # In our paper, the case with the type 'INB' is excluded
                gold_label["gold_cases"] = [case for case in gold_label["gold_cases"] if case["case_type"] is not INB]
                if gold_label["case_type"] == INB:
                    gold_label["case_type"] = NULL

                # check the bug
                if gold_label["case_type"] in NO_ANSWERS:
                    assert len(gold_label["gold_cases"]) == 0
                else:
                    assert len(gold_label["gold_cases"]) != 0

            pas = Pas(
                prd_sent_idx=prd.sent_index,
                prd_word_ids=prd_word_ids,
                gold_labels=prd.gold_labels,
                alt_type=prd.alt_type,
            )
            pas_list.append(pas)

    return NTCPasDocument(
        file_path=file_path,
        sents=sents,
        sent_ids=sent_ids,
        pas_list=pas_list,
    )
