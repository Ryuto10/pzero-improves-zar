# coding=utf-8

from typing import List, Dict, Optional, TypedDict
from abc import ABC
from logzero import logger
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
import re


class SpecialTokens(TypedDict):
    pad_token: str
    unk_token: str
    cls_token: str
    sep_token: str
    mask_token: str


class Tokenizer(ABC):
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = None
        self.model_max_length: int = None

        self.unk_token: str = None
        self.sep_token: str = None
        self.pad_token: str = None
        self.cls_token: str = None
        self.mask_token: str = None

        self.unk_token_id: int = None
        self.sep_token_id: int = None
        self.pad_token_id: int = None
        self.cls_token_id: int = None
        self.mask_token_id: int = None

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens"""
        raise NotImplementedError()

    def tokenize_into_words(self, text: str) -> List[str]:
        """Tokenize text into words
        Args:
            text (str): Target text to tokenize

        Returns:
            words (List[str]): Words
        """
        raise NotImplementedError()

    def tokenize_into_subwords(self, word: str) -> List[str]:
        """Tokenize a word into subwords
        Args:
            word (str): Target word to tokenize

        Returns:
            subwords (List[str]): Subwords
        """
        raise NotImplementedError()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to embed ids"""
        raise NotImplementedError()

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert embed ids to tokens"""
        raise NotImplementedError()

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]]) -> List[int]:
        """Build model inputs from a sequence or a pair of sequence for sequence classification tasks
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        """
        raise NotImplementedError()

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
            spaces_between_special_tokens: bool,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary
        with options to remove special tokens and clean up tokenization spaces.

        Args:
            - token_ids (List[int]):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            _ skip_special_tokens (bool):
                Whether or not to remove special tokens in the decoding.
            _ clean_up_tokenization_spaces (bool):
                Whether or not to clean up the tokenization spaces.
            - spaces_between_special_tokens (bool):
                Whether or not to add spaces around special tokens.
                The behavior of Fast tokenizers is to have this to :obj:`False`.
                This is setup to :obj:`True` in slow tokenizers for backward compatibility.

        Returns:
            - The decoded sentence (str)
        """
        raise NotImplementedError()


class JapaneseTokenizer(Tokenizer):
    """Tokenizer used in the paper"""

    def __init__(self):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking'
        )
        self.vocab = self.tokenizer.vocab
        self.model_max_length = self.tokenizer.model_max_length

        self.unk_token = self.tokenizer.unk_token
        self.sep_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.cls_token = self.tokenizer.cls_token
        self.mask_token = self.tokenizer.mask_token

        self.unk_token_id = self.tokenizer.unk_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer.tokenize(text)

    def tokenize_into_words(self, text: str) -> List[str]:
        return self.tokenizer.word_tokenizer.tokenize(text)

    def tokenize_into_subwords(self, word: str) -> List[str]:
        return self.tokenizer.subword_tokenizer.tokenize(word)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return self.tokenizer.build_inputs_with_special_tokens(token_ids_0, token_ids_1)

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
            spaces_between_special_tokens: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens,
            clean_up_tokenization_spaces,
            spaces_between_special_tokens
        )


class JapaneseCharTokenizer(Tokenizer):
    """Character-based tokenizer"""

    def __init__(self, vocab: Dict[str, int], special_tokens: SpecialTokens, model_max_length: int = 512) -> None:
        super().__init__()
        self.vocab = vocab
        self.model_max_length = model_max_length

        self.special_tokens = special_tokens
        self.unk_token = self.special_tokens["unk_token"]
        self.sep_token = self.special_tokens["sep_token"]
        self.pad_token = self.special_tokens["pad_token"]
        self.cls_token = self.special_tokens["cls_token"]
        self.mask_token = self.special_tokens["mask_token"]

        self.unk_token_id = self.vocab[self.unk_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.pad_token_id = self.vocab[self.pad_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.mask_token_id = self.vocab[self.mask_token]

        self.ids_to_tokens: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.remove_ids = {self.cls_token_id, self.sep_token_id}

        re_special_tokens = {"*", "+", ".", "?", "{", "}", "(", ")", "[", "]", "^", "$", "-", "|", "/"}
        sp_tokens_str = "|".join(
            "".join(f'\{ch}' if ch in re_special_tokens else ch for ch in token)
            for token in self.special_tokens.values()
        )
        self.separate_str = f"({sp_tokens_str})"

    def tokenize(self, text: str) -> List[str]:
        tokenized_tokens = []
        for chunk in re.split(self.separate_str, text):
            if chunk in self.special_tokens.values():
                tokenized_tokens.append(chunk)
            else:
                tokenized_tokens.extend(list(chunk))

        return tokenized_tokens

    def tokenize_into_words(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize_into_subwords(self, word: str) -> List[str]:
        return self.tokenize(word)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab[t] if t in self.vocab else self.unk_token_id for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens[idx] for idx in ids]

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]

        return cls + token_ids_0 + sep + token_ids_1 + sep

    def decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
            spaces_between_special_tokens: bool = False,
    ) -> str:
        tokens: List[str] = []
        for idx in token_ids:
            if skip_special_tokens and idx in self.remove_ids:
                continue
            tokens.append(self.ids_to_tokens[idx])

        if spaces_between_special_tokens:
            text = " ".join(tokens)
        else:
            text = "".join(tokens)

        return text


def load_tokenizer() -> Tokenizer:
    logger.info("Loading tokenizer ...")
    tokenizer = JapaneseTokenizer()

    return tokenizer
