# coding=utf-8

import unittest

from collections import defaultdict
from src.tokenizer import JapaneseCharTokenizer, JapaneseTokenizer, SpecialTokens


class TestJapaneseCharTokenizer(unittest.TestCase):
    sp_tokens = SpecialTokens(
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    use_chars = "テストの文です。東京タワー"

    vocab = {}
    for idx, token in enumerate(sp_tokens.values()):
        vocab[token] = idx
    for idx, char in enumerate(use_chars, len(sp_tokens)):
        vocab[char] = idx

    tokenizer = JapaneseCharTokenizer(vocab=vocab, special_tokens=sp_tokens)

    def test_tokenize(self):
        text = "[CLS]テストの[MASK]です。[SEP][PAD][PAD]"
        expected = ['[CLS]', 'テ', 'ス', 'ト', 'の', '[MASK]', 'で', 'す', '。', '[SEP]', '[PAD]', '[PAD]']
        actual = self.tokenizer.tokenize(text)
        assert expected == actual

    def test_tokenize_into_words(self):
        text = "[CLS]テストの[MASK]です。[SEP][PAD][PAD]"
        expected = ['[CLS]', 'テ', 'ス', 'ト', 'の', '[MASK]', 'で', 'す', '。', '[SEP]', '[PAD]', '[PAD]']
        actual = self.tokenizer.tokenize_into_words(text)
        assert expected == actual

    def test_tokenize_into_subwords(self):
        word = "東京タワー"
        expected = ['東', '京', 'タ', 'ワ', 'ー']
        actual = self.tokenizer.tokenize_into_words(word)
        assert expected == actual

    def test_convert_tokens_to_ids(self):
        tokens = ['[CLS]', 'テ', 'ス', 'ト', 'の', '絵', 'で', 'す', '。', '[SEP]', '[PAD]', '[PAD]']
        expected = [2, 5, 6, 7, 8, 1, 10, 11, 12, 3, 0, 0]
        actual = self.tokenizer.convert_tokens_to_ids(tokens)
        assert expected == actual

    def test_convert_ids_to_tokens(self):
        ids = [2, 5, 6, 7, 8, 1, 10, 11, 12, 3]
        expected = ['[CLS]', 'テ', 'ス', 'ト', 'の', '[UNK]', 'で', 'す', '。', '[SEP]']
        actual = self.tokenizer.convert_ids_to_tokens(ids)
        assert expected == actual

    def test_build_inputs_with_special_tokens(self):
        ids = [5, 6, 7, 8, 9, 10, 11, 12]
        expected = [2, 5, 6, 7, 8, 9, 10, 11, 12, 3]
        actual = self.tokenizer.build_inputs_with_special_tokens(ids)
        assert expected == actual

    def test_decode(self):
        ids = [2, 5, 6, 7, 8, 9, 10, 11, 12, 3]
        expected = '[CLS]テストの文です。[SEP]'
        actual = self.tokenizer.decode(ids)
        assert expected == actual

        expected = 'テストの文です。'
        actual = self.tokenizer.decode(ids, skip_special_tokens=True)
        assert expected == actual


class TestJapaneseTokenizer(unittest.TestCase):
    tokenizer = JapaneseTokenizer()

    def test_tokenize(self):
        text = "これはテストの文です。"
        expected = ['これ', 'は', 'テスト', 'の', '文', 'です', '。']
        actual = self.tokenizer.tokenize(text)
        assert expected == actual

    def test_tokenize_into_words(self):
        text = "これはテストの文です。"
        expected = ['これ', 'は', 'テスト', 'の', '文', 'です', '。']
        actual = self.tokenizer.tokenize_into_words(text)
        assert expected == actual

    def test_tokenize_into_subwords(self):
        word = "東京タワー"
        expected = ['東京', '##タワー']
        actual = self.tokenizer.tokenize_into_subwords(word)
        assert expected == actual

    def test_convert_tokens_to_ids(self):
        tokens = ['これ', 'は', 'テスト', 'の', '文', 'です', '。']
        expected = [171, 9, 4609, 5, 214, 2992, 8]
        actual = self.tokenizer.convert_tokens_to_ids(tokens)
        assert expected == actual

    def test_convert_ids_to_tokens(self):
        ids = [171, 9, 4609, 5, 214, 2992, 8]
        expected = ['これ', 'は', 'テスト', 'の', '文', 'です', '。']
        actual = self.tokenizer.convert_ids_to_tokens(ids)
        assert expected == actual

    def test_build_inputs_with_special_tokens(self):
        ids = [171, 9, 4609, 5, 214, 2992, 8]
        expected = [2, 171, 9, 4609, 5, 214, 2992, 8, 3]
        actual = self.tokenizer.build_inputs_with_special_tokens(ids)
        assert expected == actual

    def test_decode(self):
        ids = [2, 171, 9, 4609, 5, 214, 2992, 8, 3]
        expected = '[CLS] これ は テスト の 文 です 。 [SEP]'
        actual = self.tokenizer.decode(ids)
        assert expected == actual

        expected = 'これ は テスト の 文 です 。'
        actual = self.tokenizer.decode(ids, skip_special_tokens=True)
        assert expected == actual
