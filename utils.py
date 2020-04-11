#!/usr/bin/env python

import torch
from vocabulary import SpecialToken


class Word2Idx(object):
    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, tokenized_sequence):
        result = []
        result.append(self._vocab.get_index(SpecialToken.START.value.word))
        result.extend([self._vocab.get_index(token.lower()) for token in tokenized_sequence])
        result.append(self._vocab.get_index(SpecialToken.END.value.word))
        return torch.LongTensor(result)


class IdxToWord(object):
    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, idx_sequence):
        result = []

        special_tokens = [token.value.word for token in SpecialToken]
        for idx in idx_sequence:
            word = self._vocab.get_word(idx)
            if word not in special_tokens or word == SpecialToken.UNK.value.word:
                result.append(word)
        return result
