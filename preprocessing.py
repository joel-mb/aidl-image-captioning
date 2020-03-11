#!/usr/bin/env python

from vocabulary import SpecialToken


class Word2Idx(object):

    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, tokenized_sequence, add_start_end=True):
        # TODO(joel): Add start and end specia tokens
        return [self._vocab.get_index(token) for token in tokenized_sequence]


class IdxToWord(object):

    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, idx_sequence):
        # TODO(joel): Remove start end special tokens?
        return [self._vocab.get_word(idx) for idx in idx_sequence]


class NormalizeImageNet(transform.Normalize):

    def __call__(self):
        super(NormalizeImageNet, self).__init__(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
