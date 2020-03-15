#!/usr/bin/env python

import torch
from torchvision import transforms

from vocabulary import SpecialToken


class Word2Idx(object):
    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, tokenized_sequence):
        result = []
        result.append(self._vocab.get_index(SpecialToken.START.value))
        result.extend([
            self._vocab.get_index(token.lower())
            for token in tokenized_sequence
        ])
        result.append(self._vocab.get_index(SpecialToken.END.value))
        return torch.LongTensor(result)


class IdxToWord(object):
    def __init__(self, vocabulary):
        self._vocab = vocabulary

    def __call__(self, idx_sequence):
        result = [self._vocab.get_word(idx) for idx in idx_sequence]
        return result


class NormalizeImageNet(transforms.Normalize):
    def __init__(self):
        super(NormalizeImageNet, self).__init__(mean=(0.485, 0.456, 0.406),
                                                std=(0.229, 0.224, 0.225))
