#!/usr/bin/env python

import collections
import enum


class SpecialToken(enum.Enum):
    START="<start>"
    END="<end>"
    PAD="<pad>"
    UNK="<unk>"


class Vocabulary(object):

    def __init__(self, min_freq=1):
        self.min_freq = min_freq

        self._word2idx = {}
        self._idx2word = {}

        self._counter = collections.Counter()

    def __len__(self):
        return len(self._word2idx)
    
    def __contains__(self, word):
        return word in self._word2idx

    def add_word(self, word):
        """
        Adds a new word to the vocabulary and updates the internal counter.

            :param word: word to be added.
            :return: True if the word is successfully added. Otherwise, False.
        """
        # Ensures that the given word is lowercase.
        _word = word.lower()
        
        self._counter.update({_word: 1})
        if self._counter[word] > self.min_freq and not _word in self._word2idx:
            new_idx = len(self._word2idx)
            self._word2idx[_word] = new_idx
            self._idx2word[new_idx] = _word
            return True
        return False

    def get_word(self, idx):
        return self._idx2word.get(idx, None)

    def get_index(self, word):
        return self._word2idx.get(word, None)

    def frequency(self, word):
        """
        Returns the number of occurrences of the given word.
        """
        return self._counter[word]

    def most_common(self, n=5):
        """
        Returns a list of the n most common elements and their counts.
        """
        return self._counter.most_common(n)

    def least_common(self, n=5):
        """
        Returns a list of the n most common elements and their counts.
        """
        return self._counter.most_common()[:-n-1:-1] 

    def encode(self, sequence=[]):
        """
        Given a tokenized sequence, returns a vector with the corresponding index of each token.
        """
        return [self.get_index(word) for word in sequence]

    def decode(self, sequence=[]):
        """
        Given an encoded sequence (i.e., each element represents the index of each token), returns
        the tokenized sequence. 
        """
        return [self.get_word(idx) for idx in sequence]


def build_vocabulary_from_tokenized_captions_flickr(filename, min_freq=1):
    vocab = Vocabulary(min_freq)

    # Processing file with tokenized captions.
    with open(filename) as f:
        for line in f.readlines():
            for token in line.split()[1:]:
                vocab.add_word(token)

    # Adding special tokens
    for special_token in SpecialToken:
        vocab.add_word(special_token.value)

    return vocab