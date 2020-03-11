#!/usr/bin/env python

import enum
import os

import imageio  # TODO(joel): Replace with PIL
from torch.utils.data import Dataset

from vocabulary import SpecialToken


class Split(enum.Enum):
    TRAIN = 0
    TEST = 1
    EVAL = 2


class Flickr8kDataset(Dataset):

    def __init__(self, img_root, ann_root, split=Split.TRAIN, transform=None, target_transform=None):
        super(Flickr8kDataset, self).__init__()

        self._img_root = img_root
        self._ann_root = ann_root
        self._split = split
        self._transform = transform
        self._target_transform = target_transform

        if self._split == Split.TRAIN:
            ann_file = '...'
        elif self._split == Split.TEST:
            ann_file = '...'
        elif self._split == Split.EVAL:
            ann_file = '...'
        self._ann_file = os.path.join(self._ann_root, ann_file)

        # Proccessing tokenized captions.
        self.samples = []
        with open(self._ann_file) as f:
            for line in f.readlines():
                line = line.split()
                image_id, tokens = line[0], line[1:]
                self.samples.append((image_id, tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        image_id, tokens = self.samples[key]
        
        # Processing image
        image = imageio.imread(os.path.join(self._img_root, image_id))
        if self.transform is not None:
            image = self.transform(image)

        # Processing caption.
        caption = tokens
        if self.target_transform is not None:
            caption = self.target_transform(tokens)

        return image, caption