#!/usr/bin/env python

import os

import imageio
from torch.utils.data import Dataset

from vocabulary import SpecialToken


class FlickrDataset(Dataset):

    def __init__(self, data_root, vocabulary, transform=None):
        super(FlickrDataset, self).__init__()

        self.data_root = data_root
        self.vocab = vocabulary
        self.transform = transform

        # Proccessing tokenized captions.
        self.samples = []
        flickr_tokenized_file = os.path.join(data_root, "Flickr8k_Text", "Flickr8k.token.txt")
        with open(flickr_tokenized_file) as f:
            for line in f.readlines():
                line = line.split()
                image_id, tokens = line[0], line[1:]
                self.samples.append((image_id, tokens))

    def __len__(self):
        return len(self.samples)

    def _get_image_path(self, image_id):
        image_name = image_id[:-2]
        return os.path.join(self.data_root, "Flickr8k_Dataset", image_name)

    def __getitem__(self, key):
        image_id, tokens = self.samples[key]
        
        # Processing image
        image = imageio.imread(self._get_image_path(image_id))
        if self.transform is not None:
            image = self.transform(image)

        # Processing caption.
        caption = []
        caption.append(self.vocab.get_index(SpecialToken.START.value))
        caption.extend([self.vocab.get_index(token) for token in tokens])
        caption.append(self.vocab.get_index(SpecialToken.END.value))

        return image, caption