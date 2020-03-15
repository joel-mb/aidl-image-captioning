#!/usr/bin/env python

import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from util import Flickr8kFolder


class Flickr8kDataset(Dataset):
    """
    """

    def __init__(self,
                 data_folder,
                 split='train',
                 transform=None,
                 target_transform=None):
        super(Flickr8kDataset, self).__init__()

        self._data_folder = data_folder
        self._transform = transform
        self._target_transform = target_transform

        if split == 'train':
            _split_file = data_folder.train_imgs_file
        elif split == 'test':
            _split_file = data_folder.test_imgs_file
        elif split == 'eval':
            _split_file = data_folder.eval_imgs_file

        # Processing split file.
        split_samples = []
        with open(_split_file) as f:
            split_samples = [line[:-1] for line in f.readlines()]

        # Proccessing tokenized captions.
        self.samples = []
        with open(data_folder.ann_file) as f:
            for line in f.readlines():
                line = line.split()
                image_id, tokens = line[0], line[1:]
                if image_id[:-2] in split_samples:
                    self.samples.append((image_id, tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        image_id, tokens = self.samples[key]

        # Processing image
        filename = os.path.join(self._data_folder.img_root, image_id[:-2])
        image = Image.open(filename).convert('RGB')
        if self._transform is not None:
            image = self._transform(image)

        # Processing caption.
        caption = tokens
        if self._target_transform is not None:
            caption = self._target_transform(tokens)

        return image, caption

def flickr_collate_fn(batch):
    #print('[flickr_collate] batch size: {}, batch type: {}'.format(len(batch), type(batch)))
    imgs = [sample[0] for sample in batch]
    captions = [sample[1] for sample in batch]
    lengths =  [len(caption) for caption in captions]

    #for caption in captions:
    #    print('[flickr_collate] caption: {}'.format(caption))
    captions = pad_sequence(captions, batch_first=True, padding_value=2)
    #print('[flickr_collate] captions shape: {}'.format(captions.shape))
    #print('[flickr_collate] legths: {}'.format(lengths))
    
    return torch.stack(imgs), captions, lengths
