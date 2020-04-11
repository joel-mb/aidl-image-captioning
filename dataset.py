#!/usr/bin/env python

import os

import torch
import PIL
from torch.nn.utils.rnn import pad_sequence

from vocabulary import SpecialToken

# ==================================================================================================
# -- helpers ---------------------------------------------------------------------------------------
# ==================================================================================================


class Flickr8kFolder(object):
    """
    Helper object to ease the accessibility to the files defining the dataset.

    WARNING: This class assumes that the data is organized as follows:
        .
        |__ Flickr8k_Dataset
        |   |_ [img_id].jpg
        |   |_ ... 
        |    
        |__ Flickr8k_Text
            |_Flickr8k.token.txt
            |_Flickr_8k.devImages.txt
            |_Flickr_8k.testImages.txt
            |_Flickr_8k.trainImages.txt
    """
    def __init__(self, root):
        self.root = root

    @property
    def img_root(self):
        return os.path.join(self.root, 'Flickr8k_Dataset')

    @property
    def text_root(self):
        return os.path.join(self.root, 'Flickr8k_Text')

    @property
    def ann_file(self):
        """
        Annotation file.

        This file contains the image id and the corresponding caption for each one of the images
        that define the dataset. Each image has five different captions and the captions are already
        tokenized.
        """
        return os.path.join(self.text_root, 'Flickr8k.token.txt')

    @property
    def train_imgs_file(self):
        """
        Train images file.

        This file contains the ids of the images that define the trainning set.
        """
        return os.path.join(self.text_root, 'Flickr_8k.trainImages.txt')

    @property
    def test_imgs_file(self):
        """
        Test images file.

        This file contains the ids of the images that define the testing set.
        """
        return os.path.join(self.text_root, 'Flickr_8k.testImages.txt')

    @property
    def eval_imgs_file(self):
        """
        Evaluation images file.

        This file contains the ids of the images that define the evaluation set.
        """
        return os.path.join(self.text_root, 'Flickr_8k.devImages.txt')


# ==================================================================================================
# -- flickr8k dataset ------------------------------------------------------------------------------
# ==================================================================================================


class Flickr8kDataset(torch.utils.data.Dataset):
    """
    Flickr8k custom dataset.
    """
    def __init__(self, data_folder, split='train', transform=None, target_transform=None):
        super(Flickr8kDataset, self).__init__()

        self._data_folder = data_folder
        self._transform = transform
        self._target_transform = target_transform

        if split == 'train':
            _split_imgs_file = data_folder.train_imgs_file
        elif split == 'test':
            _split_imgs_file = data_folder.test_imgs_file
        elif split == 'eval':
            _split_imgs_file = data_folder.eval_imgs_file

        # Processing split imgs file.
        split_imgs = []  # list of image ids
        with open(_split_imgs_file) as f:
            split_imgs = [line[:-1] for line in f.readlines()]

        # Proccessing tokenized captions. Each sample is defined with the tuple (image_id, [tokens])
        self.samples = []
        with open(data_folder.ann_file) as f:
            for line in f.readlines():
                line = line.split()
                image_id, tokens = line[0], line[1:]
                if image_id[:-2] in split_imgs:
                    self.samples.append((image_id, tokens))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        image_id, tokens = self.samples[key]

        # Processing image
        filename = os.path.join(self._data_folder.img_root, image_id[:-2])
        image = PIL.Image.open(filename).convert('RGB')
        if self._transform is not None:
            image = self._transform(image)

        # Processing caption.
        caption = tokens
        if self._target_transform is not None:
            caption = self._target_transform(tokens)

        return image, caption


def flickr_collate_fn(batch):
    """
    Custom collate function to be used with the data loader.

    This methos returns a padded caption set to be used in the trainning stage (with the <START>
    special token removed) and a padded caption set to be used during the computation loss (with the
    <END> special token removed). The returned data is sorted in descending order based on the
    caption length.

        :param bacth: list of tuples (image, caption) returned by the dataset.
            - img - float tensor of shape (channels, height, width).
            - caption - long tensor of variable length.
        :returns: tuple (imgs, captions_train, captions_loss, lengths)
            - imgs - float tensor of shape (batch_size, channels, height, width)
            - captions_train - long tensor of shape (batch_size, max_seq_length)
            - captions_loss - long tensor of shape (batch_size, max_seq_length)
            - lengths - long tensor of shape (batch_size)
    """
    # Sort data by caption length in descending order.
    batch.sort(key=lambda x: x[1].size(0), reverse=True)

    imgs = [sample[0] for sample in batch]
    captions_train = [sample[1][:-1] for sample in batch]  # Remove <end> special token.
    captions_loss = [sample[1][1:] for sample in batch]  # Remove <start> special token.
    lengths = [len(caption) for caption in captions_train]

    # Padding captions.
    captions_train = pad_sequence(captions_train,
                                  batch_first=True,
                                  padding_value=SpecialToken.PAD.value.index)
    captions_loss = pad_sequence(captions_loss,
                                 batch_first=True,
                                 padding_value=SpecialToken.PAD.value.index)

    # Length list to long tensor.
    lengths = torch.tensor(lengths, dtype=torch.int64)

    return torch.stack(imgs), captions_train, captions_loss, lengths
