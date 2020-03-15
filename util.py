#!/usr/bin/env python

import os


class Flickr8kFolder(object):
    """
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
        return os.path.join(self.text_root, 'Flickr8k.token.txt')

    @property
    def train_imgs_file(self):
        return os.path.join(self.text_root, 'Flickr_8k.trainImages.txt')

    @property
    def test_imgs_file(self):
        return os.path.join(self.text_root, 'Flickr_8k.testImages.txt')

    @property
    def eval_imgs_file(self):
        return os.path.join(self.text_root, 'Flickr_8k.devImages.txt')

def show_image_captioning(self, image):
    # Show image and caption. This can be used for inference when sampling.
    pass