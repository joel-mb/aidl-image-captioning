#!/usr/bin/env python

import logging

import torch
import torch.nn as nn

import torchvision

import pretrainedmodels

# ==================================================================================================
# -- encoder factory -------------------------------------------------------------------------------
# ==================================================================================================


class EncoderFactory(object):
    @staticmethod
    def get_encoder(type, encoder_size):
        if type == 'resnet101':
            logging.info('Building resnet101 encoder...')
            return get_resnet101_encoder(encoder_size)
        elif type == 'senet154':
            logging.info('Building senet154 encoder...')
            return get_senet154_encoder(encoder_size)
        elif type == 'vgg19':
            logging.info('Building vgg19 encoder...')
            return get_vgg19_encoder(encoder_size)


def get_resnet101_encoder(encoder_size):
    model = pretrainedmodels.__dict__['resnet101'](pretrained='imagenet')
    modules = list(model.children())

    # Feature extractor. Removes fc and avgpool.
    feature_extractor = nn.Sequential(*modules[:-2])
    linear = nn.Linear(2048, encoder_size)

    return Encoder(feature_extractor, linear, num_pixels=7*7)


def get_senet154_encoder(encoder_size):
    model = pretrainedmodels.__dict__['senet154'](pretrained='imagenet')
    modules = list(model.children())

    # Feature extractor. Removes fc, dropout and avgpool.
    feature_extractor = nn.Sequential(*modules[:-3])
    linear = nn.Linear(2048, encoder_size)

    return Encoder(feature_extractor, linear, num_pixels=7*7)


def get_vgg19_encoder(encoder_size):
    model = pretrainedmodels.__dict__['vgg19'](pretrained='imagenet')
    modules = list(model.children())

    # Feature extractor.
    feature_extractor = nn.Sequential(*modules[:-8])
    linear = nn.Linear(3, encoder_size)

    return Encoder(feature_extractor, linear, num_pixels=7*7)


# ==================================================================================================
# -- encoder ---------------------------------------------------------------------------------------
# ==================================================================================================


class Encoder(nn.Module):
    def __init__(self, feature_extractor, linear, num_pixels):
        super(Encoder, self).__init__()
        self.num_pixels = num_pixels

        # Feature extractor. Removes fc and avgpool.
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Linear layer to encoder size.
        self.linear = linear

    def trainable_parameters(self):
        return list(self.linear.parameters())

    def train(self, status=True):
        super(Encoder, self).train(status)
        # As we are applying transfer learning, the encoder feature extractor should be always in
        # evaluation mode.
        self.feature_extractor.eval()

    def forward(self, img):
        """
        Forward method.

            :param img: float tensor of shape (batch_size, channels, height, width)
            :returns: float tensor of shape (batch_size, encoded_size)
        """
        x = self.feature_extractor(img)
        x = x.permute(0, 2, 3, 1)
        x = self.linear(x) # (batch_size, num_pixels, num_pixels, encoder_size)

        out = x.view(x.size(0), -1, x.size(3))  # (batch_size, num_pixels, encoder_size)
        return out