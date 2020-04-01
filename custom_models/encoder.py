#!/usr/bin/env python

import torch
import torch.nn as nn

import torchvision

# ==================================================================================================
# -- baseline encoder ------------------------------------------------------------------------------
# ==================================================================================================


class EncoderToRemove(nn.Module):
    def __init__(self, encoded_size, fine_tune=False):
        super(Encoder, self).__init__()

        self.encoded_size = encoded_size

        model = torchvision.models.resnet101(pretrained=True)
        modules = list(model.children())

        # Feature extractor.
        self.feature_extractor = nn.Sequential(*modules[:-1])
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Linear layer.
        num_ftrs = modules[-1].in_features
        self.linear = nn.Linear(num_ftrs, self.encoded_size)

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
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


# ==================================================================================================
# -- attention encoder -----------------------------------------------------------------------------
# ==================================================================================================


class Encoder(nn.Module):
    def __init__(self, encoder_size, fine_tune=False):
        super(Encoder, self).__init__()

        model = torchvision.models.resnet101(pretrained=True)
        modules = list(model.children())

        # Feature extractor. Removes fc and avgpool.
        self.feature_extractor = nn.Sequential(*modules[:-2])
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # Linear layer.
        #num_ftrs = modules[-1].in_features
        self.linear = nn.Linear(2048, encoder_size)

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
        x = self.linear(x)
        # out shape -- (batch_size, 7, 7, encoder_size)
        out = x.view(x.size(0), -1, x.size(3))  # (batch_size, num_pixels, encoder_size)

        #print('Shape ouput encoder: {}'.format(out.shape))
        return out