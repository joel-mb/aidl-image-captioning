#!/usr/bin/env python

import torch
import torch.nn as nn

import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        super(AdditiveAttention, self).__init__()

        self.Wc = nn.Linear(hidden_size, attention_size)
        self.Wh = nn.Linear(encoder_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, features, hidden):
        """
        Forward method.

            :param features: encoder output (input)
                - features - float tensor of shape (batch_size, num_pixels, encoder_size)
            :param hidden: decoder output (context)
                -hidden - float tensor of shape (batch_size, hidden_size)
        """
        # ------
        # Scores
        # ------
        # scores -- float tensor of shape (batch_size, num_pixels, attention_size)
        # Internally, the features and hidden tensor have the following shape after the linear
        # layer:
        #    features -- (batch_size, num_pixels, attention_size)
        #    hidden -- (batch_size, 1, attention_size)
        scores = torch.tanh(self.Wh(features) + self.Wc(hidden).unsqueeze(1))

        # float tensor of shape (bath_size, num_pixels, 1)
        scores = self.V(scores)
        scores = scores.squeeze(2)  # Remove last dimension. shape -- (batch_size, num_pixels)

        # ------
        # Alphas
        # ------
        # alphas -- float tensor of shape (batch_size, num_pixels)
        alphas = F.softmax(scores, dim=1)

        # -------
        # Context
        # -------
        context = features * alphas.unsqueeze(2)  # (batch_size, num_pixels, encoder_size)
        context = torch.sum(context, dim=1)  # (batch_size, encoder_size)

        return context, alphas


class MultiplicativeAttention(nn.Module):
    def __init__(self):
        super(MultiplicativeAttention, self).__init__()
