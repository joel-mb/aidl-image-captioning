import torch
import torch.nn as nn

import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        super(AdditiveAttention, self).__init__()

        self.Wh = nn.Linear(encoder_size, attention_size)
        self.Wc = nn.Linear(hidden_size, attention_size)
        self.V = nn.Linear(attention_size, 1)

    def forward(self, features, hidden):
        """
        Forward method.

            :param features: float tensor of shape (batch_size, num_pixels, encoder_size) -- encoder out
            :param hidden: float tensor of shape (batch_size, hidden_size). -- decoder out
        """
        #print('----Attention---')
        #print('Input: features: {}, hidden: {}'.format(features.shape, hidden.shape))
        # ------
        # Scores
        # ------
        # float tensor of shape (batch_size, num_pixels, attention_size)
        scores = torch.tanh(self.Wh(features) + self.Wc(hidden).unsqueeze(1))
        #print('Scores 1: {}'.format(scores.shape))

        # float tensor of shape (bath_size, num_pixels, 1)
        scores = self.V(scores)
        #print('Scores 2: {}'.format(scores.shape))
        scores = scores.squeeze(2)  # Remove last dimension. shape -- (batch_size, num_pixels)
        #print('Scores 3: {}'.format(scores.shape))

        # ------
        # Alphas
        # ------
        # float tensor of shape (batch_size, num_pixels)
        alphas = F.softmax(scores, dim=1)
        #print('alphas: {}'.format(alphas.shape))

        # -------
        # Context
        # -------
        context = features * alphas.unsqueeze(2)
        #context = torch.bmm(t, features)  # (batch_size, num_pixels, encoder_size)
        #print('Context 1: {}'.format(context.shape))
        context = torch.sum(context, dim=1)  # (batch_size, encoder_size)
        #print('Context 2: {}'.format(context.shape))
        
        #print('Out context: {}, out alphas: {}'.format(context.shape, alphas.shape))
        return context, alphas


class MultiplicativeAttention(nn.Module):
    def __init__(self):
        super(MultiplicativeAttention, self).__init__()
