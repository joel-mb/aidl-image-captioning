#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocabulary import SpecialToken
from .attention import AdditiveAttention

# Checking if GPU is available.
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# ==================================================================================================
# -- decoder factory -------------------------------------------------------------------------------
# ==================================================================================================


class DecoderFactory(object):
    @staticmethod
    def get_decoder(attention_type, embed_size, vocab_size, encoder_size, num_pixels, hidden_size,
                    attention_size):

        if attention_type == 'none':
            logging.info('Building baseline decoder...')
            return Decoder(embed_size, vocab_size, encoder_size, num_pixels, hidden_size)
        elif attention_type == 'additive':
            logging.info('Building additive attention decoder...')
            return DecoderWithAttention(embed_size, vocab_size, encoder_size, num_pixels,
                                        hidden_size, attention_size)


# ==================================================================================================
# -- baseline decoder ------------------------------------------------------------------------------
# ==================================================================================================


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_size, num_pixels, hidden_size):
        super(Decoder, self).__init__()

        self.linear_features = nn.Linear(encoder_size * num_pixels, hidden_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def trainable_parameters(self):
        return list(self.parameters())

    def init_hidden(self, features):
        # Unsqueeze first dimension of features to apply num_layers = 1.
        # Shape after unsqueeze:
        #    features -- (num_layers(1), batch_size, hidden_size).
        features = features.unsqueeze(0)  # Adds num_layer dimension.

        h, c = torch.zeros(features.shape).to(device), features
        return h, c

    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: float tensor of shape (batch_size, num_pixels, encoder_size)
            :param captions: long tensor of shape (batch_size, max_seq_length)
            :param lengths: long tensor of shape (batch_size, ) containing original length for each
                            caption.

        In training mode we use teacher forcing as we know the targets.
        """
        # ----------
        # Embeddings
        # ----------
        # Word indices to embeddings representation. Captions shape after embeddings:
        #    captions -- (batch_size, max_length, embed_size)
        captions = self.embedding(captions)

        # -------------
        # Flat features
        # -------------
        # Shape after linear:
        #    features -- (batch_size, hidden_size)
        features = self.linear_features(features.view(features.size(0), -1))

        # ----
        # LSTM
        # ----
        # In training mode we use teacher forcing as we know the targets.
        #
        # Shape after lstm:
        #    output -- (batch_size, max_seq_length, hidden_size)
        #    state(h) -- (num_layers(1), batch_size, hidden_size)
        #    state(c) -- (num_layers(1), batch_size, hidden_size)
        packed = pack_padded_sequence(captions, lengths, batch_first=True)
        hidden, state = self.lstm(packed, self.init_hidden(features))
        output, _ = pad_packed_sequence(hidden, batch_first=True)

        # -------------------
        # Output linear layer
        # -------------------
        # Shape after linear layer:
        #    output -- (batch_size, max_length, vocab_size)
        output = self.linear(output)

        return output, state

    def _forward_step(self, x, state=None):
        """
        Single forward step.
        
            :param x: input. Shape: (1, ) (word index)
            :param state: tuple (h, c).
                - h - float tensor of shape (num_layers(1), batch_size(1), hidden_size)
                - c - float tensor of shape (num_layers(1), batch_size(1), hidden_size)
        """
        # ---------
        # Embedding
        # ---------
        # Shape after embedding and unsqueeze.
        #    (batch_size(1), seq_length(1), embed_size)
        x = self.embedding(x)
        x = x.unsqueeze(1)  # Unsqueeze to add seq_length(1) dimension.

        # ----------------
        # LSTM single step
        # ----------------
        # Shape after lstm:
        #    hidden -- (batch_size(1), max_length(1), hidden_size)
        #    state -- (num_layers(1), batch_size(1), hidden_size)
        hidden, state = self.lstm(x, state)

        # -------------------
        # Output linear layer
        # -------------------
        # Shape after linear output:
        #    output -- (batch_size(1), max_length(1), vocab_size)
        output = self.linear(hidden)

        # -------
        # Softmax
        # -------
        # Shape after softmax:
        #    output -- (batch_size(1), max_length(1), vocab_size)
        output = F.log_softmax(output, dim=2)
        output = output.squeeze(1)  # Remove max_length(1) dimension.

        return output, state, None

    def sample(self, features, max_seq_length=25):
        """
        Predicts a caption.
            :param features: float tensor of shape (batch_size(1), num_pixels, encoder_size)
            :param max_seq_len: maximum length of the predicted caption.

        In prediction time, the model uses embedding of the previously predicted word and the last
        hidden state.
        """
        output = []

        last_idx_predicted = -1
        while len(output) < max_seq_length and last_idx_predicted != SpecialToken.END.value.index:

            if len(output) == 0:  # First iteration
                init_word = torch.tensor([SpecialToken.START.value.index],
                                         dtype=torch.int64,
                                         device=device)

                hidden = self.linear_features(features.view(features.size(0), -1))
                state = self.init_hidden(hidden)
                out, state, _ = self._forward_step(init_word, state)

            else:
                out, state, _ = self._forward_step(pred, state)

            # out shape -- (batch_size(1), vocab_size)
            _, pred = out.max(1)
            last_idx_predicted = pred.item()

            output.append(last_idx_predicted)

        return output, None


# ==================================================================================================
# -- attention decoder -----------------------------------------------------------------------------
# ==================================================================================================


class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_size, num_pixels, hidden_size,
                 attention_size):
        super(DecoderWithAttention, self).__init__()

        self.linear_features = nn.Linear(encoder_size * num_pixels, hidden_size)

        self.attention = AdditiveAttention(encoder_size, hidden_size, attention_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def trainable_parameters(self):
        return list(self.parameters())

    def init_hidden(self, features):
        h, c = torch.zeros(features.shape).to(device), features
        return h, c

    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: float tensor of shape (batch_size, num_pixels, encoder_size)
            :param captions: long tensor of shape (batch_size, max_seq_length)
            :param lengths: long tensor of shape (batch_size, ) containing original length for each
                            caption.

        In training mode we use teacher forcing as we know the targets.
        """
        alphas_list = []
        predictions = []

        # ----------
        # Embeddings
        # ----------
        # captions -- (batch_size, max_length, embed_size)
        captions = self.embedding(captions)

        # ------------------
        # Initial LSTM state
        # ------------------
        # hidden -- (batch_size, hidden_size)
        hidden = self.linear_features(features.view(features.size(0), -1))
        state = self.init_hidden(hidden)

        # -------------
        # Decoder steps
        # -------------
        # Using teacher forcing
        for i in range(captions.size(1)):
            embeddings = captions[:, i]

            # ---------
            # Attention
            # ---------
            # alphas -- (batch_size, num_pixels)
            # context -- (batch_size, encoder_size)
            context, alphas = self.attention(features, state[0])

            # ---------
            # LSTM step
            # ---------
            # hidden - (batch_size, hidden_size)
            # state -- h: (batch_size, num_layers, hidden_size)
            #          c: (batch_size, num_layers, hidden_size)
            i = torch.cat([embeddings, context], dim=1)
            state = self.lstm_cell(i, state)
            hidden = state[0]

            # ----------
            # Vocabulary
            # ----------
            # Shape after linear layer:
            #    out - (batch_size, vocab_size)
            out = self.linear(hidden)

            # Store predictions and alphas.
            predictions.append(out)
            alphas_list.append(alphas)

        # predictions -- shape (batch_size, max_length, vocab_size)
        # alphas -- shape (batch_size, max_length, num_pixels)
        predictions = torch.stack(predictions, dim=1)
        alphas_list = torch.stack(alphas_list, dim=1)

        return predictions, state

    def _forward_step(self, features, x, state=None):
        """
        Forward decoder single step.

            :param features: float tensor of shape (batch_size(1), num_pixels, encoder_size)
            :param x: long tensor of shape (1, ) containing the index of the predicted word.
            :param state: tuple (h, c).
                - h - float tensor of shape (batch_size(1), hidden_size)
                - c - float tensor of shape (batch_size(1), hidden_size)

        In prediction time, the model uses embedding of the previously predicted word and the last
        hidden state.
        """
        # ----------
        # Embeddings
        # ----------
        # captions -- (batch_size, embed_size)
        x = self.embedding(x)

        # -------------
        # Decoder steps
        # -------------
        context, alphas = self.attention(features, state[0])

        # ---------
        # LSTM step
        # ---------
        # hidden - (batch_size, hidden_size)
        # state -- h: (batch_size, num_layers, hidden_size)
        #          c: (batch_size, num_layers, hidden_size)
        i = torch.cat([x, context], dim=1)
        state = self.lstm_cell(i, state)
        hidden = state[0]

        # ----------
        # Vocabulary
        # ----------
        # out -- (batch_size, vocab_size)
        out = self.linear(hidden)

        # -------
        # Softmax
        # -------
        # Shape after softmax:
        #    output -- (batch_size(1), vocab_size)
        out = F.log_softmax(out, dim=1)

        return out, state, alphas

    def sample(self, features, max_seq_length):
        """
        Predicts a caption.

            :param features: float tensor of shape (batch_size(1), num_pixels, encoder_size)
            :param max_seq_len: maximum length of the predicted caption.

        In prediction time, the model uses embedding of the previously predicted word and the last
        hidden state.
        """
        output = []
        alphas_list = []

        last_idx_predicted = -1
        while len(output) < max_seq_length and last_idx_predicted != SpecialToken.END.value.index:

            if len(output) == 0:  # First iteration
                init_word = torch.tensor([SpecialToken.START.value.index],
                                         dtype=torch.int64,
                                         device=device)

                hidden = self.linear_features(features.view(features.size(0), -1))
                state = self.init_hidden(hidden)
                out, state, alphas = self._forward_step(features, init_word, state)

            else:
                out, state, alphas = self._forward_step(features, pred, state)

            # out shape -- (batch_size(1), vocab_size)
            _, pred = out.max(1)
            last_idx_predicted = pred.item()

            output.append(last_idx_predicted)
            alphas_list.append(alphas)

        return output, alphas_list