#!/usr/bin/env python

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabulary import SpecialToken
from .attention import AdditiveAttention

# ==================================================================================================
# -- baseline decoder ------------------------------------------------------------------------------
# ==================================================================================================


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        self._vocab_size = vocab_size

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

        h, c = torch.zeros(features.shape).to('cuda'), features
        return h, c

    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: float tensor of shape (batch_size, hidden_size==encoder_size)
            :param captions: long tensor of shape (batch_size, max_seq_length)
            :param lengths: long tensor of shape (batch_size, ) containing original length for each
                            caption.

        In training mode we use teacher forcing as we know the targets.
        """
        # ----------
        # Embeddings
        # ----------
        # Word indices to embeddings representation. Captions shape after
        # embeddings:
        #    captions -- (batch_size, max_length, embed_size)
        captions = self.embedding(captions)

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

    def forward_step(self, x, state=None):
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

        return output, state

    def sample(self, features):
        out = 0
        state = 0
        alphas = 0
        return out, alphas


# ==================================================================================================
# -- attention decoder -----------------------------------------------------------------------------
# ==================================================================================================


class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_size, hidden_size, attention_size):
        super(DecoderWithAttention, self).__init__()

        self.hidden_size = hidden_size

        self.attention = AdditiveAttention(encoder_size, hidden_size, attention_size)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def trainable_parameters(self):
        return list(self.parameters())

    def init_hidden(self, features):
        # Unsqueeze first dimension of features to apply num_layers = 1.
        # Shape after unsqueeze:
        #    features -- (num_layers(1), batch_size, hidden_size).
        # FIXME: LSTMcell no num_layers!!!!!!!!!!!!!
        h, c = torch.zeros(features.shape).to('cuda'), features
        return h, c

    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: float tensor of shape (batch_size, hidden_size==encoder_size)
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

        #
        #
        #
        # hidden -- (batch_size, hidden_size)
        # fc1 = nn.Linear(encoder_size, hidden_size)ç
        # use F.linear instead!!!
        flat_features = features.view(features.size(0), -1)  # (batch_size, 7*7*encoder_size)
        fc1 = nn.Linear(flat_features.size(1), self.hidden_size).to('cuda')
        hidden = fc1(flat_features)
        #print('First hidden shape: {}'.format(hidden.shape))

        #
        # initial state
        #
        state = self.init_hidden(hidden)
        #print('State shape: h->{}, c->{}'.format(state[0].shape, state[1].shape))

        # -------------
        # Decoder steps
        # -------------
        # Using teacher forcing
        # FIXME: What happends withs PAD????
        for i in range(captions.size(1)):
            #print('===== IRETATION {} ====='.format(i + 1))
            embeddings = captions[:, i]
            #print('Embedding_size: {}'.format(embeddings.shape))

            # ---------
            # Attention
            # ---------
            # alphas -- (batch_size, num_pixels)
            # context -- (batch_size, encoder_size)
            context, alphas = self.attention(features, hidden)
            #print('Attention--> Context: {}, alphas: {}'.format(context.shape, alphas.shape))

            # ---------
            # LSTM step
            # ---------
            # hidden - (batch_size, hidden_size)
            # state -- h: (batch_size, num_layers, hidden_size)
            #          c: (batch_size, num_layers, hidden_size)
            i = torch.cat([embeddings, context], dim=1)
            #print('Input lstm cell: {}'.format(i.shape))
            h, c = self.lstm_cell(i, state)
            hidden = h
            state = (h, c)
            #print('Output lstm cell --> hidden: {}, h, c: {}'.format(hidden.shape, state[0].shape))

            # ----------
            # Vocabulary
            # ----------
            # out -- (batch_size, vocab_size)
            out = self.linear(hidden)
            #print('Out shape: {}'.format(out.shape))

            # Store predictions and alphas.
            predictions.append(out)
            alphas_list.append(alphas)

        # predictions -- shape (batch_size, max_length, vocab_size)
        # alphas -- shape (batch_size, max_length, num_pixels)
        predictions = torch.stack(predictions, dim=1)
        alphas_list = torch.stack(alphas_list, dim=1)
        return predictions, alphas_list

    def _forward_step(self, features, x, state=None):
        """
        Forward decoder single step.

        features -- (batch_size(1), num_pixels(49), encoder_size(512)) -- encoder out
        x shape (1, )  # last prediction word.
        state: h -- (batch_size, hidden_size)
               c -- (batch_size, hidden_size)

        In training mode we use teacher forcing as we know the targets.
        """
        print(' ===== Prediction step =====')
        # ----------
        # Embeddings
        # ----------
        # captions -- (batch_size, embed_size)
        x = self.embedding(x)
        print('Caption shape {}'.format(x.shape))

        #
        # initial state
        #
        if state is None:
            flat_features = features.view(features.size(0), -1)  # (batch_size, 7*7*encoder_size)
            fc1 = nn.Linear(flat_features.size(1), self.hidden_size).to('cuda')
            hidden = fc1(flat_features)

            state = self.init_hidden(hidden)
        else:
            hidden = state[0]

        print('Shape hidden: {}'.format(hidden.shape))
        print('Shape state: {}'.format(state[1].shape))

        # -------------
        # Decoder steps
        # -------------
        context, alphas = self.attention(features, hidden)
        print('Attention--> Context: {}, alphas: {}'.format(context.shape, alphas.shape))

        # ---------
        # LSTM step
        # ---------
        # hidden - (batch_size, hidden_size)
        # state -- h: (batch_size, num_layers, hidden_size)
        #          c: (batch_size, num_layers, hidden_size)
        i = torch.cat([x, context], dim=1)
        print('Input lstm cell: {}'.format(i.shape))
        h, c = self.lstm_cell(i, state)
        hidden = h
        state = (h, c)
        print('Output lstm cell --> hidden: {}, h, c: {}'.format(hidden.shape, state[0].shape))

        # ----------
        # Vocabulary
        # ----------
        # out -- (batch_size, vocab_size)
        out = self.linear(hidden)
        print('Out shape: {}'.format(out.shape))

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

            :param img: float tensor of shape (channels, height, width) # check!!!
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
                                         device='cuda')

                out, state, alphas = self._forward_step(features, init_word)

            else:
                out, state, alphas = self._forward_step(features, pred, state)

            # out shape -- (batch_size(1), vocab_size)
            _, pred = out.max(1)
            print('Pred shape: {}'.format(pred.shape))
            #pred = pred.squeeze(1)
            last_idx_predicted = pred.item()
            print('Idx predices: {}'.format(last_idx_predicted))
            #last_idx_predicted = pred[0].item()

            output.append(last_idx_predicted)
            alphas_list.append(alphas)

        return output, alphas_list