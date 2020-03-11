#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from vocabulary import Vocabulary, SpecialToken


class Encoder(nn.Module):
    
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self._encoder_model = models.resnet101(pretrained=True)
        self._encoder_model.fc = nn.Linear(2048, self.embed_size)

    def forward(self, x):
        # User iteration?
        return self._encoder_model(x)


class Decoder(nn.Model):

    def __init__(self, embed_size, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self._linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        pass

    def forward(self, features, captions, lengths):
        """
        In training mode we use teacher forcing as we know the targets. In prediction
        time the model uses embedding of the previously predicted word and the last
        hidden state.
        # Random entre correcta y generada en el timestep anterior.
        """
        # Caption shape: [batch size, sequence_length, vocab_size]

        # 1. Word index to embedding
        # Entrar feautures como contexto.
        captions = self._embedding(captions)

        # 2. Concatenate captions and fetures, as the features is the first input to our rnn.
        # x = torch.cat(features, captions)

        # 3. LSTM
        #pack_padded_sequence
        hidden, state = self._lstm(captions, features)
        #pad_packed_sequence

        # 4. out
        out = self._linear(hidden)
        out = F.log_softmax(out, dim=-1)
        return out, state

    def _forward_step(self, x, state=None):
        x = self._embedding(x)
        hidden, state = self._lstm(x, state)
        out = self._linear(hidden)
        return F.log_softmax(out, dim=-1), state

    def sample(self, features, max_len):
        output = []  # TODO(joel): pytorch tensor.

        i, pred = 0, None
        while i <= max_len and pred != SpecialToken.END.value:
            if i == 0:  # First iteration.
                pred, state = self._forward_step("<start>", features)
            else:
                pred, state = self._forward_step(pred, state)

            output.append(pred)
            i += 1

        return output