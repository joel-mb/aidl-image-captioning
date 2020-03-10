#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Encoder(nn.Module):
    
    def __init__(self, embed_size):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self._encoder_model = models.resnet101(pretrained=True)
        self._encoder_model.fc = nn.Linear(2048, self.embed_size)

    def forward(self, x):
        return self._encoder_model(x)


class Decoder(nn.Model):

    def __init__(self, embed_size, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()

        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self._linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        pass

    def forward(self, features, captions):
        """
        In function forward there's a for-loop that computes the decoder hidden states one time step
        at a time. In training mode we use teacher forcing as we know the targets. In prediction
        time the model uses embedding of the previously predicted word and the last hidden state.
        """
        # Caption shape: [batch size, sequence_length, ...]

        # 1. Word index to embedding
        captions = self._embedding(captions)

        # 2. Concatenate captions and fetures, as the features is the first input to our rnn.
        input = torch.cat(featues, captions)

        # 3. LSTM
        #pack_padded_sequence
        hidden, state = self._lstm(input)
        #pad_packed_sequence

        # 4. ou
        out = self._linear(hidden)
        out = F.log_softmax(out, dim=-1)
        return out, state

    def _forward_step(self, x, state=None):
        x = self._embedding(x)
        hidden, state = self._lstm(x, state)
        out = self._linear(hidden)
        return F.log_softmax(out, dim=-1), state

    def sample(self, features, max_len):
        output = []
        while i <= max_len and pred != '<END>':
            if i == 0:
                # First iteration
                pred, state = self._forward_step(features)
            else:
                pred, state = self._forward_step(pred, state)

            output.append(pred)
            i += 1
        return output


if __name__ == '__main__':
    print("Taking inception model!!!")
    model = models.resnet101(pretrained=False)
    model = model[0:-1]
    print(model)