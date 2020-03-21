#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocabulary import Vocabulary, SpecialToken


class Encoder(nn.Module):
    def __init__(self, encoded_size):
        super(Encoder, self).__init__()

        self.encoded_size = encoded_size

        self.model = models.resnet101(pretrained=True)
        self.model.eval()  # Check (always eval)
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by
        # default
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.encoded_size)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size):
        super(Decoder, self).__init__()

        self._vocab_size = vocab_size

        self._embedding = nn.Embedding(vocab_size, embed_size)
        self._lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self._linear = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        pass

    #TODO: Random entre correcta y generada en el timestep anterior.
    #TODO: Lengths should be a tensor instead of python list.
    def forward(self, features, captions, lengths):
        """
        Forward decoder.

            :param features: shape [batch_size, hidden_size==encoder_size]
            :param captions: shape [batch_size, max_length]
            :param lengths: list with length of each caption.

        In training mode we use teacher forcing as we know the targets.
        """        
        # ----------
        # Embeddings
        # ----------
        # Word indices to embeddings representation. Captions shape after
        # embeddings:
        #    captions -- [batch_size, max_length, embed_size]
        captions = self._embedding(captions)
        
        # ----
        # LSTM
        # ----
        # In training mode we use teacher forcing as we know the targets.
        #
        # Unsqueeze first dimension features to apply num_layers = 1.
        # Shape after unsqueeze:
        #    features -- [num_layers(1), batch_size, hidden_size].
        # Shape after lstm:
        #    output -- [batch_size, max_length, hidden_size]
        #    state(h) -- [num_layers, batch_size, hidden_size]
        #    state(c) -- [num_layers, batch_size, hidden_size]
        features = features.unsqueeze(0)  # Adds num_layer dimension.

        # FIXME: Sort captions by length to avoid enforce_sorted=Fasle
        packed = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)
        # FIXME: How to init c? h=features, c=?
        hidden, state = self._lstm(packed, (torch.zeros(features.shape).to('cuda'), features))
        output, _ = pad_packed_sequence(hidden, batch_first=True)

        # -------------------
        # Output linear layer
        # -------------------
        # Shape after linear layer:
        #    [batch_size, max_length, vocab_size]
        output = self._linear(output)

        # -------
        # Softmax
        # -------
        # Shape after softmax
        #    [batch_size, max_length, vocab_size]
        #output = F.log_softmax(output, dim=2)

        return output, state

    def _forward_step(self, x, state=None):
        """
        Single forward step.
        
            :param x: input. Shape: [1] (word index)
            :param state: tuple (h, c).
                Shape: [num_layers(1), batch_size(1), hidden_size]
        """
        # ---------
        # Embedding
        # ---------
        # Shape after embedding:
        #    [batch_size(1), max_lenght(1), embed_size]
        x = self._embedding(x)
        x = x.unsqueeze(1)  # Unsqueeze to add max_length(1) dimension.

        # ----------------
        # LSTM single step
        # ----------------
        # Shape after lstm:
        #    hidden -- [batch_size(1), max_length(1), hidden_size]
        #    state -- [num_layers(1), batch_size(1), hidden_size]
        hidden, state = self._lstm(x, state)

        # -------------------
        # Output linear layer
        # -------------------
        # Shape after linear output:
        #    output -- [batch_size(1), max_length(1), vocab_size]
        output = self._linear(hidden)

        # -------
        # Softmax
        # -------
        # Shape after softmax:
        #    output -- [batch_size(1), max_length(1), vocab_size]
        output = F.log_softmax(output, dim=2)

        return output, state

    def sample(self, features, max_len):
        """
        Predicts a caption.

            :param features: shape [batch_size(1), hidden_size].
            :param max_len: maximum length of the predicted caption.

        In prediction time the model uses embedding of the previously predicted
        word and the last hidden state.
        """
        output = []

        i, index_pred = 0, None
        while i <= max_len and index_pred != 1: # END
            if i == 0:  # First iteration.
                start = torch.tensor([0], dtype=torch.int64, device='cuda')

                features = features.unsqueeze(0)  # Add num_layers dimension
                h, c = torch.zeros(features.shape), features
                h = h.to('cuda')
                c = c.to('cuda')

                out, state = self._forward_step(start, (h, c))

            else:
                out, state = self._forward_step(pred, state)

            _, pred = out.max(2)
            index_pred = pred[0][0]
            pred = pred.squeeze(1)
            output.append(pred)
            i += 1

        #print(output)
        output = torch.stack(output, 1).cpu().numpy().tolist()
 
        return output[0], state


class ImageCaptioningModel(nn.Module):

    def __init__(self, encoder, decoder, fine_tune=False):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fine_tune = fine_tune  # TODO(joel): Implement!

    def forward(self, imgs, train_caps, lengths):
        features = self.encoder(imgs)
        return self.decoder(features, train_caps, lengths)

    def sample(self, img):
        pass