#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchvision

from vocabulary import SpecialToken

# ==================================================================================================
# -- encoder ---------------------------------------------------------------------------------------
# ==================================================================================================


class Encoder(nn.Module):
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
# -- decoder ---------------------------------------------------------------------------------------
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


# ==================================================================================================
# -- model -----------------------------------------------------------------------------------------
# ==================================================================================================


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, fine_tune=False):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fine_tune = fine_tune  # TODO(joel): Implement!

    def trainable_parameters(self):
        return self.encoder.trainable_parameters() + self.decoder.trainable_parameters()

    def forward(self, imgs, train_caps, lengths):
        features = self.encoder(imgs)
        out, state = self.decoder(features, train_caps, lengths)
        return out, state

    def predict(self, img, max_seq_length=25):
        """
        Predicts a caption.

            :param img: float tensor of shape (channels, height, width) # check!!!
            :param max_seq_len: maximum length of the predicted caption.

        In prediction time, the model uses embedding of the previously predicted word and the last
        hidden state.
        """
        output = []

        # -------
        # Encoder
        # -------
        # Shape and encoder:
        #    features -- (batch_size(1), num_pixels, encoder_size)
        img = img.unsqueeze(0)  #  Adding batch size dim: (batch_size(1), channels, height, width)
        features = self.encoder(img)

        # -------
        # Decoder
        # -------
        last_idx_predicted = -1
        while len(output) < max_seq_length and last_idx_predicted != SpecialToken.END.value.index:

            if len(output) == 0:  # First iteration
                init_word = torch.tensor([SpecialToken.START.value.index],
                                         dtype=torch.int64,
                                         device='cuda')
                
                out, state, alphas = self.decoder.forward_step(features, init_word)

            else:
                out, state, alphas = self.decoder.forward_step(features, pred, state)

            # out shape -- (batch_size(1), vocab_size)
            _, pred = out.max(1)
            print('Pred shape: {}'.format(pred.shape))
            #pred = pred.squeeze(1)
            last_idx_predicted = pred.item()
            print('Idx predices: {}'.format(last_idx_predicted))
            #last_idx_predicted = pred[0].item()

            output.append(last_idx_predicted)

        return output

    # def predict(self, img, max_seq_length=25):
    #     """
    #     Predicts a caption.

    #         :param img: float tensor of shape (batch_size, channels, height, width)
    #         :param max_seq_len: maximum length of the predicted caption.

    #     In prediction time, the model uses embedding of the previously predicted word and the last
    #     hidden state.
    #     """
    #     output = []

    #     # -------
    #     # Encoder
    #     # -------
    #     # Shape and encoder:
    #     #    features -- (batch_size(1), encoded_size==hidden_size)
    #     img = img.unsqueeze(0)  #  Adding batch size dim: (batch_size(1), channels, height, width)
    #     features = self.encoder(img)

    #     # -------
    #     # Decoder
    #     # -------
    #     last_idx_predicted = -1
    #     while len(output) < max_seq_length and last_idx_predicted != SpecialToken.END.value.index:

    #         if len(output) == 0:  # First iteration
    #             init_word = torch.tensor([SpecialToken.START.value.index],
    #                                      dtype=torch.int64,
    #                                      device='cuda')
    #             h, c = self.decoder.init_hidden(features)

    #             out, state = self.decoder.forward_step(init_word, (h, c))

    #         else:
    #             out, state = self.decoder.forward_step(pred, state)

    #         # out shape -- (batch_size(1), seq_length(1), vocab_size)
    #         _, pred = out.max(2)
    #         pred = pred.squeeze(1)
    #         last_idx_predicted = pred[0].item()

    #         output.append(last_idx_predicted)

    #     return output