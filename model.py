#!/usr/bin/env python

import torch.nn as nn

# ==================================================================================================
# -- model -----------------------------------------------------------------------------------------
# ==================================================================================================


class ImageCaptioningNet(nn.Module):
    def __init__(self, encoder, decoder):
        super(ImageCaptioningNet, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def trainable_parameters(self):
        return self.encoder.trainable_parameters() + self.decoder.trainable_parameters()

    def forward(self, imgs, train_caps, lengths):
        features = self.encoder(imgs)
        out, state = self.decoder(features, train_caps, lengths)
        return out, state

    def predict(self, img, max_seq_length=25):
        """
        Predicts a caption.

            :param img: float tensor of shape (channels, height, width)
            :param max_seq_len: maximum length of the predicted caption.
        """
        # -------
        # Encoder
        # -------
        # features shape -- (batch_size(1), num_pixels, encoder_size)
        img = img.unsqueeze(0)  #  Adding batch size dim: (batch_size(1), channels, height, width)
        features = self.encoder(img)

        # -------
        # Decoder
        # -------
        #  out -- (seq_length, )
        #  state -- (num_layers, hidden_size)
        #  alphas -- (batch_size, num_pixels)
        out, alphas = self.decoder.sample(features, max_seq_length)

        return out, alphas