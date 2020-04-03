#!/usr/bin/env python

import argparse
import logging
import math
import os

import torch
import torchvision

import matplotlib.pyplot as plt
import plotly.express as px
import PIL

import model
import utils
import vocabulary
import dataset

from custom_models.encoder import Encoder
from custom_models.decoder import DecoderWithAttention

import numpy as np

# ==================================================================================================
# -- hyperparameters -------------------------------------------------------------------------------
# ==================================================================================================

hparams = {
    'data_root': '/home/jmoriana/workspace/aidl/aidl-image-captioning/data/flickr8k',
    'batch_size': 10,
    'num_workers': 1,
    'num_epochs': 100,
    'encoder_size': 256,
    'hidden_size': 128,
    'embedding_size': 256,
    'attention_size': 256,
    'learning_rate': 1e-3,
    'log_interval': 100,
    'min_freq': 1,
    'max_seq_length': 25,
    'device': 'cuda'
}

# ==================================================================================================
# -- helpers ---------------------------------------------------------------------------------------
# ==================================================================================================


def show_prediction(img, caption, alphas=None):
    """
    Shows image and the predicted caption using matplotlib.

        :param img: PIL imgae.
        :param caption: tokenized list
    """
    grid = plt.GridSpec(1 + math.ceil(len(caption) / 5), 5, wspace=0.4, hspace=0.3)

    main_axis = plt.subplot(grid[0, :])
    main_axis.imshow(img)
    main_axis.title.set_text(' '.join(caption))
    main_axis.axis('off')

    ncol = 0
    nrow = 1
    for index, word in enumerate(caption):

        print(nrow, ncol)
        word_axis = plt.subplot(grid[nrow, ncol])
        word_axis.imshow(img)
        word_axis.title.set_text(word)
        word_axis.axis('off')

        alpha = alphas[index].view(1, 7, 7)
        alpha = alpha.cpu().detach().numpy()[0]
        print(alpha)
        print('Alpha size: '.format(alpha.size))
        #mask = np.repeat(alpha, 32, axis=1)
        mask = np.repeat(np.repeat(alpha, 32, axis=0), 32, axis=1)
        print('Mask size: {}'.format(mask.size))
        word_axis.imshow(mask, alpha=0.6)

        ncol = ncol + 1 if ncol < 4 else 0
        nrow = 1 + math.trunc((index + 1) / 5.0)

    plt.show()


# ==================================================================================================
# -- inference -------------------------------------------------------------------------------------
# ==================================================================================================


def predict(img_path):
    """
    Caption prediction.

        :param img_path: path of the image to predict.
    """
    # -----------
    # Data folder
    # -----------
    data_folder = dataset.Flickr8kFolder(hparams['data_root'])

    # -------------------
    # Building vocabulary
    # -------------------
    logging.info('Building vocabulary...')
    vocab = vocabulary.build_flickr8k_vocabulary(data_folder.ann_file, min_freq=hparams['min_freq'])
    logging.debug('Vocabulary size: {}'.format(len(vocab)))

    # ---------
    # Transform
    # ---------
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # -------------
    # Builing model
    # -------------
    logging.info('Building model...')

    encoder = Encoder(hparams['encoder_size'])
    decoder = DecoderWithAttention(hparams['embedding_size'], len(vocab), hparams['encoder_size'],
                                   hparams['hidden_size'], hparams['attention_size'])
    net = model.ImageCaptioningNet(encoder, decoder)

    net.to(hparams['device'])
    net.load_state_dict(torch.load('models/model.pt'))

    encoder.eval()
    decoder.eval()

    # -----------
    # Other utils
    # -----------
    idx2word_fn = utils.IdxToWord(vocab)

    # ----------
    # Prediction
    # ----------
    pil_image = PIL.Image.open(img_path).convert('RGB')
    image = transform(pil_image).to('cuda')

    caption, alphas = net.predict(image, hparams['max_seq_length'])
    caption = idx2word_fn(caption)

    print(image.cpu().numpy().size)
    print('Alphas: {}'.format(alphas[0].cpu().detach().numpy()))

    #    print('Caption: {}'.format(caption))
    #    print('Len caption: {}'.format(len(caption)))
    #    print('Len alphas: {}'.format(len(alphas)))
    #    for word, alpha in zip(caption, alphas[:-1]):
    #        alpha = alpha.view(1, 7, 7)
    #        print('\nToken: {}'.format(word))
    #        print(alpha)
    #        print('Sum: {}'.format(alpha.sum()))

    show_prediction(image.permute(1, 2, 0).cpu().numpy(), caption, alphas[1:])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--image-path', '-i', metavar='PATH', type=str)
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')
    args = argparser.parse_args()

    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    predict(args.image_path)