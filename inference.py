#!/usr/bin/env python

import argparse
import json
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

from custom_models.encoder import EncoderFactory
from custom_models.decoder import DecoderFactory

import numpy as np

# ==================================================================================================
# -- helpers ---------------------------------------------------------------------------------------
# ==================================================================================================

MAXIMUM_NCOL = 5
PIXELS = 224


def show_prediction(img, caption, alphas=None):
    """
    Shows image and the predicted caption using matplotlib.

        :param img: PIL imgae.
        :param caption: tokenized list
    """
    grid = plt.GridSpec(1 + math.ceil(len(caption) / MAXIMUM_NCOL), MAXIMUM_NCOL)

    if alphas is not None:
        main_axis = plt.subplot(grid[0, :])
        main_axis.imshow(img)
        main_axis.title.set_text(' '.join(caption))
        main_axis.axis('off')

        ncol = 0
        nrow = 1
        for index, word in enumerate(caption):
            word_axis = plt.subplot(grid[nrow, ncol])
            word_axis.imshow(img)
            word_axis.title.set_text(word)
            word_axis.axis('off')

            alpha = alphas[index]
            num_pixels = alpha.size(1) / 2.0

            alpha = alpha.view(1, num_pixels, num_pixels)
            alpha = alpha.cpu().detach().numpy()[0]
            mask = np.repeat(np.repeat(alpha, PIXELS / num_pixels, axis=0),
                             PIXELS / num_pixels,
                             axis=1)

            word_axis.imshow(mask, cmap='gray', alpha=0.6)

            ncol = ncol + 1 if ncol < MAXIMUM_NCOL - 1 else 0
            nrow = 1 + math.trunc((index + 1) / float(MAXIMUM_NCOL))

    else:
        main_axis = plt.subplot(grid[:, :])
        main_axis.imshow(img)
        main_axis.title.set_text(' '.join(caption))
        main_axis.axis('off')

    plt.show()


# ==================================================================================================
# -- inference -------------------------------------------------------------------------------------
# ==================================================================================================


def predict(img_path, model_path, args, data_root='data/flickr8k', max_seq_length=25):
    """
    Caption prediction.

        :param img_path: path of the image to predict.
    """
    # -----------
    # Data folder
    # -----------
    data_folder = dataset.Flickr8kFolder(data_root)

    # -------------------
    # Building vocabulary
    # -------------------
    logging.info('Building vocabulary...')
    vocab = vocabulary.build_flickr8k_vocabulary(data_folder.ann_file,
                                                 min_freq=args['vocab_min_freq'])
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

    encoder = EncoderFactory.get_encoder(args['encoder_type'], args['encoder_size'])
    decoder = DecoderFactory.get_decoder(args['attention_type'], args['embedding_size'], len(vocab),
                                         args['encoder_size'], encoder.num_pixels,
                                         args['hidden_size'], args['attention_size'])

    net = model.ImageCaptioningNet(encoder, decoder)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # -----------
    # Other utils
    # -----------
    idx2word_fn = utils.IdxToWord(vocab)

    # ----------
    # Prediction
    # ----------
    pil_image = PIL.Image.open(img_path).convert('RGB')
    image = transform(pil_image).to(device)

    caption, alphas = net.predict(image, max_seq_length)
    caption = idx2word_fn(caption)

    # Show result.
    plot_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize(256),
         torchvision.transforms.CenterCrop(224)])

    if alphas is not None:
        alphas = alphas[:-1]

    show_prediction(plot_transform(pil_image), caption, alphas)


# ==================================================================================================
# -- main ------------------------------------------------------------------------------------------
# ==================================================================================================

# Checking if GPU is available.
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('model_name', type=str, help='name of the model to be used')
    argparser.add_argument('image_path', metavar='PATH', type=str)

    # Data parameters.
    argparser.add_argument('--data-root',
                           metavar='PATH',
                           type=str,
                           default='data/flickr8k',
                           help='path for FLickr8k data (default: data/flickr8k)')
    argparser.add_argument('--max-seq-length',
                           type=int,
                           default=25,
                           help='maximum sequence length (default: 25)')
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')

    args = argparser.parse_args()
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Loading arguments model.
    model_path = os.path.join('models', args.model_name + '.pt')
    model_args_path = os.path.join('models', args.model_name + '.json')
    if not (os.path.exists(model_path) and os.path.exists(model_args_path)):
        raise RuntimeError('The provided model does not exist')

    with open(model_args_path, 'r') as f:
        model_args = json.load(f)

    predict(args.image_path, model_path, model_args, args.data_root, args.max_seq_length)