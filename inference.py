#!/usr/bin/env python

import argparse
import logging
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

# ==================================================================================================
# -- hyperparameters -------------------------------------------------------------------------------
# ==================================================================================================

hparams = {
    'data_root': '/home/jmoriana/workspace/aidl/aidl-image-captioning/data/flickr8k',
    'batch_size': 10,
    'num_workers': 1,
    'num_epochs': 100,
    'hidden_size': 128,
    'embedding_size': 600,
    'learning_rate': 1e-3,
    'log_interval': 100,
    'min_freq': 1,
    'max_seq_length': 25,
    'device': 'cuda'
}

# ==================================================================================================
# -- helpers ---------------------------------------------------------------------------------------
# ==================================================================================================


def show_prediction(img, caption):
    """
    Shows image and the predicted caption using matplotlib.

        :param img: PIL imgae.
        :param caption: tokenized list
    """
    fig = px.imshow(img)

    fig.update_layout(
        title_text=' '.join(caption)
    )
    fig.show()


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
    
    encoder_size = 512
    encoder = Encoder(encoder_size)
    decoder = DecoderWithAttention(hparams['embedding_size'], len(vocab), encoder_size, hparams['hidden_size'], 512)
    ic_model = model.ImageCaptioningModel(encoder, decoder)

    ic_model.to(hparams['device'])
    ic_model.load_state_dict(torch.load('models/model.pt'))

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

    caption = ic_model.predict(image, hparams['max_seq_length'])
    caption = idx2word_fn(caption)

    show_prediction(pil_image, caption)

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