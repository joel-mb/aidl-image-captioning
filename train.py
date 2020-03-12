#!/usr/bin/env python

import logging

import torch
from torchvision import transforms

import dataset
import model
import preprocessing
import util
import vocabulary

# ------------------------
# Defining hyperparameters
# ------------------------
hparams = {
    'data_root': '/home/jmoriana/workspace/aidl/aidl-image-captioning/data/flickr8k',
    'batch_size': 32,
    'num_workers': 2,
    'num_epochs': 10,
    'hidden_size': 128,
    'embedding_size': 600,
    'learning_rate': 1e-3,
    'log_interval': 100,
    'min_freq': 1,
    'max_length': 25,
    'device': 'cuda'
}


def train_loop():
    # -----------
    # Data folder
    # -----------
    data_folder = util.Flickr8kFolder(hparams['data_root'])

    # -------------------
    # Building vocabulary
    # -------------------
    logging.info('Building vocabulary')
    vocab = vocabulary.build_flickr8k_vocabulary(data_folder.ann_file,
                                                 min_freq=hparams['min_freq'])
    logging.info('Vocabulary size: {}'.format(len(vocab)))

    # ---------
    # Transform
    # ---------
    logging.info('Building transforms')
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        preprocessing.NormalizeImageNet()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        preprocessing.NormalizeImageNet()
    ])

    # --------
    # Datasets
    # --------
    logging.info('Building datasets')
    flickr_trainset = dataset.Flickr8kDataset(
        data_folder,
        split='train',
        transform=train_transforms,
        target_transform=preprocessing.Word2Idx(vocab))

    flickr_testset = dataset.Flickr8kDataset(
        data_folder,
        split='test',
        transform=test_transforms,
        target_transform=preprocessing.Word2Idx(vocab))

    # -----------
    # Data loader
    # -----------
    logging.info('Logging data loader')
    train_loader = torch.utils.data.DataLoader(
        flickr_trainset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        num_workers=hparams['num_workers'])

    test_loader = torch.utils.data.DataLoader(
        flickr_testset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=hparams['num_workers'])

    # -------------
    # Builing model
    # -------------
    encoder = model.Encoder(hparams['hidden_size'])
    decoder = model.Decoder(hparams['embedding_size'], len(vocab),
                            hparams['hidden_size'])

    encoder.to(hparams['device'])
    decoder.to(hparams['device'])

    #########
    img = flickr_testset[0][0].to('cuda').unsqueeze(0)
    features = encoder(img)
    print('After encoder: {}'.format(features.shape))
    output, _ = decoder.sample(features, 25)
    idx_to_word = preprocessing.IdxToWord(vocab)
    print(idx_to_word(output))

    return
    #########

    # ------------------
    # Loss and optimizer
    # ------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = ...

    # -------------
    # Training loop
    # -------------

    # Activate the train=True flag inside the model
    encoder.train()
    decoder.train()

    #TODO(joel): with no_grad for convolutional.
    #TODO(joel): Pad captions?

    # For each batch
    for batch_idx, (data, target) in enumerate(train_loader):
        img = data.to(hparams['device'])
        captions = data.to(hparams['device'])

        # Padding? --> hacerla a mano o con pytorch.

        # 0) Clear gradients
        optimizer.zero_grad()

        # 1) Forward the data through the network
        features = encoder(img)
        out = decoder(features, captions)

        # 2) Compute the loss
        loss = criterion(out, captions)

        # 3) Backprop with repsect to the loss function
        loss.backward()

        # 4) Apply the optimizer with a learning step
        optimizer.step()

        # Print loss and accuracy

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    train_loop()