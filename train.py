#!/usr/bin/env python

import logging

import torch
import torch.nn as nn
import torch.optim as optim
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
    'data_root':
    '/home/jmoriana/workspace/aidl/aidl-image-captioning/data/flickr8k',
    'batch_size': 10,
    'num_workers': 1,
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
        num_workers=hparams['num_workers'],
        collate_fn=dataset.flickr_collate_fn)

    test_loader = torch.utils.data.DataLoader(
        flickr_testset,
        batch_size=hparams['batch_size'],
        shuffle=False,
        num_workers=hparams['num_workers'],
        collate_fn=dataset.flickr_collate_fn)

    # -------------
    # Builing model
    # -------------
    encoder = model.Encoder(hparams['hidden_size'])
    decoder = model.Decoder(hparams['embedding_size'], len(vocab),
                            hparams['hidden_size'])

    encoder.to(hparams['device'])
    decoder.to(hparams['device'])

    # ------------------
    # Loss and optimizer
    # ------------------
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final encoder layer are being optimized.
    params = list(encoder.model.fc.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=hparams['learning_rate'])

    # -------------
    # Training loop
    # -------------
    # Activate the train=True flag inside the model. some layers have different
    # behavior during train/and evaluation (like BatchNorm, Dropout) so setting
    # it matters.
    encoder.train()
    decoder.train()

    # Test overfitting
    # img = flickr_trainset[0][0].to('cuda').unsqueeze(0)  # Adding bacth_size
    # features = encoder(img)
    # output, _ = decoder.sample(features, 25)
    # #idx_to_word = preprocessing.IdxToWord(vocab)
    # #print(idx_to_word(output))
    # #return

    # For each batch
    for epoch in range(hparams['num_epochs']):
        for i, (data, target, lengths) in enumerate(train_loader):
            
            img = data.to(hparams['device'])          # [batch_size, channel, w, h]
            captions = target.to(hparams['device'])   # [batch_size, max_lenght]

            print('Shape captions: {}'.format(captions.shape))
            print(captions[0])

            # 0) Clear gradients
            optimizer.zero_grad()

            # 1) Forward the data through the network
            features = encoder(img)                  # [batch_size, hidden_size]
            out, _ = decoder(features, captions, lengths)  # --> Quitar end al captions!!!!!!

            print('Out shape: {}'.format(out.shape))

            # 2) Compute loss
            # Nuestro modelo no tiene que predecir start.
            # input con start sin end (training).
            # loss sin start y con end.

            out = out.view(-1, len(vocab))
            target = captions[:][1:]
            print("Target shape: {}".format(target.shape))
            print("Target: {}".format(target[0]))
            return
            target = captions.view(-1)  # Quitar el start al target!!!!!!

            print('out after view: {}'.format(out.shape))
            print('target after view: {}'.format(target.shape))
            loss = criterion(out, target)

            # 3) Backprop with repsect to the loss function
            loss.backward()

            # 4) Apply the optimizer with a learning step
            optimizer.step()

            print("-----------------------------")
            print('Batch idx: {}'.format(i))
            print("Loss: {}".format(loss))

            return

            # Print loss and accuracy
            if i == 1000:
                break

    # Test overfitting
    img = flickr_trainset[0][0].to('cuda').unsqueeze(0)
    features = encoder(img)
    print('After encoder: {}'.format(features.shape))
    output, _ = decoder.sample(features, 25)
    idx_to_word = preprocessing.IdxToWord(vocab)
    print(idx_to_word(output))


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    train_loop()

    # Sampling
    # #########
    # img = flickr_testset[0][0].to('cuda').unsqueeze(0)
    # features = encoder(img)
    # print('After encoder: {}'.format(features.shape))
    # output, _ = decoder.sample(features, 25)
    # idx_to_word = preprocessing.IdxToWord(vocab)
    # print(idx_to_word(output))

    # return
    #########
