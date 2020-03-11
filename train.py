#!/usr/bin/env python

import dataset
import model
import vocabulary

# ------------------------
# Defining hyperparameters
# ------------------------
hparams = {
    'img_root': '...',
    'ann_root': '...',
    'ann_file': '...',
    'batch_size':64,
    'num_workers': 2,
    'num_epochs':10,
    'hidden_size':128,
    'embedding_size': 600,
    'learning_rate':1e-3,
    'log_interval':100,
    'min_freq': 5,
    'max_length': 25,
    'device': 'cuda'
}

def train_loop():
    # -------------------
    # Building vocabulary
    # -------------------
    vocab = vocabulary.build_flickr8k_vocabulary(ann_file=hparams['ann_file'], min_freq=hparams['min_freq'])
    print('Size vocabulary: {}'.format(len(vocabulary)))

    # ---------
    # Transform
    # ---------
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
    flickr_trainset = dataset.FlickrDataset(
        hparams['img_root'],
        hparams['ann_root'],
        split=Split.TRAIN,
        transform=train_transform,
        target_transform=preprocessing.Word2Idx(vocab)
    )

    flickr_testset = custom_dataset.Flickr8k(
        hparams['img_root'],
        hparams['ann_root'],
        split=Split.TEST,
        transform=test_transform,
        target_transform=preprocessing.Word2Idx(vocab)
    )

    # -----------
    # Data loader
    # -----------
    train_loader = torch.utils.data.DataLoader(
        flickr_trainset,
        batch_size=hparams['batch_size'], 
        shuffle=True,
        num_workers=hparams['num_workers']
    )

    test_loader = torch.utils.data.DataLoader(
        flickr_testset,
        batch_size=hparams['test_batch_size'], 
        shuffle=False,
        num_workers=hparams['num_workers']
    )

    # -------------
    # Builing model
    # -------------
    encoder = model.Encoder(hparams['embed_size'])
    decoder = model.Decoder(hparams['embed_size'], len(vocab), hparams['hidden_size'])

    encoder.to(hparams['device'])
    decoder.to(hparams['device'])

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
