#!/usr/bin/env python

from torch.utils.data import Dataset

from vocabulary import Vocabulary

# ------------------------
# Defining hyperparameters
# ------------------------
hparams = {
    'batch_size':64,
    'num_workers': 2,
    'num_epochs':10,
    'hidden_size':128,
    'embedding_size': 600l
    'learning_rate':1e-3,
    'log_interval':100,
    'min_freq': 5,
    'max_length': 25
    'device': 'cuda'
}

def train_loop():
    # -------------------
    # Building vocabulary
    # -------------------
    vocabulary = Vocabulary(ann_file=..., min_freq=hparams['min_freq'])
    print('Size vocabulary: {}'.format(len(vocabulary)))

    # ---------
    # Transform
    # ---------
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # TODO(joel): Create custom transform for vocabulary (captions).

    # --------
    # Datasets
    # --------
    flickr_trainset = dataset.Flickr8k(
        img_root,
        ann_root,
        train=True # TODO(joel): Alomejor añadir tipo (TRAIN, TEST, EVAL)
        transform=train_transform,
        target_transform=...
    )

    flickr_testset = dataset.Flickr8k(
        img_root,
        ann_root,
        train=False
        transform=test_transform,
        target_transform=...
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
    encoder = Encoder(...)
    decoder = Decoder(...)

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

    # TODO(joel): Pad captions?
    # TODO(joel): with no_grad para no hacer fine tunning de la convolucional.

    # For each batch
    for batch_idx, (data, target) in enumerate(train_loader):
        img, caption = data.to(device), target.to(device)

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
