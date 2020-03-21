#!/usr/bin/env python

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import torchvision
from torch.utils.tensorboard import SummaryWriter

from torchtext.data.metrics import bleu_score

import dataset
import model
import preprocessing
import util
import vocabulary

# ==================================================================================================
# -- Hyperparameters -------------------------------------------------------------------------------
# ==================================================================================================

hparams = {
    'data_root':
    '/home/jmoriana/workspace/aidl/aidl-image-captioning/data/flickr8k',
    'batch_size': 10,
    'num_workers': 1,
    'num_epochs': 100,
    'hidden_size': 128,
    'embedding_size': 600,
    'learning_rate': 1e-3,
    'log_interval': 100,
    'min_freq': 1,
    'max_length': 25,
    'device': 'cuda'
}


# ==================================================================================================
# -- Training class --------------------------------------------------------------------------------
# ==================================================================================================


class Train(object):

    def __init__(self, hparams):
        self.hparams = hparams

        # -----------
        # Data folder
        # -----------
        self.data_folder = util.Flickr8kFolder(self.hparams['data_root'])

        # -------------------
        # Building vocabulary
        # -------------------
        logging.info('Building vocabulary...')
        self.vocab = vocabulary.build_flickr8k_vocabulary(self.data_folder.ann_file,
                                                          min_freq=self.hparams['min_freq'])
        logging.debug('Vocabulary size: {}'.format(len(self.vocab)))

        # ----------
        # Transforms
        # ----------
        logging.info('Building transforms...')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        test_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # --------
        # Datasets
        # --------
        logging.info('Building datasets...')
        self.flickr_trainset = dataset.Flickr8kDataset(
            self.data_folder,
            split='train',
            transform=train_transforms,
            target_transform=preprocessing.Word2Idx(self.vocab))

        self.flickr_testset = dataset.Flickr8kDataset(
            self.data_folder,
            split='test',
            transform=test_transforms,
            target_transform=preprocessing.Word2Idx(self.vocab))

        # -----------
        # Data loader
        # -----------
        logging.info('Building data loader...')
        self.train_loader = torch.utils.data.DataLoader(
            self.flickr_trainset,
            batch_size=self.hparams['batch_size'],
            shuffle=True,
            num_workers=self.hparams['num_workers'],
            collate_fn=dataset.flickr_collate_fn)

        self.test_loader = torch.utils.data.DataLoader(
            self.flickr_testset,
            batch_size=self.hparams['batch_size'],
            shuffle=False,
            num_workers=self.hparams['num_workers'],
            collate_fn=dataset.flickr_collate_fn)

        # -------------
        # Builing model
        # -------------
        logging.info('Builing model...')
        self.encoder = model.Encoder(self.hparams['hidden_size'])
        self.decoder = model.Decoder(self.hparams['embedding_size'], len(self.vocab), self.hparams['hidden_size'])
        self.model = model.ImageCaptioningModel(self.encoder, self.decoder)

        self.encoder.to(self.hparams['device'])
        self.decoder.to(self.hparams['device'])
        self.model.to(self.hparams['device'])

        # ------------------
        # Loss and optimizer
        # ------------------
        self.criterion = nn.CrossEntropyLoss()

        # Observe that only parameters of final encoder layer are being optimized.
        params = list(self.encoder.model.fc.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=self.hparams['learning_rate'])

        # -----------
        # Other utils
        # -----------
        self.idx2word_fn = preprocessing.IdxToWord(self.vocab)

    def compute_accuracy(self, predicted, target):
        """
        Computes accuracy based on BLEU.
        
            :param predicted: Predicted captions. Shape: [batch_size, max_length]
            :param target: Target captions. Shape: [batch_size, max_length]
        """
        total_bleu = 0
        for predicted_cap, target_cap in zip(predicted, target):
            predicted_cap = self.idx2word_fn(predicted_cap.tolist())
            target_cap = self.idx2word_fn(target_cap.tolist())

            bleu = bleu_score([predicted_cap], [[target_cap]])
            total_bleu += bleu
        return total_bleu / self.hparams['batch_size']

    def train_epoch(self):
        """
        Training epoch.

            : return: loss and accuracy
        """
        # Activate the train=True flag inside the model. some layers have
        # different behavior during train and evaluation (like BatchNorm,
        # Dropout) so setting it matters.
        self.encoder.train()
        self.decoder.train()

        total_loss = 0
        total_accuracy = 0
        for i, (data, train_caps, loss_caps, lengths) in enumerate(self.train_loader):

            img = data.to(self.hparams['device'])                  # [batch_size, channel, w, h]
            train_caps = train_caps.to(self.hparams['device'])     # [batch_size, max_lenght]
            loss_caps = loss_caps.to(self.hparams['device'])       # [batch_size, max_length]

            # 0. Clear gradients.
            self.optimizer.zero_grad()

            # 1. Forward the data through the network.
            #features = self.encoder(img)                           # [batch_size, hidden_size]
            #out, _ = self.decoder(features, train_caps, lengths)
            out, _ = self.model(img, train_caps, lengths)

            # 2. Compute loss.
            loss = self.criterion(out.view(-1, len(self.vocab)), loss_caps.view(-1))

            # 3) Backprop with repsect to the loss function.
            loss.backward()

            # 4) Apply the optimizer with a learning step.
            self.optimizer.step()

            # 5. Computing loss and accuracy.
            _, predicted_caps = out.max(2)                        # [batch_size, max_length]
            total_loss += loss.item()
            total_accuracy += self.compute_accuracy(train_caps, out.max(2)[1])

        return total_loss / len(self.train_loader), total_accuracy / (len(self.train_loader))

    def validate_epoch(self):
        # Activate the train=True flag inside the model. some layers have
        # different behavior during train and evaluation (like BatchNorm,
        # Dropout) so setting it matters.
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0
        total_accuracy = 0
        for i, (data, train_caps, loss_caps, lengths) in enumerate(self.test_loader):

            img = data.to(self.hparams['device'])                  # [batch_size, channel, w, h]
            train_caps = train_caps.to(self.hparams['device'])     # [batch_size, max_lenght]
            loss_caps = loss_caps.to(self.hparams['device'])       # [batch_size, max_length]

            # 1. Forward the data through the network.
            #features = self.encoder(img)                           # [batch_size, hidden_size]
            #out, _ = self.decoder(features, train_caps, lengths)
            out, _ = self.model(img, train_caps, lengths)

            # 2. Compute loss.
            loss = self.criterion(out.view(-1, len(self.vocab)), loss_caps.view(-1))

            # 3. Computing loss and accuracy.
            total_loss += loss.item()
            total_accuracy = 0.0  # TODO(joel): Compute BLEU

        return total_loss / len(self.test_loader), total_accuracy / (len(self.test_loader))

    def train(self):
        """
        Training loop

        """
        # Starting tensorboard writer.
        self.writer = SummaryWriter()  # TODO(joel): Add this to ctor?

        for epoch in range(self.hparams['num_epochs']):
            
            train_loss, train_acc = self.train_epoch()
            eval_loss, eval_acc = self.validate_epoch()

            if epoch == 0:
                dummy_imgs = torch.randn(32, 3, 224, 224, dtype=torch.float32).to('cuda')
                dummy_caps = torch.randint(low=0, high=8921, size=(32, 25), dtype=torch.int64).to('cuda')
                dummy_lengths = torch.randint(low=1, high=25, size=(32, ), dtype=torch.int64).to('cuda')

                self.writer.add_graph(self.model, (dummy_imgs, dummy_caps, dummy_lengths))

            # Writing scalars.
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', eval_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/test', eval_acc, epoch)

            logging.info(
                '''***** Epoch {epoch:} *****:
                \t[TRAIN] Loss: {train_loss:} | Acc: {train_acc:}
                \t[EVAL] Loss: {eval_loss:} | Acc: {eval_acc:}'''.format(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                eval_loss=eval_loss,
                eval_acc=eval_acc))
 
        # Writing hparams.
        self.writer.add_hparams(self.hparams, {
                'hparam/train-loss': train_loss,
                'hparam/train-accuracy': train_acc,
                'hparam/eval-loss': eval_loss,
                'hparam/eval-train': eval_acc
            }
        )

        # Writing embeddings.
        logging.debug('Adding embeddings...')
        self.writer.add_embedding(self.decoder._embedding.weight, metadata=self.vocab.get_words(), global_step=0)

        # Closing tensorboard writer
        self.writer.close()

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    
    trainer = Train(hparams)
    trainer.train()
