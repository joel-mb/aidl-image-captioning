#!/usr/bin/env python
""" Trainning utils."""

import argparse
import json
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchtext

import dataset
import model
import utils
import vocabulary

from custom_models.encoder import EncoderFactory
from custom_models.decoder import DecoderFactory

# ==================================================================================================
# -- training  -------------------------------------------------------------------------------------
# ==================================================================================================


class Train(object):
    def __init__(self, args):
        self.args = args

        # -----------
        # Data folder
        # -----------
        data_folder = dataset.Flickr8kFolder(args.data_root)

        # -------------------
        # Building vocabulary
        # -------------------
        logging.info('Building vocabulary...')
        self.vocab = vocabulary.build_flickr8k_vocabulary(data_folder.ann_file,
                                                          min_freq=args.vocab_min_freq)
        logging.debug('Vocabulary size: {}'.format(len(self.vocab)))

        # ----------
        # Transforms
        # ----------
        logging.info('Building transforms...')
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        val_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # --------
        # Datasets
        # --------
        logging.info('Building datasets...')
        flickr_trainset = dataset.Flickr8kDataset(data_folder,
                                                  split='train',
                                                  transform=train_transforms,
                                                  target_transform=utils.Word2Idx(self.vocab))

        flickr_valset = dataset.Flickr8kDataset(data_folder,
                                                split='eval',
                                                transform=val_transforms,
                                                target_transform=utils.Word2Idx(self.vocab))

        # -----------
        # Data loader
        # -----------
        logging.info('Building data loader...')
        self.train_loader = torch.utils.data.DataLoader(flickr_trainset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        collate_fn=dataset.flickr_collate_fn)

        self.val_loader = torch.utils.data.DataLoader(flickr_valset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers,
                                                      collate_fn=dataset.flickr_collate_fn)

        # -------------
        # Builing model
        # -------------
        logging.info('Building model...')
        encoder = EncoderFactory.get_encoder(args.encoder_type, args.encoder_size)
        decoder = DecoderFactory.get_decoder(args.attention_type, args.embedding_size,
                                             len(self.vocab), args.encoder_size, encoder.num_pixels,
                                             args.hidden_size, args.attention_size)

        self.model = model.ImageCaptioningNet(encoder, decoder)
        self.model.to(args.device)

        # ------------------
        # Loss and optimizer
        # ------------------
        self.criterion = nn.CrossEntropyLoss()

        # Only the parameters of the final encoder layer are being optimized.
        params = self.model.trainable_parameters()
        self.optimizer = optim.Adam(params, lr=args.learning_rate)

        # ------
        # Others
        # ------
        self.idx2word_fn = utils.IdxToWord(self.vocab)

    @property
    def hparams(self):
        return {
            'encoder_type': self.args.encoder_type,
            'attention_type': self.args.attention_type,
            'num_epochs': self.args.num_epochs,
            'batch_size': self.args.batch_size,
            'learning_rate': self.args.learning_rate,
            'vocab_min_freq': self.args.vocab_min_freq,
            'embedding_size': self.args.embedding_size,
            'hidden_size': self.args.hidden_size,
            'attention_size': self.args.attention_size
        }

    def _dummy_input(self):
        """
        Returns a tuple with a dummy input (random) for the model. This method is used to ease the
        call of the add_graph method of the tensorboard summary writer.
        """
        dummy_imgs = torch.randn(self.args.batch_size, 3, 224, 224, dtype=torch.float32)
        dummy_caps = torch.randint(low=0,
                                   high=len(self.vocab) - 1,
                                   size=(self.args.batch_size, self.args.max_seq_length),
                                   dtype=torch.int64)
        dummy_lens = torch.randint(low=1,
                                   high=self.args.max_seq_length,
                                   size=(self.args.batch_size, ),
                                   dtype=torch.int64)
        dummy_lens, _ = torch.sort(dummy_lens, descending=True)

        return (dummy_imgs.to(self.args.device), dummy_caps.to(self.args.device), dummy_lens)

    def _compute_accuracy(self, predicted, target):
        """
        Computes accuracy based on BLEU.

            :param predicted: Predicted captions. Shape: (batch_size, max_length).
            :param target: Target captions. Shape: (batch_size, max_length).
            :returns: average of the bleu score of each predicted caption.
        """
        total_bleu = 0
        for predicted_cap, target_cap in zip(predicted, target):
            predicted_cap = self.idx2word_fn(predicted_cap.tolist())
            target_cap = self.idx2word_fn(target_cap.tolist())

            bleu = torchtext.data.metrics.bleu_score([predicted_cap], [[target_cap]])
            total_bleu += bleu
        return (total_bleu / self.args.batch_size) * 100.0

    def _train_epoch(self, epoch):
        """
        Training step for one epoch.

            :param epoch: current epoch (int)
            :return: average of loss and accurancy for the current epoch.
        """
        self.model.train()

        total_loss = 0
        total_accuracy = 0
        for i, (data, train_caps, loss_caps, lengths) in enumerate(self.train_loader):

            imgs = data.to(self.args.device)  # (batch_size, channels, h, w)
            train_caps = train_caps.to(self.args.device)  # (batch_size, max_length)
            loss_caps = loss_caps.to(self.args.device)  # (batch_size, max_length)

            # 0. Clear gradients.
            self.optimizer.zero_grad()

            # 1. Forward the data through the network.
            out, _ = self.model(imgs, train_caps, lengths)

            # 2. Compute loss.
            loss = self.criterion(out.view(-1, len(self.vocab)), loss_caps.view(-1))

            # 3. Backprop with repsect to the loss function.
            loss.backward()

            # 4) Apply the optimizer with a learning step.
            self.optimizer.step()

            # 5. Computing loss and accuracy.
            _, predicted_caps = out.max(2)  # predicted_caps = (batch_size, max_length)
            loss_value = loss.item()
            acc_value = self._compute_accuracy(train_caps, predicted_caps)

            if self.args.log_interval > 0 and i % self.args.log_interval == 0:
                print('Epoch [{}/{}] - [{}/{}] [TRAIN] Loss: {} | Acc: {}'.format(
                    epoch + 1, self.args.num_epochs, i, len(self.train_loader), loss_value,
                    acc_value))

                # Writing scalars to tensorboard.
                step = epoch * (len(self.train_loader)) + i
                self.writer.add_scalar('Loss/train', loss_value, step)
                self.writer.add_scalar('Accuracy/train', acc_value, step)

            # Adding loss and accuracy to totals.
            total_loss += loss_value
            total_accuracy += acc_value

        return total_loss / len(self.train_loader), total_accuracy / (len(self.train_loader))

    def _validate_epoch(self, epoch):
        """
        Validation step for one epoch.

            :param epoch: current epoch (int)
            :return: average of loss and accurancy for the current epoch.
        """
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            total_accuracy = 0
            for i, (data, train_caps, loss_caps, lengths) in enumerate(self.val_loader):

                imgs = data.to(self.args.device)  # (batch_size, channels, h, w)
                train_caps = train_caps.to(self.args.device)  # (batch_size, max_length)
                loss_caps = loss_caps.to(self.args.device)  # (batch_size, max_length)

                # 1. Forward the data through the network.
                out, _ = self.model(imgs, train_caps, lengths)

                # 2. Compute loss.
                loss = self.criterion(out.view(-1, len(self.vocab)), loss_caps.view(-1))

                # 3. Computing loss and accuracy.
                _, predicted_caps = out.max(2)  # predicted_caps = (batch_size, max_length)
                loss_value = loss.item()
                acc_value = self._compute_accuracy(train_caps, predicted_caps)

                if self.args.log_interval > 0 and i % self.args.log_interval == 0:
                    print('Epoch [{}/{}] - [{}/{}] [EVAL] Loss: {} | Acc: {}'.format(
                        epoch + 1, self.args.num_epochs, i, len(self.val_loader), loss_value,
                        acc_value))

                    # Writing scalars to tensorboard.
                    step = epoch * (len(self.val_loader)) + i
                    self.writer.add_scalar('Loss/eval', loss_value, step)
                    self.writer.add_scalar('Accuracy/eval', acc_value, step)

                # Adding loss and accuracy to totals.
                total_loss += loss_value
                total_accuracy += acc_value

        return total_loss / len(self.val_loader), total_accuracy / (len(self.val_loader))

    def train(self):
        """
        Training loop
        """
        # Starting tensorboard writer.
        if self.args.log_interval > 0:
            if args.session_name is not None:
                self.writer = SummaryWriter(os.path.join('runs', args.session_name))
            else:
                self.writer = SummaryWriter()

            self.writer.add_graph(self.model, self._dummy_input())

        for epoch in range(self.args.num_epochs):

            train_loss, train_acc = self._train_epoch(epoch)
            eval_loss, eval_acc = self._validate_epoch(epoch)

            # Save checkpoint of the model.
            if self.args.save_checkpoints:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_loss
                    }, 'checkpoints/checkpoints-{}.tar'.format(epoch + 1))

        if self.args.log_interval > 0:
            logging.info('Logging hparams...')
            self.writer.add_hparams(
                self.hparams, {
                    'hparam/train-loss': train_loss,
                    'hparam/train-accuracy': train_acc,
                    'hparam/eval-loss': eval_loss,
                    'hparam/eval-train': eval_acc
                })

            logging.info('Logging embeddings...')
            self.writer.add_embedding(self.model.decoder.embedding.weight,
                                      metadata=self.vocab.get_words(),
                                      global_step=0)

            self.writer.close()

        if not self.args.no_save_model:
            model_name = self.args.session_name if self.args.session_name is not None else 'model'

            logging.info('Saving model as {}...'.format(model_name))
            torch.save(self.model.state_dict(), os.path.join('models', model_name + '.pt'))

            logging.info('Saving arguments of the model...')
            arguments = {
                'encoder_type': self.args.encoder_type,
                'attention_type': self.args.attention_type,
                'vocab_min_freq': self.args.vocab_min_freq,
                'encoder_size': self.args.encoder_size,
                'hidden_size': self.args.hidden_size,
                'embedding_size': self.args.embedding_size,
                'attention_size': self.args.attention_size,
                'overfitting': self.args.overfitting
            }
            with open(os.path.join('models', model_name + '.json'), 'w') as f:
                json.dump(arguments, f)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)

    # Data parameters.
    argparser.add_argument('--session-name', type=str, help='session name')
    argparser.add_argument('--data-root',
                           metavar='PATH',
                           type=str,
                           default='',
                           help='path for FLickr8k data')

    # Training parameters.
    argparser.add_argument('--num-epochs',
                           type=int,
                           default=10,
                           help='number of epochs (default: 10)')
    argparser.add_argument('--batch-size', type=int, default=32, help='batch size (default: 32)')
    argparser.add_argument('--learning-rate',
                           type=float,
                           default=1e-3,
                           help='learning rate (default: 1e-3)')
    argparser.add_argument('--num-workers',
                           type=int,
                           default=4,
                           help='number of workers used in the data loader (default: 4)')

    # Model parameters.
    argparser.add_argument('--encoder-type',
                           type=str,
                           choices=['resnet101', 'senet154'],
                           help="select the encoder type",
                           default='resnet101')
    argparser.add_argument('--attention-type',
                           type=str,
                           choices=['none', 'additive'],
                           help="select the decoder type",
                           default='additive')
    argparser.add_argument('--vocab-min-freq',
                           type=int,
                           default=1,
                           help='minimum frequency of a word to be added in the vocab (default: 1)')
    argparser.add_argument('--max-seq-length',
                           type=int,
                           default=25,
                           help='maximum sequence length (default: 25)')
    argparser.add_argument('--encoder-size',
                           type=int,
                           default=64,
                           help='encoder size (default: 128)')
    argparser.add_argument('--hidden-size',
                           type=int,
                           default=256,
                           help='hidden size (default: 256)')
    argparser.add_argument('--embedding-size',
                           type=int,
                           default=128,
                           help='embedding size (default: 128)')
    argparser.add_argument('--attention-size',
                           type=int,
                           default=64,
                           help='attention size (default: 256)')

    # Logging parameters
    argparser.add_argument('--no-save-model', action='store_true', help='do not save trained model')
    argparser.add_argument('--log-interval',
                           type=int,
                           default=25,
                           help='logging step with tensorboard (per batch) (default: 25)')
    argparser.add_argument('--save-checkpoints', action='store_true', help='save checkpoints')
    argparser.add_argument('--overfitting', action='store_true', help='use overfitting dataset')

    # Other parameters.
    argparser.add_argument('--debug', action='store_true', help='enable debug messages')

    args = argparser.parse_args()
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Checking is GPU is available.
    if torch.cuda.is_available():
        logging.info('GPU available, setting device argument to "cuda"')
        args.device = 'cuda'
    else:
        logging.warning('GPU not available, setting device argument to "cpu"')
        args.device = 'cpu'

    # Finding data root if not set by the user.
    repo_path = os.path.dirname(os.path.realpath(__file__))
    if args.data_root == '':
        if args.overfitting is True:
            args.data_root = os.path.join(repo_path, 'data/flickr8k_overfitting')
        else:
            args.data_root = os.path.join(repo_path, 'data/flickr8k')

        if not os.path.exists(args.data_root):
            raise RuntimeError('Could not find Flickr8k data')

    # Checking whether the models folder exists.
    models_path = os.path.join(repo_path, 'models')
    if not args.no_save_model and not os.path.exists(models_path):
        logging.info('Creating models folder to save the trained model')
        os.mkdir(models_path)

    # Checking whether the checkpoints folder exists.
    checkpoints_path = os.path.join(repo_path, 'checkpoints')
    if args.save_checkpoints and not os.path.exists(checkpoints_path):
        logging.info('Creating checkpoints folder')
        os.mkdir(checkpoints_path)

    # Trainning model.
    trainer = Train(args)
    trainer.train()
