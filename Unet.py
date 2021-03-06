#!/usr/bin/env python
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as utils

from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl

from dataset import DirDataset


class Unet(pl.LightningModule):
    def __init__(self, hparams):
        super(Unet, self).__init__()
        self.hparams = hparams
        self.lr = hparams.lr
        self.n_channels = hparams.n_channels
        self.n_classes = hparams.n_classes
        self.bilinear = True
        # self.save_hyperparameters()

        def double_conv(in_channels, out_channels, kernel_size_val, padding_val):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_val, padding=padding_val, stride=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_val, padding=padding_val, stride=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.ReLU()
            )

        def down(in_channels, out_channels, kernel_size_val, padding_val):
            return nn.Sequential(
                double_conv(in_channels, out_channels, kernel_size_val, padding_val),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        def contract_block(in_channels, out_channels, kernel_size_val, padding_val):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size_val, padding=padding_val, stride=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size_val, padding=padding_val, stride=1),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

        class up(pl.LightningModule):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                else:
                    self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2,
                                                kernel_size=2, stride=2)

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                                diffY // 2, diffY - diffY // 2])
                x = torch.cat([x2, x1], dim=1) ## why 1?
                return self.conv(x)

        def expand_block(in_channels, out_channels, kernel_size_val, padding_val):

            expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size_val, stride=1, padding=padding_val),
                                torch.nn.BatchNorm2d(out_channels),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(out_channels, out_channels, kernel_size_val, stride=1, padding=padding_val),
                                torch.nn.BatchNorm2d(out_channels),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                                )
            return expand

        def expand_end_block(in_channels, out_channels, kernel_size_val, padding_val):

            expand_end = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size_val, stride=1, padding=padding_val),
                                torch.nn.BatchNorm2d(out_channels),
                                torch.nn.ReLU(),
                                torch.nn.Conv2d(out_channels, out_channels, kernel_size_val, stride=1, padding=padding_val),
                                torch.nn.BatchNorm2d(out_channels),
                                torch.nn.ReLU(),
                                torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                                torch.nn.Sigmoid()
                                )
            return expand_end

        self.conv1 = contract_block(self.n_channels, 32, 7, 3)
        self.conv2 = contract_block(32, 64, 3, 1)
        self.conv3 = contract_block(64, 128, 3, 1)

        self.upconv3 = expand_block(128, 64, 3, 1)
        self.upconv2 = expand_block(64*2, 32, 3, 1)
        # self.upconv1 = expand_end_block(32*2, self.n_classes, 3, 1)
        self.upconv1 = expand_block(32*2, self.n_classes, 3, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)

        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        return upconv1

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)

        y_hat = torch.squeeze(y_hat)

        # log 4 example images
        sample_y = y[:4]
        sample_y_hat = y_hat[:4]
        grid_1 = utils.make_grid(sample_y)
        grid_2 = utils.make_grid(sample_y_hat)
        self.logger.experiment.add_image('y', grid_1, 0)
        self.logger.experiment.add_image('y_hat', grid_2, 0)

        y = torch.squeeze(y)
        y = y.type_as(y_hat)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 2 else \
                    F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = nn.BCELoss(y_hat)
        tensorboard_logs = {'train_loss': loss}        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def acc_metric(self, predb, yb):
        a = predb.argmax(dim=0)
        b = yb.argmax(dim=0)
        return  (a == b).float().mean()

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        y = torch.squeeze(y)
        y = y.type_as(y_hat)
        y_hat = torch.squeeze(y_hat)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 2 else \
                    F.binary_cross_entropy_with_logits(y_hat, y)
        # loss = nn.BCELoss(y_hat)

        # calculate acc
        val_acc = self.acc_metric(y_hat, y)
        self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc,on_step=True, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': val_acc}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        self.log('avg_val_loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        # return torch.optim.RMSprop(self.parameters(), lr=0.001, weight_decay=1e-8)
        # return torch.optim.RMSprop(self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=1e-8)
        
    def __dataloader(self):
        dataset = self.hparams.dataset
        dataset = DirDataset(f'{dataset}/images',f'{dataset}/gt')
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=16, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, pin_memory=True, shuffle=False)

        return {'train': train_loader,'val': val_loader}

    

    # @pl.data_loader
    def train_dataloader(self):
        return self.__dataloader()['train']

    # @pl.data_loader
    def val_dataloader(self):
        return self.__dataloader()['val']

    # @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--gpu', type=int, default=0)
        parser.add_argument('--max_epoch', type=int, default=10)
        return parser
        