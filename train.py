#!/usr/bin/env python
import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet_04 import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams, pretrained=False):
    model = Unet(hparams)

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        #filepath=os.path.join(log_dir, 'checkpoints'),
        #save_best_only=False,
        # dirpath='/content/drive/MyDrive/ETH/drive_lightning_logs/checkpoints',
        # filepath=os.path.join(log_dir, 'checkpoints'),
        # filepath='/content/drive/MyDrive/ETH/drive_lightning_logs/checkpoints',
        # filename='sample-unet-{epoch}-{val_loss:.2f}',       
        # mode='min',
        # save_best_only=True,
        verbose=True
    )
    
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=True,
    )    
    trainer = Trainer(
        gpus=hparams.gpu,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hparams.max_epoch,
        auto_lr_find=True,
        # default_root_dir='/content/drive/MyDrive/ETH/drive_lightning_logs'
        # early_stop_callback=stop_callback,
    )

    lr_finder = trainer.tuner.lr_find(model)

    # Inspect results
    fig = lr_finder.plot(); fig.show()
    suggested_lr = lr_finder.suggestion()

    # Overwrite lr and create new model
    hparams.lr = suggested_lr

    model = Unet(hparams)
    if pretrained:
      # checkpoint = 
      # 'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth'
      model.load_from_checkpoint('/content/drive/MyDrive/ETH/drive_lightning_logs/train_2021-02-07_17-23-31/checkpoints.ckpt')
        
    # Ready to train with new learning rate
    trainer.fit(model)
    trainer.save_checkpoint("/content/drive/MyDrive/ETH/drive_lightning_logs/checkpoints/after_train.ckpt")


# class Params: # hparams
#   def __init__(self, dataset, log_dir, n_channels, n_classes, lr):
#     self.dataset = dataset
#     self.log_dir = log_dir
#     self.n_channels = n_channels
#     self.n_classes = n_classes
#     self.lr = lr

# hparams = Params(dataset='/content/train_patch_01',
#                  log_dir='/content/drive/MyDrive/ETH/drive_lightning_logs',
#                  n_channels=3,
#                  n_classes=1,
#                  lr=0.1)
# main(hparams, pretrained = True)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='/content/drive/MyDrive/ETH/drive_lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)



