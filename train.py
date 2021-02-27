#!/usr/bin/env python
import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    model = Unet(hparams)

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # dirpath=os.path.join(log_dir, 'checkpoints'),
        save_top_k=2,
        # save_besct_only=False,
        verbose=True,
    )
    
    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='auto',
    #     patience=5,
    #     verbose=True,
    # )    

    trainer = Trainer(
        overfit_batches=0.1111111111111111,
        fast_dev_run=False,
        gpus=hparams.gpu,
        checkpoint_callback=checkpoint_callback,
        # early_stop_callback=stop_callback,
        max_epochs=hparams.max_epoch,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='/content/drive/MyDrive/ETH/drive_lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
