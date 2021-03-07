#!/usr/bin/env python
import os
from argparse import ArgumentParser

import numpy as np
import torch

from Unet import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger

def weights_update(model, checkpoint):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def main(hparams):
    # model = Unet(hparams)
    checkpoint_path = '/content/drive/MyDrive/ETH/drive_lightning_logs/train_logs_2021-03-06_15-00-53/version_0/checkpoints/epoch=19-step=7679.ckpt'
    model = weights_update(model=Unet(hparams), checkpoint=torch.load(checkpoint_path))

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
        # dirpath='/home/sebastian/workspace/ETH/scratch/lightning_logs',
        verbose=True,
    )


    logger = TestTubeLogger(
        save_dir='/content/drive/MyDrive/ETH/drive_lightning_logs/train_logs_2021-03-07_13-19-24/'',
        version=0  # An existing version with a saved checkpoint
    )

    # stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     mode='auto',
    #     patience=5,
    #     verbose=True,
    # )

    trainer = Trainer(
        logger=logger,
        default_save_path='/content/lightning_logs'
        # overfit_batches=0.1111111111111111,
        fast_dev_run=False,
        gpus=hparams.gpu,
        auto_lr_find=True,
        checkpoint_callback=checkpoint_callback,
        max_epochs=hparams.max_epoch,
    )
    """
    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(model)

    # Results can be found in
    lr_finder.results
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    print("Suggested lr:", new_lr)

    # update hparams of the model
    model.hparams.lr = new_lr
    """

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--dataset', required=True)
    parent_parser.add_argument('--log_dir', default='/content/drive/MyDrive/ETH/drive_lightning_logs')

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
