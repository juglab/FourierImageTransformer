from fit.datamodules.tomo_rec import MNIST_TRecFITDM
from fit.utils.tomo_utils import get_polar_rfft_coords_2D, get_polar_rfft_coords_sinogram
from fit.utils import denormalize, convert2DFT
from fit.modules import TRecTransformerModule

from matplotlib import pyplot as plt

import torch

import numpy as np

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import wget
from os.path import exists

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from os.path import join, exists


import configparser
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    
    config.read(args.config)
    
    wandb_logger = WandbLogger(project="fit", id=config['DEFAULT']['wandbid'], resume='allow')
    
    
    
    
    seed_everything(22122020)
    dm = MNIST_TRecFITDM(root_dir='/tungstenfs/scratch/gmicro/buchtimo/gitrepos/FourierImageTransformer/examples/datamodules/data/', batch_size=int(config['DEFAULT']['batch_size']), num_angles=int(config['DEFAULT']['num_angles']))
    # FIT: TRec + FBP vs FIT: TRec
    with_fbp = bool(config['DEFAULT']['with_fbp'])

    dm.prepare_data()
    dm.setup()
    
    angles = dm.gt_ds.get_ray_trafo().geometry.angles
    det_len = dm.gt_ds.get_ray_trafo().geometry.detector.shape[0]
    
    img_shape = dm.gt_shape
    
    proj_r, proj_phi, src_flatten = get_polar_rfft_coords_sinogram(angles=angles, 
                                                               det_len=det_len)
    target_r, target_phi, dst_flatten, order = get_polar_rfft_coords_2D(img_shape=img_shape)
    
    n_heads = int(config['DEFAULT']['n_heads'])
    d_query = int(config['DEFAULT']['d_query'])
    model = TRecTransformerModule(d_model=n_heads * d_query, 
                                  sinogram_coords=(proj_r, proj_phi),
                                  target_coords=(target_r, target_phi),
                                  src_flatten_coords=src_flatten, 
                                  dst_flatten_coords=dst_flatten, 
                                  dst_order=order,
                                  angles=angles, 
                                  img_shape=img_shape,
                                  detector_len=det_len,
                                  loss=str(config['DEFAULT']['loss']), 
                                  use_fbp=with_fbp, 
                                  init_bin_factor=1, 
                                  bin_factor_cd=5,
                                  lr=0.0001, 
                                  weight_decay=0.01, 
                                  attention_type='linear', 
                                  n_layers=4,
                                  n_heads=n_heads, 
                                  d_query=d_query, 
                                  dropout=0.1, 
                                  attention_dropout=0.1)
    
    trainer = Trainer(logger=wandb_logger,max_epochs=-1,
                  gpus=1,
                  enable_checkpointing=True, 
                  callbacks=[ModelCheckpoint(
                                            dirpath=str(config['DEFAULT']['dirpath']),
                                            save_top_k=1,
                                            verbose=False,
                                            save_last=True,
                                            monitor='Train/avg_val_mse',
                                            mode='min'
                                        ),
                            LearningRateMonitor()],
                  deterministic=True)
    
    
    last_ckpt = join(str(config['DEFAULT']['dirpath']), 'last.ckpt')
    if exists(last_ckpt):
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt);
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=None);


if __name__ == '__main__':
    main()