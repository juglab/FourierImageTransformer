import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.baselines.ConvBlockBaseline import ConvBlockBaseline
from fit.datamodules.tomo_rec import MNISTTomoFourierTargetDataModule
from fit.utils import PSNR
from fit.utils.RAdam import RAdam

import numpy as np

from torch.nn import functional as F
import torch.fft

from fit.utils.utils import denormalize


class ConvBlockBaselineModule(LightningModule):
    def __init__(self, img_shape=27,
                 lr=0.0001,
                 weight_decay=0.01,
                 d_query=4):
        super().__init__()

        self.save_hyperparameters("img_shape",
                                  "lr",
                                  "weight_decay",
                                  "d_query")

        self.cblock = ConvBlockBaseline(d_query=self.hparams.d_query)

        x, y = torch.meshgrid(torch.arange(-MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1),
                              torch.arange(-MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1))
        self.register_buffer('circle', torch.sqrt(x ** 2. + y ** 2.) <= MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2)

    def forward(self, x):
        return self.cblock.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.cblock.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Train/avg_val_mse'
        }

    def training_step(self, batch, batch_idx):
        x, y = batch

        pred = self.cblock.forward(x)

        loss = F.mse_loss(pred, y)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss = [d['loss'] for d in outputs]
        self.log('Train/loss', torch.mean(torch.stack(loss)), logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred = self.cblock.forward(x)

        val_loss = F.mse_loss(pred, y)
        if batch_idx == 0:
            self.log_val_images(pred, x, y)
        return {'val_loss': val_loss, 'val_mse': val_loss}

    def log_val_images(self, pred_img, x, y):

        for i in range(3):
            x_img = x[i]
            x_img = torch.clamp((x_img - x_img.min()) / (x_img.max() - x_img.min()), 0, 1)
            pred_img_ = pred_img[i]
            pred_img_ = torch.clamp((pred_img_ - pred_img_.min()) / (pred_img_.max() - pred_img_.min()), 0, 1)
            y_img = y[i]
            y_img = torch.clamp((y_img - y_img.min()) / (y_img.max() - y_img.min()), 0, 1)
            print(x_img.shape, y_img.shape, pred_img_.shape)
            self.trainer.logger.experiment.add_image('inputs/img_{}'.format(i), x_img,
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('predcitions/img_{}'.format(i), pred_img_,
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('targets/img_{}'.format(i), y_img,
                                                     global_step=self.trainer.global_step)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        val_mse = [o['val_mse'] for o in outputs]
        mean_val_mse = torch.mean(torch.stack(val_mse))

        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_mse', mean_val_mse, logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        assert len(x) == 1, 'Test images have to be evaluated independently.'

        pred_img = self.cblock.forward(x)


        gt = denormalize(y[0,0], self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img[0,0], self.trainer.datamodule.mean, self.trainer.datamodule.std)

        return PSNR(self.circle * gt, self.circle * pred_img, drange=torch.tensor(255., dtype=torch.float32))

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs)
        self.log('Mean PSNR', torch.mean(outputs).detach().cpu().numpy(), logger=True)
        self.log('SEM PSNR', torch.std(outputs / np.sqrt(len(outputs))).detach().cpu().numpy(),
                 logger=True)
