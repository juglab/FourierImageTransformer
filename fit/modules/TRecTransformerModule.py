import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.core.step_result import TrainResult, EvalResult

from fit.transformers.TRecTransformer import TRecTransformer
from fit.utils import convert2FC, fft_interpolate
from fit.utils.RAdam import RAdam

import numpy as np

from torch.nn import functional as F
import torch.fft


class TRecTransformerModule(LightningModule):
    def __init__(self, d_model, y_coords_proj, x_coords_proj, y_coords_img, x_coords_img, angles, img_shape=362,
                 lr=0.0001,
                 weight_decay=0.01,
                 loss_switch=0.5,
                 attention_type="linear", n_layers=4, n_heads=4, d_query=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "lr",
                                  "weight_decay",
                                  "loss_switch",
                                  "attention_type",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout")
        self.y_coords_proj = y_coords_proj
        self.x_coords_proj = x_coords_proj
        self.y_coords_img = y_coords_img
        self.x_coords_img = x_coords_img
        self.angles = angles
        self.dft_shape = (img_shape, img_shape // 2 + 1)

        self.trec = TRecTransformer(d_model=self.hparams.d_model,
                                    y_coords_proj=y_coords_proj, x_coords_proj=x_coords_proj,
                                    y_coords_img=y_coords_img, x_coords_img=x_coords_img,
                                    attention_type=self.hparams.attention_type,
                                    n_layers=self.hparams.n_layers,
                                    n_heads=self.hparams.n_heads,
                                    d_query=self.hparams.d_query,
                                    dropout=self.hparams.dropout,
                                    attention_dropout=self.hparams.attention_dropout)

        self.criterion = self._fc_loss
        self.using_real_loss = False

    def forward(self, x):
        return self.trec.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.trec.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer

    def _real_loss(self, pred_fc, target_fc, target_real, mag_min, mag_max):
        mag = pred_fc[..., 0]
        phi = pred_fc[..., 1]
        mag = (mag * (mag_max - mag_min)) + mag_min
        mag = torch.exp(mag)

        phi = phi * 2 * np.pi
        dft = torch.complex(mag * torch.cos(phi), mag * torch.sin(phi))
        dft = dft.reshape(-1, *self.dft_shape)
        y_hat = torch.roll(torch.fft.irfftn(dft, dim=[1, 2]),
                           (self.hparams.img_shape // 2, self.hparams.img_shape // 2), (1, 2))
        return F.mse_loss(y_hat, target_real)

    def _fc_loss(self, pred_fc, target_fc, target_real, mag_min, mag_max):
        return F.mse_loss(pred_fc, target_fc)

    def training_step(self, batch, batch_idx):
        x_fc, y_fc, y_real, (mag_min, mag_max) = batch
        pred = self(x_fc)
        loss = self.criterion(pred, y_fc, y_real, mag_min, mag_max)
        return loss

    def on_train_epoch_start(self):
        if not self.using_real_loss and self.current_epoch >= (self.trainer.max_epochs * self.hparams.loss_switch):
            self.criterion = self._real_loss
            print('Epoch {}/{}: Switched to real loss.'.format(self.current_epoch, self.trainer.max_epochs - 1))
            self.using_real_loss = True

    def training_epoch_end(self, outputs):
        loss = [d['loss'] for d in outputs]
        self.log('Train/loss', torch.mean(torch.stack(loss)), logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x_fc, y_fc, y_real, (mag_min, mag_max) = batch
        pred = self(x_fc)
        val_loss = self.criterion(pred, y_fc, y_real, mag_min, mag_max)
        val_mse = self._real_loss(pred, y_fc, y_real, mag_min, mag_max)
        self.log_dict({'val_loss': val_loss})
        self.log_dict({'val_mse': val_mse})
        if batch_idx == 0:
            self.log_val_images(pred, x_fc, y_real, mag_min, mag_max)
        return {'val_loss': val_loss, 'val_mse': val_mse}

    def log_val_images(self, pred, x, y_real, mag_min, mag_max):
        x_fc = convert2FC(x, mag_min, mag_max)
        pred_fc = convert2FC(pred, mag_min, mag_max)

        for i in range(3):
            x_dft = fft_interpolate(self.x_coords_proj.cpu().numpy(), self.y_coords_proj.cpu().numpy(),
                                    self.x_coords_img.cpu().numpy(), self.y_coords_img.cpu().numpy(),
                                    x_fc[i].cpu().numpy(), target_shape=self.dft_shape)
            x_img = torch.roll(torch.fft.irfftn(torch.from_numpy(x_dft)),
                                  2 * (self.hparams.img_shape // 2,), (0, 1))
            x_img = torch.clamp((x_img - x_img.min()) / (x_img.max() - x_img.min()), 0, 1)

            pred_img = torch.roll(torch.fft.irfftn(pred_fc[i].reshape(self.dft_shape)),
                                  2 * (self.hparams.img_shape // 2,), (0, 1))
            pred_img = torch.clamp((pred_img - pred_img.min()) / (pred_img.max() - pred_img.min()), 0, 1)

            y_img = y_real[i]
            y_img = torch.clamp((y_img - y_img.min()) / (y_img.max() - y_img.min()), 0, 1)

            self.trainer.logger.experiment.add_image('inputs/img_{}'.format(i), x_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('predcitions/img_{}'.format(i), pred_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('targets/img_{}'.format(i), y_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        val_mse = [o['val_mse'] for o in outputs]
        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_mse', torch.mean(torch.stack(val_mse)), logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, y_real, (mag_min, mag_max) = batch
        assert len(x) == 1, 'Test images have to be evaluated independently.'

        pred = self(x)

        pred_fc = convert2FC(pred, mag_min, mag_max)
        pred_img = torch.roll(torch.fft.irfftn(pred_fc[0].reshape(self.dft_shape)),
                              2 * (self.hparams.img_shape // 2,), (0, 1))
        return self.PSNR(y_real[0], pred_img, drange=torch.tensor(255., dtype=torch.float32))

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs)
        self.log('Mean PSNR', torch.mean(outputs).detach().cpu().numpy(), logger=True)
        self.log('SEM PSNR', torch.std(outputs/ np.sqrt(len(outputs))).detach().cpu().numpy(),
                    logger=True)

    def normalize_minmse(self, x, target):
        """Affine rescaling of x, such that the mean squared error to target is minimal."""
        cov = np.cov(x.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())
        alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
        beta = target.mean() - alpha * x.mean()
        return alpha * x + beta

    def PSNR(self, gt, img, drange):
        img = self.normalize_minmse(img, gt)
        mse = torch.mean(torch.square(gt - img))
        return 20 * torch.log10(drange) - 10 * torch.log10(mse)
