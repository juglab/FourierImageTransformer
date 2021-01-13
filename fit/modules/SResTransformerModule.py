import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.transformers.SResTransformer import SResTransformer
from fit.utils import convert2FC, PSNR, convert_to_dft
from fit.utils.RAdam import RAdam

import numpy as np

import torch.fft

from fit.utils.utils import denormalize


class SResTransformerModule(LightningModule):
    def __init__(self, d_model, img_shape=27,
                 lr=0.0001,
                 weight_decay=0.01,
                 n_layers=4, n_heads=4, d_query=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "lr",
                                  "weight_decay",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout")

        self.x_coords_img, self.y_coords_img, self.dst_flatten_coords, self.dst_order = self._get_fc_coords(
            img_shape=img_shape)
        self.dft_shape = (img_shape, img_shape // 2 + 1)

        self.sres = SResTransformer(d_model=self.hparams.d_model,
                                    y_coords_img=self.y_coords_img, x_coords_img=self.x_coords_img,
                                    attention_type='causal-linear',
                                    n_layers=self.hparams.n_layers,
                                    n_heads=self.hparams.n_heads,
                                    d_query=self.hparams.d_query,
                                    dropout=self.hparams.dropout,
                                    attention_dropout=self.hparams.attention_dropout)

    def forward(self, x):
        return self.sres.forward(x)

    def _get_fc_coords(self, img_shape):
        xcoords, ycoords = np.meshgrid(np.linspace(0, img_shape // 2, num=img_shape // 2 + 1, endpoint=True),
                                       np.concatenate([np.linspace(0, img_shape // 2, img_shape // 2, False),
                                                       np.linspace(img_shape // 2, img_shape - 1, img_shape // 2 + 1)]))

        order = np.sqrt(xcoords ** 2 + (ycoords - (img_shape // 2)) ** 2)
        order = np.roll(order, img_shape // 2 + 1, 0)
        xcoords = np.roll(xcoords, img_shape // 2 + 1, 0)
        ycoords = np.roll(ycoords, img_shape // 2 + 1, 0)
        flatten_indices = np.argsort(order.flatten())
        xcoords = xcoords.flatten()[flatten_indices]
        ycoords = ycoords.flatten()[flatten_indices]
        return torch.from_numpy(xcoords), torch.from_numpy(ycoords), torch.from_numpy(flatten_indices), order

    def configure_optimizers(self):
        optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Train/avg_val_loss'
        }

    def _fc_loss(self, pred_fc, target_fc, mag_min, mag_max):
        c1 = convert2FC(pred_fc, mag_min=mag_min, mag_max=mag_max)
        c1 = torch.stack([c1.real, c1.imag], dim=-1)
        c2 = convert2FC(target_fc, mag_min=mag_min, mag_max=mag_max)
        c2 = torch.stack([c2.real, c2.imag], dim=-1)
        amp1 = torch.linalg.norm(c1, dim=-1).unsqueeze(-1)
        amp2 = torch.linalg.norm(c2, dim=-1).unsqueeze(-1)
        c1_unit = c1 / amp1
        c2_unit = c2 / amp2

        amp_loss = (1 + torch.pow(amp1 - amp2, 2))
        phi_loss = (2 - torch.sum(c1_unit * c2_unit, dim=-1, keepdim=True))
        return torch.mean(amp_loss * phi_loss), torch.mean(amp_loss), torch.mean(phi_loss)

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self._fc_loss(pred_fc=pred_fc, target_fc=target_fc, mag_min=mag_min,
                                                    mag_max=mag_max)
        return fc_loss, amp_loss, phi_loss

    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_coords][:, :-1]
        y_fc = fc[:, self.dst_flatten_coords][:, 1:]

        pred = self.sres.forward(x_fc)

        fc_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def training_epoch_end(self, outputs):
        loss = [d['loss'] for d in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]
        self.log('Train/loss', torch.mean(torch.stack(loss)), logger=True, on_epoch=True)
        self.log('Train/amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_coords][:, :-1]
        y_fc = fc[:, self.dst_flatten_coords][:, 1:]

        pred = self.sres.forward(x_fc)

        val_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)

        self.log_dict({'val_loss': val_loss})
        if batch_idx == 0:
            self.log_val_images(pred, x_fc, y_fc, mag_min, mag_max)
        return {'val_loss': val_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def log_val_images(self, pred, x, y_fc, mag_min, mag_max):
        pred = torch.cat([x[:, :1], pred], dim=1)
        y_fc = torch.cat([x[:, :1], y_fc], dim=1)

        dft_pred = convert_to_dft(fc=pred, mag_min=mag_min, mag_max=mag_max,
                                  dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)
        dft_target = convert_to_dft(fc=y_fc, mag_min=mag_min, mag_max=mag_max,
                                    dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)

        for i in range(3):
            pred_img = torch.fft.irfftn(dft_pred[i], s=2 * (self.hparams.img_shape,))
            y_img = torch.fft.irfftn(dft_target[i], s=2 * (self.hparams.img_shape,))

            pred_img = torch.clamp((pred_img - pred_img.min()) / (pred_img.max() - pred_img.min()), 0, 1)
            y_img = torch.clamp((y_img - y_img.min()) / (y_img.max() - y_img.min()), 0, 1)

            self.trainer.logger.experiment.add_image('predcitions/img_{}'.format(i), pred_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('groundtruth/img_{}'.format(i), y_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]

        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        assert len(fc) == 1, 'Test images have to be evaluated independently.'
        x_fc = fc[:, self.dst_flatten_coords][:, :-1]
        y_fc = fc[:, self.dst_flatten_coords][:, 1:]

        pred = self.sres.forward(x_fc)

        pred = torch.cat([x_fc[:, :1], pred], dim=1)
        pred_dft = convert_to_dft(fc=pred, mag_min=mag_min, mag_max=mag_max, dst_flatten_coords=self.dst_flatten_coords,
                                  img_shape=self.hparams.img_shape)
        pred_img = torch.fft.irfftn(pred_dft[0], s=2 * (self.hparams.img_shape,))

        y = torch.cat([x_fc[:, :1], y_fc], dim=1)
        y = convert_to_dft(fc=y, mag_min=mag_min, mag_max=mag_max, dst_flatten_coords=self.dst_flatten_coords,
                           img_shape=self.hparams.img_shape)
        y_img = torch.fft.irfftn(y[0], s=2 * (self.hparams.img_shape,))

        gt = denormalize(y_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)

        return PSNR(gt, pred_img, drange=torch.tensor(255., dtype=torch.float32))

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs)
        self.log('Mean PSNR', torch.mean(outputs).detach().cpu().numpy(), logger=True)
        self.log('SEM PSNR', torch.std(outputs / np.sqrt(len(outputs))).detach().cpu().numpy(),
                 logger=True)
