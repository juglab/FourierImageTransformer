import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.datamodules.tomo_rec import MNISTTomoFourierTargetDataModule
from fit.modules.loss import _fc_prod_loss, _fc_sum_loss
from fit.transformers.TRecTransformer import TRecTransformer, TRecEncoder, TRecConvBlock, TRecEncDec
from fit.utils import convert2FC, fft_interpolate, PSNR, convert_to_dft, psfft
from fit.utils.RAdam import RAdam

import numpy as np

from torch.nn import functional as F
import torch.fft

from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi


class TRecTransformerModule(LightningModule):
    def __init__(self, d_model, y_coords_proj, x_coords_proj, y_coords_img, x_coords_img, src_flatten_coords,
                 dst_flatten_coords, dst_order, angles, img_shape=27, detector_len=27,
                 loss='prod',
                 use_fbp=True,
                 init_bin_factor=4,
                 bin_factor_cd=10,
                 lr=0.0001,
                 weight_decay=0.01,
                 attention_type="linear", n_layers=4, n_heads=4, d_query=4, dropout=0.1, attention_dropout=0.1,
                 encoder_only=False,
                 convblock_only=False,
                 no_convblock=False):
        super().__init__()

        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "bin_factor_cd",
                                  "init_bin_factor",
                                  "detector_len",
                                  "loss",
                                  "use_fbp",
                                  "lr",
                                  "weight_decay",
                                  "attention_type",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout",
                                  "convblock_only")
        self.y_coords_proj = y_coords_proj
        self.x_coords_proj = x_coords_proj
        self.y_coords_img = y_coords_img
        self.x_coords_img = x_coords_img
        if not type(src_flatten_coords) is torch.Tensor:
            self.src_flatten_coords = torch.from_numpy(src_flatten_coords)
        else:
            self.src_flatten_coords = src_flatten_coords
        if not type(dst_flatten_coords) is torch.Tensor:
            self.dst_flatten_coords = torch.from_numpy(dst_flatten_coords)
        else:
            self.dst_flatten_coords = dst_flatten_coords
        self.dst_order = dst_order
        self.angles = angles
        self.num_angles = len(self.angles)
        self.dft_shape = (img_shape, img_shape // 2 + 1)
        self.bin_factor = init_bin_factor
        self.bin_count = 0
        self.best_mean_val_mse = 9999999
        self.bin_factor_patience = 10
        self.register_buffer('mask', psfft(self.bin_factor, pixel_res=img_shape))

        if loss == 'prod':
            self.loss = _fc_prod_loss
        else:
            self.loss = _fc_sum_loss

        if not self.hparams.use_fbp:
            self.register_buffer('random_cond', torch.rand(1, self.dst_flatten_coords.shape[0], 2))
        else:
            self.random_cond = None

        if no_convblock:
            self.trec = TRecEncDec(d_model=self.hparams.d_model,
                                    y_coords_proj=y_coords_proj, x_coords_proj=x_coords_proj,
                                    flatten_proj=self.src_flatten_coords,
                                    y_coords_img=y_coords_img, x_coords_img=x_coords_img,
                                    flatten_img=self.dst_flatten_coords,
                                    attention_type=self.hparams.attention_type,
                                    n_layers=self.hparams.n_layers,
                                    n_heads=self.hparams.n_heads,
                                    d_query=self.hparams.d_query,
                                    dropout=self.hparams.dropout,
                                    attention_dropout=self.hparams.attention_dropout)
        else:
            if convblock_only:
                self.trec = TRecConvBlock(d_model=self.hparams.d_model,
                                        y_coords_proj=y_coords_proj, x_coords_proj=x_coords_proj,
                                        flatten_proj=self.src_flatten_coords,
                                        y_coords_img=y_coords_img, x_coords_img=x_coords_img,
                                        flatten_img=self.dst_flatten_coords,
                                        attention_type=self.hparams.attention_type,
                                        n_layers=self.hparams.n_layers,
                                        n_heads=self.hparams.n_heads,
                                        d_query=self.hparams.d_query,
                                        dropout=self.hparams.dropout,
                                        attention_dropout=self.hparams.attention_dropout)
            else:
                if encoder_only:
                    self.trec = TRecEncoder(d_model=self.hparams.d_model,
                                            y_coords_proj=y_coords_proj, x_coords_proj=x_coords_proj,
                                            flatten_proj=self.src_flatten_coords,
                                            y_coords_img=y_coords_img, x_coords_img=x_coords_img,
                                            flatten_img=self.dst_flatten_coords,
                                            attention_type=self.hparams.attention_type,
                                            n_layers=self.hparams.n_layers,
                                            n_heads=self.hparams.n_heads,
                                            d_query=self.hparams.d_query,
                                            dropout=self.hparams.dropout,
                                            attention_dropout=self.hparams.attention_dropout)
                else:
                    self.trec = TRecTransformer(d_model=self.hparams.d_model,
                                                y_coords_proj=y_coords_proj, x_coords_proj=x_coords_proj,
                                                flatten_proj=self.src_flatten_coords,
                                                y_coords_img=y_coords_img, x_coords_img=x_coords_img,
                                                flatten_img=self.dst_flatten_coords,
                                                attention_type=self.hparams.attention_type,
                                                n_layers=self.hparams.n_layers,
                                                n_heads=self.hparams.n_heads,
                                                d_query=self.hparams.d_query,
                                                dropout=self.hparams.dropout,
                                                attention_dropout=self.hparams.attention_dropout)

        x, y = torch.meshgrid(torch.arange(-self.hparams.img_shape // 2 + 1,
                                           self.hparams.img_shape // 2 + 1),
                              torch.arange(-self.hparams.img_shape // 2 + 1,
                                           self.hparams.img_shape // 2 + 1))
        self.register_buffer('circle', torch.sqrt(x ** 2. + y ** 2.) <= self.hparams.img_shape // 2)

    def forward(self, x, out_pos_emb):
        return self.trec.forward(x, out_pos_emb)

    def configure_optimizers(self):
        optimizer = RAdam(self.trec.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Train/avg_val_mse'
        }

    def _real_loss(self, pred_img, target_fc, mag_min, mag_max):
        dft_target = convert_to_dft(fc=target_fc, mag_min=mag_min, mag_max=mag_max,
                                    dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)
        if self.bin_factor > 1:
            dft_target *= self.mask

        y_target = torch.roll(torch.fft.irfftn(dft_target, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                              2 * (self.hparams.img_shape // 2,), (1, 2))
        return F.mse_loss(pred_img, y_target)

    def _fc_loss(self, pred_fc, target_fc, mag_min, mag_max):
        pred_amp = denormalize_amp(pred_fc[..., 0], mag_min=mag_min, mag_max=mag_max)
        target_amp = denormalize_amp(target_fc[..., 0], mag_min=mag_min, mag_max=mag_max)

        pred_phi = denormalize_phi(pred_fc[..., 1])
        target_phi = denormalize_phi(target_fc[..., 1])

        amp_loss = 0 + torch.pow(pred_amp - target_amp, 2)
        phi_loss = 1 - torch.cos(pred_phi - target_phi)
        return torch.mean(amp_loss + phi_loss), torch.mean(amp_loss), torch.mean(phi_loss)

    def criterion(self, pred_fc, pred_img, target_fc, mag_min, mag_max):
        if self.hparams.convblock_only:
            return self._real_loss(pred_img=pred_img, target_fc=target_fc, mag_min=mag_min,
                                    mag_max=mag_max), torch.tensor(0.0), torch.tensor(0.0)
        else:
            fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, mag_min=mag_min,
                                                    mag_max=mag_max)
            real_loss = self._real_loss(pred_img=pred_img, target_fc=target_fc, mag_min=mag_min,
                                        mag_max=mag_max)
            return fc_loss + real_loss, amp_loss, phi_loss

    def _bin_data(self, x_fc, fbp_fc, y_fc):
        shells = (self.hparams.detector_len // 2 + 1) / self.bin_factor
        num_sino_fcs = np.clip(self.num_angles * int(shells + 1), 1, x_fc.shape[1])

        if self.bin_factor > 1:
            num_target_fcs = np.sum(self.dst_order <= shells)
        else:
            num_target_fcs = fbp_fc.shape[1]

        x_fc_ = x_fc[:, self.src_flatten_coords][:, :num_sino_fcs]
        if self.hparams.use_fbp:
            fbp_fc_ = fbp_fc[:, self.dst_flatten_coords][:, :num_target_fcs]
        else:
            fbp_fc_ = self.random_cond[:, self.dst_flatten_coords][:, :num_target_fcs]
            fbp_fc_ = torch.repeat_interleave(fbp_fc_, x_fc.shape[0], dim=0)

        y_fc_ = y_fc[:, self.dst_flatten_coords][:, :num_target_fcs]

        return x_fc_, fbp_fc_, y_fc_

    def training_step(self, batch, batch_idx):
        x_fc, fbp_fc, y_fc, y_real, (mag_min, mag_max) = batch
        x_fc_, fbp_fc_, y_fc_ = self._bin_data(x_fc, fbp_fc, y_fc)

        pred_fc, pred_img = self.trec.forward(x_fc_, fbp_fc_, mag_min=mag_min, mag_max=mag_max,
                                              dst_flatten_coords=self.dst_flatten_coords,
                                              img_shape=self.hparams.img_shape,
                                              attenuation=self.mask)

        fc_loss, amp_loss, phi_loss = self.criterion(pred_fc, pred_img, y_fc_, mag_min, mag_max)
        return {'loss': fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def training_epoch_end(self, outputs):
        loss = [d['loss'] for d in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]
        self.log('Train/loss', torch.mean(torch.stack(loss)), logger=True, on_epoch=True)
        self.log('Train/amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def _gt_bin_mse(self, y_fc, y_real, mag_min, mag_max):
        dft_y = convert_to_dft(fc=y_fc, mag_min=mag_min, mag_max=mag_max,
                               dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)
        y_hat = torch.roll(torch.fft.irfftn(dft_y, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                           2 * (self.hparams.img_shape // 2,), (1, 2))

        return F.mse_loss(y_hat, y_real)

    def _val_psnr(self, pred_img, y_real):
        pred_img_norm = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        y_real_norm = denormalize(y_real, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        psnrs = []
        for i in range(len(pred_img_norm)):
            gt = self.circle * y_real_norm[i]
            psnrs.append(PSNR(gt, self.circle * pred_img_norm[i],
                              drange=gt.max() - gt.min()))

        return torch.mean(torch.stack(psnrs))

    def validation_step(self, batch, batch_idx):
        x_fc, fbp_fc, y_fc, y_real, (mag_min, mag_max) = batch
        x_fc_, fbp_fc_, y_fc_ = self._bin_data(x_fc, fbp_fc, y_fc)
        pred_fc, pred_img = self.trec.forward(x_fc_, fbp_fc_, mag_min=mag_min, mag_max=mag_max,
                                              dst_flatten_coords=self.dst_flatten_coords,
                                              img_shape=self.hparams.img_shape,
                                              attenuation=self.mask)

        val_loss, amp_loss, phi_loss = self.criterion(pred_fc, pred_img, y_fc_, mag_min, mag_max)

        val_mse = F.mse_loss(pred_img, y_real)
        val_psnr = self._val_psnr(pred_img, y_real)
        bin_mse = self._gt_bin_mse(y_fc_, y_real, mag_min=mag_min, mag_max=mag_max)
        self.log_dict({'val_loss': val_loss})
        self.log_dict({'val_mse': val_mse})
        self.log_dict({'val_psnr': val_psnr})
        self.log_dict({'bin_mse': bin_mse})
        if batch_idx == 0:
            self.log_val_images(pred_img, fbp_fc[:, self.dst_flatten_coords], y_fc_, y_real, mag_min, mag_max)
        return {'val_loss': val_loss, 'val_mse': val_mse, 'val_psnr': val_psnr, 'bin_mse': bin_mse,
                'amp_loss': amp_loss,
                'phi_loss': phi_loss}

    def log_val_images(self, pred_img, fbp_fc, y_fc, y_real, mag_min, mag_max):
        dft_fbp = convert_to_dft(fc=fbp_fc, mag_min=mag_min, mag_max=mag_max,
                                 dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)
        dft_target = convert_to_dft(fc=y_fc, mag_min=mag_min, mag_max=mag_max,
                                    dst_flatten_coords=self.dst_flatten_coords, img_shape=self.hparams.img_shape)

        for i in range(min(3, len(pred_img))):

            if self.bin_factor == 1:
                fbp_img = torch.roll(torch.fft.irfftn(self.mask * dft_fbp[i], s=2 * (self.hparams.img_shape,)),
                           2 * (self.hparams.img_shape // 2,), (0, 1))
                y_img = y_real[i]
            else:
                fbp_img = torch.roll(torch.fft.irfftn(self.mask * dft_fbp[i],
                                                    s=2 * (self.hparams.img_shape,)),
                                   2 * (self.hparams.img_shape // 2,), (0, 1))
                y_img = torch.roll(torch.fft.irfftn(self.mask * dft_target[i], s=2 * (self.hparams.img_shape,)),
                                   2 * (self.hparams.img_shape // 2,), (0, 1))

            fbp_img = torch.clamp((fbp_img - fbp_img.min()) / (fbp_img.max() - fbp_img.min()), 0, 1)
            pred_img_ = pred_img[i]
            pred_img_ = torch.clamp((pred_img_ - pred_img_.min()) / (pred_img_.max() - pred_img_.min()), 0, 1)
            y_img = torch.clamp((y_img - y_img.min()) / (y_img.max() - y_img.min()), 0, 1)

            self.trainer.logger.experiment.add_image('inputs/img_{}'.format(i), fbp_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('predcitions/img_{}'.format(i), pred_img_.unsqueeze(0),
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('targets/img_{}'.format(i), y_img.unsqueeze(0),
                                                     global_step=self.trainer.global_step)

    def _is_better(self, mean_val_mse):
        return mean_val_mse < self.best_mean_val_mse * (1. - 0.0001)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        val_mse = [o['val_mse'] for o in outputs]
        val_psnr = [o['val_psnr'] for o in outputs]
        bin_mse = [o['bin_mse'] for o in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]
        mean_val_mse = torch.mean(torch.stack(val_mse))
        mean_val_psnr = torch.mean(torch.stack(val_psnr))
        bin_factor_threshold = torch.mean(torch.stack(bin_mse)) * self.bin_factor

        if self._is_better(mean_val_mse):
            self.best_mean_val_mse = mean_val_mse
            self.bin_factor_patience = 10
        else:
            self.bin_factor_patience -= 1

        reduce_bin_factor = (self.bin_factor_patience < 1) or (
                self.bin_count > self.hparams.bin_factor_cd and mean_val_mse < bin_factor_threshold)
        if reduce_bin_factor and self.bin_factor > 1:
            self.bin_count = 0
            self.bin_factor_patience = 10
            self.best_mean_val_mse = mean_val_mse
            self.bin_factor = max(1, self.bin_factor // 2)
            self.register_buffer('mask', psfft(self.bin_factor, pixel_res=self.hparams.img_shape).to(self.device))
            print('Reduced bin_factor to {}.'.format(self.bin_factor))

        if self.bin_factor > 1:
            self.trainer.lr_schedulers[0]['scheduler']._reset()

        self.bin_count += 1

        if self.bin_factor > 1:
            self.trainer.lr_schedulers[0]['scheduler']._reset()

        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_mse', mean_val_mse, logger=True, on_epoch=True)
        self.log('Train/avg_val_psnr', mean_val_psnr, logger=True, on_epoch=True)
        self.log('Train/avg_bin_mse', bin_factor_threshold, logger=True, on_epoch=True)
        self.log('Train/avg_val_amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x_fc, fbp_fc, y, y_real, (mag_min, mag_max) = batch
        assert len(x_fc) == 1, 'Test images have to be evaluated independently.'
        if self.bin_factor != 1:
            print('bin_factor set to 1.')
            self.bin_factor = 1
        x_fc_, fbp_fc_, y_fc_ = self._bin_data(x_fc, fbp_fc, y)

        _, pred_img = self.trec.forward(x_fc_, fbp_fc_, mag_min=mag_min, mag_max=mag_max,
                                        dst_flatten_coords=self.dst_flatten_coords,
                                        img_shape=self.hparams.img_shape,
                                        attenuation=self.mask)

        gt = denormalize(y_real[0], self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img[0], self.trainer.datamodule.mean, self.trainer.datamodule.std)

        gt = self.circle * gt
        return PSNR(gt, self.circle * pred_img, drange=gt.max() - gt.min())

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs)
        self.log('Mean PSNR', torch.mean(outputs).detach().cpu().numpy(), logger=True)
        self.log('SEM PSNR', torch.std(outputs / np.sqrt(len(outputs))).detach().cpu().numpy(),
                 logger=True)
