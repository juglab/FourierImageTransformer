import torch
from pytorch_lightning import LightningModule

from fit.modules.loss import _fc_prod_loss
from fit.transformers.TRecTransformer import TRecTransformer
from fit.utils import PSNR, convert2DFT
from fit.utils.RAdam import RAdam

import numpy as np

from torch.nn import functional as F
import torch.fft

from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi, denormalize_FC

import wandb


class TRecTransformerModule(LightningModule):
    def __init__(self, d_model, sinogram_coords, target_coords, src_flatten_coords,
                 dst_flatten_coords, dst_order, angles, img_shape=27, detector_len=27,
                 init_bin_factor=1,
                 lr=0.0001,
                 t_0=100,
                 weight_decay=0.01,
                 attention_type="linear", n_layers=4, n_heads=4, d_query=4, dropout=0.1, attention_dropout=0.1,
                 d_conv=8):
        super().__init__()

        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "detector_len",
                                  "init_bin_factor",
                                  "lr",
                                  "t_0",
                                  "weight_decay",
                                  "attention_type",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout")
        self.sinogram_coords = sinogram_coords
        self.target_coords = target_coords
        if not type(src_flatten_coords) is torch.Tensor:
            self.src_flatten_coords = torch.from_numpy(src_flatten_coords)
        else:
            self.src_flatten_coords = src_flatten_coords
        if not type(dst_flatten_coords) is torch.Tensor:
            self.dst_flatten_order = torch.from_numpy(dst_flatten_coords)
        else:
            self.dst_flatten_order = dst_flatten_coords
        self.dst_order = dst_order
        self.angles = angles
        self.num_angles = len(self.angles)
        self.dft_shape = (img_shape, img_shape // 2 + 1)
        self.best_mean_val_mse = 9999999
        self.bin_factor = init_bin_factor

        self.loss = _fc_prod_loss

        self.trec = TRecTransformer(d_model=self.hparams.d_model,
                                    coords_sinogram=self.sinogram_coords,
                                    flatten_order_sinogram=self.src_flatten_coords,
                                    coords_target=self.target_coords,
                                    flatten_order_target=self.dst_flatten_order,
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

    def forward(self, x, zero_fbp):
        return self.trec.forward(x, zero_fbp)

    def configure_optimizers(self):
        optimizer = RAdam(self.trec.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.hparams.t_0, T_mult=2,
                                                                         eta_min=self.hparams.lr * 0.01,
                                                                         last_epoch=-1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'interval': 'step',
            "frequency": 1
        }

    def criterion(self, pred_fc, target_fc, amp_min, amp_max):
            fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=amp_min,
                                                    amp_max=amp_max)
            return fc_loss, amp_loss, phi_loss

    def _flatten_data(self, x_fc, fbp_fc, y_fc):

        shells = (self.hparams.detector_len // 2 + 1) / self.bin_factor
        num_sino_fcs = np.clip(self.num_angles * int(shells + 1), 1, x_fc.shape[1])

        if self.bin_factor > 1:
            num_target_fcs = np.sum(self.dst_order <= shells)
        else:
            num_target_fcs = fbp_fc.shape[1]

        x_fc_ = x_fc[:, self.src_flatten_coords][:, :num_sino_fcs]

        fbp_fc_ = fbp_fc[:, self.dst_flatten_order][:, :num_target_fcs]

        y_fc_ = y_fc[:, self.dst_flatten_order][:, :num_target_fcs]
        zero_fbp = fbp_fc_ * 0.
        return x_fc_, zero_fbp, y_fc_

    def training_step(self, batch, batch_idx):
        x_fc, fbp_fc, y_fc, y_real, (amp_min, amp_max) = batch
        x_fc_, zero_fbp, y_fc_ = self._flatten_data(x_fc, fbp_fc, y_fc)

        pred_fc = self.trec.forward(x_fc_, zero_fbp)

        dft_pred_fc = convert2DFT(x=pred_fc, amp_min=amp_min, amp_max=amp_max,
                                  dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
        pred_img = torch.roll(torch.fft.irfftn(dft_pred_fc, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                              2 * (self.hparams.img_shape // 2,), (1, 2))

        mse = F.mse_loss(pred_img, y_real)
        fc_loss, amp_loss, phi_loss = self.criterion(pred_fc, y_fc_, amp_min, amp_max)
        return {'loss': mse + fc_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def training_epoch_end(self, outputs):
        loss = [d['loss'] for d in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]
        self.log('Train/loss', torch.mean(torch.stack(loss)), logger=True, on_epoch=True)
        self.log('Train/amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def _gt_bin_mse(self, y_fc, y_real, amp_min, amp_max):
        dft_y = convert2DFT(x=y_fc, amp_min=amp_min, amp_max=amp_max,
                            dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
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
        x_fc, fbp_fc, y_fc, y_real, (amp_min, amp_max) = batch
        x_fc_, zero_fbp, y_fc_ = self._flatten_data(x_fc, fbp_fc, y_fc)
        pred_fc = self.trec.forward(x_fc_, zero_fbp)

        val_loss, amp_loss, phi_loss = self.criterion(pred_fc, y_fc_, amp_min, amp_max)

        pred_fc = self.trec.forward(x_fc_, zero_fbp)

        dft_pred_fc = convert2DFT(x=pred_fc, amp_min=amp_min, amp_max=amp_max,
                                  dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
        pred_img = torch.roll(torch.fft.irfftn(dft_pred_fc, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                                          2 * (self.hparams.img_shape // 2,), (1, 2))

        val_mse = F.mse_loss(pred_img, y_real)
        val_psnr = self._val_psnr(pred_img, y_real)
        bin_mse = self._gt_bin_mse(y_fc_, y_real, amp_min=amp_min, amp_max=amp_max)
        self.log_dict({'val_loss': val_loss + val_mse})
        self.log_dict({'val_mse': val_mse})
        self.log_dict({'val_psnr': val_psnr})
        self.log_dict({'bin_mse': bin_mse})
        if batch_idx == 0:
            self.log_val_images(pred_img, fbp_fc[:, self.dst_flatten_order], y_fc_, y_real, amp_min, amp_max)
        return {'val_loss': val_loss, 'val_mse': val_mse, 'val_psnr': val_psnr, 'bin_mse': bin_mse,
                'amp_loss': amp_loss,
                'phi_loss': phi_loss}

    def log_val_images(self, pred_img, fbp_fc, y_fc, y_real, amp_min, amp_max):
        dft_fbp = convert2DFT(x=fbp_fc, amp_min=amp_min, amp_max=amp_max,
                              dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
        # dft_target = convert2DFT(x=y_fc, amp_min=amp_min, amp_max=amp_max,
        #                          dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)

        fbp_imgs = []
        pred_imgs = []
        target_imgs = []
        for i in range(min(3, len(pred_img))):

            fbp_img = torch.roll(torch.fft.irfftn(dft_fbp[i], s=2 * (self.hparams.img_shape,)),
                                 2 * (self.hparams.img_shape // 2,), (0, 1))
            y_img = y_real[i]

            fbp_img = torch.clamp((fbp_img - fbp_img.min()) * 255 / (fbp_img.max() - fbp_img.min()), 0, 255)
            pred_img_ = pred_img[i]
            pred_img_ = torch.clamp((pred_img_ - pred_img_.min()) * 255 / (pred_img_.max() - pred_img_.min()), 0, 255)
            y_img = torch.clamp((y_img - y_img.min()) * 255 / (y_img.max() - y_img.min()), 0, 255)
            
            fbp_imgs.append(wandb.Image(fbp_img.detach().cpu().numpy().astype(np.uint8), mode='L'))
            pred_imgs.append(wandb.Image(pred_img_.detach().cpu().numpy().astype(np.uint8), mode='L'))
            target_imgs.append(wandb.Image(y_img.detach().cpu().numpy().astype(np.uint8), mode='L'))
            
        self.trainer.logger.log_image(key="FBP", images=fbp_imgs)
        self.trainer.logger.log_image(key="Prediction", images=pred_imgs)
        self.trainer.logger.log_image(key="Ground Truth", images=target_imgs)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        val_mse = [o['val_mse'] for o in outputs]
        val_psnr = [o['val_psnr'] for o in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]
        mean_val_mse = torch.mean(torch.stack(val_mse))
        mean_val_psnr = torch.mean(torch.stack(val_psnr))

        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_mse', mean_val_mse, logger=True, on_epoch=True)
        self.log('Train/avg_val_psnr', mean_val_psnr, logger=True, on_epoch=True)
        self.log('Train/avg_val_amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x_fc, fbp_fc, y, y_real, (amp_min, amp_max) = batch
        assert len(x_fc) == 1, 'Test images have to be evaluated independently.'
        x_fc_, zero_fbp, y_fc_ = self._flatten_data(x_fc, fbp_fc, y)

        pred_fc = self.trec.forward(x_fc_, zero_fbp)
        dft_pred_fc = convert2DFT(x=pred_fc, amp_min=amp_min, amp_max=amp_max,
                                  dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
        pred_img = torch.roll(torch.fft.irfftn(dft_pred_fc, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                              2 * (self.hparams.img_shape // 2,), (1, 2))

        gt = denormalize(y_real[0], self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img[0], self.trainer.datamodule.mean, self.trainer.datamodule.std)

        gt = self.circle * gt
        return PSNR(gt, self.circle * pred_img, drange=gt.max() - gt.min())

    def test_epoch_end(self, outputs):
        outputs = torch.stack(outputs)
        print(torch.mean(outputs).detach().cpu().numpy())
        self.log('Mean PSNR', torch.mean(outputs), logger=True)
        self.log('SEM PSNR', torch.std(outputs / np.sqrt(len(outputs))),
                 logger=True)

    def get_imgs(self, x, fbp, y, amp_min, amp_max):
        self.eval()
        x_fc_, zero_fbp, y_fc_ = self._flatten_data(x, fbp, y)

        pred_fc = self.trec.forward(x_fc_, zero_fbp)

        pred_fc = denormalize_FC(pred_fc, amp_min=amp_min, amp_max=amp_max)

        dft_pred_fc = convert2DFT(x=pred_fc, amp_min=amp_min, amp_max=amp_max,
                                  dst_flatten_order=self.dst_flatten_order, img_shape=self.hparams.img_shape)
        img_pred_before_conv = torch.roll(torch.fft.irfftn(dft_pred_fc, dim=[1, 2], s=2 * (self.hparams.img_shape,)),
                                          2 * (self.hparams.img_shape // 2,), (1, 2))

        return img_pred_before_conv
