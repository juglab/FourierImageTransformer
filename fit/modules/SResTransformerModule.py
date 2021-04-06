import torch
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fit.modules.loss import _fc_prod_loss, _fc_sum_loss
from fit.transformers.SResTransformer import SResTransformerTrain, SResTransformerPredict
from fit.utils import denormalize_FC, PSNR, convert2DFT
from fit.utils.RAdam import RAdam

import numpy as np

import torch.fft

from fit.utils.utils import denormalize, denormalize_amp, denormalize_phi


class SResTransformerModule(LightningModule):
    def __init__(self, d_model, img_shape,
                 coords, dst_flatten_order, dst_order,
                 loss='prod',
                 lr=0.0001,
                 weight_decay=0.01,
                 n_layers=4, n_heads=4, d_query=4, dropout=0.1, attention_dropout=0.1):
        super().__init__()

        self.save_hyperparameters("d_model",
                                  "img_shape",
                                  "loss",
                                  "lr",
                                  "weight_decay",
                                  "n_layers",
                                  "n_heads",
                                  "d_query",
                                  "dropout",
                                  "attention_dropout")

        self.coords = coords
        self.dst_flatten_order = dst_flatten_order
        self.dst_order = dst_order
        self.dft_shape = (img_shape, img_shape // 2 + 1)

        if loss == 'prod':
            self.loss = _fc_prod_loss
        else:
            self.loss = _fc_sum_loss

        self.sres = SResTransformerTrain(d_model=self.hparams.d_model,
                                         coords=self.coords,
                                         flatten_order=self.dst_flatten_order,
                                         attention_type='causal-linear',
                                         n_layers=self.hparams.n_layers,
                                         n_heads=self.hparams.n_heads,
                                         d_query=self.hparams.d_query,
                                         dropout=self.hparams.dropout,
                                         attention_dropout=self.hparams.attention_dropout)
        self.sres_pred = None
        x, y = np.meshgrid(range(self.dft_shape[1]), range(-self.dft_shape[0] // 2, self.dft_shape[0] // 2 + 1))
        radii = np.roll(np.sqrt(x ** 2 + y ** 2, dtype=np.float32), self.dft_shape[0] // 2 + 1, 0)
        num_shells = 4
        self.input_seq_length = np.sum(np.round(radii) < num_shells)

    def forward(self, x):
        return self.sres.forward(x)

    def configure_optimizers(self):
        optimizer = RAdam(self.sres.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'Train/avg_val_loss'
        }

    def criterion(self, pred_fc, target_fc, mag_min, mag_max):
        fc_loss, amp_loss, phi_loss = self.loss(pred_fc=pred_fc, target_fc=target_fc, amp_min=mag_min,
                                                amp_max=mag_max)
        return fc_loss, amp_loss, phi_loss

    def training_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]

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
        x_fc = fc[:, self.dst_flatten_order][:, :-1]
        y_fc = fc[:, self.dst_flatten_order][:, 1:]

        pred = self.sres.forward(x_fc)

        val_loss, amp_loss, phi_loss = self.criterion(pred, y_fc, mag_min, mag_max)
        if batch_idx == 0:
            self.log_val_images(fc, mag_min, mag_max)
        self.log_dict({'val_loss': val_loss})
        return {'val_loss': val_loss, 'amp_loss': amp_loss, 'phi_loss': phi_loss}

    def log_val_images(self, fc, mag_min, mag_max):
        self.load_test_model(self.trainer.checkpoint_callback.last_model_path)
        lowres, pred, gt = self.get_lowres_pred_gt(fc, mag_min=mag_min,
                                                   mag_max=mag_max)
        for i in range(min(3, len(lowres))):
            lowres_ = torch.clamp((lowres[i].unsqueeze(0) - lowres.min()) / (lowres.max() - lowres.min()), 0, 1)
            pred_ = torch.clamp((pred[i].unsqueeze(0) - pred.min()) / (pred.max() - pred.min()), 0, 1)
            gt_ = torch.clamp((gt[i].unsqueeze(0) - gt.min()) / (gt.max() - gt.min()), 0, 1)

            self.trainer.logger.experiment.add_image('inputs/img_{}'.format(i), lowres_,
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('predcitions/img_{}'.format(i), pred_,
                                                     global_step=self.trainer.global_step)
            self.trainer.logger.experiment.add_image('targets/img_{}'.format(i), gt_,
                                                     global_step=self.trainer.global_step)

    def validation_epoch_end(self, outputs):
        val_loss = [o['val_loss'] for o in outputs]
        amp_loss = [d['amp_loss'] for d in outputs]
        phi_loss = [d['phi_loss'] for d in outputs]

        self.log('Train/avg_val_loss', torch.mean(torch.stack(val_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_amp_loss', torch.mean(torch.stack(amp_loss)), logger=True, on_epoch=True)
        self.log('Train/avg_val_phi_loss', torch.mean(torch.stack(phi_loss)), logger=True, on_epoch=True)

    def load_test_model(self, path):
        self.sres_pred = SResTransformerPredict(self.hparams.d_model,
                                                coords=self.coords,
                                                flatten_order=self.dst_flatten_order,
                                                attention_type='causal-linear',
                                                n_layers=self.hparams.n_layers,
                                                n_heads=self.hparams.n_heads,
                                                d_query=self.hparams.d_query,
                                                dropout=self.hparams.dropout,
                                                attention_dropout=self.hparams.attention_dropout)
        if len(path) > 0:
            weights = torch.load(path)
            sd = {}
            for k in weights['state_dict'].keys():
                if k[:5] == 'sres.':
                    sd[k[5:]] = weights['state_dict'][k]
            self.sres_pred.load_state_dict(sd)

        self.sres_pred.to(self.device)

    def predict_with_recurrent(self, fcs, n, seq_len):
        memory = None
        y_hat = []
        x_hat = []

        with torch.no_grad():
            for i in range(n):
                x_hat.append(fcs[:, i])
                yi, memory = self.sres_pred(x_hat[-1], i=i, memory=memory)
                y_hat.append(yi)

            for i in range(n, seq_len - 1):
                x_hat.append(y_hat[-1])
                yi, memory = self.sres_pred(x_hat[-1], i=i, memory=memory)
                y_hat.append(yi)

            x_hat.append(y_hat[-1])
            x_hat = torch.stack(x_hat, dim=1)

        return x_hat

    def convert2img(self, fc, mag_min, mag_max):
        dft = convert2DFT(x=fc, amp_min=mag_min, amp_max=mag_max, dst_flatten_order=self.dst_flatten_order,
                          img_shape=self.hparams.img_shape)
        return torch.fft.irfftn(dft, s=2 * (self.hparams.img_shape,), dim=[1, 2])

    def test_step(self, batch, batch_idx):
        fc, (mag_min, mag_max) = batch
        lowres_img, pred_img, gt_img = self.get_lowres_pred_gt(fc=fc, mag_min=mag_min, mag_max=mag_max)

        lowres_img = denormalize(lowres_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        pred_img = denormalize(pred_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)
        gt_img = denormalize(gt_img, self.trainer.datamodule.mean, self.trainer.datamodule.std)

        lowres_psnr = [PSNR(gt_img[i], lowres_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                       range(gt_img.shape[0])]
        pred_psnr = [PSNR(gt_img[i], pred_img[i], drange=torch.tensor(255., dtype=torch.float32)) for i in
                     range(gt_img.shape[0])]
        return (lowres_psnr, pred_psnr)

    def get_lowres_pred_gt(self, fc, mag_min, mag_max):
        x_fc = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        pred = self.predict_with_recurrent(x_fc, self.input_seq_length, fc.shape[1])
        pred_img = self.convert2img(fc=pred, mag_min=mag_min, mag_max=mag_max)
        lowres = torch.zeros_like(pred)
        lowres += fc.min()
        lowres[:, :self.input_seq_length] = fc[:, self.dst_flatten_order][:, :self.input_seq_length]
        lowres_img = self.convert2img(fc=lowres, mag_min=mag_min, mag_max=mag_max)
        gt_img = self.convert2img(fc=fc[:, self.dst_flatten_order], mag_min=mag_min, mag_max=mag_max)
        return lowres_img, pred_img, gt_img

    def test_epoch_end(self, outputs):
        lowres_psnrs = torch.cat([torch.stack(o[0]) for o in outputs])
        pred_psnrs = torch.cat([torch.stack(o[1]) for o in outputs])
        self.log('Input Mean PSNR', torch.mean(lowres_psnrs).detach().cpu().numpy(), logger=True)
        self.log('Input SEM PSNR', torch.std(lowres_psnrs / np.sqrt(len(lowres_psnrs))).detach().cpu().numpy(),
                 logger=True)
        self.log('Prediction Mean PSNR', torch.mean(pred_psnrs).detach().cpu().numpy(), logger=True)
        self.log('Prediction SEM PSNR', torch.std(pred_psnrs / np.sqrt(len(pred_psnrs))).detach().cpu().numpy(),
                 logger=True)
