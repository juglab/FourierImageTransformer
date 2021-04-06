import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset

from fit.utils.utils import normalize_FC, log_amplitudes


class SResFourierCoefficientDataset(Dataset):
    def __init__(self, ds, amp_min, amp_max):
        self.ds = ds
        if amp_min == None and amp_max == None:
            tmp_imgs = []
            for i in np.random.permutation(len(self.ds))[:200]:
                img = self.ds[i]
                tmp_imgs.append(img)

            tmp_imgs = torch.stack(tmp_imgs)
            tmp_ffts = torch.fft.rfftn(tmp_imgs, dim=[1, 2])
            log_amps = log_amplitudes(tmp_ffts.abs())
            self.amp_min = log_amps.min()
            self.amp_max = log_amps.max()
        else:
            self.amp_min = amp_min
            self.amp_max = amp_max

    def __getitem__(self, item):
        img = self.ds[item]
        img_fft = torch.fft.rfftn(img, dim=[0, 1])

        img_amp, img_phi = normalize_FC(img_fft, amp_min=self.amp_min, amp_max=self.amp_max)

        img_fft = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1)
        return img_fft, (self.amp_min.unsqueeze(-1), self.amp_max.unsqueeze(-1))

    def __len__(self):
        return len(self.ds)
