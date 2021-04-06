import numpy as np
import torch
import torch.fft
from skimage.transform import iradon
from torch.utils.data import Dataset

from fit.utils.utils import log_amplitudes, normalize_FC


class TRecFourierCoefficientDataset(Dataset):
    def __init__(self, ds, angles, mag_min, mag_max, img_shape=42, inner_circle=True):
        self.ds = ds
        self.img_shape = img_shape
        self.inner_circle = inner_circle
        self.angles = angles
        if mag_min == None and mag_max == None:
            tmp_sinos = []
            for i in np.random.permutation(len(self.ds))[:200]:
                sino, _ = self.ds[i]
                tmp_sinos.append(sino)

            tmp_sinos = torch.stack(tmp_sinos)
            tmp_ffts = torch.fft.rfftn(tmp_sinos, dim=[1, 2])
            tmp_amps = log_amplitudes(tmp_ffts.abs())
            self.amp_min = tmp_amps.min()
            self.amp_max = tmp_amps.max()
        else:
            self.amp_min = mag_min
            self.amp_max = mag_max

    def __getitem__(self, item):
        sino, img = self.ds[item]
        fbp = torch.from_numpy(
            np.array(iradon(sino.numpy().T, theta=np.rad2deg(-self.angles), circle=self.inner_circle,
                            output_size=self.img_shape).astype(np.float32).T))
        sino_fft = torch.fft.rfftn(torch.roll(sino, sino.shape[1] // 2 + 1, 1), dim=[-1])
        fbp_fft = torch.fft.rfftn(torch.roll(fbp, 2 * (img.shape[0] // 2 + 1,), (0, 1)), dim=[0, 1])
        img_fft = torch.fft.rfftn(torch.roll(img, 2 * (img.shape[0] // 2 + 1,), (0, 1)), dim=[0, 1])

        sino_amp, sino_phi = normalize_FC(sino_fft, amp_min=self.amp_min, amp_max=self.amp_max)
        fbp_amp, fbp_phi = normalize_FC(fbp_fft, amp_min=self.amp_min, amp_max=self.amp_max)
        img_amp, img_phi = normalize_FC(img_fft, amp_min=self.amp_min, amp_max=self.amp_max)

        sino_fc = torch.stack([sino_amp.flatten(), sino_phi.flatten()], dim=-1)
        fbp_fc = torch.stack([fbp_amp.flatten(), fbp_phi.flatten()], dim=-1)
        img_fc = torch.stack([img_amp.flatten(), img_phi.flatten()], dim=-1)
        return sino_fc, fbp_fc, img_fc, img, (self.amp_min.unsqueeze(-1), self.amp_max.unsqueeze(-1))

    def __len__(self):
        return len(self.ds)
