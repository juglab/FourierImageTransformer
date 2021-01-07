import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset


class FCDataset(Dataset):
    def __init__(self, ds, part='train', img_shape=42):
        self.ds = ds.create_torch_dataset(part=part)
        self.img_shape = img_shape
        self.angles = ds.ray_trafo.geometry.angles

    def __getitem__(self, i):
        raise NotImplementedError()

    def __len__(self):
        return len(self.ds)


class FourierCoefficientDataset(FCDataset):
    def __init__(self, ds, part='train', img_shape=42):
        super().__init__(ds, part=part, img_shape=img_shape)

    def __getitem__(self, item):
        sino, img = self.ds[item]
        sino_fft = torch.fft.rfftn(torch.roll(sino, sino.shape[1] // 2 + 1, 1), dim=[-1])
        img_fft = torch.fft.rfftn(torch.roll(img, 2 * (img.shape[0] // 2 + 1,), (0, 1)), dim=[0, 1])

        sino_mag = sino_fft.abs()
        sino_mag[sino_mag == 0] = 1.
        sino_mag = torch.log(sino_mag)
        sino_phi = sino_fft.angle()

        img_mag = img_fft.abs()
        img_mag[img_mag == 0] = 1.
        img_mag = torch.log(img_mag)
        img_phi = img_fft.angle()

        mag_min, mag_max = sino_mag.min(), sino_mag.max()

        sino_mag = (sino_mag - mag_min) / (mag_max - mag_min)
        img_mag = (img_mag - mag_min) / (mag_max - mag_min)

        sino_phi = sino_phi / (2 * np.pi)
        img_phi = img_phi / (2 * np.pi)

        sino_fft = torch.stack([sino_mag.flatten(), sino_phi.flatten()], dim=-1)
        img_fft = torch.stack([img_mag.flatten(), img_phi.flatten()], dim=-1)
        return sino_fft, img_fft, img, (mag_min.unsqueeze(-1), mag_max.unsqueeze(-1))

