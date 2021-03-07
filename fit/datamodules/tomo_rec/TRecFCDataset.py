import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset


class TRecFourierCoefficientDataset(Dataset):
    def __init__(self, ds, mag_min, mag_max, part='train', img_shape=42):
        self.ds = ds.create_torch_dataset(part=part)
        self.img_shape = img_shape
        self.angles = ds.ray_trafo.geometry.angles
        if mag_min == None and mag_max == None:
            tmp_sinos = []
            for i in np.random.permutation(len(self.ds))[:200]:
                sino, _ = self.ds[i]
                tmp_sinos.append(sino)

            tmp_sinos = torch.stack(tmp_sinos)
            tmp_sinos = torch.fft.rfftn(tmp_sinos, dim=[1, 2]).abs()
            tmp_sinos[tmp_sinos == 0] = 1.
            tmp_sinos = torch.log(tmp_sinos)
            self.mag_min = tmp_sinos.min()
            self.mag_max = tmp_sinos.max()
        else:
            self.mag_min = mag_min
            self.mag_max = mag_max

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

        sino_mag = 2 * (sino_mag - self.mag_min) / (self.mag_max - self.mag_min) - 1
        img_mag = 2 * (img_mag - self.mag_min) / (self.mag_max - self.mag_min) - 1

        sino_phi = sino_phi / np.pi
        img_phi = img_phi / np.pi

        sino_fft = torch.stack([sino_mag.flatten(), sino_phi.flatten()], dim=-1)
        img_fft = torch.stack([img_mag.flatten(), img_phi.flatten()], dim=-1)
        return sino_fft, img_fft, img, (self.mag_min.unsqueeze(-1), self.mag_max.unsqueeze(-1))

    def __len__(self):
        return len(self.ds)
