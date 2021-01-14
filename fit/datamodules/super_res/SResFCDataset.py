import numpy as np
import torch
import torch.fft
from torch.utils.data import Dataset


class SResFourierCoefficientDataset(Dataset):
    def __init__(self, ds, mag_min, mag_max, part='train', img_shape=42):
        self.ds = ds.create_torch_dataset(part=part)
        self.img_shape = img_shape
        if mag_min == None and mag_max == None:
            tmp_imgs = []
            for i in np.random.permutation(len(self.ds))[:200]:
                img = self.ds[i]
                tmp_imgs.append(img)

            tmp_imgs = torch.stack(tmp_imgs)
            tmp_imgs = torch.fft.rfftn(tmp_imgs, dim=[1, 2]).abs()
            tmp_imgs[tmp_imgs == 0] = 1.
            tmp_imgs = torch.log(tmp_imgs)
            self.mag_min = tmp_imgs.min()
            self.mag_max = tmp_imgs.max()
        else:
            self.mag_min = mag_min
            self.mag_max = mag_max

    def __getitem__(self, item):
        img = self.ds[item]
        img_fft = torch.fft.rfftn(img, dim=[0, 1])

        img_mag = img_fft.abs()
        img_mag[img_mag == 0] = 1.
        img_mag = torch.log(img_mag)
        img_phi = img_fft.angle()

        img_mag = 2 * (img_mag - self.mag_min) / (self.mag_max - self.mag_min) - 1

        img_phi = 2 * img_phi / (2 * np.pi) - 1

        img_fft = torch.stack([img_mag.flatten(), img_phi.flatten()], dim=-1)
        return img_fft, (self.mag_min.unsqueeze(-1), self.mag_max.unsqueeze(-1))

    def __len__(self):
        return len(self.ds)
