import numpy as np
import torch
import torch.fft
from skimage.transform import iradon
from torch.utils.data import Dataset


class BaselineDataset(Dataset):
    def __init__(self, ds, mean, std, part='train', img_shape=42):
        self.ds = ds.create_torch_dataset(part=part)
        self.img_shape = img_shape
        self.angles = ds.ray_trafo.geometry.angles
        if mean == None and std == None:
            tmp_recos = []
            for i in np.random.permutation(len(self.ds))[:200]:
                sino, _ = self.ds[i]
                reco = iradon(sino.numpy().T, theta=-np.rad2deg(self.angles), circle=True,
                       filter_name='cosine').T
                tmp_recos.append(torch.from_numpy(reco))

            tmp_recos = torch.stack(tmp_recos)
            self.mean = tmp_recos.mean()
            self.std = tmp_recos.std()
        else:
            self.mean = mean
            self.std = std

    def __getitem__(self, item):
        sino, img = self.ds[item]
        reco = iradon(sino.numpy().T, theta=-np.rad2deg(self.angles), circle=True,
                      filter_name='cosine').T
        reco = torch.from_numpy(reco)
        reco = (reco - self.mean)/self.std
        return reco.unsqueeze(0), img.unsqueeze(0)

    def __len__(self):
        return len(self.ds)
