from typing import Optional, Union, List

import dival
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit.datamodules.baselines.BaselineDataset import BaselineDataset
from fit.datamodules.tomo_rec.TRecDataModule import get_projection_dataset
from fit.datamodules.tomo_rec.TRecFCDataset import TRecFourierCoefficientDataset
from fit.datamodules.GroundTruthDataset import GroundTruthDataset
import odl
from skimage.transform import resize

from fit.utils.tomo_utils import get_detector_length
from fit.utils.utils import normalize




class MNISTBaselineDataModule(LightningDataModule):
    IMG_SHAPE = 27

    def __init__(self, root_dir, batch_size, num_angles=15, inner_circle=True):
        """
        :param root_dir:
        :param batch_size:
        :param num_angles:
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_angles = num_angles
        self.inner_circle = inner_circle
        self.gt_ds = None
        self.mean = None
        self.std = None
        self.mag_min = None
        self.mag_max = None

    def setup(self, stage: Optional[str] = None):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data.type(torch.float32)
        mnist_train_val = MNIST(self.root_dir, train=True, download=True).data.type(torch.float32)
        np.random.seed(1612)
        perm = np.random.permutation(mnist_train_val.shape[0])
        mnist_train = mnist_train_val[perm[:55000], 1:, 1:]
        mnist_val = mnist_train_val[perm[55000:], 1:, 1:]
        mnist_test = mnist_test[:, 1:, 1:]

        assert mnist_train.shape[1] == MNISTBaselineDataModule.IMG_SHAPE
        assert mnist_train.shape[2] == MNISTBaselineDataModule.IMG_SHAPE
        x, y = torch.meshgrid(torch.arange(-MNISTBaselineDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTBaselineDataModule.IMG_SHAPE // 2 + 1),
                              torch.arange(-MNISTBaselineDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTBaselineDataModule.IMG_SHAPE // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= MNISTBaselineDataModule.IMG_SHAPE // 2
        mnist_train = circle * np.clip(mnist_train, 50, 255)
        mnist_val = circle * np.clip(mnist_val, 50, 255)
        mnist_test = circle * np.clip(mnist_test, 50, 255)

        self.mean = mnist_train.mean()
        self.std = mnist_train.std()

        mnist_train = normalize(mnist_train, self.mean, self.std)
        mnist_val = normalize(mnist_val, self.mean, self.std)
        mnist_test = normalize(mnist_test, self.mean, self.std)
        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(mnist_train, mnist_val, mnist_test),
            num_angles=self.num_angles, im_shape=70, impl='astra_cpu', inner_circle=self.inner_circle)

        tmp_ds = BaselineDataset(self.gt_ds, mean=None, std=None, part='train',
                                                 img_shape=MNISTBaselineDataModule.IMG_SHAPE)
        self.mag_min = tmp_ds.mean
        self.mag_max = tmp_ds.std

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            BaselineDataset(self.gt_ds, mean=self.mean, std=self.std, part='train',
                                          img_shape=MNISTBaselineDataModule.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            BaselineDataset(self.gt_ds, mean=self.mean, std=self.std, part='validation',
                                          img_shape=MNISTBaselineDataModule.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            BaselineDataset(self.gt_ds, mean=self.mean, std=self.std, part='test',
                                          img_shape=MNISTBaselineDataModule.IMG_SHAPE),
            batch_size=1)
