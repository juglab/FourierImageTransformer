from typing import Optional, Union, List

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit.datamodules.tomo_rec.FCDataset import FourierCoefficientDataset
from fit.datamodules.tomo_rec.GroundTruthDataset import GroundTruthDataset
from fit.utils.tomo_utils import get_projection_dataset


class MNISTTomoFourierTargetDataModule(LightningDataModule):
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

    def setup(self, stage: Optional[str] = None):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data
        mnist_train_test = MNIST(self.root_dir, train=True, download=True).data
        np.random.seed(1612)
        perm = np.random.permutation(mnist_train_test.shape[0])
        mnist_train = mnist_train_test[perm[:55000], 1:, 1:]
        mnist_val = mnist_train_test[perm[55000:], 1:, 1:]
        mnist_test = mnist_test[:, 1:, 1:]
        assert mnist_train.shape[1] == MNISTTomoFourierTargetDataModule.IMG_SHAPE
        assert mnist_train.shape[2] == MNISTTomoFourierTargetDataModule.IMG_SHAPE
        x, y = torch.meshgrid(torch.arange(-MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1),
                              torch.arange(-MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1,
                                           MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= MNISTTomoFourierTargetDataModule.IMG_SHAPE // 2
        mnist_train = circle * np.clip(mnist_train, 50, 255)
        mnist_val = circle * np.clip(mnist_val, 50, 255)
        mnist_test = circle * np.clip(mnist_test, 50, 255)
        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(mnist_train, mnist_val, mnist_test),
            num_angles=self.num_angles, im_shape=70, impl='astra_cpu', inner_circle=self.inner_circle)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            FourierCoefficientDataset(self.gt_ds, part='train', img_shape=MNISTTomoFourierTargetDataModule.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            FourierCoefficientDataset(self.gt_ds, part='validation',
                                      img_shape=MNISTTomoFourierTargetDataModule.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            FourierCoefficientDataset(self.gt_ds, part='test', img_shape=MNISTTomoFourierTargetDataModule.IMG_SHAPE),
            batch_size=1)
