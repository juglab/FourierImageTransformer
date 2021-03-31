from os.path import join, exists
from typing import Optional, Union, List

import numpy as np
import torch
import wget
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit.datamodules.super_res import SResFourierCoefficientDataset
from fit.datamodules.GroundTruthDatasetFactory import GroundTruthDatasetFactory
from fit.utils.utils import normalize


class SResFITDataModule(LightningDataModule):
    def __init__(self, root_dir, batch_size, gt_shape):
        """

        :param root_dir:
        :param batch_size:
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.gt_shape = gt_shape
        self.gt_ds = None
        self.mean = None
        self.std = None
        self.mag_min = None
        self.mag_max = None

    def setup(self, stage: Optional[str] = None):
        tmp_fcds = SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'), amp_min=None,
                                                 amp_max=None)
        self.mag_min = tmp_fcds.amp_min
        self.mag_max = tmp_fcds.amp_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'), amp_min=self.mag_min,
                                          amp_max=self.mag_max),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='validation'), amp_min=self.mag_min,
                                          amp_max=self.mag_max),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            SResFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='test'), amp_min=self.mag_min,
                                          amp_max=self.mag_max),
            batch_size=self.batch_size)


class MNIST_SResFITDM(SResFITDataModule):

    def __init__(self, root_dir, batch_size):
        """
        Uses the MNIST[1] dataset via the PyTorch API.

        :param root_dir:
        :param batch_size:

        References:
            [1] Yann LeCun and Corinna Cortes.
            MNIST handwritten digit database. 2010.
        """
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=27)

    def prepare_data(self, *args, **kwargs):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data.type(torch.float32)
        mnist_train_val = MNIST(self.root_dir, train=True, download=True).data.type(torch.float32)
        np.random.seed(1612)
        perm = np.random.permutation(mnist_train_val.shape[0])
        mnist_train = mnist_train_val[perm[:55000], 1:, 1:]
        mnist_val = mnist_train_val[perm[55000:], 1:, 1:]
        mnist_test = mnist_test[:, 1:, 1:]

        self.mean = mnist_train.mean()
        self.std = mnist_train.std()

        mnist_train = normalize(mnist_train, self.mean, self.std)
        mnist_val = normalize(mnist_val, self.mean, self.std)
        mnist_test = normalize(mnist_test, self.mean, self.std)

        self.gt_ds = GroundTruthDatasetFactory(mnist_train, mnist_val, mnist_test)


class CelebA_SResFITDM(SResFITDataModule):

    def __init__(self, root_dir, batch_size):
        """
        Uses the CelebA[1] dataset.

        :param root_dir:
        :param batch_size:

        References:
            [1] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.
            Deep learning face attributes in the wild.
            In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
        """
        super().__init__(root_dir=root_dir, batch_size=batch_size, gt_shape=63)

    def prepare_data(self, *args, **kwargs):
        if not exists(join(self.root_dir, 'gt_data.npz')):
            wget.download('https://cloud.mpi-cbg.de/index.php/s/Wtuy9IqUsSpjKav/download',
                          out=join(self.root_dir, 'gt_data.npz'))

        gt_data = np.load(join(self.root_dir, 'gt_data.npz'))

        gt_train = torch.from_numpy(gt_data['gt_train'])
        gt_val = torch.from_numpy(gt_data['gt_val'])
        gt_test = torch.from_numpy(gt_data['gt_test'])
        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)
        self.gt_ds = GroundTruthDatasetFactory(gt_train, gt_val, gt_test)
