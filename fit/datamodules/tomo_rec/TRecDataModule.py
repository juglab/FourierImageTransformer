from os.path import join, exists
from typing import Optional, Union, List

import dival
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit.datamodules.tomo_rec.TRecFCDataset import TRecFourierCoefficientDataset
from fit.datamodules.GroundTruthDatasetFactory import GroundTruthDatasetFactory
from skimage.transform import resize

from fit.utils.utils import normalize

import wget


class TomoFITDataModule(LightningDataModule):
    def __init__(self, root_dir, gt_shape, batch_size, num_angles=15):
        """

        :param root_dir: path to downloaded data
        :param gt_shape: size of the ground truth data
        :param batch_size:
        :param num_angles: for projection
        """
        super().__init__()
        self.root_dir = root_dir
        self.gt_shape = gt_shape
        self.batch_size = batch_size
        self.num_angles = num_angles
        self.inner_circle = True
        self.gt_ds = None
        self.mean = None
        self.std = None
        self.mag_min = None
        self.mag_max = None

    def __get_circle__(self):
        x, y = torch.meshgrid(torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1),
                              torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1))
        return torch.sqrt(x ** 2. + y ** 2.) <= self.gt_shape // 2

    def prepare_data(self, *args, **kwargs):
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'),
                                                 angles=self.gt_ds.ray_trafo.geometry.angles, mag_min=None,
                                                 mag_max=None, img_shape=self.gt_shape)
        self.mag_min = tmp_fcds.amp_min
        self.mag_max = tmp_fcds.amp_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='train'),
                                          angles=self.gt_ds.ray_trafo.geometry.angles, mag_min=self.mag_min,
                                          mag_max=self.mag_max, img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='validation'),
                                          angles=self.gt_ds.ray_trafo.geometry.angles, mag_min=self.mag_min,
                                          mag_max=self.mag_max, img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds.create_torch_dataset(part='test'),
                                          angles=self.gt_ds.ray_trafo.geometry.angles, mag_min=self.mag_min,
                                          mag_max=self.mag_max, img_shape=self.gt_shape),
            batch_size=1)


class MNIST_TRecFITDM(TomoFITDataModule):

    def __init__(self, root_dir, batch_size, num_angles=15):
        """
        Uses the MNIST[1] dataset via the PyTorch API.

        :param root_dir: path to downloaded data
        :param batch_size:
        :param num_angles: for projection

        References:
            [1] Yann LeCun and Corinna Cortes.
            MNIST handwritten digit database. 2010.
        """
        super().__init__(root_dir=root_dir, gt_shape=27, batch_size=batch_size, num_angles=num_angles)

    def prepare_data(self, *args, **kwargs):
        mnist_test = MNIST(self.root_dir, train=False, download=True).data.type(torch.float32)
        mnist_train_val = MNIST(self.root_dir, train=True, download=True).data.type(torch.float32)
        np.random.seed(1612)
        perm = np.random.permutation(mnist_train_val.shape[0])
        mnist_train = mnist_train_val[perm[:55000], 1:, 1:]
        mnist_val = mnist_train_val[perm[55000:], 1:, 1:]
        mnist_test = mnist_test[:, 1:, 1:]

        assert mnist_train.shape[1] == self.gt_shape
        assert mnist_train.shape[2] == self.gt_shape

        circle = self.__get_circle__()
        mnist_train = circle * np.clip(mnist_train, 50, 255)
        mnist_val = circle * np.clip(mnist_val, 50, 255)
        mnist_test = circle * np.clip(mnist_test, 50, 255)

        self.mean = mnist_train.mean()
        self.std = mnist_train.std()

        mnist_train = normalize(mnist_train, self.mean, self.std)
        mnist_val = normalize(mnist_val, self.mean, self.std)
        mnist_test = normalize(mnist_test, self.mean, self.std)

        mnist_train *= circle
        mnist_val *= circle
        mnist_test *= circle

        ds_factory = GroundTruthDatasetFactory(mnist_train, mnist_val, mnist_test, inner_circle=self.inner_circle)
        self.gt_ds = ds_factory.build_projection_dataset(num_angles=self.num_angles,
                                                         upscale_shape=70,
                                                         impl='astra_cpu')


class LoDoPaB_TRecFITDM(TomoFITDataModule):
    IMG_SHAPE = 361

    def __init__(self, batch_size, gt_shape=111, num_angles=33):
        """
        Uses the LoDoPaB[1] dataset.

        :param batch_size:
        :param gt_shape: size of the ground truth data
        :param num_angles: for projection

        References:
            [1]  Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maaß.
            The lodopab-ct dataset: A benchmark dataset for low-dose ct reconstruction methods.
            arXiv preprint arXiv:1910.01113, 2019.
        """
        super().__init__(root_dir=None, gt_shape=gt_shape, batch_size=batch_size, num_angles=num_angles)

    def prepare_data(self, *args, **kwargs):
        lodopab = dival.get_standard_dataset('lodopab', impl='astra_cpu')
        assert self.gt_shape <= self.IMG_SHAPE, 'GT is larger than original images.'
        if self.gt_shape < self.IMG_SHAPE:
            gt_train = np.array([resize(lodopab.get_sample(i, part='train', out=(False, True))[1][1:, 1:],
                                        output_shape=(self.gt_shape, self.gt_shape), anti_aliasing=True) for i in
                                 range(4000)])
            gt_val = np.array([resize(lodopab.get_sample(i, part='validation', out=(False, True))[1][1:, 1:],
                                      output_shape=(self.gt_shape, self.gt_shape), anti_aliasing=True) for i in
                               range(400)])
            gt_test = np.array([resize(lodopab.get_sample(i, part='test', out=(False, True))[1][1:, 1:],
                                       output_shape=(self.gt_shape, self.gt_shape), anti_aliasing=True) for i in
                                range(3553)])
        else:
            gt_train = np.array(
                [lodopab.get_sample(i, part='train', out=(False, True))[1][1:, 1:] for i in range(4000)])
            gt_val = np.array(
                [lodopab.get_sample(i, part='validation', out=(False, True))[1][1:, 1:] for i in range(400)])
            gt_test = np.array([lodopab.get_sample(i, part='test', out=(False, True))[1][1:, 1:] for i in range(3553)])

        gt_train = torch.from_numpy(gt_train)
        gt_val = torch.from_numpy(gt_val)
        gt_test = torch.from_numpy(gt_test)

        assert gt_train.shape[1] == self.gt_shape
        assert gt_train.shape[2] == self.gt_shape

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        circle = self.__get_circle__()
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        ds_factory = GroundTruthDatasetFactory(gt_train, gt_val, gt_test, inner_circle=self.inner_circle)
        self.gt_ds = ds_factory.build_projection_dataset(num_angles=self.num_angles,
                                                         upscale_shape=self.gt_shape + (self.gt_shape // 2 - 7),
                                                         impl='astra_cpu')


class CropLoDoPaB_TRecFITDM(TomoFITDataModule):
    IMG_SHAPE = 361

    def __init__(self, batch_size, gt_shape=111, num_angles=33):
        """
        Uses the LoDoPaB[1] dataset.

        :param batch_size:
        :param gt_shape: size of the ground truth data
        :param num_angles: for projection

        References:
            [1]  Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maaß.
            The lodopab-ct dataset: A benchmark dataset for low-dose ct reconstruction methods.
            arXiv preprint arXiv:1910.01113, 2019.
        """
        super().__init__(root_dir=None, gt_shape=gt_shape, batch_size=batch_size, num_angles=num_angles)

    def prepare_data(self, *args, **kwargs):
        lodopab = dival.get_standard_dataset('lodopab', impl='astra_cpu')
        assert self.gt_shape <= self.IMG_SHAPE, 'GT is larger than original images.'
        if self.gt_shape < self.IMG_SHAPE:
            crop_off = (362 - self.gt_shape) // 2
            gt_train = np.array([lodopab.get_sample(i, part='train', out=(False, True))[1][crop_off:-(crop_off + 1),
                                 crop_off:-(crop_off + 1)] for i in
                                 range(4000)])
            gt_val = np.array([lodopab.get_sample(i, part='validation', out=(False, True))[1][crop_off:-(crop_off + 1),
                               crop_off:-(crop_off + 1)] for i in
                               range(400)])
            gt_test = np.array([lodopab.get_sample(i, part='test', out=(False, True))[1][crop_off:-(crop_off + 1),
                                crop_off:-(crop_off + 1)] for i in
                                range(3553)])
        else:
            gt_train = np.array(
                [lodopab.get_sample(i, part='train', out=(False, True))[1][1:, 1:] for i in range(4000)])
            gt_val = np.array(
                [lodopab.get_sample(i, part='validation', out=(False, True))[1][1:, 1:] for i in range(400)])
            gt_test = np.array(
                [lodopab.get_sample(i, part='test', out=(False, True))[1][1:, 1:] for i in range(3553)])

        gt_train = torch.from_numpy(gt_train)
        gt_val = torch.from_numpy(gt_val)
        gt_test = torch.from_numpy(gt_test)

        assert gt_train.shape[1] == self.gt_shape
        assert gt_train.shape[2] == self.gt_shape

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        circle = self.__get_circle__()
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        ds_factory = GroundTruthDatasetFactory(gt_train, gt_val, gt_test, inner_circle=self.inner_circle)
        self.gt_ds = ds_factory.build_projection_dataset(num_angles=self.num_angles,
                                                         upscale_shape=self.gt_shape + (self.gt_shape // 2 - 7),
                                                         impl='astra_cpu')


class Kanji_TRecFITDM(TomoFITDataModule):

    def __init__(self, root_dir, batch_size, num_angles=33):
        """
        The underlying data is Kanji[1] saved in a npz file, which is downloaded if it is not available on your machine.

        :param root_dir: path to downloaded data
        :param batch_size:
        :param num_angles: for projection

        References:
            [1] Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha.
            Deep learning for classical japanese literature, 2018.
        """
        super().__init__(root_dir=root_dir, gt_shape=63, batch_size=batch_size, num_angles=num_angles)

    def prepare_data(self, *args, **kwargs):
        if not exists(join(self.root_dir, 'gt_data.npz')):
            wget.download('https://cloud.mpi-cbg.de/index.php/s/7MK9vNUnq4Ndkhg/download',
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

        circle = self.__get_circle__()
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        ds_factory = GroundTruthDatasetFactory(gt_train, gt_val, gt_test, inner_circle=self.inner_circle)
        self.gt_ds = ds_factory.build_projection_dataset(num_angles=self.num_angles,
                                                         upscale_shape=133,
                                                         impl='astra_cpu')
