from glob import glob
from os.path import join
from typing import Optional, Union, List

import dival
import numpy as np
import torch
from imageio import imread
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from fit.datamodules.tomo_rec.TRecFCDataset import TRecFourierCoefficientDataset
from fit.datamodules.GroundTruthDataset import GroundTruthDataset
import odl
from skimage.transform import resize

from fit.utils.tomo_utils import get_detector_length
from fit.utils.utils import normalize


def get_projection_dataset(dataset, num_angles, im_shape=70, impl='astra_cpu', inner_circle=True):
    assert isinstance(dataset, GroundTruthDataset)
    reco_space = dataset.space
    if inner_circle:
        space = odl.uniform_discr(min_pt=reco_space.min_pt,
                                  max_pt=reco_space.max_pt,
                                  shape=(im_shape, im_shape), dtype=np.float32)
        min_pt = reco_space.min_pt
        max_pt = reco_space.max_pt
        proj_space = odl.uniform_discr(min_pt, max_pt, 2 * (2 * int(reco_space.max_pt[0]) - 1,), dtype=np.float32)
        detector_length = get_detector_length(proj_space)
        det_partition = odl.uniform_partition(-np.sqrt((reco_space.shape[0] / 2.) ** 2 / 2),
                                              np.sqrt((reco_space.shape[0] / 2.) ** 2 / 2),
                                              detector_length)
    else:
        space = odl.uniform_discr(min_pt=reco_space.min_pt,
                                  max_pt=reco_space.max_pt,
                                  shape=(im_shape, im_shape), dtype=np.float32)
        min_pt = reco_space.min_pt
        max_pt = reco_space.max_pt
        proj_space = odl.uniform_discr(min_pt, max_pt, 2 * (reco_space.shape[0],), dtype=np.float32)
        detector_length = get_detector_length(proj_space)
        det_partition = odl.uniform_partition(-reco_space.shape[0] / 2., reco_space.shape[0] / 2., detector_length)

    angle_partition = odl.uniform_partition(0, np.pi, num_angles)
    reco_geometry = odl.tomo.Parallel2dGeometry(angle_partition, det_partition)

    ray_trafo = odl.tomo.RayTransform(space, reco_geometry, impl=impl)

    def get_reco_ray_trafo(**kwargs):
        return odl.tomo.RayTransform(reco_space, reco_geometry, **kwargs)

    reco_ray_trafo = get_reco_ray_trafo(impl=impl)

    class _ResizeOperator(odl.Operator):
        def __init__(self):
            super().__init__(reco_space, space)

        def _call(self, x, out, **kwargs):
            out.assign(space.element(resize(x, (im_shape, im_shape), order=1)))

    # forward operator
    resize_op = _ResizeOperator()
    forward_op = ray_trafo * resize_op

    ds = dataset.create_pair_dataset(
        forward_op=forward_op, noise_type=None)

    ds.get_ray_trafo = get_reco_ray_trafo
    ds.ray_trafo = reco_ray_trafo
    return ds


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

        assert mnist_train.shape[1] == self.IMG_SHAPE
        assert mnist_train.shape[2] == self.IMG_SHAPE
        x, y = torch.meshgrid(torch.arange(-self.IMG_SHAPE // 2 + 1,
                                           self.IMG_SHAPE // 2 + 1),
                              torch.arange(-self.IMG_SHAPE // 2 + 1,
                                           self.IMG_SHAPE // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= self.IMG_SHAPE // 2
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

        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds, mag_min=None, mag_max=None, part='train',
                                                 img_shape=self.IMG_SHAPE)
        self.mag_min = tmp_fcds.mag_min
        self.mag_max = tmp_fcds.mag_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='train',
                                          img_shape=self.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='validation',
                                          img_shape=self.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='test',
                                          img_shape=self.IMG_SHAPE),
            batch_size=1)


class LoDoPaBFourierTargetDataModule(LightningDataModule):
    IMG_SHAPE = 361

    def __init__(self, batch_size, gt_shape=361, num_angles=15):
        """
        :param root_dir:
        :param batch_size:
        :param num_angles:
        """
        super().__init__()
        self.batch_size = batch_size
        self.gt_shape = gt_shape
        self.num_angles = num_angles
        self.inner_circle = True
        self.gt_ds = None
        self.mean = None
        self.std = None

    def setup(self, stage: Optional[str] = None):
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
        x, y = torch.meshgrid(torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1),
                              torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= self.gt_shape // 2
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(gt_train, gt_val, gt_test),
            num_angles=self.num_angles, im_shape=self.gt_shape + (self.gt_shape // 2 - 7), impl='astra_cpu',
            inner_circle=self.inner_circle)

        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds, mag_min=None, mag_max=None, part='train',
                                                 img_shape=self.gt_shape)
        self.mag_min = tmp_fcds.mag_min
        self.mag_max = tmp_fcds.mag_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='train',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='validation',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='test',
                                          img_shape=self.gt_shape),
            batch_size=1)


class CropLoDoPaBFourierTargetDataModule(LightningDataModule):
    IMG_SHAPE = 361

    def __init__(self, batch_size, gt_shape=361, num_angles=15):
        """
        :param root_dir:
        :param batch_size:
        :param num_angles:
        """
        super().__init__()
        self.batch_size = batch_size
        self.gt_shape = gt_shape
        self.num_angles = num_angles
        self.inner_circle = True
        self.gt_ds = None
        self.mean = None
        self.std = None

    def setup(self, stage: Optional[str] = None):
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
        x, y = torch.meshgrid(torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1),
                              torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= self.gt_shape // 2
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(gt_train, gt_val, gt_test),
            num_angles=self.num_angles, im_shape=self.gt_shape + (self.gt_shape // 2 - 7), impl='astra_cpu',
            inner_circle=self.inner_circle)

        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds, mag_min=None, mag_max=None, part='train',
                                                 img_shape=self.gt_shape)
        self.mag_min = tmp_fcds.mag_min
        self.mag_max = tmp_fcds.mag_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='train',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='validation',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='test',
                                          img_shape=self.gt_shape),
            batch_size=1)


class KanjiFourierTargetDataModule(LightningDataModule):
    IMG_SHAPE = 63

    def __init__(self, root_dir, batch_size, num_angles=33):
        """
        :param root_dir:
        :param batch_size:
        :param num_angles:
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_angles = num_angles
        self.inner_circle = True
        self.gt_ds = None
        self.mean = None
        self.std = None

    def setup(self, stage: Optional[str] = None):
        gt_data = np.load(join(self.root_dir, 'gt_data.npz'))

        gt_train = torch.from_numpy(gt_data['gt_train'])
        gt_val = torch.from_numpy(gt_data['gt_val'])
        gt_test = torch.from_numpy(gt_data['gt_test'])

        x, y = torch.meshgrid(torch.arange(-self.IMG_SHAPE // 2 + 1,
                                           self.IMG_SHAPE // 2 + 1),
                              torch.arange(-self.IMG_SHAPE // 2 + 1,
                                           self.IMG_SHAPE // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= self.IMG_SHAPE // 2
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(gt_train, gt_val, gt_test),
            num_angles=self.num_angles, im_shape=133, impl='astra_cpu', inner_circle=self.inner_circle)

        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds, mag_min=None, mag_max=None, part='train',
                                                 img_shape=self.IMG_SHAPE)
        self.mag_min = tmp_fcds.mag_min
        self.mag_max = tmp_fcds.mag_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='train',
                                          img_shape=self.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='validation',
                                          img_shape=self.IMG_SHAPE),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='test',
                                          img_shape=self.IMG_SHAPE),
            batch_size=1)


class CelebAFourierTargetDataModule(LightningDataModule):
    IMG_SHAPE = 127

    def __init__(self, root_dir, batch_size, num_angles=33):
        """
        :param root_dir:
        :param batch_size:
        :param num_angles:
        """
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.gt_shape = 63
        self.num_angles = num_angles
        self.inner_circle = True
        self.gt_ds = None
        self.mean = None
        self.std = None

    def setup(self, stage: Optional[str] = None):
        gt_data = np.load(join(self.root_dir, 'gt_data.npz'))

        gt_train = torch.from_numpy(gt_data['gt_train'])
        gt_val = torch.from_numpy(gt_data['gt_val'])
        gt_test = torch.from_numpy(gt_data['gt_test'])

        assert gt_train.shape[1] == self.gt_shape
        assert gt_train.shape[2] == self.gt_shape
        x, y = torch.meshgrid(torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1),
                              torch.arange(-self.gt_shape // 2 + 1,
                                           self.gt_shape // 2 + 1))
        circle = torch.sqrt(x ** 2. + y ** 2.) <= self.gt_shape // 2
        gt_train *= circle
        gt_val *= circle
        gt_test *= circle

        self.mean = gt_train.mean()
        self.std = gt_train.std()

        gt_train = normalize(gt_train, self.mean, self.std)
        gt_val = normalize(gt_val, self.mean, self.std)
        gt_test = normalize(gt_test, self.mean, self.std)

        self.gt_ds = get_projection_dataset(
            GroundTruthDataset(gt_train, gt_val, gt_test),
            num_angles=self.num_angles, im_shape=153, impl='astra_cpu', inner_circle=self.inner_circle)

        tmp_fcds = TRecFourierCoefficientDataset(self.gt_ds, mag_min=None, mag_max=None, part='train',
                                                 img_shape=self.gt_shape)
        self.mag_min = tmp_fcds.mag_min
        self.mag_max = tmp_fcds.mag_max

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='train',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='validation',
                                          img_shape=self.gt_shape),
            batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            TRecFourierCoefficientDataset(self.gt_ds, mag_min=self.mag_min, mag_max=self.mag_max, part='test',
                                          img_shape=self.gt_shape),
            batch_size=1)
