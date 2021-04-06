import numpy as np
import odl
import torch
from dival import Dataset
from dival.datasets.dataset import ObservationGroundTruthPairDataset
from odl import uniform_discr
from skimage.transform import resize

from fit.utils.tomo_utils import get_detector_length


class GroundTruthDatasetFactory(Dataset):
    """
    Factory to create projection datasets from any 2D image-data.

    This is essentially a simple version of dival[1] without any noise contribution.

    References:
        [1] Johannes Leuschner, Maximilian Schmidt, Daniel Otero Baguer, and Peter Maaß.
        The lodopab-ct dataset: A benchmark dataset for low-dose ct reconstruction methods.
        arXiv preprint arXiv:1910.01113, 2019.
    """

    def __init__(self, train_gt_images, val_gt_images, test_gt_images, inner_circle=True):
        """
        Note: Currently only odd sized images are supported.

        :param train_gt_images:
        :param val_gt_images:
        :param test_gt_images:
        :param inner_circle: all pixels outside the largest circle around the center are set to zero i.e.
        the detector length is equal to the image height
        """
        self.train_gt_images = train_gt_images
        self.val_gt_images = val_gt_images
        self.test_gt_images = test_gt_images
        assert self.train_gt_images.shape[1] == self.train_gt_images.shape[2], 'Train images are not square.'
        assert self.train_gt_images.shape[1] % 2 == 1, 'Train image size has to be odd.'
        assert self.val_gt_images.shape[1] == self.val_gt_images.shape[2], 'Val images are not square.'
        assert self.val_gt_images.shape[1] % 2 == 1, 'Val image size has to be odd.'
        assert self.test_gt_images.shape[1] == self.test_gt_images.shape[2], 'Test images are not square.'
        assert self.test_gt_images.shape[1] % 2 == 1, 'Test image size has to be odd.'

        self.shape = (self.train_gt_images.shape[1], self.train_gt_images.shape[2])
        self.inner_circle = inner_circle
        if self.inner_circle:
            circ_space = np.sqrt((self.shape[0] / 2.) ** 2 / 2.)
            min_pt = [-circ_space, -circ_space]
            max_pt = [circ_space, circ_space]
        else:
            min_pt = [-self.shape[0] / 2., -self.shape[1] / 2.]
            max_pt = [self.shape[0] / 2., self.shape[1] / 2.]

        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = self.train_gt_images.shape[0]
        self.validation_len = self.val_gt_images.shape[0]
        self.test_len = self.test_gt_images.shape[0]
        self.random_access = True
        super().__init__(space=space)

    def _create_pair_dataset(self, forward_op, post_processor=None,
                             noise_type=None, noise_kwargs=None,
                             noise_seeds=None):

        dataset = ObservationGroundTruthPairDataset(
            self.generator, forward_op, post_processor=post_processor,
            train_len=self.train_len, validation_len=self.validation_len,
            test_len=self.test_len, noise_type=noise_type,
            noise_kwargs=noise_kwargs, noise_seeds=noise_seeds)
        return dataset

    def build_projection_dataset(self, num_angles, upscale_shape=70, impl='astra_cpu'):
        """
        Builds the forward projection operator. The ground truth images are upscaled during the forward
        operation to avoid the the [inverse crime](https://arxiv.org/abs/math-ph/0401050).

        :param num_angles: number of projection angles
        :param upscale_shape: to avoid inverse crime
        :param impl: radon transform implementation
        :return:
        """
        forward_op, get_reco_ray_trafo, reco_ray_trafo = self._build_forward_op(upscale_shape, impl,
                                                                                num_angles)

        ds = self._create_pair_dataset(
            forward_op=forward_op, noise_type=None)

        ds.get_ray_trafo = get_reco_ray_trafo
        ds.ray_trafo = reco_ray_trafo
        return ds

    def _build_forward_op(self, upscale_shape, impl, num_angles):
        reco_space = self.space
        if self.inner_circle:
            space = odl.uniform_discr(min_pt=reco_space.min_pt,
                                      max_pt=reco_space.max_pt,
                                      shape=(upscale_shape, upscale_shape), dtype=np.float32)
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
                                      shape=(upscale_shape, upscale_shape), dtype=np.float32)
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
                out.assign(space.element(resize(x, (upscale_shape, upscale_shape), order=1)))

        # forward operator
        resize_op = _ResizeOperator()
        forward_op = ray_trafo * resize_op
        return forward_op, get_reco_ray_trafo, reco_ray_trafo

    def generator(self, part='train'):
        if part == 'train':
            gen = self._train_generator()
        elif part == 'validation':
            gen = self._val_generator()
        elif part == 'test':
            gen = self._test_generator()
        else:
            raise NotImplementedError

        for gt in gen:
            yield gt

    def _train_generator(self):
        for i in range(self.train_len):
            yield (self.train_gt_images[i].type(torch.float32))

    def _test_generator(self):
        for i in range(self.test_len):
            yield (self.test_gt_images[i].type(torch.float32))

    def _val_generator(self):
        for i in range(self.validation_len):
            yield (self.val_gt_images[i].type(torch.float32))

    def get_sample(self, index, part='train', out=None):
        if out == None:
            if part == 'train':
                return self.train_gt_images[index].type(torch.float32)
            elif part == 'validation':
                return self.val_gt_images[index].type(torch.float32)
            elif part == 'test':
                return self.test_gt_images[index].type(torch.float32)
            else:
                raise NotImplementedError
        else:
            if part == 'train':
                out = self.train_gt_images[index].type(torch.float32)
            elif part == 'validation':
                out = self.val_gt_images[index].type(torch.float32)
            elif part == 'test':
                out = self.test_gt_images[index].type(torch.float32)
            else:
                raise NotImplementedError
