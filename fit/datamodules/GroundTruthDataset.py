import numpy as np
import torch
from dival import Dataset
from dival.datasets.dataset import ObservationGroundTruthPairDataset
from odl import uniform_discr


class GroundTruthDataset(Dataset):
    def __init__(self, train_gt_images, val_gt_images, test_gt_images, inner_circle=True):
        self.train_gt_images = train_gt_images
        self.val_gt_images = val_gt_images
        self.test_gt_images = test_gt_images
        assert self.train_gt_images.shape[1] == self.train_gt_images.shape[2], 'Train images are not square.'
        # assert self.train_gt_images.shape[1] % 2 == 1, 'Train image size has to be odd.'
        assert self.val_gt_images.shape[1] == self.val_gt_images.shape[2], 'Val images are not square.'
        # assert self.val_gt_images.shape[1] % 2 == 1, 'Val image size has to be odd.'
        assert self.test_gt_images.shape[1] == self.test_gt_images.shape[2], 'Test images are not square.'
        # assert self.test_gt_images.shape[1] % 2 == 1, 'Test image size has to be odd.'

        self.shape = (self.train_gt_images.shape[1], self.train_gt_images.shape[2])
        if inner_circle:
            circ_space = np.sqrt((self.shape[0]/2.)**2 /2.)
            min_pt = [-circ_space, -circ_space]
            max_pt = [circ_space, circ_space]
        else:
            min_pt = [-self.shape[0]/2., -self.shape[1]/2.]
            max_pt = [self.shape[0]/2., self.shape[1]/2.]

        space = uniform_discr(min_pt, max_pt, self.shape, dtype=np.float32)
        self.train_len = self.train_gt_images.shape[0]
        self.validation_len = self.val_gt_images.shape[0]
        self.test_len = self.test_gt_images.shape[0]
        self.random_access = True
        super().__init__(space=space)

    def create_pair_dataset(self, forward_op, post_processor=None,
                            noise_type=None, noise_kwargs=None,
                            noise_seeds=None):

        dataset = ObservationGroundTruthPairDataset(
            self.generator, forward_op, post_processor=post_processor,
            train_len=self.train_len, validation_len=self.validation_len,
            test_len=self.test_len, noise_type=noise_type,
            noise_kwargs=noise_kwargs, noise_seeds=noise_seeds)
        return dataset

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
