import unittest

from fit.utils.tomo_utils import get_cartesian_rfft_coords_2D, get_polar_rfft_coords_2D, get_polar_rfft_coords_sinogram, \
    get_cartesian_rfft_coords_sinogram
import torch

import numpy as np


class TestTomoUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.img_shape = 27
        self.angles = np.array([0, np.pi / 2, np.pi])

    def test_cartesian_rfft_2D(self):
        x, y, flatten_indices, order = get_cartesian_rfft_coords_2D(self.img_shape)
        x_ordered = torch.zeros_like(x)
        x_ordered[flatten_indices] = x
        x_ordered = x_ordered.reshape(self.img_shape, -1)

        y_ordered = torch.zeros_like(y)
        y_ordered[flatten_indices] = y
        y_ordered = y_ordered.reshape(self.img_shape, -1)
        y_ordered = torch.roll(y_ordered, -(self.img_shape // 2 + 1), 0)

        y_target, x_target = torch.meshgrid(torch.arange(self.img_shape), torch.arange(self.img_shape // 2 + 1))

        self.assertEqual(order[0, 0], 0, 'Top left pixel should have index 0.')
        self.assertTrue(torch.all(x_target == x_ordered) and torch.all(y_target == y_ordered),
                        'rFFT coordinates are wrong.')

    def test_polar_rfft_2D(self):
        r, phi, flatten_indices, order = get_polar_rfft_coords_2D(img_shape=self.img_shape)

        self.assertEqual(order[0, 0], 0, 'Top left pixel should have index 0.')

        r_ordered = torch.zeros_like(r)
        r_ordered[flatten_indices] = r
        r_ordered = r_ordered.reshape(self.img_shape, -1)
        self.assertEqual(r_ordered[0, 0], 0, 'Top left pixel does not have radius 0.')

        phi_ordered = torch.zeros_like(phi)
        phi_ordered[flatten_indices] = phi
        phi_ordered = phi_ordered.reshape(self.img_shape, -1)
        self.assertEqual(phi_ordered[0, 0], 0, 'Top left pixel angle does not correspond to 0.')
        self.assertEqual(phi_ordered[self.img_shape // 2, 0], np.pi / 2, 'Phi component is of (test 1).')
        self.assertEqual(phi_ordered[self.img_shape - 1, 0], -np.pi / 2, 'Phi component is of (test 2).')

    def test_polar_sinogram(self):
        r, phi, flatten_indices = get_polar_rfft_coords_sinogram(self.angles, self.img_shape)
        self.assertTrue(torch.all((r[0::3] == r[1::3]) == (r[1::3] == r[2::3])),
                        'Radii of polar sinogram coords are off.')

        phi_ordered = torch.zeros_like(phi)
        phi_ordered[flatten_indices] = phi
        self.assertTrue(torch.all(phi_ordered[:self.img_shape // 2 + 1] == np.pi / 2.),
                        'Phi of polar sinogram coords are off (test1).')
        self.assertTrue(torch.all(phi_ordered[self.img_shape // 2 + 1:-(self.img_shape // 2 + 1)] == 0),
                        'Phi of polar sinogram coords are off (test1).')
        self.assertTrue(torch.all(phi_ordered[-(self.img_shape // 2 + 1):] == -np.pi / 2.),
                        'Phi of polar sinogram coords are off (test2).')

    def test_cartesian_sinogram(self):
        x, y, flatten_indices = get_cartesian_rfft_coords_sinogram(self.angles, self.img_shape)
        print(x)
        self.assertTrue(torch.all(x <= self.img_shape // 2 + 1))
        self.assertTrue(torch.all(x >= 0))
        self.assertTrue(torch.all(y <= self.img_shape))
        self.assertTrue(torch.all(y >= 0))


if __name__ == '__main__':
    unittest.main()
