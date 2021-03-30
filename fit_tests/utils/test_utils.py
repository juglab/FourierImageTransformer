import unittest

from fit.utils import cart2pol, pol2cart
import torch

from fit.utils import normalize, denormalize
from fit.utils import normalize_amp, denormalize_amp

import numpy as np

from fit.utils.utils import log_amplitudes, normalize_phi, denormalize_phi, psf_real, normalize_FC, denormalize_FC, \
    convert2DFT


class TestUtils(unittest.TestCase):

    def test_cart2pol2cart(self):
        x = torch.arange(1, 6, dtype=torch.float32)
        y = torch.arange(-2, 3, dtype=torch.float32)

        r, phi = cart2pol(x, y)
        x_, y_ = pol2cart(r, phi)
        self.assertTrue(torch.allclose(x, x_) and torch.allclose(y, y_),
                        'Cartesian to polar coordinate transformations are broken.')

    def test_normlize_denormalize_realspace(self):
        data = torch.from_numpy(np.array([-1, 2, 4, 0, -5], dtype=np.float32))
        mean = torch.mean(data)
        std = torch.std(data)
        data_n = normalize(data, mean, std)
        self.assertAlmostEqual(torch.mean(data_n).item(), 0, 7)
        self.assertAlmostEqual(torch.std(data_n).item(), 1, 7)

        data_dn = denormalize(data_n, mean, std)
        self.assertTrue(torch.allclose(data, data_dn))

    def test_normalize_denormalize_amplitudes(self):
        amps = torch.exp(torch.arange(6, dtype=torch.float32))
        log_amps = log_amplitudes(amps)
        min_amp = log_amps.min()
        max_amp = log_amps.max()

        n_amps = normalize_amp(amps, amp_min=min_amp, amp_max=max_amp)
        amps_ = denormalize_amp(n_amps, amp_min=min_amp, amp_max=max_amp)

        self.assertTrue(torch.allclose(amps, amps_))

    def test_normalize_denormalize_phases(self):
        phases = torch.linspace(-np.pi, np.pi, 10)

        phases_n = normalize_phi(phases)
        phases_ = denormalize_phi(phases_n)

        self.assertTrue(torch.allclose(phases, phases_))

    def test_normalize_denormalize_FC(self):
        img = psf_real(7, 27)
        rfft = torch.fft.rfftn(img)
        log_amps = log_amplitudes(rfft.abs())
        min_amp = log_amps.min()
        max_amp = log_amps.max()

        amp_n, phi_n = normalize_FC(rfft, amp_min=min_amp, amp_max=max_amp)
        fc_n = torch.stack([amp_n, phi_n], -1)
        rfft_ = denormalize_FC(fc_n, amp_min=min_amp, amp_max=max_amp)

        self.assertTrue(torch.allclose(rfft, rfft_))

    def test_convert2DFT(self):
        img = psf_real(7, 27)
        rfft = torch.fft.rfftn(img)
        log_amps = log_amplitudes(rfft.abs())
        min_amp = log_amps.min()
        max_amp = log_amps.max()

        order = torch.from_numpy(np.random.permutation(27 * 14))
        amp_n, phi_n = normalize_FC(rfft, amp_min=min_amp, amp_max=max_amp)
        fc_n = torch.stack([amp_n.flatten(), phi_n.flatten()], dim=-1)[order]

        dft = convert2DFT(fc_n.unsqueeze(0), amp_min=min_amp, amp_max=max_amp, dst_flatten_order=order, img_shape=27)
        img_ = torch.fft.irfftn(dft, s=(27, 27))

        self.assertTrue(torch.allclose(img, img_))


if __name__ == '__main__':
    unittest.main()
