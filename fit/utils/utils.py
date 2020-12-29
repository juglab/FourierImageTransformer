import numpy as np
import scipy
import torch


def convert2FC(x, mag_min, mag_max):
    mag = x[..., 0]
    phi = x[..., 1]
    mag = (mag * (mag_max - mag_min)) + mag_min
    mag = torch.exp(mag)

    phi = phi * 2 * np.pi
    return torch.complex(mag * torch.cos(phi), mag * torch.sin(phi))


def fft_interpolate(srcx, srcy, dstx, dsty, sino_fft, target_shape):
    return scipy.interpolate.griddata(
        (srcx, srcy),
        sino_fft,
        (dstx, dsty),
        method='nearest',
        fill_value=1.0
    ).reshape(target_shape)
