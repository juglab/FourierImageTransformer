import numpy as np
import torch

from fit.utils import cart2pol, pol2cart


def get_detector_length(proj_space):
    """
    Compute detector length based on an odl projection space.

    Note: This is based on `odl.tomo.geometry.parallel.parallel_beam_geometry`

    :param proj_space: odl projection space
    :return: detector_length
    """
    corners = proj_space.domain.corners()[:, :2]
    rho = np.max(np.linalg.norm(corners, axis=1))
    # Find default values according to Nyquist criterion.
    # We assume that the function is bandlimited by a wave along the x or y
    # axis. The highest frequency we can measure is then a standing wave with
    # period of twice the inter-node distance.
    min_side = min(proj_space.partition.cell_sides[:2])
    omega = np.pi / min_side
    num_px_horiz = 2 * int(np.ceil(rho * omega / np.pi)) + 1
    # based on odl.tomo.geometry.parallel.parallel_beam_geometry
    return num_px_horiz


def get_polar_rfft_coords_sinogram(angles, det_len):
    """
    Compute polar coordinates of the sinogram 1D Fourier coefficients (rFFT1D).

    :param angles: Projection angles
    :param det_len: Detector length
    :return: radii, phi, flatten_indices
    """
    assert det_len % 2 == 1, '`det_len` has to be odd.'
    a = np.rad2deg(-angles + np.pi / 2.)
    r = np.arange(0, det_len // 2 + 1)
    r, a = np.meshgrid(r, a)
    flatten_indices = np.argsort(r.flatten())
    r = r.flatten()[flatten_indices]
    a = a.flatten()[flatten_indices]
    return torch.from_numpy(r), torch.from_numpy(np.deg2rad(a)), torch.from_numpy(flatten_indices)


def get_polar_rfft_coords_2D(img_shape):
    """
    Compute polar coordinates of the 2D Fourier coefficients obtained with rFFT.

    :param img_shape:
    :return: x, y, flatten_indices, rings
    """
    assert img_shape % 2 == 1, '`img_shape` has to be odd.'
    x, y, flatten_indices, order = get_cartesian_rfft_coords_2D(img_shape)
    y -= img_shape // 2
    r, phi = cart2pol(x, y)
    return r, phi, flatten_indices, order


def get_cartesian_rfft_coords_sinogram(angles, det_len):
    """
    Compute cartesian coordinates of the sinogram 1D Fourier coefficients (rFFT1D).

    :param angles: Projection angles
    :param det_len: Detector length
    :return: x, y, flatten_indices
    """
    assert det_len % 2 == 1, '`det_len` has to be odd.'
    r, a, flatten_indices = get_polar_rfft_coords_sinogram(angles, det_len)
    x, y = pol2cart(r, a)
    y += (det_len // 2)
    return x, y, flatten_indices


def get_cartesian_rfft_coords_2D(img_shape):
    """
    Compute cartesian coordinates of the 2D Fourier coefficients obtained with rFFT.

    :param img_shape:
    :return: x, y, flatten_indices, rings
    """
    assert img_shape % 2 == 1, '`img_shape` has to be odd.'
    xcoords, ycoords = np.meshgrid(np.arange(img_shape // 2 + 1), np.arange(img_shape))
    xcoords = xcoords.astype(np.float32)
    ycoords = ycoords.astype(np.float32)
    rings = np.sqrt(xcoords ** 2 + (ycoords - (img_shape // 2)) ** 2)

    rings = np.roll(rings, img_shape // 2 + 1, 0)
    xcoords = np.roll(xcoords, img_shape // 2 + 1, 0)
    ycoords = np.roll(ycoords, img_shape // 2 + 1, 0)

    flatten_order = np.argsort(rings.flatten())
    xcoords = xcoords.flatten()[flatten_order]
    ycoords = ycoords.flatten()[flatten_order]

    return torch.from_numpy(xcoords), torch.from_numpy(ycoords), torch.from_numpy(flatten_order), torch.from_numpy(
        rings)
