import numpy as np
import torch


def get_detector_length(proj_space):
    # based on odl.tomo.geometry.parallel.parallel_beam_geometry
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


def get_proj_coords_pol(angles, det_len):
    tmp = det_len // 2 + 1
    a = np.rad2deg(-angles + np.pi / 2.)
    r = np.arange(0, tmp)
    r, a = np.meshgrid(r, a)
    flatten_indices = np.argsort(r.flatten())
    r = r.flatten()[flatten_indices]
    a = a.flatten()[flatten_indices]
    return torch.from_numpy(r), torch.from_numpy(np.deg2rad(a)), flatten_indices


def get_proj_coords_cart(angles, det_len):
    r, a, flatten_indices = get_proj_coords_pol(angles, det_len)
    xcoords = r * torch.cos(a)
    ycoords = (det_len // 2) + r * torch.sin(a)
    return xcoords, ycoords, flatten_indices


def get_img_coords_cart(img_shape, det_len):
    xcoords, ycoords = np.meshgrid(np.linspace(0, det_len // 2, num=img_shape // 2 + 1, endpoint=True),
                                   np.concatenate([np.linspace(0, det_len // 2, img_shape // 2, False),
                                                   np.linspace(det_len // 2, det_len - 1, img_shape // 2 + 1)]))

    order = np.sqrt(xcoords ** 2 + (ycoords - (det_len // 2)) ** 2)
    order = np.roll(order, img_shape // 2 + 1, 0)
    xcoords = np.roll(xcoords, img_shape // 2 + 1, 0)
    ycoords = np.roll(ycoords, img_shape // 2 + 1, 0)
    flatten_indices = np.argsort(order.flatten())
    xcoords = xcoords.flatten()[flatten_indices]
    ycoords = ycoords.flatten()[flatten_indices]
    return torch.from_numpy(xcoords), torch.from_numpy(ycoords), flatten_indices, order


def get_img_coords_pol(img_shape, det_len):
    xcoords, ycoords, flatten_indices, order = get_img_coords_cart(img_shape, det_len)
    ycoords -= img_shape // 2
    r = torch.sqrt(xcoords ** 2 + ycoords ** 2)
    phi = torch.atan2(ycoords, xcoords)
    return r, phi, flatten_indices, order


def pol2cart(rho, phi):
    x = rho * torch.cos(phi)
    y = rho * torch.sin(phi)
    return (x, y)
