import numpy as np
import odl
import torch
from skimage.transform import resize

from ..datamodules.tomo_rec.GroundTruthDataset import GroundTruthDataset


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


def get_proj_coords(angles, det_len):
    tmp = det_len // 2 + 1
    a = np.rad2deg(-angles + np.pi / 2.)
    r = np.arange(0, tmp)
    r, a = np.meshgrid(r, a)
    flatten_indices = np.argsort(r.flatten())
    r = r.flatten()[flatten_indices]
    a = a.flatten()[flatten_indices]
    xcoords = r * np.cos(np.deg2rad(a))
    ycoords = (tmp) + r * np.sin(np.deg2rad(a)) - 1
    return torch.from_numpy(xcoords), torch.from_numpy(ycoords), flatten_indices


def get_img_coords(img_shape, det_len):
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
