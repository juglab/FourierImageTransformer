import numpy as np
import odl
import torch
from skimage.transform import resize

from fit.datamodules.tomo_rec.GroundTruthDataset import GroundTruthDataset


def get_projection_dataset(dataset, num_angles, IM_SHAPE=(400, 400), impl='astra_cpu'):
    assert isinstance(dataset, GroundTruthDataset)
    reco_space = dataset.space

    space = odl.uniform_discr(min_pt=reco_space.min_pt,
                              max_pt=reco_space.max_pt,
                              shape=IM_SHAPE, dtype=np.float32)

    reco_geometry = odl.tomo.parallel_beam_geometry(
        reco_space, num_angles=num_angles)
    geometry = odl.tomo.parallel_beam_geometry(
        space, num_angles=num_angles,
        det_shape=reco_geometry.detector.shape)

    ray_trafo = odl.tomo.RayTransform(space, geometry, impl=impl)

    def get_reco_ray_trafo(**kwargs):
        return odl.tomo.RayTransform(reco_space, reco_geometry, **kwargs)

    reco_ray_trafo = get_reco_ray_trafo(impl=impl)

    class _ResizeOperator(odl.Operator):
        def __init__(self):
            super().__init__(reco_space, space)

        def _call(self, x, out, **kwargs):
            out.assign(space.element(resize(x, IM_SHAPE, order=1)))

    # forward operator
    resize_op = _ResizeOperator()
    forward_op = ray_trafo * resize_op

    ds = dataset.create_pair_dataset(
        forward_op=forward_op, noise_type=None)

    ds.get_ray_trafo = get_reco_ray_trafo
    ds.ray_trafo = reco_ray_trafo
    return ds


def get_proj_coords(angles, img_shape):
    tmp = np.round(np.sqrt(2 * (img_shape // 2) ** 2)) + 1
    a = np.rad2deg(-angles + np.pi / 2.)
    r = np.arange(tmp)
    r, a = np.meshgrid(r, a)
    r = r.flatten()
    a = a.flatten()
    xcoords = r * np.cos(np.deg2rad(a))
    ycoords = (tmp) + r * np.sin(np.deg2rad(a))
    return torch.from_numpy(xcoords), torch.from_numpy(ycoords)


def get_img_coords(img_shape, endpoint=False):
    fs_size = np.round(np.sqrt(2 * (img_shape // 2) ** 2)) + 1
    xcoords, ycoords = np.meshgrid(np.linspace(0, fs_size, num=img_shape // 2 + 1, endpoint=endpoint),
                                   np.linspace(0, 2 * fs_size, num=img_shape, endpoint=endpoint))
    xcoords = np.roll(xcoords, img_shape // 2, 0)
    ycoords = np.roll(ycoords, img_shape // 2, 0)
    xcoords = xcoords.flatten()
    ycoords = ycoords.flatten()
    return torch.from_numpy(xcoords), torch.from_numpy(ycoords)
