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


def fft_interpolate(srcx, srcy, dstx, dsty, sino_fft, target_shape, dst_flatten_order):
    vals = scipy.interpolate.griddata(
        (srcx, srcy),
        sino_fft,
        (dstx, dsty),
        method='nearest',
        fill_value=1.0
    )
    output = np.zeros_like(vals)
    output[dst_flatten_order] = vals
    return output.reshape(target_shape)

def gaussian(x, mu, sig):
    return torch.exp(-torch.pow(x - mu, torch.tensor(2.)) / (2 * torch.pow(sig, torch.tensor(2.))))

def gaussian_psf(x,c,r):
    return torch.maximum((1+gaussian(r,0,r/2.))*gaussian(x,c,r/2.)-gaussian(r,0,r/2.),torch.zeros_like(x))

def psf_real(r,pixel_res=32):
    c=int(pixel_res/2.)
    x, y = torch.meshgrid(torch.arange(pixel_res), torch.arange(pixel_res))
    psfimg = gaussian_psf(x, c, r) * gaussian_psf(y, c, r)
    psfimg=torch.roll(psfimg, -c, dims=0)
    psfimg=torch.roll(psfimg, -c, dims=1)
    psfimg/=torch.sum(psfimg)
    return psfimg

def psfft(r,pixel_res=32):
    return torch.fft.rfftn(psf_real(r,pixel_res))

def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta

def PSNR(gt, img, drange):
    img = normalize_minmse(img, gt)
    mse = torch.mean(torch.square(gt - img))
    return 20 * torch.log10(drange) - 10 * torch.log10(mse)

def convert_to_dft(fc, mag_min, mag_max, dst_flatten_coords, img_shape=28):
    fc = convert2FC(fc, mag_min, mag_max)

    dft = torch.ones(fc.shape[0], img_shape*(img_shape//2 + 1), dtype=fc.dtype, device=fc.device)
    dft[:, :fc.shape[1]] = fc

    dft[:,dst_flatten_coords] = torch.flatten(dft.clone(), start_dim=1)
    return dft.reshape(-1, img_shape, img_shape//2 + 1)