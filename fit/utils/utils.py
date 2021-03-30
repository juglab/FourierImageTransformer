import numpy as np
import scipy
import torch
import torch.fft


def log_amplitudes(amp):
    """
    Compute log-amplitudes of a Fourier spectrum.

    :param fft: 
    :return: log-amplitudes
    """
    amp[amp == 0] = 1.
    return torch.log(amp)


def normalize_FC(rfft, amp_min, amp_max):
    """
    Convert Fourier coefficients of rFFT into normalized amplitudes and phases.
    
    :param rfft: 
    :param amp_min: 
    :param amp_max: 
    :return: 
    """
    amp = rfft.abs()
    phi = rfft.angle()

    amp = normalize_amp(amp, amp_min=amp_min, amp_max=amp_max)
    phi = normalize_phi(phi)

    return amp, phi


def denormalize_FC(fc, amp_min, amp_max):
    """
    Convert normalized amplitudes and phases `x` into Fourier coefficients.
    
    :param fc: 
    :param amp_min: 
    :param amp_max: 
    :return: 
    """
    amp = denormalize_amp(fc[..., 0], amp_min=amp_min, amp_max=amp_max)
    phi = denormalize_phi(fc[..., 1])
    return torch.complex(amp * torch.cos(phi), amp * torch.sin(phi))


def convert2DFT(x, amp_min, amp_max, dst_flatten_order, img_shape=27):
    """
    Convert normalized amplitudes and phases `x` into discrete Fourier transform.
    
    :param x: 
    :param amp_min: 
    :param amp_max: 
    :param dst_flatten_order: flattening order of `x`
    :param img_shape: real-space image shape
    :return: 
    """
    x = denormalize_FC(x, amp_min, amp_max)

    dft = torch.ones(x.shape[0], img_shape * (img_shape // 2 + 1), dtype=x.dtype, device=x.device)
    dft[:, :x.shape[1]] = x

    dft[:, dst_flatten_order] = torch.flatten(dft.clone(), start_dim=1)
    return dft.reshape(-1, img_shape, img_shape // 2 + 1)


def normalize_phi(phi):
    """
    Normalize phi to [-1, 1].

    :param phi:
    :return:
    """
    return phi / np.pi


def denormalize_phi(phi):
    """
    Invert `normalize_phi`.

    :param phi:
    :return:
    """
    return phi * np.pi


def normalize_amp(amp, amp_min, amp_max):
    """
    Normalize amplitudes to [-1, 1].

    :param amp:
    :param amp_min:
    :param amp_max:
    :return:
    """
    log_amps = log_amplitudes(amp)
    return 2 * (log_amps - amp_min) / (amp_max - amp_min) - 1


def denormalize_amp(amp, amp_min, amp_max):
    """
    Invert `normalize_amp`.

    :param amp:
    :param amp_min:
    :param amp_max:
    :return:
    """
    amp = (amp + 1) / 2.
    amp = (amp * (amp_max - amp_min)) + amp_min
    amp = torch.exp(amp)
    return amp


def fft_interpolate(srcx, srcy, dstx, dsty, src_fourier_coefficients, target_shape, dst_flatten_order):
    """
    Interpolates Fourier coefficients at (dstx, dsty) from Fourier coefficients in `sinogram_FC`.
    
    :param srcx: source x-coordinates
    :param srcy: source y-coordinates
    :param dstx: interpolated x-coordinates
    :param dsty: interolated y-coordinates
    :param src_fourier_coefficients: source Fourier coefficients
    :param target_shape: output shape of the interpolated Fourier coefficients
    :param dst_flatten_order: flattening order of (dstx, dsty)
    :return: 
    """
    vals = scipy.interpolate.griddata(
        (srcx, srcy),
        src_fourier_coefficients,
        (dstx, dsty),
        method='nearest',
        fill_value=1.0
    )
    output = np.zeros_like(vals)
    output[dst_flatten_order] = vals
    return output.reshape(target_shape)


def gaussian(x, mu, sig):
    """
    Compute Gaussian.

    :param x:
    :param mu:
    :param sig:
    :return:
    """
    return torch.exp(-torch.pow(x - mu, torch.tensor(2.)) / (2 * torch.pow(sig, torch.tensor(2.))))


def gaussian_psf(x, c, r):
    """
    Create 1D Gaussian shaped point spread function (PSF) profile normalized to [0, 1].

    :param x: range
    :param c: psf center
    :param r: psf sigma (radius)
    :return:
    """
    return torch.maximum((1 + gaussian(r, 0, r / 2.)) * gaussian(x, c, r / 2.) - gaussian(r, 0, r / 2.),
                         torch.zeros_like(x))


def psf_real(r, pixel_res=32):
    """
    Compute point spread function (PSF) with radius `r`.

    :param r: radius
    :param pixel_res: resolution in pixels (size of the PSF image)
    :return: PSF image
    """
    c = int(pixel_res / 2.)
    x, y = torch.meshgrid(torch.arange(pixel_res), torch.arange(pixel_res))
    psfimg = gaussian_psf(x, c, r) * gaussian_psf(y, c, r)
    psfimg = torch.roll(psfimg, -c, dims=0)
    psfimg = torch.roll(psfimg, -c, dims=1)
    psfimg /= torch.sum(psfimg)
    return psfimg


def psf_rfft(r, pixel_res=32):
    """
    Real FFT of PSF.

    :param r: radius of PSF
    :param pixel_res: resolution in pixel (size of PSF image)
    :return: rFFT of PSF
    """
    return torch.fft.rfftn(psf_real(r, pixel_res))


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.detach().cpu().numpy().flatten(), target.detach().cpu().numpy().flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta


def PSNR(gt, img, drange):
    """
    Computes the highest PSNR by affine rescaling of x,
    such that the mean squared error to gt is minimal.

    :param gt: ground truth
    :param img: image
    :param drange: data range
    :return: PSNR
    """
    img = normalize_minmse(img, gt)
    mse = torch.mean(torch.square(gt - img))
    return 20 * torch.log10(drange) - 10 * torch.log10(mse)


def normalize(data, mean, std):
    """
    Zero-mean, one standard dev. normalization

    :param data:
    :param mean:
    :param std:
    :return: normalized data
    """
    return (data - mean) / std


def denormalize(data, mean, std):
    """
    Invert `normalize`

    :param data:
    :param mean:
    :param std:
    :return: denormalized data
    """
    return (data * std) + mean


def pol2cart(r, phi):
    """
    Polar coordinates to cartesian coordinates.
    
    :param r: 
    :param phi: 
    :return: x, y
    """
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    return (x, y)


def cart2pol(x, y):
    """
    Cartesian coordinates to polar coordinates.

    :param x:
    :param y:
    :return: r, phi
    """
    r = torch.sqrt(x ** 2 + y ** 2)
    phi = torch.atan2(y, x)
    return r, phi
