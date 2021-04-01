import torch

from fit.utils.utils import denormalize_amp, denormalize_phi


def _fc_prod_loss(pred_fc, target_fc, amp_min, amp_max):
    pred_amp = denormalize_amp(pred_fc[..., 0], amp_min=amp_min, amp_max=amp_max)
    target_amp = denormalize_amp(target_fc[..., 0], amp_min=amp_min, amp_max=amp_max)

    pred_phi = denormalize_phi(pred_fc[..., 1])
    target_phi = denormalize_phi(target_fc[..., 1])

    amp_loss = 1 + torch.pow(pred_amp - target_amp, 2)
    phi_loss = 2 - torch.cos(pred_phi - target_phi)
    return torch.mean(amp_loss * phi_loss), torch.mean(amp_loss), torch.mean(phi_loss)


def _fc_sum_loss(pred_fc, target_fc, amp_min, amp_max):
    pred_amp = denormalize_amp(pred_fc[..., 0], amp_min=amp_min, amp_max=amp_max)
    target_amp = denormalize_amp(target_fc[..., 0], amp_min=amp_min, amp_max=amp_max)

    pred_phi = denormalize_phi(pred_fc[..., 1])
    target_phi = denormalize_phi(target_fc[..., 1])

    amp_loss = torch.pow(pred_amp - target_amp, 2)
    phi_loss = 1 - torch.cos(pred_phi - target_phi)
    return torch.mean(amp_loss + phi_loss), torch.mean(amp_loss), torch.mean(phi_loss)
