import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule


class MipLRDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.max_steps = max_steps
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        super(MipLRDecay, self).__init__(optimizer)

    def get_lr(self):
        step = self.last_epoch
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(self.last_epoch / self.lr_delay_steps, 0, 1))
        else:
            delay_rate = 1.
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        return [delay_rate * log_lerp]


class PositionalEncoding(LightningModule):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret


class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay

    def forward(self, input, target, mask=None):
        losses = []
        psnrs = []
        for rgb in input:
            if mask is None:
                mse = ((rgb - target[..., :3]) ** 2).mean()
            else:
                mse = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)
        loss = self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1]
        return loss, torch.Tensor(psnrs)


class SARNeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1):
        super(SARNeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay
        self.eikonal_weight = .1

    def forward(self, pulse_data, gt, target):
        losses = []
        psnrs = []
        for rgb in pulse_data:
            mse = ((rgb - target) ** 2).mean()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)
        loss = self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1]
        return loss + self.eikonal_weight * self.eikonal_loss(gt), torch.Tensor(psnrs)

    def eikonal_loss(self, grad_theta):
        return ((grad_theta.norm(2, dim=1) - 1)**2).mean()


def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)