import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from tqdm import tqdm
import mcubes
import numpy as np
from utils import sample_along_rays, resample_along_rays, volumetric_rendering, rays_from_pose, volumetric_scattering

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
c0 = 299792458.0
DTR = np.pi / 180


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


# Model
class NeRF(LightningModule):
    def __init__(self, D=8, hidden=256,
                 randomized=False,
                 ray_shape="cone",
                 num_levels=2,
                 num_samples=128,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
                 return_raw=False,
                 learn_params=None):
        """
        """
        super(NeRF, self).__init__()
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_noise = density_noise
        self.density_bias = density_bias
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.return_raw = return_raw
        self.params = learn_params
        self.automatic_optimization = False
        self.input_ch = int((max_deg - min_deg) * 2 * 3)
        self.density_activation = nn.Softplus()
        self.hidden = hidden
        self.white_bkgd = False
        self.encoding = PositionalEncoding(min_deg, max_deg)

        net0 = [nn.Sequential(
            nn.Linear(self.input_ch, self.hidden),
            nn.GELU(),
        )]
        net0 = net0 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        ) for _ in range(D)]

        self.density_net0 = nn.Sequential(*net0)

        net1 = [nn.Sequential(
            nn.Linear(self.input_ch + self.hidden, self.hidden),
            nn.GELU(),
        )]
        net1 = net1 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        ) for _ in range(D)]

        self.density_net1 = nn.Sequential(*net1)

        self.final_density = nn.Sequential(
            nn.Linear(self.hidden, 1),
        )

        self.final_rgb = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Sigmoid()
        )

    def forward(self, rays_o, rays_d, radii, near, far):
        comp_rgbs = []
        distances = []
        accs = []
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(rays_o, rays_d, radii, self.num_samples,
                                                        near, far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays_o, rays_d, radii,
                                                          t_vals.to(self.device),
                                                          weights.to(self.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype,
                                                               device=self.device)

            # volumetric rendering
            raw_rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            raw_density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(raw_rgb, raw_density, t_vals,
                                                                           rays_d,
                                                                           self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(raw_rgb).detach(), torch.clone(raw_density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        if self.params['scheduler_gamma'] is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'], verbose=True)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        self.log('LR', sch.get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def train_val_get(self, batch, batch_idx, kind='train'):
        rays_o, rays_d, radii, near, far, target = batch

        # Generate rays for random sampling
        rgb, disp, acc = self.forward(rays_o, rays_d, radii, near, far)

        train_loss, psnrs = self.loss_function(rgb, target)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss

    def loss_function(self, input, target):
        losses = []
        psnrs = []
        for rgb in input:
            mse = ((rgb - target[..., :3]) ** 2).mean()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)
        loss = self.params['coarse_weight_decay'] * torch.sum(losses[:-1]) + losses[-1]
        return loss, torch.Tensor(psnrs)

    def render_image(self, pose, height, width, focal, chunks=1024):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        rgbs = []
        dists = []
        accs = []
        rays_d, rays_o, radii, near, far = rays_from_pose(width, height, focal, pose, 0., 1.)
        with torch.no_grad():
            for i in range(0, rays_o.shape[0], chunks):
                rgb, distance, acc = self(rays_o[i:i + chunks], rays_d[i:i + chunks], radii[i:i + chunks],
                                          near[i:i + chunks], far[i:i + chunks])
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, accs

    def render_model(self, xrange: tuple[float, float], yrange: tuple[float, float], zrange: tuple[float, float],
                     xpts: int = 400, ypts: int = 400, zpts: int = 100, sigma_threshold: float = 50.,
                     chunks: int = 1024):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        self.return_raw = True
        x, y, z = np.meshgrid(np.linspace(xrange[0], xrange[1], xpts), np.linspace(yrange[0], yrange[1], ypts),
                              np.linspace(zrange[0], zrange[1], zpts))
        rays_o = torch.FloatTensor(np.stack([x, y, z], -1).reshape(-1, 3)).to(self.device)
        rays_d = torch.zeros_like(rays_o).to(self.device)
        radii = torch.ones_like(rays_o[..., :1]).to(self.device) * 0.0005
        near = torch.ones_like(radii).to(self.device)
        far = torch.ones_like(radii).to(self.device)
        raws = []
        with torch.no_grad():
            for i in tqdm(range(0, rays_o.shape[0], chunks)):
                _, _, _, raw = self(rays_o[i:i + chunks], rays_d[i:i + chunks], radii[i:i + chunks],
                                          near[i:i + chunks], far[i:i + chunks])
                raws.append(torch.mean(raw, dim=1).cpu())

        sigma = torch.cat(raws, dim=0)
        sigma = np.maximum(sigma[:, -1].cpu().numpy(), 0)
        sigma = sigma.reshape(x.shape)
        print("Extracting mesh")
        print("fraction occupied", np.mean(np.array(sigma > sigma_threshold), dtype=np.float32))
        vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
        mcubes.export_obj(vertices, triangles, 'render_output.obj')
        return vertices, triangles


class SARNeRF(LightningModule):
    def __init__(self, D=8, hidden=256,
                 randomized=False,
                 ray_shape="cone",
                 num_levels=2,
                 num_samples=128,
                 density_noise=1,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=32,
                 return_raw=False,
                 pulse_std=1,
                 learn_params=None):
        """
        """
        super(SARNeRF, self).__init__()
        self.randomized = randomized
        self.ray_shape = ray_shape
        self.num_levels = num_levels
        self.num_samples = num_samples
        self.density_noise = density_noise
        self.density_bias = density_bias
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.wavelength = c0 / 9.6e9
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.return_raw = return_raw
        self.params = learn_params
        self.pulse_std = pulse_std
        self.automatic_optimization = False
        self.input_ch = int((max_deg - min_deg) * 2 * 3)
        self.density_activation = nn.Softplus()
        self.hidden = hidden
        self.encoding = PositionalEncoding(min_deg, max_deg)

        net0 = [nn.Sequential(
            nn.Linear(self.input_ch, self.hidden),
            nn.GELU(),
        )]
        net0 = net0 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        ) for _ in range(D)]

        self.density_net0 = nn.Sequential(*net0)

        net1 = [nn.Sequential(
            nn.Linear(self.input_ch + self.hidden, self.hidden),
            nn.GELU(),
        )]
        net1 = net1 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        ) for _ in range(D)]

        self.density_net1 = nn.Sequential(*net1)

        self.final_density = nn.Sequential(
            nn.Linear(self.hidden, 1),
        )

        self.final_norm = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Softsign()
        )

        self.final_rcs = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Sigmoid()
        )

        self.norm_net1 = nn.Sequential(
            nn.Linear(self.input_ch + self.hidden, self.hidden),
            nn.GELU(),
        )

        self.norm_net0 = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
        )

        _xavier_init(self)

    def forward(self, rays_o, rays_d, rays_p, mfilt, radii, _near, _far, nsam, mpp):
        comp_rgbs = []
        distances = []
        accs = []
        scaling = _far - _near
        near = torch.zeros_like(radii, device=self.device)
        far = torch.ones_like(radii, device=self.device)

        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(rays_o.reshape((-1, 3)), rays_d.reshape((-1, 3)), radii.reshape((-1, 1)), self.num_samples,
                                                        near.reshape((-1, 1)), far.reshape((-1, 1)), randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays_o.reshape((-1, 3)), rays_d.reshape((-1, 3)), radii.reshape((-1, 1)),
                                                          t_vals.to(self.device),
                                                          weights.reshape((-1, self.num_samples)).to(self.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))

            new_encodings = self.norm_net0(new_encodings)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.norm_net1(new_encodings)
            raw_norms = self.final_norm(new_encodings).reshape((-1, radii.shape[1], self.num_samples, 3))
            raw_rcs = self.final_rcs(new_encodings).reshape((-1, radii.shape[-1], self.num_samples, 3))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype,
                                                               device=self.device)

            # volumetric rendering
            raw_density = self.density_activation(raw_density + self.density_bias).reshape((-1, radii.shape[-1], self.num_samples, 1))
            comp_rp, distance, acc, weights, alpha = volumetric_scattering(raw_norms, raw_density, raw_rcs, rays_p,
                                                                           t_vals.reshape((-1, radii.shape[-1], self.num_samples + 1)), rays_d,
                                                                           torch.arange(nsam, device=self.device) * mpp + _near,
                                                                           _near, self.wavelength)
            comp_pulse = torch.view_as_real(torch.fft.ifft(mfilt * torch.fft.fft(torch.view_as_complex(comp_rp),
                                                                                 mfilt.shape[-1], dim=-1), dim=-1))[..., :nsam, :]

            comp_rgbs.append(comp_pulse)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(raw_norms).detach(), torch.clone(raw_rcs).detach(), torch.clone(raw_density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                lr=self.params['LR'],
                                weight_decay=self.params['weight_decay'],
                                betas=self.params['betas'],
                                eps=1e-7)
        if self.params['scheduler_gamma'] is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'], verbose=True)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()
        self.log('LR', sch.get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def train_val_get(self, batch, batch_idx, kind='train'):
        rays_o, rays_d, rays_p, mfilt, radii, near, far, mpp, target = batch

        # Generate rays for random sampling
        rgb, disp, acc = self.forward(rays_o, rays_d, rays_p, mfilt, radii, near, far, target.shape[1], mpp)

        train_loss = self.loss_function(rgb, target)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss

    def loss_function(self, input, target):
        losses = []
        for rgb in input:
            mse = ((rgb - target) ** 2).mean()
            losses.append(mse)
        losses = torch.stack(losses)
        loss = self.params['coarse_weight_decay'] * torch.sum(losses[:-1]) + losses[-1]
        return loss

    def render_model(self, xrange: tuple[float, float], yrange: tuple[float, float], zrange: tuple[float, float],
                     xpts: int = 400, ypts: int = 400, zpts: int = 100, sigma_threshold: float = 50.,
                     chunks: int = 256):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        self.return_raw = True
        x, y, z = np.meshgrid(np.linspace(xrange[0], xrange[1], xpts), np.linspace(yrange[0], yrange[1], ypts),
                              np.linspace(zrange[0], zrange[1], zpts))
        rays_o = torch.FloatTensor(np.stack([x, y, z], -1).reshape(-1, chunks, 3)).to(self.device)
        rays_d = torch.zeros_like(rays_o).to(self.device)
        radii = torch.ones_like(rays_o[..., 1]).to(self.device) * 0.0005
        near = torch.ones_like(radii).to(self.device)
        far = torch.ones_like(radii).to(self.device)
        rays_p = torch.ones_like(radii).to(self.device)
        mfilt = torch.ones((1, 8192)).to(self.device)
        mpp = torch.tensor([1.]).to(self.device)

        raws = []

        with torch.no_grad():
            for i in tqdm(range(0, rays_o.shape[0])):
                _, _, _, raw = self(rays_o[i].unsqueeze(0), rays_d[i].unsqueeze(0), rays_p[i].unsqueeze(0),
                                   mfilt,
                                   radii[i].unsqueeze(0),
                                   near[i].unsqueeze(0), far[i].unsqueeze(0), 3062, mpp)
                raws.append(torch.mean(raw, dim=2).cpu())
        sigma = torch.cat(raws, dim=1)
        sigma = np.maximum(sigma[..., -1].cpu().numpy(), 0)
        sigma = sigma.reshape(x.shape)
        print("Extracting mesh")
        print("fraction occupied", np.mean(np.array(sigma > sigma_threshold), dtype=np.float32))
        vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
        mcubes.export_obj(vertices, triangles, 'render_output.obj')
        return vertices, triangles


def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)