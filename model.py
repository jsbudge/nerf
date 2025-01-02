import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
import torch.nn as nn
from torch import optim

from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b
from pytorch_lightning import LightningModule
from util_modules import PositionalEncoding, MipLRDecay, NeRFLoss
from tqdm import tqdm
import mcubes
import numpy as np

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
c0 = 299792458.0
DTR = np.pi / 180





class MipNeRF(LightningModule):
    def __init__(self,
                 config=None,
                 return_raw=False,
                 ):
        super(MipNeRF, self).__init__()
        self.config = config
        self.use_viewdirs = config.use_viewdirs
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.ray_shape = config.ray_shape
        self.white_bkgd = config.white_bkgd
        self.num_levels = config.num_levels
        self.num_samples = config.num_samples
        self.density_input = (config.max_deg - config.min_deg) * 3 * 2
        self.rgb_input = 3 + ((config.viewdirs_max_deg - config.viewdirs_min_deg) * 3 * 2)
        self.density_noise = config.density_noise
        self.rgb_padding = config.rgb_padding
        self.resample_padding = config.resample_padding
        self.density_bias = config.density_bias
        self.hidden = config.hidden
        self.return_raw = return_raw
        self.automatic_optimization = False
        self.density_activation = nn.Softplus()

        self.loss_function = NeRFLoss(config.coarse_weight_decay)

        self.positional_encoding = PositionalEncoding(config.min_deg, config.max_deg)
        self.density_net0 = nn.Sequential(
            nn.Linear(self.density_input, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
        )
        self.density_net1 = nn.Sequential(
            nn.Linear(self.density_input + config.hidden, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
            nn.Linear(config.hidden, config.hidden),
            nn.ReLU(True),
        )
        self.final_density = nn.Sequential(
            nn.Linear(config.hidden, 1),
        )

        input_shape = config.hidden
        if self.use_viewdirs:
            input_shape = config.num_samples

            self.rgb_net0 = nn.Sequential(
                nn.Linear(config.hidden, config.hidden)
            )
            self.viewdirs_encoding = PositionalEncoding(config.viewdirs_min_deg, config.viewdirs_max_deg)
            self.rgb_net1 = nn.Sequential(
                nn.Linear(config.hidden + self.rgb_input, config.num_samples),
                nn.ReLU(True),
            )
        self.final_rgb = nn.Sequential(
            nn.Linear(input_shape, 3),
            nn.Sigmoid()
        )
        _xavier_init(self)

    def forward(self, rays):
        comp_rgbs = []
        distances = []
        accs = []
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(rays.origins, rays.directions, rays.radii, self.num_samples,
                                                        rays.near, rays.far, randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(rays.origins, rays.directions, rays.radii,
                                                          t_vals.to(rays.origins.device),
                                                          weights.to(rays.origins.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((-1, self.num_samples, 1))

            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(rays.viewdirs.to(self.device))
                viewdirs = torch.cat((viewdirs, rays.viewdirs.to(self.device)), -1)
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                new_encodings = self.rgb_net0(new_encodings)
                new_encodings = torch.cat((new_encodings, viewdirs), -1)
                new_encodings = self.rgb_net1(new_encodings)
            raw_rgb = self.final_rgb(new_encodings).reshape((-1, self.num_samples, 3))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, alpha = volumetric_rendering(rgb, density, t_vals, rays.directions.to(rgb.device), self.white_bkgd)
            comp_rgbs.append(comp_rgb)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

    def render_image(self, rays, height, width, chunks=8192):
        """
        Return image, disparity map, accumulated opacity (shaped to height x width) created using rays as input.
        Rays should be all of the rays that correspond to this one single image.
        Batches the rays into chunks to not overload memory of device
        """
        length = rays[0].shape[0]
        rgbs = []
        dists = []
        accs = []
        with torch.no_grad():
            for i in range(0, length, chunks):
                # put chunk of rays on device
                chunk_rays = namedtuple_map(lambda r: r[i:i+chunks].to(self.device), rays)
                rgb, distance, acc = self(chunk_rays)
                rgbs.append(rgb[-1].cpu())
                dists.append(distance[-1].cpu())
                accs.append(acc[-1].cpu())

        rgbs = to8b(torch.cat(rgbs, dim=0).reshape(height, width, 3).numpy())
        dists = torch.cat(dists, dim=0).reshape(height, width).numpy()
        accs = torch.cat(accs, dim=0).reshape(height, width).numpy()
        return rgbs, dists, accs

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay)
        scheduler = MipLRDecay(optimizer, lr_init=self.config.lr_init, lr_final=self.config.lr_final,
                               max_steps=self.config.max_steps, lr_delay_steps=self.config.lr_delay_steps,
                               lr_delay_mult=self.config.lr_delay_mult)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def train_val_get(self, batch, batch_idx, kind='train'):
        rays, pixels = batch

        # Generate rays for random sampling
        rgb, _, _ = self.forward(rays)

        train_loss, psnrs = self.loss_function(rgb, pixels, rays.lossmult.to(self.device))

        self.log_dict({f'{kind}_loss': train_loss, 'coarse_psnr': torch.mean(psnrs[:-1]), 'fine_psnr': psnrs[-1],
                       'avg_psnr': torch.mean(psnrs), 'LR': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)


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
            nn.ReLU(),
        )]
        net0 = net0 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ) for _ in range(D)]

        self.density_net0 = nn.Sequential(*net0)

        net1 = [nn.Sequential(
            nn.Linear(self.input_ch + self.hidden, self.hidden),
            nn.ReLU(),
        )]
        net1 = net1 + [nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        ) for _ in range(D)]

        self.density_net1 = nn.Sequential(*net1)

        self.final_density = nn.Sequential(
            nn.Linear(self.hidden, 1),
        )

        self.final_norm = nn.Sequential(
            nn.Linear(self.hidden, 2),
            nn.Sigmoid()
        )

        self.final_rcs = nn.Sequential(
            nn.Linear(self.hidden, 3),
            nn.Sigmoid()
        )

        self.norm_net1 = nn.Sequential(
            nn.Linear(self.input_ch + self.hidden, self.hidden),
            nn.ReLU(),
        )

        self.norm_net0 = nn.Sequential(
            nn.Linear(self.input_ch, self.hidden),
            nn.ReLU(),
        )

        _xavier_init(self)

    def forward(self, rays_o, rays_d, rays_p, mfilt, radii, _near, _far, nsam, mpp):
        comp_rgbs = []
        distances = []
        accs = []
        # scaling = _far - _near
        near = torch.ones_like(radii, device=self.device) * _near
        far = torch.ones_like(radii, device=self.device) * _far

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
            raw_rcs = self.final_rcs(new_encodings).reshape((-1, radii.shape[-1], self.num_samples, 3))

            norm_encodings = self.norm_net0(samples_enc)
            norm_encodings = torch.cat((norm_encodings, samples_enc), -1)
            norm_encodings = self.norm_net1(norm_encodings)
            raw_norms = self.final_norm(norm_encodings).reshape((-1, radii.shape[1], self.num_samples, 2))


            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype,
                                                               device=self.device)
            raw_rcs = raw_rcs * (1 + 2 * self.rgb_padding) - self.rgb_padding

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
        scheduler = MipLRDecay(optimizer, lr_init=self.params['LR'], max_steps=200000, lr_final=self.params['LR_final'], lr_delay_steps=2500, lr_delay_mult=.1)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
            self.log('LR', sch.get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def train_val_get(self, batch, batch_idx, kind='train'):
        rays_o, rays_d, rays_p, mfilt, radii, near, far, mpp, target = batch

        # Generate rays for random sampling
        rgb, disp, acc = self.forward(rays_o, rays_d, rays_p, mfilt, radii, near, far, target.shape[1], mpp)

        train_loss = self.loss_function(rgb, target)

        self.log_dict({f'{kind}_loss': train_loss, 'LR': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
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
        near = 0.
        far = 1.
        rays_p = torch.ones_like(radii).to(self.device)
        mfilt = torch.ones((1, 8192)).to(self.device)
        mpp = torch.tensor([1.]).to(self.device)

        raws = []

        with torch.no_grad():
            for i in tqdm(range(0, rays_o.shape[0])):
                _, _, _, raw = self(rays_o[i].unsqueeze(0), rays_d[i].unsqueeze(0), rays_p[i].unsqueeze(0),
                                   mfilt,
                                   radii[i].unsqueeze(0),
                                   near, far, 3062, mpp)
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