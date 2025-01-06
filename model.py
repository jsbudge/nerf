import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch import optim

from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b, \
    volumetric_scattering
from pytorch_lightning import LightningModule
from util_modules import PositionalEncoding, MipLRDecay, NeRFLoss, SARNeRFLoss
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
    def __init__(self, config=None, return_raw: bool = False, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config = config
        self.use_viewdirs = config.use_viewdirs
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.ray_shape = config.ray_shape
        self.num_levels = config.num_levels
        self.num_samples = config.num_samples
        self.density_input = (config.max_deg - config.min_deg) * 3 * 2
        self.rgb_input = 3 + ((config.viewdirs_max_deg - config.viewdirs_min_deg) * 3 * 2)
        self.density_noise = config.density_noise
        self.rgb_padding = config.rgb_padding
        self.resample_padding = config.resample_padding
        self.density_bias = config.density_bias
        self.hidden = config.hidden
        self.wavelength = config.wavelength
        self.return_raw = return_raw
        self.automatic_optimization = False
        self.density_activation = nn.Softplus()

        self.loss_function = SARNeRFLoss(config.coarse_weight_decay)

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
            nn.LayerNorm(config.hidden),
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
        self.final_angle = nn.Sequential(
            nn.Linear(input_shape, 2),
            nn.Sigmoid()
        )
        _xavier_init(self)

    def forward(self, ray_o, ray_d, ray_p, mfilt, radii, _near, _far, nsam, mpp):
        comp_rgbs = []
        distances = []
        accs = []
        near = torch.ones_like(radii, device=self.device) * _near[:, None]
        far = torch.ones_like(radii, device=self.device) * _far[:, None]
        pulse_bins = torch.arange(nsam, device=self.device)[None, :] * mpp[:, None] + _near[:, None]
        for l in range(self.num_levels):
            # sample
            if l == 0:  # coarse grain sample
                t_vals, (mean, var) = sample_along_rays(ray_o, ray_d, radii.unsqueeze(2), self.num_samples,
                                                        near.unsqueeze(2), far.unsqueeze(2), randomized=self.randomized, lindisp=False,
                                                        ray_shape=self.ray_shape)
            else:  # fine grain sample/s
                t_vals, (mean, var) = resample_along_rays(ray_o, ray_d, radii.unsqueeze(2),
                                                          t_vals.to(self.device),
                                                          weights.to(self.device), randomized=self.randomized,
                                                          stop_grad=True, resample_padding=self.resample_padding,
                                                          ray_shape=self.ray_shape)
            # do integrated positional encoding of samples
            samples_enc = self.positional_encoding(mean, var)[0]
            samples_enc = samples_enc.reshape([-1, samples_enc.shape[-1]])

            # predict density
            new_encodings = self.density_net0(samples_enc)
            new_encodings = torch.cat((new_encodings, samples_enc), -1)
            new_encodings = self.density_net1(new_encodings)
            raw_density = self.final_density(new_encodings).reshape((ray_o.shape[0], -1, self.num_samples, 1))

            # predict rgb
            if self.use_viewdirs:
                #  do positional encoding of viewdirs
                viewdirs = self.viewdirs_encoding(ray_d)
                viewdirs = torch.cat((viewdirs, ray_d), -1)
                viewdirs = torch.tile(viewdirs[:, None, :], (1, self.num_samples, 1))
                viewdirs = viewdirs.reshape((-1, viewdirs.shape[-1]))
                new_encodings = self.rgb_net0(new_encodings)
                new_encodings = torch.cat((new_encodings, viewdirs), -1)
                new_encodings = self.rgb_net1(new_encodings)
            raw_rgb = self.final_rgb(new_encodings).reshape((ray_o.shape[0], -1, self.num_samples, 3))
            angle = self.final_angle(new_encodings).reshape((ray_o.shape[0], -1, self.num_samples, 2))

            # Add noise to regularize the density predictions if needed.
            if self.randomized and self.density_noise:
                raw_density += self.density_noise * torch.rand(raw_density.shape, dtype=raw_density.dtype, device=raw_density.device)

            # volumetric rendering
            rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            # angle = raw_angles * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rp, distance, acc, weights, alpha = volumetric_scattering(rgb, angle, density, t_vals, ray_d,
                                                                           pulse_bins,
                                                                           self.wavelength)
            comp_rp = torch.fft.fft(torch.view_as_complex(comp_rp), mfilt.shape[-1], dim=-1)
            comp_pulse = torch.fft.ifft(mfilt * comp_rp, dim=-1)[..., :nsam]
            comp_pulse = comp_pulse / torch.std(comp_pulse, dim=-1)[..., None]
            comp_pulse = torch.view_as_real(comp_pulse)
            comp_rgbs.append(comp_pulse)
            distances.append(distance)
            accs.append(acc)
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(accs)

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
        ray_o, ray_d, ray_p, mfilt, radii, near, far, mpp, pulse_data = batch

        # Generate rays for random sampling
        rgb, _, _ = self.forward(ray_o, ray_d, ray_p, mfilt, radii, near, far, pulse_data.shape[1], mpp)

        train_loss, psnrs = self.loss_function(rgb, pulse_data)

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

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)