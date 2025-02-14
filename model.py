import math
from typing import Any
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib as mplib
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from simulib.platform_helper import SDRPlatform
from torch import optim, Tensor
from torch.optim import Optimizer
from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b, \
    plot_grad_flow, eikonal_loss, positional_encoding, laplace_cdf, error_bound_sample, uniform_sample
from pytorch_lightning import LightningModule
from util_modules import PositionalEncoding, MipLRDecay, NeRFLoss
from torch.distributions import Uniform
import diff_operators as diff_ops
from sdrparse import load
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
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)


class SARNeRF(LightningModule):
    def __init__(self, config=None, return_raw: bool = False, eik_loss_baseline: np.array = None,
                 scene_bbox: np.array = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config = config
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.num_levels = config.num_levels
        self.ray_samples = config.ray_samples
        self.num_samples = config.num_samples
        self.fine_samples = config.fine_samples
        self.density_input = config.encoder_size * 3 * 4
        self.density_noise = config.density_noise
        self.rgb_padding = config.rgb_padding
        self.resample_padding = config.resample_padding
        self.density_bias = config.density_bias
        self.hidden = config.hidden
        self.wavelength = config.wavelength
        self.return_raw = return_raw
        # self.automatic_optimization = False
        self.scene_bbox = scene_bbox.astype(np.float32)
        self.temperature = .1
        self.pulse_std = config.pulse_std
        self.eikonal_weight = config.eikonal_weight
        self.acc_weight = config.acc_weight
        self.sigma = config.encoder_sigma
        self.encoder_size = config.encoder_size

        sdr_f = load(config.sdr_file)
        rp = SDRPlatform(sdr_f, origin=config.data_center, channel=0, fs=sdr_f[0].fs)
        self.nsam, _, pulse_bins, _, near_range_s, _, fft_sz, _ = rp.getRadarParams(0., 0., 1)
        self.near_range = np.float32(near_range_s * c0)
        self.mpp = np.float32(c0 / rp.fs / 2)
        self.far_range = np.float32(self.near_range + self.nsam * self.mpp)
        self.mfilt = torch.tensor(sdr_f.genMatchedFilter(0, fft_len=fft_sz) * np.fft.fft(sdr_f[0].cal_chirp, fft_sz), dtype=torch.complex64)
        self.pulse_bins = torch.tensor(self.near_range + self.mpp * np.arange(self.nsam), dtype=torch.float32)
        self.az_bw = np.float32(rp.az_half_bw)
        self.el_bw = np.float32(rp.el_half_bw)

        self.sdf0 = nn.Sequential(
            nn.Linear(self.density_input, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )

        self.sdf1 = nn.Sequential(
            nn.Linear(config.hidden + self.density_input, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 1),
            nn.ReLU(),
        )

        self.param0 = nn.Sequential(
            nn.Linear(self.density_input, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, config.param_hidden),
            nn.SiLU(),
        )

        self.param1 = nn.Sequential(
            nn.Linear(config.param_hidden + self.density_input, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, 1),
            nn.Sigmoid(),
        )

        _xavier_init(self)

        # self.example_input_array = torch.tensor(np.random.rand(1, 5), dtype=torch.float32, requires_grad=True)

    def forward(self, ray_d, ray_o, ray_p, return_occ=False):
        # Calculate sphere intersections for near and far
        pos_enc = positional_encoding(torch.cat([ray_o, ray_d], dim=-1), self.sigma, self.encoder_size)
        distance = self.sdf0(pos_enc)
        distance = torch.cat([distance, pos_enc], dim=-1)
        distance = self.sdf1(distance) * self.mpp + self.near_range

        # z_vals, _ = error_bound_sample(ray_d, ray_o, self.get_beta(), self.num_samples, near, far, self.sdf_network)
        pts = ray_o + ray_d * distance

        # predict params and reshape for use later
        param_enc = positional_encoding(torch.cat([pts, ray_d], dim=-1), self.sigma, self.encoder_size)
        params = self.param0(param_enc)
        params = torch.cat([params, param_enc], dim=-1)
        params = self.param1(params) + .001

        # Phong reflection formulation, phong to be added later
        '''ref_model = (torch.sum(ray_d * normals, dim=-1) + torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * ray_p.squeeze(-1)'''
        ref_model = params / distance ** 4 * ray_p
        # ref_model = ray_p.squeeze(-1) / torch.square(distance)

        # Get soft buckets to preserve gradients across histogramming step
        pulse_bins = self.pulse_bins.to(self.device)
        bound_dist = torch.abs(distance - pulse_bins.unsqueeze(0))
        soft_buckets = torch.softmax(-bound_dist / self.temperature, dim=-1)

        # Calculate out expected phase as well
        ret = torch.view_as_real(ref_model.squeeze(-1) * torch.exp(-4j * torch.pi / self.wavelength * distance.squeeze(-1)))

        # Soft histogram step, along with applying filtered chirp in frequency domain
        comp_pulse = ret[:, :, None, :] * soft_buckets[:, :, :, None]
        comp_pulse = torch.sum(comp_pulse, dim=-3)
        comp_pulse = torch.view_as_complex(comp_pulse)
        comp_pulse = torch.fft.ifft(torch.fft.fft(comp_pulse, self.mfilt.shape[-1], dim=-1) *
                                    self.mfilt.to(self.device), dim=-1)[..., :self.nsam]
        comp_pulse = torch.view_as_real(comp_pulse) / self.pulse_std
        if return_occ:
            return distance
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return comp_pulse, distance, pts
            # return sdf.reshape((ray_o.shape[0], -1, self.fine_samples, 1))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay, foreach=False)
        scheduler = MipLRDecay(optimizer, lr_init=self.config.lr_init, lr_final=self.config.lr_final,
                               max_steps=self.config.max_steps, lr_delay_steps=self.config.lr_delay_steps,
                               lr_delay_mult=self.config.lr_delay_mult)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        pts = self.train_val_get(batch, True)
        if self.global_step % 100 == 0 and self.trainer.is_global_zero:
            mplib.use('Agg')
            pts_np = pts.cpu().data.numpy()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2])
            ax.set_xlim(self.scene_bbox[0, 0], self.scene_bbox[1, 0])
            ax.set_ylim(self.scene_bbox[0, 1], self.scene_bbox[1, 1])
            ax.set_zlim(self.scene_bbox[0, 2], self.scene_bbox[1, 2])
            self.logger.experiment.add_figure('Density Plot', fig, global_step=self.global_step)
            plt.close(fig)
        if self.automatic_optimization:
            return self.loss

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(self.loss, retain_graph=True)
            self.loss = None
            opt.step()
        self.lr_schedulers().step()


    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, False)

    def train_val_get(self, batch, do_train = True):
        ray_d, ray_o, ray_p, target_data = batch

        pulses, dists, pts = self.forward(ray_d, ray_o, ray_p)

        loss = torch.square(
            torch.abs(torch.view_as_complex(target_data)) - torch.abs(torch.view_as_complex(pulses))).mean()

        with torch.no_grad():
            psnr = mse_to_psnr(((target_data - pulses) ** 2).mean())

        self.loss = loss

        loss_name = 'train_loss' if do_train else 'val_loss'
        self.log_dict({loss_name: loss, 'psnr': psnr, 'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True, sync_dist=True)

        return pts

    # def on_fit_start(self) -> None:
    #     self.logger.log_graph(self, self.example_input_array())

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self, norm_type=2)  # Compute 2-norm for each layer
        self.log_dict(norms)

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)