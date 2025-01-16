import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from pytorch_lightning.utilities import grad_norm
from torch import optim
from torch.optim import Optimizer
from torch.distributed.fsdp.wrap import wrap
from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b, \
    volumetric_scattering, get_sphere_intersection, distance_calculation, plot_grad_flow
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
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.ray_shape = config.ray_shape
        self.num_levels = config.num_levels
        self.num_samples = config.num_samples
        self.density_input = (config.max_deg - config.min_deg) * 3 * 2
        self.params_input = 3 + ((config.viewdirs_max_deg - config.viewdirs_min_deg) * 3 * 2)
        self.density_noise = config.density_noise
        self.rgb_padding = config.rgb_padding
        self.resample_padding = config.resample_padding
        self.density_bias = config.density_bias
        self.hidden = config.hidden
        self.wavelength = config.wavelength
        self.return_raw = return_raw
        self.automatic_optimization = False
        self.scene_bounding_sphere = 100.0

        self.loss_function = SARNeRFLoss(config.coarse_weight_decay)

        self.positional_encoding = PositionalEncoding(config.min_deg, config.max_deg)
        self.sdf_network = SDFNetwork(config.min_deg, config.max_deg, config.hidden, self.density_input, self.scene_bounding_sphere)
        self.param_network = ParamNetwork(config.min_deg, config.max_deg, config.hidden, self.density_input)

        self.alpha = nn.Parameter(data=torch.Tensor([1.]), requires_grad=True)
        self.alpha_pos = nn.Softplus()
        self.beta = nn.Parameter(data=torch.Tensor([.01]), requires_grad=True)
        self.beta_pos = nn.Softplus()

        _xavier_init(self)

    def forward(self, ray_o, ray_d, ray_p, mfilt, radii, _near, _far, nsam, mpp, return_occ=False):
        comp_rgbs = []
        distances = []
        sdfs = []
        near = torch.zeros_like(radii, device=self.device)#  * _near[:, None]
        far = torch.ones_like(radii, device=self.device)#  * _far[:, None]
        pulse_bins = torch.arange(nsam - 1, device=self.device)[None, :] * mpp[:, None] + _near[:, None]
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
                                                          stop_grad=True,
                                                          ray_shape=self.ray_shape)

            sdf = self.sdf_network(mean, var)
            gradients = self.sdf_network.gradient(mean).detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)

            # predict params and reshape for use later
            params = self.param_network(mean, var).reshape((ray_o.shape[0], -1, self.num_samples, 3))

            # Laplace distribution CDF
            # density = self.alpha_beta_pos(self.alpha) * (.5 + .5 * torch.sign(raw_density) * (1 - torch.exp(-torch.abs(raw_density) / self.alpha_beta_pos(self.beta))))
            density = self.laplace_cdf(sdf + self.density_bias).reshape((ray_o.shape[0], -1, self.num_samples, 1))
            # density = self.final_occ(self.density_activation(raw_density) + self.density_bias).reshape((ray_o.shape[0], -1, self.num_samples, 1))
            distance, acc, weights, alpha, trans = distance_calculation(density, t_vals, ray_d)
            distance = distance * (_far - _near) + _near
            with torch.no_grad():
                normals = torch.sum(weights.unsqueeze(-1) * normals, dim=2)
                normals = normals / normals.norm(2, -1, keepdim=True)

            comp_rp = volumetric_scattering(params, weights, normals, distance, ray_d, pulse_bins, self.wavelength)
            comp_rp = torch.fft.fft(torch.view_as_complex(comp_rp), mfilt.shape[-1], dim=-1)
            comp_pulse = torch.fft.ifft(mfilt * comp_rp, dim=-1)[..., :nsam]
            comp_pulse = comp_pulse / torch.std(comp_pulse, dim=-1)[..., None]
            comp_pulse = torch.nan_to_num(torch.view_as_real(comp_pulse))
            comp_rgbs.append(comp_pulse)
            distances.append(distance)
            sdfs.append(sdf.reshape((ray_o.shape[0], -1, self.num_samples, 1)))
        if return_occ:
            return density
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(comp_rgbs), torch.stack(distances), torch.stack(sdfs)

    def laplace_cdf(self, x):
        return (.00001 + self.alpha_pos(self.alpha)) * (.5 + .5 * torch.sign(-x) * (
                    1 - torch.exp(-torch.abs(x) / (.00001 + self.beta_pos(self.beta)))))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay)
        scheduler = MipLRDecay(optimizer, lr_init=self.config.lr_init, lr_final=self.config.lr_final,
                               max_steps=self.config.max_steps, lr_delay_steps=self.config.lr_delay_steps,
                               lr_delay_mult=self.config.lr_delay_mult)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        self.train_val_get(batch, True)

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, False)

    def train_val_get(self, batch, do_train = True):
        if do_train:
            opt = self.optimizers()
        ray_o, ray_d, ray_p, mfilt, radii, near, far, mpp, pulse_data = batch

        # Generate rays for random sampling
        pulses, dists, sdfs = self.forward(ray_o, ray_d, ray_p, mfilt, radii, near, far, pulse_data.shape[1], mpp)

        # Calculate out Eikonal loss
        eik_pts = torch.empty(ray_o.shape[1], 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).to(
            self.device)
        eik_near_pts = ray_o + ray_d * dists[-1][..., None] + torch.randn(ray_o.shape, device=ray_o.device)
        eik_pts = torch.cat([eik_pts, eik_near_pts.squeeze(0)], dim=0)
        grad_theta = self.sdf_network.gradient(eik_pts)

        train_loss, psnrs = self.loss_function(pulses, grad_theta, pulse_data)

        loss_name = 'train_loss' if do_train else 'val_loss'
        self.log_dict({loss_name: train_loss, 'coarse_psnr': torch.mean(psnrs[:-1]), 'fine_psnr': psnrs[-1],
                       'avg_psnr': torch.mean(psnrs), 'LR': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        if do_train:


            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            # self.clip_gradients(opt, gradient_clip_val=.5, gradient_clip_algorithm='norm')
            # plot_grad_flow(self.named_parameters())
            opt.step()
            self.lr_schedulers().step()

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self, norm_type=2)  # Compute 2-norm for each layer
        self.log_dict(norms)


class SDFNetwork(LightningModule):

    def __init__(self, min_deg: int = 0, max_deg: int = 4, hidden: int = 256, input_layer_sz: int = 6, scene_bounding_sphere: float = 1., *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.scene_bounding_sphere = scene_bounding_sphere
        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.sdf_net0 = nn.Sequential(
            nn.Linear(input_layer_sz, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.sdf_net1 = nn.Sequential(
            nn.Linear(input_layer_sz + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.final_sdf = nn.Sequential(
            nn.Linear(hidden, 1),
        )

    def forward(self, x, y=None):
        if y is None:
            enc = self.positional_encoding(x, y)
        else:
            enc = self.positional_encoding(x, y)[0]
        enc = enc.reshape((-1, enc.shape[-1]))
        # predict density
        new_encodings = self.sdf_net0(enc)
        new_encodings = torch.cat((new_encodings, enc), -1)
        new_encodings = self.sdf_net1(new_encodings)
        raw_sdf = self.final_sdf(new_encodings)

        # Clamp inside of bounding sphere
        sphere_sdf = self.scene_bounding_sphere - raw_sdf.norm(2, 1, keepdim=True)
        return torch.minimum(raw_sdf, sphere_sdf)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)
        y.requires_grad_(True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients


class ParamNetwork(LightningModule):

    def __init__(self, min_deg: int = 0, max_deg: int = 4, hidden: int = 256, input_layer_sz: int = 6, *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.param_net0 = nn.Sequential(
            nn.Linear(input_layer_sz, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.param_net1 = nn.Sequential(
            nn.Linear(input_layer_sz + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.final_param = nn.Sequential(
            nn.Linear(hidden, 3),
            nn.Softplus(),
        )

    def forward(self, x, y=None):
        if y is None:
            enc = self.positional_encoding(x, y)
        else:
            enc = self.positional_encoding(x, y)[0]
        enc = enc.reshape((-1, enc.shape[-1]))
        # predict density
        new_encodings = self.param_net0(enc)
        new_encodings = torch.cat((new_encodings, enc), -1)
        new_encodings = self.param_net1(new_encodings)
        return self.final_param(new_encodings)

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)