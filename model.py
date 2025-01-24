import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
from pytorch_lightning.utilities import grad_norm
from torch import optim, Tensor
from torch.optim import Optimizer
from torch.distributed.fsdp.wrap import wrap
from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b, \
    volumetric_scattering, distance_calculation, plot_grad_flow, get_sphere_intersections
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
    def __init__(self, config=None, return_raw: bool = False, mfilt: np.array = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config = config
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.ray_shape = config.ray_shape
        self.num_levels = config.num_levels
        self.num_samples = config.num_samples
        self.fine_samples = config.fine_samples
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
        self.scene_bounding_sphere = 1500.0
        self.temperature = .01
        self.mfilt = mfilt

        self.loss_function = SARNeRFLoss(config.coarse_weight_decay)

        self.positional_encoding = PositionalEncoding(config.min_deg, config.max_deg)
        self.sdf_network = SDFNetwork(config.min_deg, config.max_deg, config.hidden, self.density_input, self.scene_bounding_sphere)
        self.param_network = ParamNetwork(config.min_deg, config.max_deg, config.hidden, self.density_input)

        self.alpha = nn.Parameter(data=torch.Tensor([1.]), requires_grad=True)
        self.alpha_pos = nn.Softplus()
        self.beta = nn.Parameter(data=torch.Tensor([.1]), requires_grad=True)
        self.beta_pos = nn.Sigmoid()
        self.radar_scaling = nn.Parameter(data=torch.Tensor([1e3]), requires_grad=True)
        self.scaling_pos = nn.Softplus()

        self.approx_sign = nn.Sigmoid()

        _xavier_init(self)

    def forward(self, ray_o, ray_d, ray_p, radii, _near, nsam, mpp, return_occ=False):
        # Calculate sphere intersections for near and far
        sph_intersections = get_sphere_intersections(ray_o.reshape(-1, 3), ray_d.reshape(-1, 3), self.scene_bounding_sphere)
        _far = _near + nsam * mpp
        pulse_bins = torch.arange(nsam, dtype=torch.float32, device=self.device) * mpp + _near
        data_range = _far - _near
        near = torch.clamp_min((sph_intersections[:, 0].view(radii.shape) - _near) / data_range, 0.)
        far = torch.clamp_max((sph_intersections[:, 1].view(radii.shape) - _near) / data_range, 1.)
        # sample
        for l in range(2):
            if l == 0:
                t_vals, (mean, var) = sample_along_rays(ray_o, ray_d, radii.unsqueeze(2), self.num_samples,
                                                        near.unsqueeze(2), far.unsqueeze(2),
                                                        randomized=False, lindisp=False, ray_shape=self.ray_shape)
                sdf = self.sdf_network(mean, var)

                # Laplace distribution CDF
                density = self.laplace_cdf(sdf + self.density_bias).reshape((ray_o.shape[0], -1, self.num_samples, 1))
            else:
                t_vals, (mean, var) = resample_along_rays(ray_o, ray_d, radii.unsqueeze(2),
                                                          t_vals.to(ray_o.device),
                                                          weights.to(ray_o.device), randomized=False,
                                                          stop_grad=True, num_samples=self.fine_samples + 1, resample_padding=.001,
                                                          ray_shape=self.ray_shape)
                sdf = self.sdf_network(mean)

                # Laplace distribution CDF
                density = self.laplace_cdf(sdf + self.density_bias).reshape((ray_o.shape[0], -1, self.fine_samples, 1))


            t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
            t_dists = t_vals[..., 1:] - t_vals[..., :-1]
            delta = t_dists * torch.linalg.norm(ray_d[..., None, :], dim=-1)
            # Note that we're quietly turning density from [..., 0] to [...].
            density_delta = density[..., 0] * delta

            alpha = 1 - torch.exp(-density_delta)
            trans = torch.exp(-torch.cat([
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], dim=-1),
            ], dim=-1))
            # trans = torch.where(trans < .5, trans, 0.)
            weights = alpha * trans

        # predict params and reshape for use later
        params = self.param_network(mean).reshape((ray_o.shape[0], -1, self.fine_samples, 2))
        acc = torch.sum(weights, dim=-1)
        weight_std = torch.std(weights, dim=-1)
        distance = torch.sum(weights * t_mids, dim=-1) / acc
        distance = torch.clamp(torch.nan_to_num(distance), t_vals[..., 0], t_vals[..., -1])
        distance = distance * data_range + _near
        params = torch.sum(weights[..., None] * params, dim=-2) / acc[..., None]

        gradients = self.sdf_network.gradient(mean).detach()
        with torch.no_grad():
            normals = torch.sum(weights[..., None] * gradients, dim=-2) / acc[..., None]
            normals = torch.nan_to_num(normals / normals.norm(2, -1, keepdim=True), 0)

        bounce = normals * torch.sum(ray_d * normals, dim=-1)[..., None] * 2 - ray_d

        '''ref_model = (params[..., 0] * torch.sum(ray_d * normals, dim=-1) + params[..., 1] * torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)) ** params[..., 2])) / distance ** 2'''
        ref_model = (params[..., 0] * torch.sum(ray_d * normals, dim=-1) + params[..., 1] * torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * (1 + self.scaling_pos(self.radar_scaling))
        '''ref_model = (torch.sum(ray_d * normals, dim=-1) + torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)) ** params[..., 0])) / distance ** 2'''
        '''ref_model = (torch.sum(-ray_d * normals, dim=-1) + torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2'''
        # Calculate out expected phase as well
        # indices = torch.stack([torch.bucketize(distance[d], pulse_bins) for d in range(ref_model.shape[0])])
        # indices = torch.dstack([indices, indices])
        bound_dist = torch.abs(distance.unsqueeze(-1) - pulse_bins.unsqueeze(0))
        soft_buckets = torch.softmax(-bound_dist / self.temperature, dim=-1)
        # Calculate out expected phase as well
        ret = torch.view_as_real(ref_model * torch.exp(-2j * torch.pi / self.wavelength * distance * 2))
        comp_pulse = ret[:, :, None, :] * soft_buckets[:, :, :, None]
        comp_pulse = torch.sum(comp_pulse, dim=-3)
        comp_pulse = torch.view_as_complex(comp_pulse)
        comp_pulse = torch.fft.ifft(torch.fft.fft(comp_pulse, self.mfilt.shape[-1], dim=-1) * self.mfilt.to(self.device), dim=-1)[..., :nsam]
        comp_pulse = torch.view_as_real(comp_pulse)
        if return_occ:
            return density
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return comp_pulse, distance, sdf.reshape((ray_o.shape[0], -1, self.fine_samples, 1)), acc, weight_std

    def laplace_cdf(self, x):
        return (.00001 + self.alpha_pos(self.alpha)) * (.5 + .5 * (2 * self.approx_sign(-x) - 1) * (
                    1 - torch.exp(-torch.abs(x) / (.1 + self.beta_pos(self.beta)))))


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
        ray_o, ray_d, ray_p, radii, near, mpp, pulse_data = batch

        # Generate rays for random sampling
        pulses, dists, sdfs, acc, wstd = self.forward(ray_o, ray_d, ray_p, radii, near, pulse_data.shape[1], mpp)

        # Calculate out Eikonal loss
        eik_pts = (torch.rand(size=(ray_o.shape[1], 3)) * -self.scene_bounding_sphere * 2 + self.scene_bounding_sphere).to(
            self.device)
        eik_near_pts = ray_o + ray_d * dists[..., None]
        eik_pts = torch.cat([eik_pts, eik_near_pts.squeeze(0)], dim=0)
        grad_theta = self.sdf_network.gradient(eik_pts)

        train_loss, psnrs, eik_loss, acc_loss, std_loss = self.loss_function(pulses, grad_theta, acc, wstd, pulse_data)

        loss_name = 'train_loss' if do_train else 'val_loss'
        self.log_dict({loss_name: train_loss, 'eik_loss': eik_loss, 'psnr': psnrs,
                       'acc_loss': acc_loss, 'std_loss': std_loss, 'lr': self.lr_schedulers().get_last_lr()[0]}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        if do_train:


            opt.zero_grad()
            self.manual_backward(train_loss, retain_graph=True)
            # self.clip_gradients(opt, gradient_clip_val=.5, gradient_clip_algorithm='norm')
            # plot_grad_flow(self.named_parameters())
            opt.step()
            self.lr_schedulers().step()

    # def on_fit_start(self) -> None:
    #     self.logger.log_graph(self, self.example_input_array())

    def train(self, mode=True):
        self.randomized = self.init_randomized
        super().train(mode)
        return self

    def eval(self):
        self.randomized = False
        return super().eval()

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
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.sdf_net1 = nn.Sequential(
            nn.Linear(input_layer_sz + hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
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
        return self.final_sdf(new_encodings)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)
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
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.param_net1 = nn.Sequential(
            nn.Linear(input_layer_sz + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.final_param = nn.Sequential(
            nn.Linear(hidden, 2),
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
        return self.final_param(new_encodings) + 1e-3

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)