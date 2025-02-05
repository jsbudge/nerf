import math
from typing import Any
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib as mplib
from pytorch_lightning.utilities import grad_norm
from simulib.platform_helper import SDRPlatform
from torch import optim, Tensor
from torch.optim import Optimizer
from utils import sample_along_rays, resample_along_rays, volumetric_rendering, namedtuple_map, to8b, \
    plot_grad_flow, eikonal_loss
from pytorch_lightning import LightningModule
from util_modules import PositionalEncoding, MipLRDecay, NeRFLoss
from torch.distributions import Uniform
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
            # nn.init.he_(module.weight)


class SARNeRF(LightningModule):
    def __init__(self, config=None, return_raw: bool = False, eik_loss_baseline: np.array = None,
                 scene_bbox: np.array = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.config = config
        self.init_randomized = config.randomized
        self.randomized = config.randomized
        self.ray_shape = config.ray_shape
        self.num_levels = config.num_levels
        self.ray_samples = config.ray_samples
        self.num_samples = config.num_samples
        self.fine_samples = config.fine_samples
        self.density_input = (config.max_deg - config.min_deg) * 3 * 2
        self.density_noise = config.density_noise
        self.rgb_padding = config.rgb_padding
        self.resample_padding = config.resample_padding
        self.density_bias = config.density_bias
        self.hidden = config.hidden
        self.wavelength = config.wavelength
        self.return_raw = return_raw
        self.automatic_optimization = False
        self.scene_bbox = scene_bbox.astype(np.float32)
        self.temperature = .01
        self.pulse_std = config.pulse_std
        self.eikonal_weight = config.eikonal_weight
        self.acc_weight = config.acc_weight

        sdr_f = load(config.sdr_file)
        rp = SDRPlatform(sdr_f, origin=config.data_center, channel=0, fs=sdr_f[0].fs)
        self.nsam, _, pulse_bins, _, near_range_s, _, fft_sz, _ = rp.getRadarParams(5., .75, 1)
        self.near_range = np.float32(near_range_s * c0)
        self.near_range_s = np.float32(near_range_s)
        self.mpp = np.float32(pulse_bins[1] - pulse_bins[0])
        self.far_range = np.float32(near_range_s * c0 + self.nsam * self.mpp)
        self.mfilt = torch.tensor(sdr_f.genMatchedFilter(0, fft_len=fft_sz) * np.fft.fft(sdr_f[0].cal_chirp, fft_sz), dtype=torch.complex64)
        self.pulse_bins = torch.tensor(pulse_bins * 2, dtype=torch.float32)
        self.az_bw = np.float32(rp.az_half_bw)
        self.el_bw = np.float32(rp.el_half_bw)

        # Calculate radius of cone
        self.radii = self.far_range * np.tan(min(self.az_bw, self.el_bw)) / self.ray_samples

        tran_gain_db = 25.
        rec_gain_db = 25.
        amp_gain_db = 30.
        tran_power_watt = 100.
        self.radar_coeff = np.float32(
                c0 ** 2 / sdr_f[0].fc ** 2 * tran_power_watt * 10 ** (
                (tran_gain_db + 2.15) / 10) * 10 ** (
                        (rec_gain_db + 2.15) / 10) *
                10 ** ((amp_gain_db + 2.15) / 10) / (4 * np.pi) ** 3)

        # Distribute rays in a regular (randomized, but unchanging) grid
        '''self.az_vals = torch.empty((1, self.ray_samples, 1), dtype=torch.float32).uniform_(-self.az_bw * 2, self.az_bw * 2)
        self.el_vals = torch.empty((1, self.ray_samples, 1), dtype=torch.float32).uniform_(-self.el_bw * 2, self.el_bw * 2)
        self.ray_p = torch.square(torch.sinc(self.az_vals / self.az_bw)) * torch.square(
            torch.sinc(self.el_vals / self.el_bw)) * self.radar_coeff'''
        self.az_vals = torch.distributions.Uniform(-self.az_bw * 2, self.az_bw * 2)
        self.el_vals = torch.distributions.Uniform(-self.el_bw * 2, self.el_bw * 2)


        self.use_eik_base = False
        if eik_loss_baseline is not None:
            self.use_eik_base = True
            self.eik_base = torch.tensor(eik_loss_baseline, dtype=torch.float32)
        else:
            self.use_eik_base = False

        self.sdf_network = SDFNetwork(config.min_deg, config.max_deg, config.hidden, self.density_input)

        self.param0 = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
        )

        self.param1 = nn.Sequential(
            nn.Linear(config.hidden + self.density_input, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, config.hidden),
            nn.SiLU(),
            nn.Linear(config.hidden, 2),
            nn.Softplus(),
        )

        self.beta = nn.Parameter(data=torch.Tensor([1.]), requires_grad=True)
        self.beta_pos = nn.Softplus()
        self.approx_sign = nn.Sigmoid()

        _xavier_init(self)

        # self.example_input_array = torch.tensor(np.random.rand(1, 5), dtype=torch.float32, requires_grad=True)

    def forward(self, ray_info, return_occ=False):
        ray_d, ray_o, ray_p = self.generate_rays(ray_info)
        # Calculate sphere intersections for near and far
        bb_enter, bb_leave, hits = self.bb_intersect(ray_o.reshape(-1, 3), ray_d.reshape(-1, 3))
        pulse_bins = self.pulse_bins.to(self.device)
        near = torch.clamp_min(bb_enter.view(ray_p.shape), self.near_range)
        far = torch.clamp_max(bb_leave.view(ray_p.shape), self.far_range)

        hit_mask = hits.squeeze(-1)
        ray_d = ray_d[:, hit_mask]
        ray_o = ray_o[:, hit_mask]
        ray_p = ray_p[:, hit_mask]
        near = near[:, hit_mask]
        far = far[:, hit_mask]
        radii = torch.ones_like(near) * self.radii

        # sample
        for l in range(2):
            if l == 0:
                t_vals, (mean, var) = sample_along_rays(ray_o, ray_d, radii, self.num_samples, near, far,
                                                        randomized=False, lindisp=False, ray_shape=self.ray_shape)
                _, sdf = self.sdf_network(mean)
            else:
                t_vals, (mean, var) = resample_along_rays(ray_o, ray_d, radii,
                                                          t_vals.to(self.device), weights.to(self.device),
                                                          randomized=False, stop_grad=True,
                                                          num_samples=self.fine_samples + 1, resample_padding=0.001,
                                                          ray_shape=self.ray_shape)
                raw_features, sdf = self.sdf_network(mean)

            # Laplace distribution CDF converts sdf to a density
            density = self.laplace_cdf(sdf + self.density_bias).reshape((ray_o.shape[0], -1, self.num_samples if l == 0 else self.fine_samples, 1))

            # Volumetric rendering equation - this calculates the transmission through the volume
            # using density calculated by network
            t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
            delta = t_vals[..., 1:] - t_vals[..., :-1]
            # Note that we're quietly turning density from [..., 0] to [...].
            density_delta = density[..., 0] * delta

            alpha = 1 - torch.exp(-density_delta)
            trans = torch.exp(-torch.cat([
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], dim=-1),
            ], dim=-1))
            weights = alpha * trans

        # predict params and reshape for use later
        '''enc_mean = self.sdf_network.positional_encoding(mean, var)[0]
        params = self.param0(raw_features)
        params = torch.cat([params, enc_mean.reshape((-1, enc_mean.shape[-1]))], dim=-1)
        params = self.param1(params).reshape((ray_o.shape[0], -1, self.fine_samples, 2))'''
        acc = torch.sum(weights, dim=-1)
        distance = torch.sum(weights * t_mids, dim=-1) / acc
        distance = torch.clamp(torch.nan_to_num(distance), t_vals[..., 0], t_vals[..., -1])
        # params = torch.sum(weights[..., None] * params, dim=-2) / acc[..., None]

        # Detach gradients from autograd so they don't influence backpropagation
        gradients = self.sdf_network.gradient(mean).detach()
        with torch.no_grad():
            normals = torch.sum(weights[..., None] * gradients, dim=-2) / acc[..., None]
            normals = torch.nan_to_num(-normals / normals.norm(2, -1, keepdim=True), 0)
        bounce = normals * torch.sum(ray_d * normals, dim=-1)[..., None] * 2 - ray_d

        # Phong reflection formulation, phong to be added later
        ref_model = (torch.sum(ray_d * normals, dim=-1) + torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * ray_p.squeeze(-1)
        '''ref_model = (params[..., 0] * torch.sum(ray_d * normals, dim=-1) + params[..., 1] * torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * ray_p.squeeze(-1)'''
        # ref_model = ray_p.squeeze(-1) / torch.square(distance)

        # Get soft buckets to preserve gradients across histogramming step
        bound_dist = torch.abs(distance.unsqueeze(-1) - pulse_bins.unsqueeze(0))
        soft_buckets = torch.softmax(-bound_dist / self.temperature, dim=-1)

        # Calculate out expected phase as well
        ret = torch.view_as_real(ref_model * torch.exp(-2j * torch.pi / self.wavelength * distance * 2))

        # Soft histogram step, along with applying filtered chirp in frequency domain
        comp_pulse = ret[:, :, None, :] * soft_buckets[:, :, :, None]
        comp_pulse = torch.sum(comp_pulse, dim=-3)
        comp_pulse = torch.view_as_complex(comp_pulse)
        comp_pulse = torch.fft.ifft(torch.fft.fft(comp_pulse, self.mfilt.shape[-1], dim=-1) *
                                    self.mfilt.to(self.device), dim=-1)[..., :self.nsam]
        comp_pulse = torch.view_as_real(comp_pulse) / self.pulse_std
        if return_occ:
            return density
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return comp_pulse, distance, sdf.reshape((ray_o.shape[0], -1, self.fine_samples, 1)), acc, weights

    def laplace_cdf(self, x):
        beta = .0001 + self.beta_pos(self.beta)
        return (.5 + .5 * x.sign() * torch.expm1(-x.abs() / beta)) / beta
        # return (.5 + .5 * (2 * self.approx_sign(-x) - 1) * torch.expm1(-torch.abs(x) / beta)) / beta

    def generate_rays(self, ray_info):
        # Generate random rays for sampling
        az_vals = self.az_vals.rsample((1, self.ray_samples, 1)).to(self.device)
        el_vals = self.el_vals.rsample((1, self.ray_samples, 1)).to(self.device)
        ray_p = torch.square(torch.sinc(az_vals / self.az_bw)) * torch.square(
            torch.sinc(el_vals / self.el_bw)) * self.radar_coeff
        az_vals = az_vals + ray_info[..., 3]
        el_vals = el_vals + ray_info[..., 4]
        ray_d = torch.cat(
            [torch.sin(az_vals) * torch.cos(el_vals), torch.cos(az_vals) * torch.cos(el_vals), -torch.sin(el_vals)],
            dim=-1)
        ray_o = torch.broadcast_to(ray_info[..., :3], ray_d.shape)
        return ray_d, ray_o, ray_p


    def sample_density_function(self, x_range: list = None, y_range: list = None, z_range: list = None,
                                npts: list = None, pts: Tensor = None) -> (Tensor, Tensor):
        # If we don't want to give a set of points, generate a cube based on ranges given
        if pts is None:
            x, y, z = torch.meshgrid(torch.linspace(x_range[0], x_range[1], npts[0], dtype=torch.float32).to(self.device),
                                     torch.linspace(y_range[0], y_range[1], npts[1], dtype=torch.float32).to(self.device),
                                     torch.linspace(z_range[0], z_range[1], npts[2], dtype=torch.float32).to(self.device))
            pts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1)
            r_shape = x.shape
        else:
            r_shape = pts.shape[:-1]
        # Adding normals in here to give another thing to plot
        normals = self.sdf_network.gradient(pts).detach()
        with torch.no_grad():
            _, sdf = self.sdf_network(pts)
            normals = torch.nan_to_num(normals / normals.norm(2, -1, keepdim=True), 0)
        return sdf.reshape(r_shape), normals.reshape((*r_shape, 3))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay)
        scheduler = MipLRDecay(optimizer, lr_init=self.config.lr_init, lr_final=self.config.lr_final,
                               max_steps=self.config.max_steps, lr_delay_steps=self.config.lr_delay_steps,
                               lr_delay_mult=self.config.lr_delay_mult)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        self.train_val_get(batch, True)
        if self.global_step % 100 == 0:
            mplib.use('Agg')
            pts = torch.cat([self.eik_base, torch.rand(size=(self.eik_base.shape[0], 3)) *
                             np.diff(self.scene_bbox, axis=0) + self.scene_bbox[0]], dim=0).to(self.device)

            sdf_test, norm_test = self.sample_density_function(pts=pts)
            sdf_test = self.laplace_cdf(sdf_test).cpu().data.numpy().flatten()
            pts_np = pts.cpu().data.numpy()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=sdf_test)
            ax.set_xlim(self.scene_bbox[0, 0], self.scene_bbox[1, 0])
            ax.set_ylim(self.scene_bbox[0, 1], self.scene_bbox[1, 1])
            ax.set_zlim(self.scene_bbox[0, 2], self.scene_bbox[1, 2])
            self.logger.experiment.add_figure('Density Plot', fig, global_step=self.global_step)
            plt.close(fig)

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, False)

    def train_val_get(self, batch, do_train = True):
        if do_train:
            opt = self.optimizers()
            opt.zero_grad()
        ray_info, target_data = batch

        # Generate rays for random sampling
        pulses, dists, sdfs, acc, weights = self.forward(ray_info)

        # Calculate out Eikonal loss
        if self.use_eik_base:
            eik_pts = (torch.rand(size=(self.eik_base.shape[0], 3)) * np.diff(self.scene_bbox, axis=0) + self.scene_bbox[0]).to(self.device)
            eik_loss = (torch.exp(-10. * torch.abs(self.sdf_network(eik_pts)[1])).mean() + torch.square(self.sdf_network(self.eik_base.to(self.device))[1]).mean())
            eik_pts = torch.cat([eik_pts, self.eik_base.to(self.device)], dim=0)
        else:
            eik_pts = (torch.rand(size=(ray_info.shape[1] * 2, 3)) * np.diff(self.scene_bbox, axis=0) + self.scene_bbox[0]).to(
                self.device)
            eik_loss = 0.
        eik_pts.requires_grad_(True)
        sdf_output = self.sdf_network(eik_pts)[1]
        grad_theta = torch.autograd.grad(sdf_output, eik_pts,
                                         grad_outputs=torch.ones_like(sdf_output, requires_grad=True),
                                         retain_graph=True, create_graph=True)[0]
        eik_loss = eikonal_loss(grad_theta) * .0001 + eik_loss


        # Compute cosine similarity between pulses
        # loss = torch.nan_to_num(torch.sum(target_data * pulses, dim=-2) / (torch.linalg.norm(target_data, dim=-2) * torch.linalg.norm(pulses, dim=-2)), 1e9)
        # loss = 1 - torch.mean(torch.abs(loss))
        loss = ((target_data - pulses) ** 2).mean()
        # loss = torch.mean(torch.square(torch.abs(torch.view_as_complex(target_data)) - torch.abs(torch.view_as_complex(pulses))))
        with torch.no_grad():
            psnr = mse_to_psnr(torch.mean(torch.square(torch.abs(torch.view_as_complex(target_data)) - torch.abs(torch.view_as_complex(pulses)))))


        acc_loss = ((1 - acc) ** 2).mean()
        loss = loss + self.eikonal_weight * eik_loss + self.acc_weight * acc_loss

        loss_name = 'train_loss' if do_train else 'val_loss'
        self.log_dict({loss_name: loss, 'eik_loss': eik_loss, 'psnr': psnr,
                       'acc_loss': acc_loss, 'lr': self.lr_schedulers().get_last_lr()[0],
                       'density_std': self.sdf_network(eik_pts)[1].std()}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        if do_train:
            self.manual_backward(loss, retain_graph=True)
            # self.clip_gradients(opt, gradient_clip_val=50, gradient_clip_algorithm='norm')
            # plot_grad_flow(self.named_parameters())
            opt.step()
            self.lr_schedulers().step()

    # def on_fit_start(self) -> None:
    #     self.logger.log_graph(self, self.example_input_array())

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        norms = grad_norm(self, norm_type=2)  # Compute 2-norm for each layer
        self.log_dict(norms)

    def bb_intersect(self, ray_o, ray_d):
        with torch.no_grad():
            tmin = (torch.tensor(self.scene_bbox[0], device=self.device) - ray_o) / ray_d
            tmax = (torch.tensor(self.scene_bbox[1], device=self.device) - ray_o) / ray_d
            t1 = torch.min(tmin, tmax)
            t2 = torch.max(tmin, tmax)
            tnear = torch.max(t1, dim=-1)[0]
            tfar = torch.min(t2, dim=-1)[0]
            hits = torch.logical_and(tnear - tfar < 0, tfar >= 0)
        return tnear, tfar, torch.logical_and(hits, tnear < self.far_range)



class SDFNetwork(LightningModule):

    def __init__(self, min_deg: int = 0, max_deg: int = 4, hidden: int = 256, input_layer_sz: int = 6,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.positional_encoding = PositionalEncoding(min_deg, max_deg)
        self.hypernetwork = nn.ModuleList()
        self.sdf_net = nn.ModuleList()
        for i in range(4):
            self.hypernetwork.append(nn.Sequential(
                nn.Linear(input_layer_sz if i == 0 else hidden, hidden),
                nn.ReLU(),
            ))
            self.sdf_net.append(Siren(input_layer_sz if i == 0 else hidden, hidden, w0=3. if i == 0 else 1., is_first = i == 0))
            '''self.sdf_net.append(nn.Sequential(
                nn.Linear(input_layer_sz if i == 0 else hidden, hidden),
                nn.SiLU(),
            ))'''
        self.final_sdf = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        _xavier_init(self)

    def forward(self, x, y=None):
        if y is None:
            enc = self.positional_encoding(x, y)
        else:
            enc = self.positional_encoding(x, y)[0]
        enc = enc.reshape((-1, enc.shape[-1]))
        henc = enc
        for hyp, sir in zip(self.hypernetwork, self.sdf_net):
            henc = hyp(henc)
            enc = sir(enc) * henc
        # predict density
        return henc, self.final_sdf(enc)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            # x.requires_grad_(True)
            y = self.forward(x)[1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
            retain_graph=True)[0]
        return gradients

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight0 = torch.zeros(dim_out, dim_in)
        # weight1 = torch.zeros(dim_out, dim_in)
        # weightquad = torch.zeros(dim_out, dim_in)
        bias0 = torch.zeros(dim_out) if use_bias else None
        # bias1 = torch.zeros(dim_out) if use_bias else None
        # biasquad = torch.zeros(dim_out) if use_bias else None

        w_std = (1 / dim_in) if self.is_first else (math.sqrt(c / dim_in) / w0)
        weight0.uniform_(-w_std, w_std)
        # weight1.uniform_(-w_std, w_std)
        # weightquad.uniform_(-w_std, w_std)

        if bias0 is not None:
            bias0.uniform_(-w_std, w_std)
            # bias1.uniform_(-w_std, w_std)
            # biasquad.uniform_(-w_std, w_std)

        self.weight0 = nn.Parameter(weight0)
        # self.weight1 = nn.Parameter(weight1)
        # self.weightquad = nn.Parameter(weightquad)
        self.bias0 = nn.Parameter(bias0) if use_bias else None
        # self.bias1 = nn.Parameter(bias1) if use_bias else None
        # self.biasquad = nn.Parameter(biasquad) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = nn.functional.linear(x, self.weight0, self.bias0)
        # out =  nn.functional.linear(x, self.weight0, self.bias0) * nn.functional.linear(x, self.weight1, self.bias1) + nn.functional.linear(torch.square(x), self.weightquad, self.biasquad)
        out = self.activation(out)
        out = self.dropout(out)
        return out