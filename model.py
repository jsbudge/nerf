import math
from typing import Any
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib as mplib
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from simulib.platform_helper import SDRPlatform
from torch import optim, Tensor, autocast
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
        self.density_input = config.encoder_size * 3 * 2
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

        self.use_eik_base = False
        if eik_loss_baseline is not None:
            self.use_eik_base = True
            self.eik_base = torch.tensor(eik_loss_baseline, dtype=torch.float32)
        else:
            self.use_eik_base = False
        # self.use_eik_base = False

        self.sdf_network = SDFNetwork(config.encoder_sigma, config.encoder_size, config.hidden, self.density_input)

        self.param0 = nn.Sequential(
            nn.Linear(config.hidden, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, config.param_hidden),
            nn.SiLU(),
        )

        self.param1 = nn.Sequential(
            nn.Linear(config.param_hidden + self.density_input, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, config.param_hidden),
            nn.SiLU(),
            nn.Linear(config.param_hidden, 2),
            nn.Softplus(),
        )

        self.beta = nn.Parameter(data=torch.Tensor([1.]), requires_grad=True)
        self.beta_pos = nn.Softplus()

        _xavier_init(self)

        # self.example_input_array = torch.tensor(np.random.rand(1, 5), dtype=torch.float32, requires_grad=True)

    def forward(self, ray_d, ray_o, ray_p, return_occ=False):
        # Calculate sphere intersections for near and far
        bb_enter, bb_leave, hits = self.bb_intersect(ray_o.reshape(-1, 3), ray_d.reshape(-1, 3))
        pulse_bins = self.pulse_bins.to(self.device)
        near = bb_enter.view(ray_p.shape)
        far = bb_leave.view(ray_p.shape)
        # near = torch.clamp_(bb_enter.view(ray_p.shape), self.near_range, self.far_range - 1)
        # far = torch.clamp_(bb_leave.view(ray_p.shape), self.near_range, self.far_range)

        hit_mask = hits.squeeze(-1)
        ray_d = ray_d[:, hit_mask]
        ray_o = ray_o[:, hit_mask]
        ray_p = ray_p[:, hit_mask]
        near = near[:, hit_mask]
        far = far[:, hit_mask]

        z_vals = uniform_sample(ray_d, self.num_samples, near, far, randomized=True)

        # z_vals, _ = error_bound_sample(ray_d, ray_o, self.get_beta(), self.num_samples, near, far, self.sdf_network)
        pts = ray_o[..., None, :] + ray_d[..., None, :] * z_vals[..., None]
        pts = pts.reshape(-1, 3)
        pts.requires_grad_(True)
        latent_features, sdf = self.sdf_network(pts, get_features=True)
        density = laplace_cdf(sdf.reshape(z_vals.shape), self.get_beta())

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.zeros(*dists.shape[:-1], 1).to(dists.device)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(dists.device), free_energy[..., :-1]],
                                        dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here

        # predict params and reshape for use later
        enc_mean = positional_encoding(pts, self.sigma, self.encoder_size)
        params = self.param0(latent_features)
        params = torch.cat([params, enc_mean], dim=-1)
        params = self.param1(params).reshape((ray_o.shape[0], -1, self.num_samples, 2)) + .001
        acc = torch.sum(weights, dim=-1)
        distance = torch.sum(weights * z_vals, dim=-1) / acc
        distance = torch.clamp(torch.nan_to_num(distance), z_vals[..., 0], z_vals[..., -1])
        params = torch.sum(weights[..., None] * params, dim=-2) / acc[..., None]

        # Detach gradients from autograd so they don't influence backpropagation
        pts.requires_grad_(True)
        gradients = diff_ops.gradient(sdf, pts).detach().reshape(*z_vals.shape, 3)
        with torch.no_grad():
            normals = torch.sum(weights[..., None] * gradients, dim=-2) / acc[..., None]
            normals = torch.nan_to_num(-normals / normals.norm(2, -1, keepdim=True), 0)
            bounce = normals * torch.sum(ray_d * normals, dim=-1)[..., None] * 2 - ray_d

        # Phong reflection formulation, phong to be added later
        '''ref_model = (torch.sum(ray_d * normals, dim=-1) + torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * ray_p.squeeze(-1)'''
        ref_model = (params[..., 0] * torch.sum(ray_d * normals, dim=-1) + params[..., 1] * torch.nan_to_num(
            torch.abs(torch.sum(bounce * normals, dim=-1)))) / distance ** 2 * ray_p.squeeze(-1) / 16.
        # ref_model = ray_p.squeeze(-1) / torch.square(distance)

        # Get soft buckets to preserve gradients across histogramming step
        bound_dist = torch.abs(distance.unsqueeze(-1) - pulse_bins.unsqueeze(0))
        soft_buckets = torch.softmax(-bound_dist / self.temperature, dim=-1)

        # Calculate out expected phase as well
        ret = torch.view_as_real(ref_model * torch.exp(-4j * torch.pi / self.wavelength * distance))

        # Soft histogram step, along with applying filtered chirp in frequency domain
        comp_pulse = ret[:, :, None, :] * soft_buckets[:, :, :, None]
        comp_pulse = torch.sum(comp_pulse, dim=-3)
        comp_pulse = torch.view_as_complex(comp_pulse)
        comp_pulse = torch.fft.ifft(torch.fft.fft(comp_pulse, self.mfilt.shape[-1], dim=-1) *
                                    self.mfilt.to(self.device), dim=-1)[..., :self.nsam]
        comp_pulse = torch.view_as_real(comp_pulse) / self.pulse_std
        if return_occ:
            return sdf.reshape((ray_o.shape[0], -1, self.num_samples, 1)), z_vals, density.reshape((ray_o.shape[0], -1, self.num_samples, 1)), weights
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return comp_pulse, distance, z_vals, weights
            # return sdf.reshape((ray_o.shape[0], -1, self.fine_samples, 1))

    def get_beta(self):
        return .0001 + self.beta_pos(self.beta)


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
        pts.requires_grad_(True)
        sdf = self.sdf_network(pts)
        normals = diff_ops.gradient(sdf, pts).detach()
        with torch.no_grad():
            normals = torch.nan_to_num(normals / normals.norm(2, -1, keepdim=True), 0)
        return sdf.reshape(r_shape), normals.reshape((*r_shape, 3))


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.config.lr_init, weight_decay=self.config.weight_decay, foreach=False)
        scheduler = MipLRDecay(optimizer, lr_init=self.config.lr_init, lr_final=self.config.lr_final,
                               max_steps=self.config.max_steps, lr_delay_steps=self.config.lr_delay_steps,
                               lr_delay_mult=self.config.lr_delay_mult)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def training_step(self, batch, batch_idx):
        self.train_val_get(batch, True)
        '''if self.global_step % 100 == 0 and self.trainer.is_global_zero:
            mplib.use('Agg')
            pts = torch.cat([self.eik_base, torch.rand(size=(self.eik_base.shape[0], 3)) *
                             np.diff(self.scene_bbox, axis=0) + self.scene_bbox[0]], dim=0).to(self.device)

            sdf_test, norm_test = self.sample_density_function(pts=pts)
            sdf_test = laplace_cdf(sdf_test, self.get_beta()).cpu().data.numpy().flatten()
            pts_np = pts.cpu().data.numpy()

            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=sdf_test)
            ax.set_xlim(self.scene_bbox[0, 0], self.scene_bbox[1, 0])
            ax.set_ylim(self.scene_bbox[0, 1], self.scene_bbox[1, 1])
            ax.set_zlim(self.scene_bbox[0, 2], self.scene_bbox[1, 2])
            self.logger.experiment.add_figure('Density Plot', fig, global_step=self.global_step)
            plt.close(fig)'''
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
        # chunk_sz = 1024

        # Calculate out Eikonal loss
        if self.use_eik_base:
            eik_pts = (torch.rand(size=(self.eik_base.shape[0], 3)) * np.diff(self.scene_bbox, axis=0) +
                       self.scene_bbox[0]).to(self.device)
            eik_loss = (torch.exp(-10. * torch.abs(self.sdf_network(eik_pts))).sum() + torch.abs(
                self.sdf_network(self.eik_base.to(self.device))).sum())
            eik_pts = torch.cat([eik_pts, self.eik_base.to(self.device)], dim=0)
        else:
            eik_pts = (torch.rand(size=(ray_d.shape[1] * 2, 3)) * np.diff(self.scene_bbox, axis=0) + self.scene_bbox[
                0]).to(
                self.device)
            eik_loss = 0.
        eik_pts.requires_grad_(True)
        grad_theta = diff_ops.gradient(self.sdf_network(eik_pts), eik_pts)
        eik_loss = eikonal_loss(grad_theta) + eik_loss

        pulses, dists, z_vals, weights = self.forward(ray_d, ray_o, ray_p)

        pdf_weights = weights + .00001
        pdf_weights = pdf_weights / torch.sum(pdf_weights, dim=-1, keepdim=True)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.zeros(*dists.shape[:-1], 1).to(dists.device)], -1)
        density_entropy = -torch.sum(torch.log2(pdf_weights) * pdf_weights * dists, dim=-1)
        spans = z_vals[..., -1] - z_vals[..., 0]
        density_entropy = torch.sum(density_entropy * spans) / torch.sum(spans)

        density_std = self.sdf_network(eik_pts).std()

        dloss = (density_entropy + eik_loss * self.eikonal_weight)

        loss = torch.square(
            torch.abs(torch.view_as_complex(target_data)) - torch.abs(torch.view_as_complex(pulses))).mean() + dloss

        with torch.no_grad():
            psnr = mse_to_psnr(((target_data - pulses) ** 2).mean())

        self.loss = loss

        loss_name = 'train_loss' if do_train else 'val_loss'
        self.log_dict({loss_name: loss, 'eik_loss': eik_loss, 'psnr': psnr, 'lr': self.lr_schedulers().get_last_lr()[0],
                       'density_std': density_std, 'density_entropy': density_entropy}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True, sync_dist=True)

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
        return tnear, tfar, hits# torch.logical_and(hits, tnear < self.far_range)



class SDFNetwork(LightningModule):

    def __init__(self, encoder_sigma: float = 10., encoder_size: int = 10, hidden: int = 256, input_layer_sz: int = 6,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.enc_sig = encoder_sigma
        self.enc_sz = encoder_size
        self.hypernetwork = nn.ModuleList()
        self.sdf_net = nn.ModuleList()
        for i in range(3):
            self.hypernetwork.append(nn.Sequential(
                nn.Linear(input_layer_sz if i == 0 else hidden + input_layer_sz, hidden),
                nn.Softplus(),
            ))
            self.sdf_net.append(Siren(input_layer_sz if i == 0 else hidden, hidden, w0=30. if i == 0 else 10., is_first = i == 0))
        self.final_sdf = nn.Sequential(
            nn.Linear(hidden, 1),
        )

        _xavier_init(self)

    def forward(self, x, get_features=False):
        enc = positional_encoding(x, self.enc_sig, self.enc_sz)
        enc = enc.reshape((-1, enc.shape[-1]))
        init_enc = enc
        for i, (hyp, sir) in enumerate(zip(self.hypernetwork, self.sdf_net)):
            henc = hyp(init_enc) if i == 0 else hyp(torch.cat([henc, init_enc], dim=-1))
            enc = sir(enc) * henc
        # predict density
        return (enc, self.final_sdf(enc)) if get_features else self.final_sdf(enc)

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
        dropout = .25
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight0 = torch.zeros(dim_out, dim_in)
        weight1 = torch.zeros(dim_out, dim_in)
        weightquad = torch.zeros(dim_out, dim_in)
        bias0 = torch.zeros(dim_out) if use_bias else None
        bias1 = torch.zeros(dim_out) if use_bias else None
        biasquad = torch.zeros(dim_out) if use_bias else None

        w_std = (1 / dim_in) if self.is_first else (math.sqrt(c / dim_in) / w0)
        weight0.uniform_(-w_std, w_std)
        weight1.uniform_(-w_std, w_std)
        weightquad.uniform_(-w_std, w_std)

        if bias0 is not None:
            bias0.uniform_(-w_std, w_std)
            bias1.uniform_(-w_std, w_std)
            biasquad.uniform_(-w_std, w_std)

        self.weight0 = nn.Parameter(weight0)
        self.weight1 = nn.Parameter(weight1)
        self.weightquad = nn.Parameter(weightquad)
        self.bias0 = nn.Parameter(bias0) if use_bias else None
        self.bias1 = nn.Parameter(bias1) if use_bias else None
        self.biasquad = nn.Parameter(biasquad) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # out = nn.functional.linear(x, self.weight0, self.bias0)
        out =  nn.functional.linear(x, self.weight0, self.bias0) * nn.functional.linear(x, self.weight1, self.bias1) + nn.functional.linear(torch.square(x), self.weightquad, self.biasquad)
        out = self.activation(out)
        out = self.dropout(out)
        return out