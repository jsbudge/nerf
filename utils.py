import contextlib
import pickle
from typing import Optional, Union, Tuple, Dict
import torch
import torchist
from torch import nn, Tensor
from pytorch_lightning import LightningModule
from torch.nn import functional as nn_func
from PIL import Image
import numpy as np
import os
import imageio
import json


def init_weights(m):
    with contextlib.suppress(ValueError):
        if hasattr(m, 'weight'):
            torch.nn.init.xavier_normal_(m.weight)
        # sourcery skip: merge-nested-ifs
        if hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.fill_(.01)


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def rays_from_pose(W, H, focal, poses, near, far):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    camera_dirs = torch.tensor(np.stack([(i - W * .5 + .5) / focal, -(j - H * .5 + .5) / focal,
                                         -np.ones_like(i)], -1)).unsqueeze(0).to(poses.device)
    rays_d = torch.einsum('bijk,bkk->bijk', camera_dirs, poses[:, :3, :3])
    rays_o = torch.ones(rays_d.shape, device=poses.device) * poses[:, :3, -1][:, None, None, :]
    dx = torch.sqrt(torch.sum((rays_d[:, :-1, :, :] - rays_d[:, 1:, :, :]) ** 2, dim=-1))
    dx = torch.cat([dx, dx[:, -2:-1, :]], 1).unsqueeze(1)
    radii = (dx * 0.5773502691896258).reshape((-1, 1))
    rays_d = rays_d.reshape((-1, 3))
    rays_o = rays_o.reshape((-1, 3))
    near = torch.zeros_like(radii) + near
    far = torch.zeros_like(radii) + far
    return rays_d, rays_o, radii, near, far


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[..., :-1], dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf), cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1]) + [1], device=cdf.device),
                     cdf,
                     torch.ones(list(cdf.shape[:-1]) + [1], device=cdf.device)],
                    dim=-1)

    # Draw uniform samples.
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples


def lift_gaussian(d, t_mean, t_var, r_var, diag):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = d[..., None, :] * t_mean[..., None]

    d_mag_sq = torch.sum(d ** 2, dim=-1, keepdim=True) + 1e-10

    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1], device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `d` is normalized.

    Args:
    d: torch.float32 3-vector, the axis of the cone
    t0: float, the starting distance of the frustum.
    t1: float, the ending distance of the frustum.
    base_radius: float, the scale of the radius as a function of distance.
    diag: boolean, whether or the Gaussian will be diagonal or full-covariance.
    stable: boolean, whether or not to use the stable computation described in
      the paper (setting this to False will cause catastrophic failure).

    Returns:
    a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: torch.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)


def cast_rays(t_vals, origins, directions, radii, ray_shape, diag=True):
    """Cast rays (cone- or cylinder-shaped) and featurize sections of it.

    Args:
      t_vals: float array, the "fencepost" distances along the ray.
      origins: float array, the ray origin coordinates.
      directions: float array, the ray direction vectors.
      radii: float array, the radii (base radii for cones) of the rays.
      diag: boolean, whether or not the covariance matrices should be diagonal.

    Returns:
      a tuple of arrays of means and covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    if ray_shape == 'cone':
        gaussian_fn = conical_frustum_to_gaussian
    elif ray_shape == 'cylinder':
        gaussian_fn = cylinder_to_gaussian
    else:
        assert False
    means, covs = gaussian_fn(directions, t0, t1, radii, diag)
    means = means + origins[..., None, :]
    return means, covs


def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, ray_shape):
    """Stratified sampling along the rays.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      num_samples: int.
      near: torch.tensor, [batch_size, 1], near clip.
      far: torch.tensor, [batch_size, 1], far clip.
      randomized: bool, use randomized stratified sampling.
      lindisp: bool, sampling linearly in disparity rather than depth.

    Returns:
      t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
      means: torch.tensor, [batch_size, num_samples, 3], sampled means.
      covs: torch.tensor, [batch_size, num_samples, 3, 3], sampled covariances.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0., 1., num_samples + 1,  device=origins.device)
    if lindisp:
        t_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    else:
        t_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights, randomized, stop_grad, resample_padding, ray_shape):
    """Resampling.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      radii: torch.tensor(float32), [batch_size, 3], ray radii.
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      weights: torch.tensor(float32), weights for t_vals
      randomized: bool, use randomized samples.
      stop_grad: bool, whether or not to backprop through sampling.
      resample_padding: float, added to the weights before normalizing.

    Returns:
      t_vals: torch.tensor(float32), [batch_size, num_samples+1].
      points: torch.tensor(float32), [batch_size, num_samples, 3].
    """
    if stop_grad:
        with torch.no_grad():
            weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
            weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
            weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

            # Add in a constant (the sampling function will renormalize the PDF).
            weights = weights_blur + resample_padding

            new_t_vals = sorted_piecewise_constant_pdf(
                t_vals,
                weights,
                t_vals.shape[-1],
                randomized,
            )
    else:
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

        # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
            t_vals,
            weights,
            t_vals.shape[-1],
            randomized,
        )
    means, covs = cast_rays(new_t_vals, origins, directions, radii, ray_shape)
    return new_t_vals, (means, covs)


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].
    white_bkgd: bool.

    Returns:
    comp_rgb: torch.tensor(float32), [batch_size, 3].
    disp: torch.tensor(float32), [batch_size].
    acc: torch.tensor(float32), [batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])
    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights, alpha


def volumetric_scattering(norm, density, rcs, rays_p, t_vals, dirs, pulse_bins, near_range, wavelength):
    """Volumetric Rendering Function.

    Args:
    norm: torch.tensor(float32), normal values, [batch_size, ray_batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, ray_batch_size, num_samples].
    t_vals: torch.tensor(float32), [batch_size, ray_batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, ray_batch_size, 3].
    pulse_bins: torch.tensor(float32), [pulse_length]

    Returns:
    pulse: torch.tensor(float32), [batch_size, pulse_length, 2].
    disp: torch.tensor(float32), [batch_size, ray_batch_size].
    acc: torch.tensor(float32), [batch_size, ray_batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_norm = (weights[..., None] * norm).sum(dim=-2)
    comp_norm = comp_norm / torch.linalg.norm(comp_norm, dim=-1)[..., None]
    comp_rcs = (weights[..., None] * rcs).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = (torch.clamp(torch.nan_to_num(distance), t_vals[..., 0], t_vals[..., -1]) *
                (pulse_bins[1] - pulse_bins[0]) + near_range)
    # Apply Phong reflection model to get reflected power
    bounce = comp_norm * torch.sum(dirs * comp_norm, dim=-1)[..., None] * 2 - dirs
    nrho = rays_p * (torch.sum(dirs * comp_norm, dim=-1) * comp_rcs[..., 0] +
                     torch.nan_to_num(torch.abs(torch.sum(bounce * comp_norm, dim=-1) *
                                                comp_rcs[..., 1])**comp_rcs[..., 2])) / distance**2
    # Calculate out expected phase as well
    ret = torch.view_as_real(nrho * torch.exp(-2j * torch.pi / wavelength * distance * 2))
    pulse = torch.zeros((nrho.shape[0], pulse_bins.shape[0], 2), dtype=torch.float, device=nrho.device)
    # Place into correct range bins
    for d in range(nrho.shape[0]):
        pulse[d, torch.bucketize(distance[d], pulse_bins.to(distance.device))] += ret[d]
    return pulse, distance, acc, weights, alpha


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_tran_roll = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_tran_pitch = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()

rot_tran_yaw = lambda th: torch.Tensor([
    [np.cos(th), -np.sin(th), 0, 0],
    [np.sin(th), np.cos(th), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()

rot_roll = lambda phi: torch.Tensor([
    [1, 0, 0],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi), np.cos(phi)]]).float()

rot_pitch = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th)],
    [0, 1, 0],
    [np.sin(th), 0, np.cos(th)]]).float()

rot_yaw = lambda th: torch.Tensor([
    [np.cos(th), -np.sin(th), 0],
    [np.sin(th), np.cos(th), 0],
    [0, 0, 1]]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_tran_roll(phi / 180. * np.pi) @ c2w
    c2w = rot_tran_pitch(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, testskip=1):
    all_imgs = []
    all_poses = []
    with open(os.path.join(basedir, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)
    if testskip == 0:
        skip = 1
    else:
        skip = testskip

    for frame in meta['frames'][::skip]:
        fname = os.path.join(basedir, f"{frame['file_path']}.png")
        all_imgs.append(imageio.v2.imread(fname))
        all_poses.append(np.array(frame['transform_matrix']))

    imgs = np.array(all_imgs).astype(np.float32) / 255
    poses = np.array(all_poses).astype(np.float32)

    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)

    return imgs, poses# , render_poses


def get_blender_params(basedir):
    with open(os.path.join(basedir, 'transforms_train.json'), 'r') as fp:
        meta = json.load(fp)

    frame = meta['frames'][0]
    image = imageio.v2.imread(os.path.join(basedir, f"{frame['file_path']}.png"))
    H, W = image.shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    return H, W, focal


def load_llff_data(basedir, factor):
    """Load data from disk"""
    img_dir = 'images'
    if factor != 1:
        img_dir = 'images_' + str(factor)
    img_dir = os.path.join(basedir, img_dir)
    img_files = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    images = []
    for img_file in img_files:
        image = imageio.v2.imread(img_file)
        images.append(image)
    images = np.stack(images, -1)

    # Load poses
    with open(os.path.join(basedir, 'poses_bounds.npy'), 'rb') as fp:
        poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale
    # Recenter poses.
    poses = recenter_poses(poses)
    return images, poses


def get_llff_params(basedir, factor):
    """Load data from disk"""
    img_dir = 'images'
    if factor != 1:
        img_dir = 'images_' + str(factor)
    img_dir = os.path.join(basedir, img_dir)
    img_files = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
    ]
    images = []
    for img_file in img_files:
        image = imageio.v2.imread(img_file)
        images.append(image)
    images = np.stack(images, -1)

    # Load poses
    with open(os.path.join(basedir, 'poses_bounds.npy'), 'rb') as fp:
        poses_arr = np.load(fp)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])
    # Update poses according to downsampling.
    poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor
    # Correct rotation matrix ordering and move variable dim to axis 0.
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    images = np.moveaxis(images, -1, 0)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    # Rescale according to a default bd factor.
    scale = 1. / (bds.min() * .75)
    poses[:, :3, 3] *= scale
    bds *= scale
    # Recenter poses.
    poses = recenter_poses(poses)
    return images[0].shape[0], images[0].shape[1], poses[0, -1, -1]


def recenter_poses(poses):
    """Recenter poses according to the original NeRF code."""
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def poses_avg(poses):
    """Average poses according to the original NeRF code."""
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([look_at(vec2, up, center), hwf], 1)
    return c2w


def look_at(z, up, pos):
    """Construct look at view matrix
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m



def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)