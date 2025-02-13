import numpy as np
import torch
import collections
import matplotlib.cm as cm
from graphviz import Digraph
from matplotlib import pyplot as plt
from scipy import signal
from torch import Tensor
from torch.autograd import Variable

Rays = collections.namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*map(fn, tup))


def sorted_piecewise_constant_pdf(bins, weights, num_samples, randomized):
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdim=True)
    padding = torch.maximum(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + (padding / weights.shape[-1])
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
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(
            to=(s - torch.finfo(torch.float32).eps))
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


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / (directions[..., 2] + 1e-15)
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / (oz + 1e-15))
    o1 = -((2 * focal) / h) * (oy / (oz+ 1e-15) )
    o2 = 1 + 2 * near / (oz+ 1e-15)

    d0 = -((2 * focal) / w) * (dx / (dz+ 1e-15) - ox / (oz+ 1e-15))
    d1 = -((2 * focal) / h) * (dy / (dz+ 1e-15) - oy / (oz+ 1e-15))
    d2 = -2 * near / (oz+ 1e-15)

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


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


'''def sample_along_rays(origins, directions, radii, num_samples, near, far, randomized, lindisp, ray_shape):
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
        t_rand = torch.rand(batch_size, origins.shape[1], num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, origins.shape[1], num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)'''

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
        t_rand = torch.rand(batch_size, origins.shape[1], num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, origins.shape[1], num_samples + 1])
    means, covs = cast_rays(t_vals, origins, directions, radii, ray_shape)
    return t_vals, (means, covs)


def resample_along_rays(origins, directions, radii, t_vals, weights, randomized, stop_grad, num_samples, resample_padding, ray_shape):
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
                num_samples,
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
            num_samples,
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


def distance_calculation(density, t_vals, dirs):
    """Volumetric Scattering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].

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
    # trans = torch.where(trans < .5, trans, 0.)
    weights = alpha * trans

    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[..., 0], t_vals[..., -1])
    return distance, acc, weights, alpha, trans


def volumetric_scattering(params, weights, normals, distance, dirs, pulse_bins, wavelength):
    """Volumetric Scattering Function.

    Args:
    rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
    density: torch.tensor(float32), density, [batch_size, num_samples, 1].
    t_vals: torch.tensor(float32), [batch_size, num_samples].
    dirs: torch.tensor(float32), [batch_size, 3].

    Returns:
    comp_rgb: torch.tensor(float32), [batch_size, 3].
    disp: torch.tensor(float32), [batch_size].
    acc: torch.tensor(float32), [batch_size].
    weights: torch.tensor(float32), [batch_size, num_samples]
    """
    # comp_rgb = (weights[..., None] * params).sum(dim=-2)
    comp_rgb = torch.sum(weights[..., None] * params, dim=-2)
    bounce = normals * torch.sum(-dirs * normals, dim=-1)[..., None] * 2 + dirs

    '''nplot = normals.cpu().data.numpy()[0]
    bplot = bounce.cpu().data.numpy()[0]
    dplot = dirs.cpu().data.numpy()[0]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.quiver(np.zeros(1444), np.zeros(1444), np.zeros(1444), nplot[:, 0], nplot[:, 1], nplot[:, 2], length=1., normalize=True, color='blue')
    ax.quiver(np.zeros(1444), np.zeros(1444), np.zeros(1444), bplot[:, 0], bplot[:, 1], bplot[:, 2], length=1.,
              normalize=True, color='green')
    ax.quiver(-dplot[:, 0], -dplot[:, 1], -dplot[:, 2], dplot[:, 0], dplot[:, 1], dplot[:, 2], length=1.,
              normalize=True, color='red')
    plt.show()'''

    ref_model = (comp_rgb[..., 0] * torch.sum(-dirs * normals, dim=-1) + comp_rgb[..., 1] * torch.nan_to_num(torch.abs(torch.sum(bounce * normals, dim=-1) *
                                                comp_rgb[..., 1])**comp_rgb[..., 2])) / distance**2 * .25
    indices = torch.stack([torch.bucketize(distance[d], pulse_bins[d]) for d in range(ref_model.shape[0])])
    indices = torch.dstack([indices, indices])
    # Calculate out expected phase as well
    ret = torch.view_as_real(ref_model * torch.exp(-2j * torch.pi / wavelength * distance * 2))
    calc_pulse_data = torch.zeros((ref_model.shape[0], pulse_bins.shape[1] + 1, 2), dtype=torch.float, device=ref_model.device)
    calc_pulse_data = calc_pulse_data.scatter_reduce(1, indices, ret, reduce='sum')
    return calc_pulse_data





def generate_spiral_cam_to_world(radii, focus_depth, n_poses=120):
    """
    Generate a spiral path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ppn7ddat
    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path
    Outputs:
        poses_spiral: (n_poses, 3, 4) the cam to world transformation matrix of a spiral path
    """

    spiral_cams = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([(np.cos(t) * 0.5) - 2, -np.sin(t) - 0.5, -np.sin(0.5 * t) * 0.75]) * radii
        # the viewing z axis is the vector pointing from the focus_depth plane to center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        # compute other axes as in average_poses
        x = normalize(np.cross(np.array([0, 1, 0]), z))
        y = np.cross(z, x)
        spiral_cams += [np.stack([y, z, x, center], 1)]
    return np.stack(spiral_cams, 0)


def generate_spherical_cam_to_world(radius, n_poses=120):
    """
    Generate a 360 degree spherical path for rendering
    ref: https://github.com/kwea123/nerf_pl/blob/master/datasets/llff.py
    ref: https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_cams: (n_poses, 3, 4) the cam to world transformation matrix of a circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ], dtype=np.float)

        rotation_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ], dtype=np.float)

        rotation_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ], dtype=np.float)
        cam_to_world = trans_t(radius)
        cam_to_world = rotation_phi(phi / 180. * np.pi) @ cam_to_world
        cam_to_world = rotation_theta(theta) @ cam_to_world
        cam_to_world = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                                dtype=np.float) @ cam_to_world
        return cam_to_world

    spheric_cams = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_cams += [spheric_pose(th, -30, radius)]
    return np.stack(spheric_cams, 0)


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


def flatten(x):
    # Always flatten out the height x width dimensions
    x = [y.reshape([-1, y.shape[-1]]) for y in x]
    # concatenate all data into one list
    x = np.concatenate(x, axis=0)
    return x


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def convolve2d(z, f):
  return signal.convolve2d(z, f, mode='same')


def depth_to_normals(depth):
    """Assuming `depth` is orthographic, linearize it to a set of normals."""
    f_blur = np.array([1, 2, 1]) / 4
    f_edge = np.array([-1, 0, 1]) / 2
    dy = convolve2d(depth, f_blur[None, :] * f_edge[:, None])
    dx = convolve2d(depth, f_blur[:, None] * f_edge[None, :])
    inv_denom = 1 / np.sqrt(1 + dx**2 + dy**2)
    normals = np.stack([dx * inv_denom, dy * inv_denom, inv_denom], -1)
    return normals


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x)**2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def visualize_normals(depth, acc, scaling=None):
    """Visualize fake normals of `depth` (optionally scaled to be isotropic)."""
    if scaling is None:
        mask = ~np.isnan(depth)
        x, y = np.meshgrid(
            np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing='xy')
        xy_var = (np.var(x[mask]) + np.var(y[mask])) / 2
        z_var = np.var(depth[mask])
        scaling = np.sqrt(xy_var / z_var)

        scaled_depth = scaling * depth
        normals = depth_to_normals(scaled_depth)
        vis = np.isnan(normals) + np.nan_to_num((normals + 1) / 2, 0)

        # Set non-accumulated pixels to white.
        if acc is not None:
            vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

        return vis


def visualize_depth(depth,
                    acc=None,
                    near=None,
                    far=None,
                    ignore_frac=0,
                    curve_fn=lambda x: -np.log(x + np.finfo(np.float32).eps),
                    modulus=0,
                    colormap=None):
    """Visualize a depth map.
    Args:
      depth: A depth map.
      acc: An accumulation map, in [0, 1].
      near: The depth of the near plane, if None then just use the min().
      far: The depth of the far plane, if None then just use the max().
      ignore_frac: What fraction of the depth map to ignore when automatically
        generating `near` and `far`. Depends on `acc` as well as `depth'.
      curve_fn: A curve function that gets applied to `depth`, `near`, and `far`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
          Note that the default choice will flip the sign of depths, so that the
          default colormap (turbo) renders "near" as red and "far" as blue.
      modulus: If > 0, mod the normalized depth by `modulus`. Use (0, 1].
      colormap: A colormap function. If None (default), will be set to
        matplotlib's turbo if modulus==0, sinebow otherwise.
    Returns:
      An RGB visualization of `depth`.
    """
    if acc is None:
        acc = np.ones_like(depth)
    acc = np.where(np.isnan(depth), np.zeros_like(acc), acc)

    # Sort `depth` and `acc` according to `depth`, then identify the depth values
    # that span the middle of `acc`, ignoring `ignore_frac` fraction of `acc`.
    sortidx = np.argsort(depth.reshape([-1]))
    depth_sorted = depth.reshape([-1])[sortidx]
    acc_sorted = acc.reshape([-1])[sortidx]
    cum_acc_sorted = np.cumsum(acc_sorted)
    mask = ((cum_acc_sorted >= cum_acc_sorted[-1] * ignore_frac) &
            (cum_acc_sorted <= cum_acc_sorted[-1] * (1 - ignore_frac)))
    depth_keep = depth_sorted[mask]

    # If `near` or `far` are None, use the highest and lowest non-NaN values in
    # `depth_keep` as automatic near/far planes.
    eps = np.finfo(np.float32).eps
    near = near or depth_keep[0] - eps
    far = far or depth_keep[-1] + eps

    # Curve all values.
    depth, near, far = [curve_fn(x) for x in [depth, near, far]]

    # Wrap the values around if requested.
    if modulus > 0:
        value = np.mod(depth, modulus) / modulus
        colormap = colormap or sinebow
    else:
        # Scale to [0, 1].
        value = np.nan_to_num(
            np.clip((depth - np.minimum(near, far)) / np.abs(far - near), 0, 1))
        colormap = colormap or cm.get_cmap('turbo')

    vis = colormap(value)[:, :, :3]

    # Set non-accumulated pixels to white.
    vis = vis * acc[:, :, None] + (1 - acc)[:, :, None]

    return vis


def to8b(img):
    if len(img.shape) >= 3:
        return np.array([to8b(i) for i in img])
    else:
        return (255 * np.clip(np.nan_to_num(img), 0, 1)).astype(np.uint8)


def to_float(img):
    if len(img.shape) >= 3:
        return np.array([to_float(i) for i in img])
    else:
        return (img / 255.).astype(np.float32)

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


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    misses = under_sqrt <= 0
    under_sqrt[misses] = 0

    sphere_intersections = torch.sqrt(under_sqrt) * torch.tensor([-1, 1], device=under_sqrt.device).float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections, misses


def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().data.numpy())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(len(ave_grads)), layers, rotation=45)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    
    
def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var in seen:
            return
        if isinstance(var, Variable):
            value = '(' + ', '.join(['%d' % v for v in var.size()]) + ')'
            dot.node(id(var), str(value), fillcolor='lightblue')
        else:
            dot.node(id(var), str(type(var).__name__))
        seen.add(var)
        if hasattr(var, 'previous_functions'):
            for u in var.previous_functions:
                dot.edge(id(u[0]), id(var))
                add_nodes(u[0])

    add_nodes(var.creator)
    return dot

def eikonal_loss(grad_theta):
    return (torch.square(grad_theta.norm(2, dim=1) - 1)).mean()


def uniform_sample(ray_d, n_samples, near, far, randomized=False):
    t_vals = torch.linspace(0., 1., n_samples).to(ray_d.device)
    z_vals = near * (1. - t_vals) + far * t_vals

    if randomized:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(ray_d.device)

        z_vals = lower + (upper - lower) * t_rand
    return z_vals


def error_bound_sample(ray_d, ray_o, _beta, n_samples, near, far, sdf_network, eps=1e-3, max_iters=5, beta_iters=2, add_samples=32):
    beta0 = _beta.detach()
    z_vals = uniform_sample(ray_d, n_samples, near, far)
    samples, samples_idx = z_vals, None
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    bound = (1.0 / (4.0 * torch.log(torch.tensor(eps + 1.0)))) * torch.sum(torch.square(dists), dim=-1)
    beta = torch.sqrt(bound)

    total_iters, not_converge = 0, True

    while not_converge and total_iters < max_iters:
        pts = ray_o[..., None, :] + ray_d[..., None, :] * samples[..., None]
        pts = pts.reshape(-1, 3)
        with torch.no_grad():
            samples_sdf = sdf_network(pts)
            if samples_idx is not None:
                sdf = sdf.reshape(1, -1, z_vals.shape[2] - samples.shape[2])
                samples_sdf = samples_sdf.reshape(1, -1, samples.shape[2])
                sdf_merge = torch.cat([sdf, samples_sdf], -1)
                sdf = torch.gather(sdf_merge, 2, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf

        d = sdf.reshape(z_vals.shape)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        a, b, c = dists, torch.abs(d[..., :-1]), torch.abs(d[..., 1:])
        first_cond = torch.square(a) + torch.square(b) <= torch.square(c)
        second_cond = torch.square(a) + torch.square(c) <= torch.square(b)
        d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1], z_vals.shape[2] - 1).to(z_vals.device)
        d_star[first_cond] = b[first_cond]
        d_star[second_cond] = c[second_cond]
        s = (a + b + c) / 2.0
        area_before_sqrt = s * (s - a) * (s - b) * (s - c)
        mask = ~first_cond & ~second_cond & (b + c - a > 0)
        d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
        d_star = (torch.sign(d[..., 1:]) * torch.sign(d[..., :-1]) == 1) * d_star  # Fixing the sign

        # Updating beta using line search
        curr_error = get_error_bound(beta0, sdf, z_vals, dists, d_star)
        beta[curr_error <= eps] = beta0
        beta_min, beta_max = torch.ones_like(beta) * beta0, beta
        for _ in range(beta_iters):
            beta_mid = (beta_min + beta_max) / 2.
            curr_error = get_error_bound(beta_mid.unsqueeze(-1), sdf, z_vals, dists, d_star)
            # err_bound = curr_error <= eps
            beta_max[curr_error <= eps] = beta_mid[curr_error <= eps]
            beta_min[curr_error > eps] = beta_mid[curr_error > eps]
        beta = beta_max.unsqueeze(-1)

        density = laplace_cdf(sdf.reshape(z_vals.shape), beta)

        dists = torch.cat([dists, torch.zeros(*dists.shape[:-1], 1).to(dists.device)], -1)
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(dists.device), free_energy[..., :-1]], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        weights = alpha * transmittance  # probability of the ray hits something here

        #  Check if we are done and this is the last sampling
        total_iters += 1
        not_converge = beta.max() > beta0

        if not_converge and total_iters < max_iters:
            ''' Sample more points proportional to the current error bound'''

            error_per_section = torch.exp(-d_star / beta) * torch.square(dists[..., :-1]) / (
                        4 * torch.square(beta))
            error_integral = torch.cumsum(error_per_section, dim=-1)
            bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * transmittance[..., :-1]

            pdf = bound_opacity + 0.0
            u = torch.linspace(0., 1., steps=add_samples).to(pdf.device)
            u = torch.broadcast_to(u, (*pdf.shape[:-1], add_samples))
            samples = sample_from_pdf(u, pdf, z_vals)
            z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)
            beta = beta.squeeze(-1)
        else:
            ''' Sample the final sample set to be used in the volume rendering integral '''

            pdf = weights[..., :-1]
            pdf = pdf + 1e-5  # prevent nans
            u = torch.rand(*pdf.shape[:-1], n_samples).to(pdf.device)
            samples = sample_from_pdf(u, pdf, z_vals)

    z_vals, _ = torch.sort(samples, -1)

    # add some of the near surface points
    '''idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
    z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))'''

    return z_vals, pdf


def sample_from_pdf(u, pdf, z_vals):
    pdf = pdf / torch.sum(pdf, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], inds_g.shape[2], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
    bins_g = torch.gather(z_vals.unsqueeze(2).expand(matched_shape), 3, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

def get_error_bound(beta, sdf, z_vals, dists, d_star):
    density = laplace_cdf(sdf.reshape(z_vals.shape), beta=beta)
    shifted_free_energy = torch.cat([torch.zeros(*dists.shape[:-1], 1).to(dists.device), dists * density[..., :-1]], dim=-1)
    integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
    error_per_section = torch.exp(-d_star / beta) * torch.square(dists) / (4 * torch.square(beta))
    error_integral = torch.cumsum(error_per_section, dim=-1)
    bound_opacity = (torch.clamp(torch.exp(error_integral), max=1.e6) - 1.0) * torch.exp(
        -integral_estimation[..., :-1])

    return bound_opacity.max(-1)[0]


def laplace_cdf(x, beta):
    return (.5 + .5 * torch.sign(x) * torch.expm1(-torch.abs(x) / beta)) / beta


def positional_encoding(v: Tensor,
        sigma: float,
        m: int) -> Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`
        where :math:`j \in \{0, \dots, m-1\}`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`

    See :class:`~rff.layers.PositionalEncoding` for more details.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)