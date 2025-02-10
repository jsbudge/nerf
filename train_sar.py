import mcubes
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from sdrparse import load
from simulib.platform_helper import SDRPlatform
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from pytorch_lightning.strategies import FSDPStrategy
from simulib.grid_helper import SDREnvironment
from simulib.simulation_functions import azelToVec
from config import get_config
from dataloader import SARNeRFModule
from model import SARNeRF
import matplotlib.pyplot as plt
import open3d as o3d
import matplotlib as mplib

from utils import laplace_cdf

pio.renderers.default = 'browser'


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()



    config = get_config(model='sarnerf', param_file='./params.yaml')

    if config.distributed:
        seed_everything(17, workers=True)
    else:
        seed_everything(np.random.randint(1, 2048), workers=True)

    from simulib.simulation_functions import enu2llh, llh2enu, db

    sdr_f = load(config.sdr_file)
    rp = SDRPlatform(sdr_f, origin=config.data_center, fs=sdr_f[0].fs)

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
        rp.getRadarParams(5., .75, 1))
    # Calculate out ground ranges for sphere intersections
    av_hght = rp.pos(rp.gpst).mean(axis=0)[2]
    near_grange = granges[0]
    far_grange = granges[-1]
    sph_inter = (far_grange - near_grange)

    bg = SDREnvironment(sdr_f, origin=config.data_center)
    gx, gy, gz = bg.getGrid(config.data_center, sph_inter, sph_inter, 80, 80)
    # Shift position
    glat, glon, galt = enu2llh(gx.flatten(), gy.flatten(), gz.flatten(), bg.ref)
    gx, gy, gz = llh2enu(glat, glon, galt, rp.origin)



    # Only get the ones inside the scene sphere
    gpts = np.dstack((gx, gy, gz))[0]
    gpts = gpts[np.linalg.norm(gpts[:, :2], axis=1) < sph_inter]
    bounding_box = np.array([[gpts[:, 0].min() - 10, gpts[:, 1].min() - 10, gpts[:, 2].min() - 10],
                             [gpts[:, 0].max() + 10, gpts[:, 1].max() + 10, gpts[:, 2].max() + 10]])

    print('Loading data...')
    data = SARNeRFModule(config=config)
    data.setup()
    logger = loggers.TensorBoardLogger(config.log_dir, name="SARNeRF", version=0, log_graph=True)

    print('Building trainer...')
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, devices=2, detect_anomaly=False, overfit_batches=2,
                          strategy=FSDPStrategy(sharding_strategy='SHARD_GRAD_OP'), num_sanity_val_steps=0,
                          check_val_every_n_epoch=100)
    else:
        trainer = Trainer(logger=logger, max_steps=config.max_steps, max_epochs=config.max_epochs, detect_anomaly=False,
                          devices=[0], num_sanity_val_steps=0, check_val_every_n_epoch=20000)

    print('Loading model...')
    model = SARNeRF(
        config=config,
        eik_loss_baseline=gpts,
        scene_bbox=bounding_box,
    )
    model.train()

    print("======= Training =======")
    try:
        trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    # torch.save(model.state_dict(), config.model_weight_path)


    print('Rendering pulse...')
    mplib.use('TkAgg')
    model.eval()
    model.to(device)

    # Generate points from sphere for sampling the function
    pts = torch.cat([model.eik_base, torch.rand(size=(model.eik_base.shape[0], 3)) * np.diff(model.scene_bbox, axis=0) + model.scene_bbox[0]], dim=0).to(model.device)

    sdf_test, norm_test = model.sample_density_function(pts=pts)
    density_test = laplace_cdf(sdf_test, model.get_beta()).cpu().data.numpy().flatten()
    norm_test = norm_test.cpu().data.numpy().reshape(-1, 3)
    sdf_test = sdf_test.cpu().data.numpy().flatten()

    pts_np = pts.cpu().data.numpy()
    ax = plt.figure('Normals').add_subplot(projection='3d')
    ax.quiver(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], norm_test[:, 0], norm_test[:, 1], norm_test[:, 2])
    plt.show()

    ax = plt.figure('Density').add_subplot(projection='3d')
    ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=density_test)
    plt.show()

    # Build out the pulse
    ray_d, ray_o, ray_p, target = next(iter(data.train_dataloader()))
    pulse, dist_tensor, sdf_tensor, weight_tensor = model(ray_d.to(model.device), ray_o.to(model.device), ray_p.to(model.device))
    np_pulse = torch.view_as_complex(pulse).cpu().data.numpy()[0]
    np_target = torch.view_as_complex(target).cpu().data.numpy()[0]
    distances = dist_tensor[0].cpu().data.numpy()
    densities = laplace_cdf(sdf_tensor, model.get_beta())

    nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
            rp.getRadarParams(5., .75, 1))

    mfilt = sdr_f.genMatchedFilter(0, fft_len=fft_len)
    test = np.fft.ifft(np.fft.fft(sdr_f.getPulse(10)[1].flatten(), fft_len) * mfilt)[:nsam]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(ranges, abs(np_pulse))
    plt.subplot(3, 1, 2)
    plt.plot(ranges, abs(np_target))
    plt.subplot(3, 1, 3)
    plt.plot(ranges, abs(test))
    plt.show()

    # Calculate sphere intersections for near and far
    bb_enter, bb_leave, misses = model.bb_intersect(ray_o.reshape(-1, 3).to(model.device), ray_d.reshape(-1, 3).to(model.device))
    misses = misses.to(ray_d.device)
    tilts = np.arcsin(-ray_d[0, :, 2].cpu().data.numpy())

    flight_path = data.train_dataset.pos
    ray_points = (ray_o[0, misses].cpu().data.numpy() + ray_d[0, misses].cpu().data.numpy() * distances[:, None])
    beampattern = ray_o[0].detach().numpy() + ray_d[0].cpu().data.numpy() * (ray_o[0, :, 2].detach().numpy() / np.sin(tilts))[:, None]
    trace_angle = azelToVec(np.arctan2(ray_d[0, :, 0].cpu().data.numpy(), ray_d[0, :, 1].cpu().data.numpy()).mean(), -np.arcsin(ray_d[0, :, 2].cpu().data.numpy()).mean())
    ray_trace = (ray_o[0, 0].detach() + trace_angle[None, :] * ranges[::nsam-1][:, None]).cpu().data.numpy()
    ray_size = db(ray_p[0].cpu().data.numpy().flatten())
    ray_size = (ray_size - ray_size.min()) / (ray_size.max() - ray_size.min()) * 50
    fig = px.scatter_3d(x=flight_path[:, 0], y=flight_path[:, 1], z=flight_path[:, 2])
    fig.add_trace(go.Scatter3d(x=gpts[:, 0], y=gpts[:, 1], z=gpts[:, 2], mode='markers', marker=dict(opacity=.5)))
    fig.add_trace(go.Scatter3d(x=ray_points[:, 0], y=ray_points[:, 1], z=ray_points[:, 2], mode='markers'))
    fig.add_trace(go.Scatter3d(x=beampattern[:, 0], y=beampattern[:, 1], z=beampattern[:, 2], marker=dict(size=ray_size), mode='markers'))
    fig.add_trace(go.Scatter3d(x=ray_trace[:, 0], y=ray_trace[:, 1], z=ray_trace[:, 2], mode='lines'))
    fig.update_layout(
        scene=dict(xaxis=dict(range=[flight_path[:, 0].min(), flight_path[:, 0].max()]),
                   yaxis=dict(range=[flight_path[:, 1].min(), flight_path[:, 1].max()]),
                   zaxis=dict(range=[min(beampattern[:, 2].min(), gpts[:, 2].min(), ray_points[:, 2].min()) - 10, flight_path[:, 2].max() + 100])),
    )
    fig.show()

    # BBOx intersections
    from simulib.mesh_functions import drawOctreeBox
    bbox = drawOctreeBox(model.scene_bbox)


    bb_enter_pos = (ray_o + ray_d * bb_enter[..., None].to(ray_o.device)).cpu().data.numpy()
    bb_leave_pos = (ray_o + ray_d * bb_leave[..., None].to(ray_o.device)).cpu().data.numpy()
    misses = misses.cpu().data.numpy()

    fig = px.scatter_3d(x=bb_enter_pos[0, misses, 0], y=bb_enter_pos[0, misses, 1], z=bb_enter_pos[0, misses, 2])
    fig.add_trace(go.Scatter3d(x=bb_leave_pos[0, misses, 0], y=bb_leave_pos[0, misses, 1], z=bb_leave_pos[0, misses, 2],
                               mode='markers'))
    fig.add_trace(bbox)

    '''for n in range(misses.shape[0]):
        if not misses[n]:
            trace_angle = azelToVec(np.arctan2(ray_d[n, 0], ray_d[n, 1]), -np.arcsin(ray_d[n, 2]))
            ray_trace = (ray_info[:, :3].detach() + trace_angle[None, :] * ranges[::nsam - 1][:, None]).cpu().data.numpy()
            fig.add_trace(go.Scatter3d(x=ray_trace[:, 0], y=ray_trace[:, 1], z=ray_trace[:, 2], mode='lines'))'''
    fig.show()

    model.to('cpu')
    sdf_cubes, _ = model.sample_density_function([gx.min(), gx.max()], [gy.min(), gy.max()], [gz.min(), gz.max()], [50, 50, 50])
    vertices, triangles = mcubes.marching_cubes(sdf_cubes.cpu().data.numpy(), 0)

    fig = go.Figure(data=[go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2])])
    fig.show()




