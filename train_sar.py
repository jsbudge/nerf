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
import os

from tqdm import tqdm

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from config import get_config
from dataloader import SARNeRFModule
from model import SARNeRF
import matplotlib.pyplot as plt
import open3d as o3d
c0 = 3e8

pio.renderers.default = 'browser'


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))



if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    force_cudnn_initialization()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.empty_cache()



    config = get_config(model='sarnerf', param_file='./params.yaml')

    if config.distributed:
        seed_everything(17, workers=True)
    else:
        seed_everything(np.random.randint(1, 2048), workers=True)

    print('Loading data...')
    data = SARNeRFModule(config=config)
    data.setup()
    logger = loggers.TensorBoardLogger(config.log_dir, name="SARNeRF", log_graph=True)

    print('Building trainer...')
    if config.distributed:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, devices=2, detect_anomaly=True, overfit_batches=2,
                          strategy=FSDPStrategy(sharding_strategy='SHARD_GRAD_OP'))
    else:
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, devices=[0], detect_anomaly=True, num_sanity_val_steps=0)

    print('Loading model...')
    model = SARNeRF(
        config=config,
    )
    model.train()

    print("======= Training =======")
    try:
        trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    # torch.save(model.state_dict(), config.model_weight_path)

    rays_o, rays_d, rays_p, mfilt, radii, near, far, mpp, target = next(iter(data.train_dataloader()))


    print('Rendering pulse...')
    model.eval()
    pulse, disp, acc = model(rays_o.to(model.device), rays_d.to(model.device), rays_p.to(model.device),
                             mfilt.to(model.device), radii.to(model.device), near.to(model.device),
                             far.to(model.device), target.shape[1], mpp.to(model.device))
    model.to('cpu')
    density = model(rays_o.to(model.device), rays_d.to(model.device), rays_p.to(model.device),
                             mfilt.to(model.device), radii.to(model.device), near.to(model.device),
                             far.to(model.device), target.shape[1], mpp.to(model.device), return_occ=True)
    density = density.cpu().data.numpy()
    np_pulse = pulse.cpu().data.numpy()
    np_target = target.cpu().data.numpy()
    model.to(device)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np_pulse[-1, 0, :, 0])
    plt.subplot(2, 1, 2)
    plt.plot(np_target[0, :, 0])
    plt.show()

    sdr_f = load(config.sdr_file)
    rp = SDRPlatform(sdr_f, origin=config.data_center, channel=0)
    flight_path = rp.txpos(rp.gpst)
    bores = rp.boresight(rp.gpst)
    fig = px.scatter_3d(x=flight_path[:, 0], y=flight_path[:, 1], z=flight_path[:, 2])
    fig.add_trace(go.Cone(x=flight_path[:, 0], y=flight_path[:, 1], z=flight_path[:, 2], u=bores[:, 0],
                          v=bores[:, 1], w=bores[:, 2], anchor='tail', sizeref=55, sizemode='absolute'))
    fig.add_trace(go.Cone(x=rays_o[0, :, 0], y=rays_o[0, :, 1], z=rays_o[0, :, 2], u=rays_d[0, :, 0],
                          v=rays_d[0, :, 1], w=rays_d[0, :, 2], anchor='tail', sizeref=55, sizemode='absolute'))
    fig.update_layout(
        scene=dict(xaxis=dict(range=[flight_path[:, 0].min(), flight_path[:, 0].max()]),
                   yaxis=dict(range=[flight_path[:, 1].min(), flight_path[:, 1].max()]),
                   zaxis=dict(range=[flight_path[:, 2].min() - 100, flight_path[:, 2].max() + 100])),
    )
    fig.show()

    gx, gy, gz = np.meshgrid(np.linspace(-500, 500, 10),np.linspace(-500, 500, 10), np.linspace(-500, 500, 10))

    fig = px.scatter_3d(x=gx.flatten(), y=gy.flatten(), z=gz.flatten())
    fig.add_trace(go.Cone(x=rays_o[0, :, 0], y=rays_o[0, :, 1], z=rays_o[0, :, 2], u=rays_d[0, :, 0],
                          v=rays_d[0, :, 1], w=rays_d[0, :, 2], anchor='tail', sizeref=55, sizemode='absolute'))
    fig.show()

    '''model.to(device)

    xmin, xmax = config.x_range
    ymin, ymax = config.y_range
    zmin, zmax = config.z_range
    x = np.linspace(xmin, xmax, config.grid_size)
    y = np.linspace(ymin, ymax, config.grid_size)
    z = np.linspace(zmin, zmax, config.grid_size)
    origins = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(1, -1, 3))
    directions = torch.zeros_like(origins)
    ray_p = torch.ones_like(origins[..., :1])
    radii = torch.ones_like(origins[..., :1]) * 0.0005

    print("Predicting occupancy")
    raws = []
    with torch.no_grad():
        for i in tqdm(range(0, origins.shape[1], config.chunks)):
            img, dist, acc, raw = model(origins[:, i:i + config.chunks].to(model.device),
                                        directions[:, i:i + config.chunks].to(model.device), ray_p[:, i:i + config.chunks].to(model.device),
                             mfilt.to(model.device), radii[:, i:i + config.chunks].to(model.device), near.to(model.device),
                             far.to(model.device), target.shape[1], mpp.to(model.device))
            raws.append(torch.mean(raw, dim=1).cpu())
    sigma = torch.cat(raws, dim=0)
    sigma = np.maximum(sigma[:, -1].cpu().numpy(), 0)
    sigma = sigma.reshape(config.grid_size, config.grid_size, config.grid_size)
    print("Extracting mesh")
    print("fraction occupied", np.mean(np.array(sigma > config.sigma_threshold), dtype=np.float32))
    vertices, triangles = mcubes.marching_cubes(sigma, config.sigma_threshold)
    vertices_ = (vertices / config.sigma_threshold).astype(np.float32)'''
    '''x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face["vertex_indices"] = triangles
    mesh_path = path.join(config.log_dir, "mesh.ply")
    PlyData([PlyElement.describe(vertices_[:, 0], "vertex"), PlyElement.describe(face, "face")]).write(mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print(f"Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.")'''

    # mesh = o3d.io.read_triangle_mesh('./render_output.obj')
    # o3d.visualization.draw_plotly([mesh])




