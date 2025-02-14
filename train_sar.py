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
from tqdm import tqdm

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
        rp.getRadarParams(0., 0., 1))
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
    data = SARNeRFModule(config=config, bounding_box=bounding_box,
                         use_data_file='/home/jeff/repo/nerf/data/SAR_12172024_113146_train.pt')
    data.setup()
    logger = loggers.TensorBoardLogger(config.log_dir, name="SARNeRF", version=0, log_graph=True)

    print('Building trainer...')
    if config.distributed:
        # strategy=FSDPStrategy(sharding_strategy='FULL_SHARD'),
        trainer = Trainer(logger=logger, max_epochs=config.max_epochs, devices=2, detect_anomaly=False,
                          strategy=FSDPStrategy(sharding_strategy='FULL_SHARD'),
                           num_sanity_val_steps=0,
                          check_val_every_n_epoch=100)
    else:
        trainer = Trainer(logger=logger, max_steps=config.max_steps, max_epochs=config.max_epochs, detect_anomaly=False,
                          devices=[1], num_sanity_val_steps=0, check_val_every_n_epoch=20000)

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

    if trainer.is_global_zero:
        print('Rendering pulse...')
        mplib.use('TkAgg')
        model.eval()
        model.to(device)

        # Build out the pulse
        ray_d, ray_o, ray_p, target = next(iter(data.train_dataloader()))
        # model.to('cpu')
        pulse, dist_tensor, pts_tensor = model(ray_d.to(model.device), ray_o.to(model.device), ray_p.to(model.device))
        np_pulse = torch.view_as_complex(pulse).cpu().data.numpy()[0]
        np_target = torch.view_as_complex(target).cpu().data.numpy()[0]
        distances = dist_tensor[0].cpu().data.numpy()

        nsam, nr, ranges, ranges_sampled, near_range_s, granges, fft_len, up_fft_len = (
                rp.getRadarParams(0, 0., 1))

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

        ray_trace = (ray_o[0, 0].detach() + ray_d[0, 0].detach() * ranges[::nsam - 1][:, None]).cpu().data.numpy()
        pts_np = pts_tensor.cpu().data.numpy()
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=ray_trace[:, 0], y=ray_trace[:, 1], z=ray_trace[:, 2], mode='lines'))
        fig.add_trace(go.Scatter3d(x=gpts[:, 0], y=gpts[:, 1], z=gpts[:, 2], mode='markers', marker=dict(opacity=.5)))
        fig.add_trace(go.Scatter3d(x=pts_np[:, 0], y=pts_np[:, 1], z=pts_np[:, 2], mode='markers'))
        fig.show()




