import pickle
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from sdrparse import load
from simulib.platform_helper import SDRPlatform
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from dataloader import SDRDataModule
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

    seed_everything(np.random.randint(1, 2048), workers=True)
    # seed_everything(17, workers=True)

    with open('./params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    exp_params = config['sar_exp_params']
    print('Loading data...')
    data = SDRDataModule(device=device, pulse_std=exp_params['model_params']['pulse_std'], **exp_params["dataset_params"])
    data.setup()
    logger = loggers.TensorBoardLogger(exp_params['train_params']['log_dir'],
                                       name="SARNeRF", log_graph=True)

    print('Loading model...')
    mdl = SARNeRF(**exp_params['model_params'], learn_params=exp_params['learn_params'])

    expected_lr = max((exp_params['learn_params']['LR'] *
                       exp_params['learn_params']['scheduler_gamma'] ** (
                               exp_params['train_params']['max_epochs'] * exp_params['learn_params']['swa_start'])),
                      1e-9)
    print('Building trainer...')
    trainer = Trainer(logger=logger, max_epochs=exp_params['train_params']['max_epochs'],
                      log_every_n_steps=exp_params['train_params']['log_epoch'], devices=[1], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=exp_params['train_params']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=exp_params['learn_params']['swa_start'])])

    print("======= Training =======")
    try:
        trainer.fit(mdl, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    rays_o, rays_d, rays_p, mfilt, radii, near, far, mpp, target = next(iter(data.train_dataloader()))


    print('Rendering pulse...')
    with torch.no_grad():
        pulse, disp, acc = mdl(rays_o.to(mdl.device), rays_d.to(mdl.device), rays_p.to(mdl.device),
                               mfilt.to(mdl.device), radii.to(mdl.device), near.to(mdl.device), far.to(mdl.device),
                               target.shape[1], mpp.to(mdl.device))

        disp_np = disp.cpu().data.numpy()

        verts, tris = mdl.render_model((-100., 100.), (-100., 100.), (-5., 10.), 100, 100, 100, sigma_threshold=50.,
                                       chunks=1000)
    np_pulse = pulse.cpu().data.numpy()
    np_target = target.cpu().data.numpy()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np_pulse[-1, 0, :, 0])
    plt.subplot(2, 1, 2)
    plt.plot(np_target[0, :, 0])
    plt.show()

    sdr_f = load(exp_params['dataset_params']['data_path'])
    rp = SDRPlatform(sdr_f, channel=0)
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

    mesh = o3d.io.read_triangle_mesh('./render_output.obj')
    o3d.visualization.draw_plotly([mesh])




