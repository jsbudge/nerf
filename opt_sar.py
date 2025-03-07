import optuna
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from sdrparse import load
from simulib.platform_helper import SDRPlatform
from simulib.grid_helper import SDREnvironment
from functools import partial
from config import get_config
from dataloader import SARNeRFModule
from model import SARNeRF

def objective(trial: optuna.Trial, config=None, gpts=None, bounding_box=None, mfilt=None):
    config.hidden_siren = trial.suggest_float('hidden_siren', .1, 100.)
    config.encoder_sigma = trial.suggest_float('encoder_sigma', .1, 1000.)
    config.beta0 = trial.suggest_float('beta0', .1, 10.)
    config.lr_init = trial.suggest_categorical('lr', [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-9])
    config.weight_decay = trial.suggest_categorical('weight_decay', [0., 1e-1, 1e-2, 1e-3, 1e-5, 1e-7])
    data = SARNeRFModule(config=config, bounding_box=bounding_box,
                         use_data_file='/home/jeff/repo/nerf/data/SAR_12172024_113146_train.pt')
    data.setup()

    trainer = Trainer(logger=False, max_steps=config.max_steps, max_epochs=10, detect_anomaly=False,
                      devices=[1], num_sanity_val_steps=0, check_val_every_n_epoch=20000, enable_checkpointing=False)

    model = SARNeRF(
        config=config,
        eik_loss_baseline=gpts,
        scene_bbox=bounding_box,
        mfilt=mfilt,
    )
    model.train()
    trainer.fit(model, datamodule=data)

    return trainer.callback_metrics['psnr'].item()


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

    from simulib.simulation_functions import enu2llh, llh2enu

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
    mfilt = torch.tensor(sdr_f.genMatchedFilter(0, fft_len=fft_len) * np.fft.fft(sdr_f[0].cal_chirp, fft_len),
                         dtype=torch.complex64)

    study = optuna.create_study(direction='maximize',
                                storage="sqlite:///db.sqlite3",
                                study_name='sarnerf')
    objective = partial(objective, config=config, gpts=gpts, bounding_box=bounding_box, mfilt=mfilt)
    study.optimize(objective, n_trials=100)