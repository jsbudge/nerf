import pickle
import numpy as np
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
import yaml
from dataloader import ImageDataModule
from model import NeRF
import matplotlib.pyplot as plt


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

    exp_params = config['exp_params']



    data = ImageDataModule(device=device, **exp_params["dataset_params"])
    data.setup()
    logger = loggers.TensorBoardLogger(exp_params['train_params']['log_dir'],
                                       name="NeRF", log_graph=True)

    mdl = NeRF(**exp_params['model_params'], learn_params=exp_params['learn_params'])

    expected_lr = max((exp_params['learn_params']['LR'] *
                       exp_params['learn_params']['scheduler_gamma'] ** (
                               exp_params['train_params']['max_epochs'] * exp_params['learn_params']['swa_start'])),
                      1e-9)
    trainer = Trainer(logger=logger, max_epochs=exp_params['train_params']['max_epochs'],
                      log_every_n_steps=exp_params['train_params']['log_epoch'], devices=[0], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=exp_params['train_params']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=exp_params['learn_params']['swa_start'])])

    print("======= Training =======")
    try:
        trainer.fit(mdl, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    target, poses = data.val_dataset.getImage(0)

    print('Rendering image...')
    with torch.no_grad():
        rgb, disp, acc = mdl.render_image(poses.unsqueeze(0).to(mdl.device), target.shape[0], target.shape[1], data.focal)

    print('Rendering model...')
    with torch.no_grad():
        verts, tris = mdl.render_model((-1., 1.), (-1., 1.), (0., 1.), 100, 100, 100)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(rgb)
    plt.subplot(2, 1, 2)
    plt.imshow(target)
    plt.show()

