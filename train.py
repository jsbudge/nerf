import os.path
import imageio
from pytorch_lightning import Trainer, loggers, seed_everything
from config import get_config
from model import MipNeRF
from os import path
from dataloader import NeRFModule
from tqdm import tqdm
import numpy as np
import torch
from utils import visualize_depth, visualize_normals, to8b

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    torch.autograd.set_detect_anomaly(True)
    # torch.cuda.empty_cache()

    seed_everything(np.random.randint(1, 2048), workers=True)
    config = get_config(param_file='./params.yaml')
    model_save_path = path.join(config.log_dir, "model.pt")
    optimizer_save_path = path.join(config.log_dir, "optim.pt")

    data = NeRFModule(config)
    data.setup()

    model = MipNeRF(
        config=config,
    )
    model.train()

    os.makedirs(config.log_dir, exist_ok=True)
    logger = loggers.TensorBoardLogger(config.log_dir, name="NeRF", log_graph=True)

    print('Building trainer...')
    trainer = Trainer(logger=logger, max_epochs=config.max_steps, devices=[0])

    print("======= Training =======")
    try:
        trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        print('Breaking out of training early.')

    torch.save(model.state_dict(), model_save_path)

    render_data = data.render_loader()
    model.eval()

    print("Generating Video using", len(render_data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []
    for ray in tqdm(render_data):
        img, dist, acc = model.render_image(ray, render_data.h, render_data.w, chunks=config.chunks)
        rgb_frames.append(img)
        if config.visualize_depth:
            depth_frames.append(to8b(visualize_depth(dist, acc, render_data.near, render_data.far)))
        if config.visualize_normals:
            normal_frames.append(to8b(visualize_normals(dist, acc)))

    imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10, codecs="hvec")
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10, codecs="hvec")
