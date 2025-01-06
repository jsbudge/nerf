import os.path
import imageio
from pytorch_lightning import Trainer, loggers, seed_everything
from torch.utils.data import DataLoader

from config import get_config
from model import MipNeRF
from os import path
from dataloader import NeRFModule, dataset_dict
from tqdm import tqdm
import numpy as np
import torch
from utils import visualize_depth, visualize_normals, to8b


def get_dataset(dataset_name, base_dir, split, factor=4, device=torch.device("cpu")):
    d = dataset_dict[dataset_name](base_dir, split, factor=factor, device=device)
    return d


def get_dataloader(dataset_name, base_dir, split, factor=4, batch_size=None, shuffle=True, device=torch.device("cpu")):
    d = get_dataset(dataset_name, base_dir, split, factor, device)
    # make the batchsize height*width, so that one "batch" from the dataloader corresponds to one
    # image used to render a video, and don't shuffle dataset
    if split == "render":
        batch_size = d.w * d.h
        shuffle = False
    loader = DataLoader(d, batch_size=batch_size, shuffle=shuffle)
    loader.h = d.h
    loader.w = d.w
    loader.near = d.near
    loader.far = d.far
    return loader

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

    render_data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)
    model.eval()

    print("Generating Video using", len(render_data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []
    for ray in tqdm(render_data):
        img, dist, acc = model.render_image(ray, render_data.h, render_data.w, chunks=config.chunks)
        break
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
