from glob import glob
from typing import List, Optional, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
import numpy as np
from multiprocessing import cpu_count
from sdrparse import load
from simulib.platform_helper import SDRPlatform
from simulib.simulation_functions import azelToVec
from sklearn.model_selection import train_test_split
from utils import load_blender_data, get_blender_params, rays_from_pose, rot_roll, rot_yaw


def collate_fun(batch):
    return (torch.stack([ccd for ccd, _, _, _, _, _ in batch]), torch.stack([tcd for _, tcd, _, _, _, _ in batch]),
            torch.stack([csd for _, _, csd, _, _, _ in batch]), torch.stack([tsd for _, _, _, tsd, _, _ in batch]),
            torch.tensor([pl for _, _, _, _, pl, _ in batch]), torch.tensor([bw for _, _, _, _, _, bw in batch]))


class BaseModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            pin_memory: bool = False,
            single_example: bool = False,
            device: str = 'cpu',
            collate: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = 0  # cpu_count() // 2
        self.pin_memory = pin_memory
        self.single_example = single_example
        self.device = device
        self.collate = collate
        self.train_sampler = None
        self.val_sampler = None

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        if self.collate:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size if self.train_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.train_sampler,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size if self.train_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.train_sampler,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.collate:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size if self.val_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.val_sampler,
                pin_memory=self.pin_memory,
                collate_fn=collate_fun,
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size if self.val_sampler is None else 1,
                num_workers=self.num_workers,
                batch_sampler=self.val_sampler,
                pin_memory=self.pin_memory,
            )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size if self.val_sampler is None else 1,
            num_workers=self.num_workers,
            batch_sampler=self.val_sampler,
            pin_memory=self.pin_memory,
            collate_fn=collate_fun,
        )


class ImageDataset(Dataset):
    def __init__(self, root_dir: str, H: int = 256, W: int = 256, focal: float = .1,
                 split: float = 1., single_example: bool = False,
                 near: float = 0., far: float = 1., is_val=False, seed=42):
        assert Path(root_dir).is_dir()
        images, poses = load_blender_data(root_dir)
        idxes = np.arange(poses.shape[0])

        if split < 1:
            Xs, Xt = train_test_split(idxes, test_size=split, random_state=seed)
        else:
            Xt = idxes
            Xs = idxes

        i_vals = Xs if is_val else Xt
        images = images[i_vals, ...]
        poses = torch.tensor(poses[i_vals])
        pixels = torch.tensor(images.reshape((-1, 4)))
        rays_d, rays_o, radii, near, far = rays_from_pose(W, H, focal, poses, near, far)

        # Remove the transparent parts, they're not good for training
        mask = pixels[:, 3] != 0
        self.rays_d = rays_d[mask]
        self.rays_o = rays_o[mask]
        self.radii = radii[mask]
        self.near = near[mask]
        self.far = far[mask]
        self.pixels = pixels[mask, :3]

        self.images = images[..., :3]
        self.poses = poses

    def __getitem__(self, idx):
        return self.rays_d[idx], self.rays_o[idx], self.radii[idx], self.near[idx], self.far[idx], self.pixels[idx]

    def __len__(self):
        return self.rays_d.shape[0]

    def getImage(self, idx):
        return self.images[idx], self.poses[idx]


class SDRPulseDataset(Dataset):
    def __init__(self, sdr_file: str, split: float = 1., fft_sz: int = 16384, az_samples: int = 32,
                 el_samples: int = 32, pulse_std: float = 1., single_example: bool = False, is_val=False, seed=42):
        sdr_f = load(sdr_file)
        idxes = np.arange(sdr_f[0].nframes)

        if split < 1:
            Xs, Xt = train_test_split(idxes, test_size=split, random_state=seed)
        else:
            Xt = idxes
            Xs = idxes

        i_vals = Xs if is_val else Xt
        rp = SDRPlatform(sdr_f, channel=0)
        self.pos = torch.tensor(rp.txpos(sdr_f[0].pulse_time[i_vals]), dtype=torch.float)
        self.pans = rp.pan(sdr_f[0].pulse_time[i_vals])
        self.tilts = rp.tilt(sdr_f[0].pulse_time[i_vals])
        self.pulses = np.fft.ifft(np.fft.fft(sdr_f.getPulses(sdr_f[0].frame_num[i_vals])[1], fft_sz, axis=0).T *
                                  sdr_f.genMatchedFilter(0, fft_len=fft_sz), axis=1)[:, :sdr_f[0].nsam]
        self.pulses = torch.view_as_real(torch.tensor(self.pulses / pulse_std))
        self.mfilt = sdr_f.genMatchedFilter(0, fft_len=fft_sz) * np.fft.fft(sdr_f[0].cal_chirp, fft_sz)
        near, far = rp.calcRanges(0, .5)

        azes, eles = np.meshgrid(np.linspace(-rp.az_half_bw, rp.az_half_bw, az_samples), np.linspace(-rp.el_half_bw, rp.el_half_bw, el_samples))
        self.pvecs = torch.tensor(azelToVec(azes.flatten(), eles.flatten()).T, dtype=torch.float)
        dx = torch.sqrt(torch.sum((self.pvecs[:-1] - self.pvecs[1:]) ** 2, dim=-1))
        dx = torch.cat([dx, dx[-2:-1]], 0)
        self.radii = (dx * 0.5773502691896258)
        self.near = torch.zeros_like(self.radii) + near
        self.far = torch.zeros_like(self.radii) + far
        self.ray_p = np.sinc(azes.flatten() / rp.az_half_bw) * np.sinc(eles.flatten() / rp.el_half_bw)


    def __getitem__(self, idx):
        ray_d = self.pvecs @ rot_yaw(self.pans[idx]) @ rot_roll(self.tilts[idx])
        return torch.outer(torch.ones_like(self.radii), self.pos[idx]), ray_d, self.ray_p, self.mfilt, self.radii, self.near, self.far, self.pulses[idx]

    def __len__(self):
        return self.pulses.shape[0]


class SDRDataModule(BaseModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            fft_sz: int = 16384,
            az_samples: int = 32,
            el_samples: int = 32,
            pulse_std: float = 1.,
            pin_memory: bool = False,
            split: float = 1.,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.data_dir = data_path
        self.fft_sz = fft_sz
        self.split = split
        self.device = device
        self.az_samples = az_samples
        self.el_samples = el_samples
        self.pulse_std = pulse_std

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SDRPulseDataset(self.data_dir, az_samples = self.az_samples,
                 el_samples=self.el_samples, split=self.split, pulse_std=self.pulse_std, fft_sz=self.fft_sz)
        self.val_dataset = SDRPulseDataset(self.data_dir, az_samples = self.az_samples,
                 el_samples=self.el_samples, split=self.split, pulse_std=self.pulse_std, fft_sz=self.fft_sz, is_val=True)


class ImageDataModule(BaseModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            near: float = 0.,
            far: float = 1.,
            pin_memory: bool = False,
            split: float = 1.,
            single_example: bool = False,
            device: str = 'cpu',
            **kwargs,
    ):
        super().__init__(train_batch_size, val_batch_size, pin_memory, single_example, device)

        self.data_dir = data_path
        self.H, self.W, self.focal = get_blender_params(data_path)
        self.near = near
        self.far = far
        self.split = split
        self.device = device

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ImageDataset(self.data_dir, split=self.split, H=self.H, W=self.W, focal=self.focal,
                                         single_example=self.single_example, near=self.near, far=self.far)

        self.val_dataset = ImageDataset(self.data_dir, split=self.split, H=self.H, W=self.W, focal=self.focal,
                                         single_example=self.single_example, near=self.near, far=self.far, is_val=True)