import torch
from sdrparse import load
from simulib.platform_helper import SDRPlatform
import numpy as np
from pathlib import Path
from simulib.simulation_functions import findPowerOf2

if __name__ == '__main__':
    fnme = '/home/jeff/SDR_DATA/RAW/12172024/SAR_12172024_113146.sar'
    data_center = np.array([40.093229, -111.768341, 1353.06885])

    sdr_f = load(fnme)
    idxes = np.arange(sdr_f[0].nframes)[::100]
    frames = sdr_f[0].frame_num[idxes]

    rp = SDRPlatform(sdr_f, origin=data_center, channel=0, fs=sdr_f[0].fs)
    fft_sz = findPowerOf2(sdr_f[0].nsam + sdr_f[0].pulse_length)
    pulses = np.fft.ifft(np.fft.fft(sdr_f.getPulses(frames)[1], fft_sz, axis=0).T *
                              sdr_f.genMatchedFilter(0, fft_len=fft_sz), axis=1)[:, :sdr_f[0].nsam]
    # Normalize pulses so that they have a standard deviation of one
    pulse_std = pulses.std(axis=1)
    valids = abs(pulse_std - pulse_std.mean()) <= pulse_std.std()
    pulses = pulses[valids]
    norm_std = np.std(pulses)
    pulses = pulses / norm_std
    pulses = torch.view_as_real(torch.tensor(pulses))

    valid_idxes = idxes[valids]
    pos = torch.tensor(rp.txpos(sdr_f[0].pulse_time[valid_idxes]), dtype=torch.float)
    pans = torch.tensor(rp.pan(sdr_f[0].pulse_time[valid_idxes]), dtype=torch.float32)
    tilts = torch.tensor(rp.tilt(sdr_f[0].pulse_time[valid_idxes]), dtype=torch.float32)
    az_bw = np.float32(rp.az_half_bw)
    el_bw = np.float32(rp.el_half_bw)

    data = torch.cat([pos, pans.unsqueeze(-1), tilts.unsqueeze(-1)], dim=-1)

    fpath = Path(fnme)

    torch.save([data, pulses, norm_std, az_bw, el_bw, sdr_f[0].fc], f'./data/{fpath.stem}_train.pt')