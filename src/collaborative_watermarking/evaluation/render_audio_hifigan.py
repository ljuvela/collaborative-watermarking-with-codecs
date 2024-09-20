#!/usr/bin/env python
"""
render_audio_hifigan

Code to render audio via hifi-gan-based watermark model.
Source: https://github.com/ljuvela/CollaborativeWatermarking/blob/master/render_audio.py

To do: 
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import itertools
import time
import argparse
import json
import soundfile as sf
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from collaborative_watermarking.third_party.hifi_gan.env import AttrDict, build_env
from collaborative_watermarking.third_party.hifi_gan.models import Generator, \
    MultiPeriodDiscriminator, MultiScaleDiscriminator, \
    feature_loss, generator_loss, discriminator_loss

from collaborative_watermarking.third_party.hifi_gan.utils import plot_spectrogram, \
    scan_checkpoint, load_checkpoint, save_checkpoint

from collaborative_watermarking.meldataset import MelDataset, mel_spectrogram, get_filelist

from collaborative_watermarking.metrics import DiscriminatorMetrics
from collaborative_watermarking.models.watermark import WatermarkModel
from collaborative_watermarking.augmentation import get_augmentations

torch.backends.cudnn.benchmark = True

def render_wrapper(rank, a, h):
    """
    """
    
    # render waveform
    wav_data, wav_path = render(rank, a, h)

    # save to disk
    os.makedirs(f"{a.output_dir}/audio", exist_ok=True)
    for wav, file_path in zip(wav_data, wav_path):
        sf.write(file_path, wav, h.sampling_rate)
    print("Rendering done.")
    
    
def render(rank, a, h):
    """wav_data_list, file_path_list = render(rank, a, h)

    render waveform and return a list of rendered waveform data and save path
    """
    print(f"Arguments: {a}")
    print(f"Config: {h}")

    
    ####
    # device & environment setup
    ####

    # multi-GPU support
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank)

    # set device
    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        device = torch.device('cpu')

    ####
    # model initialization
    ####
    # model that renders watermark (hifi-gan generator)
    generator = Generator(h)
    generator = generator.to(device)

    # load checkpoint
    assert os.path.isdir(a.checkpoint_path), \
        "Cannot found checkpoint {:s}".format(a.checkpoint_path)
    
    cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])    

    # wrapper for multi-GPU
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)


    ####
    # data preparation
    ####
    # filelist
    test_filelist = get_filelist(a.input_test_file, a.input_wavs_dir, ext=a.wavefile_ext)

    # torch dataset
    testset = MelDataset(test_filelist, h.segment_size, h.n_fft, h.num_mels,
                         h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                         shuffle=False, fmax_loss=h.fmax_for_loss, device=device, split=False,
                         fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    test_sampler = DistributedSampler(testset) if h.num_gpus > 1 else None

    test_loader = DataLoader(testset, num_workers=h.num_workers, shuffle=False,
                             sampler=test_sampler,
                             batch_size=1,
                             pin_memory=True,
                             drop_last=True)

    ####
    # rendering
    ####
    print("Rendering waveforms")
    
    generator.eval()
    torch.cuda.empty_cache()
    
    max_wav_files = a.max_wav_files_out
    if max_wav_files is None:
        max_wav_files = len(test_loader)

    wav_data = []
    wav_path = []
    with torch.no_grad():
        for j, batch in enumerate(test_loader):
            
            if j >= max_wav_files:
                break

            # mel, audio, filename, mel_loss
            x, y, filename, y_mel = batch

            # rendering
            y_g_hat = generator(x.to(device))
            y_g_np = y_g_hat.detach().cpu().numpy()

            # save
            for i, (y_g, name) in enumerate(zip(y_g_np, filename)):
                bname = os.path.splitext(os.path.basename(name))[0]
            
                # save to output
                print(f"Rendering file '{bname}', ({j * len(filename) + i} / {len(test_loader)})")
                wav_data.append(np.squeeze(y_g))
                wav_path.append(f"{a.output_dir}/{bname}.wav")

    return wav_data, wav_path
    

            
def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_test_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--config', default='')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--output_dir')
    parser.add_argument('--max_wav_files_out', default=None, type=int)
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    
    a = parser.parse_args()
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(render_wrapper, nprocs=h.num_gpus, args=(a, h,))
    else:
        render_wrapper(0, a, h)

if __name__ == '__main__':
    main()

