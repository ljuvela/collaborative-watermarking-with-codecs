#!/usr/bin/env python
"""
eval_watermark_hifigan

Code to evaluating waveforms rendered by hifi-gan-based watermark model.
Source: https://github.com/ljuvela/CollaborativeWatermarking/blob/master/eval_watermark.py
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import itertools
import time
import argparse
import json

import numpy as np
import pandas as pd

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
from collaborative_watermarking.augmentation import get_augmentations_eval
from collaborative_watermarking.utils import Labels, ScoreColumns

torch.backends.cudnn.benchmark = True

def evaluation(rank, a, h):
    """
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
    # we only need WatermarkModel
    watermark = WatermarkModel(
        model_type=h.watermark_model,
        sample_rate=h.sampling_rate,
        config=h
        ).to(device)

    # load checkpoint
    assert os.path.isdir(a.checkpoint_path), \
        "Cannot found checkpoint {:s}".format(a.checkpoint_path)

    cp_wm = scan_checkpoint(a.checkpoint_path, 'wm_')
    state_dict_wm = load_checkpoint(cp_wm, device)
    watermark.load_state_dict(state_dict_wm['watermark'])

    # wrapper for multi-GPU
    if h.num_gpus > 1:
        watermark = DistributedDataParallel(watermark, device_ids=[rank]).to(device)
    
    if a.pretrained_watermark_path is not None:
        state_dict = torch.load(a.pretrained_watermark_path, map_location='cpu')
        print(f"Loading pre-trained watermark model from {a.pretrained_watermark_path}")
        watermark.load_pretrained_state_dict(state_dict)

    ####
    # data preparation
    ####
    # We need two parallel datasets
    #  one for watermarked audio
    #  one for non-watermarked
    
    # filelist for watermarked 
    marked_filelist = get_filelist(a.input_wavs_marked_file,
                                   a.input_wavs_marked_dir,
                                   ext=a.wavefile_ext)

    # torch dataset
    # no need to load mel anymore, but MelDataset is re-used for conveience
    marked_testset = MelDataset(marked_filelist,
                                h.segment_size, h.n_fft, h.num_mels,
                                h.hop_size, h.win_size, h.sampling_rate,
                                h.fmin, h.fmax, n_cache_reuse=0,
                                shuffle=False, fmax_loss=h.fmax_for_loss, device=device, split=False,
                                fine_tuning=a.fine_tuning)
    marked_sampler = DistributedSampler(marked_testset) if h.num_gpus > 1 else None

    marked_loader = DataLoader(marked_testset, num_workers=h.num_workers, shuffle=False,
                               sampler=marked_sampler,
                               batch_size=1,
                               pin_memory=True,
                               drop_last=True)

    # filelist for non-watermarked 
    nomark_filelist = get_filelist(a.input_wavs_nomark_file,
                                   a.input_wavs_nomark_dir,
                                   ext=a.wavefile_ext)

    nomark_testset = MelDataset(nomark_filelist, h.segment_size, h.n_fft, h.num_mels,
                                 h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax,
                                 n_cache_reuse=0,
                                 shuffle=False, fmax_loss=h.fmax_for_loss, device=device, split=False,
                                 fine_tuning=a.fine_tuning)

    nomark_sampler = DistributedSampler(nomark_testset) if h.num_gpus > 1 else None

    nomark_loader = DataLoader(nomark_testset, num_workers=h.num_workers, shuffle=False,
                                sampler=nomark_sampler,
                                batch_size=1,
                                pin_memory=True,
                                drop_last=True)
    
    ####
    # augmentation methods
    ####
    # each minibatch contains: real, fake, and their repeations
    aug_bs = 2 * a.num_bootstrap_reps
    
    # create augmentation wrappers
    augmentations_test, augmentation_names = get_augmentations_eval(
        h, device=device, batch_size = aug_bs, num_workers=1)
    
    # currently, we use each augmentation specified in the list independently
    #  each time, only apply one type of augmentation to all the data
    list_augmentation = augmentations_test.augmentations

    ####
    # evaluation
    ####
    print("Watermark detection")

    watermark.eval()
    torch.cuda.empty_cache()

    scores = []
    tag_augmentation = []
    lab_truth = []
    sent_name = []
    with torch.no_grad():

        # for each augmentation (watermark detection condition)
        for func_augmentation, name_augmentation in zip(list_augmentation, augmentation_names):
            func_augmentation.eval()
            
            # for each utterance
            for j, (marked_batch, nomark_batch) in enumerate(zip(marked_loader, nomark_loader)):
                
                # mel, audio, filename, mel_loss
                _, marked_audio, marked_filename, _ = marked_batch
                _, nomark_audio, nomark_filename, _ = nomark_batch

                bname = marked_filename[0].split('/')[-1]
                print(f"Evaluating file '{name_augmentation}', '{bname}', ({j+1}/{len(marked_loader)})")                
                # in case the watermarked and original audio slightly differ in length
                wav_len = min([marked_audio.shape[-1], nomark_audio.shape[-1]])
                
                # duplicat the data for args.num_bootstrap_reps
                #  treat them as a single mini-batch
                #  repeat (1, length) -> (reps, 1, length)
                # we evaluation all the repetitions at the same time
                audio_batch = torch.concat(
                    [
                        marked_audio[:, 0:wav_len].unsqueeze(1).repeat([a.num_bootstrap_reps, 1, 1]),
                        nomark_audio[:, 0:wav_len].unsqueeze(1).repeat([a.num_bootstrap_reps, 1, 1]),
                    ])
                
                # apply augmentation
                audio_aug = func_augmentation(audio_batch).to(device)
                
                # detection
                scores_n, scores_p = watermark(audio_aug[:aug_bs//2], audio_aug[aug_bs//2:])

                # save scores and labels as csv file
                #  scores_n and scores_p have shape (num_data, 1)
                scores.append(scores_n.squeeze(1).cpu().numpy())
                scores.append(scores_p.squeeze(1).cpu().numpy())
                # labels (repeat for aug_bs // 2 times)
                lab_truth += [Labels.FAKE] * (aug_bs // 2)
                lab_truth += [Labels.REAL] * (aug_bs // 2)
                # tag of augmentation method
                tag_augmentation += [name_augmentation] * aug_bs
                # utterance name
                sent_name += [x.split('/')[-1] for x in marked_filename] * (aug_bs // 2)
                sent_name += [x.split('/')[-1] for x in nomark_filename] * (aug_bs // 2)

    # merge score
    scores = np.concatenate(scores, axis=0)

    # write to csv via pandas API
    result_pd = pd.DataFrame(
        data={ScoreColumns.SCORE: scores,
              ScoreColumns.LABEL: lab_truth,
              ScoreColumns.FILENAME: sent_name,
              ScoreColumns.AUGMENTATION: tag_augmentation})
    result_pd.to_csv(a.output_score_csv, index=False)
    
    print("Writing detection score to {:s}".format(a.output_score_csv))
    return


def main():
    """
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_marked_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_wavs_nomark_dir', default='LJSpeech-1.1/wavs')    
    parser.add_argument('--input_wavs_marked_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--input_wavs_nomark_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--config_model', default='')
    parser.add_argument('--config_eval', default='')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--pretrained_watermark_path', default=None)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--output_score_csv', default='scores.csv', type=str)
    parser.add_argument('--num_bootstrap_reps', default=5, type=int)
    
    
    a = parser.parse_args()
    print(a)
    # load model configuration file
    with open(a.config_model) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    # load evaluation configuration file
    with open(a.config_eval) as f_eval:
        data_eval = f_eval.read()
    json_config_eval = json.loads(data_eval)
    h_eval = AttrDict(json_config_eval)

    # over-write configuration in h with h_eval
    for key in h_eval.keys():
        print("Evaluation configuration {:s}: {:s}".format(key, str(h_eval[key])))
        h[key] = h_eval[key]
    
    # setup
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        #print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(evaluation, nprocs=h.num_gpus, args=(a, h,))
    else:
        evaluation(0, a, h)

if __name__ == '__main__':
    main()

