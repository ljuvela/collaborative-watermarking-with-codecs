import matplotlib.pyplot as plt

import soundfile as sf

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


from collaborative_watermarking.third_party.vits import commons
from collaborative_watermarking.third_party.vits import utils

from collaborative_watermarking.third_party.vits.data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from collaborative_watermarking.third_party.vits.models import SynthesizerTrn
from collaborative_watermarking.third_party.vits.text.symbols import symbols
from collaborative_watermarking.third_party.vits.text import text_to_sequence



def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def test_vits_inference():

    hps = utils.get_hparams_from_file("./src/collaborative_watermarking/third_party/vits/configs/ljs_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint("./pre_trained/vits/pretrained_ljs.pth", net_g, None)

    stn_tst = get_text("We propose VITS, Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.", hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    sf.write(os.path.join(output_dir, 'vits.wav'), audio, hps.data.sampling_rate)


    # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate))



# def test_vits_ljs_huggingface():

#     from transformers import VitsModel, AutoTokenizer

#     model = VitsModel.from_pretrained("kakao-enterprise/vits-ljs")
#     import ipdb; ipdb.set_trace()
#     tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-ljs")

#     text = "Hey, it's Hugging Face on the phone"
#     inputs = tokenizer(text, return_tensors="pt")

#     with torch.no_grad():
#         output = model(**inputs).waveform

# from scipy.io.wavfile import write