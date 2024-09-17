import torch
import os
import numpy as np

def test_generator():
    
    from collaborative_watermarking.third_party.hifi_gan.models import Generator
    from collaborative_watermarking.third_party.hifi_gan.env import AttrDict

    config = {
        "resblock": "2",
        "num_gpus": 0,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,

        "upsample_rates": [8, 8, 4],
        "upsample_kernel_sizes": [16, 16, 8],
        "upsample_initial_channel": 256,
        "resblock_kernel_sizes": [3, 5, 7],
        "resblock_dilation_sizes": [[1, 2], [2, 6], [3, 12]],
    }

    h = AttrDict(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    generator = Generator(h).to(device)

    num_mels = 80
    num_frames = 30
    batch_size = 1

    x = torch.randn(batch_size, num_mels, num_frames).to(device)

    y = generator(x)

    assert y.shape == (batch_size, 1, np.prod(config['upsample_rates']) * num_frames)

