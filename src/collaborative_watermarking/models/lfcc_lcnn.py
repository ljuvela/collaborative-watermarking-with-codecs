import os
import sys
import torch

from ..utils import add_path

from torchaudio.transforms import Resample

import importlib.util
from types import SimpleNamespace


from ..third_party.asvspoof2021.lfcc_lcnn.lfcc_lcnn import Model as LFCC_LCNN_Base

class LFCC_LCNN(LFCC_LCNN_Base):

    def __init__(self, in_dim, out_dim,
                 sample_rate,
                 sigmoid_output=True,
                 dropout_prob=0.7,
                 use_batch_norm=True):
        """
        Args: 
            in_dim: input dimension, default 1 for single channel wav
            out_dim: output dim, default 1 for single value classifier
        """

        prj_conf = SimpleNamespace()
        prj_conf.optional_argument = [""]
        args = None
        mean_std = None
        super().__init__(
            in_dim, out_dim,
            args=args, prj_conf=prj_conf,
            mean_std=mean_std,
            dropout_prob=dropout_prob,
            use_batch_norm=use_batch_norm)
        
        self.sample_rate = sample_rate
        if self.sample_rate != 16000:
            self.resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        else:
            self.resampler = None

        self.sigmoid_out = sigmoid_output


    def eval(self, pass_gradients=True):
        """
        Set model to eval mode

        cuDNN RNNs are not passing gradients in eval mode 
        and these need to be exempted if gradient flow is needed
        """
        if not pass_gradients:
            self.eval()
        else:
            for name, module in self.named_children():
                print(name)
                if name != 'gru':
                    module.eval()
            self.training = False
            

    def forward(self, x):
        """
        Args:
            x: (batch, channels=1, length)

        Returns:
            scores: (batch, length=1)

        """

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        if self.resampler is not None:
            x = self.resampler(x)

        feature_vec = self._compute_embedding(x[:, 0, :], datalength=None)
        # return feature_vec
        scores = self._compute_score(feature_vec, inference=(not self.sigmoid_out))
        scores = scores.reshape(-1, 1)
        return scores
