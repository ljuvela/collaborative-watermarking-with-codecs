#!/usr/bin/env python
"""
metrics.py

functions and classes for computing the evaluation metrics
"""
import argparse

import torch
import numpy as np
import pandas as pd

from collaborative_watermarking.utils import Labels, ScoreColumns

def compute_det_curve(target_scores, nontarget_scores):
    """compute DET curve values
                                                                           
    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores
    
    output
    ------
      frr:   np.array, FRR, (#N, ), where #N is total number of scores + 1
      far:   np.array, FAR, (#N, ), where #N is total number of scores + 1
      thr:   np.array, threshold, (#N, )

    source: https://github.com/asvspoof-challenge/2021/blob/main/eval-package/eval_metrics.py
    """    
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    # false acceptance rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    # Thresholds are the sorted scores
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """
    compute eer
                                                                           
    input
    -----
      target_scores:    np.array, target trial scores
      nontarget_scores: np.array, nontarget trial scores
    
    output
    ------
      eer:   float, EER value in [0, 1]
      thr: float, threshold corresponding to EER

    source: https://github.com/asvspoof-challenge/2021/blob/main/eval-package/eval_metrics.py
    """    
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]




class DiscriminatorMetrics():

    def __init__(self):
        
        self._scores_real = []
        self._scores_fake = []
        
    @property
    def scores_real(self):
        return torch.cat(self._scores_real, dim=0)
    
    @property
    def scores_fake(self):
        return torch.cat(self._scores_fake, dim=0)

    @property
    def eer(self):
        scores_real = self.scores_real.detach().cpu().numpy()
        scores_fake = self.scores_fake.detach().cpu().numpy()
        eer, thresholds = compute_eer(scores_real, scores_fake)
        return eer

    @property
    def accuracy(self):
        return 1.0 - self.eer


    def accumulate(self, disc_real_outputs, disc_fake_outputs):
        """ 
        Args:
            disc_real_outputs: 
                shape is [(batch, timesteps)] * num_models
            disc_fake_outputs
                shape is [(batch, timesteps)] * num_models
        """
      
        scores_real = []
        scores_fake = []
        # classifications for each discriminator
        for d_real, d_fake in zip(disc_real_outputs, disc_fake_outputs):
            # mean prediction over time and channels
            scores_real.append(torch.mean(d_real, dim=(-1,)))
            scores_fake.append(torch.mean(d_fake, dim=(-1,)))

        # Stack scores from different discriminators
        scores_real = torch.stack(scores_real, dim=1) # -> (batch, num_discriminators)
        scores_fake = torch.stack(scores_fake, dim=1) # -> (batch, num_discriminators)

        # Voting by averaging scores
        scores_real_voted = torch.mean(scores_real, dim=-1) # -> (batch,)
        scores_fake_voted = torch.mean(scores_fake, dim=-1) # -> (batch,)

        if scores_real_voted.shape != scores_fake_voted.shape:
            raise ValueError("Real and generated batch sizes must match")
        
        self._scores_real.append(scores_real_voted)
        self._scores_fake.append(scores_fake_voted)


class WatermarkMetric():

    def __init__(self, tag:str,
                model: torch.nn.Module, 
                augmentation: torch.nn.Module):
        self.tag = tag
        self.augmentation = augmentation
        self.model = model
        self.metrics = [DiscriminatorMetrics() for i in range(model.get_num_models())]

    def accumulate(self, input_real, input_fake):

        real_aug = self.augmentation(input_real)
        fake_aug = self.augmentation(input_fake)

        score_real, score_fake = self.model(real_aug, fake_aug)

        for metrics, y_wm_real_i, y_wm_fake_i in zip(self.metrics, score_real, score_fake):
            metrics.accumulate(
                disc_real_outputs = y_wm_real_i,
                disc_fake_outputs = y_wm_fake_i
            )

    def get_eer(self):
        out_val = []
        model_labels = self.model.get_labels()
        for model_label, metric in zip(model_labels, self.metrics):
            out_val.append((model_label, self.tag, metric.eer))
        return out_val
    


class WatermarkEER(torch.nn.Module):

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, real, fake):
        metric = DiscriminatorMetrics()
        with torch.no_grad():
            wm_real, wm_fake = self.detector(real, fake)
        
        metric.accumulate([wm_real], [wm_fake])
        return metric.eer


def compute_metrics_csv(score_pd):
    """
    input: score_pd, pandas dataFrame, containing score and labels
    """
    def _compute_eer_pd(score_pd):
        # retrive scores from dataFrame
        scores_p = score_pd.query('{:s} == "{:s}"'.format(ScoreColumns.LABEL, Labels.REAL))
        scores_n = score_pd.query('{:s} == "{:s}"'.format(ScoreColumns.LABEL, Labels.FAKE))
        # compute EER
        eer, threshold = compute_eer(scores_p[ScoreColumns.SCORE].to_numpy(),
                                     scores_n[ScoreColumns.SCORE].to_numpy())
        return eer

    # eer pooled over all the conditions
    eers = []
    tags = []
    eer = _compute_eer_pd(score_pd)
    eers.append(eer)
    tags.append('pooled')

    # eer in each augmentation method
    aug_methods = sorted(score_pd[ScoreColumns.AUGMENTATION].unique())
    for aug_method in aug_methods:
        score_pd_tmp = score_pd.query('{:s} == "{:s}"'.format(ScoreColumns.AUGMENTATION, aug_method))
        eer = _compute_eer_pd(score_pd_tmp)
        eers.append(eer)
        tags.append(aug_method)

    # print results
    for eer, tag in zip(eers, tags):
        print("EER, {:s}, {:.3f} %".format(tag, eer * 100))

def main():

    print("Evaluating detection performance")
    
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_file', default='scores.csv', type=str)
    parser.add_argument('--score_format', default='csv', type=str)
    a = parser.parse_args()

    # load the score file
    if a.score_format == 'csv':
        # csv format
        score_pd = pd.read_csv(a.score_file)
        compute_metrics_csv(score_pd)
    else:
        print("Format {:s} is not supported".format(a.score_format))
        
if __name__ == "__main__":
    main()
