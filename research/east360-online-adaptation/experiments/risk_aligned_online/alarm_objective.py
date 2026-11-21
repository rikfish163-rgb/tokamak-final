"""Offline, shot-level surrogate for the fixed first-crossing alarm rule.

This objective is our diagnostic design, not a reproduction of a published
fusion algorithm. It changes the OUTER adaptation objective, not stored
max(25 ms, AdvancedTime) endpoint labels. True outcomes are forbidden in the
within-shot learner. Unbounded early alarms are allowed by the current metric;
therefore this loss alone cannot establish useful physical warning timing.
"""
import math

import numpy as np
import torch
from torch.nn import functional as F


def timely_shot_loss(logits, rows, threshold, minimum_lead_ms=10.):
    """One BCE per shot, with the fixed alarm threshold as logit origin.

    Positive bag: observed endpoints strictly before the timely deadline.
    Negative bag: all observed endpoints of a safe shot. Hard max implements
    the existence of a threshold crossing, without a time-point independence
    assumption. Whole-shot labels and the deadline are offline targets only.
    """
    if logits.ndim != 2 or logits.shape[0] != len(rows) or not rows:
        raise ValueError('nonempty matching [batch,time] logits and rows required')
    if not 0 < threshold < 1 or not math.isfinite(minimum_lead_ms) or minimum_lead_ms < 0:
        raise ValueError('finite threshold in (0,1) and nonnegative lead required')
    selected, targets = [], []
    for i, row in enumerate(rows):
        times = np.asarray(row['times'], dtype=np.float64)
        if not len(times) or len(times) > logits.shape[1] or not np.isfinite(times).all():
            raise ValueError('valid observation times required')
        mask = np.ones(len(times), bool)
        if row['is_disrupt']:
            if row['down'] is None or not np.isfinite(row['down']):
                raise ValueError('disruptive training shot requires down time')
            # Same numerical convention as simple_train.metrics.
            mask = np.round((row['down']-times)*1000., 9) > minimum_lead_ms
        if not mask.any():
            raise ValueError('training shot has no timely observation')
        selected.append(logits[i, :len(times)][torch.as_tensor(mask, device=logits.device)].max())
        targets.append(float(row['is_disrupt']))
    margin = torch.stack(selected) - math.log(threshold/(1.-threshold))
    return F.binary_cross_entropy_with_logits(margin, logits.new_tensor(targets))
