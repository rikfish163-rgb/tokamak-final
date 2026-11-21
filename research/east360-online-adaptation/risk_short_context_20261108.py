"""Source teaching task construction for short causal histories."""

from numbers import Integral

import numpy as np
import torch

from experiments.risk_aligned_online import simple_data
from probe_risk_masked_utility_20261107 import masked_view


GROUPS = ('0_to_0', '0_to_1', '1_to_1')


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _labels_and_valid(s):
    valid = np.asarray(s['valid'], dtype=bool)
    y = np.asarray(s['y'])
    values = np.asarray(s['values'])
    available = np.asarray(s['available'])
    if (valid.ndim != 1 or y.shape != valid.shape or values.ndim != 2
            or values.shape != (len(valid), 23) or available.shape != values.shape):
        raise ValueError('aligned 23-channel source signals, masks and labels required')
    return y, valid


def context_pools(shots, *, minimum_origin=7):
    """Group complete four-feedback tasks by their two native risk labels."""
    if not _integer(minimum_origin) or minimum_origin not in (7, 127):
        raise ValueError('minimum origin must be 7 or 127')
    pools = {group: [] for group in GROUPS}
    for i, s in enumerate(shots):
        y, valid = _labels_and_valid(s)
        origins = np.arange(minimum_origin, len(valid) - 147, dtype=np.int64)
        if not len(origins):
            continue
        invalid_prefix = np.concatenate(([0], np.cumsum(~valid, dtype=np.int64)))
        starts = np.maximum(0, origins - 127)
        origins = origins[invalid_prefix[origins + 148] == invalid_prefix[starts]]
        for clock in (origins + 122, origins + 147):
            labels = y[clock]
            if not np.all((labels == 0) | (labels == 1)):
                raise ValueError('binary native labels required at task clocks')
        for group in GROUPS:
            current, future = map(int, group.split('_to_'))
            selected = origins[(y[origins + 122] == current) & (y[origins + 147] == future)]
            if len(selected):
                pools[group].append((i, selected))
    if any(not rows for rows in pools.values()):
        raise ValueError('all three native current/future source classes need complete episodes')
    return pools


def sequence_from_note(s, note, device):
    """Rebuild four matured H25 updates and the separate future risk target."""
    y, valid = _labels_and_valid(s)
    origins, q, t = note['origins'], note['query'], note['label_time']
    if (len(origins) != 4 or not all(_integer(o) for o in origins)
            or not _integer(q) or not _integer(t) or origins[0] < 7
            or any(b - a != 32 for a, b in zip(origins, origins[1:]))
            or q != origins[-1] + 26 or t != q + 25 or t >= len(valid)
            or not valid[max(0, origins[0] - 127):t + 1].all()):
        raise ValueError('complete source sequence and its recorded clock required')
    group = note['group']
    if group not in GROUPS or (y[q], y[t]) != tuple(map(int, group.split('_to_'))):
        raise ValueError('recorded group no longer matches native source labels')

    qx, qa = simple_data.window(s, q, 128)
    qx, qa = torch.as_tensor(qx[None], device=device), torch.as_tensor(qa[None], device=device)
    episodes = []
    for origin in origins:
        x, a = simple_data.window(s, origin, 128)
        episode = {key: torch.as_tensor(value[None], device=device) for key, value in
                   dict(x=x, a=a, truth=s['values'][origin + 1:origin + 26],
                        mask=s['available'][origin + 1:origin + 26]).items()}
        episode, masked_query = masked_view(episode, qa)
        episodes.append(episode)
    label = torch.as_tensor(s['y'][t:t + 1], device=device)
    return episodes, qx, masked_query, label, {
        key: note[key] for key in ('shot', 'origins', 'query', 'label_time', 'group')}
