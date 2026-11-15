"""Fixed cold20 teaching selection and matched-context controls."""

import numpy as np
import torch


def teaching_mask(records):
    if any(r['point_label'] not in (0., 1.) for r in records):
        raise ValueError('binary historical point labels required')
    return np.array([not r['disruptive'] or r['point_label'] == 1. for r in records])


def stratified_folds(ids, labels):
    if len(ids) != 20 or len(set(ids)) != 20 or set(ids) != set(labels):
        raise ValueError('exactly twenty labeled cold shots required')
    safe = [s for s in ids if not labels[s]]
    disruptive = [s for s in ids if labels[s]]
    if len(safe) != 3:
        raise ValueError('this frozen design requires three safe cold shots')
    result = []
    for index in range(3):
        held = disruptive[index::3] + [safe[index]]
        result.append(dict(fold=index, fit=[s for s in ids if s not in held], held=held))
    return result


def context_controls(context, mean):
    if context.ndim != 2 or context.shape[1] != 358 or mean.shape != (358,):
        raise ValueError('B-by-358 context and training-only mean required')
    reverse = torch.cat((context[:, :128], context[:, 128:].reshape(-1, 5, 46)
                         .flip(1).flatten(1)), dim=1)
    return dict(real=context, reverse=reverse,
                mean_feedback=torch.cat((context[:, :128], mean[None, 128:].expand(len(context), -1)), dim=1),
                constant=mean[None].expand(len(context), -1))
