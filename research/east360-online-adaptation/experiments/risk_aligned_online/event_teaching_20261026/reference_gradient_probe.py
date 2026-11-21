"""Disentangle moving-reference gradients on16 source VAL shots; no fitting."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..direction_teaching_20261026.learning import make_adapter
from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import load_event_memory


def gradient_geometry(after, before, parameters):
    if after.ndim != 1 or after.shape != before.shape or not len(after):
        raise ValueError('nonempty paired point losses required')
    def vector(loss, retain):
        gs = torch.autograd.grad(loss, parameters, retain_graph=retain, allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g in zip(parameters, gs, strict=True)]).detach().double()
    ga = vector(.02*F.softplus((after-before.detach())/.02).mean(), True)
    gb = vector(.02*F.softplus((after.detach()-before)/.02).mean(), True)
    useful = vector(after.mean(), False)
    net = ga+gb
    total = ga.norm()+gb.norm()
    return dict(after_gradient_norm=float(ga.norm()), reference_gradient_norm=float(gb.norm()),
                net_gradient_norm=float(net.norm()),
                cancellation_ratio=float(net.norm()/total) if total > 0 else None,
                net_dot_after_loss_gradient=float(net@useful),
                after_only_dot_after_loss_gradient=float(ga@useful),
                after_reference_cosine=float(F.cosine_similarity(ga,gb,dim=0))
                    if ga.norm()>0 and gb.norm()>0 else None)


def probe(root):
    target = root/'source_reference_gradient16.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(131)
    torch.backends.mha.set_fastpath_enabled(False)
    protocol = json.loads((root/'protocol.json').read_text())
    done = json.loads((root/'complete.json').read_text())
    if sha(root/'targets.pt') != done['targets_sha256']:
        raise ValueError('source labels changed')
    targets = torch.load(root/'targets.pt', map_location='cpu', weights_only=True)['risk']
    cache = Path(protocol['cache_root'])/'cache'
    # Predetermined by source VAL ID and outcome, never by diagnostic results.
    groups = {False: [], True: []}
    for sid in sorted(protocol['val_ids']):
        example = torch.load(cache/f'{sid}.pt', map_location='cpu', weights_only=True)
        if len(groups[example['flag']]) < 8:
            groups[example['flag']].append(example)
        if all(len(v) == 8 for v in groups.values()):
            break
    assert all(len(v) == 8 for v in groups.values())
    base, parent = load_parent(DEFAULT_PARENT)
    model = make_adapter(base)
    receipt = load_event_memory(model, root/'true_epoch2.pt')
    parameters = tuple(model.memory.parameters())
    shift = math.log(parent['threshold']/(1-parent['threshold']))
    rows = []
    for e in groups[False]+groups[True]:
        e.update(targets[e['shot']])
        trace = model.memory.trace(e['means'], e['complete'], mode='memory', return_before=True)
        logits = []
        for name in ('delta', 'before_delta'):
            delta = endpoint_corrections(trace[name], e['ends'])
            logits.append(torch.cat([model.risk(e['h'][i:i+32], e['valid'][i:i+32], delta[i:i+32])
                                     for i in range(0,len(delta),32)]))
        losses = [F.binary_cross_entropy_with_logits(z-shift,e['labels'],reduction='none')[e['direction_mask']]
                  for z in logits]
        if not len(losses[0]):
            rows.append(dict(shot=e['shot'],flag=e['flag'],status='NO_ELIGIBLE_DIRECTION_POINTS'))
            continue
        values = dict(after_loss=float(losses[0].detach().mean()), before_loss=float(losses[1].detach().mean()))
        rows.append(dict(shot=e['shot'],flag=e['flag'],status='ANALYZED',points=len(losses[0]),
                         **values,**gradient_geometry(*losses,parameters)))
        print(json.dumps(rows[-1]),flush=True)
    result = dict(scope='POST_HOC_SOURCE_VAL16_COMPONENT_GRADIENT_NOT_EAST_PERFORMANCE',
        selection='first8_safe_and_first8_disruptive_sorted_VAL_ID',receipt=receipt,rows=rows,
        model_parameters_modified=False,east_outcomes_read=0,seconds=time.monotonic()-started,
        interpretation='Positive gradient dot product predicts improvement only for an infinitesimal '
                       'negative-gradient step; not an optimizer or full-stream result.',
        code_sha256=sha(Path(__file__)),cache_audit_sha256=sha(cache.parent/'audit.json'))
    _durable_json(target,result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    probe(p.parse_args().root)
