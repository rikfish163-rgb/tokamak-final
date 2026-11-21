"""Post-training SOURCE VAL diagnostic: can the event head ignore fast memory?"""
import argparse
import json
from pathlib import Path
import time

import torch
from torch import nn

from ..redesign_20261004.prepare_prior import load_parent, DEFAULT_PARENT
from ..delayed_recurrent_memory_20261025.model import endpoint_corrections
from ..source_task_memory_20261026.learning import to_device
from ..direction_teaching_20261026.learning import make_adapter
from ..simple_online import _durable_json
from ..simple_train import sha
from .checkpoint import load_event_memory
from .learning import KINDS, risk_and_features, event_objective


@torch.no_grad()
def probe(root):
    output = root/'source_event_fast_ablation.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(131)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    p = json.loads((root/'protocol.json').read_text())
    done = json.loads((root/'complete.json').read_text())
    assert sha(root/'targets.pt') == done['targets_sha256']
    labels = torch.load(root/'targets.pt', map_location='cpu', weights_only=True)['true']
    cache = Path(p['cache_root'])
    expected = json.loads((root/'val_epoch2.json').read_text())['summary']
    base, _ = load_parent(DEFAULT_PARENT)
    base.cuda()
    results = {}
    for kind in KINDS:
        model = make_adapter(base).cuda()
        path = root/f'{kind}_epoch2.pt'
        receipt = load_event_memory(model, path)
        a = torch.load(path, map_location='cpu', weights_only=True)
        head = nn.Linear(base.cfg.dim, 6).cuda().eval()
        head.load_state_dict(a['event_head_training_only'])
        rows = {mode: [] for mode in ('on', 'off')}
        for sid in p['val_ids']:
            e = torch.load(cache/'cache'/f'{sid}.pt', map_location='cpu', weights_only=True)
            e.update(labels[sid])
            e = to_device(e, 'cuda')
            trace = model.memory.trace(e['means'], e['complete'], mode='memory')
            delta = endpoint_corrections(trace['delta'], e['ends'])
            for mode in rows:
                current = delta if mode == 'on' else torch.zeros_like(delta)
                features = []
                for i in range(0, len(current), 32):
                    _, h = risk_and_features(model, e['h'][i:i+32], e['valid'][i:i+32], current[i:i+32])
                    features.append(h)
                logits = head(torch.cat(features))
                correct = (logits.argmax(-1) == e['event_y']).to(logits)
                rows[mode].append(dict(shot=sid, loss=float(event_objective(logits,e['event_y'],e['event_weight'])),
                    correct_weight=float((correct*e['event_weight']).sum()),
                    weight=float(e['event_weight'].sum())))
        total = {mode: {key: sum(r[key] for r in data) for key in ('loss','correct_weight','weight')}
                 for mode, data in rows.items()}
        for mode in total:
            total[mode]['weighted_accuracy'] = total[mode]['correct_weight']/total[mode]['weight']
        assert abs(total['on']['loss']-expected[kind]['event_loss']) < 1e-6
        assert abs(total['on']['correct_weight']-expected[kind]['event_correct_weight']) < 1e-6
        results[kind] = dict(receipt=receipt, totals=total, per_shot=rows)
    result = dict(scope='POST_HOC_SOURCE_VAL_ONLY_NOT_EAST_PERFORMANCE', source_val=160,
                  results=results, seconds=time.monotonic()-started,
                  interpretation='Known-interval family classification only; no normal class or event-onset detection.',
                  code_sha256=sha(Path(__file__)), source_audit_sha256=sha(root/'audit.json'))
    _durable_json(output, result)
    print(json.dumps({k:v['totals'] for k,v in results.items()}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    probe(p.parse_args().root)
