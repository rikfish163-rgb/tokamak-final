"""Post-hoc localization of failed target transfer, NEVER threshold selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import io23
from ..anchored_fast_window_20261023.run import ranking
from ..redesign_20261004.prepare_prior import load_parent,DEFAULT_PARENT
from ..simple_online import _durable_json
from ..simple_train import metrics,sha
from .east import load_aligned,read_signal,transform,MAP
from .stream import predict_static
from .train import public


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root
    if (root/'stage_diagnostic.json').exists():raise FileExistsError(root/'stage_diagnostic.json')
    # This diagnostic is not allowed before the full paired run is sealed.
    done=json.loads((root/'completed.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    original,ck=load_parent(DEFAULT_PARENT);original.cuda().eval()
    candidate,_=load_aligned(Path(protocol['source']));candidate.requires_grad_(False)
    mapping=json.loads(MAP.read_text());assert sha(MAP)==protocol['input_map_sha256']
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False);start=time.monotonic()
    rows={a:[] for a in ('original_raw','original_mapped','aligned_raw','aligned_mapped','prepared_off')}
    for i,r in enumerate(json.loads((root/'outcomes.json').read_text())):
        sid=r['shot'];s=read_signal(sid);raw={k:s[k] for k in ('values','available')};mapped=transform(raw,mapping)
        _,valid=io23.labels_at(s['times'],r['is_disrupt'],r['down'],125.)
        predictions={name:predict_static(model,signal) for name,model,signal in
                     (('original_raw',original,raw),('original_mapped',original,mapped),
                      ('aligned_raw',candidate,raw),('aligned_mapped',candidate,mapped))}
        with np.load(root/'shots'/f'{sid}.npz') as z:predictions['prepared_off']=z['off'].copy()
        for name,p in predictions.items():rows[name].append(dict(r,times=s['times'][valid],scores=p[valid]))
        if (i+1)%60==0:print(dict(stage='posthoc_stage_localization',shots=i+1,seconds=time.monotonic()-start),flush=True)
    result={name:dict(ranking=ranking(rr),original_threshold=public(metrics(rr,float(ck['threshold']),lead=30.))) for name,rr in rows.items()}
    for name,rr in rows.items():
        np.savez_compressed(root/f'stage_{name}_curves.npz',**{f'{r["shot"]}_{k}':r[k] for r in rr for k in ('times','scores')})
    report=dict(scope='POSTHOC_TEST_LABEL_DIAGNOSTIC; not deployable thresholds or an independent test',
        completed_sha256=sha(root/'completed.json'),source_sha256=protocol['source_sha256'],
        original_source_sha256=sha(DEFAULT_PARENT),no_new_training=True,no_threshold_selected_for_deployment=True,
        results=result,seconds=time.monotonic()-start,
        decision='DO_NOT_PROMOTE: static target start collapsed and real feedback does not beat constant/reverse controls',
        gate_warning='completed.intra_gate is a local J/FPR numerical screen only, NOT deployment qualification')
    _durable_json(root/'stage_diagnostic.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
