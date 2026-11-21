"""Independent count/hash audit plus three fixed-shot CUDA repeatability checks."""
import hashlib
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.attention_basis_20261029.learning import MatrixAttention
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.input_factors_20261031.stream import predict
from experiments.risk_aligned_online.maturity_jacobian_20261030.run import configure
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    root = Path(__file__).parent
    run = root/'probe_outputs/input_factors360_v1'
    p, done = read(run/'protocol.json'),read(run/'complete.json')
    assert done['complete'] and done['neural_updates']==0
    for name,sha in done['artifact_sha256'].items():
        assert digest(run/name)==sha, name
    # The immutable execution snapshot is authoritative. The live runner has
    # since gained seal-integrity checks, which do not enter numerical replay.
    # Do not mistake that explicit post-run edit for changed historical code.
    live_differences = []
    permitted_postrun_edits = {
        'experiments/risk_aligned_online/input_factors_20261031/run.py',
        'INPUT_FACTORS_PLAN_20261031.md',
    }
    with zipfile.ZipFile(run/'code_snapshot.zip') as archive:
        for name,sha in p['code_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha, name
            if digest(root/name)!=sha:
                assert name in permitted_postrun_edits, name
                live_differences.append(name)
    predictions = read(run/'per_shot.json')
    barrier = read(run/'all_predictions_sealed.json')
    access = read(run/'accesses.json')
    assert len(access)==360 and all(r['sealed_count']==360 for r in access)
    assert [r['shot'] for r in access]==p['order']
    ref = root/p['reference']
    mapping = read(ref/'input_map.json')
    # Independent coefficient calculation, not the production fit_maps helper.
    for arm,column,kind in [('n1_gain',11,'gain'),('CIII_gain',1,'gain'),
                            ('dx_shift',7,'offset'),('dy_shift',8,'offset')]:
        source,target = mapping['source_statistics']['median'][column],mapping['target_statistics']['median'][column]
        gain,offset = np.ones(23),np.zeros(23)
        if kind=='gain':
            gain[column] = source/target
        else:
            offset[column] = source-target
        assert np.array_equal(gain,p['transformations'][arm]['gain'])
        assert np.array_equal(offset,p['transformations'][arm]['offset'])
    stored = {n:[] for n in p['arms']}
    for i,sid in enumerate(p['order']):
        f = run/'shots'/f'{i+1:04d}_{sid}.npz'
        assert digest(f)==barrier['seals'][i]['sha256']
        assert barrier['seals'][i]['shot']==sid
        with h5py.File(io23.path_for('EAST',sid),'r') as h:
            flag = bool(h['meta/IsDisrupt'][()])
            down = float(h['meta/DownTime'][()]) if flag else None
            ds = h['data/ip']
            times = float(ds.attrs['StartTime'])+np.arange(len(ds))/float(ds.attrs['SampleRate'])
        with np.load(f,allow_pickle=False) as a, np.load(Path(p['archive'])/'shots'/f'{i+1:04d}_{sid}.npz',allow_pickle=False) as old:
            assert np.array_equal(a['times'],times)
            assert np.array_equal(a['cold'],old['cold'])
            assert np.array_equal(a['identical'],a['cold'])
            for name in p['arms']:
                score = a[name]
                assert score.shape==times.shape and np.isfinite(score).all()
                hit = np.flatnonzero(score>=p['threshold'])
                lead = round((down-times[hit[0]])*1000.,9) if flag and len(hit) else None
                row = dict(shot=sid,is_disrupt=flag,alarm=bool(len(hit)),
                           lead_ms=lead,timely=lead is not None and lead>30.)
                assert row==predictions[name][i],(name,sid)
                if i<20:
                    assert np.array_equal(score,old['cold'])
                stored[name].append(row)
    sections = {'full360':slice(None),'cold20':slice(20),'suffix340':slice(20,None),'last90':slice(-90,None)}
    for name,rows in stored.items():
        for section,sl in sections.items():
            subset = rows[sl]
            nd = sum(r['is_disrupt'] for r in subset)
            ns = len(subset)-nd
            tp = sum(r['timely'] for r in subset)
            fp = sum(r['alarm'] and not r['is_disrupt'] for r in subset)
            expected = done['results'][name][section]
            assert (tp,fp,nd,ns)==tuple(expected[k] for k in ('tp','fp','n_disrupt','n_safe'))
            assert tp/nd==expected['tpr'] and fp/ns==expected['fpr']
    configure()
    base,_ = load_parent(DEFAULT_PARENT)
    net = MatrixAttention(base.cuda()).requires_grad_(False).eval()
    slow = torch.load(ref/'models/cold.pt',map_location='cuda',weights_only=True)['slow']
    repeats = []
    for i in (20,113,359):
        sid = p['order'][i]
        raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values','available')},mapping)
        # Preserve protocol arm order: JSON writers sort dictionary keys.
        maps = {n:p['transformations'][n] for n in p['arms']}
        actual = predict(net,signal,slow,maps)
        with np.load(run/'shots'/f'{i+1:04d}_{sid}.npz',allow_pickle=False) as a:
            assert all(np.array_equal(actual[n],a[n]) for n in p['arms'])
        repeats.append(sid)
    result = dict(complete=True,source_sha256=digest(DEFAULT_PARENT),full_stream_curves_checked=2160,
                  metric_sections_checked=24,archived_cold_exact=True,all_labels_after360_seals=True,
                  cuda_repeat_shots=repeats,cuda_repeat_curves_exact=18,
                  run_complete_sha256=digest(run/'complete.json'),script_sha256=digest(Path(__file__)),
                  executed_snapshot_checked=True,live_non_numerical_edits=live_differences)
    with Path(__file__).with_name('INPUT_FACTORS_VERIFY_20261031_v2.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(result)


if __name__=='__main__':
    main()
