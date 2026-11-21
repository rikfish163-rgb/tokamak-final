"""Order-aware verification; align comparisons by shot identity, not position."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.risk_aligned_online import io23, simple_data
from experiments.risk_aligned_online.causal_input import apply_input_contract
from experiments.risk_aligned_online.context_memory_20261028.checkpoint import load_context_memory
from experiments.risk_aligned_online.lifecycle_stream_20261025.audit import audit, flips
from experiments.risk_aligned_online.lifecycle_stream_20261025.run import build_adapter, predict
from experiments.risk_aligned_online.maturity_jacobian_20261030.run import configure
from experiments.risk_aligned_online.redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from experiments.risk_aligned_online.warmstart_alignment_20261022.alignment import transform


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(records):
    return {r['shot']:dict(r,disruptive=r['is_disrupt']) for r in records}


def guard_summary(folder):
    receipts = read(folder/'receipts.json')
    result = {}
    for arm in ('inter','both'):
        attempts = [r['updates'][arm] for r in receipts if arm in r['updates']]
        rejected = [r for r in attempts if not r['accepted']]
        failures = [[k for k,old in r['guard_before'].items() if r['guard_after'][k]>old+1e-7]
                    for r in rejected]
        result[arm] = dict(attempted=len(attempts),accepted=sum(r['accepted'] for r in attempts),
            rejected=len(rejected),
            source_only_rejections=sum(keys==['source_event'] for keys in failures),
            by_constraint={k:sum(k in keys for keys in failures)
                           for k in ('source_event','historical_D','historical_S')})
    return result


def main():
    root = Path(__file__).parent
    old = root/'experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028'
    new = root/'probe_outputs/current_order_context360_v2'
    completed = read(new/'completed.json')
    assert completed['status']=='COMPLETED'
    # Existing independent post-run scorer verifies raw files, each sealed
    # curve, actual labels, update histories/guards and archived code hashes.
    if (new/'independent_audit.json').exists():
        raise FileExistsError(new/'independent_audit.json')
    with redirect_stdout(io.StringIO()):
        audit(new)
    p,q = read(old/'protocol.json'),read(new/'protocol.json')
    for key in ('source_sha256','threshold','warmup','source_statistics_sha256',
                'target_loss','readout','feedback','source_memory','seed','arms','torch','gpu'):
        assert p[key]==q[key],key
    assert q['order_policy']=='current_only'
    old_rows,new_rows = read(old/'order.json'),read(new/'order.json')
    assert new_rows==read(root/'probe_outputs/current_order_manifest_v1/order.json')
    assert read(old/'source_replay.json')==read(new/'source_replay.json')
    old_pos = {r['shot']:i for i,r in enumerate(old_rows)}
    assert set(old_pos)=={r['shot'] for r in new_rows}
    initial,source_exact = [],0
    for i,r in enumerate(new_rows):
        sid = r['shot']
        with h5py.File(io23.path_for('EAST',sid),'r') as h:
            ds = h['data/ip']
            rate,start = float(ds.attrs['SampleRate']),float(ds.attrs['StartTime'])
            n = int(np.ceil(.1*rate-1e-10))
            current = float(np.median(abs(np.asarray(ds[:n],np.float64))))
            times = start+np.arange(len(ds))/rate
        assert current==r['initial_ip_A']
        initial.append(((current/100000.)**2,sid))
        with np.load(new/'shots'/f'{i+1:04d}_{sid}.npz',allow_pickle=False) as a, \
                np.load(old/'shots'/f'{old_pos[sid]+1:04d}_{sid}.npz',allow_pickle=False) as b:
            assert np.array_equal(a['times'],times) and np.array_equal(a['times'],b['times'])
            assert np.array_equal(a['source'],b['source']),sid
            source_exact += 1
    assert initial==sorted(initial)
    old_pred,new_pred = read(old/'per_shot.json'),read(new/'per_shot.json')
    across = {}
    for arm in q['arms']:
        a,b = keyed(old_pred[arm]),keyed(new_pred[arm])
        ordered = sorted(a)
        across[arm] = flips([a[s] for s in ordered],[b[s] for s in ordered])
    within = {}
    for first,second in [('cold','intra'),('cold','inter'),('inter','both'),
                         ('intra','both'),('both_shadow','both'),('intra_reverse','intra')]:
        a,b = keyed(new_pred[first]),keyed(new_pred[second])
        ordered = sorted(a)
        within[second+'__minus__'+first] = flips([a[s] for s in ordered],[b[s] for s in ordered])
    configure()
    base,_ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(),readout='full',feedback='progressive',context=True)
    load_context_memory(model,Path(q['source_memory']['path']))
    model.restore(torch.load(new/'models/cold.pt',map_location='cuda',weights_only=True))
    mapping = read(new/'input_map.json')
    replayed = []
    for index in (20,113,359):
        sid = new_rows[index]['shot']
        raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values','available')},mapping)
        curves,_,_ = predict(dict(cold=model,inter=model,both=model),signal)
        with np.load(new/'shots'/f'{index+1:04d}_{sid}.npz',allow_pickle=False) as a:
            for arm in ('aligned','cold','intra','intra_reverse'):
                assert np.array_equal(curves[arm],a[arm]),(sid,arm)
        replayed.append(sid)
    result = dict(complete=True,source_curves_exact_by_shot=source_exact,
        order_readonly_ip_recomputed=True,new_order_sha256=sha(new/'order.json'),
        unchanged_source_event_replay=True,cuda_replayed_shots=replayed,
        cuda_replayed_fixed_parent_curves_exact=12,
        persistent_inter_models_not_reexecuted=True,
        across_order_flips_by_identity=across,within_new_order_flips=within,
        guard_rejections=dict(old=guard_summary(old),new=guard_summary(new)),
        old_results=read(old/'completed.json')['metrics'],new_results=completed['metrics'],
        new_completed_sha256=sha(new/'completed.json'),source_checkpoint_sha256=sha(DEFAULT_PARENT),
        script_sha256=sha(Path(__file__)),independent_confirmation=False)
    with Path(__file__).with_suffix('.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(dict(source_exact=source_exact,cuda_replays=replayed,complete=True))


if __name__=='__main__':
    main()
