"""Independent EAST recount from seals, raw outcomes and FFN state receipts.

No model training or GPU; the recount does not call the runner's metric helper.
This verifies artifacts/timing, not physical correctness or future performance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from . import io23
from .redesign_20261004.run import DEFAULT_SOURCE
from .simple_online import _durable_json, _hash


DIMENSION = 16576
ZERO_SHA = hashlib.sha256(np.zeros(DIMENSION, np.float32).tobytes()).hexdigest()


def check_timing(a, dimension=DIMENSION):
    zero_sha=hashlib.sha256(np.zeros(dimension, np.float32).tobytes()).hexdigest()
    if (a['scope'] != 'EAST_CAUSAL_REPLAY' or a['fast_parameter_count'] != dimension
            or not a['fast_discarded'] or a['model_parameter_change_norm'] != 0.
            or a['model_buffer_change_norm'] != 0.):
        raise ValueError('temporary/persistent/source scope changed during prediction')
    attempts = a['attempts']
    if len(attempts) != a['attempted_updates'] or sum(r['accepted'] for r in attempts) != a['accepted_updates']:
        raise ValueError('attempt counts disagree')
    previous = zero_sha
    for i, r in enumerate(attempts):
        at = r['fast_at_origin']
        if at['count'] != dimension or at['dtype'] != 'torch.float32':
            raise ValueError('FFN state dimension/dtype changed')
        if i == 0 and (r['origin_index'] != 0 or at['l2'] != 0. or at['sha256'] != zero_sha):
            raise ValueError('each shot must start at zero temporary state')
        if at['sha256'] != previous or r['fast_before']['sha256'] != at['sha256']:
            raise ValueError('temporary state continuity changed')
        origin, maturity = r['origin_index'], r['maturity_index']
        expected = maturity + 1 if maturity + 1 < a['score_count'] else None
        if (maturity != origin + 25 or r['predictions_sealed_through_index'] != maturity
                or (r['accepted'] and r['effective_index'] != expected)
                or (not r['accepted'] and r['effective_index'] is not None)):
            raise ValueError('signal maturity/sealing/effect boundary violated')
        after = r['fast_after']
        if after['count'] != dimension or not np.isfinite(after['l2']) or after['l2'] > 1.000001:
            raise ValueError('temporary norm bound changed')
        if not r['accepted'] and after['sha256'] != at['sha256']:
            raise ValueError('rejected proposal was retained')
        previous = after['sha256']
    return dict(attempted=len(attempts), accepted=sum(r['accepted'] for r in attempts))


def counts(rows):
    nd = int(sum(r['is_disrupt'] for r in rows)); ns = len(rows) - nd
    tp = int(sum(r['timely'] for r in rows))
    fp = int(sum(r['alarm'] and not r['is_disrupt'] for r in rows))
    return dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns,
                tpr=tp/nd if nd else None, fpr=fp/ns if ns else None,
                early_over_3s=int(sum(r['lead_ms'] is not None and r['lead_ms'] > 3000. for r in rows)))


def site_contract(site):
    contracts=dict(persistent_ffn=(16576,'final_persistent_ffn.pt',(16576,)),
                   gradient_transport=(16576,'final_gradient_transport.pt',(16576,)),
                   routed_ffn=(512,'final_routed_ffn.pt',(8,64)),
                   routed_ffn_weight=(65536,'final_routed_ffn_weight.pt',(8,64,128)))
    if site not in contracts: raise ValueError('unknown internal update site')
    return contracts[site]


def verify(directory):
    protocol = json.loads((directory/'protocol.json').read_text())
    reported = json.loads((directory/'comparison.json').read_text())
    site=protocol['internal_site']
    if (site not in ('persistent_ffn','routed_ffn','routed_ffn_weight','gradient_transport') or protocol['first_shot_target_information'] != 0
            or protocol['source_training'] or protocol['jtext_online'] or protocol['alarm_tuning']
            or protocol['source_sha256'] != _hash(DEFAULT_SOURCE)):
        raise ValueError('source/zero-target-information/model-site contract changed')
    dimension,artifact_name,memory_shape=site_contract(site)
    if site=='gradient_transport':
        receipt=protocol['source_gradient_transport']
        if (_hash(directory/'source_gradient_operator.pt')!=receipt['artifact_sha256']
                or receipt['protocol']['parent_sha256']!=protocol['source_sha256']
                or not receipt['protocol']['source_only'] or not receipt['protocol']['no_east_reads']
                or receipt['used_for_persistent_gradient']):
            raise ValueError('source-only fast gradient operator contract changed')
    if site in ('routed_ffn','routed_ffn_weight'):
        from .routed_ffn import load_router
        router=protocol['source_router']
        if (_hash(directory/'source_router.pt') != router['artifact_sha256']
                or router['source_role']!='TRAIN' or router['target_records_accessed']!=0
                or router['label_used_for_key_fitting'] or router['routing'] not in ('local','uniform')):
            raise ValueError('source-only fixed router contract changed')
        loaded=load_router(directory/'source_router.pt',bank_sha256=protocol['source_bank_content_sha256'])
        if router['source_shot_ids']!=loaded['source_shot_ids']:
            raise ValueError('source routing identities changed')
        if site=='routed_ffn_weight' and router['gate_normalization']!='unit_l2':
            raise ValueError('conditional weight routing norm budget changed')
    order = json.loads((directory/'order.json').read_text())[:protocol['selected_shots']]
    identities = [r['shot'] for r in order]
    with zipfile.ZipFile(directory/'code_snapshot.zip') as archive:
        for name, digest in protocol['code_sha256'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('archived code hash mismatch')
    outputs, records, alignments = {}, {}, {}
    for arm in protocol['arms']:
        paths = sorted((directory/arm).glob('*_seal.json'))
        rows = []; attempts = accepted = inter_accepted = 0
        if len(paths) != len(identities): raise ValueError('incomplete sealed cohort')
        for path, sid in zip(paths, identities):
            receipt = json.loads(path.read_text()); curve = path.parent/receipt['npz']
            if receipt['shot'] != sid or receipt['threshold'] != protocol['threshold'] or _hash(curve) != receipt['sha256']:
                raise ValueError('changed shot identity, threshold or prediction seal')
            with np.load(curve, allow_pickle=False) as sealed:
                times, scores = sealed['times'].copy(), sealed['scores'].copy()
                if int(sealed['shot']) != sid: raise ValueError('NPZ identity changed')
            if (times.shape != scores.shape or not len(times) or not np.isfinite(times).all()
                    or (np.diff(times) <= 0).any() or not np.isfinite(scores).all()
                    or ((scores < 0) | (scores > 1)).any()):
                raise ValueError('invalid sealed curve')
            disruptive, down = io23.read_outcome('EAST', sid)
            disruptive = bool(disruptive)
            eligible = times <= down if disruptive else np.ones(len(times), bool)
            hit = np.flatnonzero(eligible & (scores >= protocol['threshold']))
            lead = round((down-times[hit[0]])*1000., 9) if disruptive and len(hit) else None
            row = dict(shot=sid, is_disrupt=disruptive, alarm=bool(len(hit)),
                       lead_ms=lead, timely=bool(lead is not None and lead > 30.))
            a = json.loads(path.with_name(path.name.replace('_seal.json', '_audit.json')).read_text())
            if a['shot'] != sid or a['score_count'] != len(scores) or a['seal_sha256'] != receipt['sha256']:
                raise ValueError('audit identity/length/seal changed')
            if row != a['prediction']: raise ValueError('per-shot first alarm differs from stored result')
            checked = check_timing(a,dimension); attempts += checked['attempted']; accepted += checked['accepted']
            p = a['persistent']
            if p is not None:
                if p['scope']['parameter_names'] != ['memory'] or p['scope']['parameter_count'] != dimension:
                    raise ValueError('post-shot update scope changed')
                if p['drift']['after'] > 1.000001: raise ValueError('persistent norm bound changed')
                inter_accepted += bool(p['accepted'])
            rows.append(row)
        m = counts(rows); original = reported[arm]['metrics']
        if any(m[k] != original[k] for k in ('tp','fp','n_disrupt','n_safe','tpr','fpr')):
            raise ValueError('independent operational recount differs')
        if (accepted != reported[arm]['accepted_intra_updates']
                or inter_accepted != reported[arm]['accepted_inter_updates']):
            raise ValueError('update count differs')
        state = torch.load(directory/arm/artifact_name, map_location='cpu', weights_only=True)
        # The first archived FFN run predates the named policy and had only
        # hard crossings checks; its immutable code snapshot is verified above.
        policy = protocol.get('source_guard_policy', 'crossings')
        if policy not in ('crossings','diagnostic'): raise ValueError('unknown source guard policy')
        if (state['source_sha256'] != protocol['source_sha256'] or state['temporary_retained']
                or state['observed_shots'] != identities
                or (policy == 'crossings' and not state['source_guard']['passed'])
                or state['memory'].shape != memory_shape or not torch.isfinite(state['memory']).all()):
            raise ValueError('invalid final persistent artifact')
        if arm in ('mask_static','static','intra') and state['memory'].any():
            raise ValueError('persistent experience leaked into a non-inter arm')
        alignments[arm] = json.loads((directory/arm/'alignment.json').read_text())
        outputs[arm] = dict(m, seals=len(paths), attempted=attempts,
            accepted_intra=accepted, accepted_inter=inter_accepted,
            startup_30=counts(rows[:30]), after_startup=counts(rows[30:]),
            final_memory_l2=float(state['memory'].double().norm()))
        outputs[arm]['source_guard_at_end'] = state['source_guard']
        records[arm] = rows
    if any(value != alignments['static'] for value in alignments.values()):
        raise ValueError('arms used different label-blind input statistics')
    baseline = {r['shot']:r for r in records['static']}
    for arm in ('intra','inter','both'):
        rescue, loss, fp_removed, fp_added = [], [], [], []
        for r in records[arm]:
            b = baseline[r['shot']]
            if r['is_disrupt']:
                if not b['timely'] and r['timely']: rescue.append(r['shot'])
                if b['timely'] and not r['timely']: loss.append(r['shot'])
            else:
                if b['alarm'] and not r['alarm']: fp_removed.append(r['shot'])
                if not b['alarm'] and r['alarm']: fp_added.append(r['shot'])
        outputs[arm]['paired_vs_static'] = dict(tp_rescued=rescue, tp_lost=loss,
            fp_removed=fp_removed, fp_added=fp_added)
    return dict(directory=str(directory.resolve()), arms=outputs,
                source_sha256=protocol['source_sha256'],
                evidence='DEVELOPMENT PREFIX; verified recount/timing, not independent confirmation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior verification')
    result = verify(args.directory)
    _durable_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
