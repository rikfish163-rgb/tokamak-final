"""Exact selected optimizer replay from each policy's saved pre-update state.

This checks execution reproducibility, not an independently implemented optimizer
or future generalization. Selection is declared before any replay and never
changes the sealed online trajectories.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import simple_data
from ..causal_input import apply_input_contract
from ..context_memory_20261028.checkpoint import load_context_memory
from ..east_context_operator_20261031.run import configure, read
from ..lifecycle_stream_20261025.learning import source_replay, target_example
from ..lifecycle_stream_20261025.run import build_adapter
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..warmstart_alignment_20261022.alignment import transform
from .forward import POLICIES
from .learning import update_with_guards


def main(root):
    output = root/'selected_update_replay.json'
    declaration = root/'selected_update_replay_plan.json'
    if output.exists() or declaration.exists():
        raise FileExistsError('selected replay already attempted; preserve evidence')
    configure()
    p, done = read(root/'protocol.json'), read(root/'completed.json')
    assert done['status'] == 'COMPLETED'
    assert sha(DEFAULT_PARENT) == p['source_sha256']
    receipts = read(root/'receipts.json')
    ids = [r['shot'] for r in p['order']]
    selected = {}
    for name in POLICIES:
        cases = [next(i for i in range(20,360) if receipts[i]['updates'][name]['accepted']),
                 next(i for i in range(20,360) if not receipts[i]['updates'][name]['paired_guard']['checks']['original_pass']),
                 ids.index(65067),359]
        extra = [i for i in range(20,360)
                 if receipts[i]['updates'][name]['paired_guard']['checks']['original_pass']
                 and not receipts[i]['updates'][name]['accepted']]
        if extra:
            cases.append(extra[0])
        selected[name] = sorted(set(cases))
    _durable_json(declaration,dict(completed_sha256=sha(root/'completed.json'),
        selection='first accepted, first original-rejected, first additional rejection if present, rank of65067, final360',
        ranks={name:[i+1 for i in indices] for name,indices in selected.items()},
        classification_results_not_used_to_choose_cases=True,
        diagnostic_case65067_previously_examined=True, verifier_sha256=sha(Path(__file__))))
    base, _ = load_parent(DEFAULT_PARENT)
    model = build_adapter(base.cuda(),readout='full',feedback='progressive',context=True)
    load_context_memory(model,Path(p['source_memory']['path']))
    source, events = source_replay(model)
    assert events == read(root/'source_replay.json')
    mapping = read(root/'input_map.json')
    raw_hashes = read(Path(p['original_parent'])/'input_hashes.json')
    cache = {}

    def example(index,name):
        key = (index,name)
        if key in cache:
            return cache[key]
        sid = ids[index]
        from ..io23 import path_for
        assert sha(path_for('EAST',sid)) == raw_hashes[str(sid)]
        raw = apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal'))
        signal = transform({k:raw[k] for k in ('values','available')},mapping)
        outcome = simple_data.attach_outcome(raw,'EAST',{})
        delta,_ = model.memory_delta(signal)
        with np.load(root/'shots'/f'{index+1:04d}_{sid}.npz',allow_pickle=False) as saved:
            row = target_example(model,outcome,saved[name+'_on'],signal,delta)
        if index >= 20:
            assert row['ends'] == receipts[index]['endpoints'][name]
        cache[key] = row
        return row

    results=[]
    for name, indices in selected.items():
        for index in indices:
            sid=ids[index]
            state_path=root/'states'/f'{index+1:04d}_{sid}.pt'
            assert sha(state_path) == receipts[index]['state_sha256']
            state=torch.load(state_path,map_location='cuda',weights_only=True)[name]
            model.restore(state)
            current=example(index,name)
            replay=[example(i,name) for i in range(index-8,index)]
            expected=receipts[index]['updates'][name]
            actual=update_with_guards(model,[current],replay,source,rule=name,
                eligible_so_far=expected['paired_guard']['eligible_count_before'],
                policy=p['target_loss'],steps=2)
            assert actual == expected,(name,sid,'receipt mismatch')
            if index<359:
                next_state=torch.load(root/'states'/f'{index+2:04d}_{ids[index+1]}.pt',
                                      map_location='cuda',weights_only=True)[name]
            else:
                next_state=torch.load(root/'models'/f'{name}.pt',map_location='cuda',weights_only=True)
            assert all(torch.equal(value,next_state[key]) for key,value in model.adapted_state().items()),(name,sid,'state mismatch')
            results.append(dict(policy=name,rank=index+1,shot=sid,accepted=actual['accepted'],
                                full_receipt_exact=True,retained_state_exact=True))
    result=dict(status='PASS',cases=results,optimizer_steps_reexecuted=2*len(results),
                independent_optimizer_implementation=False,
                sealed_online_outputs_unchanged=True,source_unchanged=sha(DEFAULT_PARENT)==p['source_sha256'],
                selection_plan_sha256=sha(declaration),verifier_sha256=sha(Path(__file__)))
    _durable_json(output,result)
    print(result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    main(parser.parse_args().root)
