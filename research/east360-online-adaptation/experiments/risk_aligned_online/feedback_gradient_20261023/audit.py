"""Independent summary arithmetic plus two full same-state diagnostic replays."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .. import io23,simple_data
from ..deepcoral_prefix_20261022.run import state_cpu, assert_unchanged
from ..diverse_support_20261023.run import READ, make_parent
from ..heldout_threshold_20261023.run import REFERENCE, models_for_calibration
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT,PARENT_SHA256
from ..simple_online import _durable_json
from ..simple_train import sha
from .run import replay_shot


def main(root):
    if (root/'verification.json').exists(): raise FileExistsError(root/'verification.json')
    done=READ(root/'completed.json');p=READ(root/'protocol.json');summary=READ(root/'summary.json')
    for name,digest in done['artifact_sha256'].items(): assert sha(root/name)==digest,name
    for name,digest in p['code_sha256'].items(): assert sha(io23.ROOT/name)==digest,name
    for name,digest in p['input_sha256'].items(): assert sha(name)==digest,name
    cases=[r for sid in p['shot_ids'] for r in READ(root/'shots'/f'{sid}.json')]
    assert len(cases)==done['cases']
    assert sum(r['status']=='MEASURED' for r in cases)==done['measured']
    # Check reported diagnostic endpoints against the OFFICIAL scorer's scalar
    # rounding convention, independently of the diagnostic helper.
    for sid in p['shot_ids']:
        raw=simple_data.read_shot('EAST',sid,rate_contract='nominal')
        disruptive,down=io23.read_outcome('EAST',sid)
        for r in READ(root/'shots'/f'{sid}.json'):
            if r['status']=='MEASURED' and disruptive:
                lead=float(round((down-raw['times'][r['query_index']])*1000.,9))
                assert lead>30. and lead==r['query_lead_ms']
    channel_summary={}
    for key,expected in summary.items():
        phase,state=key.split('/')
        rows=[r for r in cases if r['status']=='MEASURED' and r['phase']==phase
              and r['alarm_already_emitted']==(state=='decision_already_alarm')]
        assert expected['records']==len(rows) and expected['shots']==len({r['shot'] for r in rows})
        cos=[r['real_vs_flat']['cosine'] for r in rows if r['real_vs_flat']['cosine'] is not None]
        assert expected['median_real_flat_cosine']==(float(np.median(cos)) if cos else None)
        for arm,counts in expected['candidates'].items():
            values=[r['candidates'][arm]['actual_risk_loss_change'] for r in rows]
            assert counts['better']==sum(v< -1e-7 for v in values)
            assert counts['worse']==sum(v>1e-7 for v in values)
            assert counts['negligible']==sum(abs(v)<=1e-7 for v in values)
            assert counts['median_change']==float(np.median(values))
        channel_summary[key]=dict(sorted({name:float(np.mean([
            next(c for c in r['channels'] if c['feature']==name)['change_output_gradient_norm']
            for r in rows])) for name in io23.RAW23}.items(), key=lambda item:item[1],reverse=True))
    torch.set_num_threads(2);torch.manual_seed(131);parent,_=make_parent()
    model=models_for_calibration(parent)['mapped_fixed'];before=state_cpu(model)
    mapping=READ(REFERENCE/'input_map.json');replayed=[]
    for sid in (62765,87537):
        actual,n=replay_shot(model,sid,mapping,p['threshold'])
        assert actual==READ(root/'shots'/f'{sid}.json')
        replayed.append(dict(shot=sid,verified_updates=n,diagnostic_points=len(actual)))
    assert_unchanged(model,before);assert sha(DEFAULT_PARENT)==PARENT_SHA256
    result=dict(status='VERIFIED_ARITHMETIC_AND_TWO_REPLAYS',completed_sha256=sha(root/'completed.json'),
        audit_sha256=sha(Path(__file__)),cases=len(cases),replayed=replayed,source_unchanged=True,
        strict_lead_boundary_checked=True,
        mean_output_gradient_norm_by_channel=channel_summary,
        note='Audit added after run snapshot. Channel norm is output derivative, not causal physical importance.')
    _durable_json(root/'verification.json',result)
    print({k:v for k,v in result.items() if k!='mean_output_gradient_norm_by_channel'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
