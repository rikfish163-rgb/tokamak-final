"""Post-hoc history-only inspection of retrieval and evaluation time alignment."""
import argparse
from pathlib import Path
from unittest.mock import patch

from .. import io23,simple_data
from ..causal_input import apply_input_contract
from ..simple_online import _durable_json
from ..simple_train import sha
from ..task_geometry_20261023.run import READ
from .learning import ARMS


def diagnose(root):
    output=root/'timing_diagnostic.json'
    if output.exists():raise FileExistsError(output)
    p,v=READ(root/'protocol.json'),READ(root/'verification.json')
    assert v['status']=='VERIFIED' and v['completed_sha256']==sha(root/'completed.json')
    rows=READ(root/'probes.json');sampling=READ(root/'sampling.json');ids=p['selected_ids']
    paired={a:{(r['shot'],r['query']):r for r in rows if r['arm']==a} for a in ARMS}
    counts={a:{g:dict(n=0,changed_set=0,changed_ordered_list=0) for g in ('safe','disruptive_timely','disruptive_early')} for a in ARMS}
    for a in ARMS:
        for identity,r in paired[a].items():
            one=r['update']['selected_indices'];other=paired['trajectory_rank'][identity]['update']['selected_indices']
            c=counts[a][r['group']];c['n']+=1;c['changed_set']+=int(set(one)!=set(other))
            c['changed_ordered_list']+=int(one!=other)
    outcome,readshot=io23.read_outcome,simple_data.read_shot;access=[]
    def allowed_outcome(device,sid):
        assert device=='EAST' and sid in ids
        access.append(sid);return outcome(device,sid)
    def allowed_signal(device,sid,**kwargs):
        assert device=='EAST' and sid in ids
        return readshot(device,sid,**kwargs)
    transitions=[]
    with patch.object(io23,'read_outcome',side_effect=allowed_outcome),patch.object(simple_data,'read_shot',side_effect=allowed_signal):
        for sid in ids:
            shot=simple_data.attach_outcome(apply_input_contract(simple_data.read_shot('EAST',sid,rate_contract='nominal')),'EAST',{})
            for r in sampling:
                if r['shot']!=sid:continue
                q,t=r['query'],r['count']-1;assert q-t==25
                transitions.append(dict(shot=sid,group=r['group'],key_end=t,query_end=q,
                    label_at_key=int(shot['y'][t]),label_at_query=int(shot['y'][q]),
                    elapsed_ms=float((shot['times'][q]-shot['times'][t])*1000),
                    key_lead_ms=round(float((shot['down']-shot['times'][t])*1000),9) if shot['is_disrupt'] else None,
                    query_lead_ms=r['lead_ms']))
    assert access==ids and len(transitions)==640
    positive=[r for r in transitions if r['group']=='disruptive_timely']
    result=dict(status='DIAGNOSTIC',verification_sha256=sha(root/'verification.json'),
        code_sha256=sha(Path(__file__)),selection_changes_relative_to_trajectory=counts,
        historical_timely_queries=len(positive),crossed_fixed125ms_label_boundary=sum(r['label_at_key']==0 and r['label_at_query']==1 for r in positive),
        rows=transitions,no_test_access=True,
        interpretation='retrieved training endpoint and lookup endpoint coincide; evaluated current-shot endpoint is25points later; valid learning protocol but untested transfer of update across this delay')
    _durable_json(output,result)
    print({k:val for k,val in result.items() if k not in ('rows','selection_changes_relative_to_trajectory')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);diagnose(p.parse_args().root)
