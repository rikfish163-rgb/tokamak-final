"""Locked test identities and strict warning-time rules from paper_131 §4.3.1.

These local test splits have participated in method development. Their role
is TEST, but they are not an untouched future confirmation set. No threshold
selection occurs here. Training-label horizons are deliberately unaffected.
"""
import json

from . import io23
from .simple_online import _hash
from .simple_train import metrics

SPECS={
    'J-TEXT':dict(path=io23.JTEXT_INDEX,
        sha256='30b7ddbd6ab9f1fc1386c250c01663f80af23f1e31a9c987101ca32c4dc6c1b5',
        name='J-TEXT TEST220',count=220,lead_ms=10.),
    'EAST':dict(path=io23.EAST_COHORT,
        sha256='657cab542bd987f9c97c3cb88867a9cc828373b32e6ee58d450451461c6ec825',
        name='EAST HDF5-authoritative TEST360',count=360,lead_ms=30.),
}


def load_test_contract(device):
    if device not in SPECS:
        raise ValueError('explicit J-TEXT or EAST test required')
    spec=SPECS[device]
    if _hash(spec['path'])!=spec['sha256']:
        raise ValueError('locked test manifest changed; review its provenance explicitly')
    if device=='J-TEXT':
        ids=io23.source_roles()['test']
    else:
        split=json.loads(spec['path'].read_text())['test']
        disruptive=split['disruption_shots'];safe=split['non_disruption_shots']
        if len(disruptive)!=180 or len(safe)!=180 or set(disruptive)&set(safe):
            raise ValueError('canonical EAST test must contain disjoint180D/180S')
        ids=disruptive+safe
    if (len(ids)!=spec['count'] or len(set(ids))!=spec['count']
            or any(type(sid) is not int or sid<=0 for sid in ids)):
        raise ValueError('distinct complete canonical test IDs required')
    return dict(device=device,name=spec['name'],shot_ids=sorted(ids),
        manifest=str(spec['path'].resolve()),manifest_sha256=spec['sha256'],
        lead_ms=spec['lead_ms'],lead_comparison='strict greater than',
        expected_disrupt=spec['count']//2,expected_safe=spec['count']//2,
        independent_confirmation=False,
        scope='LOCKED TEST SPLIT, DEVELOPMENT-EXPOSED; NOT INDEPENDENT CONFIRMATION')


def validate_test_ids(device,ids):
    contract=load_test_contract(device)
    actual=list(ids)
    if (any(type(sid) is not int for sid in actual)
            or len(actual)!=len(contract['shot_ids'])
            or len(set(actual))!=len(actual) or set(actual)!=set(contract['shot_ids'])):
        raise ValueError('full locked test identities required; no validation, subset or replacement shots')
    return contract


def score_fixed_test(device,rows,threshold):
    """Score already-sealed predictions with the unchanged caller threshold."""
    contract=validate_test_ids(device,[r['shot'] for r in rows])
    nd=sum(bool(row['is_disrupt']) for row in rows)
    if nd!=contract['expected_disrupt'] or len(rows)-nd!=contract['expected_safe']:
        raise ValueError('revealed HDF5 test class counts disagree with the locked contract')
    return metrics(rows,threshold,lead=contract['lead_ms'])
