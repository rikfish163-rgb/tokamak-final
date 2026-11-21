import json

import numpy as np
import pytest

from experiments.risk_aligned_online import commissioning_radial_east_20261111 as replay


def test_resume_compares_json_semantics_and_does_not_change_the_fit(tmp_path):
    prefix = tmp_path/'previous'
    archived = prefix/'executed_code'
    archived.mkdir(parents=True)
    producer = tmp_path/'worker.py'
    previous = archived/producer.name
    previous.write_text('def predict():\n    return 1\n\ndef run():\n    return 0\n')
    producer.write_text('def predict():\n    return 1\n\ndef run():\n    return 1\n')
    old = dict(arms=['a','b'],fit_ids=[2,3],code_sha256={producer.name:replay.sha(previous)})
    (prefix/'protocol.json').write_text(json.dumps(old))
    (prefix/'fit.json').write_text(json.dumps({'ids':[2,3]}))
    signed = np.ones((6,24),np.float32)
    np.savez(prefix/'mapper.npz',signed=signed,unsigned=signed*2.)
    live = dict(old,arms=('a','b'),code_sha256={producer.name:replay.sha(producer)})
    fitted = replay._resume_fit(prefix,producer,live).numpy()
    np.testing.assert_array_equal(fitted,np.stack((signed,signed*2.)))
    with pytest.raises(ValueError,match='different'):
        replay._resume_fit(prefix,producer,dict(live,arms=('b','a')))


def test_retained_shot_requires_complete_curves_and_same_parent(tmp_path):
    path = tmp_path/'retained.npz'
    t = np.arange(4,dtype=float)
    arms = {k:np.zeros_like(t) for k in replay.ARMS}
    saved_parent = {k:v for k,v in arms.items() if k in replay.BASES}
    np.savez(path,times=t,**arms)
    replay._retained(path,t,saved_parent)
    with pytest.raises(AssertionError):
        replay._retained(path,t,dict(saved_parent,fixed=np.ones_like(t)))
    np.savez(path,times=t,**dict(arms,response_plus_old=np.zeros(2)))
    with pytest.raises(ValueError,match='incomplete retained curve'):
        replay._retained(path,t,saved_parent)
