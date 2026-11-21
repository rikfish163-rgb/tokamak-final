import hashlib
import json

import pytest

import audit_sequence_event_coverage_20261108 as audit


def test_csv_metadata_is_validated_before_any_source_signal_read(tmp_path,monkeypatch):
    history=[dict(shot=1,origins=[127,159,191,223],query=249,label_time=274,group='0_to_0')]*512
    (tmp_path/'fit.json').write_text(json.dumps(dict(history=history)))
    (tmp_path/'protocol.json').write_text(json.dumps(dict(source_steps=512,sequence_steps=4,source_train_ids=[1])))
    csv=tmp_path/'events.csv'
    csv.write_text('shot_no,cohort,disruption_label,event_index,event_class,event_start_s,event_end_s,annotation_status\n'
                   '1,training_pool,0,1,A,0.15,0.20,event\n'
                   '1,training_pool,1,2,A,0.20,0.25,event\n')
    monkeypatch.setattr(audit,'FIT',tmp_path)
    monkeypatch.setattr(audit,'OUTPUT',tmp_path/'coverage.json')
    monkeypatch.setattr(audit.event_probe,'DEFAULT_LABELS',csv)
    monkeypatch.setattr(audit.event_probe,'LABEL_SHA256',hashlib.sha256(csv.read_bytes()).hexdigest())
    monkeypatch.setattr(audit.simple_data,'source_roles',lambda:dict(train=[1],val=[],test=[]))
    def forbidden(*args,**kwargs):
        raise AssertionError('signal read occurred before inconsistent CSV metadata was rejected')
    monkeypatch.setattr(audit.io23,'read_signals',forbidden)
    with pytest.raises(ValueError,match='inconsistent shot metadata'):
        audit.main()


def test_unknown_and_different_interval_ids_do_not_become_one_event():
    assert not any(audit.flags([-1]*5).values())
    assert not audit.flags([1,1,1,2,2])['all_origins_query_one_known_event']
    assert audit.flags([2]*5)['all_origins_query_one_known_event']
