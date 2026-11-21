import hashlib
import json

import pytest

from . import prepare_stream


def test_composition_cannot_change_already_selected_order(tmp_path,monkeypatch):
    events=[]
    order=[{'shot':2},{'shot':4},{'shot':7}]
    def build(**kwargs):
        events.append('selection_finalized')
        assert kwargs['order']=='shot_number'
        return order,{'roster':{'shot_ids':[2,4,7],'selection':'all compatible'},'errors':[]}
    def outcome(device,sid):
        assert events[0]=='selection_finalized'
        events.append(sid)
        return sid==4, .5 if sid==4 else None
    monkeypatch.setattr(prepare_stream,'build_manifest',build)
    monkeypatch.setattr(prepare_stream.io23,'read_outcome',outcome)
    output=tmp_path/'stream'
    result=prepare_stream.save_stream(output)
    assert json.loads((output/'order.json').read_text())==order
    assert events==['selection_finalized',2,4,7]
    assert result['total']['n_disrupt']==1 and result['total']['n_safe']==2
    assert (output/'order.json').read_bytes()==prepare_stream.manifest_bytes(order)
    with pytest.raises(FileExistsError):
        prepare_stream.save_stream(output)
