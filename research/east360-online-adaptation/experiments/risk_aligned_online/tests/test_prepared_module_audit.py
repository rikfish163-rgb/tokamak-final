import pytest
import torch

from experiments.risk_aligned_online.prepared_module_audit import hybrid_state, GROUPS


def states():
    original={prefix+'weight':torch.zeros(2,3) for prefix in GROUPS.values()}
    original.update({'base.risk_head.weight':torch.zeros(1,3),'base.center':torch.zeros(23),
                     'decoder.weight':torch.zeros(2,3),'memory':torch.zeros(4)})
    prepared={k:v.clone() for k,v in original.items()}
    for prefix in GROUPS.values():prepared[prefix+'weight'].fill_(1)
    prepared['decoder.weight'].fill_(2)
    return original,prepared


@pytest.mark.parametrize('group',list(GROUPS))
def test_only_one_original_module_is_replaced(group):
    original,prepared=states()
    result=hybrid_state(original,prepared,group)
    for key,value in result.items():
        assert torch.equal(value,prepared[key] if key.startswith(GROUPS[group]) else original[key])
        assert value.data_ptr()!=original[key].data_ptr()
    assert all(not v.any() for v in original.values())


def test_undeclared_preparation_or_malformed_states_fail():
    original,prepared=states()
    with pytest.raises(ValueError,match='module'):
        hybrid_state(original,prepared,'all')
    prepared['base.risk_head.weight'].fill_(1)
    with pytest.raises(ValueError,match='undeclared'):
        hybrid_state(original,prepared,'embedding')
    original,prepared=states();prepared['base.embedding.weight']=torch.zeros(1)
    with pytest.raises(ValueError,match='shape'):
        hybrid_state(original,prepared,'embedding')
    original,prepared=states();prepared.pop('memory')
    with pytest.raises(ValueError,match='keys'):
        hybrid_state(original,prepared,'embedding')


def test_no_noop_or_nonfinite_candidate_is_silently_accepted():
    original,prepared=states();prepared['base.embedding.weight'].zero_()
    with pytest.raises(ValueError,match='unchanged'):
        hybrid_state(original,prepared,'embedding')
    original,prepared=states();prepared['base.embedding.weight'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):
        hybrid_state(original,prepared,'embedding')
