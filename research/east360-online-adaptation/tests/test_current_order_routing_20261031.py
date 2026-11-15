import inspect

import pytest

from experiments.risk_aligned_online.lifecycle_stream_20261025 import run


def test_legacy_is_default_and_uses_same_order_function(monkeypatch):
    calls=[]
    monkeypatch.setattr(run,'order_stream',lambda ids:calls.append(ids) or [{'shot':9}])
    assert run.choose_order([9],'legacy_proxy')==[{'shot':9}]
    assert calls==[[9]]
    assert inspect.signature(run.main).parameters['order_policy'].default=='legacy_proxy'


def test_current_only_routes_explicitly_without_legacy(monkeypatch):
    from experiments.risk_aligned_online.current_order_20261031 import order
    monkeypatch.setattr(order,'order_stream',lambda ids:[{'shot':7,'proxy':1.}])
    def forbidden(ids):
        raise AssertionError('legacy Bt must not be used')
    monkeypatch.setattr(run,'order_stream',forbidden)
    assert run.choose_order([7],'current_only')==[{'shot':7,'proxy':1.}]


def test_unknown_order_fails():
    with pytest.raises(ValueError):
        run.choose_order([1],'best_order')
def test_stratified_order_requires_explicit_route(monkeypatch):
    import sys
    import types
    from experiments.risk_aligned_online.lifecycle_stream_20261025 import run
    name = 'experiments.risk_aligned_online.current_order_20261031.stratified'
    fake = types.ModuleType(name)
    calls = []
    def choose(ids):
        calls.append(ids)
        return [{'shot': 17, 'commissioning_phase': 'cold_coverage20'}]
    fake.order_stream = choose
    monkeypatch.setitem(sys.modules, name, fake)
    assert run.choose_order([17], 'current_stratified') == choose([17])
    assert calls == [[17], [17]]

