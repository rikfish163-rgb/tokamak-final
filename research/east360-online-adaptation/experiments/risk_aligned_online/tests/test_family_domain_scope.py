from argparse import Namespace

import pytest
import torch

from experiments.risk_aligned_online import context_experiment, family_experiment
from experiments.risk_aligned_online.family_feedback import signal_target23
from experiments.risk_aligned_online.simple23 import Config, Simple23


@pytest.mark.parametrize('legacy_source_online_file', [False, True])
def test_east_replay_never_reads_or_requires_a_jtext_online_val_file(tmp_path, monkeypatch,
                                                                  legacy_source_online_file):
    checkpoint = tmp_path/'adapter.pt'
    if legacy_source_online_file:
        (tmp_path/'val.json').write_text('{"source_adapted_gate":{"passed":false}}')
    protocol = dict(candidate='family_patch_input', grouping='physical', patch=8, rank=4,
                    horizon=8, parent='qualified_source.pt', parent_sha256='static_parent_hash')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: dict(protocol=protocol))
    calls = []

    def replay(args, *, adapter_factory, target_fn):
        calls.append(args)
        assert target_fn is signal_target23
        assert adapter_factory(64, 8).state_dim == 16
        assert args.mask_channel is None

    monkeypatch.setattr(family_experiment, 'replay', replay)
    args = Namespace(adapter=checkpoint, output=tmp_path/'east', limit=12, sample_count=None,
                     arms=['static', 'learned'])
    family_experiment.replay_east(args)
    assert calls == [args]


def test_source_fit_only_uses_offline_train_and_never_runs_jtext_online_eval(tmp_path, monkeypatch):
    parent = Simple23(Config(window=32, fast_site='tokens'), torch.zeros(23), torch.ones(23))
    parent.eval().requires_grad_(False)
    monkeypatch.setattr(family_experiment, 'qualify_source',
                        lambda _: dict(qualified=True, checkpoint_sha256='static_parent_hash'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(family_experiment, 'load_parent', lambda _: (parent, dict(threshold=.7)))
    monkeypatch.setattr(family_experiment.simple_data, 'source_roles',
                        lambda: dict(train=[11], val=[22], test=[33]))
    monkeypatch.setattr(family_experiment.simple_data, 'advanced_times', lambda: {})
    seen = []

    def offline_shots(ids, advanced):
        seen.append(ids)
        assert ids == [11]
        return []

    def forbidden(*a, **k):
        raise AssertionError('J-TEXT online evaluation is outside this workflow')

    monkeypatch.setattr(family_experiment.simple_train, 'load_shots', offline_shots)
    monkeypatch.setattr(context_experiment, 'evaluate', forbidden)
    # Also catch the old imported alias until the implementation removes it.
    monkeypatch.setattr(family_experiment, 'evaluate', forbidden, raising=False)
    monkeypatch.setattr(family_experiment, 'episode_pools', lambda *a: {})
    monkeypatch.setattr(family_experiment, 'sample', lambda *a: {})
    monkeypatch.setattr(family_experiment, 'train_step', lambda *a, **k: dict(signal_loss=.1))
    args = Namespace(parent=tmp_path/'source.pt', output=tmp_path/'offline_fit', steps=1,
                     batch_size=2, unroll=1, mode='aligned', grouping='physical')
    family_experiment.train(args)
    assert seen == [[11]]
    assert (args.output/'adapter.pt').exists()
    assert not (args.output/'val.json').exists()
    stored = torch.load(args.output/'adapter.pt', map_location='cpu', weights_only=False)
    assert stored['protocol']['online_domain'] == 'EAST'
    assert stored['protocol']['jtext_online_evaluation'] is False
