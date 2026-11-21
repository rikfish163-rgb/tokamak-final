from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.risk_aligned_online import online
from experiments.risk_aligned_online.model import ModelConfig, RiskAlignedModel


torch.set_num_threads(1)


class _WindowData:
    @staticmethod
    def window(shot: dict, end: int, length: int = 8) -> tuple[np.ndarray, np.ndarray]:
        first = max(0, end - length + 1)
        values = np.zeros((length, shot["values"].shape[1]), dtype=np.float32)
        available = np.zeros_like(values, dtype=bool)
        count = end - first + 1
        values[-count:] = shot["values"][first : end + 1]
        available[-count:] = shot["available"][first : end + 1]
        return values, available


def _model() -> RiskAlignedModel:
    torch.manual_seed(29)
    cfg = ModelConfig(channels=3, window=8, patch=2, dim=8, heads=2,
                      rank=2, horizon=3, fast_lr=0.3, fast_radius=1.0)
    return RiskAlignedModel(cfg, torch.zeros(3), torch.ones(3)).eval()


def _shot(length: int = 15, *, outcomes: bool = False) -> dict:
    rng = np.random.default_rng(117)
    values = rng.normal(size=(length, 3)).astype(np.float32)
    available = np.ones_like(values, dtype=bool)
    shot = {
        "shot": 42,
        "values": values,
        "available": available,
        "times": 0.8 + np.arange(length, dtype=np.float64) / 1000.0,
        "rate": 1000.0,
    }
    if outcomes:
        shot.update(
            y=(np.arange(length) % 4 == 0).astype(np.float32),
            y125=(np.arange(length) % 3 == 0).astype(np.float32),
            valid=np.ones(length, dtype=bool),
            is_disrupt=True,
            down=float(shot["times"][-1] + 0.1),
        )
    return shot


def test_intra_predictions_are_prefix_causal_under_future_signal_perturbation():
    model = _model()
    original = _shot(length=12)
    changed = _shot(length=12)
    changed["values"][:5] = original["values"][:5]
    changed["values"][5:] += 1000.0

    first = online.predict_shot(model, original, intra=True, target_domain="EAST", device="cpu",
                                data_api=_WindowData)
    second = online.predict_shot(model, changed, intra=True, target_domain="EAST", device="cpu",
                                 data_api=_WindowData)

    # A support block matures only after its H future values arrive. Changes
    # beginning at sample 5 therefore cannot affect earlier sealed predictions.
    np.testing.assert_allclose(first["scores"][:5], second["scores"][:5], rtol=0, atol=1e-7)
    assert first["fast_updates"][0]["target_end"] == model.cfg.horizon
    assert first["fast_updates"][0]["observed_through"] == model.cfg.horizon


def test_prediction_api_rejects_outcome_fields_before_reveal():
    model = _model()
    shot = _shot()
    shot["y"] = np.zeros(len(shot["times"]), dtype=np.float32)

    with pytest.raises(ValueError, match="outcome|reveal|label"):
        online.predict_shot(model, shot, intra=False, target_domain="EAST", device="cpu",
                            data_api=_WindowData)


def test_intra_fast_state_resets_per_shot_and_never_mutates_persistent_model():
    model = _model()
    shot = _shot()
    before = online.state_hash(model)

    first = online.predict_shot(model, shot, intra=True, target_domain="EAST", device="cpu",
                                data_api=_WindowData)
    second = online.predict_shot(model, shot, intra=True, target_domain="EAST", device="cpu",
                                 data_api=_WindowData)

    np.testing.assert_array_equal(first["scores"], second["scores"])
    assert first["fast_updates"]
    assert online.state_hash(model) == before
    assert first["fast_reset"] is True


def test_sealed_predictions_are_written_before_outcome_attachment(tmp_path: Path):
    model = _model()
    events: list[str] = []

    class FakeData:
        @staticmethod
        def read_shot(device: str, sid: int) -> dict:
            events.append("read")
            shot = _shot(length=7)
            shot['shot'] = sid
            return shot

        window = staticmethod(_WindowData.window)

        @staticmethod
        def attach_outcome(shot: dict, device: str, advanced: dict) -> None:
            events.append("reveal")
            sealed = tmp_path / "1_sealed.npz"
            assert sealed.exists()
            payload = np.load(sealed)
            assert set(payload.files) == {"times", "scores", "shadow_scores"}
            assert not any(name in payload.files for name in ("y", "y125", "valid", "is_disrupt", "down"))
            shot.update(y=np.zeros(7), y125=np.zeros(7), valid=np.ones(7, dtype=bool),
                        is_disrupt=False, down=float(shot["times"][-1]))

    result = online.run_arm(
        model,
        target_domain="EAST",
        order=[{"shot": 1}],
        arm="static",
        threshold=0.5,
        output_dir=tmp_path,
        data_api=FakeData,
        advanced={},
        device="cpu",
    )

    assert events == ["read", "reveal"]
    assert result["audits"][0]["sealed_sha256"]
    assert result["audits"][0]["parent_hash"] == result["audits"][0]["next_hash"]


def test_runner_rejects_mismatched_shot_identity(tmp_path):
    class WrongShot(_WindowData):
        @staticmethod
        def read_shot(domain, sid):
            return _shot()

    with pytest.raises(ValueError, match='shot.*identity'):
        online.run_arm(_model(), target_domain='EAST', order=[{'shot': 1}],
                       arm='static', threshold=.5, output_dir=tmp_path, advanced={},
                       device='cpu', data_api=WrongShot)
    assert not list(tmp_path.glob('*sealed*'))


def test_persistent_update_changes_only_declared_scope_and_measures_controller_norm():
    model = _model()
    shot = _shot(length=24, outcomes=True)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    controller_keys = [key for key in before if key.startswith("update_controller.")]
    allowed = {
        name
        for name, parameter in model.named_parameters()
        if any(parameter is candidate for candidate in model.persistent_parameters())
    }

    audit, replay_record = online.persistent_update(
        model, shot, replay=[], device="cpu", lr=0.05, steps=1,
        data_api=_WindowData, sealed_scores=np.full(len(shot['times']), .5)
    )

    changed = {key for key, value in model.state_dict().items()
               if not torch.equal(value, before[key])}
    assert changed
    assert changed <= allowed
    assert audit["meta_episode_count"] > 0
    expected_controller_delta = np.sqrt(sum(
        float((model.state_dict()[key] - before[key]).double().square().sum())
        for key in controller_keys
    ))
    assert audit["controller_delta_norm"] == pytest.approx(expected_controller_delta, rel=1e-5, abs=1e-9)
    assert audit["controller_delta_norm"] > 0
    assert audit["controller_learned"] is True
    assert replay_record["risk_windows"].ndim == 3
    assert replay_record["meta_episodes"]
    assert all("state" not in episode for episode in replay_record["meta_episodes"])


def test_one_class_prefix_reports_null_rates_and_does_not_select_threshold():
    rows = [{
        "shot": 9,
        "times": np.array([0.8, 0.9]),
        "scores": np.array([0.2, 0.8]),
        "is_disrupt": True,
        "down": 1.0,
    }]

    summary = online.summarize_rows(rows, threshold=0.5, target_domain="EAST")

    assert summary["n_disrupt"] == 1
    assert summary["n_safe"] == 0
    assert summary["tpr"] is None
    assert summary["fpr"] is None
    assert summary["J"] is None
    assert online.posthoc_operating_points(rows, target_domain="EAST") == {}


def test_limit_is_applied_only_after_full_low_to_high_order_is_built():
    class FakeData:
        calls: list[tuple[str, list[int], str]] = []

        @classmethod
        def build_order(cls, device: str, ids: list[int], *, order: str) -> list[dict]:
            cls.calls.append((device, ids.copy(), order))
            return [{"shot": sid, "proxy": rank} for rank, sid in enumerate([4, 2, 8, 1])]

    prefix = online.ordered_prefix(FakeData, "EAST", [1, 2, 4, 8], limit=2)

    assert FakeData.calls == [("EAST", [1, 2, 4, 8], "low_to_high")]
    assert [row["shot"] for row in prefix] == [4, 2]
    assert [row["proxy"] for row in prefix] == [0, 1]


def test_meta_episodes_use_only_valid_preevent_mature_support():
    model = _model()
    shot = _shot(24, outcomes=True)
    shot['down'] = shot['times'][12]
    record = online.select_experience(model, shot, sealed_scores=np.full(24, .5), data_api=_WindowData)
    assert max(record['selected_endpoints']) <= 12
    assert record['meta_episodes']
    for episode in record['meta_episodes']:
        assert episode['support_end'] + model.cfg.horizon == episode['target_end']
        assert episode['target_end'] < episode['query_end'] <= 12
    shot['valid'][:] = False
    audit, empty = online.persistent_update(model, shot, [], 'cpu', .01, 1, data_api=_WindowData, sealed_scores=np.full(24, .5))
    assert audit['steps'] == 0
    assert audit['controller_delta_norm'] == 0
    assert not audit['controller_learned']
    assert not empty['meta_episodes']


def test_fixed_gate_and_frozen_model_support_label_free_adapt():
    model = _model().requires_grad_(False)
    result = online.predict_shot(model, _shot(), intra=True, target_domain='EAST',
                                 device='cpu', data_api=_WindowData, control='fixed')
    assert result['fast_updates']
    assert all(update['gate'] == 1. for update in result['fast_updates'])
    assert all(not p.requires_grad for p in model.parameters())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('domain', ['EAST', 'J-TEXT'])
def test_active_selection_preserves_rare_positive_and_hard_windows(domain):
    shot = _shot(10000, outcomes=True)
    shot['device'] = domain
    shot['y'][:] = 0
    shot['y125'][:] = 0
    labels = shot['y125'] if domain == 'EAST' else shot['y']
    labels[5101:5226] = 1
    scores = labels.copy() * .9 + .05
    scores[1234] = .999
    scores[5123] = .001
    before = scores.copy()
    record = online.select_experience(_model(), shot, sealed_scores=scores, data_api=_WindowData)
    ends = record['selected_endpoints']
    assert len(ends) == len(set(ends)) == 32
    assert {1234, 5123} <= set(ends)
    assert set(labels[ends]) == {0, 1}
    queries = [e['query_end'] for e in record['meta_episodes']]
    assert len(queries) == len(set(queries)) == 8
    assert set(labels[queries]) == {0, 1}
    assert {1234, 5123} <= set(queries)
    np.testing.assert_array_equal(before, scores)


def test_source_smoke_provenance_and_code_hashes():
    result = online.run_provenance({'config': {'evidence': 'SMOKE'}}, limit=0)
    assert result['source_evidence'] == 'SMOKE'
    assert result['evidence'] == 'SMOKE'
    assert set(result['code_sha256']) == {'online.py', 'data.py', 'model.py', 'learning.py'}
    assert all(len(v) == 64 for v in result['code_sha256'].values())


def test_four_arm_two_shot_synthetic_smoke(tmp_path):
    import json

    class FakeData(_WindowData):
        @staticmethod
        def read_shot(domain, sid):
            shot = _shot(12)
            shot['shot'] = sid
            shot['device'] = domain
            return shot

        @staticmethod
        def attach_outcome(shot, domain, advanced):
            n = len(shot['times'])
            positive = shot['shot'] == 2
            shot.update(y=np.full(n, positive, np.float32),
                        y125=np.full(n, positive, np.float32),
                        valid=np.ones(n, bool), is_disrupt=positive,
                        down=float(shot['times'][-1]+.1))

    results = {}
    for arm in ('static', 'intra', 'inter', 'both'):
        results[arm] = online.run_arm(
            _model(), target_domain='EAST', order=[{'shot': 1}, {'shot': 2}],
            arm=arm, threshold=.5, output_dir=tmp_path/arm, advanced={},
            device='cpu', data_api=FakeData, control='fixed')
        audits = results[arm]['audits']
        assert audits[1]['parent_hash'] == audits[0]['next_hash']
        assert audits[0]['cumulative']['tpr'] is None
        assert audits[1]['cumulative_revealed'] == 2
        assert set(results[arm]['posthoc_diagnostic_operating_points']) == {'0.1', '0.15', '0.2'}
        if arm in ('inter', 'both'):
            assert audits[0]['inter_update']['replay_shots'] == 0
            assert audits[1]['inter_update']['replay_shots'] == 1
            assert audits[1]['inter_update']['controller_delta_norm'] == 0
            assert not audits[1]['inter_update']['controller_learned']
            assert audits[1]['inter_update']['control'] == 'fixed'
            assert 'update_controller' not in audits[1]['inter_update']['scope']
        else:
            assert results[arm]['parent_hash'] == results[arm]['final_hash']
        for sid in (1, 2):
            path = tmp_path/arm/f'{sid}_sealed.npz'
            receipt = json.loads((tmp_path/arm/f'{sid}_seal.json').read_text())
            assert receipt['sealed_sha256'] == online.file_hash(path)
        snapshot = torch.load(tmp_path/arm/'snapshot_1.pt', weights_only=False)
        assert snapshot['revealed_shots'] == [1]
    assert len({r['parent_hash'] for r in results.values()}) == 1
    with np.load(tmp_path/'static'/'1_sealed.npz') as static, np.load(tmp_path/'intra'/'1_sealed.npz') as intra:
        np.testing.assert_array_equal(static['scores'], intra['shadow_scores'])
