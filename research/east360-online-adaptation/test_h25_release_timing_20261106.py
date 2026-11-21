"""Synthetic, CPU-only checks for the bounded H25 release-clock pilot."""

import numpy as np
import pytest
import torch

import run_h25_release_timing_20261106 as pilot
from experiments.risk_aligned_online.simple23 import Config, Simple23


def test_clocks_share_origins_and_keep_complete_nonoverlapping_windows():
    entries = pilot.schedule(130)
    assert [(e["origin"], e["maturity"], e["early"], e["late"]) for e in entries] == [
        (31, 56, 56, 63),
        (63, 88, 88, 95),
        (95, 120, 120, 127),
    ]
    for clock in ("early", "late"):
        blocks = [pilot.block_indices(130, e[clock]) for e in entries]
        assert all(len(a) <= 32 for a in blocks)
        assert all((a > e[clock]).all() for a, e in zip(blocks, entries, strict=True))
        assert len(np.unique(np.concatenate(blocks))) == sum(map(len, blocks))
    np.testing.assert_array_equal(pilot.block_indices(130, 56), np.arange(57, 89))
    np.testing.assert_array_equal(pilot.block_indices(130, 127), np.arange(128, 130))
    assert pilot.schedule(56) == []
    assert len(pilot.schedule(57)) == 1


def test_guarded_block_commit_forbids_rewrites_and_pre_release_changes():
    native = np.arange(130, dtype=float)
    curve = np.repeat(native[:, None], 2, axis=1)
    written = np.zeros(len(native), dtype=bool)
    pilot.write_block(curve, written, 56, np.full((32, 2), -1.0))
    np.testing.assert_array_equal(curve[:57, 0], native[:57])
    np.testing.assert_array_equal(curve[89:, 0], native[89:])
    with pytest.raises(ValueError, match="overlap"):
        pilot.write_block(curve, written, 60, np.full((32, 2), 2.0))
    with pytest.raises(ValueError, match="length"):
        pilot.write_block(curve, written, 88, np.zeros((33, 2)))
    pilot.write_block(curve, written, 88, np.full((32, 2), 3.0))
    assert written.sum() == 64


def test_matured_gradients_are_extracted_once_for_both_clocks(monkeypatch):
    n = 130
    values = np.arange(n * 23, dtype=np.float32).reshape(n, 23)
    raw = {"values": values, "available": np.ones_like(values, dtype=bool)}
    x = torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 128, 23)
    available = torch.ones_like(x, dtype=torch.bool)
    entries = pilot.schedule(n)
    seen = []
    original = pilot.h25.arrived_targets

    def extract(v, a, origins, maturity):
        seen.append((origins.copy(), maturity.copy()))
        return original(v, a, origins, maturity)

    def gradients(base, xo, ao, target, mask):
        assert base is None and ao.shape == xo.shape
        for i, e in enumerate(entries):
            np.testing.assert_array_equal(target[i], values[e["origin"] + 1 : e["maturity"] + 1])
            assert xo[i, 0, 0] == e["origin"]
        assert mask.all()
        return {"true": torch.ones((len(entries), 3, 128)),
                "reverse": torch.ones((len(entries), 3, 128)) * 2,
                "groups": torch.ones((len(entries), 3), dtype=torch.bool)}

    monkeypatch.setattr(pilot.h25, "arrived_targets", extract)
    monkeypatch.setattr(pilot.h25, "gradients", gradients)
    got = pilot.target_gradients(None, x, available, raw, entries)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0][1], seen[0][0] + 25)
    assert got["true"].shape == got["reverse"].shape == (3, 3, 128)


def test_ambient_projection_preserves_residual_family_span_would_lose():
    past = np.eye(128)[:3]
    family = np.zeros((2, 3, 128))
    family[0, 0, [0, 3]] = 1
    family[0, 1, [1, 4]] = 1
    family[0, 2, [2, 5]] = 1
    family[1] = family[0]
    states = pilot.ambient_directions(family, np.ones(3, dtype=bool), past)
    np.testing.assert_allclose(past @ states.T, 0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 0.35, atol=1e-12)
    assert np.all(states[:, 3:6] < 0)
    q = pilot.geo.basis(family[0])
    assert np.linalg.norm(q @ pilot.mix.projected(q, past, -pilot.mech.unit_average(family[0]))) < 1e-9


def test_candidate_grid_is_common_and_uses_direct_plus_minus(monkeypatch):
    n = 96
    x = torch.zeros((n, 1, 1))
    x[:, 0, 0] = torch.arange(n)
    a = torch.ones_like(x, dtype=torch.bool)
    native = np.arange(n, dtype=np.float32)
    family = np.zeros((2, 3, 128))
    family[0, :2, 3] = 1
    family[1, :2, 4] = 1
    reached = np.eye(128)[:3]
    seen = []

    def direct(base, xx, aa, u):
        seen.append(u.detach().clone())
        return xx[:, 0, 0] + 0.4 * u[:, 3] + 0.5 * u[:, 4]

    monkeypatch.setattr(pilot.mix, "gradient", lambda *args: reached)
    monkeypatch.setattr(pilot.bank.old, "risk", direct)
    states, scale, drift = pilot.choose_states(
        None, x, a, native, 56, family, np.array([True, True, False])
    )
    assert scale == 0.5
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), [0.175, 0.175], atol=1e-7)
    assert 0 < drift <= 0.10
    flat = torch.cat(seen)
    assert bool((flat[:, 3] > 0).any() and (flat[:, 3] < 0).any())
    assert bool((flat[:, 4] > 0).any() and (flat[:, 4] < 0).any())


def test_fewer_than_two_groups_never_attempts_update(monkeypatch):
    def forbidden(*args):
        raise AssertionError("risk and jacobian must not be evaluated")

    monkeypatch.setattr(pilot.mix, "gradient", forbidden)
    states, scale, drift = pilot.choose_states(
        None, torch.empty(0), torch.empty(0), np.zeros(0), 56,
        np.ones((2, 3, 128)), np.array([True, False, False]),
    )
    assert not states.any() and scale == drift == 0


def test_nonfeasible_common_grid_leaves_both_arms_native(monkeypatch):
    x = torch.zeros((80, 1, 1))
    a = torch.ones_like(x, dtype=torch.bool)
    family = np.zeros((2, 3, 128))
    family[0, :2, 3] = 1
    family[1, :2, 4] = 1
    monkeypatch.setattr(pilot.mix, "gradient", lambda *args: np.eye(128)[:3])
    monkeypatch.setattr(
        pilot.bank.old, "risk", lambda base, xx, aa, u: 100 * (u[:, 3] + u[:, 4])
    )
    states, scale, drift = pilot.choose_states(
        None, x, a, np.zeros(80), 56, family, np.array([True, True, False])
    )
    assert not states.any() and scale == drift == 0


def test_full_synthetic_model_replay_matches_truncated_prefix():
    torch.manual_seed(61106)
    prior_threads = torch.get_num_threads()
    prior_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        base = Simple23(
            Config(fast_site="tokens", fast_rule="normalized"),
            torch.zeros(23), torch.ones(23),
        ).eval().requires_grad_(False)
        rng = np.random.default_rng(61106)
        values = rng.normal(size=(100, 23)).astype(np.float32)
        mask = np.ones_like(values, dtype=bool)
        raw = {"values": values, "available": mask, "times": np.arange(100) / 1000}
        short = {key: value[:75].copy() for key, value in raw.items()}
        full_trace = pilot.trace_shot(base, raw)
        prefix_trace = pilot.trace_shot(base, short)
        np.testing.assert_allclose(full_trace["native"][:75], prefix_trace["native"], atol=2e-5, rtol=0)
        for clock, first_release in (("early", 56), ("late", 63)):
            np.testing.assert_allclose(
                full_trace["curves"][clock][:75], prefix_trace["curves"][clock], atol=2e-5, rtol=0
            )
            np.testing.assert_array_equal(
                full_trace["curves"][clock][: first_release + 1, 0],
                full_trace["native"][: first_release + 1],
            )
            assert full_trace["events"][0][clock + "_scale"] > 0
            assert full_trace["events"][0][clock + "_trust_drift"] <= 0.10
        assert not any(p.grad is not None for p in base.parameters())
    finally:
        torch.backends.mha.set_fastpath_enabled(prior_fastpath)
        torch.set_num_threads(prior_threads)


def test_offline_coverage_distinguishes_new_gradient_from_previous_late_action():
    n = 130
    native = np.linspace(-1, 1, n)
    entries = pilot.schedule(n)
    trace = {
        "native": native,
        "curves": {c: np.repeat(native[:, None], 2, axis=1) for c in pilot.CLOCKS},
        "written": {}, "supported": {}, "active": {}, "events": [],
    }
    for clock in pilot.CLOCKS:
        covered = np.zeros(n, dtype=bool)
        for e in entries:
            covered[pilot.block_indices(n, e[clock])] = True
        trace["written"][clock] = covered.copy()
        trace["supported"][clock] = covered.copy()
        trace["active"][clock] = covered.copy()
    for e in entries:
        trace["events"].append({
            **e, "groups": 3, "early_scale": 1.0, "late_scale": 1.0,
            "early_trust_drift": 0.03, "late_trust_drift": 0.04,
        })
    scored = pilot.score_shot(trace, np.arange(n) / 1000, 0.112, 25.0)
    assert scored["coverage"]["canonical_positive_points"] == 15
    assert scored["coverage"]["early"]["first_action_index"] == 57
    second = scored["per_release"][1]
    assert (second["newly_actionable_start"], second["newly_actionable_end"]) == (89, 95)
    assert second["newly_actionable_positive_points"] == 7
    assert second["early_new_gradient_action_positive_points"] == 7
    assert second["previous_late_action_positive_points"] == 7
    assert scored["metrics"]["original"]["canonical_peak_index"] == 101


def test_all_six_traces_are_fixed_before_any_outcome_reveal(monkeypatch):
    trace_ids, revealed, outputs = [], [], []

    class FakeHashes:
        cache = {}

        def __call__(self, path):
            return f"hash-{path}"

        def check(self):
            pass

    class FakeOutput:
        def exists(self):
            return False

    class FakeBase:
        def parameters(self):
            return []

    config = {"source_state_sha256": "state", "threshold": pilot.r.THRESHOLD}
    raw_hashes = {sid: f"hash-{sid}" for sid in pilot.LATE_IDS}
    monkeypatch.setattr(pilot, "Hashes", FakeHashes)
    monkeypatch.setattr(pilot.torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(pilot.torch.backends.mha, "set_fastpath_enabled", lambda enabled: None)
    monkeypatch.setattr(pilot, "locked_inputs", lambda h: (config, raw_hashes))
    monkeypatch.setattr(pilot.r, "load_parent", lambda _: (FakeBase(), {"threshold": pilot.r.THRESHOLD}))
    monkeypatch.setattr(pilot.r, "state_digest", lambda _: "state")
    monkeypatch.setattr(pilot.r.io23, "path_for", lambda device, sid: sid)
    monkeypatch.setattr(pilot.r.simple_data, "read_shot", lambda device, sid: {"shot": sid, "times": np.arange(3)})

    def trace(base, raw):
        trace_ids.append(raw["shot"])
        return {"id": raw["shot"]}

    def advanced():
        assert tuple(trace_ids) == pilot.LATE_IDS
        return {}

    def outcome(device, sid):
        assert tuple(trace_ids) == pilot.LATE_IDS
        revealed.append(sid)
        return True, 0.5

    monkeypatch.setattr(pilot, "trace_shot", trace)
    monkeypatch.setattr(pilot.r.simple_data, "advanced_times", advanced)
    monkeypatch.setattr(pilot.r.io23, "read_outcome", outcome)
    monkeypatch.setattr(pilot.r.io23, "effective_horizon_ms", lambda sid, adv: (25.0, None, False))
    monkeypatch.setattr(pilot, "score_shot", lambda *args: {"metrics": {"original": {"timely": False}}})
    monkeypatch.setattr(pilot.r, "write_new", lambda path, value: outputs.append(value))
    pilot.run(FakeOutput())
    assert tuple(revealed) == pilot.LATE_IDS
    assert len(outputs) == 1 and len(outputs[0]["shots"]) == 6


def test_input_lock_includes_frozen_protocol_and_only_original_train(monkeypatch):
    from pathlib import Path

    prior = Path("/synthetic/output_trust")
    inventory_path = Path("/synthetic/train_coverage.json")
    protocol_path = Path("/synthetic/release_protocol.md")
    monkeypatch.setattr(pilot, "PRIOR", prior)
    monkeypatch.setattr(pilot, "INVENTORY", inventory_path)
    monkeypatch.setattr(pilot, "PROTOCOL", protocol_path)
    train = [*pilot.LATE_IDS, *range(1, 1348)]
    inventory = [{"shot": sid, "sha256": f"raw-{sid}"} for sid in train]
    paths = {
        protocol_path: pilot.PROTOCOL_SHA256,
        prior / "train_crossfit.json": pilot.AUDIT_SHA256,
        prior / "completed.json": "table-hash",
        prior / "protocol.json": "table-protocol-hash",
        inventory_path: "inventory-hash",
        Path(pilot.r.DEFAULT_PARENT).resolve(): "parent-hash",
        pilot.r.io23.JTEXT_ADVANCED_TIME_PATH.resolve(): "advanced-hash",
    }
    config = {
        "status": "LOCKED_OUTPUT_TRUST_TABLE", "roles": {"train": train},
        "threshold": pilot.r.THRESHOLD, "logit_threshold": pilot.mech.TAU,
        "limit": pilot.LIMIT, "scales": pilot.bank.tr.SCALES,
        "source_state_sha256": "state",
        "input_sha256": {
            str(inventory_path.resolve()): "inventory-hash",
            str(Path(pilot.r.DEFAULT_PARENT).resolve()): "parent-hash",
            str(pilot.r.io23.JTEXT_ADVANCED_TIME_PATH.resolve()): "advanced-hash",
        },
    }
    docs = {
        prior / "fit_completed.json": {
            "train_audit_sha256": pilot.AUDIT_SHA256, "train_table_sha256": "table-hash",
            "source_state_sha256": "state",
        },
        prior / "train_crossfit.json": {"geometry": {"non_timely_shots": list(pilot.LATE_IDS)}},
        prior / "completed.json": {"protocol_sha256": "table-protocol-hash"},
        prior / "protocol.json": config,
        inventory_path: {"inventory": inventory},
    }
    touched = []

    def fake_hash(path):
        path = Path(path)
        touched.append(path)
        return paths.get(path, "other-hash")

    monkeypatch.setattr(pilot.r, "read", lambda path: docs[Path(path)])
    _, selected = pilot.locked_inputs(fake_hash)
    assert protocol_path in touched
    assert selected == {sid: f"raw-{sid}" for sid in pilot.LATE_IDS}
    paths[protocol_path] = "different-protocol"
    with pytest.raises(ValueError, match="protocol changed"):
        pilot.locked_inputs(fake_hash)
