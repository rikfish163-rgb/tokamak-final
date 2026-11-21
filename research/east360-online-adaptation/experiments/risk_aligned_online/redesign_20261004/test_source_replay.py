"""CPU-only acceptance tests; any serialized artifacts live in pytest tmp_path."""
import copy
import hashlib

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.simple23 import Config, Simple23
from experiments.risk_aligned_online.redesign_20261004 import source_replay as replay
from experiments.risk_aligned_online.redesign_20261004.embedding_model import EmbeddingModel
from experiments.risk_aligned_online.redesign_20261004.model import ContextModel


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class TrainData:
    """Synthetic signals/outcomes through the real label and window utilities."""
    def __init__(self, count=8):
        self.ids = list(range(1, 2 * count + 1))
        self.reads = []
        self.outcomes = []

    def source_roles(self):
        return dict(train=self.ids, val=[9001], test=[9002])

    def advanced_times(self):
        return {sid: 40. if sid % 2 else 5. for sid in self.ids + [9001, 9002]}

    def read_shot(self, device, sid):
        assert device == "J-TEXT" and sid in self.ids
        self.reads.append(sid)
        rng = np.random.default_rng(sid)
        return dict(device=device, shot=sid, times=np.arange(240) / 1000.,
                    values=rng.normal(size=(240, 23)).astype(np.float32),
                    available=np.ones((240, 23), bool))

    def attach_outcome(self, shot, device, advanced):
        assert set(advanced) <= set(self.ids)
        self.outcomes.append(shot["shot"])
        disruptive = bool(shot["shot"] % 2)
        horizon = io23.effective_horizon_ms(shot["shot"], advanced)[0]
        down = .24 if disruptive else None
        y, valid = io23.labels_at(shot["times"], disruptive, down, horizon)
        return dict(shot, y=y, valid=valid, down=down,
                    is_disrupt=disruptive, horizon=horizon)


def teacher():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131)
        base = Simple23(Config(fast_site="tokens", fast_rule="normalized", layers=1),
                        torch.linspace(-1, 1, 23), torch.linspace(.5, 2, 23))
    with torch.no_grad():
        base.fast_origin.copy_(torch.linspace(-.2, .3, 128))
    return base.requires_grad_(False).eval()


def bank_case(count=8):
    base, data = teacher(), TrainData(count)
    bank = replay.build_source_bank(base, source_parent_sha256="a" * 64,
                                    data_api=data, per_class=count, seed=131)
    return base, data, bank


def test_train_only_balanced_distinct_deterministic_raw_causal_windows():
    base, data, bank = bank_case()
    other = replay.build_source_bank(base, source_parent_sha256="a" * 64,
                                     data_api=TrainData(), per_class=8, seed=131)
    replay.validate_bank(bank)
    assert bank["bank_sha256"] == other["bank_sha256"]
    assert bank["x"].shape == (16, 128, 23)
    assert bank["available"].dtype == torch.bool
    assert bank["labels"].sum().item() == 8
    assert len(set(bank["shotids"].tolist())) == 16
    assert set(data.reads) <= set(data.ids)
    assert set(data.outcomes) <= set(data.ids)
    for row, sid in enumerate(bank["shotids"].tolist()):
        advanced = {k: v for k, v in data.advanced_times().items() if k in data.ids}
        shot = data.attach_outcome(data.read_shot("J-TEXT", sid), "J-TEXT", advanced)
        end = bank["endindices"][row].item()
        np.testing.assert_array_equal(bank["x"][row], shot["values"][end - 127:end + 1])
        assert shot["valid"][end] and shot["y"][end] == bank["labels"][row].item()
        if bank["labels"][row]:
            lead = round((shot["down"] - shot["times"][end]) * 1000, 9)
            assert 10 < lead <= max(25, data.advanced_times()[sid])
        else:
            assert not shot["is_disrupt"]
    assert bank["protocol"]["source_role"] == "TRAIN"
    assert bank["protocol"]["target_labels_accessed"] == 0
    with torch.no_grad():
        torch.testing.assert_close(bank["teacher_logits"], base(bank["x"], bank["available"])["logit"])


def test_insufficient_shots_or_overlap_or_fabricated_labels_raise():
    with pytest.raises(ValueError, match="insufficient"):
        replay.build_source_bank(teacher(), source_parent_sha256="a" * 64,
                                 data_api=TrainData(1), per_class=2)
    data = TrainData()
    data.source_roles = lambda: dict(train=data.ids, val=[1], test=[9002])
    with pytest.raises(ValueError, match="overlap"):
        replay.build_source_bank(teacher(), source_parent_sha256="a" * 64, data_api=data, per_class=2)
    data = TrainData()
    attach = data.attach_outcome
    def bad_labels(*args):
        shot = attach(*args)
        shot["y"] = 1 - shot["y"]
        return shot
    data.attach_outcome = bad_labels
    with pytest.raises(ValueError, match="label"):
        replay.build_source_bank(teacher(), source_parent_sha256="a" * 64, data_api=data, per_class=2)


def test_strict_lead_boundary_and_no_future_suffix_dependence():
    data = TrainData(2)
    read = data.read_shot
    def changed(device, sid):
        s = read(device, sid)
        if sid % 2:
            s["values"][230:] = 1e8  # exactly 10 ms or later: never in a positive window
        return s
    data.read_shot = changed
    bank = replay.build_source_bank(teacher(), source_parent_sha256="a" * 64, data_api=data, per_class=2)
    assert (bank["endindices"][bank["labels"] == 1] < 230).all()
    assert (bank["x"][bank["labels"] == 1].abs() < 1e8).all()


def test_sampling_is_rng_balanced_distinct_and_accessors_do_not_alias():
    _, _, bank = bank_case(64)
    indices = replay.sample_source(bank, np.random.default_rng(3))
    assert len(indices) == len(set(indices.tolist())) == 64
    assert bank["labels"][indices].sum().item() == 32
    assert torch.equal(indices, replay.sample_source(bank, np.random.default_rng(3)))
    batch = replay.source_batch(bank, indices, "cpu")
    batch["x"].add_(99)
    replay.validate_bank(bank)
    with pytest.raises(ValueError):
        replay.sample_source(bank, np.random.default_rng(3), batch_size=3)
    with pytest.raises(ValueError, match="insufficient"):
        replay.sample_source(bank_case(2)[2], np.random.default_rng(3))


@pytest.mark.parametrize("wrapper", [ContextModel, EmbeddingModel])
def test_loss_only_persistent_gradients_no_updates_and_same_batch_reusable(wrapper):
    base, _, bank = bank_case()
    model = wrapper(base).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    indices = replay.sample_source(bank, np.random.default_rng(3), batch_size=8)
    loss, audit = replay.source_loss(model, bank, "cpu", np.random.default_rng(5),
                                     batch_size=8, indices=indices)
    batch = replay.source_batch(bank, indices, "cpu")
    logits = model(batch["x"], batch["available"], model.initial_fast(8))["logit"]
    torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(logits, batch["labels"]))
    assert audit["indices"] == indices.tolist()
    assert audit["positive_count"] == audit["safe_count"] == 4
    assert all(p.grad is None for p in model.parameters())
    loss.backward()
    assert all(p.grad is None for p in base.parameters())
    gradients = [p.grad for p in model.persistent_parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    replay.validate_bank(bank)


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
        self.fast_seen = None

    def initial_fast(self, batch):
        return torch.zeros(batch, 128)

    def forward(self, x, available, fast):
        self.fast_seen = fast
        return {"logit": self.logits}


def test_guard_original_correct_points_only_boundary_no_confidence_margin():
    _, _, bank = bank_case(2)
    # Builder orders positives then safes: TP just above threshold, FN, TN, FP.
    baseline = torch.logit(torch.tensor([.501, .1, .499, .9]))
    bank["teacher_logits"] = baseline
    bank["bank_sha256"] = replay.bank_sha256(bank)
    model = FixedLogits(torch.logit(torch.tensor([.5, .99, .499, .01])))
    guard = replay.evaluate_source_guard(model, bank, "cpu", threshold=.5)
    assert guard["passed"] and guard["protected_tp"] == guard["protected_tn"] == 1
    assert not model.fast_seen.requires_grad and model.fast_seen.count_nonzero() == 0
    model.logits = torch.logit(torch.tensor([.499, .99, .5, .01]))
    guard = replay.evaluate_source_guard(model, bank, "cpu", threshold=.5)
    assert not guard["passed"] and guard["lost_tp"] == guard["lost_tn"] == 1
    assert guard["violating_indices"] == [0, 2]
    with pytest.raises(ValueError, match="original"):
        replay.evaluate_source_guard(model, bank, "cpu", threshold=.5, baselineoriginal_teacher=baseline + 1)
    with pytest.raises(ValueError):
        replay.evaluate_source_guard(model, bank, "cpu", threshold=float("nan"))
    model.logits[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replay.evaluate_source_guard(model, bank, "cpu", threshold=.5)


@pytest.mark.parametrize("field", ["x", "labels", "teacher_logits", "horizon_ms", "lead_ms"])
def test_nonfinite_bank_rejected(field):
    _, _, bank = bank_case(2)
    bank[field].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        replay.validate_bank(bank)


def test_artifact_content_mutation_and_unfrozen_teacher_rejected():
    _, _, bank = bank_case(2)
    bank["x"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="hash"):
        replay.validate_bank(bank)
    base = teacher().requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        replay.build_source_bank(base, source_parent_sha256="a" * 64, data_api=TrainData(2), per_class=2)


def test_guard_all_points_no_correct_teacher_fails_closed_and_subset_reuses_indices():
    base, _, bank = bank_case(2)
    model = EmbeddingModel(base).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    guard = replay.evaluate_source_guard(model, bank, "cpu", .5)
    assert guard["passed"] and guard["checked_points"] == 4
    indices = torch.tensor([0, 2])
    assert replay.evaluate_source_guard(model, bank, "cpu", .5, indices=indices)["checked_points"] == 2
    assert all(p.grad is None for p in model.parameters())
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    bank["teacher_logits"] = torch.where(bank["labels"] == 1, -torch.ones(4), torch.ones(4))
    bank["bank_sha256"] = replay.bank_sha256(bank)
    guard = replay.evaluate_source_guard(FixedLogits(torch.zeros(4)), bank, "cpu", .5)
    assert not guard["passed"] and guard["reason"] == "no_original_correct_points"


def test_fixed_safe_index_provenance_and_invalid_selected_batches():
    bank = replay.build_source_bank(teacher(), source_parent_sha256="a" * 64,
                                    data_api=TrainData(2), per_class=2, safe_end_index=150)
    assert (bank["endindices"][bank["labels"] == 0] == 150).all()
    for bad in (torch.tensor([0, 0]), torch.tensor([-1]), torch.tensor([4]), torch.tensor([0.])):
        with pytest.raises(ValueError):
            replay.source_batch(bank, bad)
    with pytest.raises(ValueError, match="balanced"):
        replay.source_loss(ContextModel(teacher()), bank, "cpu", np.random.default_rng(1),
                           batch_size=2, indices=torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="bounded"):
        replay.build_source_bank(teacher(), source_parent_sha256="a" * 64,
                                 data_api=TrainData(2), per_class=65)


@pytest.mark.parametrize("change", ["device", "nonfinite", "horizon"])
def test_bad_source_data_fail_closed(change):
    data = TrainData(2)
    attach = data.attach_outcome
    def bad(*args):
        shot = attach(*args)
        if change == "device":
            shot["device"] = "EAST"
        elif change == "nonfinite":
            shot["values"][0, 0] = np.nan
        else:
            shot["horizon"] = 125.
        return shot
    data.attach_outcome = bad
    with pytest.raises(ValueError):
        replay.build_source_bank(teacher(), source_parent_sha256="a" * 64, data_api=data, per_class=2)


def test_cli_explicit_new_output_cpu_only_roundtrip_and_parent_hashlock(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        replay.main([])
    bad_parent = tmp_path / "parent.pt"
    bad_parent.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        replay.load_teacher(bad_parent)
    base, _, bank = bank_case(2)
    monkeypatch.setattr(replay, "load_teacher", lambda path: base)
    monkeypatch.setattr(replay, "build_source_bank", lambda *args, **kw: copy.deepcopy(bank))
    monkeypatch.setattr(replay, "sha256", lambda path: replay.PARENT_SHA256)
    out = tmp_path / "bank.pt"
    assert replay.main(["--output", str(out), "--per-class", "2"]) == 0
    restored = replay.load_source_bank(out, expected_parent_sha256="a" * 64)
    assert restored["bank_sha256"] == bank["bank_sha256"]
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        replay.main(["--output", str(out)])
    assert hashlib.sha256(out.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="parent"):
        replay.load_source_bank(out)
    assert not torch.cuda.is_initialized()
