"""Two real prefix-only GPU steps; no future-shot metric or parameter selection."""

from .. import io23, simple_data
from ..causal_input import apply_input_contract
from ..east_context_operator_20261031.run import configure, read
from ..redesign_20261004.prepare_prior import DEFAULT_PARENT, load_parent
from ..simple_online import _durable_json
from ..simple_train import sha
from ..slow_fast_attribution_20261031.run import REFERENCE, REFERENCE_SHA
from .prepare import prepare


def main():
    root = io23.ROOT / "probe_outputs/operational_gain_preflight_v1"
    if root.exists():
        raise FileExistsError(root)
    configure()
    assert sha(REFERENCE / "completed.json") == REFERENCE_SHA
    mapping = read(REFERENCE / "input_map.json")
    target = []
    seals = {}
    ids = [r["shot"] for r in read(REFERENCE / "order.json")]
    assert mapping["calibration_ids"] == ids[:20]
    for i, sid in enumerate(ids[:20]):
        path = REFERENCE / "shots" / f"{i + 1:04d}_{sid}.npz"
        seals[str(sid)] = sha(path)
        assert (
            seals[str(sid)]
            == read(REFERENCE / "shots" / f"{i + 1:04d}_{sid}_seal.json")["sha256"]
        )
        raw = apply_input_contract(
            simple_data.read_shot("EAST", sid, rate_contract="nominal")
        )
        target.append(simple_data.attach_outcome(raw, "EAST", {}))
    base, parent = load_parent(DEFAULT_PARENT)
    base.cuda()
    root.mkdir()
    _durable_json(
        root / "protocol.json",
        dict(
            exact_ended_prefix=ids[:20],
            prior_prediction_seals=seals,
            steps=2,
            purpose="numeric gradient and timing check; no suffix read; results not model selection",
            source_sha256=sha(DEFAULT_PARENT),
        ),
    )
    prepare(
        base, target, mapping, float(parent["threshold"]), root / "preparation", steps=2
    )


if __name__ == "__main__":
    main()
