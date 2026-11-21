"""Re-execute the frozen local-memory trajectory, intervening only on episode origin."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import layer0_local_memory_20261105 as memory
import layer_interaction_replay_20261105 as layer
import moving_peak_replay_20261105 as r
from run_layer0_query_memory_20261105 import as_case, parents, predict, raw_for


PLAN = r.ROOT / "LAYER0_MEMORY_POLARITY_PLAN_20261105.md"
LOCAL = r.ROOT / "probe_outputs/layer0_query_memory360_20261105_v1"
ARMS = (
    "static",
    "intra",
    "inter",
    "both",
    "inter_safe",
    "both_safe",
    "inter_disruptive",
    "both_disruptive",
)


def isolate_origin(episodes, flags, disruptive):
    if (
        len(episodes) != len(flags)
        or type(disruptive) is not bool
        or any(type(f) is not bool for f in flags)
    ):
        raise ValueError("exactly one revealed origin per memory required")
    return tuple(
        episode
        if flag == disruptive
        else memory.Episode(episode.keys, torch.zeros_like(episode.delta))
        for flag, episode in zip(flags, episodes, strict=True)
    )


def run(output):
    start = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    r.configure()
    order, ref, _, base, source, mapping, map_path, fit_path = parents()
    old = r.read(LOCAL / "completed.json")
    old_cfg = r.read(LOCAL / "protocol.json")
    raw_hashes = ref["raw_sha256"]
    if (
        old["status"] != "COMPLETE_LAYER0_QUERY_MEMORY"
        or old["n_reveals"] != 36
        or "memory_variant" in old_cfg
        or old_cfg["order"] != order
        or old_cfg["raw_sha256"] != raw_hashes
        or old_cfg["threshold"] != r.THRESHOLD
        or r.sha(LOCAL / "protocol.json") != old["protocol_sha256"]
    ):
        raise ValueError("unchanged causal local-memory parent required")
    paths = [
        Path(__file__),
        PLAN,
        LOCAL / "completed.json",
        LOCAL / "protocol.json",
        Path(as_case.__code__.co_filename),
        Path(predict.__code__.co_filename),
        Path(memory.__file__),
        Path(r.__file__),
        Path(r.policy.__file__),
        Path(r.latent.__file__),
        map_path,
        fit_path,
        r.DEFAULT_PARENT,
    ]
    hashes = {str(p.resolve()): r.sha(p) for p in paths}
    output.mkdir(parents=True)
    (output / "shots").mkdir()
    r.write_new(
        output / "protocol.json",
        dict(
            order=order,
            arms=ARMS,
            raw_sha256=raw_hashes,
            threshold=r.THRESHOLD,
            source_state_sha256=source,
            label_budget=36,
            additional_label_reads=0,
            training_steps=0,
            development_only=True,
            input_sha256=hashes,
            local_producer_sha256=r.sha(LOCAL / "completed.json"),
        ),
    )
    rule = bandwidth = state = previous_seal = None
    episodes, flags, sealed = (), [], []
    errors = dict.fromkeys(ARMS[:4], 0.0)
    with patch.object(
        r.io23,
        "read_outcome",
        side_effect=RuntimeError("no new outcomes are available in this intervention"),
    ):
        for rank, sid in enumerate(order, 1):
            raw = raw_for(sid, raw_hashes)
            case = as_case(base, raw, rank, mapping, rule)
            p, _ = predict(base, case, episodes, bandwidth, all_arms=True)
            for flag, name in ((False, "safe"), (True, "disruptive")):
                if episodes:
                    sub, _ = predict(
                        base,
                        case,
                        isolate_origin(episodes, flags, flag),
                        bandwidth,
                        all_arms=False,
                    )
                else:
                    sub = {k: p[k] for k in ("inter", "both")}
                p.update({f"{k}_{name}": v for k, v in sub.items()})
            stem = f"{rank:04d}_{sid}"
            archive = LOCAL / "shots" / (stem + ".npz")
            bound = old["sealed"][rank - 1]
            if r.sha(archive) != bound["curve_sha256"] or (
                bound["rank"],
                bound["shot"],
                bound["state_sha256"],
                bound["state_size"],
            ) != (rank, sid, state, len(episodes)):
                raise ValueError(f"wrong archived state at prediction: {sid}")
            with np.load(archive, allow_pickle=False) as ref_curve:
                if not np.array_equal(ref_curve["times"], raw["times"]):
                    raise ValueError("signal clock changed")
                for k in ARMS[:4]:
                    error = layer.require_curve_parity(p[k], ref_curve[k], r.THRESHOLD)
                    errors[k] = max(errors[k], error)
            array = output / "shots" / (stem + ".npz")
            seal = output / "shots" / (stem + "_seal.json")
            r.save_arrays(array, times=raw["times"], **p)
            row = dict(
                rank=rank,
                shot=sid,
                curve_sha256=r.sha(array),
                raw_sha256=raw_hashes[str(sid)],
                state_sha256=state,
                n_safe=sum(not f for f in flags),
                n_disruptive=sum(flags),
                seal_ns=time.monotonic_ns(),
                previous_seal_sha256=previous_seal,
            )
            r.write_new(seal, row)
            previous_seal = r.sha(seal)
            sealed.append(dict(row, seal_sha256=previous_seal))
            if rank % 20:
                continue
            block = rank // 20
            receipt = LOCAL / "blocks" / f"{block:02d}_update.json"
            revealed = LOCAL / "blocks" / f"{block:02d}_reveals.json"
            prior = old["blocks"][block - 1]
            if (
                r.sha(receipt) != prior["update_sha256"]
                or r.sha(revealed) != prior["reveals_sha256"]
            ):
                raise ValueError("changed prior feedback receipt")
            update = r.read(receipt)
            if update["state_before_sha256"] != state:
                raise ValueError("noncausal parent memory transition")
            if block == 1:
                rule = r.load_rule(fit_path, base.center.device)
                bandwidth = update["bandwidth"]
            elif update["decision"] == "accepted":
                one = r.read(revealed)["outcomes"]
                if len(one) != 1 or type(one[0]["disruptive"]) is not bool:
                    raise ValueError("invalid released origin")
                flags.append(one[0]["disruptive"])
            if update["state_file"]:
                path = LOCAL / "blocks" / update["state_file"]
                state = r.sha(path)
                with np.load(path, allow_pickle=False) as m:
                    if m["size"].item() != len(flags):
                        raise ValueError(
                            "state does not match all and only past accepted origins"
                        )
                    episodes = tuple(
                        memory.Episode(
                            torch.as_tensor(m[f"keys_{i}"], device=base.center.device),
                            torch.as_tensor(m[f"delta_{i}"], device=base.center.device),
                        )
                        for i in range(len(flags))
                    )
            if update["state_after_sha256"] != state:
                raise ValueError("unbound memory state")
            print(
                f"sealed {rank}/360 origin intervention, {len(flags)} committed origins",
                flush=True,
            )
    if (
        len(flags) != old["n_accepted"]
        or r.state_digest(base) != source
        or any(r.sha(p) != h for p, h in hashes.items())
    ):
        raise ValueError("frozen parent or origin ledger changed")
    r.write_new(
        output / "completed.json",
        dict(
            status="COMPLETE_LAYER0_MEMORY_POLARITY",
            n_shots=len(order),
            n_curves=len(order) * len(ARMS),
            sealed=sealed,
            max_ref_error=errors,
            protocol_sha256=r.sha(output / "protocol.json"),
            seconds=time.monotonic() - start,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
