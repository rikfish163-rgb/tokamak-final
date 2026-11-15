"""Source-only offline task records; no signal, outcome, model or GPU readers.

``load_records(role)`` returns (copied rows with role/device, canonical role IDs,
provenance). Input hashes are archived evidence, NOT freshly rehashed raw data.
``load_events()`` returns (grouped, metadata) for TRAIN/VAL only. Families 0..5
are nominal categories, never ordered stages; -1 is unknown, never normal.
"""

from collections import Counter
import hashlib
import json
from numbers import Integral

import numpy as np

from .. import io23

ARCHIVE = io23.ROOT / "probe_outputs/context_vector_source_v1"
SOURCE_SHA = "c7270c1ea3652db2e30e972d4ca9a4a2d33220a68e4a3765c8b7f5bcceeaeb02"
EVIDENCE_SHA = {
    "complete.json": "4911c2ceac1ab60a3778b6f38c599e2281cfd7b84acb1dbc9f7e43e3bff800aa",
    "audit_cpu_v1.json": "1e6c13f19b6b87c34464aacae2518587b6b59cd94b4f9132022a1bfe47b58d02",
}
GROUPS = ("safe", "danger", "early")
TASKS = {"native": (), "no_MNM": ("MNM",), "no_MNM_CIII": ("MNM", "CIII")}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _read(name):
    content = (ARCHIVE / name).read_bytes()
    return json.loads(content), hashlib.sha256(content).hexdigest()


def _check_record(r, role, ids, advanced):
    _require(isinstance(r, dict), "record must be a mapping")
    _require(all(_integer(r.get(k)) for k in ("shot", "origin", "maturity", "query")),
             "integer shot and clock indices required")
    _require(r["shot"] in ids and r.get("role", role) == role
             and r.get("device", "J-TEXT") == "J-TEXT", "record outside source role")
    o, q = r["origin"], r["query"]
    _require(o >= 127 and (o - 127) % 25 == 0 and r["maturity"] == o + 25
             and o + 26 <= q <= o + 50, "invalid origin/maturity/query clock")
    g, d, lead = r.get("group"), r.get("disruptive"), r.get("lead_ms")
    _require(g in GROUPS and isinstance(d, bool) and r.get("feedback_observed") is True,
             "invalid group, disruption or feedback metadata")
    _require(r.get("label") == float(g == "danger") and d == (g != "safe"),
             "group/label/disruption mismatch; early is diagnostic only")
    if d:
        _require(isinstance(lead, (int, float)) and not isinstance(lead, bool)
                 and np.isfinite(lead) and lead > 10., "query requires finite lead >10ms")
        horizon = io23.effective_horizon_ms(r["shot"], advanced)[0]
        _require((g == "danger") == (lead <= horizon + 1e-6),
                 "label disagrees with max(25ms, AdvancedTime)")
    else:
        _require(lead is None, "safe record cannot have a disruption lead")


def load_records(role):
    """Bind the fixed completed archive and current roles without opening HDF.

    Provenance includes artifact_sha256, input_hashes, source_sha256, role and
    raw_inputs_rehashed=False. Returned IDs include role shots with no episodes.
    """
    _require(role in ("train", "val"), "only J-TEXT train/val roles are allowed")
    evidence, digests = {}, {}
    for name, expected in EVIDENCE_SHA.items():
        evidence[name], digests[name] = _read(name)
        _require(digests[name] == expected, f"archive hash mismatch: {name}")
    complete, audit = evidence["complete.json"], evidence["audit_cpu_v1.json"]
    _require(complete.get("complete") is True and complete.get("source_unchanged") is True
             and audit.get("verified") is True, "incomplete or unaudited archive")
    names = ("protocol.json", f"{role}_records.json", f"{role}_input_hashes.json")
    for name in names:
        evidence[name], digests[name] = _read(name)
        _require(digests[name] == complete["artifact_sha256"].get(name),
                 f"archive hash mismatch: {name}")
    protocol = evidence["protocol.json"]
    expected = dict(scope="JTEXT_TRAIN_CONTEXT_VECTOR_TEACHING_NOT_JTEXT_ONLINE",
        source_sha256=SOURCE_SHA, label="max(25ms, AdvancedTime)", query_lead_strict_ms=10.,
        first_origin=127, maturity_offset=25, query_offset=[26, 50], max_per_group_per_shot=8)
    _require(all(protocol.get(k) == v for k, v in expected.items()), "source protocol changed")
    _require(all(v.get(k) == 0 for v in (complete, protocol)
                 for k in ("test_signals_read", "east_signals_read")), "forbidden role exposure")
    roles = io23.source_roles()
    _require(all(protocol.get(f"{r}_ids") == roles[r] for r in ("train", "val"))
             and not any(set(roles[a]) & set(roles[b]) for a, b in
                         (("train", "val"), ("train", "test"), ("val", "test"))),
             "current source roles disagree or overlap")
    ids, advanced = list(roles[role]), io23.advanced_times()
    hashes = evidence[f"{role}_input_hashes.json"]
    _require(set(hashes) == {str(sid) for sid in ids} and all(
        isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)
        for h in hashes.values()), "source input hash inventory mismatch")
    rows = evidence[f"{role}_records.json"]
    _require(isinstance(rows, list) and bool(rows), "nonempty records required")
    counts, seen = Counter(), set()
    for r in rows:
        _check_record(r, role, set(ids), advanced)
        key = r["shot"], r["origin"]
        _require(key not in seen, "duplicate shot/origin")
        seen.add(key)
        counts[r["shot"], r["group"]] += 1
    _require(max(counts.values()) <= 8, "per-shot/group cap exceeded")
    _require(audit.get("raw_inputs") == len(roles["train"]) + len(roles["val"])
             and (role != "val" or audit.get("validation_windows") == len(rows)),
             "audit population mismatch")
    return [dict(r, role=role, device="J-TEXT") for r in rows], ids, dict(
        archive=str(ARCHIVE), role=role, source_sha256=SOURCE_SHA, artifact_sha256=digests,
        input_hashes=hashes, raw_inputs_rehashed=False)


def load_events():
    """Hash-check the existing absolute-clock CSV parser; return TRAIN/VAL only."""
    from ..event_prefix_probe import DEFAULT_LABELS, LABEL_SHA256, read_events

    content = DEFAULT_LABELS.read_bytes()
    _require(hashlib.sha256(content).hexdigest() == LABEL_SHA256, "event CSV hash mismatch")
    grouped, metadata = read_events(DEFAULT_LABELS)
    roles = io23.source_roles()
    allowed = set(roles["train"]) | set(roles["val"])
    return ({sid: events for sid, events in grouped.items() if sid in allowed},
            {sid: meta for sid, meta in metadata.items() if sid in allowed})


def attach_event(rows, times_by_shot, grouped):
    """Copy rows and attach origin/query families on supplied absolute clocks.

    No I/O, interpolation, event-end-as-normal or A-F ordinal interpretation.
    """
    from ..event_prefix_probe import assign_events

    assigned = {}
    for r in rows:
        sid = r["shot"]
        if sid not in assigned:
            _require(sid in times_by_shot, "missing source shot clock")
            events = grouped.get(sid, [])
            _require(all(e.shot == sid for e in events), "mixed-shot event intervals")
            assigned[sid] = assign_events(times_by_shot[sid], events)[0]
        families = assigned[sid]
        _require(all(_integer(r[k]) and 0 <= r[k] < len(families)
                     for k in ("origin", "query")), "event lookup outside shot clock")
    return [dict(r, **{f"{key}_family": int(assigned[r["shot"]][r[key]])
                      for key in ("origin", "query")}) for r in rows]


def mask_task(signal, task):
    """Copy values/available only; mask channels without changing any values."""
    _require(task in TASKS and set(signal) == {"values", "available"},
             "unknown task or non-signal fields")
    values, available = np.asarray(signal["values"]), np.asarray(signal["available"])
    _require(values.ndim == 2 and values.shape[1] == 23 and values.dtype.kind in "iuf"
             and available.shape == values.shape and available.dtype.kind == "b",
             "numeric Nx23 values and boolean availability required")
    result = dict(values=values.copy(), available=available.copy())
    for channel in TASKS[task]:
        result["available"][:, io23.RAW23.index(channel)] = False
    return result


def sample_indices(records, rng, per_class=1):
    """Safe then danger: equal family (including -1), shot, record; replacement.

    Call ONCE per step and share the returned list across all three models.
    Early records never enter training and an empty class is an error.
    """
    _require(_integer(per_class) and per_class > 0, "positive per_class required")
    buckets, shot_groups = {g: {} for g in GROUPS[:2]}, {}
    for i, r in enumerate(records):
        g, sid, family = r.get("group"), r.get("shot"), r.get("origin_family", -1)
        _require(g in GROUPS and _integer(sid) and _integer(family) and -1 <= family <= 5
                 and r.get("role", "train") == "train"
                 and r.get("device", "J-TEXT") == "J-TEXT", "invalid training record")
        shot_groups.setdefault(sid, set()).add(g)
        _require(not ("safe" in shot_groups[sid] and len(shot_groups[sid]) > 1),
                 "shot cannot be both safe and disruptive")
        if g != "early":
            buckets[g].setdefault(int(family), {}).setdefault(int(sid), []).append(i)
    _require(all(buckets.values()), "both safe and danger groups required")
    result = []
    for g in GROUPS[:2]:
        for _ in range(per_class):
            family = rng.choice(sorted(buckets[g]))
            sid = rng.choice(sorted(buckets[g][family]))
            result.append(int(rng.choice(buckets[g][family][sid])))
    return result


def summarize_records(rows):
    """Role/group/disruption census and family x group DISTINCT-shot counts."""
    def count(selected):
        return dict(records=len(selected), shots=len({r["shot"] for r in selected}))

    roles = {}
    for role in sorted({r.get("role", "unspecified") for r in rows}):
        local = [r for r in rows if r.get("role", "unspecified") == role]
        item = dict(**count(local), groups={g: count([r for r in local if r["group"] == g])
                                          for g in GROUPS})
        item["disruptive"] = {str(d).lower(): count([r for r in local if r["disruptive"] == d])
                              for d in (False, True)}
        for key in ("origin_family", "query_family"):
            item[key] = {str(f): {g: count([r for r in local
                if r.get(key, -1) == f and r["group"] == g]) for g in GROUPS}
                for f in range(-1, 6)}
        roles[role] = item
    return dict(**count(rows), roles=roles)
