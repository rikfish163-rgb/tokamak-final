"""Re-score the sealed EAST 360 comparisons and list exactly which shots false-alarmed."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBES = ROOT / "probe_outputs"
DELAY = PROBES / "commissioning_write_delay_east360_20261112_v2"
STEPS = PROBES / "commissioning_parent_readout_east2953_20261113_v1"
BETWEEN = PROBES / "commissioning_query_strata_east360_20261113_v2"
ROWS = (
    (DELAY, "fixed", "静态基线"),
    (STEPS, "inherited", "记住本炮预测偏差"),
    (STEPS, "legacy_current4", "再看这次误差比平常多多少"),
    (STEPS, "signed_path_true", "再改输入信号的走势"),
    (DELAY, "bias_only", "保留前四步炮内修正"),
    (DELAY, "current_5", "再加第 5 步：用最新误差修正，等 5 点"),
    (DELAY, "earlier_5", "改用上一笔误差，等 5 点"),
    (DELAY, "current_0", "用最新误差，不额外等 5 点"),
    (BETWEEN, "bias_only_recent", "前四步加炮间修正"),
    (BETWEEN, "current_5_recent", "前五步加炮间修正"),
    (BETWEEN, "current_5_recent_supported", "限制炮间修正的使用范围"),
    (BETWEEN, "current_5_matched_supported", "在同样范围内换同组未破裂样本"),
    (BETWEEN, "current_5_high_supported", "在同样范围内换高磁场未破裂样本"),
)
COMPARE = (None, 0, 1, 2, 3, 4, 5, 5, 4, 5, 9, 10, 10)
CATEGORIES = {"cold": "前20炮准备段",
              "before_write": "后340炮：第5步首次改动前已误报",
              "after_write": "第5步已改动；前四步版本也误报",
              "zero_after_write": "第5步已运行；首报点没有改动",
              "outside_row5": "第5行未误报，其他某行误报"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def truth(shot):
    f = ROOT / "data/EAST/unified_hdf5" / str(shot//100*100) / f"{shot}.hdf5"
    with h5py.File(f, "r") as hdf:
        disrupted, = np.asarray(hdf["meta/IsDisrupt"][()]).reshape(-1)
        if disrupted not in (0, 1):
            raise ValueError(f"nonbinary shot outcome: {shot}")
        down, = np.asarray(hdf["meta/DownTime"][()]).reshape(-1) if disrupted else (None,)
        ip = hdf["data/ip"]
        start, rate = (float(np.asarray(ip.attrs[k]).reshape(-1)[0]) for k in ("StartTime", "SampleRate"))
        return bool(disrupted), None if down is None else float(down), start+np.arange(len(ip))/rate


def packs(name, seals):
    result = {}
    for path, seal in seals.items():
        f = path / "shots" / name
        if hashlib.sha256(f.read_bytes()).hexdigest() != seal[name]:
            raise ValueError(f"changed sealed curve: {f}")
        with np.load(f, allow_pickle=False) as archive:
            result[path] = dict(archive)
    return result


def occurrences(hit):
    longest = max(map(len, np.split(hit, np.flatnonzero(np.diff(hit) > 1)+1))) if len(hit) else 0
    return len(hit), longest


def main():
    p = read(DELAY / "protocol.json")
    order = p["order"]
    between, prev = (read(path / "protocol.json") for path in (BETWEEN, STEPS))
    protocols = (p, between, prev)
    for key in ("startup", "calibration", "original"):
        if any(d[key] != p[key] for d in protocols):
            raise ValueError(f"inconsistent operating rule: {key}")
    if order != between["order"] or order != [r["shot"] for r in prev["order"] if r["cohort"] == "fixed360"]:
        raise ValueError("methods do not share the fixed 360 replay roster")
    seals = {path: read(path / "seal.json")[k]
             for path, k in ((DELAY, "curve_sha256"), (STEPS, "curve_sha256"), (BETWEEN, "shots"))}
    stored = {path: read(path / "per_shot.json")["0.2"] for path in seals}
    reported_support = {r["shot"]: r for r in read(DELAY / "alarm_support.json")["records"]["0.2"]["5"]}
    counts = [dict(row=i, method=method, timely=0, late=0, missed=0,
                   false_alarms=0, correct_no_alarm=0, false_alarm_shots=[])
              for i, (_, _, method) in enumerate(ROWS)]
    cases = []
    for rank, sid in enumerate(order, start=1):
        files = packs(f"{rank:04d}_{sid}.npz", seals)
        outcome, down, t = truth(sid)
        np.testing.assert_array_equal(files[BETWEEN]["times"], t)
        threshold = p["original"] if rank <= p["startup"] else p["calibration"]["0.2"]
        td = down-t if outcome else None
        mask = (td*1000 >= -1e-6) if outcome else np.ones(len(t), bool)
        used = files[DELAY]["dose"][-1] > 0
        positions = np.flatnonzero(used)
        curves, hits = [], []
        for row, (path, k, _) in enumerate(ROWS):
            a = files[path][k]
            ix = np.flatnonzero(mask & (a >= threshold))
            lead = round(float(td[ix[0]]*1000), 9) if outcome and len(ix) else None
            timely = bool(lead is not None and lead > 30)
            if dict(shot=sid, disruptive=outcome, down_s=down, alarm=bool(len(ix)),
                    lead_ms=lead, timely=timely) != stored[path][k][rank-1]:
                raise ValueError(f"sealed result differs: {rank}, row {row}")
            if outcome:
                key = "timely" if timely else "late" if len(ix) else "missed"
            else:
                key = "false_alarms" if len(ix) else "correct_no_alarm"
                if len(ix):
                    counts[row]["false_alarm_shots"].append(sid)
            counts[row][key] += 1
            curves.append(a)
            hits.append(ix)
        if outcome or not any(map(len, hits)):
            continue
        ix = hits[5]
        case = dict(shot=sid, rank=rank, threshold=threshold, category="outside_row5",
                    first_step5_write_s=float(t[positions[0]]) if len(positions) else None)
        if len(ix):
            first = ix[0]
            zero = ix[~used[ix]]
            category = ("cold" if rank <= p["startup"] else "after_write" if used[first]
                        else "before_write" if not len(positions) or first < positions[0] else "zero_after_write")
            old_cat = {"cold": "cold", "after_write": "exposed_only", "before_write": "before_first_write",
                       "zero_after_write": "unchanged_later"}[category]
            rec = reported_support[sid]
            if rec["category"] != old_cat or rec["first_index"] != first or bool(len(zero)) == bool(used[first]):
                raise ValueError(f"sealed write timing differs: {sid}")
            count, longest = occurrences(ix)
            case.update(category=category, row5_score_at_first=float(curves[5][first]),
                        row4_score_at_row5_first=float(curves[4][first]),
                        row5_points_at_or_above=count, row5_longest_run=longest,
                        row5_points_also_above_in_row4=int(np.count_nonzero(curves[4][ix] >= threshold)))
        case["alarm_changes_0_to_5"] = " ".join(
            f"{'+' if len(hits[i]) else '-'}{i}" for i in range(1, 6) if bool(len(hits[i])) != bool(len(hits[i-1]))
        )
        for row, (a, hit) in enumerate(zip(curves, hits, strict=True)):
            case.update({f"row{row}_first_s": float(t[hit[0]]) if len(hit) else None,
                         f"row{row}_peak_score": float(a.max())})
        # Keep a small first-alarm window for cases in the reader's guide.
        if sid in (67900, 73790, 72984, 73743):
            plot_row = 7 if sid == 73743 else 5
            pivot = hits[plot_row][0]
            w = slice(max(0, pivot-22), min(len(t), pivot+24))
            with (HERE / f"_case_{sid}.npz").open("wb") as stream:
                np.savez_compressed(stream, seconds=t[w], center=t[pivot], threshold=threshold,
                                    **{f"row{i}": curves[i][w] for i in (0, 4, 5, 7)})
        cases.append(case)
    assert counts[0]["timely"]+counts[0]["late"]+counts[0]["missed"] == 180
    assert counts[0]["false_alarms"]+counts[0]["correct_no_alarm"] == 180
    for m, reference in zip(counts, COMPARE, strict=True):
        assert m["timely"]+m["late"]+m["missed"] == 180
        assert m["false_alarms"]+m["correct_no_alarm"] == 180
        m["reference"] = reference
        m.update(timely_correct=m["timely"]+m["correct_no_alarm"],
                 binary_correct=m["timely"]+m["late"]+m["correct_no_alarm"])
        if reference is not None:
            a, b = set(counts[reference]["false_alarm_shots"]), set(m["false_alarm_shots"])
            m.update(false_alarms_added=sorted(b-a), false_alarms_removed=sorted(a-b))
    for category, expected in (("cold", 6), ("before_write", 7), ("after_write", 21)):
        assert sum(r["category"] == category for r in cases) == expected
    assert counts[5]["false_alarm_shots"] == counts[4]["false_alarm_shots"]
    assert {v["shot"] for v in cases if v["category"] != "outside_row5"} == set(reported_support)
    fields = (
        ("shot", "炮号"), ("rank", "回放序号"), ("category", "情况"), ("threshold", "报警线"),
        ("alarm_changes_0_to_5", "相对上一行的误报变化（+新增，-去掉，数字是表格行号）"),
        ("first_step5_write_s", "第5步第一次实际改动（秒）"),
        ("row5_score_at_first", "第5行首报分数"), ("row4_score_at_row5_first", "同一时刻第4行分数"),
        ("row5_points_at_or_above", "第5行总越线采样点数"), ("row5_longest_run", "第5行最长连续越线点数"),
        ("row5_points_also_above_in_row4", "第5行越线的点中第4行也越线的点数"),
        *((f"row{i}_{key}", f"第{i}行{label}") for i in range(len(ROWS))
          for key, label in (("first_s", "首次误报（秒）"), ("peak_score", "最高分"))),
    )
    with (HERE / "EAST360_误报逐炮.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in fields])
        for r in cases:
            writer.writerow([CATEGORIES[r[k]] if k == "category" else
                             f"{r[k]:.7g}" if isinstance(r.get(k), float) else r.get(k) for k, _ in fields])
    with (HERE / "EAST360_完整计分.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("表格行号", "方法", "用于比较的行号", "及时报警数", "及时报警率", "晚报警数", "漏报警数",
                         "误报数", "误报率", "未破裂且不报警数", "按时才算对的炮数", "按时才算对的准确率",
                         "只看报没报的正确炮数", "只看报没报的准确率", "新添误报炮号", "去掉误报炮号"))
        for m in counts:
            writer.writerow((m["row"], m["method"], m["reference"], m["timely"], f'{m["timely"]/180:.2%}',
                             m["late"], m["missed"], m["false_alarms"], f'{m["false_alarms"]/180:.2%}',
                             m["correct_no_alarm"], m["timely_correct"], f'{m["timely_correct"]/360:.2%}',
                             m["binary_correct"], f'{m["binary_correct"]/360:.2%}',
                             " ".join(map(str, m.get("false_alarms_added", []))),
                             " ".join(map(str, m.get("false_alarms_removed", [])))))
    data = dict(n=360, n_disruptive=180, n_safe=180, rows=counts, false_alarm_cases=cases)
    (HERE / "EAST360_评分核对.json").write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for m in counts:
        print(m["row"], *(m[k] for k in ("timely", "late", "missed", "false_alarms", "timely_correct", "binary_correct")),
              "+", m.get("false_alarms_added"), "-", m.get("false_alarms_removed"))
    print("360 HDF5 outcomes and 4,680 sealed method-shot results matched.")


if __name__ == "__main__":
    main()
