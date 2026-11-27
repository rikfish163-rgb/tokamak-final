"""Single-pass EAST360 replay separating between-discharge statistics and alarm feedback."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from zero_cold_start import backbone, data, engine, experiments, metrics
from shot_memory import contract, digest, pack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from risk_aligned_online.warmstart_alignment_20261022 import alignment


OUTPUT = Path(__file__).with_suffix('.json')
ORDER_HASH = '3fefd51a0a1e16bb82735f59e8b332857eecd1c0d66102e6b4b413371a93ce97'
SOURCE_HASH = '45b17a4d56b710c8ba349af53778c175a46112776ed1525dcd508953bb1d51e7'
ARMS = ('none', 'recent', 'equal', 'old_stats', 'old_feedback', 'old_joint', 'equal_feedback')
PAIRS = (('none', 'equal'), ('recent', 'equal'), ('old_stats', 'equal'),
         ('none', 'old_stats'), ('none', 'old_feedback'), ('old_stats', 'old_joint'),
         ('equal', 'equal_feedback'), ('old_joint', 'equal'), ('old_joint', 'equal_feedback'))


class PrefixHistory:
    """A channel's qualified past discharges each contribute one empirical distribution."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = {j: [] for j in cfg.align_channels}
        self.coverage = {data.RAW23[j]: Counter() for j in cfg.align_channels}
        self.last_shot = -1

    def update(self, shot, times, values, available, equal, recent, source_center, source_scale):
        if shot <= self.last_shot:
            raise ValueError('duplicate or out-of-order discharge in prefix history')
        self.last_shot = shot
        prefix = (times - times[0]) * 1000. < self.cfg.align_prefix_ms
        for j, samples in self.samples.items():
            audit = self.coverage[data.RAW23[j]]
            selected = prefix & available[:, j] & np.isfinite(values[:, j])
            v = values[selected, j].astype(float)
            if len(v) < 2:
                audit['missing' if not len(v) else 'one_point'] += 1
                continue
            if np.any(np.diff(times[selected]) * 1000. > self.cfg.max_gap_ms + 1e-7):
                audit['gapped'] += 1
                continue
            if np.subtract(*np.percentile(v, [75, 25])) < self.cfg.scale_floor:
                audit['observed_unidentifiable_scale'] += 1
                continue
            audit['used'] += 1
            if len(samples) == 0:
                audit['first_used_shot'] = shot
            v = v[np.linspace(0, len(v) - 1, min(128, len(v)), dtype=int)]
            samples.append(v)
            pooled = np.concatenate(samples)
            n = len(samples)
            sizes = np.array([len(s) for s in samples])
            age = np.arange(n - 1, -1, -1)
            for st, votes in ((equal, np.ones(n)), (recent, (1. - self.cfg.align_ema) ** age)):
                point_weights = np.repeat(votes / sizes, sizes)
                lo, c, hi = alignment._weighted_quantile(pooled, point_weights, [.25, .50, .75])
                if hi - lo >= self.cfg.scale_floor:
                    st.target_center[j], st.target_scale[j] = c, hi - lo
                    st.target_n_eff[j] = min(n, self.cfg.n_eff_cap)
                else:
                    st.target_center[j], st.target_scale[j] = source_center[j], source_scale[j]
                    st.target_n_eff[j] = 0


def retune(pred, cfg, multiplier):
    """Reuse identical network scores when only the previous multiplier has changed."""
    baseline, tau, idx, r0, spread = engine._detect(pred.risk_curve, pred.times, cfg, multiplier)
    assert (r0, spread) == (pred.ref_risk, pred.ref_spread)
    baseline.flags.writeable = tau.flags.writeable = False
    return replace(pred, baseline_curve=baseline, threshold_curve=tau, alarm_idx=idx,
                   alarm=idx is not None, alarm_time=None if idx is None else float(pred.times[idx]))


def fingerprint(state):
    return json.dumps(pack(state), sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def seal(predictions, states, source_line, cfg):
    """Hash issued forecasts before requesting metadata, including score-only control alarms."""
    fixed_alarms = {}
    signature = sha256(source_line.tobytes())
    for name, pred in predictions.items():
        idx = engine._persistence(pred.risk_curve, source_line, pred.times, cfg)
        fixed_alarms[name] = None if idx is None else float(pred.times[idx])
        signature.update(json.dumps([name, pred.shot, pred.alarm_time, fixed_alarms[name]]).encode())
        signature.update(fingerprint(states[name]))
        for curve in (pred.times, pred.risk_curve, pred.threshold_curve):
            assert not curve.flags.writeable
            signature.update(curve.tobytes())
    return fixed_alarms, signature.hexdigest()


def scored(pred, line, first, disruptive, down, cfg):
    row = metrics.score_shot(pred.times, pred.risk_curve, line, disruptive, down,
                             hold_ms=cfg.hold_ms, deadline_ms=cfg.min_lead_ms)
    eligible_first = None if disruptive and first is not None and first > down + 1e-9 else first
    assert eligible_first == row['alarm_time']
    early = np.round((down - pred.times) * 1000., 6) > cfg.min_lead_ms if disruptive else slice(None)
    row.update(shot=pred.shot, timely_peak=float(pred.risk_curve[early].max(initial=-np.inf)),
               startup_fp=(row['outcome'] == 'FP' and (first - pred.times[0]) * 1000. < cfg.ref_ms))
    return row


def summary(rows):
    stats = metrics.summarize_stream(rows, checkpoints=())
    keys = ('n_disruptive', 'n_normal', 'tp_timely', 'tardy', 'miss', 'fp', 'tn',
            'timely_tpr', 'fpr', 'premature_gt3000_count')
    out = {k: stats[k] for k in keys}
    normal = sorted((r['timely_peak'] for r in rows if not r['disruptive']), reverse=True)
    assert normal and len(normal) == out['n_normal']
    budget = (len(normal) - 1) // 5
    cutoff = np.nextafter(normal[budget], np.inf)
    out.update(max_fp=budget, passed={str(p): 100 * out['fp'] < p * len(normal) for p in (20, 15, 10)},
               startup_fp=int(sum(r['startup_fp'] for r in rows)),
               posthoc_fixed_tp=sum(int(r['timely_peak'] >= cutoff) for r in rows if r['disruptive']),
               posthoc_fixed_fp=sum(int(s >= cutoff) for s in normal))
    return out


def paired(before, after):
    flips = {key: [] for key in ('gained_timely', 'lost_timely', 'added_fp', 'removed_fp')}
    transitions = Counter()
    for a, b in zip(before, after, strict=True):
        assert (a['shot'], a['disruptive']) == (b['shot'], b['disruptive'])
        left, right = a['outcome'], b['outcome']
        transitions[f'{left}->{right}'] += 1
        if left != right:
            for flag, is_flip in (("gained_timely", right == 'TP_timely'), ("lost_timely", left == 'TP_timely'),
                                  ("added_fp", right == 'FP'), ("removed_fp", left == 'FP')):
                if is_flip:
                    flips[flag].append(a['shot'])
    return {**flips, 'transitions': dict(transitions)}


def evaluate():
    ids = data._east360()
    assert sha256(','.join(map(str, ids)).encode()).hexdigest() == ORDER_HASH
    model, ckpt = backbone.load_source_model()
    cfg = experiments.locked_configs(float(ckpt['val_threshold_fpr5']))['COH6_Reflective_M100']
    provenance = contract(cfg)
    assert provenance['source_sha256'] == SOURCE_HASH and cfg.hold_ms == 1
    sc, ss = experiments.source_coords(ckpt, cfg.align_mode)
    states = {n: engine.InterShotState.from_source(sc, ss, cfg) for n in ARMS}
    settings = {n: replace(cfg, reflect_align=n in ('old_stats', 'old_joint'),
                           reflect_threshold=n in ('old_joint', 'old_feedback', 'equal_feedback')) for n in ARMS}
    history = PrefixHistory(cfg)
    frozen, witnesses, checkpoints = {}, {}, {}
    native, common, ledger = defaultdict(list), defaultdict(list), []
    for rank, sid in enumerate(ids, 1):
        sig = data.read_signals('EAST', sid)
        t, v, mask = sig['times'], sig['values'], sig['available']
        active = {**states, **(frozen if rank > 270 else {})}
        predictions = {}
        for name, state in active.items():
            base = {'old_feedback': 'none', 'equal_feedback': 'equal'}.get(name)
            if base:
                assert all(np.array_equal(getattr(state, k), getattr(states[base], k))
                           for k in ('target_center', 'target_scale', 'target_n_eff'))
                predictions[name] = retune(predictions[base], cfg, state.multiplier)
            else:
                predictions[name] = engine.run_shot_causal(sid, t, v, mask,
                    lambda x, m: backbone.predict_shot_proba(model, x, m), state, cfg, sc, ss)
        source_line = np.full_like(t, cfg.source_threshold)
        source_line.flags.writeable = False
        fixed_alarms, signature = seal(predictions, active, source_line, cfg)
        issued = {n: p.alarm_time for n, p in predictions.items()}
        # One row is sealed before the first read of the current HDF5 outcome.
        ledger.append([sid, signature, list(issued.values()), list(fixed_alarms.values())])
        y, down = data.read_outcome('EAST', sid)
        for n, p in predictions.items():
            native[n].append(scored(p, p.threshold_curve, issued[n], y, down, cfg))
            common[n].append(scored(p, source_line, fixed_alarms[n], y, down, cfg))
        ledger[-1].extend((y, down))
        for n in ('old_stats', 'old_feedback', 'old_joint', 'equal_feedback'):
            engine.reflect_and_update(predictions[n], v, mask, y, down, states[n], settings[n], sc, ss)
        history.update(sid, t, v, mask, states['equal'], states['recent'], sc, ss)
        for n in ('none', 'recent', 'equal'):
            states[n].shots_seen += 1
        for k in ('target_center', 'target_scale', 'target_n_eff'):
            setattr(states['equal_feedback'], k, getattr(states['equal'], k).copy())
        if rank in (20, 270):
            name = f'equal_frozen_{rank}'
            frozen[name] = deepcopy(states['equal'])
            witnesses[name] = fingerprint(frozen[name])
        if rank in (1, 20, 90, 180, 270, 360):
            checkpoints[rank] = {n: pack(states[n]) for n in ARMS}
        if rank % 90 == 0:
            print(f'Sealed and scored {rank}/360', flush=True)

    assert all(fingerprint(st) == witnesses[n] for n, st in frozen.items())
    assert all(st.shots_seen == len(ids) for st in states.values())
    for n in ARMS:
        assert native[n][0] == native['none'][0] and common[n][0] == common['none'][0]
    actual = {n: summary(native[n]) for n in ARMS}
    old = actual['old_joint']
    assert tuple(old[k] for k in ('tp_timely', 'tardy', 'miss', 'fp', 'tn')) == (135, 23, 22, 31, 149)
    assert old['n_disruptive'] == old['n_normal'] == 180
    for n, base in (('old_feedback', 'none'), ('equal_feedback', 'equal')):
        assert common[n] == common[base]
    for key, counts in history.coverage.items():
        assert sum(v for k, v in counts.items() if k != 'first_used_shot') == 360, key
    probe_names = ('none', *frozen, 'equal', 'recent', 'old_joint', 'equal_feedback')
    return dict(provenance={**provenance, 'experiment_sha256': digest(__file__),
                            'quantile_module_sha256': digest(alignment.__file__), 'id_order_sha256': ORDER_HASH},
                arms=list(ARMS), frozen_on_final90=list(frozen), all360=actual,
                same_source_threshold={n: summary(common[n]) for n in ARMS},
                paired={f'{a}_to_{b}': paired(native[a], native[b]) for a, b in PAIRS},
                paired_source={f'{a}_to_equal': paired(common[a], common['equal'])
                               for a in ('none', 'recent', 'old_stats', 'old_joint')},
                blocks90=[{n: summary(native[n][i:i+90]) for n in ARMS} for i in range(0, len(ids), 90)],
                common_last90={n: summary(native[n][-90:]) for n in probe_names},
                last90_pairs={f'{a}_to_{b}': paired(native[a][-90:], native[b][-90:]) for a, b in
                              (('none', 'equal_frozen_20'), ('equal_frozen_20', 'equal_frozen_270'),
                               ('equal_frozen_270', 'equal'), ('old_joint', 'equal_feedback'))},
                coverage=history.coverage, checkpoint_states=checkpoints,
                ledger_fields=['shot', 'pre_reveal_sha256', 'issued_firsts_seconds_in_arm_order',
                               'issued_source_line_firsts_seconds_in_arm_order', 'IsDisrupt', 'DownTime_if_disruptive'],
                ledger=ledger)


if __name__ == '__main__':
    result = evaluate()
    temp = OUTPUT.with_suffix('.pending')
    temp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    temp.replace(OUTPUT)
    for arm, row in result['all360'].items():
        print(arm, row['tp_timely'], row['fp'], flush=True)
