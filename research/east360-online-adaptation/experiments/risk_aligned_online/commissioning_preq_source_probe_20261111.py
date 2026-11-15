"""J-TEXT TRAIN-only prequential readout; rank the archived VAL pairs.

Run as a module with --source pointing to the archived lag-source run, --mean
to its sealed graph_memory/source_mean.npz, and --output to a new JSON file.
The four arms share one ridge readout. This is a pair-ranking diagnostic, not
an alarm evaluation, threshold selection, or an independent TEST result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ('fixed', 'real', 'shifted', 'lag')
RIDGE = .001


def fit_readout(fixed, real, risk, shot, *, ridge=RIDGE):
    """Weighted ridge on both views, with an unpenalized intercept.

    Each endpoint label receives mass 1/2, each shot within a label receives
    equal mass, and its rows share that mass. Fixed and real each get half of
    every row's mass, so adding the second view does not change total weight.
    """
    from .commissioning_transfer_20261108 import shot_weights

    fixed = np.asarray(fixed, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    risk = np.asarray(risk)
    shot = np.asarray(shot)
    if (fixed.ndim != 2 or fixed.shape[1] != 18 or real.shape != fixed.shape
            or risk.shape != (len(fixed),) or shot.shape != risk.shape
            or not np.isfinite(fixed).all() or not np.isfinite(real).all()
            or not np.isin(risk, (0, 1)).all()
            or not np.isfinite(ridge) or ridge <= 0):
        raise ValueError('finite paired 18-feature views, binary labels and positive ridge required')
    w = shot_weights(shot, risk.astype(bool))
    y = risk.astype(np.float64)
    mean_x = .5 * (w @ fixed + w @ real)
    mean_y = float(w @ y)
    f, r = fixed - mean_x, real - mean_x
    covariance = .5 * ((f.T * w) @ f + (r.T * w) @ r)
    target = .5 * ((f.T + r.T) @ (w * (y - mean_y)))
    weights = np.linalg.solve(covariance + ridge * np.eye(18), target)
    intercept = mean_y - float(mean_x @ weights)
    if not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise FloatingPointError('nonfinite ridge readout')
    return dict(weights=weights, intercept=intercept, ridge=float(ridge),
                weight_sum=float(w.sum()))


def locked_pair_ids(archived, val_ids):
    """Recover, without rematching, the old lag-source fifty VAL identities."""
    held = archived['held']
    pairs = held['pair_ids']
    if held['pairs'] != 50 or len(pairs) != 50:
        raise ValueError('expected the original 50 archived lag-source pairs')
    val = set(val_ids)
    identities = []
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in pair)
                or pair[0] not in val or pair[2] not in val
                or pair[1] < 0 or pair[3] < 0):
            raise ValueError('invalid archived VAL pair identity')
        identities.append(list(pair))
    if len({tuple(p) for p in identities}) != 50:
        raise ValueError('duplicate archived pair identity')
    return identities


def collect_shot(sid, advanced, center, scale, prior):
    """Sample one supervised source shot, preserving the old endpoint rule."""
    from . import commissioning_lag_data_20261110 as lag
    from . import commissioning_lag_prequential_20261111 as prequential
    from . import commissioning_transition_source_20261110 as source
    from . import simple_data
    from .commissioning_transition_net_20261110 import FIRST, HORIZON, STRIDE

    raw = simple_data.read_shot('J-TEXT', sid)
    if raw['device'] != 'J-TEXT' or int(raw['shot']) != sid:
        raise ValueError('source shot identity changed')
    labeled = simple_data.attach_outcome(raw, 'J-TEXT', advanced)
    endpoints = source.decision_points(labeled['y'].astype(bool), labeled['valid'])
    z, mask, _, _ = source.shot_coordinates(raw, center, scale)
    origins = np.arange(FIRST, len(z) - HORIZON, STRIDE)
    aligned, shifted, eligible = lag.feedback(z, mask, origins + HORIZON)
    views, audit = prequential.prequential_views(len(z), origins, aligned, shifted, eligible, prior)
    if set(views) != set(ARMS) or any(
        views[arm].shape != (len(z), 18) or views[arm].dtype != np.float32
        or not np.isfinite(views[arm]).all() for arm in ARMS
    ):
        raise ValueError('invalid prequential view contract')
    lead = (np.round((labeled['down'] - labeled['times'][endpoints]) * 1000., 9)
            if labeled['is_disrupt'] else np.zeros(len(endpoints)))
    return dict(shot=int(sid), end=endpoints, risk=labeled['y'][endpoints],
                disruptive=bool(labeled['is_disrupt']), lead=lead,
                views={arm: views[arm][endpoints] for arm in ARMS}, audit=audit)


def compare_pairs(pair_ids, records, readout):
    """Compare the same positive/safe rows; exact score differences define ties."""
    weights = np.asarray(readout['weights'], dtype=np.float64)
    intercept = float(readout['intercept'])
    if weights.shape != (18,) or not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError('invalid ranking readout')
    arms = {arm: dict(wins=0, losses=0, ties=0) for arm in ARMS}
    comparisons = []
    for identity in pair_ids:
        ps, pe, ss, se = identity
        try:
            positive, safe = records[(ps, pe)], records[(ss, se)]
        except KeyError as exc:
            raise ValueError(f'archived pair endpoint was not sampled: {identity}') from exc
        if (positive['risk'] != 1 or not positive['disruptive'] or positive['lead'] <= 10.
                or safe['risk'] != 0 or safe['disruptive']):
            raise ValueError(f'archived pair no longer meets label and strict lead rules: {identity}')
        margins, scores = {}, {}
        for arm in ARMS:
            a = np.asarray(positive['views'][arm], dtype=np.float64)
            b = np.asarray(safe['views'][arm], dtype=np.float64)
            if a.shape != (18,) or b.shape != (18,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
                raise ValueError('invalid pair feature')
            # Subtract before applying the intercept: it cancels algebraically
            # and cannot round a small nonzero ranking margin into a tie.
            margin = float((a - b) @ weights)
            margins[arm] = margin
            scores[arm] = dict(positive=float(a @ weights + intercept),
                               safe=float(b @ weights + intercept))
            arms[arm]['wins' if margin > 0 else 'losses' if margin < 0 else 'ties'] += 1
        comparisons.append(dict(identity=list(identity), margins=margins, scores=scores))
    return dict(pair_ids=[list(p) for p in pair_ids], arms=arms, pairs=comparisons)


def diagnose(roles, advanced, center, scale, prior, pair_ids, *, expected_train_shots=None):
    """Fit on TRAIN only and replay the exact requested VAL identities."""
    train, val, test = (list(roles[k]) for k in ('train', 'val', 'test'))
    if (any(len(ids) != len(set(ids)) for ids in (train, val, test))
            or set(train) & set(val) or set(train) & set(test) or set(val) & set(test)):
        raise ValueError('source role lists must be distinct and disjoint')
    required_val = {sid for p in pair_ids for sid in (p[0], p[2])}
    if not required_val <= set(val):
        raise ValueError('pair contains a non-VAL shot')

    fixed, real, risks, shots = [], [], [], []
    used = []
    train_blocks = 0
    for i, sid in enumerate(train, 1):
        rows = collect_shot(sid, advanced, center, scale, prior)
        if len(rows['end']):
            fixed.append(rows['views']['fixed'])
            real.append(rows['views']['real'])
            risks.append(rows['risk'])
            shots.append(np.full(len(rows['end']), sid, dtype=np.int64))
            used.append(int(sid))
            train_blocks += rows['audit']['observed_blocks']
        if i % 300 == 0 or i == len(train):
            print(dict(stage='train_features', shots=i, total=len(train)), flush=True)
    if expected_train_shots is not None and used != expected_train_shots:
        raise ValueError('training sampled shots differ from the archived lag-source run')
    if not used:
        raise ValueError('no sampled TRAIN endpoints')
    fixed, real, risk, shot = (np.concatenate(a) for a in (fixed, real, risks, shots))
    readout = fit_readout(fixed, real, risk, shot)
    train_summary = dict(rows=len(risk), positive_rows=int(np.sum(risk == 1)),
                         negative_rows=int(np.sum(risk == 0)),
                         positive_shots=int(len(np.unique(shot[risk == 1]))),
                         negative_shots=int(len(np.unique(shot[risk == 0]))),
                         observed_blocks=int(train_blocks))

    records = {}
    val_used = []
    for sid in val:
        if sid not in required_val:
            continue
        rows = collect_shot(sid, advanced, center, scale, prior)
        val_used.append(int(sid))
        for i, endpoint in enumerate(rows['end']):
            key = (int(sid), int(endpoint))
            if key in records:
                raise ValueError('duplicate sampled endpoint')
            records[key] = dict(risk=int(rows['risk'][i]), disruptive=rows['disruptive'],
                                lead=float(rows['lead'][i]),
                                views={arm: rows['views'][arm][i] for arm in ARMS})
    held = compare_pairs(pair_ids, records, readout)
    held['val_shots'] = val_used
    held['unique_positive_shots'] = len({p[0] for p in pair_ids})
    held['unique_safe_shots'] = len({p[2] for p in pair_ids})
    return dict(train_shots=used, train=train_summary, readout=readout, held=held)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run(source, mean, output):
    from . import commissioning_event_roughness_20261110 as event_roughness
    from . import commissioning_lag_data_20261110 as lag
    from . import commissioning_lag_prequential_20261111 as prequential
    from . import commissioning_roughness_20261110 as roughness
    from . import commissioning_transfer_20261108 as transfer
    from . import commissioning_transition_net_20261110 as transition_net
    from . import commissioning_transition_source_20261110 as transition
    from . import io23, simple_data
    from .redesign_20261004 import prepare_prior
    from .redesign_20261004.prepare_prior import DEFAULT_PARENT, PARENT_SHA256, load_parent

    source, mean, output = (Path(p).resolve() for p in (source, mean, output))
    if output.suffix.lower() != '.json' or output.exists():
        raise ValueError('--output must be a new JSON file')
    if not output.parent.is_dir():
        raise ValueError('output parent directory must already exist')
    source_hashes = {name: sha256(source / name) for name in ('protocol.json', 'model.npz', 'result.json')}
    protocol, archived = (read_json(source / name) for name in ('protocol.json', 'result.json'))
    graph_protocol = read_json(mean.parent / 'protocol.json')
    graph_seal = read_json(mean.parent / 'seal.json')
    mean_hash = sha256(mean)
    if (protocol['parent_sha256'] != PARENT_SHA256
            or archived['model_sha256'] != source_hashes['model.npz']
            or graph_protocol['source_sha256'] != PARENT_SHA256
            or graph_protocol['source_assets'] != source_hashes
            or graph_seal['artifacts'].get(mean.name) != mean_hash):
        raise ValueError('source run, parent or sealed graph mean lineage changed')
    with np.load(mean, allow_pickle=False) as archive:
        prior = np.asarray(archive['mean'], dtype=np.float32)
    if prior.shape != (70,) or not np.isfinite(prior).all() or not np.linalg.norm(prior) > 0:
        raise ValueError('sealed source mean must be a finite nonzero 70-vector')
    roles = io23.source_roles()
    pair_ids = locked_pair_ids(archived, roles['val'])
    advanced = io23.advanced_times()
    base, _ = load_parent(DEFAULT_PARENT)
    center, scale = (v.detach().cpu().numpy().copy() for v in (base.center, base.scale))
    result = diagnose(roles, advanced, center, scale, prior, pair_ids,
                      expected_train_shots=archived['train_shots'])
    result['readout']['weights'] = result['readout']['weights'].tolist()
    result.update(
        scope='JTEXT_TRAIN_FIT_ARCHIVED_VAL50_PAIR_RANKING_NOT_TEST_OR_ALARMS',
        protocol=dict(label='HDF5 IsDisrupt and DownTime; max(25 ms, AdvancedTime)',
                      endpoint_sampling='existing transition_source decision_points',
                      feedback='q=127+32k; complete H25 only; view at exact archived endpoint',
                      objective='label-then-shot equal weights, fixed and real half each, ridge MSE',
                      ridge=RIDGE, arms=list(ARMS), tie='exact zero score difference',
                      no_test_or_east_waveforms=True),
        input_sha256=dict(source_assets=source_hashes, source_mean=mean_hash,
                          graph_protocol=sha256(mean.parent / 'protocol.json'),
                          graph_seal=sha256(mean.parent / 'seal.json'),
                          parent=PARENT_SHA256, advanced_times=sha256(io23.JTEXT_ADVANCED_TIME_PATH)),
        code_sha256={name: sha256(Path(module.__file__)) for name, module in (
            ('probe', __import__(__name__, fromlist=[''])), ('prequential', prequential),
            ('lag_data', lag), ('transition_source', transition), ('shot_weights', transfer),
            ('transition_net', transition_net), ('roughness', roughness),
            ('event_roughness', event_roughness), ('prepare_prior', prepare_prior),
            ('io23', io23), ('simple_data', simple_data))})
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(dict(stage='complete', train_shots=len(result['train_shots']),
               pairs=len(pair_ids), arms=result['held']['arms'], output=str(output)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--mean', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.mean, args.output)
