"""Replace the extra within-shot direction with matched, already-ended evidence."""

import argparse
from pathlib import Path
from shutil import copyfile
import time
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree
import torch

from . import commissioning_persistence_20261111 as persistent
from . import commissioning_persistence_learning_20261111 as history
from . import commissioning_radial_east_20261111 as reference_model
from . import io23, simple_data
from .commissioning_transfer_20261108 import features
from .commissioning_transition_east_20261110 import scores
from .hdf5_score_recheck_20261103 import score_curve
from .history_gradient_metric_20261102.run import read, sha, write
from .lifecycle_stream_20261025.audit import flips
from .mask_repair_20261104.east_evaluation import configure, load_inputs, prepare_actual
from .mask_separated_intra_20261103.run import state_digest
from .source_anticipation_20261104.static_stream import timing_decomposition
from .warning_lead_metrics import report


BASES, STARTUP = reference_model.BASES, reference_model.STARTUP
FULL_DONOR, COMMON_SELF, COMMON_DONOR = ('ended_donor_same_budget',
                                         'own_on_common_support', 'donor_on_common_support')
ARMS = (*BASES, FULL_DONOR, COMMON_SELF, COMMON_DONOR)
CALIPERS = np.array([np.log(1.10), np.log(1.10), .02, np.log(2.)])
PHYSICAL = [io23.RAW23.index(name) for name in ('ip', 'Bt')]


def match_context(values, observed, releases):
    """Current, field, signed fractional current change, and observed prefix age."""
    index = releases[:, None] + np.arange(-32, 0)
    available = observed[index] & np.isfinite(values[index])
    enough = ((available[:, :16, 0].sum(1) >= 8)
              & (available[:, 16:].sum(1) >= 8).all(1))
    x = np.where(available[enough], values[index[enough]], np.nan).astype(np.float64)
    ip_before = np.nanmedian(abs(x[:, :16, 0]), axis=1)
    ip_now = np.nanmedian(abs(x[:, 16:, 0]), axis=1)
    bt = np.nanmedian(x[:, 16:, 1], axis=1)
    nonzero = (ip_now > 0.) & (bt != 0.)
    i = np.flatnonzero(enough)[nonzero]
    valid, field = np.zeros(len(releases), bool), np.zeros(len(releases), bool)
    context = np.zeros((len(releases), 4), np.float64)
    valid[i], field[i] = True, bt[nonzero] > 0.
    a, b = ip_before[nonzero], ip_now[nonzero]
    context[i] = np.column_stack((np.log(b), np.log(abs(bt[nonzero])),
                                  (b-a)/np.maximum(a, b), np.log(releases[i]))) / CALIPERS
    return context, field, valid


def select(query, bank, rank):
    """Nearest context within one scaled Euclidean unit, exact mask and Bt sign."""
    choice = np.full(len(query['context']), -1, dtype=np.int64)
    distance = np.full(len(choice), np.inf)
    for g in np.unique(query['group']):
        target = np.flatnonzero(query['group'] == g)
        past = np.flatnonzero((bank['rank'] < rank) & (bank['group'] == g))
        if past.size:
            d, j = cKDTree(bank['context'][past]).query(query['context'][target],
                                                     distance_upper_bound=1.)
            use = np.isfinite(d)
            choice[target[use]], distance[target[use]] = past[j[use]], d[use]
    return choice, distance


def release_states(length, release, selected, pool):
    """Hold each selected completed window for its original 32-point block only."""
    state = np.zeros((length, pool.shape[1]), dtype=pool.dtype)
    support = np.zeros(length, bool)
    for begin, donor in zip(release, selected, strict=True):
        if donor >= 0:
            sl = slice(begin, begin+persistent.STRIDE)
            state[sl], support[sl] = pool[donor], True
    return state, support


def equalized_delta(own, donor, support):
    """Match the L2 norm *after* projecting both directions in recipient coordinates."""
    a, b = (t.norm(dim=1) for t in (own, donor))
    # A projected zero is not a direction. It is outside common support, even
    # when the donor's raw diagnostic vector and its match are both nonzero.
    applied = torch.from_numpy(support) & (a > 0.) & (b > 0.)
    result = torch.zeros_like(own)
    result[applied] = (donor[applied]/b[applied, None])*a[applied, None]
    return result, applied


@torch.no_grad()
def components(raw, signal, args, rank):
    _,net,params,old,quiet,_,_,_ = args
    length = len(raw['times'])
    tr,pre,ready = history.observed_track(net,signal,np.arange(length),params,old)
    states,_ = persistent.replay(*ready,quiet)
    state = states['current_equal_norm']
    window = np.arange(2,len(tr['q']))
    release = tr['q'][window]+persistent.WINDOW+1
    window = window[release < length]
    release = tr['q'][window]+persistent.WINDOW+1
    active = state[release].any(1)
    release,window = release[active],window[active]
    mask = ready[1][window[:, None]+np.arange(-2,1)].all(1)
    context,field,valid = match_context(raw['values'][:,PHYSICAL],raw['available'][:,PHYSICAL],release)
    mask_code = mask.astype(np.int64) @ (1 << np.arange(mask.shape[1]))
    rows = dict(context=context[valid], group=2*mask_code[valid]+field[valid],
                direction=state[release[valid]], release=release[valid],
                rank=np.full(int(valid.sum()),rank,dtype=np.int64))
    return torch.from_numpy(tr['pre']),pre,torch.from_numpy(tr['key']),state,rows,len(release)


@torch.no_grad()
def encode(raw, signal, args, rank, bank):
    bare,pre,key,state,rows,n_active = components(raw,signal,args,rank)
    picked,distance = select(rows,bank,rank)
    selected = picked >= 0
    donor,support = release_states(len(state),rows['release'],picked,bank['direction'])
    own,other = (torch.einsum('...k,...f,kfw->...w', key, 4.*torch.from_numpy(s), args[-3])
                 for s in (state,donor))
    other,applied = equalized_delta(own,other,support)
    own_common = torch.where(applied[:,None],own,0.)
    swapped = torch.where(applied[:,None],other,own)
    latent = (bare,pre,pre+own,pre+swapped,pre+own_common,pre+other)
    x,_ = features(args[0],signal,'cuda')
    curves = scores(x,{k:v.tanh().numpy() for k,v in zip(ARMS,latent,strict=True)},args[-2],ARMS)
    if not all(np.isfinite(p).all() for p in curves.values()):
        raise FloatingPointError('nonfinite sealed probability')
    a,b = (d.norm(dim=1).numpy() for d in (own,swapped))
    np.testing.assert_allclose(b,a,rtol=5e-7,atol=1e-8)
    on = applied.numpy()
    np.testing.assert_array_equal(curves[FULL_DONOR][~on],curves['legacy_current4'][~on])
    for k in (COMMON_SELF,COMMON_DONOR):
        np.testing.assert_array_equal(curves[k][~on],curves['inherited'][~on])
    j = picked[selected]
    u,v = (d[on].double() for d in (own,swapped))
    direction_cos = ((u*v).sum(1)/(u.norm(dim=1)*v.norm(dim=1))).numpy()
    receipt = dict(applied=on, own_norm=a, donor_norm=b, direction_cos=direction_cos,
                   active_releases=np.array(n_active), eligible_releases=rows['release'],
                   matched_release=rows['release'][selected], donor_rank=bank['rank'][j],
                   donor_release=bank['release'][j], distance=distance[selected],
                   group=rows['group'][selected], donor_group=bank['group'][j])
    return curves,receipt,rows


def run(reference, coordinates, output):
    configure()
    started = time.monotonic()
    lock = load_inputs()
    ids = lock['ids']
    p,old_seal,args,assets = reference_model._frozen(reference,coordinates,lock)
    frozen = tuple(state_digest(m) for m in args[:2])
    cutoffs,original = p['calibration'],p['original']
    producer = Path(__file__)
    loader = Path(reference_model.__file__)
    code = dict(p['code_sha256'], **{f.name:sha(f) for f in (producer,loader)})
    output.mkdir(parents=True)
    (output/'shots').mkdir()
    archive = output/'executed_code'
    archive.mkdir()
    for name,value in code.items():
        copyfile(producer.parent/name,archive/name)
        if sha(archive/name)!=value:
            raise ValueError(f'archived producer differs: {name}')
    write(output/'protocol.json',dict(
        scope='EAST360_ENDED_DONOR_DIRECTION_CONTROL_DEVELOPMENT',order=ids,startup=STARTUP,
        reference=str(reference),assets=assets,code_sha256=code,source_sha256=p['source_sha256'],
        arms=ARMS,original=original,calibration=cutoffs,first_effect=persistent.FIRST_EFFECT,
        window=persistent.WINDOW,stride=persistent.STRIDE,no_suffix_fit=True,
        outcome_access='all 360 curves sealed before outcome access',trainable_parameter_updates=0,
        donor_bank='ended preparation mapped after their original predictions; later append only after own seal',
        context='last two 16-point Ip medians, most recent Bt median, log release age; no full-shot length',
        calipers=CALIPERS.tolist(),max_scaled_euclidean_distance=1.,
        exact_match='Bt sign and ten-channel effective three-window support mask',
        intervention='swap 10D signed residual; keep recipient key, release and exact projected L2',
        unmatched='full-dose comparison retains own direction; common-support comparison sets both increments zero'))

    def load_raw(sid):
        if sha(io23.path_for('EAST',sid))!=lock['hashes'][str(sid)]:
            raise ValueError('locked raw file changed')
        return simple_data.read_shot('EAST',sid,rate_contract='nominal')

    history_bank,hashes,witnesses = [],{},{}
    with patch.object(io23,'read_outcome',side_effect=RuntimeError('unsealed EAST outcome access')):
        for rank,sid in enumerate(ids,1):
            name = f'{rank:04d}_{sid}.npz'
            path = reference/'shots'/name
            if sha(path)!=old_seal['curve_sha256'][name]:
                raise ValueError('locked reference curve changed')
            with np.load(path,allow_pickle=False) as f:
                times,arch = f['times'],{k:f[k] for k in BASES}
            receipt,rows = {},None
            if rank<=STARTUP:
                curves = dict.fromkeys(ARMS,arch['fixed'])
            else:
                if rank==STARTUP+1:
                    for prev,older in enumerate(ids[:STARTUP],1):
                        cold = load_raw(older)
                        sig = prepare_actual(cold,rank,lock['mapping'])
                        history_bank.append(components(cold,sig,args,prev)[-2])
                bank = {k:np.concatenate([r[k] for r in history_bank]) for k in history_bank[0]}
                raw = load_raw(sid)
                np.testing.assert_array_equal(raw['times'],times)
                sig = prepare_actual(raw,rank,lock['mapping'])
                curves,receipt,rows = encode(raw,sig,args,rank,bank)
                for k in BASES:
                    np.testing.assert_array_equal(curves[k],arch[k])
                if rank in (21,120,360):
                    end = persistent.FIRST_EFFECT+3*persistent.STRIDE+5
                    raw_prefix = {k:raw[k][:end] for k in ('times','values','available')}
                    sig_prefix = {k:sig[k][:end] for k in ('values','available')}
                    cs,rr,_ = encode(raw_prefix,sig_prefix,args,rank,bank)
                    for k,v in cs.items():
                        np.testing.assert_allclose(v,curves[k][:end],atol=2e-6,rtol=0.)
                        for cutoff in cutoffs.values():
                            np.testing.assert_array_equal(v>=cutoff,curves[k][:end]>=cutoff)
                    np.testing.assert_array_equal(rr['applied'],receipt['applied'][:end])
                    for k in ('matched_release','donor_rank','donor_release','distance'):
                        np.testing.assert_array_equal(rr[k],receipt[k][receipt['matched_release']<end])
                    witnesses[str(sid)] = end
            target = output/'shots'/name
            with target.open('xb') as f:
                np.savez_compressed(f,times=times,**curves,**receipt)
            hashes[name] = sha(target)
            if rows is not None:
                history_bank.append(rows)
            if rank%40==0 or rank==STARTUP+1:
                print(dict(stage='sealed',shots=rank,seconds=round(time.monotonic()-started,2)),flush=True)
    if tuple(state_digest(m) for m in args[:2])!=frozen:
        raise ValueError('frozen model changed during donor control')
    write(output/'seal.json',dict(curve_sha256=hashes,prefix_witnesses=witnesses))

    per = {c:{k:[] for k in ARMS} for c in cutoffs}
    for rank,sid in enumerate(ids,1):
        name = f'{rank:04d}_{sid}.npz'
        path = output/'shots'/name
        if sha(path)!=hashes[name]:
            raise ValueError('unsealed curve')
        d,down = io23.read_outcome('EAST',sid)
        with np.load(path,allow_pickle=False) as f:
            for c,g in per.items():
                threshold = original if rank<=STARTUP else cutoffs[c]
                for k in ARMS:
                    g[k].append(dict(shot=sid,disruptive=d,down_s=down,
                                     **score_curve(f['times'],f[k],d,down,threshold)))
    old = read(reference/'per_shot.json')
    if any(per[c][k]!=old[c][k] for c in cutoffs for k in BASES):
        raise ValueError('frozen parent scoring changed')
    write(output/'per_shot.json',per)
    metrics = {c:{k:timing_decomposition(v) for k,v in g.items()} for c,g in per.items()}
    comparisons = (('fixed','legacy_current4'),('inherited','legacy_current4'),
                   (FULL_DONOR,'legacy_current4'),(COMMON_DONOR,COMMON_SELF))
    paired = {c:{f'{a}__to__{b}':flips(g[a],g[b]) for a,b in comparisons} for c,g in per.items()}
    write(output/'lead_sensitivity.json',report([output/'per_shot.json'],('fixed',FULL_DONOR,COMMON_DONOR)))
    write(output/'result.json',dict(status='COMPLETED_EAST360_ENDED_DONOR_CONTROL_DEVELOPMENT',
        metrics=metrics,paired=paired,seconds=time.monotonic()-started))
    print({c:{k:(r['tp'],r['fp']) for k,r in g.items()} for c,g in metrics.items()},flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('reference','coordinates','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    a = parser.parse_args()
    run(a.reference.resolve(),a.coordinates.resolve(),a.output.resolve())
