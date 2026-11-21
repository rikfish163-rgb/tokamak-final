"""Direct layer-0 transfer on the frozen limited-label EAST query schedule."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments.risk_aligned_online.warning_lead_metrics import summarize
from layer0_direct_transfer_fit_20261112 import fit_direct
import layer0_local_memory_20261105 as memory
import layer0_query_protocol_20261105 as protocol
import moving_peak_replay_20261105 as r
import run_layer0_query_memory_20261105 as core
from verify_confirmation_raw_replay_20261105 import counts, pair, raw_truth, score


LOCAL = r.ROOT/'probe_outputs/layer0_query_memory360_20261105_v1'
REFS = dict(local=LOCAL,
            uniform=r.ROOT/'probe_outputs/layer0_uniform_kernel360_20261105_v1',
            positive=r.ROOT/'probe_outputs/layer0_disruptive_feedback360_20261105_v1',
            phase=r.ROOT/'probe_outputs/layer0_phase_matched360_20261106_v1')
DIRECT = ('direct', 'direct_with_intra')
PARENT = ('static', 'intra')


def block_files(run, block):
    return tuple(run/'blocks'/f'{block:02d}_{k}.json' for k in ('commit', 'reveals', 'update'))


def curves_for(run, done, rank, sid):
    note = done['sealed'][rank-1]
    path = core.shot_file(run, note, '.npz')
    if (note['rank'], note['shot'])!=(rank, sid) or r.sha(path)!=note['curve_sha256']:
        raise ValueError('reference shot order or curve seal changed')
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def direct_curves(base, case, delta):
    ctx, fast, _ = case
    if not len(fast):
        return dict.fromkeys(protocol.PERSISTENT_ARMS, ctx[2])
    slow = delta.expand_as(fast)
    zero = torch.zeros_like(fast)
    curves, _ = r.forecast(base, ctx, dict(inter=memory.total_action(zero, slow),
                                           both=memory.total_action(fast, slow)))
    return curves


def warning_rows(rows):
    return [dict(shot=a['shot'], disruptive=a['disruptive'], lead_ms=a['lead_ms'],
                 alarm=a['first_index'] is not None, timely=a['status']=='tp') for a in rows]


def learn(base, shot, case, truth, archived, committed, guard, train, delta, out, block, *, old_info):
    t, y, down = truth
    if 'fit' not in old_info:
        return delta, dict(decision='matched_no_fit', matched_reference_decision=old_info['decision'])
    old_paths = {k:archived[k] for k in protocol.PERSISTENT_ARMS}
    ends = protocol.teaching_indices(t, old_paths, y, down, r.THRESHOLD)
    if ends.tolist()!=old_info['teaching_indices'] or old_info['fit']['steps']!=memory.STEPS:
        raise ValueError('direct transfer teaching support or step budget changed')
    train.append((case, ends, y))
    candidate, fit = fit_direct(base, train, delta)
    old = {k:committed[a] for k,a in zip(protocol.PERSISTENT_ARMS, DIRECT, strict=True)}
    after = direct_curves(base, case, candidate)
    helpful, diagnostics = protocol.query_progress(t, old, after, y, down, r.THRESHOLD)
    witness, violations = {}, []
    core.guard_case(witness, violations, shot, truth, old, after)
    for sid, other, earlier in guard:
        core.guard_case(witness, violations, sid, earlier, direct_curves(base, other, delta),
                        direct_curves(base, other, candidate))
    accept = helpful and not violations
    proposal = out/'blocks'/f'{block:02d}_proposal.npz'
    curves = out/'blocks'/f'{block:02d}_candidate_curves.npz'
    r.save_arrays(proposal, delta=candidate.cpu().numpy())
    r.save_arrays(curves, **witness)
    return (candidate if accept else delta), dict(
        decision='accepted' if accept else ('rejected_guard' if violations else 'no_query_progress'),
        fit=fit, teaching_indices=ends.tolist(), guard_ids=[g[0] for g in guard],
        violations=violations, diagnostics=diagnostics, proposal_sha256=r.sha(proposal),
        witness_sha256=r.sha(curves))


def run(out):
    started = time.monotonic()
    r.configure()
    order, cfg, _, base, digest, mapping, _, fit_path = core.parents()
    historical = r.read(LOCAL/'completed.json')
    locked = r.read(LOCAL/'protocol.json')
    if (r.sha(LOCAL/'protocol.json')!=historical['protocol_sha256']
            or locked['order']!=order or locked['threshold']!=r.THRESHOLD
            or locked['raw_sha256']!=cfg['raw_sha256'] or locked['source_state_sha256']!=digest
            or locked['fast_fit_sha256']!=core.RULE_SHA or historical['n_reveals']!=36):
        raise ValueError('matched frozen local query run is required')
    (out/'shots').mkdir(parents=True)
    (out/'blocks').mkdir()
    code = {str(Path(f).resolve()):r.sha(f) for f in (__file__, fit_direct.__code__.co_filename)}
    r.write_new(out/'protocol.json', dict(scope='EAST360_LIMITED_LABEL_DIRECT_LAYER0_DEVELOPMENT',
        order=order, threshold=r.THRESHOLD, raw_sha256=cfg['raw_sha256'], source_state_sha256=digest,
        local_completed_sha256=r.sha(LOCAL/'completed.json'), code_sha256=code,
        arms=(*PARENT,*DIRECT), block=core.BLOCK, cold=core.COLD_BUDGET, query=core.ONLINE_BUDGET,
        fits='same scheduled fits/indices as local; joint refit on all such released shots',
        objective='shot-balanced shared vector fitted on direct and direct-plus-intra',
        protection='same full-curve query and guard rule; current 4D/4S cold guard and 2 recent queries'))
    zero = base.center.new_zeros(memory.ACTION_WIDTH)
    delta = zero
    rule = state_sha = None
    cold_guard, recent, train, sealed, blocks, batch = [], [], [], [], [], []
    rank_by_shot = {sid:rank for rank,sid in enumerate(order,1)}
    reader = r.io23.read_outcome

    def committed_truth(dev, sid):
        if dev!='EAST' or r.sha(r.io23.path_for(dev,sid))!=cfg['raw_sha256'][str(sid)]:
            raise ValueError('committed raw input differs from locked original')
        return reader(dev,sid)

    with patch.object(r.io23, 'read_outcome', side_effect=RuntimeError('label forbidden before commit')):
        for rank, sid in enumerate(order,1):
            before = time.monotonic_ns()
            old = curves_for(LOCAL, historical, rank, sid)
            if state_sha is None:
                direct = {k:old[a] for k,a in zip(DIRECT,PARENT,strict=True)}
            else:
                raw = core.raw_for(sid, cfg['raw_sha256'])
                np.testing.assert_array_equal(raw['times'], old['times'])
                case = core.as_case(base, raw, rank, mapping, rule)
                candidate = direct_curves(base, case, delta)
                direct = {k:candidate[a] for k,a in zip(DIRECT,protocol.PERSISTENT_ARMS,strict=True)}
                del case, raw
            path = out/'shots'/f'{rank:04d}_{sid}.npz'
            r.save_arrays(path, times=old['times'], **{k:old[k] for k in PARENT}, **direct)
            seal = dict(rank=rank, shot=sid, curve_sha256=r.sha(path), state_sha256=state_sha,
                        predict_start_ns=before, sealed_ns=time.monotonic_ns())
            sealed.append(seal)
            batch.append((seal, old, direct))
            if rank%core.BLOCK:
                continue
            block = rank//core.BLOCK
            old_files = block_files(LOCAL, block)
            keys = ('commit_sha256','reveals_sha256','update_sha256')
            if any(r.sha(f)!=historical['blocks'][block-1][k] for f,k in zip(old_files,keys,strict=True)):
                raise ValueError('changed reference query commitments')
            old_commit = r.read(old_files[0])
            if block==1:
                selected = order[:core.COLD_BUDGET]
            elif rank==core.N:
                selected = []
            else:
                selected = [protocol.choose_query(
                    [(n['rank'],n['shot'],o['static'],o['intra']) for n,o,_ in batch], r.THRESHOLD)[0]]
            if old_commit['selected']!=selected or old_commit['rank_end']!=rank:
                raise ValueError('query ID or boundary differs from local control')
            commit, reveal, update = block_files(out, block)
            r.write_new(commit, dict(block=block, rank_end=rank, selected=selected,
                curve_sha256=[n['curve_sha256'] for n,_,_ in batch], state_sha256=state_sha,
                committed_ns=time.monotonic_ns()))
            revealed = protocol.reveal_block(commit, selected, committed_truth)
            r.write_new(reveal, revealed)
            history = r.read(old_files[1])['outcomes']
            identity = ('shot','disruptive','down')
            if [[a[k] for k in identity] for a in revealed['outcomes']]!=[[a[k] for k in identity] for a in history]:
                raise ValueError('same query does not have the same raw outcome')
            info = dict(decision='terminal_no_query')
            if block==1:
                rule, _, cold_guard = core.cold_setup(
                    base, revealed['outcomes'], cfg['raw_sha256'], mapping, fit_path)
                info = dict(decision='cold_release_no_weight_update')
            elif selected:
                label, = revealed['outcomes']
                qid, y, down = (label[k] for k in identity)
                _, archived, committed = next((n,o,v) for n,o,v in batch if n['shot']==qid)
                raw = core.raw_for(qid, cfg['raw_sha256'])
                np.testing.assert_array_equal(raw['times'], archived['times'])
                case = core.as_case(base, raw, rank_by_shot[qid], mapping, rule)
                truth = (raw['times'], y, down)
                delta, info = learn(base,qid,case,truth,archived,committed,
                                    [*cold_guard,*recent],train,delta,out,block,old_info=r.read(old_files[2]))
                recent = [*recent,(qid,case,truth)][-2:]
            previous = state_sha
            if info['decision']=='accepted':
                path = out/'blocks'/f'{block:02d}_state.npz'
                r.save_arrays(path, delta=delta.cpu().numpy())
                state_sha = r.sha(path)
            r.write_new(update, dict(block=block, first_next_rank=rank+1 if rank<core.N else None,
                commit_sha256=r.sha(commit), reveal_sha256=r.sha(reveal),
                state_before=previous, state_after=state_sha, updated_ns=time.monotonic_ns(), **info))
            blocks.append(dict(zip(keys, (r.sha(f) for f in (commit,reveal,update)), strict=True)))
            batch = []
            print(dict(sealed=rank, fits=len(train), decision=info['decision']), flush=True)
    if (r.state_digest(base)!=digest or any(p.grad is not None for p in base.parameters())
            or any(r.sha(path)!=sha for path,sha in code.items())):
        raise ValueError('frozen parent changed during direct fitting')
    r.write_new(out/'seal.json', dict(shots=sealed, blocks=blocks, seconds=time.monotonic()-started,
        protocol_sha256=r.sha(out/'protocol.json'), fitted_shots=len(train)))


def score_run(out):
    p,s = (r.read(out/k) for k in ('protocol.json','seal.json'))
    refs = {k:r.read(folder/'completed.json') for k,folder in REFS.items()}
    if (len(s['shots'])!=len(p['order']) or len(p['order'])!=360 or len(s['blocks'])!=18
            or r.sha(out/'protocol.json')!=s['protocol_sha256']
            or r.sha(LOCAL/'completed.json')!=p['local_completed_sha256']
            or any(r.sha(path)!=h for path,h in p['code_sha256'].items())):
        raise ValueError('missing sealed direct replay')
    arms = (*PARENT,*DIRECT,*(f'{k}_{a}' for k in refs for a in protocol.PERSISTENT_ARMS))
    rows = {k:[] for k in arms}
    queried, events, truth, clocks = [], [], {}, {}
    state, last, fit_steps = None, -1, 0
    starts = {sid:idx for idx,sid in enumerate(p['order'])}
    guard_ids = []
    for rank,sid in enumerate(p['order'],1):
        stamp = s['shots'][rank-1]
        file = out/'shots'/f'{rank:04d}_{sid}.npz'
        if (stamp['rank'],stamp['shot'],stamp['state_sha256'])!=(rank,sid,state) or (
                not last<=stamp['predict_start_ns']<=stamp['sealed_ns'] or r.sha(file)!=stamp['curve_sha256']):
            raise ValueError('prediction missing or used an early/reordered state')
        last = stamp['sealed_ns']
        meta,t = raw_truth(r.io23.path_for('EAST',sid),sid)
        truth[sid] = meta
        clocks[sid] = t
        with np.load(file,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['times'],t)
            curves = {k:z[k] for k in p['arms']}
        for k,folder in REFS.items():
            saved = curves_for(folder,refs[k],rank,sid)
            np.testing.assert_array_equal(saved['times'],t)
            for a in PARENT:
                np.testing.assert_allclose(curves[a],saved[a],atol=4e-6,rtol=0.)
                np.testing.assert_array_equal(curves[a]>=p['threshold'],saved[a]>=p['threshold'])
            curves.update({f'{k}_{a}':saved[a] for a in protocol.PERSISTENT_ARMS})
        for k,a in zip(DIRECT,PARENT,strict=True):
            n = protocol.FIRST if state is not None else len(t)
            np.testing.assert_array_equal(curves[k][:n],curves[a][:n])
        for k in arms:
            rows[k].append(score(t,curves[k],meta,p['threshold']))
        if rank%p['block']:
            continue
        block = rank//p['block']
        paths = block_files(out,block)
        old = block_files(LOCAL,block)
        commit,reveal,update = map(r.read,paths)
        old_commit,old_reveal,old_update = map(r.read,old)
        hashes = dict(zip(('commit_sha256','reveals_sha256','update_sha256'),map(r.sha,paths),strict=True))
        if (hashes!=s['blocks'][block-1] or commit['selected']!=old_commit['selected']
                or commit['rank_end']!=rank or update['first_next_rank']!=old_update['first_next_rank']
                or commit['curve_sha256']!=[x['curve_sha256'] for x in s['shots'][rank-p['block']:rank]]
                or not last<=commit['committed_ns']<=update['updated_ns']
                or update['state_before']!=state or commit['state_sha256']!=state
                or reveal['commit_sha256']!=hashes['commit_sha256']
                or update['commit_sha256']!=hashes['commit_sha256']
                or update['reveal_sha256']!=hashes['reveals_sha256']
                or [x['shot'] for x in reveal['outcomes']]!=commit['selected']):
            raise ValueError('direct label commit differs from sealed chronology')
        last = commit['committed_ns']
        for label,other in zip(reveal['outcomes'],old_reveal['outcomes'],strict=True):
            meta = {k:label[k] for k in ('shot','disruptive','down')}
            if (meta!=truth[label['shot']] or any(meta[k]!=other[k] for k in meta)
                    or not last<=label['read_start_ns']<=label['read_end_ns']<=update['updated_ns']):
                raise ValueError('direct label was not the agreed post-commit raw label')
            last = label['read_end_ns']
        if block==1:
            for flag in (True,False):
                guard_ids.extend([qid for qid in commit['selected'] if truth[qid]['disruptive']==flag][:4])
            guard_ids.sort(key=starts.__getitem__)
        if 'fit' in update:
            fit = update['fit']
            if fit['steps']!=old_update['fit']['steps'] or update['teaching_indices']!=old_update['teaching_indices']:
                raise ValueError('direct fit differs from the fixed sample/step budget')
            fit_steps += fit['steps']
            guards = [*guard_ids,*queried[-2:]]
            [qid] = commit['selected']
            if update['guard_ids']!=guards:
                raise ValueError('direct transfer guard had a different label exposure')
            block_dir = out/'blocks'
            prop = block_dir/f'{block:02d}_proposal.npz'
            evidence = block_dir/f'{block:02d}_candidate_curves.npz'
            if r.sha(prop)!=update['proposal_sha256'] or r.sha(evidence)!=update['witness_sha256']:
                raise ValueError('direct transfer guard or fitted vector changed')
            with np.load(prop,allow_pickle=False) as w:
                candidate = w['delta']
                if candidate.shape!=(memory.ACTION_WIDTH,) or not np.isfinite(candidate).all() or (
                        np.linalg.norm(candidate.astype(np.float64))>memory.EPISODE_RADIUS+1e-6):
                    raise ValueError('candidate outside the declared action radius')
            observed = []
            with np.load(evidence,allow_pickle=False) as z:
                for q in (qid,*guards):
                    t = clocks[q]
                    y,down = (truth[q][k] for k in ('disruptive','down'))
                    np.testing.assert_array_equal(z[f'{q}_times'],t)
                    before,after = ({k:z[f'{q}_{k}_{suffix}'] for k in protocol.PERSISTENT_ARMS}
                                    for suffix in ('before','after'))
                    for k in protocol.PERSISTENT_ARMS:
                        np.testing.assert_array_equal(before[k][:protocol.FIRST],after[k][:protocol.FIRST])
                        observed.extend(dict(shot=q,arm=k,reason=v) for v in
                            protocol.guard_violations(t,before[k],after[k],y,down,p['threshold']))
                    if q==qid:
                        idx = starts[q]
                        with np.load(out/'shots'/f'{idx+1:04d}_{q}.npz',allow_pickle=False) as saved:
                            for k,arm in zip(protocol.PERSISTENT_ARMS,DIRECT,strict=True):
                                np.testing.assert_array_equal(before[k],saved[arm])
                        helpful,diag = protocol.query_progress(t,before,after,y,down,p['threshold'])
            expected = 'accepted' if helpful and not observed else ('rejected_guard' if observed else
                                                                    'no_query_progress')
            if (observed!=update['violations'] or diag!=update['diagnostics'] or expected!=update['decision']):
                raise ValueError('guard replay disagrees with the committed update')
        elif 'fit' in old_update:
            raise ValueError('missed required matched fit')
        if update['decision']=='accepted':
            path = out/'blocks'/f'{block:02d}_state.npz'
            if r.sha(path)!=update['state_after']:
                raise ValueError('missing accepted vector')
            with np.load(path,allow_pickle=False) as w:
                np.testing.assert_array_equal(w['delta'],candidate)
        elif update['state_after']!=state:
            raise ValueError('rejected or absent direct update changed state')
        if block>1:
            queried.extend(commit['selected'])
        state, last = update['state_after'], update['updated_ns']
        events.append({k:update[k] for k in ('block','first_next_rank','decision')})
    if len(set(queried))!=p['query'] or p['cold']!=20 or sum(a['disruptive'] for a in truth.values())!=180:
        raise ValueError('fixed query or cohort size differs')
    sets = dict(full360=range(360), after20=range(20,360), last180=range(180,360),
                unqueried_after20=[j for j in range(20,360) if p['order'][j] not in queried])
    sections = {}
    for label, idx in sets.items():
        d = {k:[v[j] for j in idx] for k,v in rows.items()}
        contrasts = [(DIRECT[0],PARENT[0]), (DIRECT[1],PARENT[1]), (DIRECT[1],DIRECT[0]),
                     *((DIRECT[1],f'{k}_both') for k in refs)]
        sections[label] = dict(metrics={k:counts(v) for k,v in d.items()},
            paired={f'{a}_vs_{b}':pair(d[a],d[b]) for a,b in contrasts},
            paired_30_to_3000={f'{a}_vs_{b}':pair(d[a],d[b],True) for a,b in contrasts})
    lead = {k:summarize(warning_rows(v)) for k,v in rows.items()}
    r.write_new(out/'per_shot.json',rows)
    r.write_new(out/'result.json',dict(status='COMPLETE_MATCHED_DIRECT_TRANSFER_DEVELOPMENT',
        raw_verified=len(arms)*360, training_steps=fit_steps, queried=queried, events=events,
        fixed_threshold=p['threshold'], sections=sections, lead_sensitivity=lead,
        guard_recount=True))
    print({k:(a['tp'],a['fp'],a['tp_30_to_3000']) for k,a in sections['full360']['metrics'].items()},flush=True)


if __name__=='__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('out',type=Path)
    cli.add_argument('--score-only',action='store_true')
    args = cli.parse_args()
    if args.score_only:
        score_run(args.out.resolve())
    else:
        run(args.out.resolve())
