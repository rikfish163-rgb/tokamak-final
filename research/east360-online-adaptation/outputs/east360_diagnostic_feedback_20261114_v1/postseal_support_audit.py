"""CPU-only, read-only audit of sealed H25 packets (outputs outside project)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from experiments.risk_aligned_online import io23
from experiments.risk_aligned_online.causal_input import CIII_MISSING_VALUE, CIII_MISSING_ATOL

ROOT = io23.ROOT
RUN = ROOT / 'outputs/east360_latent_teacher_20261114_v3_value_scope'
NAMES = ('Mir_avg_amp', 'Mir_avg_fre', 'CIII', 'sxr_mean', 'sxr_kurt',
         'xuv_kurt', 'xuv_ratio', 'Mir_VV')
CORE = ('dx_a', 'dy_a', 'n=1 amplitude', 'sxr_skew', 'sxr_var',
        'v_loop', 'xuv_skew', 'xuv_var')
MIR = ('Mir_avg_amp', 'Mir_avg_fre')
ALL = NAMES + CORE
LOOKUP = {name: io23.RAW23.index(name) for name in ALL}

def read(name):
    return json.loads((RUN / name).read_text())

def score(t, s, y, down, th):
    valid = ((down-t)*1000 >= -1e-6) if y else np.ones(len(t),bool)
    hits = np.flatnonzero(valid & (s >= th))
    first = int(hits[0]) if len(hits) else None
    lead = round(float((down-t[first])*1000),9) if y and first is not None else None
    return first, lead, bool(lead is not None and lead > 30.)

def audit_packet(raw, q):
    x, avail = raw['v'], raw['a']
    y = x[q+1:q+26].astype(np.float64)
    m = avail[q+1:q+26].copy()
    ci = LOOKUP['CIII']
    m[:,ci] &= ~np.isclose(y[:,ci],CIII_MISSING_VALUE,atol=CIII_MISSING_ATOL,rtol=0.)
    ai, fi = (LOOKUP[a] for a in MIR)
    amp, fre = y[:,ai],y[:,fi]
    same_zero = bool(np.array_equal(amp == 0., fre == 0.))
    finite_mir = bool(m[:,[ai,fi]].all())
    positive = finite_mir and bool((fre > 0.).all()) and bool((amp != 0.).all())
    zero = finite_mir and bool((fre == 0.).all()) and bool((amp == 0.).all())
    # Full target values and five chronological five-sample means.
    core_bins = {}
    for name in CORE:
        i = LOOKUP[name]
        if m[:,i].all():
            core_bins[name] = y[:,i].reshape(5,5).mean(1)
    # An intentionally generous in-packet linear fit of the extra channel on
    # all eight core channels simultaneously.  This is not held-out prediction
    # and cannot establish independence from the nonlinear parent model.
    core_full = m[:, [LOOKUP[name] for name in CORE]].all()
    if core_full:
        design = y[:, [LOOKUP[name] for name in CORE]].copy()
        design -= design.mean(0,keepdims=True)
        norms = np.linalg.norm(design,axis=0)
        design = design[:,norms>0] / norms[norms>0]
    result = {}
    for name in NAMES:
        i = LOOKUP[name]
        complete = bool(m[:,i].all())
        valid = complete and (positive if name in MIR else True)
        raw_finite = bool(avail[q+1:q+26,i].all())
        raw_flat = raw_finite and bool(np.ptp(y[:,i]) == 0.)
        item = {'raw_finite25':raw_finite, 'raw_flat25':raw_flat,
                'available25':complete, 'continuous25':valid}
        if valid:
            arr = y[:,i]
            bins = arr.reshape(5,5).mean(1)
            exact_flat = bool(np.ptp(arr)==0.)
            # Scale is based only on arrived data q-127..q+25; it is not a
            # sensor noise floor or a claim of physical relevance.
            hist=x[q-127:q+26,i].astype(np.float64)
            hm=avail[q-127:q+26,i].copy()
            if name == 'CIII':
                hm &= ~np.isclose(hist,CIII_MISSING_VALUE,atol=CIII_MISSING_ATOL,rtol=0.)
            if name in MIR:
                hf=x[q-127:q+26,fi]
                hm &= avail[q-127:q+26,fi] & (hf>0.)
            scale=float(np.quantile(np.abs(hist[hm]),.95)) if hm.any() else 0.
            bdiff=float(np.ptp(bins))
            rel=bdiff/max(scale,1e-12)
            correl=[]
            for v in core_bins.values():
                if np.std(v)>0 and np.std(bins)>0:
                    c=float(np.corrcoef(bins,v)[0,1]);
                    if np.isfinite(c): correl.append(abs(c))
            centered=arr-arr.mean()
            ynorm=np.linalg.norm(centered)
            lin_resid=None
            if core_full and ynorm>0:
                target=centered/ynorm
                beta=np.linalg.lstsq(design,target,rcond=1e-6)[0]
                lin_resid=float(np.linalg.norm(target-design@beta))
            item.update(exact_flat=exact_flat, bin_range=bdiff, rel=rel,
                        change_1pct=bool(rel>=.01), change_0_1pct=bool(rel>=.001),
                        best_core_corr=max(correl) if correl else None,
                        core_linear_residual=lin_resid)
        result[name]=item
    return {'same_zero':same_zero, 'mir_positive':positive, 'mir_zero':zero,
            'mir_mixed_or_other':finite_mir and not (positive or zero), 'fields':result}

def summarize(rows):
    info={'packets':len(rows), 'shots':len({r['shot'] for r in rows}), 'fields':{}}
    for name in NAMES:
        total=[r for r in rows if r['audit']['fields'][name]['continuous25']]
        nontriv=[r for r in total if r['audit']['fields'][name]['change_1pct']]
        weak=[r for r in total if r['audit']['fields'][name]['change_0_1pct']]
        decor=[r for r in nontriv if r['audit']['fields'][name]['best_core_corr'] is not None
               and r['audit']['fields'][name]['best_core_corr'] < .90]
        residual=[r for r in nontriv if r['audit']['fields'][name]['core_linear_residual'] is not None
                  and r['audit']['fields'][name]['core_linear_residual'] >= .5]
        info['fields'][name]={'complete_packets':len(total),
          'complete_shots':len({r['shot'] for r in total}),
          'raw_finite_packets':sum(r['audit']['fields'][name]['raw_finite25'] for r in rows),
          'raw_flat_packets':sum(r['audit']['fields'][name]['raw_flat25'] for r in rows),
          'constant_packets':sum(r['audit']['fields'][name]['exact_flat'] for r in total),
          'constant_shots':len({r['shot'] for r in total if r['audit']['fields'][name]['exact_flat']}),
          'change_1pct_packets':len(nontriv), 'change_1pct_shots':len({r['shot'] for r in nontriv}),
          'change_0_1pct_packets':len(weak), 'change_0_1pct_shots':len({r['shot'] for r in weak}),
          'less_correlated_packets':len(decor), 'less_correlated_shots':len({r['shot'] for r in decor}),
          'core_linear_residual_ge_half_packets':len(residual),
          'core_linear_residual_ge_half_shots':len({r['shot'] for r in residual})}
    for kind in ('same_zero','mir_positive','mir_zero','mir_mixed_or_other'):
        hit=[r for r in rows if r['audit'][kind]]
        info[kind]={'packets':len(hit),'shots':len({r['shot'] for r in hit})}
    return info

def main():
    protocol,seal,per = (read(x) for x in ('protocol.json','seal.json','per_shot.json'))
    order=protocol['order']; tab=per['0.2']; threshold=protocol['calibration']['0.2']
    assert len(order)==360 and len(set(order))==360 and protocol['startup']==20
    assert seal['shots']==[f'{rank:04}_{sid}.npz' for rank,sid in enumerate(order,1)]
    assert protocol['clock']=={'horizon':25,'release':26,'offset':31,'handoff':8,
                               'dose':'same archived cap times a common transition; no persisted state'}
    allrows=[]; partrows=[]; cases=[]; counts=defaultdict(int)
    for rank,sid in enumerate(order,1):
        if rank<=20: continue
        a,b=tab['current_5'][rank-1],tab['bias_only'][rank-1]
        assert a['shot']==sid==b['shot'] and a['disruptive']==b['disruptive']
        if a['disruptive'] and not a['timely']:
            kind='late' if a['alarm'] else 'never'
        elif not a['disruptive'] and b['alarm']:
            kind='fp'
        else: continue
        y,down=io23.read_outcome('EAST',sid)
        assert y==a['disruptive'] and down==a['down_s']==b['down_s']
        raw=io23.read_signals('EAST',sid,rate_contract='nominal'); t=raw['times']
        with np.load(RUN/'shots'/seal['shots'][rank-1],allow_pickle=False) as z:
            assert len(t)==len(z['dose'])==len(z['current_5'])==len(z['bias_only'])
            first_old,lead_old,timely_old=score(t,z['current_5'],y,down,threshold)
            first_parent,lead_parent,timely_parent=score(t,z['bias_only'],y,down,threshold)
            for row,first,lead,timely in ((a,first_old,lead_old,timely_old),
                                          (b,first_parent,lead_parent,timely_parent)):
                assert row['alarm']==(first is not None) and row['lead_ms']==lead and row['timely']==timely
            q=z['q']; r=q+39; eligible=z['eligible']
            assert len(q)==len(eligible) and np.array_equal(q,np.arange(127,127+32*len(q),32))
            safe=np.flatnonzero(eligible & (r<len(t)))
            safe=safe[z['dose'][r[safe]]>0.]
            if y:
                ms=np.round((down-t[r[safe]])*1000,9)
                safe=safe[(ms>30.) & (ms<=125.)]
            else:
                assert first_parent is not None
                safe=safe[r[safe]<=first_parent]
            # Count extra opportunities before complete handoff.  The chosen
            # slot follows the archived action_clock exactly: if a preceding
            # slot is eligible, new direction is chosen at f>=1/2 and the
            # middle point has zero common dose.  Never credit another slot's
            # nonzero dose to the current packet.
            full_set=set(safe.tolist()); partial=[]
            for ix in np.flatnonzero(eligible):
                if int(ix) in full_set: continue
                origin=int(q[ix]); older=bool(ix>0 and eligible[ix-1])
                for at in range(origin+32,min(origin+39,len(t))):
                    f=(at-(origin+31))/8.
                    if older and f<.5: continue
                    if z['dose'][at]<=0.: continue
                    in_time=(30.<round(float((down-t[at])*1000),9)<=125.) if y else at<=first_parent
                    if in_time:
                        partial.append({'q':origin,'first_partial_index':at,'fraction':f,
                                        'dose':float(z['dose'][at])})
                        break
            counts[kind]+=1
            cases.append({'kind':kind,'shot':sid,'rank':rank,'outcome':int(y),'down_s':down,
                          'first_old_idx':first_old,'first_parent_idx':first_parent,
                          'n_q39':int(len(safe)), 'q39':q[safe].tolist(),
                          'last95_q39':q[safe][np.round((t[first_parent]-t[r[safe]])*1000,9)<=95].tolist() if not y else [],
                          'partial_only':partial})
            for ix in safe:
                origin=int(q[ix]); at=int(r[ix]);
                one={'kind':kind,'shot':sid,'rank':rank,'q':origin,'write':at,
                     'write_s':float(t[at]), 'dose':float(z['dose'][at]),
                     'to_cutoff_ms':float((down-t[at])*1000) if y else float((t[first_parent]-t[at])*1000),
                     'last95':bool(not y and round(float((t[first_parent]-t[at])*1000),9)<=95.),
                     'audit':audit_packet(raw,origin)}
                allrows.append(one)
            for extra in partial:
                origin,at=extra['q'],extra['first_partial_index']
                partrows.append({'kind':kind,'shot':sid,'rank':rank,'q':origin,'write':at,
                    'write_s':float(t[at]),'dose':extra['dose'],'handoff_fraction':extra['fraction'],
                    'to_cutoff_ms':float((down-t[at])*1000) if y else float((t[first_parent]-t[at])*1000),
                    'last95':bool(not y and round(float((t[first_parent]-t[at])*1000),9)<=95.),
                    'audit':audit_packet(raw,origin)})
    assert dict(counts)=={'never':12,'late':21,'fp':28},dict(counts)
    groups={kind:summarize([r for r in allrows if r['kind']==kind]) for kind in ('never','late','fp')}
    groups['fp_last95']=summarize([r for r in allrows if r['kind']=='fp' and r['last95']])
    combined=allrows+partrows
    for kind in ('never','late','fp'):
        groups[kind+'_including_partial']=summarize([r for r in combined if r['kind']==kind])
    groups['fp_last95_including_partial']=summarize([r for r in combined if r['kind']=='fp' and r['last95']])
    out={'protocol':{'run':str(RUN),'raw_template':str(io23.DATA_ROOT/'EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5'),
          'selection':'post-cold old current_5 never/late, bias_only actual false alarm; eligible q and dose[q+39]>0',
          'break':'30 < (DownTime-time[q+39])*1000 <= 125; safe q+39 <= parent first alarm',
          'metrics':'25 target points; positivity required Mir frequency and nonzero amplitude; CIII finite and outside -10+/-0.01; actual variation is 5-bin range / p95(abs(valid raw q-127:q+25)) >= 0.01, sensitivity 0.001',
          'corr':'maximum absolute Pearson correlation between five bin means and any complete nonconstant core channel; <0.9 is only lack of close single-channel shape, not conditional information or causality',
          'linear_residual':'least-squares fit of 25 target samples after centering against all 8 complete core channels centered and normalized, rcond=1e-6; report residual norm / original centered norm >=0.5. This is fitted on the same packet; no generalization, noise-floor or causality claim.'},
          'cases':cases,'summary':groups,'packets':allrows,'partial_packets':partrows}
    print(json.dumps(out,ensure_ascii=False,allow_nan=False,indent=2))

if __name__=='__main__': main()
