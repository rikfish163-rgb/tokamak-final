"""Independently reconstruct held dense effect predictions and shot means."""

import argparse
from pathlib import Path
import numpy as np
import run_patch_utility_shadow_response_20261106 as s
import verify_patch_utility_selection_20261106 as independent
from verify_confirmation_raw_replay_20261105 import Hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-receipt", type=Path, help="optional new output; never overwrites")
    args = parser.parse_args()
    r, m, b = Path("probe_outputs"), s.m, s.b
    root = r / "patch_utility_shadow_effect_20261106_v1"
    expected = b.r.read(root/"result.json")
    models = b.r.read(root/"fits.json")["fits"]
    h = Hashes()
    for p, sha in expected["input_sha256"].items():
        assert h(p) == sha, p
    assert h(root/"fits.json")==expected["fits_sha256"] and h(root/"held_errors.npz")==expected["errors_sha256"]
    tapes, ids, _ = m.load(r/"patch_utility_bank_20261106_v1", r/"patch_utility_target_views_20261106_v1",
                           r/"patch_utility_selection_20261106_v1/result.json",h)
    folds = independent.deterministic_folds(ids)
    tol = 2e-8
    arms = tuple(expected["metrics"])
    checked = {}
    saved = np.load(root/"held_errors.npz",allow_pickle=False)
    assert np.array_equal(saved["ids"],ids) and np.array_equal(saved["folds"],folds)
    arrays = {a:[] for a in arms}
    for vi,v in enumerate(m.VIEWS):
        tape = tapes[v]
        c = tape.data["curves"].astype(np.float64)
        pref = c[:,:,:8]
        d0 = pref-pref[:,[0]]
        last = d0[:,:,7]
        truth = np.stack([(c[:,:,8+8*j:16+8*j]-c[:,[0],8+8*j:16+8*j]-last[...,None]).mean(2) for j in range(3)],axis=2)
        rows = []
        for q in tape.shots:
            for i in range(q.blocks.start,q.blocks.stop):
                clock = int(tape.data["clock"][i])
                pairs = np.maximum(np.abs(last[i,1::2]),np.abs(last[i,2::2]))
                reached = (clock+32<len(q.native) and q.native[:clock+9].max() < b.TAU)
                safe = np.r_[False, np.repeat(pairs <= b.pair.LIMIT-b.pair.GUARD_HEADROOM,2)] & tape.allowed[i]
                rows.append(safe & reached)
        keep = np.stack(rows)
        assert int(keep.sum())==expected["available_actions"][v]
        risk = pref[:,[0]]-b.TAU
        risk_centered = pref[:,[0]]-pref[:,[0]].mean(2,keepdims=True)
        d_centered = d0-d0.mean(2,keepdims=True)
        amp = np.stack([d0.mean(2), (d0**2).mean(2)**.5, (d_centered**2).mean(2)**.5,
                        d0.min(2),d0.max(2)],axis=-1)
        base = independent.features(tape.data,np.arange(len(c)),"risk",b.TAU)
        actual = truth[keep]
        owner = tape.owner
        for arm in arms:
            if arm == "hold":
                pred = np.zeros_like(truth)
            elif arm == "linear":
                pred = (last-d0[:,:,0])[...,None]*np.array([4.5,12.5,20.5])/7
            else:
                shape = d_centered if arm=="real" else d_centered[:,:,::-1] if arm=="reverse" else np.zeros_like(d_centered)
                x = np.concatenate([base,np.broadcast_to(risk,pref.shape),
                                    amp if arm!="blind" else np.zeros_like(amp),
                                    shape, shape*risk_centered],axis=-1).astype(np.float32)
                pred = np.empty_like(truth)
                for k,fit in enumerate(models[arm]):
                    mask = folds[owner]==k
                    pred[mask] = np.asarray(fit["target"]) + (
                        (x[mask]-np.asarray(fit["center"]))/np.asarray(fit["scale"])
                    )@np.asarray(fit["coefficient"])
            if arm=="real":
                maximum = float(np.max(np.abs(pred[keep]-saved[f"real_prediction__{v}"])))
                assert maximum < tol
                checked[f"real_prediction_{v}"] = maximum
            err = np.abs(pred[keep]-actual)
            counts = keep.sum(1)
            start = np.r_[0,np.cumsum(counts)]
            shot_err = np.full((len(ids),3),np.nan)
            for j,q in enumerate(tape.shots):
                block_err = [err[start[i]:start[i+1]].mean(0)
                             for i in range(q.blocks.start,q.blocks.stop) if counts[i]]
                if block_err:
                    shot_err[j] = np.asarray(block_err).mean(0)
            assert np.array_equal(np.isfinite(shot_err),np.isfinite(saved[arm][vi]))
            worst = float(np.nanmax(np.abs(shot_err-saved[arm][vi])))
            assert worst < tol
            checked[f"{arm}_{v}_error"] = worst
            arrays[arm].append(shot_err)
        print(v,int(keep.sum()),flush=True)
    for arm,scores in arrays.items():
        a = np.asarray(scores)
        denom = np.isfinite(a).sum(0)
        pooled = np.divide(np.nansum(a,axis=0),denom,out=np.full(a.shape[1:],np.nan),where=denom>0)
        arrays[arm]=pooled
        record = expected["metrics"][arm]["pooled"]
        assert abs(np.nanmean(pooled)-record["mae"])<tol
        assert np.allclose(np.nanmean(pooled,0),record["segment_mae"],atol=tol,rtol=0)
        assert int(np.isfinite(pooled[:,0]).sum())==record["shots"]
    for arm,pairs in expected["real_comparisons"].items():
        d = (arrays["real"].mean(1)-arrays[arm].mean(1))
        d = d[np.isfinite(d)]
        assert abs(d.mean()-pairs["real_minus_control"])<tol
        assert [int((d<0).sum()),int((d>0).sum()),int((d==0).sum())]==[
            pairs[k] for k in ("real_better","real_worse","tied")]
    h.check()
    receipt = dict(
        status="INDEPENDENT_DENSE_HELD_ERRORS_RECOUNT",
        source_result_sha256=h(root/"result.json"),checked_inputs=len(expected["input_sha256"]),
        actions=expected["available_actions"],maximum_differences=checked,
        metrics_recomputed=list(arms),new_val_test_east_reads=0)
    if args.write_receipt is not None:
        b.r.write_new(args.write_receipt, receipt)
    print("verified",len(expected["input_sha256"]),max(checked.values()),flush=True)


if __name__ == "__main__":
    main()
