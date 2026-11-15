"""Require matched J-TEXT TEST evidence before an EAST performance replay.

This is a development acceptance gate, not independent generalization proof.
Artifact integrity failures raise; an explicit diagnostic override may bypass
performance failure, never missing or inconsistent evidence.
"""
import hashlib
import json
import math
from pathlib import Path


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def qualify_source(checkpoint):
    checkpoint = Path(checkpoint)
    report_path = checkpoint.parent/'test220.json'
    report = json.loads(report_path.read_text())
    hashes = json.loads((checkpoint.parent/'hashes.json').read_text())
    checkpoint_hash, report_hash = _hash(checkpoint), _hash(report_path)
    if (checkpoint_hash != hashes['source_pt_sha256'] or
            checkpoint_hash != report['provenance']['checkpoint_sha256'] or
            report_hash != hashes['test220_sha256']):
        raise ValueError('source acceptance artifact hash mismatch')
    metrics = report['jtext']
    rows = metrics['predictions']
    ids = [row['shot'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source test shot')
    tp = fp = nd = ns = 0
    for row in rows:
        if row['is_disrupt']:
            nd += 1
            lead = row['lead_ms']
            if row['alarm']:
                if lead is None or not math.isfinite(lead):
                    raise ValueError('source disruptive alarm lacks finite lead')
                timely = round(lead, 9) > 10.
            else:
                if lead is not None:
                    raise ValueError('source non-alarm has lead')
                timely = False
            if row['timely'] != timely:
                raise ValueError('source timely definition does not match strict >10ms')
            tp += int(timely)
        else:
            ns += 1
            if row['timely'] or row['lead_ms'] is not None:
                raise ValueError('safe source shot has disruption timing')
            fp += int(row['alarm'])
    if any(metrics[key] != value for key, value in
           dict(tp=tp, fp=fp, n_disrupt=nd, n_safe=ns).items()):
        raise ValueError('source counts differ from per-shot evidence')
    tpr, fpr = (tp/nd if nd else None), (fp/ns if ns else None)
    for name, value in (('tpr', tpr), ('fpr', fpr)):
        reported = metrics[name]
        if value is None or reported is None or not math.isclose(reported, value, abs_tol=1e-12):
            raise ValueError('source rates differ from per-shot evidence')
    reasons = []
    if len(rows) != 220 or nd != 110 or ns != 110 or report['provenance']['selected_test_count'] != 220:
        reasons.append('not_full_balanced_TEST220')
    if 'SMOKE' in str(report.get('evidence', '')).upper():
        reasons.append('smoke_source')
    if tpr < .9:
        reasons.append('TPR_below_0.90')
    if fpr >= .2:
        reasons.append('FPR_not_below_0.20')
    return dict(qualified=not reasons, reasons=reasons, tpr=tpr, fpr=fpr,
                lead_rule='round lead_ms to 9 decimals then >10',
                checkpoint_sha256=checkpoint_hash, report_sha256=report_hash,
                evaluation_set='J-TEXT TEST220 development acceptance')
