"""CPU-only VAL audit of the actual latent_training feature-cache schema.

Read stored endpoint labels, never infer missing AdvancedTime or derive
replacement labels. This audit cannot establish physical validity of an alarm.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def audit_rows(rows, threshold):
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be finite and in (0,1)')
    records, leads, timely_leads = [], [], []
    n0 = n1 = fp = total_positive = timely_positive = 0
    for row in rows:
        times = np.asarray(row['times'], np.float64)
        logits = np.asarray(row['base_logits'], np.float32)
        labels = np.asarray(row['y'])
        if (times.ndim != 1 or not len(times) or logits.shape != times.shape
                or labels.shape != times.shape or not np.isfinite(times).all()
                or not np.isfinite(logits).all() or not np.isin(labels, (0, 1)).all()):
            raise ValueError('finite matching cached time/logit/binary-label vectors required')
        if np.any(np.diff(times) <= 0):
            raise ValueError('strictly increasing observed times required')
        # Match float32 source sigmoid. The model has no additional availability veto.
        scores = torch.from_numpy(logits).sigmoid().numpy()
        indices = np.flatnonzero(scores >= threshold)
        first = int(indices[0]) if len(indices) else None
        maximum = int(np.argmax(scores))
        disruptive = bool(row['is_disrupt'])
        lead, first_label, timely = None, None, False
        if disruptive:
            down = row['down']
            if down is None or not np.isfinite(down):
                raise ValueError('disruptive cached row requires finite down time')
            lead_vector = np.round((down-times)*1000., 9)
            positive = labels == 1
            total_positive += int(positive.sum())
            timely_positive += int((positive & (lead_vector > 10.)).sum())
            if first is not None:
                lead = float(lead_vector[first])
                first_label = int(labels[first])
                timely = lead > 10.
                leads.append(lead)
                if timely:
                    timely_leads.append(lead)
                    n0 += first_label == 0
                    n1 += first_label == 1
        safe_fp = not disruptive and first is not None
        fp += safe_fp
        records.append(dict(shot=int(row['shot']), first_alarm_index=first,
                            maximum_risk_index=maximum,
                            first_alarm_is_maximum_risk=first == maximum,
                            first_alarm_label=first_label, lead_ms=lead,
                            timely_tp=timely, safe_fp=safe_fp))
    return dict(n_rows=len(rows), threshold=float(threshold),
                timely_TP=dict(count=n0+n1, first_alarm_label0=n0, first_alarm_label1=n1),
                safeFP=fp, positive_label_windows=dict(total=total_positive,
                lead_gt_10=timely_positive, lead_le_10=total_positive-timely_positive),
                lead_distribution_ms=dict(first_alarm=leads, timely=timely_leads,
                median_timely=float(np.median(timely_leads)) if timely_leads else None), rows=records)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--threshold', required=True, type=float)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError('new output required; existing audit cannot be overwritten')
    # Only locally produced project caches, never untrusted uploads.
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    if not isinstance(cache, dict) or not {'parent_sha256', 'train', 'val'} <= cache.keys():
        raise ValueError('expected TRAIN/VAL source_features cache, never TEST')
    report = audit_rows(cache['val'], args.threshold)
    report.update(evidence='VAL160 DEVELOPMENT_ONLY; labels read as stored',
                  parent_sha256=cache['parent_sha256'], cache_sha256=sha(args.cache),
                  script_sha256=sha(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ('n_rows', 'timely_TP', 'safeFP',
                                                   'positive_label_windows')}))
    return 0


if __name__ == '__main__':
    main()
