"""Post-hoc redundancy check, not physical calibration or a feature deletion."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import io23
from ..simple_online import _durable_json
from ..simple_train import sha
from .diagnostic import read_signal


def audit(root):
    output = root/'derived_relations.json'
    if output.exists(): raise FileExistsError(output)
    rows = []
    for entry in json.loads((root/'per_shot.json').read_text()):
        sid = entry['shot']; raw, s = read_signal(sid); x, a = s['values'], s['available']
        valid = a[:, 0] & a[:, 9] & a[:, 13] & (x[:, 0] != 0)
        ratio = x[valid, 13].astype('float64')*x[valid, 9]/x[valid, 0]
        if not len(ratio) or not np.isfinite(ratio).all(): raise ValueError('no finite qa*ip/Bt relation')
        median = float(np.median(ratio))
        if median == 0: raise ValueError('zero median cannot define relative redundancy')
        # Physical units and geometry are not inferred from this identity.
        rows.append(dict(shot=sid, measured_points=len(ratio), excluded_points=int((~valid).sum()),
            median_ratio=median, q05=float(np.quantile(ratio, .05)), q95=float(np.quantile(ratio, .95)),
            relative_abs_error_q95=float(np.quantile(abs(ratio/median-1), .95)),
            hdf5_sha256=sha(io23.path_for('EAST', sid))))
    result = dict(scope='POSTHOC_20_SHOT_NUMERICAL_DEPENDENCE_NOT_CALIBRATION',
        code_sha256=sha(Path(__file__)), formula='qa_proxy * ip / Bt', rows=rows,
        physical_units_verified=False, feature_deletions=0, model_changes=0,
        median_relative_error_q95=float(np.median([r['relative_abs_error_q95'] for r in rows])),
        max_relative_error_q95=max(r['relative_abs_error_q95'] for r in rows))
    _durable_json(output, result)
    print({k: v for k, v in result.items() if k != 'rows'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); audit(p.parse_args().root)
