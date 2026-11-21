"""Four predeclared single-coordinate changes using ended cold20 statistics."""
import numpy as np

NAMES = ('cold', 'n1_gain', 'CIII_gain', 'dx_shift', 'dy_shift', 'identical')
SPECS = {'n1_gain':(11,'gain'), 'CIII_gain':(1,'gain'),
         'dx_shift':(7,'offset'), 'dy_shift':(8,'offset')}


def fit_maps(parent, order):
    source, target = parent['source_statistics'], parent['target_statistics']
    if (len(order)!=360 or len(set(order))!=360 or parent['policy']!='density_gain'
            or parent['calibration_ids']!=order[:20] or target['shot_ids']!=order[:20]
            or source['device']!='J-TEXT' or target['device']!='EAST'):
        raise ValueError('exact ended cold20 and source statistics required')
    result = {}
    for name in NAMES:
        gain, offset = np.ones(23), np.zeros(23)
        support = None
        if name in SPECS:
            c, kind = SPECS[name]
            s, t = source['median'][c], target['median'][c]
            support = dict(source_records=source['record_counts'][c],
                           target_records=target['record_counts'][c])
            if (min(support.values())<1 or s is None or t is None
                    or not np.isfinite([s,t]).all()):
                raise ValueError('observed finite source/target medians required')
            if kind=='gain':
                if min(s,t)<=0:
                    raise ValueError('positive medians required for a positive gain')
                gain[c] = s/t
            else:
                offset[c] = s-t
        if not np.isfinite(gain).all() or not np.isfinite(offset).all() or (gain<=0).any():
            raise ValueError('finite positive affine coefficients required')
        result[name] = dict(gain=gain.tolist(), offset=offset.tolist(), support=support)
    return result


def apply_map(signal, mapping):
    if set(signal)!={'values','available'}:
        raise ValueError('only values and availability allowed')
    x, mask = signal['values'], signal['available']
    if (x.ndim!=2 or x.shape[1]!=23 or x.dtype!=np.float32
            or mask.shape!=x.shape or mask.dtype!=bool or not np.isfinite(x[mask]).all()):
        raise ValueError('finite observed float32 Nx23 and matching bool mask required')
    gain, offset = np.asarray(mapping['gain']), np.asarray(mapping['offset'])
    if (gain.shape!=(23,) or offset.shape!=(23,) or not np.isfinite(gain).all()
            or not np.isfinite(offset).all() or (gain<=0).any()):
        raise ValueError('finite positive affine coefficients required')
    out = x.copy()
    for c in np.flatnonzero((gain!=1)|(offset!=0)):
        valid = mask[:,c]
        values = x[valid,c].astype(np.float64)*gain[c]+offset[c]
        if not np.isfinite(values).all() or (abs(values)>np.finfo(np.float32).max).any():
            raise FloatingPointError('input change is not representable')
        out[valid,c] = values.astype(np.float32)
    return dict(values=out, available=mask.copy())


def chunks(length):
    if type(length) is not int or length<1:
        raise ValueError('positive integer length required')
    cursor, event = 0, 127
    while cursor<length:
        stop = min(cursor+32,event+1,length)
        yield cursor,stop
        cursor = stop
        if cursor==event+1:
            event += 25
