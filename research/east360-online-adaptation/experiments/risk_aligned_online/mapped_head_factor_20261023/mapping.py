"""Fixed single-coordinate statistical map, never a certified unit conversion."""
from ..warmstart_alignment_20261022.alignment import fit_statistics,make_map,transform


def fit_mapping(source,target,*,source_ids,target_ids):
    if (len(target_ids)!=20 or len(set(target_ids))!=20
            or [s['shot'] for s in target]!=list(target_ids)
            or [s['shot'] for s in source]!=list(source_ids)):
        raise ValueError('exact source TRAIN and fixed twenty target identities required')
    a=fit_statistics(source,device='J-TEXT',allowed_ids=source_ids)
    b=fit_statistics(target,device='EAST',allowed_ids=target_ids)
    mapping=make_map(a,b,'density_gain')
    if mapping['channels'][12]['status']!='mapped':
        raise ValueError('ne_nG statistics required; no identity substitute for this contrast')
    return mapping


def map_record(record,mapping):
    signal=transform({k:record[k] for k in ('values','available')},mapping)
    return dict(record,**signal)
