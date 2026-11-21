"""Four identity-disjoint teaching folds; no held-shot mapping statistics."""
from ..warmstart_alignment_20261022.alignment import fit_statistics,make_map


def folds(ids,flags):
    ids=sorted(ids)
    if len(ids)!=20 or len(set(ids))!=20 or set(ids)!=set(flags):
        raise ValueError('exact20 distinct authorized historical identities required')
    positive=[s for s in ids if flags[s]];negative=[s for s in ids if not flags[s]]
    if len(positive)!=4 or len(negative)!=16:raise ValueError('declared4D16S history required')
    result=[]
    for i,sid in enumerate(positive):
        held=sorted([sid]+negative[i::4]);train=[s for s in ids if s not in held]
        result.append(dict(fold=i,train=train,held=held))
    return result


def fit_fold_map(source_statistics,records,train_ids):
    if len(train_ids)!=15 or len(set(train_ids))!=15:raise ValueError('exact15 training IDs required')
    lookup={r['shot']:r for r in records}
    if len(lookup)!=len(records) or not set(train_ids)<=set(lookup):raise ValueError('unique available training records required')
    target=fit_statistics([lookup[s] for s in train_ids],device='EAST',allowed_ids=train_ids)
    mapping=make_map(source_statistics,target,'density_gain')
    if mapping['channels'][12]['status']!='mapped':raise ValueError('observed positive density medians required')
    return mapping
