"""Export fresh static TEST220 evidence from an existing paired evaluation."""
import argparse
import json
from pathlib import Path

from .simple_train import sha
from .source_gate import qualify_source


def export(checkpoint,paired,*,expected_ids):
    checkpoint,paired=Path(checkpoint),Path(paired)
    destination=checkpoint.parent
    if any((destination/name).exists() for name in ('test220.json','hashes.json')):
        raise FileExistsError('preserve existing source evidence')
    data=json.loads(paired.read_text())
    digest=sha(checkpoint)
    if data['split']!='test' or data['checkpoint_sha256']!=digest:
        raise ValueError('matched TEST checkpoint required')
    metrics=data['arms']['static']['metrics']
    ids=[r['shot'] for r in metrics['predictions']]
    if len(ids)!=220 or len(set(ids))!=220 or set(ids)!=set(expected_ids):
        raise ValueError('complete fixed TEST220 roster required')
    report=dict(evidence='DEVELOPMENT_ONLY',jtext=metrics,
                provenance=dict(checkpoint_sha256=digest,selected_test_count=len(ids),
                                selection_source='J-TEXT VAL160; no test threshold selection',
                                paired_evaluation_sha256=sha(paired)))
    (destination/'test220.json').write_text(json.dumps(report,indent=2))
    (destination/'hashes.json').write_text(json.dumps(dict(source_pt_sha256=digest,
                test220_sha256=sha(destination/'test220.json')),indent=2))
    return qualify_source(checkpoint)


if __name__=='__main__':
    from .simple_data import source_roles
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--paired',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(export(args.checkpoint,args.paired,expected_ids=source_roles()['test'])))
