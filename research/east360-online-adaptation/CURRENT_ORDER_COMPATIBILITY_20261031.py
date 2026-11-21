"""Verify the unchanged context8_v1 numerical path against its archived class."""
import hashlib
import json
from pathlib import Path
import types
import zipfile

import torch

from experiments.risk_aligned_online.context_memory_20261028.model import ContextMemory


def main():
    torch.set_num_threads(1)
    root = Path(__file__).parent
    ref = root/'experiments/risk_aligned_online/runs/lifecycle_context8360_service_20261028'
    p = json.loads((ref/'protocol.json').read_text())
    name = 'experiments/risk_aligned_online/context_memory_20261028/model.py'
    with zipfile.ZipFile(ref/'code_snapshot.zip') as z:
        code = z.read(name)
    assert hashlib.sha256(code).hexdigest()==p['code_sha256'][name]
    old = types.ModuleType('archived_context8_for_audit')
    old.__package__ = 'experiments.risk_aligned_online.context_memory_20261028'
    exec(compile(code,'trusted-code-snapshot!'+name,'exec'),old.__dict__)
    path = Path(p['source_memory']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==p['source_memory']['sha256']
    weights = torch.load(path,map_location='cpu',weights_only=True)
    assert weights['profile']=='context8_v1'
    previous, current = old.ContextMemory(64),ContextMemory(64)
    for m in (previous,current):
        m.load_state_dict(weights['memory'],strict=True)
    torch.manual_seed(91)
    checks = []
    for length in (8,32,130):
        means = torch.randn(length,23)
        mask = torch.rand(length,23)>.2
        norm = previous.trace(means,mask,mode='memory')['norm'].detach()
        for mode in ('memory','off','local','hold','reverse'):
            a = previous.trace(means,mask,mode=mode,reference_norm=norm,return_before=True)
            b = current.trace(means,mask,mode=mode,reference_norm=norm,return_before=True)
            assert set(a)==set(b)
            for key in a:
                if isinstance(a[key],torch.Tensor):
                    assert torch.equal(a[key],b[key]),(length,mode,key)
                else:
                    assert a[key]==b[key]
            previous.zero_grad(set_to_none=True)
            current.zero_grad(set_to_none=True)
            for result in (a,b):
                (result['forecast_loss']+result['delta'].square().mean()).backward()
            nonfinite = {}
            for (ka,va),(kb,vb) in zip(previous.named_parameters(),current.named_parameters(),strict=True):
                assert ka==kb
                if va.grad is None or vb.grad is None:
                    assert va.grad is None and vb.grad is None
                    continue
                finite = torch.isfinite(va.grad)
                assert torch.equal(finite,torch.isfinite(vb.grad))
                assert torch.equal(va.grad[finite],vb.grad[finite])
                for condition in (torch.isnan,torch.isposinf,torch.isneginf):
                    assert torch.equal(condition(va.grad),condition(vb.grad))
                if not finite.all():
                    nonfinite[ka] = int((~finite).sum())
            # Explicit negative finding, not an equal-NaN success claim: the
            # norm-matched controls have an undefined backward at zero norm.
            # EAST memory_delta runs ALL traces under no_grad and detaches
            # their outputs; no control-trace backward enters its update().
            if nonfinite:
                assert mode in ('hold','local','reverse')
            checks.append(dict(length=length,mode=mode,forward_exact=True,
                               gradient_finite_and_exact=not nonfinite,
                               shared_control_nonfinite_gradients=nonfinite))
    result = dict(complete=True,source_memory_sha256=p['source_memory']['sha256'],
                  archived_context_code_sha256=p['code_sha256'][name],
                  current_context_code_sha256=hashlib.sha256((root/name).read_bytes()).hexdigest(),
                  scope='normal context8_v1 path, synthetic signals and masks, not a full historical training replay',
                  checks=checks)
    with Path(__file__).with_suffix('.json').open('x') as f:
        json.dump(result,f,indent=2)
    print('Exact forwards:',len(checks),'finite exact backwards:',
          sum(r['gradient_finite_and_exact'] for r in checks),
          'shared inference-only control backward limitations:',
          [(r['length'],r['mode']) for r in checks if r['shared_control_nonfinite_gradients']])


if __name__=='__main__':
    main()
