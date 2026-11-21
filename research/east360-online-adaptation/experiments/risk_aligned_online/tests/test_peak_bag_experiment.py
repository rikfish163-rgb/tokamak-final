import json
import sys
from types import SimpleNamespace

import pytest

from experiments.risk_aligned_online import peak_bag_experiment as driver


@pytest.mark.parametrize('arm',driver.PEAK_ARMS)
def test_all_arm_readers_use_runner_seal_directory(tmp_path,monkeypatch,arm):
    ref=tmp_path/'reference';ref.mkdir()
    (ref/'order.json').write_text(json.dumps([{'shot':i} for i in range(31)]))
    monkeypatch.setattr(driver,'REFERENCE',ref)
    monkeypatch.setattr(driver,'load_parent',lambda _: (None,dict(
        threshold=.7,source_center=None,source_scale=None)))
    fake=SimpleNamespace();fake.to=lambda _:fake;fake.eval=lambda:fake
    monkeypatch.setattr(driver,'DecoderFfnModel',lambda _:fake)
    monkeypatch.setattr(driver,'RollingDecoder',lambda *a,**kw:None)
    monkeypatch.setattr(driver,'BagRollingDecoder',lambda *a,**kw:None)
    monkeypatch.setattr(driver,'EndedDensityReplay',lambda *a,**kw:SimpleNamespace(**kw))
    class Checked(Exception):pass
    def check(model,*,output_dir,data_api,**kw):
        assert output_dir==data_api.output_dir
        raise Checked
    monkeypatch.setattr(driver,'run_arm',check)
    monkeypatch.setattr(sys,'argv',['driver','--output',str(tmp_path/'arm'),
                                  '--arm',arm,'--limit','31'])
    with pytest.raises(Checked):driver.main()
