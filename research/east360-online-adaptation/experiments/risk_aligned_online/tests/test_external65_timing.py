import numpy as np
from ..external65_20261023.timing import first_alarm


def test_first_alarm_is_relative_to_record_and_boundary_is_not_past():
    p=np.zeros(300);p[255]=.9
    assert first_alarm(p,.7,256)==dict(index=255,before_update=True)
    p[255]=0;p[256]=.9
    assert first_alarm(p,.7,256)==dict(index=256,before_update=False)
    assert first_alarm(np.zeros(300),.7,256)==dict(index=None,before_update=False)
