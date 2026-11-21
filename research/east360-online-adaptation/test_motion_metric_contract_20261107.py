from run_motion_metric_20261107 import teaching_group


def test_source_contract_does_not_use_east_warning_time():
    assert teaching_group(True, 20., 25., "source") == "timely"
    assert teaching_group(True, 10., 25., "source") is None
    assert teaching_group(True, 80., 25., "source") == "earlier"
    assert teaching_group(False, None, 25., "source") == "safe"


def test_advanced_time_extends_source_window_without_becoming_an_outcome():
    assert teaching_group(True, 150., 200., "source") == "timely"
    assert teaching_group(False, None, 200., "source") == "safe"


def test_auxiliary125_is_a_distinct_teaching_target_not_relabeling():
    assert teaching_group(True, 20., 25., "aux125") is None
    assert teaching_group(True, 80., 25., "aux125") == "timely"
    assert teaching_group(True, 150., 200., "aux125") == "earlier"
