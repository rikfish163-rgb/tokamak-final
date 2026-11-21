from alarm_order_conditional_20261106 import replication_summary


def make(rank, tag, shot, value):
    return dict(
        rank=rank,
        shot=shot,
        class_tag=tag,
        index=128,
        logit=1.0,
        coverage=0.9,
        motion_recent=0.3,
        motion_old=0.3,
        delta_recent=value,
        delta_old=0.0,
    )


def test_conditional_ranges_use_the_actual_ids_including_early_ranks():
    rows = [
        make(1, "fp", 60010, 0.0),
        make(2, "reasonable_tp", 60020, 0.1),
        make(834, "fp", 81000, 0.0),
        make(835, "reasonable_tp", 82000, 0.2),
        make(836, "reasonable_tp", 83000, 2.0),
    ]
    x = replication_summary(rows)
    assert x["splits"]["full"]["n_pairs"] == 2
    assert x["splits"]["first_half"]["n_pairs"] == 1
    assert x["splits"]["second_half"]["n_pairs"] == 1
    assert x["splits"]["first_half"]["pairs"][0]["fp_shot"] == 60010
    assert x["splits"]["second_half"]["pairs"][0]["tp_shot"] == 82000
    assert not x["gate"]["passed"]
