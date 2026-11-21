from experiments.risk_aligned_online.commissioning_seed_replay_20261111 import capacity


def test_postseal_capacity_never_spends_cold_start_false_alarms_twice():
    rows = [
        dict(shot=1, disruptive=False, alarm=True, timely=False, maximum=.4),
        dict(shot=2, disruptive=True, alarm=False, timely=False, maximum=.99),
        dict(shot=3, disruptive=False, alarm=False, timely=False, maximum=.95),
        dict(shot=4, disruptive=False, alarm=False, timely=False, maximum=.91),
        dict(shot=5, disruptive=True, alarm=False, timely=False, maximum=.97),
        dict(shot=6, disruptive=True, alarm=False, timely=False, maximum=.92),
    ]
    assert capacity(rows, 2, .20) is None
    r = capacity(rows, 2, 1 / 3)
    assert r['fp'] == 1 and r['tp'] == 1
    assert r['timely_shots'] == [5] and r['false_alarm_shots'] == [1]
