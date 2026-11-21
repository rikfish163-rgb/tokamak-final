from audit_layer0_signed_flips_20261106 import caliper_pairs


def test_match_maximizes_cardinality_even_when_nearest_greedy_would_lose_a_pair():
    a = [
        {"native_at_alarm": 0.50, "elapsed_s": 0.7},
        {"native_at_alarm": 0.51, "elapsed_s": 0.0},
    ]
    b = [
        {"native_at_alarm": 0.50, "elapsed_s": 0.4},
        {"native_at_alarm": 0.53, "elapsed_s": 1.0},
    ]
    assert caliper_pairs(a, b) == [(0, 1), (1, 0)]


def test_disjoint_margin_is_not_a_matched_observation():
    a = [{"native_at_alarm": 0.20, "elapsed_s": 1.0}]
    b = [{"native_at_alarm": 0.80, "elapsed_s": 1.0}]
    assert caliper_pairs(a, b) == []
