import pytest

from experiments.risk_aligned_online.commissioning_results import paired_changes, build_report


def row(sid, disruption, timely=False, alarm=False):
    return dict(shot=sid, is_disrupt=disruption, timely=timely, alarm=alarm)


def test_paired_changes_show_rescues_and_collateral_errors_separately():
    before = [row(1, True), row(2, True, True, True),
              row(3, False, alarm=True), row(4, False)]
    after = [row(1, True, True, True), row(2, True),
             row(3, False), row(4, False, alarm=True)]
    assert paired_changes(before, after) == dict(tp_rescued=[1], tp_lost=[2],
                                                fp_removed=[3], fp_added=[4])


def test_different_order_or_outcomes_cannot_be_called_a_paired_gain():
    with pytest.raises(ValueError, match='identities'):
        paired_changes([row(1, True)], [row(2, True)])
    with pytest.raises(ValueError, match='outcomes'):
        paired_changes([row(1, True)], [row(1, False)])


def test_changed_commissioning_manifest_cannot_be_reported(tmp_path):
    import json
    (tmp_path / 'protocol.json').write_text(json.dumps(dict(order_sha256='bad', roster_sha256='bad')))
    (tmp_path / 'order.json').write_text('[]')
    (tmp_path / 'roster.json').write_text('{}')
    with pytest.raises(ValueError, match='locked order or roster hash'):
        build_report(tmp_path)
