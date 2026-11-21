from experiments.risk_aligned_online.source_anticipation_20261104.prepare import (
    inventory,
)


def test_inventory_keeps_original_and_auxiliary_labels_and_early_disruption_distinct():
    rows = [
        dict(
            shot=1,
            hdf5_disruptive=False,
            original_source_point_label=0,
            future_disruption_125ms=0,
            anticipation_stratum="safe",
        ),
        dict(
            shot=2,
            hdf5_disruptive=True,
            original_source_point_label=0,
            future_disruption_125ms=1,
            anticipation_stratum="disruptive_30_to_125ms",
        ),
        dict(
            shot=2,
            hdf5_disruptive=True,
            original_source_point_label=0,
            future_disruption_125ms=0,
            anticipation_stratum="disruptive_earlier_than_125ms",
        ),
        dict(
            shot=3,
            hdf5_disruptive=True,
            original_source_point_label=1,
            future_disruption_125ms=1,
            anticipation_stratum="disruptive_0_to_30ms",
        ),
    ]
    result = inventory(rows)
    assert result["records"] == 4 and result["shots"] == 3
    assert result["strata"]["safe"]["shots"] == 1
    assert (
        result["strata"]["disruptive_earlier_than_125ms"]["hdf_disruptive_shots"] == 1
    )
    assert result["label_pairs"] == {"0_to_0": 2, "0_to_1": 1, "1_to_0": 0, "1_to_1": 1}
