"""Common, post-seal local scoring with a fixed primary comparison."""

from ..mask_separated_intra_20261103.pilot_core import aggregate, gate

CONFIGS = (
    "native_absolute",
    "native_centered",
    "source_decoder_absolute",
    "source_decoder_centered",
)
MODES = ("static", "real", "reverse", "hold")


def check_static_reference(rows, reference):
    expected = {
        (r["shot"], r["query"]): r["after_logit"]
        for r in reference
        if r["stem"] == "original" and r["mode"] == "static"
    }
    if set(expected) != {(r["shot"], r["query"]) for r in rows}:
        raise AssertionError("same fixed coordinates required for source static parity")
    maximum = max(
        abs(r["before_logit"] - expected[r["shot"], r["query"]]) for r in rows
    )
    if maximum != 0:
        raise AssertionError("static parent risk changed")
    return dict(n_coordinates=len(expected), max_abs_logit_difference=maximum)


def gates(grouped):
    values = {
        cfg: gate({"reference_bt_" + mode: grouped[cfg][mode] for mode in MODES})
        for cfg in CONFIGS
    }
    return dict(
        primary_configuration="native_absolute",
        primary_gate=values["native_absolute"],
        configuration_gates=values,
        operational_online_gain_measured=False,
    )


def analyze(rows):
    grouped = {
        cfg: {
            mode: aggregate(
                [r for r in rows if r["configuration"] == cfg and r["mode"] == mode]
            )
            for mode in MODES
        }
        for cfg in CONFIGS
    }
    return dict(configurations=grouped, **gates(grouped))
