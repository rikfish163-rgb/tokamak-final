"""Reproducible J-TEXT Adv25 online-active diagnostic.

This package is intentionally separate from the retained EAST V3C replay.
Its source checkpoints use ``max(25 ms, AdvancedTime)`` as the J-TEXT
positive-window contract.  The online controller is a delayed-feedback
decision-level mixture of frozen source experts; it is not an EAST parent and
does not claim Transformer representation learning.
"""

from .labels import ADV25_POLICY, FLOOR_MS, resolve_positive_horizon_ms

__all__ = [
    "ADV25_POLICY",
    "FLOOR_MS",
    "resolve_positive_horizon_ms",
]
