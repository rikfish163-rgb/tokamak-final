"""Label-aware, post-hoc state-capacity diagnostics; not online evaluation."""

from .optimize import probe, risk_with_shift

__all__ = ['probe', 'risk_with_shift']
