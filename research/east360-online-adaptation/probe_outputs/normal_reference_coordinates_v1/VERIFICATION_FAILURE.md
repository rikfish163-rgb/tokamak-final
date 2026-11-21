# First verification attempt: strict roundtrip comparison failed

2026-11-01. Main command returned exit1 at the declared 2e-6 probability tolerance: maximum observed probability difference was 3.4570693969726562e-6. No full-stream experiment was restarted or altered. This is not an input-file NaN and not a source/density baseline mismatch.

The initial verifier source is preserved as `first_verifier_executed.py`, SHA256 `48eef46a14134b97b9758130cf9238bac0a2a1037077c6fe81a9cee2b1a56187`. It stopped before writing `count_audit.json` or `independent_audit.json` and before selected no-hook neural replay. The exception was `AssertionError: ('roundtrip numerical equivalence failed', 3.4570693969726562e-06)`.

The revised verifier will retain the same 2e-6 limit and record an explicit FAILED roundtrip verdict, while allowing orthogonal counting, provenance and no-hook replay checks to finish. Overall status must remain `AUDITED_WITH_ROUNDTRIP_MISMATCH`, not PASS. No probability tolerance is relaxed and no alarm change is ignored. This diagnostic continuation must not be represented as the failed test later passing.
