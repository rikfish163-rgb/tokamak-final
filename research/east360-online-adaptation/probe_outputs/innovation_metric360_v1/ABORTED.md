# Incomplete infrastructure run, not an experimental result

The main process stopped this run by SIGINT (exit130). Last observed progress
was47/360 at62.31seconds; no completed.json was produced. Partial curves and
their receipts are retained, not promoted or merged into another run.

The driver globbed every package .py into the immutable inference snapshot,
including the independently edited, unused CPU verifier. The verifier hash in
the snapshot was0a1ae0ab07d436291753db998662c1274bd12630106d27904805acd94a6c103d;
its observed current hash became
d104b2c4c5e9a602ef3f6f1b299b59b6e5ca7c51a170e1ccd268dedb7cdfadb8.
Thus the run could not pass its end-of-run snapshot consistency check.

The repair binds the entry point, plan and actually imported experiment
dependencies, not unrelated sidecar code. The fresh v2 uses IDENTICAL learned
heads, source/cold/input map, scientific parameters, threshold and shot order.
No partial performance was used to choose a different method or parameter.
