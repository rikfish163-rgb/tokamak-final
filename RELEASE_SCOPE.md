# Release scope

## Included

- The Python modules imported by the J-TEXT source evaluator and the locked
  A1 → dynamic adapter → unified online runner.
- The exact split manifests and HDF5 data contract used by those modules.
- The reproducibility CLI and tests.
- The paper-grade machine evidence, compact prediction/query/reveal ledgers,
  causal-random-query, EAST-only and fixed-shape selector-roster controls,
  publication figures and compiled manuscript.
- A portable per-file SHA-256 manifest for all 2,203 required HDF5 inputs. Raw
  HDF5 bytes remain external.
- The three small, hash-bound J-TEXT parent checkpoints required to reproduce
  the current v3c target chain.
- The three fidelity-verified post-support EAST target-parent states used to
  start static and online branches from identical weights and thresholds.
- The twelve small final online states from the fixed-shape selector-roster
  control. They are evidence-bearing outputs bound by the compact artifact
  manifest, not additional source/target parents.
- Method documentation needed to interpret the results without treating the 360-shot stream as an untouched final test.
- The compact 2026-08-31 current-worktree diagnostic receipt under
  `results/current_mainline_20260831/`, including its source-profile separation,
  chronological ledgers, attribution summaries and compiled Chinese reports.

## Excluded

- Raw HDF5 data and derived NPZ/NPY feature arrays.
- PyTorch checkpoints outside the six paper-lineage source/target parents and
  the twelve explicitly released fixed-shape-control final states; optimizer
  states and large training caches remain excluded.
- Historical failed methods, obsolete reports and non-paper generated figures.
  In particular, the parent-fidelity-invalid equal-label-budget v1 run is not
  released; the corrected, fidelity-verified v2 control is included.
- Regenerable online-development run directories and optimizer caches; their
  evidence-bearing compact ledgers and parent states are in the paper package.
- Tar archives and copied workspace backups.
- Per-shot risk-trace NPZ files and duplicated per-run feature payloads from the
  local mounted-disk replay directories; the published JSON/JSONL ledgers and
  manifests are the compact evidence surface.

The raw inputs are supplied through `data/`; the excluded training caches are
regenerable.  Bundled parent checkpoints are evidence-bearing inputs, not a
substitute for source retraining.  The repository must fail closed when
required data are absent; a committed metric summary alone never counts as a
successful reproduction.
