# Reproducibility protocol

## Current paper-grade lineage

The current result is the v3c same-lineage development chain:

```text
J-TEXT TRAIN1353 source fit
→ VAL160 epoch and threshold selection
→ TEST220 within-run balanced comparator
→ EAST support110 target adaptation and threshold selection
→ balanced360 static prediction and seal
→ reset to the identical transferred parent
→ balanced360 chronological online-active replay
```

The package also contains a same-label-access counterfactual. For each seed,
the exact queried shots are front-loaded into ordinary target refitting and
then excluded from that seed's evaluation complement. Three label-agnostic,
per-block count-matched random front-load rosters are retained per seed. A
separate causal random-query online replay is complete with three schedules per
seed. It matches parent lineage, label counts and configured update budgets,
but active selection produces more realized error-conditioned updates, so it
does not isolate selector quality at equal FLOPs. A separate fixed-shape roster
screen now matches 16 current and 16 support windows plus two shared and two
private optimizer attempts per query. It produces active-minus-random utility
differences of `-0.00185/0/-0.00556`. Guard acceptance makes committed updates
differ, and archived identities under a new operator do not estimate the
adaptive selector-policy effect or prove equivalence. The complete evidence is
under `paper_grade_package/evidence/equal_compute_selector_control_v1/`.

All outcome and disruption-time labels come only from HDF5
`/meta/IsDisrupt` and `/meta/DownTime`.  J-TEXT TEST220 is non-chronological
and outcome balanced.  EAST support110 is 15 disruptive/95 safe by HDF5 and
is reused for target fitting and threshold selection.  balanced360 was used
for method development and 972-candidate screening.  Consequently the release
status is `DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_CONFIRMATION`, not an untouched
test or deployment result.

The authoritative package documentation is:

```text
experiments/transfer_methods_2026/paper_grade_package/README.md
experiments/transfer_methods_2026/paper_grade_package/NATURE_PORTFOLIO_READINESS.md
```

## Data and environment

Use `rl_env` and point `FUSION_DATA_ROOT` to a directory containing:

```text
J-TEXT/unified_hdf5/<shot//100*100>/<shot>.hdf5
EAST/unified_hdf5/<shot//100*100>/<shot>.hdf5
```

For this workstation the verified environment is PyTorch 2.11.0+cu128 on an
RTX 5070 Laptop GPU.  Raw HDF5 data are external to Git.  Generated ledgers
should be written to an ext4 location; high-frequency fsync on ntfs3 can block.

## Rebuild the paper evidence

From the repository root:

```bash
export FUSION_DATA_ROOT=/absolute/path/to/data
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/build_evidence.py
conda run -n rl_env python \
  experiments/transfer_methods_2026/build_equal_label_budget_report.py
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/generate_latex_tables.py
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/generate_figures.py
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/validate_paper_package.py
```

The evidence builder recomputes labels and disruption times from HDF5,
recomputes J-TEXT lead from the archived alarm endpoint, checks source and
target per-shot ledgers, and regenerates tables.  It does not retrain or tune a
model.

The portable content inventory
`experiments/transfer_methods_2026/paper_grade_package/evidence/required_hdf5_content_manifest.json`
contains SHA-256, byte size, shot id and relative path for all 2,203 required
HDF5 files (1,733 J-TEXT and 470 EAST). Rebuild it with:

```bash
conda run -n rl_env python -m reproducibility.cli audit \
  --data-root "$FUSION_DATA_ROOT" --full --content-hash --portable \
  --output experiments/transfer_methods_2026/paper_grade_package/evidence/required_hdf5_content_manifest.json
```

## Re-execute one full seed

Inspect the exact frozen commands without changing state:

```bash
conda run -n rl_env python \
  experiments/transfer_methods_2026/paper_grade_package/scripts/reproduce_primary_chain.py \
  --seed 131 --stage plan --output-root /path/on/ext4/reproduced-v3c
```

Use `--stage all` to execute the bundled source checkpoint plus the
fidelity-verified archived EAST target parent. Add `--retrain-source` to
retrain the J-TEXT parent; replay then fails closed unless that checkpoint hash
matches the archived target lineage. Every new output directory must be empty.
Current-shot outcomes remain lazy: they are read only after the shot prediction
and query decision are sealed.

GPU kernel and library differences can change floating-point scores.  The
source evaluator explicitly restores the registered deterministic settings;
the packaged replay reproduces source alarm decisions and lead endpoints from
the frozen checkpoints.

## Default CLI

`python -m reproducibility.cli reproduce` and the installed
`fusion-reproduce reproduce` entry now invoke this v3c archived-parent path and
verify against `results/reference/paper_v3c_multiseed_summary.json`. Historical
v8 artifacts are not part of the final paper-lineage reproduction contract.

## Missing confirmation evidence

No local command can make balanced360 independent again.  A claim-changing
experiment requires a newly acquired chronological EAST stream, a target
validation rule separate from support fitting, and static/active/no-label/
equal-budget-random branches frozen before outcomes are opened.  A future Git
release also needs a code DOI and a data DOI or controlled-access procedure.

The repository can now fail-closed on the pre-outcome part of that experiment.
After the separate target-validation cohort is complete and its threshold
artifact is committed, copy
`configs/prospective_east_confirmation.template.json` outside the checkout,
fill its Git SHA, validation-threshold, random-schedule and execution-plan
placeholders, and run:

```bash
conda run -n rl_env python -m reproducibility.cli prospective-seal \
  --spec /external/preregistration.json \
  --validation-roster /external/target-validation-shots.txt \
  --roster /external/future-confirmation-shots.txt \
  --output /external/future-confirmation.seal.json
```

Both rosters must contain one strictly increasing shot ID per line and no
metadata or outcomes. The command binds the validation and confirmation
rosters, clean Git SHA, exact method spec, all three source checkpoints, all
three migrated target parents, split/config files and the new validation
artifact. `prospective-verify` fails if any bound byte changes. This is a
preregistration mechanism, not evidence that the external run exists.
