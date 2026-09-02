# EAST V3C shot-first compatibility lineage

This directory is an intentionally retained compatibility snapshot of the
EAST V3C implementation that produced the previously audited Adv25 online
gain.  It is not a fallback selected at runtime.  The snapshot is used only
by `run_east_v3c_success_mainline.py`, which copies these exact files into a
temporary `/dev/shm` execution tree and records their SHA-256 values before
running.

## Why this snapshot is retained

The current development tree contains a later class-balanced sampler and
additional fail-closed contracts.  That sampler is a different parent model
and does not reproduce the retained V3C gain.  The retained V3C lineage uses
the historical `shot-first -> endpoint` support sampler: a support shot is
chosen uniformly, then a positive or negative endpoint is drawn within that
shot.  Mixing the two changes the migrated parent before the online phase and
makes the result scientifically uninterpretable.

The snapshot is therefore named explicitly rather than silently replacing the
current development modules.  It is the code lineage for the diagnostic
experiment reported in `fuison_ext4_runs/v3c_adv25_epoch3_20260830`.

## Execution contract

* The source checkpoint is the static J-TEXT Adv25 checkpoint trained with
  `max(25 ms, AdvancedTime)`; the retained candidates are the VAL-selected
  early checkpoints (epoch 3 for seeds 131/133 and epoch 2 for seed 137).
* Only EAST support110 is used to fit the target parent and threshold.
* The balanced360 stream is sorted chronologically and is not used for
  support fitting or threshold selection.
* The stream executes `predict -> seal -> query -> reveal -> update`.
* The J-TEXT online checkpoint is never passed to EAST.
* All generated artifacts are copied only to the mounted project filesystem;
  `/home/hetaisheng/fuison_ext4_runs` is rejected.

The balanced360 result remains `TEST_DRIVEN_EXPLORATION`/`ONLINE_DIAGNOSTIC`:
it is evidence of a reproducible causal online gain on the development stream,
not an untouched future-stream confirmation.

The companion `v3c_classfirst_compat` directory is retained as an explicitly
named negative/lineage control.  It must not be silently substituted for this
snapshot when reproducing the shot-first result.
