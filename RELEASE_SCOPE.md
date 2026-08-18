# Release scope

## Included

- The 22 Python modules imported by the locked A1 → dynamic adapter → unified online runner.
- The exact split manifests and HDF5 data contract used by those modules.
- The reproducibility CLI and tests.
- The locked three-seed reference summary and the small A1 reference seed JSONs.
- Method documentation needed to interpret the results without treating the 360-shot stream as an untouched final test.

## Excluded

- Raw HDF5 data and derived NPZ/NPY feature arrays.
- PyTorch checkpoints and large training caches.
- Historical failed methods, parameter sweeps, obsolete reports and generated figures.
- Tar archives and copied workspace backups.

The excluded items are either supplied through `data/` or are regenerable artifacts. The repository must fail closed when required data is absent; a committed metric summary alone never counts as a successful reproduction.
