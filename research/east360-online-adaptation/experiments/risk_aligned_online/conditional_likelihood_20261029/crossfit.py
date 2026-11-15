"""Shot-crossfitted prediction-error scales for conditional densities.

The final class-conditional means and scoring schema are those of
``density.fit_density``. For each supported class and channel, every complete,
valid training row is predicted by a ridge fit that excludes its entire shot.
The Student-t scale uses the uncentered, equally shot-weighted second moment of
those errors. It is a prediction-error scale, not proof of calibration: the
crossfit mean models use fewer training shots than the final mean model, and
the result remains a two-class Student-t variant.

The root ``crossfit`` metadata describes the procedure. Each class's
``crossfit`` metadata records eligible and residual coverage and the held and
fitting shots and row counts for each inner fold. Unsupported channels keep
the original empty fits and report zero crossfit residual coverage.
"""

import numpy as np
import torch

from ..readout_identifiability_20261029.learning import predict, ridge
from . import density


def fit_crossfit_density(x, target, valid, ids, labels, *, device="cpu"):
    """Fit the original means with leave-one-shot-out residual scales.

    Folds are formed independently within each training class and channel,
    using exactly the valid rows with all 25 finite targets. All rows for a
    held shot in that class/channel are predicted together, and none of that
    shot is used for its inner fit. A supported class can therefore have an
    inner model trained on just one other shot. The caller must exclude any
    outer-held shots from the supplied training rows. No output is calibrated
    and no channels or hyperparameters are selected by this procedure.
    """
    fitted = density.fit_density(x, target, valid, ids, labels, device=device)
    x, target, valid = density._inputs(x, target, valid)
    ids, labels = density._numpy(ids), density._numpy(labels)
    complete = valid & np.isfinite(target).all(axis=1)
    fitted["crossfit"] = {
        "method": "leave_one_shot_out",
        "scale": "uncentered_shot_weighted_prediction_error_second_moment",
        "note": (
            "The caller must exclude outer-held shots from the supplied rows. "
            "Inner mean models use fewer training shots and are smaller than "
            "the final mean model. This two-class Student-t prediction-error "
            "scale is not proof of calibration."
        ),
    }

    for channel, channel_fit in enumerate(fitted["channels"]):
        for label, entry in enumerate(channel_fit["classes"]):
            metadata = {
                "eligible_rows": entry["nrows"],
                "eligible_shots": entry["nshots"],
                "residual_rows": 0,
                "residual_shots": 0,
                "residual_shot_ids": [],
                "folds": [],
            }
            entry["crossfit"] = metadata
            if not channel_fit["supported"]:
                continue

            selected = complete[:, channel] & (labels == label)
            xc, yc, shot_ids = x[selected], target[selected, :, channel], ids[selected]
            residual = np.empty_like(yc)
            for shot in np.unique(shot_ids):
                held = shot_ids == shot
                train = ~held
                inner = ridge(
                    xc[train],
                    yc[train],
                    shot_ids[train],
                    density._RIDGE_PENALTY,
                    device=device,
                )
                residual[held] = yc[held] - predict(inner, xc[held])
                fit_shots = np.unique(shot_ids[train])
                metadata["folds"].append(
                    {
                        "held_shot_id": shot.item()
                        if isinstance(shot, np.generic)
                        else shot,
                        "held_rows": int(held.sum()),
                        "fit_rows": int(train.sum()),
                        "fit_shots": len(fit_shots),
                        "fit_shot_ids": [
                            value.item() if isinstance(value, np.generic) else value
                            for value in fit_shots
                        ],
                    }
                )

            if not np.isfinite(residual).all():
                raise FloatingPointError("nonfinite crossfit prediction errors")
            weights = density._shot_weights(shot_ids)
            moment = residual.T @ (weights[:, None] * residual)
            scale = 0.5 * moment + 0.5 * np.diag(np.diag(moment))
            scale += density._JITTER * np.eye(density.HORIZON)
            if not np.isfinite(scale).all():
                raise FloatingPointError("nonfinite crossfit Student-t scale")
            chol = np.linalg.cholesky(scale)
            entry["t_scale"] = torch.from_numpy(scale.copy())
            entry["t_cholesky"] = torch.from_numpy(chol.copy())
            metadata["residual_rows"] = len(residual)
            metadata["residual_shots"] = entry["nshots"]
            metadata["residual_shot_ids"] = entry["shot_ids"].copy()

    return fitted
